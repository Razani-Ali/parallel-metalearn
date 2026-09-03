from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union, Callable
import torch
from metalearn.loss.base import BaseLoss
from metalearn.algorithms.BaseLearner import MetaOptimizer
from metalearn.algorithms.MetaUtils import get_per_step_loss_weights
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from torch.func import grad, vmap
from metalearn.model_wrappers.MAMLWrapper import MAML_Model
from metalearn.loss import LabelEncoder


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# This implementation incorporates the Task-Agnostic Regularization (TAR) penalty 
# term and the task-unbiased optimization formulation proposed in:
#
# [1] X. Yang, L. Zhang, and J. Wang, "Task-Agnostic Generalized Meta-learning 
#     Based on MAML for Few-Shot Bearing Fault Diagnosis," in Image and Graphics 
#     (ICIG 2023), Lecture Notes in Computer Science, vol. 14355, 
#     Springer, Cham, 2023, pp. 118–129. 
#     DOI: 10.1007/978-3-031-46305-1_10.
#
# DISCLAIMER & INDEPENDENT IMPLEMENTATION NOTICE:
# This module is an independent, clean-room implementation developed from scratch 
# based strictly on the theoretical descriptions and mathematical equations 
# presented in the paper. It has been completely re-architected to integrate 
# seamlessly with our functional, vectorized (`torch.func.vmap`), and memory-capped 
# sequential backend architecture.
#
# Architectural scope:
# - The core contribution adapted directly in this meta-optimizer is the 
#   Task-Agnostic Regular (TAR) entropy reduction penalty term added to the 
#   meta-objective function.
# - External model design elements (e.g., 1D-CNN backbones with Squeeze-and-Excitation 
#   channel attention modules) and outer-loop optimization schedules (e.g., Cosine 
#   Annealing learning rate schedulers) are modularly decoupled and intended to be 
#   configured externally via standard PyTorch model wrappers and lr_schedulers.
# ==============================================================================


class TAGML(MetaOptimizer):
    """
    Task-Agnostic Generalized Meta-Learning (TAGML) implementation with modular design.

    Features:
    - Decoupled deployment/inference adaptation logic (`adapt_and_update`).
    - Encapsulated functional forward passes and loss evaluation.
    - Support for multi-step loss weighting and custom inner optimizers.
    - Fully vectorized task processing via `torch.func.vmap`.
    """

    def __init__(
        self,
        *,
        model: MAML_Model,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: BaseInnerOptimizer,
        support_loss_fn: BaseLoss,
        query_loss_fn: Optional[BaseLoss] = None,
        encoder: LabelEncoder = None,
        inner_steps: int = 1,
        multi_step_loss: bool = False,
        lambda_tar: float = 0.005,
        chunk_size: int = 80,
        device: Optional[torch.device] = None,
        backend: str = "vmap",
        **kwargs,
    ):
        """
        Initializes the TAGML meta-optimizer module.

        Args:
            model (MAML_Model): The base neural network model.
            optimizer (torch.optim.Optimizer): The outer-loop optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): The task-level inner-loop optimizer.
            support_loss_fn (BaseLoss): Loss module used during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module used for query evaluation (defaults to support_loss_fn).
            encoder (LabelEncoder): One-Hot Encoder for categorical labels
            inner_steps (int): Number of gradient adaptation steps in the inner loop.
            multi_step_loss (bool): Whether to compute weighted query losses across all inner steps.
            lambda_tar (float): Task-Agnostic Regular Term
            chunk_size (int): Chunk size for vmap memory management during parallel task execution.
            device (Optional[torch.device]): Target compute device (CPU or GPU).
            backend (str): If you set it to "sequential", Vmap would be ignored and algorithm switches to for loop
        """
        super().__init__()
        # Initialize core components
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn
        self.encoder = encoder
        self.backend = backend.lower()
        self.lambda_tar = lambda_tar
        
        self.num_inner_steps = inner_steps
        self.multi_step_loss = multi_step_loss
        self.chunk_size = chunk_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transfer modules to device
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner learning rates to the outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for n, p in inner_params]})

        self.step_weights = None
        
        # Wrap the inner loss function with functional autograd.
        # has_aux=True allows returning auxiliary data (like updated buffers) alongside the loss.
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0, has_aux=True)

    @staticmethod
    def _compute_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Computes the mean prediction entropy on query logits: H(p) = -sum(p * log(p))[cite: 10]
        """
        probs = torch.softmax(logits, dim=-1) #[cite: 10]
        log_probs = torch.log(probs + eps) #[cite: 10]
        entropy_per_sample = -torch.sum(probs * log_probs, dim=-1) #[cite: 10]
        return entropy_per_sample.mean() #[cite: 10]

    def _inner_loss_fn(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        static_params: Dict[str, torch.Tensor], 
        buffers: Dict[str, torch.Tensor], 
        x_s: torch.Tensor, 
        y_s: torch.Tensor, 
        inner_step: int = 0,
        training: bool = False, 
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the support loss and returns updated model buffers.
        """
        combined_params = {**fast_weights, **static_params, **buffers}
        (loss, _), out_dict = self.compute_loss(
            X=x_s, Y=y_s, model_states=combined_params,
            loss_module=self.support_loss_fn, inner_step=inner_step,
            training=training, **kwargs
        )
        # Return loss (for grad computation) and updated buffers (auxiliary output)
        return loss, out_dict.get("buffers", buffers)

    def compute_loss(
        self, *, X: torch.Tensor, Y: torch.Tensor, model_states: Dict[str, torch.Tensor], 
        loss_module: BaseLoss, inner_step: int = 0, training: bool = False, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Executes a functional forward pass and evaluates the loss.
        """
        forward_kwargs = {"training": training, "num_step": inner_step}
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)
        return loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs), out_dict

    def _extract_model_states(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """Separates fixed parameters, buffers, and fast-adapting parameters."""
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())
        
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})
        
        return static_params, all_buffers, initial_fast_weights

    def _accumulate_chunked_step(
        self,
        vectorized_processor: Callable,
        Xsupport: Union[torch.Tensor, Dict],
        Ysupport: Union[torch.Tensor, Dict],
        Xquery: Union[torch.Tensor, Dict],
        Yquery: Union[torch.Tensor, Dict],
        all_buffers: Dict[str, torch.Tensor],
        training: bool
    ) -> Tuple[float, float, Dict[str, torch.Tensor]]:
        """
        A generic, memory-efficient manual gradient accumulation engine.
        Slices the batch into chunks, accumulates gradients on the fly, and clears CUDA cache.
        Can be safely reused by other meta-learning algorithms.
        """
        total_meta_loss = 0.0
        total_metric = 0.0
        accumulated_buffers = {}

        # Determine total tasks in the batch
        batch_size = Xsupport.shape[0] if isinstance(Xsupport, torch.Tensor) else Xsupport["labels"].shape[0]
        num_chunks = (batch_size + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, batch_size)

            # Helper to slice both Tensors and Dictionaries of Tensors
            def slice_data(obj):
                if isinstance(obj, torch.Tensor):
                    return obj[start_idx:end_idx]
                elif isinstance(obj, dict):
                    return {k: v[start_idx:end_idx] for k, v in obj.items()}
                return obj

            X_s_chunk = slice_data(Xsupport)
            Y_s_chunk = slice_data(Ysupport)
            X_q_chunk = slice_data(Xquery)
            Y_q_chunk = slice_data(Yquery)

            # Process the chunk through the vmap processor
            meta_losses, metrics, batched_buffers = vectorized_processor(
                X_s_chunk, Y_s_chunk, X_q_chunk, Y_q_chunk, all_buffers
            )

            # Weight the loss mathematically to preserve true batch mean
            chunk_weight = (end_idx - start_idx) / batch_size
            chunk_loss = meta_losses.sum() / batch_size

            if training:
                # Immediate backward pass frees the graph for this chunk
                chunk_loss.backward()

                with torch.no_grad():
                    # Accumulate batched buffers (e.g., running mean/var) for final outer update
                    for name in batched_buffers:
                        mean_buf = batched_buffers[name].mean(dim=0) * chunk_weight
                        if name not in accumulated_buffers:
                            accumulated_buffers[name] = mean_buf
                        else:
                            accumulated_buffers[name] += mean_buf

            total_meta_loss += chunk_loss.item()
            total_metric += metrics.mean().item() * chunk_weight

            # 🔥 Aggressively clear cache after each chunk's backward to prevent memory spikes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_meta_loss, total_metric, accumulated_buffers

    def _run_sequential(
        self,
        process_single_task: Callable,
        Xsupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Xquery: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Yquery: Union[torch.Tensor, Dict[str, torch.Tensor]],
        all_buffers: Dict[str, torch.Tensor],
        training: bool
    ) -> Tuple[float, float, Dict[str, torch.Tensor]]:
        """
        Executes strictly sequential task processing for meta-learning loops.
        
        Instead of batching all tasks into memory simultaneously via vectorized transforms (vmap),
        this dispatcher unbinds tasks sequentially. In training mode, backward passes are executed 
        immediately inside `process_single_task` per task to destroy intermediate forward/backward
        computation graphs on the fly and aggressively purge the CUDA allocator cache.
        
        Args:
            process_single_task (Callable): Pure task processing closure accepting single task tensors.
            Xsupport (Union[torch.Tensor, Dict]): Support inputs batched across tasks at dim 0.
            Ysupport (Union[torch.Tensor, Dict]): Support targets batched across tasks at dim 0.
            Xquery (Union[torch.Tensor, Dict]): Query inputs batched across tasks at dim 0.
            Yquery (Union[torch.Tensor, Dict]): Query targets batched across tasks at dim 0.
            all_buffers (Dict[str, torch.Tensor]): Master dictionary of persistent model buffers.
            training (bool): If True, triggers task-level backward accumulation and cache clearing.

        Returns:
            Tuple[float, float, Dict[str, torch.Tensor]]:
                - total_meta_loss (float): True mathematical mean meta-loss across the entire batch.
                - total_metric (float): True mathematical mean query evaluation metric.
                - accumulated_buffers (Dict[str, torch.Tensor]): Linearly aggregated model running buffers.
        """
        # Helper closure to unbind tensors along the task dimension (dim=0)
        def _unbind_batch(data: Union[torch.Tensor, Dict[str, torch.Tensor]]):
            if isinstance(data, torch.Tensor):
                # Unbind tensor along batch dimension into a tuple of single-task slices
                return torch.unbind(data, dim=0)
            elif isinstance(data, dict):
                # Unbind dictionary of tensors key-by-key
                unbound_dict = {k: torch.unbind(v, dim=0) for k, v in data.items()}
                num_items = len(next(iter(unbound_dict.values())))
                # Reconstruct list of task-specific dictionaries
                return [{k: unbound_dict[k][i] for k in unbound_dict} for i in range(num_items)]
            return data

        # Unbind all batched support and query structures into task-level collections
        xs_tasks = _unbind_batch(Xsupport)
        ys_tasks = _unbind_batch(Ysupport)
        xq_tasks = _unbind_batch(Xquery)
        yq_tasks = _unbind_batch(Yquery)

        # Determine batch size from the unbound elements
        batch_size = len(xs_tasks)
        # Weight per task to preserve the exact mathematical batch mean
        task_weight = 1.0 / batch_size

        total_meta_loss = 0.0
        total_metric = 0.0
        accumulated_buffers = {}

        # Iterate over tasks one by one
        for i in range(batch_size):
            # Extract individual task tensors
            x_s, y_s = xs_tasks[i], ys_tasks[i]
            x_q, y_q = xq_tasks[i], yq_tasks[i]

            # Execute single task: computes loss/metric, executes backward if training=True, and returns detached items
            task_loss_val, task_metric_val, task_buf = process_single_task(
                x_s, y_s, x_q, y_q, all_buffers, task_weight=task_weight, training=training
            )

            # Accumulate reporting scalar metrics
            loss_val = task_loss_val.item() if isinstance(task_loss_val, torch.Tensor) else task_loss_val
            metric_val = task_metric_val.item() if isinstance(task_metric_val, torch.Tensor) else task_metric_val

            total_meta_loss += loss_val
            total_metric += metric_val * task_weight

            # Accumulate running model buffers (e.g., BatchNorm statistics)
            with torch.no_grad():
                for name, buf in task_buf.items():
                    if buf.dtype == torch.bool:
                        accumulated_buffers[name] = accumulated_buffers.get(name, False) | buf
                    else:
                        mean_buf = buf * task_weight
                        if name not in accumulated_buffers:
                            accumulated_buffers[name] = mean_buf
                        else:
                            accumulated_buffers[name] += mean_buf

            # Aggressively release references to task-specific tensors
            del x_s, y_s, x_q, y_q, task_buf

            # Purge the CUDA cache after the task's backward pass and graph destruction
            if training and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_meta_loss, total_metric, accumulated_buffers

    def step(self, task_itrator, training: bool = True, **kwargs) -> Tuple[float, float]:
        """
        Executes a full MAML meta-training or evaluation step safely and efficiently.
        """
        self._init_step_weights(self.num_inner_steps, training, **kwargs)
        Xs, Ys, Xq, Yq = next(task_itrator)

        def to_device(obj, device):
            if isinstance(obj, torch.Tensor): return obj.to(device)
            elif isinstance(obj, dict): return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        Xsupport = to_device(Xs, self.device)
        Ysupport = to_device(Ys, self.device)
        Xquery = to_device(Xq, self.device)
        Yquery = to_device(Yq, self.device)

        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers, task_weight = 1.0, training = True):
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            is_zero_shot = (x_s.shape[0] == 0)
            accumulated_meta_loss = 0.0 if (self.backend == "sequential" and training) else torch.tensor(0.0, device=self.device)

            # Label Encoding via Encoder Module
            if isinstance(y_s, dict) and "labels" in y_s and self.encoder is not None:
                y_s_enc, y_q_enc = self.encoder(y_s["labels"], y_q["labels"])
                y_s, y_q = {**y_s, "labels": y_s_enc}, {**y_q, "labels": y_q_enc}

            # Task entropy before parameter update
            initial_combined = {**initial_fast_weights, **static_params, **task_buffers}
            if training and self.lambda_tar > 0:
                forward_kwargs = {"training": True, "num_step": 0}
                out_init = torch.func.functional_call(self.model, initial_combined, (x_q,), forward_kwargs)
                init_entropy = self._compute_entropy(out_init["logits"]) #[cite: 10]
            else:
                init_entropy = torch.tensor(0.0, device=self.device)
                tar_penalty = torch.tensor(0.0, device=self.device)

            if not is_zero_shot:
                for inner_step in range(self.num_inner_steps):

                    should_detach = self.inner_optimizer.first_order or not training
                    eval_weights = {k: v.detach() for k, v in fast_weights.items()} if should_detach else fast_weights

                    # Functional Autograd Step
                    grads, updated_task_buffers = self.inner_step_fn(
                        eval_weights, static_params, task_buffers,
                        x_s, y_s, inner_step=inner_step, training=True, **kwargs
                    )
                    
                    # Store updated buffers
                    task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}
                    
                    # Apply Inner Optimizer
                    fast_weights, opt_state = self.inner_optimizer(
                        fast_weights=fast_weights, gradients=grads,
                        state=opt_state, step=inner_step, training=training
                    )

                    # Multi-Step Loss Accumulation (Intermediate Steps)
                    last_step = (inner_step == self.num_inner_steps - 1)
                    if self.multi_step_loss and not last_step:
                        combined = {**fast_weights, **static_params, **task_buffers}
                        
                        # 🛡️ EVALUATION FIX: Do not build graph if not training
                        if training:
                            (q_step_loss, _), _ = self.compute_loss(
                                X=x_q, Y=y_q, model_states=combined,
                                loss_module=self.query_loss_fn, inner_step=inner_step,
                                training=True, **kwargs
                            )

                            if self.backend == "sequential":
                                # Step loss weighting
                                step_weight = self.step_weights[inner_step] if self.step_weights is not None else 1.0
                                scaled_step_loss = (q_step_loss / step_weight) * task_weight
                                
                                # Accumuate gradients
                                scaled_step_loss.backward()
                                
                                accumulated_meta_loss += scaled_step_loss.item()
                                del q_step_loss, scaled_step_loss, combined, grads, updated_task_buffers
                            else:
                                # Graph accumulation for vmap backward
                                accumulated_meta_loss = self._update_meta_loss(accumulated_meta_loss, q_step_loss, inner_step)

                        else:
                            with torch.no_grad():
                                (q_step_loss, _), _ = self.compute_loss(
                                    X=x_q, Y=y_q, model_states=combined,
                                    loss_module=self.query_loss_fn, inner_step=inner_step,
                                    training=False, **kwargs
                                )
                                accumulated_meta_loss = self._update_meta_loss(accumulated_meta_loss, q_step_loss, inner_step)

            # Final Step Query Evaluation
            combined = {**fast_weights, **static_params, **task_buffers}
            
            if training:
                (q_step_loss, q_metric), out_dict = self.compute_loss(
                    X=x_q, Y=y_q, model_states=combined,
                    loss_module=self.query_loss_fn,
                    inner_step=self.num_inner_steps-1 if not is_zero_shot else 0,
                    training=True, **kwargs
                )

                if self.lambda_tar > 0:
                    final_entropy = self._compute_entropy(out_dict["logits"])
                    tar_penalty = self.lambda_tar * (-init_entropy + final_entropy)

            else:
                with torch.no_grad():
                    (q_step_loss, q_metric), out_dict = self.compute_loss(
                        X=x_q, Y=y_q, model_states=combined,
                        loss_module=self.query_loss_fn,
                        inner_step=self.num_inner_steps-1 if not is_zero_shot else 0,
                        training=False, **kwargs
                    )
            
            target_step_idx = max(0, self.num_inner_steps - 1)
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            if training and self.backend == "sequential":
                metric_val = q_metric.item() if isinstance(q_metric, torch.Tensor) else q_metric
                step_weight = self.step_weights[target_step_idx] if (self.step_weights is not None and self.multi_step_loss) else 1.0
                scaled_final_loss = (tar_penalty + q_step_loss / step_weight) * task_weight
                scaled_final_loss.backward(retain_graph=True)
                accumulated_meta_loss += scaled_final_loss.item()
                del q_step_loss, scaled_final_loss, combined, out_dict
                return accumulated_meta_loss, metric_val, updated_task_buffers
            
            else:
                accumulated_meta_loss = self._update_meta_loss(accumulated_meta_loss, q_step_loss, target_step_idx)
                accumulated_meta_loss = accumulated_meta_loss + tar_penalty
                return accumulated_meta_loss, q_metric, updated_task_buffers

        if training:
            self.optimizer.zero_grad()

        if getattr(self, "backend", "vmap") == "sequential":
            total_meta_loss, total_metric, accumulated_buffers = self._run_sequential(
                process_single_task, Xsupport, Ysupport, Xquery, Yquery, all_buffers, training
            )

        else:
            vectorized_processor = vmap(process_single_task, in_dims=(0, 0, 0, 0, None), randomness="different")
            total_meta_loss, total_metric, accumulated_buffers = self._accumulate_chunked_step(
                vectorized_processor, Xsupport, Ysupport, Xquery, Yquery, all_buffers, training
            )

        if training:
            self.optimizer.step()
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in accumulated_buffers:
                        buffer_tensor.copy_(accumulated_buffers[name])

        return total_meta_loss, total_metric

    def _init_step_weights(self, num_inner_steps: int, training: bool, **kwargs) -> None:
        if training and self.multi_step_loss and num_inner_steps > 0:
            m = float(num_inner_steps)
            # v_i = i + 1 (1-indexed) => factor = (i + 1) / m => divisor = m / (i + 1)
            self.step_weights = torch.tensor(
                [m / float(i + 1) for i in range(num_inner_steps)],
                device=self.device,
                dtype=torch.float32
            )

    def _update_meta_loss(self, current_meta_loss: torch.Tensor, q_loss: torch.Tensor, inner_step: int) -> torch.Tensor:
        if self.step_weights is not None and self.multi_step_loss:
            return current_meta_loss + (q_loss / self.step_weights[inner_step])
        return current_meta_loss + q_loss


    # =========================================================================
    # 1. Deployment / Inference Logic (Decoupled from Training Loop)
    # =========================================================================
    def adapt_and_update(
        self, 
        Xsupport: torch.Tensor, 
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        **kwargs
    ) -> None:
        """
        Adapts the base neural network parameters and persistent running buffers on a 
        single target task's support set and updates `self.model` directly in-place.

        This method is designed exclusively for real-world inference deployment on a single machine/task.
        It eliminates multi-task batch dimensions and vectorized mapping overhead (`vmap`), executing 
        stateless functional inner-loop optimization sequentially on the target support set before 
        writing the adapted parameters and synchronized buffers permanently into the base model.

        Args:
            Xsupport (torch.Tensor): Support set input feature tensor of shape `(K, ...)`.
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support set target labels of shape `(K,)` 
                or a target dictionary containing the `'labels'` key.
            **kwargs: Additional keyword arguments forwarded to the loss module and forward pass operations.
        """
        # Define a recursive helper closure to transfer tensors or nested dictionaries to the target device
        def to_device(obj, device):
            # Check if the object is a standard PyTorch Tensor
            if isinstance(obj, torch.Tensor):
                # Move tensor directly to compute device
                return obj.to(device)
            # Check if the object is a dictionary container
            elif isinstance(obj, dict):
                # Recursively migrate all tensor elements within the dictionary
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            # Return unsupported object unchanged
            return obj

        # Transfer support features and target label structures to the active compute device (CPU or GPU)
        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        # Extract fixed parameters, dynamic model buffers, and initial fast-adapting weights from the base model
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        # Create an isolated local clone of model buffers for this specific adaptation instance
        task_buffers = {k: v.clone() for k, v in all_buffers.items()}

        # Instantiate a dedicated OrderedDict copy of initial fast weights to track gradient updates
        fast_weights = OrderedDict(initial_fast_weights)

        # Initialize the stateless inner optimizer state (e.g., zero momentum buffers) for these weights
        opt_state = self.inner_optimizer.init_state(fast_weights)

        # Re-index categorical labels if a label encoder is registered on the meta-optimizer instance
        if isinstance(Ysupport, dict) and "labels" in Ysupport and getattr(self, "encoder", None) is not None:
            # Map raw labels into 0-indexed categorical space using the support set as its own reference
            y_s_enc, _ = self.encoder(Ysupport["labels"], Ysupport["labels"])
            # Reconstruct dictionary with transformed categorical labels
            Ysupport = {**Ysupport, "labels": y_s_enc}

        # Check whether the support set is empty to safely support zero-shot evaluation without gradient steps
        is_zero_shot = (Xsupport.shape[0] == 0)

        # Execute inner-loop gradient adaptation steps if support samples are present
        if not is_zero_shot:
            # Iterate through the configured number of inner gradient descent steps
            for inner_step in range(self.num_inner_steps):
                # Detach fast weights to truncate the autograd history graph and prevent activation memory accumulation
                eval_weights = {k: v.detach() for k, v in fast_weights.items()}

                # Compute support loss gradients and capture updated dynamic buffers using functional autograd
                grads, updated_task_buffers = self.inner_step_fn(
                    eval_weights, static_params, task_buffers,
                    Xsupport, Ysupport, inner_step=inner_step, training=True, **kwargs
                )

                # Detach and update local task buffers for subsequent inner adaptation steps
                task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                # Apply the functional inner optimizer update rule to compute adapted fast weights for this step
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights,
                    gradients=grads, 
                    state=opt_state,
                    training=False, 
                    step=inner_step
                )

                # Delete temporary gradient references to free memory immediately
                del grads, updated_task_buffers, eval_weights

        # Assemble the final model state by combining adapted fast weights, frozen parameters, and task buffers
        combined = {**fast_weights, **static_params, **task_buffers}

        # Execute a final stateless forward pass to capture finalized model buffers (e.g., running stats)
        with torch.no_grad():
            # Run functional forward pass without tracking autograd computation graphs
            _, out_dict = self.compute_loss(
                X=Xsupport, 
                Y=Ysupport, 
                model_states=combined,
                loss_module=self.support_loss_fn,
                inner_step=self.num_inner_steps if not is_zero_shot else 0,
                training=False, 
                **kwargs
            )

        # Extract finalized running buffers from the forward pass output, falling back to local task buffers
        raw_buffers = out_dict.get("buffers", task_buffers)
        # Detach all finalized buffer tensors from any potential computation context
        updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

        # Permanently copy the adapted fast weights and updated buffers back into the base model instance in-place
        with torch.no_grad():
            # Iterate over named parameters and copy corresponding adapted weights directly
            for name, param in self.model.named_parameters():
                # Check if this parameter was designated as an adapting fast weight
                if name in fast_weights:
                    # In-place parameter copy
                    param.copy_(fast_weights[name])

            # Iterate over named persistent buffers and copy corresponding updated values directly
            for name, buffer_tensor in self.model.named_buffers():
                # Check if this buffer exists in the finalized updated buffers dictionary
                if name in updated_task_buffers:
                    # In-place buffer copy
                    buffer_tensor.copy_(updated_task_buffers[name])

        # Purge temporary CUDA allocations from the allocator cache to maintain a minimal VRAM footprint
        if torch.cuda.is_available():
            # Free unused cached GPU memory
            torch.cuda.empty_cache()