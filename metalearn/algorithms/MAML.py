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
# The Multi-Step Loss (MSL) optimization strategy and per-step gradient loss 
# aggregation in this implementation are based on the MAML++ architecture:
#
# [1] A. Antoniou, H. Edwards, and A. Storkey, "How to train your MAML," 
#     in Int. Conf. Learn. Representations (ICLR), 2019. 
#     arXiv:1810.09502.
#
# MAML original paper:
# [2] C. Finn, P. Abbeel and S. Levine,
#       "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
#       2019, arXiv:1703.03400.
#
# MetaSGD can be implemented using same class (per layer learnable lr + only one task)
# [3] Z. Li, F. Zhou, F. Chen, and H. Li, "Meta-SGD: Learning to Learn 
#        Quickly for Few-Shot Learning," arXiv preprint arXiv:1707.09835, 2017.
#
# [4] ANIL (Almost No Inner Loop):
#     Raghu, A., Raghu, M., Bengio, S., & Vinyals, O. (2019).
#     "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML."
#     International Conference on Learning Representations (ICLR), 2020.
#     arXiv:1909.09157.
#
# [5] BOIL (Body Only Inner Loop):
#     Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2020).
#     "BOIL: Towards Representation Change for Few-shot Learning."
#     International Conference on Learning Representations (ICLR), 2021.
#     arXiv:2008.08882.
# ==============================================================================


class MAML(MetaOptimizer):
    """
    Model-Agnostic Meta-Learning (MAML) implementation with modular design.

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
        chunk_size: int = 80,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initializes the MAML meta-optimizer module.

        Args:
            model (MAML_Model): The base neural network model.
            optimizer (torch.optim.Optimizer): The outer-loop optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): The task-level inner-loop optimizer.
            support_loss_fn (BaseLoss): Loss module used during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module used for query evaluation (defaults to support_loss_fn).
            encoder (LabelEncoder): One-Hot Encoder for categorical labels
            inner_steps (int): Number of gradient adaptation steps in the inner loop.
            multi_step_loss (bool): Whether to compute weighted query losses across all inner steps.
            chunk_size (int): Chunk size for vmap memory management during parallel task execution.
            device (Optional[torch.device]): Target compute device (CPU or GPU).
        """
        super().__init__()
        # Initialize core components
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn
        self.encoder = encoder
        
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

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers):
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            meta_loss = torch.tensor(0.0, device=self.device)
            is_zero_shot = (x_s.shape[0] == 0)

            # Label Encoding via Encoder Module
            if isinstance(y_s, dict) and "labels" in y_s and self.encoder is not None:
                y_s_enc, y_q_enc = self.encoder(y_s["labels"], y_q["labels"])
                y_s, y_q = {**y_s, "labels": y_s_enc}, {**y_q, "labels": y_q_enc}

            if not is_zero_shot:
                for inner_step in range(self.num_inner_steps):

                    # Functional Autograd Step
                    grads, updated_task_buffers = self.inner_step_fn(
                        fast_weights, static_params, task_buffers,
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
                        else:
                            with torch.no_grad():
                                (q_step_loss, _), _ = self.compute_loss(
                                    X=x_q, Y=y_q, model_states=combined,
                                    loss_module=self.query_loss_fn, inner_step=inner_step,
                                    training=False, **kwargs
                                )
                        meta_loss = self._update_meta_loss(meta_loss, q_step_loss, inner_step)

            # Final Step Query Evaluation
            combined = {**fast_weights, **static_params, **task_buffers}
            
            # 🛡️ EVALUATION FIX: No graph retention during pure inference
            if training:
                (q_step_loss, q_metric), out_dict = self.compute_loss(
                    X=x_q, Y=y_q, model_states=combined,
                    loss_module=self.query_loss_fn,
                    inner_step=self.num_inner_steps if not is_zero_shot else 0,
                    training=True, **kwargs
                )
            else:
                with torch.no_grad():
                    (q_step_loss, q_metric), out_dict = self.compute_loss(
                        X=x_q, Y=y_q, model_states=combined,
                        loss_module=self.query_loss_fn,
                        inner_step=self.num_inner_steps if not is_zero_shot else 0,
                        training=False, **kwargs
                    )
            
            target_step_idx = max(0, self.num_inner_steps - 1)
            meta_loss = self._update_meta_loss(meta_loss, q_step_loss, target_step_idx)
            
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            return meta_loss, q_metric, updated_task_buffers

        # Vectorize task processor (Note: no chunk_size passed to vmap, we chunk externally)
        vectorized_processor = vmap(process_single_task, in_dims=(0, 0, 0, 0, None), randomness="different")

        if training:
            self.optimizer.zero_grad()

        # Execute generic chunked accumulator
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
            self.step_weights = get_per_step_loss_weights(
                num_inner_steps, kwargs.get("epoch"), kwargs.get("epochs"), self.device
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
        Adapts model parameters and running buffers on the provided support set and 
        permanently updates `self.model` in-place.

        This method is designed exclusively for test-time task adaptation or model deployment.
        It runs vectorized inner-loop optimization across tasks using `torch.func.vmap`,
        aggregates the resulting adapted fast weights and updated buffers across tasks,
        and copies them back into the base neural network.

        Args:
            Xsupport (torch.Tensor): Support set input feature tensor of shape `(num_tasks, K, ...)`.
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support set target labels or dictionary.
            **kwargs: Additional keyword arguments forwarded to the loss module and model forward passes.
        """
        # Helper function to recursively transfer tensors or nested dictionaries to the target device
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        # Transfer input features and target labels to the designated hardware device (CPU/GPU)
        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        # Extract fixed parameters, dynamic model buffers, and initial fast weights
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        # Define single-task adaptation closure compatible with torch.func.vmap
        def _adapt_single_task(x_s, y_s, global_buffers):
            # Create an isolated clone of model buffers for this specific task
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            
            # Initialize a task-specific copy of fast weights
            fast_weights = OrderedDict(initial_fast_weights)
            
            # Initialize task-specific inner optimizer state (e.g., momentum buffers)
            opt_state = self.inner_optimizer.init_state(fast_weights)

            # Re-index categorical labels if a label encoder is configured
            if isinstance(y_s, dict) and "labels" in y_s and self.encoder is not None:
                # In single-set adaptation (support only), encode labels using support set itself
                y_s_enc, _ = self.encoder(y_s["labels"], y_s["labels"])
                y_s = {**y_s, "labels": y_s_enc}

            # Check if support set contains valid samples (skips adaptation for zero-shot)
            is_zero_shot = (x_s.shape[0] == 0)

            if not is_zero_shot:
                # Execute inner adaptation gradient steps
                for inner_step in range(self.num_inner_steps):
                    # Check if first-order approximation is enabled in the inner optimizer

                    # Compute support gradients and capture updated buffers using functional autograd
                    grads, updated_task_buffers = self.inner_step_fn(
                        fast_weights, static_params, task_buffers,
                        x_s, y_s, inner_step=inner_step, training=True, **kwargs
                    )

                    # Update persistent task buffers for subsequent inner adaptation steps
                    task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                    # Apply inner optimizer update rule to compute adapted fast weights
                    fast_weights, opt_state = self.inner_optimizer(
                        fast_weights=fast_weights,
                        gradients=grads, state=opt_state,
                        training=False, step=inner_step
                    )

            # Combine adapted fast weights with static parameters and updated task buffers
            combined = {**fast_weights, **static_params, **task_buffers}
            
            # Execute final functional forward pass to capture finalized model buffers
            with torch.no_grad():
                _, out_dict = self.compute_loss(
                    X=x_s, Y=y_s, model_states=combined,
                    loss_module=self.support_loss_fn,
                    inner_step=self.num_inner_steps if not is_zero_shot else 0,
                    training=False, **kwargs
                )

            # Extract finalized buffers from output dictionary, falling back to task_buffers if empty
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            # Return adapted task parameters and final running buffers
            return fast_weights, updated_task_buffers

        # Vectorize single-task adaptation across the task batch dimension
        vectorized_adapt = vmap(
            _adapt_single_task, 
            in_dims=(0, 0, None), 
            chunk_size=self.chunk_size, 
            randomness="different"
        )
        
        # Execute vectorized task adaptation across all support tasks simultaneously
        fast_weights_batched, batched_buffers = vectorized_adapt(Xsupport, Ysupport, all_buffers)

        # Permanently copy adapted parameters and task-averaged buffers back to the base model
        with torch.no_grad():
            # Update trainable model parameters with mean adapted values across tasks
            for name, param in self.model.named_parameters():
                if name in fast_weights_batched:
                    param.copy_(fast_weights_batched[name].mean(dim=0))
            
            # Update persistent model buffers with mean updated values across tasks
            for name, buffer_tensor in self.model.named_buffers():
                if name in batched_buffers:
                    buffer_tensor.copy_(batched_buffers[name].mean(dim=0))

        # Clear GPU cache after parameter deployment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()