from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union, Callable, Any
import torch
import torch.utils._pytree as pytree
from torch.func import grad, vmap
from metalearn.loss.base import BaseLoss
from metalearn.algorithms.BaseLearner import MetaOptimizer
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from metalearn.model_wrappers.MAMLWrapper import MAML_Model
from metalearn.loss import LabelEncoder


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# Nichol, A., Achiam, J., & Schulman, J. (2018).
# "On First-Order Meta-Learning Algorithms." 
# arXiv preprint arXiv:1803.02999.
# ==============================================================================


class Reptile(MetaOptimizer):
    """
    Reptile Meta-Learning Algorithm.

    A simple, highly scalable first-order meta-learning algorithm. Instead of differentiating
    through the inner adaptation path (like MAML), Reptile executes task-level gradient descent
    and performs outer updates by treating weight deltas (W_adapted - W_initial) as effective pseudo-gradients.

    Key Features:
    - Stateless vectorized task computation via `torch.func.vmap`.
    - Chunked execution engine preventing GPU Out-Of-Memory (OOM) failures.
    - Zero computation graph overhead during pure inference/evaluation phases.
    - Task-level running buffer synchronization (e.g., BatchNorm updates).
    """

    def __init__(
        self,
        *,
        model: MAML_Model,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: BaseInnerOptimizer,
        support_loss_fn: BaseLoss,
        query_loss_fn: Optional[BaseLoss] = None,
        encoder: Optional[LabelEncoder] = None,
        inner_steps: int = 5,
        chunk_size: int = 8,
        device: Optional[torch.device] = None,
        backend: str = "vmap",
        **kwargs: Any,
    ):
        """
        Initializes the Reptile meta-optimizer instance.

        Args:
            model (MAML_Model): The base neural network wrapper model.
            optimizer (torch.optim.Optimizer): The outer-loop meta-optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): Task-level inner optimizer adapting weights.
            support_loss_fn (BaseLoss): Loss module evaluated during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module evaluated for query performance.
            encoder (Optional[LabelEncoder]): Label re-indexing encoder.
            inner_steps (int): Total number of inner gradient steps applied on the support set.
            chunk_size (int): Sub-batch size processed concurrently to cap VRAM consumption.
            device (Optional[torch.device]): Target hardware compute device (CPU or GPU).
            backend (str): If you set it to "sequential", Vmap would be ignored and algorithm switches to for loop
            **kwargs: Additional operational parameters.
        """
        # Initialize base MetaOptimizer class
        super().__init__()

        # Store primary components and hyper-parameters
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer

        # Configure loss calculation modules
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn

        self.num_inner_steps = inner_steps
        self.chunk_size = chunk_size
        self.encoder = encoder
        self.backend = backend.lower()

        # Resolve target compute device and transfer all modules
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner-loop parameters to outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for _, p in inner_params]})

        # Functional gradient calculation wrapper for inner loop steps (First-Order gradients)
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0, has_aux=True)

    def _extract_model_states(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """
        Splits model states into static parameters, persistent buffers, and initial fast weights.

        Returns:
            Tuple[Dict, Dict, OrderedDict]: 
                (static_parameters, all_buffers, initial_fast_weights)
        """
        # Extract all named parameters and buffers as dictionaries
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())

        # Identify trainable keys designated for fast adaptation
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())

        # Separate static parameters from fast-adapting parameters
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})

        return static_params, all_buffers, initial_fast_weights

    def _inner_loss_fn(
        self,
        fast_weights: Dict[str, torch.Tensor],
        static_params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x_s: torch.Tensor,
        y_s: Union[torch.Tensor, Dict[str, torch.Tensor]],
        inner_step: int = 0,
        training: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes support loss for a single inner-step adaptation.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                Evaluated scalar loss tensor and updated model buffers dictionary.
        """
        # Combine fast weights, static parameters, and model buffers
        combined_params = {**fast_weights, **static_params, **buffers}

        # Calculate support loss using functional evaluation
        (loss, _), out_dict = self.compute_loss(
            X=x_s,
            Y=y_s,
            model_states=combined_params,
            loss_module=self.support_loss_fn,
            inner_step=inner_step,
            training=training,
            **kwargs,
        )

        # Return loss for autograd and captured buffer modifications
        return loss, out_dict.get("buffers", buffers)

    def compute_loss(
        self,
        *,
        X: torch.Tensor,
        Y: Union[torch.Tensor, Dict[str, torch.Tensor]],
        model_states: Dict[str, torch.Tensor],
        loss_module: BaseLoss,
        inner_step: int = 0,
        training: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs a functional forward pass and computes evaluation metrics.
        """
        # Pack operational context flags to ensure functional_call compatibility
        forward_kwargs = {"training": training, "num_step": inner_step}

        # Execute stateless functional forward pass on model
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)

        # Evaluate loss and performance metric
        loss_and_metric = loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs)

        return loss_and_metric, out_dict

    def _accumulate_chunked_step(
        self,
        vectorized_processor: Callable,
        Xsupport: Union[torch.Tensor, Dict],
        Ysupport: Union[torch.Tensor, Dict],
        Xquery: Union[torch.Tensor, Dict],
        Yquery: Union[torch.Tensor, Dict],
        all_buffers: Dict[str, torch.Tensor],
        training: bool,
    ) -> Tuple[Dict[str, torch.Tensor], float, float, Dict[str, torch.Tensor]]:
        """
        Processes Reptile tasks in memory-capped chunks.
        Aggregates parameter deltas across chunks while strictly capping GPU memory footprint.

        Args:
            vectorized_processor (Callable): Function mapped across tasks via torch.func.vmap.
            Xsupport (Union[torch.Tensor, Dict]): Batched support inputs.
            Ysupport (Union[torch.Tensor, Dict]): Batched support targets.
            Xquery (Union[torch.Tensor, Dict]): Batched query inputs.
            Yquery (Union[torch.Tensor, Dict]): Batched query targets.
            all_buffers (Dict[str, torch.Tensor]): Master model buffers dictionary.
            training (bool): Operational training flag.

        Returns:
            Tuple[Dict[str, torch.Tensor], float, float, Dict[str, torch.Tensor]]:
                - accumulated_deltas: Weighted parameter deltas (W_adapted - W_initial).
                - total_query_loss: Mean query loss across all tasks.
                - total_metric: Mean query performance metric across all tasks.
                - accumulated_buffers: Weighted running buffers dictionary.
        """
        total_query_loss = 0.0
        total_metric = 0.0
        accumulated_deltas: Dict[str, torch.Tensor] = {}
        accumulated_buffers: Dict[str, torch.Tensor] = {}

        # Resolve total batch size across tasks
        batch_size = Xsupport.shape[0] if isinstance(Xsupport, torch.Tensor) else Xsupport["labels"].shape[0]
        num_chunks = (batch_size + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, batch_size)

            # Helper to slice both tensors and dictionaries of tensors
            def slice_data(obj):
                if isinstance(obj, torch.Tensor):
                    return obj[start_idx:end_idx]
                elif isinstance(obj, dict):
                    return {k: v[start_idx:end_idx] for k, v in obj.items()}
                return obj

            # Extract slices for current chunk
            X_s_chunk = slice_data(Xsupport)
            Y_s_chunk = slice_data(Ysupport)
            X_q_chunk = slice_data(Xquery)
            Y_q_chunk = slice_data(Yquery)

            # Process chunk tasks in parallel
            chunk_deltas, q_losses, q_metrics, batched_buffers = vectorized_processor(
                X_s_chunk, Y_s_chunk, X_q_chunk, Y_q_chunk, all_buffers
            )

            # Calculate chunk weight factor relative to total batch size
            chunk_weight = (end_idx - start_idx) / batch_size

            # Accumulate parameter deltas
            with torch.no_grad():
                for name, delta_tensor in chunk_deltas.items():
                    weighted_delta = delta_tensor.mean(dim=0) * chunk_weight
                    if name not in accumulated_deltas:
                        accumulated_deltas[name] = weighted_delta
                    else:
                        accumulated_deltas[name] += weighted_delta

                # Accumulate buffer updates across chunks
                for name in batched_buffers:
                    buf = batched_buffers[name]
                    if buf.dtype == torch.bool:
                        mean_buf = buf.any(dim=0)
                        accumulated_buffers[name] = mean_buf
                    else:
                        mean_buf = buf.mean(dim=0) * chunk_weight
                        if name not in accumulated_buffers:
                            accumulated_buffers[name] = mean_buf
                        else:
                            accumulated_buffers[name] += mean_buf

            # Accumulate reporting statistics
            total_query_loss += q_losses.mean().item() * chunk_weight
            total_metric += q_metrics.mean().item() * chunk_weight

            # Clear CUDA allocation cache per chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return accumulated_deltas, total_query_loss, total_metric, accumulated_buffers

    def _run_sequential(
        self,
        process_single_task: Callable,
        Xsupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Xquery: Union[torch.Tensor, Dict[str, torch.Tensor]],
        Yquery: Union[torch.Tensor, Dict[str, torch.Tensor]],
        all_buffers: Dict[str, torch.Tensor],
        training: bool
    ) -> Tuple[Dict[str, torch.Tensor], float, float, Dict[str, torch.Tensor]]:
        """
        Executes strictly sequential task processing for the Reptile algorithm.

        Instead of vectorizing all tasks across memory simultaneously via `torch.func.vmap`,
        this dispatcher unbinds tasks sequentially. It computes the parameter deltas 
        (W_adapted - W_initial) per task and accumulates them. Because Reptile relies on 
        weight differences rather than outer-loop query gradients, this completely prevents 
        GPU OOM failures by aggressively freeing the CUDA cache after each task.

        Args:
            process_single_task (Callable): Pure single-task execution closure.
            Xsupport (Union[torch.Tensor, Dict]): Support inputs batched across tasks at dim 0.
            Ysupport (Union[torch.Tensor, Dict]): Support targets batched across tasks at dim 0.
            Xquery (Union[torch.Tensor, Dict]): Query inputs batched across tasks at dim 0.
            Yquery (Union[torch.Tensor, Dict]): Query targets batched across tasks at dim 0.
            all_buffers (Dict[str, torch.Tensor]): Master dictionary of persistent model buffers.
            training (bool): Operational training flag.

        Returns:
            Tuple[Dict[str, torch.Tensor], float, float, Dict[str, torch.Tensor]]:
                - accumulated_deltas (Dict): Weighted parameter deltas for the outer update.
                - total_query_loss (float): Mean query loss across all tasks (for logging).
                - total_metric (float): Mean query performance metric across all tasks.
                - accumulated_buffers (Dict): Weighted running buffers dictionary.
        """
        # Helper closure to unbind batched data structures along dimension 0
        def _unbind_batch(data: Union[torch.Tensor, Dict[str, torch.Tensor]]):
            if isinstance(data, torch.Tensor):
                return torch.unbind(data, dim=0)
            elif isinstance(data, dict):
                unbound_dict = {k: torch.unbind(v, dim=0) for k, v in data.items()}
                num_items = len(next(iter(unbound_dict.values())))
                return [{k: unbound_dict[k][i] for k in unbound_dict} for i in range(num_items)]
            return data

        # Unbind batched support and query structures into isolated single-task instances
        xs_tasks = _unbind_batch(Xsupport)
        ys_tasks = _unbind_batch(Ysupport)
        xq_tasks = _unbind_batch(Xquery)
        yq_tasks = _unbind_batch(Yquery)

        batch_size = len(xs_tasks)
        task_weight = 1.0 / batch_size

        total_query_loss = 0.0
        total_metric = 0.0
        accumulated_deltas: Dict[str, torch.Tensor] = {}
        accumulated_buffers: Dict[str, torch.Tensor] = {}

        # Iterate sequentially over unbound tasks
        for i in range(batch_size):
            x_s, y_s = xs_tasks[i], ys_tasks[i]
            x_q, y_q = xq_tasks[i], yq_tasks[i]

            # Execute single task and extract deltas, loss, metric, and buffers
            task_deltas, task_loss_val, task_metric_val, task_buf = process_single_task(
                x_s, y_s, x_q, y_q, all_buffers, task_weight=task_weight, training=training
            )

            # Accumulate reporting scalar metrics
            loss_val = task_loss_val.item() if isinstance(task_loss_val, torch.Tensor) else task_loss_val
            metric_val = task_metric_val.item() if isinstance(task_metric_val, torch.Tensor) else task_metric_val

            total_meta_loss += loss_val
            total_metric += metric_val * task_weight

            # Accumulate weight deltas and running buffers
            with torch.no_grad():
                # Accumulate pseudo-gradients (deltas)
                if training:
                    for name, delta_tensor in task_deltas.items():
                        weighted_delta = delta_tensor * task_weight
                        if name not in accumulated_deltas:
                            accumulated_deltas[name] = weighted_delta
                        else:
                            accumulated_deltas[name] += weighted_delta

                # Accumulate running stats (e.g., BatchNorm)
                for name, buf in task_buf.items():
                    if buf.dtype == torch.bool:
                        accumulated_buffers[name] = accumulated_buffers.get(name, False) | buf
                    else:
                        mean_buf = buf * task_weight
                        if name not in accumulated_buffers:
                            accumulated_buffers[name] = mean_buf
                        else:
                            accumulated_buffers[name] += mean_buf

            # Aggressively release references to single-task tensors to break graph links
            del x_s, y_s, x_q, y_q, task_deltas, task_buf

            # Purge the CUDA allocator cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return accumulated_deltas, total_query_loss, total_metric, accumulated_buffers

    def step(
        self,
        task_itrator: Any,
        training: bool = True,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """
        Executes a single Reptile meta-training step.
        Adapts weights on the support set, computes weight deltas, and injects them 
        into the outer optimizer as effective gradients.
        """
        Xs, Ys, Xq, Yq = next(task_itrator)

        # Device transfer utility
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        Xsupport = to_device(Xs, self.device)
        Ysupport = to_device(Ys, self.device)
        Xquery = to_device(Xq, self.device)
        Yquery = to_device(Yq, self.device)

        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers, task_weight: float = 1.0, training: bool = True):
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            is_zero_shot = x_s.shape[0] == 0

            # Dynamic categorical label re-indexing
            if isinstance(y_s, dict) and "labels" in y_s and isinstance(y_q, dict) and "labels" in y_q and self.encoder is not None:
                y_s_enc, y_q_enc = self.encoder(y_s["labels"], y_q["labels"])
                y_s = {**y_s, "labels": y_s_enc}
                y_q = {**y_q, "labels": y_q_enc}

            if not is_zero_shot:
                # 1. Execute inner adaptation steps towards task manifold
                for inner_step in range(self.num_inner_steps):
                    # 🛠️ GHOST GRAPH FIX: Detach weights early for First-Order MAML
                    should_detach = self.inner_optimizer.first_order or not training
                    eval_weights = {k: v.detach() for k, v in fast_weights.items()} if should_detach else fast_weights

                    # Compute support gradients using First-Order autograd
                    grads, updated_task_buffers = self.inner_step_fn(
                        eval_weights,
                        static_params,
                        task_buffers,
                        x_s,
                        y_s,
                        inner_step=inner_step,
                        training=True,
                        **kwargs,
                    )

                    # Update local task buffers
                    task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                    # Update fast weights via inner optimizer
                    fast_weights, opt_state = self.inner_optimizer(
                        fast_weights=fast_weights,
                        gradients=grads,
                        state=opt_state,
                        step=inner_step,
                        training=False
                    )

                    del grads, updated_task_buffers

            # 2. Compute Reptile Deltas: (W_adapted - W_initial)
            param_deltas = pytree.tree_map(lambda w_new, w_old: w_new - w_old, fast_weights, initial_fast_weights)

            # 3. Final Query Evaluation (For evaluation/metrics logging)
            combined = {**fast_weights, **static_params, **task_buffers}
            with torch.no_grad():
                (q_loss, q_metric), out_dict = self.compute_loss(
                    X=x_q,
                    Y=y_q,
                    model_states=combined,
                    loss_module=self.query_loss_fn,
                    inner_step=self.num_inner_steps if not is_zero_shot else 0,
                    training=False,
                    **kwargs,
                )

            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            if self.backend == "sequential":
                loss_val = q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss
                metric_val = q_metric.item() if isinstance(q_metric, torch.Tensor) else q_metric
                return param_deltas, loss_val, metric_val, updated_task_buffers
            else:
                return param_deltas, q_loss, q_metric, updated_task_buffers

        if training:
            self.optimizer.zero_grad()

        if getattr(self, "backend", "vmap") == "sequential":
            vectorized_deltas, query_loss, query_metric, accumulated_buffers = self._run_sequential(
                process_single_task, Xsupport, Ysupport, Xquery, Yquery, all_buffers, training
            )
        else:
            def _vmap_task(x_s, y_s, x_q, y_q, bufs):
                return process_single_task(x_s, y_s, x_q, y_q, bufs, task_weight=1.0, training=training)

            vectorized_processor = vmap(
                _vmap_task,
                in_dims=(0, 0, 0, 0, None),
                randomness="different",
            )

            vectorized_deltas, query_loss, query_metric, accumulated_buffers = self._accumulate_chunked_step(
                vectorized_processor, Xsupport, Ysupport, Xquery, Yquery, all_buffers, training
            )

        # Outer loop meta-optimization step (First-Order Pseudo-Gradient Injection)
        if training:
            for name, param in self.model.named_parameters():
                if name in vectorized_deltas:
                    # Reptile Update Rule: Theta = Theta + lr * Delta
                    # Standard optimizers: Theta = Theta - lr * Grad  ==>  Grad = -Delta
                    param.grad = -vectorized_deltas[name].detach()

            # Execute outer optimizer step
            self.optimizer.step()

            # Synchronize model running buffers
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in accumulated_buffers:
                        buffer_tensor.copy_(accumulated_buffers[name])

        return query_loss, query_metric

    def adapt_and_update(
        self,
        Xsupport: torch.Tensor,
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """
        Adapts parameters and buffers on a single target task's support set and permanently 
        updates `self.model` in-place. Intended exclusively for real-world deployment.

        This method eliminates the multi-task batch dimension overhead (`vmap`). It directly 
        executes functional inner-loop optimization sequentially on the target support set 
        before writing the adapted weights and synchronized buffers back to the base model.

        Args:
            Xsupport (torch.Tensor): Support set inputs tensor of shape `(K, ...)`.
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support target label structure of shape `(K,)`.
            **kwargs: Operational arguments passed down to loss modules and model forward pass.
        """
        # Device migration helper
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        # Transfer support dataset to compute device
        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        # Extract current model state split
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        # Create isolated local clone of model buffers
        task_buffers = {k: v.clone() for k, v in all_buffers.items()}
        fast_weights = OrderedDict(initial_fast_weights)
        opt_state = self.inner_optimizer.init_state(fast_weights)

        # Re-index labels if encoder is present
        if isinstance(Ysupport, dict) and "labels" in Ysupport and self.encoder is not None:
            y_s_enc, _ = self.encoder(Ysupport["labels"], Ysupport["labels"])
            Ysupport = {**Ysupport, "labels": y_s_enc}

        is_zero_shot = Xsupport.shape[0] == 0

        if not is_zero_shot:
            # Execute inner adaptation steps towards task manifold
            for inner_step in range(self.num_inner_steps):
                # Detach weights to prevent massive activation memory build-up in deployment
                eval_weights = {k: v.detach() for k, v in fast_weights.items()}

                grads, updated_task_buffers = self.inner_step_fn(
                    eval_weights,
                    static_params,
                    task_buffers,
                    Xsupport,
                    Ysupport,
                    inner_step=inner_step,
                    training=True,
                    **kwargs,
                )

                task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights,
                    gradients=grads,
                    state=opt_state,
                    step=inner_step,
                    training=False
                )

                # Delete temporary gradient references
                del grads, updated_task_buffers, eval_weights

        # Capture finalized buffers via stateless forward pass
        combined = {**fast_weights, **static_params, **task_buffers}
        with torch.no_grad():
            _, out_dict = self.compute_loss(
                X=Xsupport,
                Y=Ysupport,
                model_states=combined,
                loss_module=self.support_loss_fn,
                inner_step=self.num_inner_steps if not is_zero_shot else 0,
                training=False,
                **kwargs,
            )

        raw_buffers = out_dict.get("buffers", task_buffers)
        updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

        # Permanently copy adapted parameters and task-averaged buffers back to base model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in fast_weights:
                    param.copy_(fast_weights[name])

            for name, buffer_tensor in self.model.named_buffers():
                if name in updated_task_buffers:
                    buffer_tensor.copy_(updated_task_buffers[name])

        # Clear GPU allocation cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Reptile: Model parameters and running buffers successfully adapted and updated.")