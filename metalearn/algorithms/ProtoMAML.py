from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union, Callable, Any
import torch
from torch.func import grad, vmap
from metalearn.loss.base import BaseLoss
from .BaseLearner import MetaOptimizer
from .MetaUtils import get_per_step_loss_weights
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from metalearn.model_wrappers.ProtoMAMLWrapper import ProtoMAML_Model


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# [1] ProtoMAML:
#     Triantafillou, E., Zhu, T., Dumoulin, V., Lamblin, P., Evci, U., Xu, K.,
#     Goroshin, R., Gelada, C., Swersky, K., Manzagol, P. A., & Larochelle, H. (2019).
#     "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples."
#     International Conference on Learning Representations (ICLR), 2020.
#     arXiv:1903.03096.
#
# [2] MAML++ Multi-Step Loss & Per-Step Adaptation:
#     Antoniou, A., Edwards, H., & Storkey, A. (2019).
#     "How to train your MAML."
#     International Conference on Learning Representations (ICLR), 2019.
#     arXiv:1810.09502.
#
# [3] Meta-SGD (Per-layer/step learnable learning rates):
#     Li, Z., Zhou, F., Chen, F., & Li, H. (2017).
#     "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning."
#     arXiv preprint arXiv:1707.09835.
#
# [4] BOIL (Body Only Inner Loop):
#     Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2021).
#     "BOIL: Towards Representation Change for Few-shot Learning."
#     International Conference on Learning Representations (ICLR), 2021.
#     arXiv:2008.08882.
# ==============================================================================


class ProtoMAML(MetaOptimizer):
    """
    Prototypical Model-Agnostic Meta-Learning (ProtoMAML) Meta-Optimizer Engine.

    Combines metric-based Prototypical initialization of classification heads 
    with gradient-based meta-optimization (MAML/MAML++).

    Key Capabilities:
    - Stateless vectorized task computation powered by `torch.func.vmap`.
    - Memory-efficient manual chunked gradient accumulation ensuring OOM-free GPU execution.
    - Zero Ghost Graph overhead under First-Order MAML optimization.
    - Robust task-level running buffer synchronization (BatchNorm / Prototype tracking).
    - Evaluation graph suppression (`torch.no_grad()`) during non-training phases.
    """

    def __init__(
        self,
        *,
        model: ProtoMAML_Model,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: BaseInnerOptimizer,
        support_loss_fn: BaseLoss,
        query_loss_fn: Optional[BaseLoss] = None,
        inner_steps: int = 1,
        multi_step_loss: bool = False,
        chunk_size: int = 8,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        """
        Initializes the ProtoMAML meta-optimizer module.

        Args:
            model (ProtoMAML_Model): Neural network wrapper implementing prototypical initialization.
            optimizer (torch.optim.Optimizer): Outer-loop meta-optimizer (e.g., Adam, AdamW).
            inner_optimizer (BaseInnerOptimizer): Inner-loop task optimizer (e.g., InnerSGD, InnerAdam).
            support_loss_fn (BaseLoss): Loss module evaluated during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module evaluated on query sets (defaults to support_loss_fn).
            inner_steps (int): Total number of inner gradient descent adaptation steps.
            multi_step_loss (bool): If True, accumulates weighted query loss across all inner steps.
            chunk_size (int): Batch chunk size processed concurrently to cap VRAM footprint.
            device (Optional[torch.device]): Target hardware compute device (CPU or GPU).
            **kwargs: Additional operational keyword arguments.
        """
        # Call parent MetaOptimizer constructor
        super().__init__()

        # Store model and optimization engines
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer

        # Configure support and query loss calculation modules
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn

        # Configure hyperparameters
        self.num_inner_steps = inner_steps
        self.multi_step_loss = multi_step_loss
        self.chunk_size = chunk_size

        # Resolve target device and move all registered modules
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner optimizer parameters (e.g., Meta-SGD per-layer LRs) to the outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for _, p in inner_params]})

        # Buffer for step-wise query loss weights
        self.step_weights = None

        # Functional autograd gradient operator for inner loop adaptation.
        # has_aux=True permits returning auxiliary outputs (updated task buffers) alongside the loss scalar.
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0, has_aux=True)

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
        Computes support adaptation loss and captures stateful buffer modifications.

        Args:
            fast_weights (Dict[str, torch.Tensor]): Trainable parameter state dictionary.
            static_params (Dict[str, torch.Tensor]): Non-trainable / frozen parameter dictionary.
            buffers (Dict[str, torch.Tensor]): Model running state buffers.
            x_s (torch.Tensor): Support set input feature tensor.
            y_s (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support target label structure.
            inner_step (int): Current inner adaptation iteration index.
            training (bool): Flag indicating if model operates in training mode.
            **kwargs: Context arguments forwarded to loss module.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                Calculated scalar loss tensor and updated model buffers dictionary.
        """
        # Merge dynamic fast weights, static weights, and model buffers into unified state dict
        combined_params = {**fast_weights, **static_params, **buffers}

        # Perform functional forward and compute support task loss
        (loss, _), out_dict = self.compute_loss(
            X=x_s,
            Y=y_s,
            model_states=combined_params,
            loss_module=self.support_loss_fn,
            inner_step=inner_step,
            training=training,
            **kwargs,
        )

        # Return computed loss for backprop and extract modified buffers (e.g., BatchNorm updates)
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
        Executes a functional forward pass through the model and calculates task loss and evaluation metric.

        Args:
            X (torch.Tensor): Batch input tensor.
            Y (Union[torch.Tensor, Dict[str, torch.Tensor]]): Ground truth targets.
            model_states (Dict[str, torch.Tensor]): Unified state dictionary containing parameters and buffers.
            loss_module (BaseLoss): Loss calculation instance.
            inner_step (int): Current inner gradient step index.
            training (bool): Operational training flag for dropout and normalization layers.
            **kwargs: Additional parameters passed to model and loss modules.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
                - ((loss_tensor, metric_tensor), model_forward_output_dictionary)
        """
        # Pack forward-specific context flags to prevent kwargs mismatches in functional_call
        forward_kwargs = {"training": training, "num_step": inner_step}

        # Execute functional stateless forward pass on model
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)

        # Evaluate scalar loss and metric via BaseLoss interface
        loss_and_metric = loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs)

        return loss_and_metric, out_dict

    def _extract_model_states(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """
        Splits model states into static parameters, persistent buffers, and initial fast weights.

        Returns:
            Tuple[Dict, Dict, OrderedDict]: 
                (static_parameters, all_buffers, initial_fast_weights)
        """
        # Extract all parameters and buffers
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())

        # Retrieve keys designated for fast inner-loop gradient adaptation
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())

        # Segregate static/frozen parameters from fast weights
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})

        return static_params, all_buffers, initial_fast_weights

    # =========================================================================
    # 1. Deployment / Inference Logic (Decoupled from Training Loop)
    # =========================================================================
    def adapt_and_update(
        self,
        Xsupport: torch.Tensor,
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """
        Adapts model parameters and running buffers on the provided support set and 
        permanently updates `self.model` in-place.

        Designed for deployment or evaluation adaptation in ProtoMAML. Prototypes are extracted
        from the support set to initialize classifier weights before fine-tuning via inner optimization.

        Args:
            Xsupport (torch.Tensor): Support set input feature tensor of shape `(num_tasks, K, ...)`.
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support target label structure.
            **kwargs: Operational arguments passed down to prototype initializers and loss modules.
        """
        # Recursive device transfer helper
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        # Transfer support data to target compute device
        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        # Extract base model states
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        # Single-task adaptation closure compatible with torch.func.vmap
        def _adapt_single_task(x_s, y_s, global_buffers):
            # Clone model buffers for local task isolation
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}

            # Initialize task-specific fast weights
            fast_weights = OrderedDict(initial_fast_weights)

            # Initialize task-specific inner optimizer state (e.g., momentum)
            opt_state = self.inner_optimizer.init_state(fast_weights)

            # Initialize classifier head weights using Support Set prototypes
            fast_weights = self.model.initialize_head_weights(
                x_s,
                y_s,
                fast_weights,
                inner_step=0,
                training=False,
                task_buffers=task_buffers,
                **kwargs,
            )

            # Skip adaptation if support set is empty (Zero-Shot)
            is_zero_shot = x_s.shape[0] == 0

            if not is_zero_shot:
                # Execute inner-loop gradient adaptation steps
                for inner_step in range(self.num_inner_steps):
                    # Compute support gradients and capture updated buffers
                    grads, updated_task_buffers = self.inner_step_fn(
                        fast_weights,
                        static_params,
                        task_buffers,
                        x_s,
                        y_s,
                        inner_step=inner_step,
                        training=True,
                        **kwargs,
                    )

                    # Detach and update local task buffers
                    task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                    # Update fast weights using inner optimizer
                    fast_weights, opt_state = self.inner_optimizer(
                        fast_weights=fast_weights,
                        gradients=grads,
                        state=opt_state,
                        step=inner_step,
                        training=False
                    )

            # Perform final forward pass to capture finalized model buffers
            combined = {**fast_weights, **static_params, **task_buffers}
            with torch.no_grad():
                _, out_dict = self.compute_loss(
                    X=x_s,
                    Y=y_s,
                    model_states=combined,
                    loss_module=self.support_loss_fn,
                    inner_step=self.num_inner_steps if not is_zero_shot else 0,
                    training=False,
                    **kwargs,
                )

            # Extract finalized task buffers
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            return fast_weights, updated_task_buffers

        # Vectorize single task adaptation across task batch dimension
        vectorized_adapt = vmap(
            _adapt_single_task,
            in_dims=(0, 0, None),
            chunk_size=self.chunk_size,
            randomness="different",
        )

        # Execute parallel adaptation across all support tasks
        fast_weights_batched, batched_buffers = vectorized_adapt(Xsupport, Ysupport, all_buffers)

        # Copy adapted weights and averaged buffers back to base model in-place
        with torch.no_grad():
            # Update trainable model parameters with mean adapted values across tasks
            for name, param in self.model.named_parameters():
                if name in fast_weights_batched:
                    param.copy_(fast_weights_batched[name].mean(dim=0))

            # Update persistent model buffers handling numeric averages and boolean masks
            for name, buffer_tensor in self.model.named_buffers():
                if name in batched_buffers:
                    buf = batched_buffers[name]
                    if buf.dtype == torch.bool:
                        buffer_tensor.copy_(buf.any(dim=0))
                    else:
                        buffer_tensor.copy_(buf.mean(dim=0))

        # Clear GPU cache after parameter deployment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ ProtoMAML: Model parameters and buffers successfully adapted and updated.")

    # =========================================================================
    # 2. Main Training/Evaluation Step & Chunked Gradient Accumulation Engine
    # =========================================================================
    def _accumulate_chunked_step(
        self,
        vectorized_processor: Callable,
        Xsupport: Union[torch.Tensor, Dict],
        Ysupport: Union[torch.Tensor, Dict],
        Xquery: Union[torch.Tensor, Dict],
        Yquery: Union[torch.Tensor, Dict],
        all_buffers: Dict[str, torch.Tensor],
        training: bool,
    ) -> Tuple[float, float, Dict[str, torch.Tensor]]:
        """
        Executes manual chunked gradient accumulation over the task batch.
        Computes backward passes per chunk immediately to free autograd memory while 
        strictly preserving the true mathematical outer batch size.

        Args:
            vectorized_processor (Callable): Function mapped across tasks via torch.func.vmap.
            Xsupport (Union[torch.Tensor, Dict]): Batched support inputs.
            Ysupport (Union[torch.Tensor, Dict]): Batched support targets.
            Xquery (Union[torch.Tensor, Dict]): Batched query inputs.
            Yquery (Union[torch.Tensor, Dict]): Batched query targets.
            all_buffers (Dict[str, torch.Tensor]): Master model buffers dictionary.
            training (bool): If True, accumulates gradients via backward().

        Returns:
            Tuple[float, float, Dict[str, torch.Tensor]]:
                - total_meta_loss (float): Mean meta-loss across the entire batch.
                - total_metric (float): Mean query metric across the entire batch.
                - accumulated_buffers (Dict[str, torch.Tensor]): Weighted buffer updates for outer step.
        """
        total_meta_loss = 0.0
        total_metric = 0.0
        accumulated_buffers = {}

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
            meta_losses, metrics, batched_buffers = vectorized_processor(
                X_s_chunk, Y_s_chunk, X_q_chunk, Y_q_chunk, all_buffers
            )

            # Scale chunk loss proportionally to the global batch size
            chunk_weight = (end_idx - start_idx) / batch_size
            chunk_loss = meta_losses.sum() / batch_size

            if training:
                # Immediate backward pass frees autograd graph and activations for this chunk
                chunk_loss.backward()

                # Accumulate buffer updates across chunks
                with torch.no_grad():
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
            total_meta_loss += chunk_loss.item()
            total_metric += metrics.mean().item() * chunk_weight

            # Clear CUDA allocation cache per chunk to prevent memory spikes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return total_meta_loss, total_metric, accumulated_buffers

    def step(
        self,
        task_itrator: Any,
        training: bool = True,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """
        Executes a single outer meta-training or meta-validation step over a batch of tasks.

        Args:
            task_itrator: Dataloader iterator yielding (Xs, Ys, Xq, Yq) batches.
            training (bool): If True, computes outer gradients and steps the optimizer.
            **kwargs: Dynamic arguments such as 'epoch' and 'epochs' for scheduler weighting.

        Returns:
            Tuple[float, float]: Mean meta-loss and mean query evaluation metric.
        """
        # Initialize multi-step loss weights if enabled
        self._init_step_weights(self.num_inner_steps, training, **kwargs)

        # Retrieve next task batch
        Xs, Ys, Xq, Yq = next(task_itrator)

        # Device transfer helper
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

        # Extract fixed/trainable model states
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers):
            """Core single-task adaptation and query loss evaluation closure."""
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            meta_loss = torch.tensor(0.0, device=self.device)
            is_zero_shot = x_s.shape[0] == 0

            # 🎯 ProtoMAML Head Initialization using Support Set Prototypes
            fast_weights = self.model.initialize_head_weights(
                x_s,
                y_s,
                fast_weights,
                task_buffers=task_buffers,
                inner_step=0,
                training=False,
                **kwargs,
            )

            if not is_zero_shot:
                # Inner Adaptation Loop
                for inner_step in range(self.num_inner_steps):
                    # 🛠️ GHOST GRAPH FIX: Detach weights early for First-Order MAML

                    # Compute support gradients and capture updated buffers
                    grads, updated_task_buffers = self.inner_step_fn(
                        fast_weights,
                        static_params,
                        task_buffers,
                        x_s,
                        y_s,
                        inner_step=inner_step,
                        training=True,
                        **kwargs,
                    )

                    # Detach and update local task buffers
                    task_buffers = {k: v.detach() for k, v in updated_task_buffers.items()}

                    # Update fast weights via inner optimizer
                    fast_weights, opt_state = self.inner_optimizer(
                        fast_weights=fast_weights,
                        gradients=grads,
                        state=opt_state,
                        step=inner_step,
                        training=training
                    )

                    # Multi-Step Loss Accumulation (Intermediate Steps)
                    last_step = inner_step == self.num_inner_steps - 1
                    if self.multi_step_loss and not last_step:
                        combined = {**fast_weights, **static_params, **task_buffers}
                        if training:
                            (q_step_loss, _), _ = self.compute_loss(
                                X=x_q,
                                Y=y_q,
                                model_states=combined,
                                loss_module=self.query_loss_fn,
                                inner_step=inner_step,
                                training=True,
                                **kwargs,
                            )
                        else:
                            with torch.no_grad():
                                (q_step_loss, _), _ = self.compute_loss(
                                    X=x_q,
                                    Y=y_q,
                                    model_states=combined,
                                    loss_module=self.query_loss_fn,
                                    inner_step=inner_step,
                                    training=False,
                                    **kwargs,
                                )
                        meta_loss = self._update_meta_loss(meta_loss, q_step_loss, inner_step)

            # Final Query Loss & Metric Evaluation on adapted weights
            combined = {**fast_weights, **static_params, **task_buffers}
            if training:
                (q_step_loss, q_metric), out_dict = self.compute_loss(
                    X=x_q,
                    Y=y_q,
                    model_states=combined,
                    loss_module=self.query_loss_fn,
                    inner_step=self.num_inner_steps if not is_zero_shot else 0,
                    training=True,
                    **kwargs,
                )
            else:
                with torch.no_grad():
                    (q_step_loss, q_metric), out_dict = self.compute_loss(
                        X=x_q,
                        Y=y_q,
                        model_states=combined,
                        loss_module=self.query_loss_fn,
                        inner_step=self.num_inner_steps if not is_zero_shot else 0,
                        training=False,
                        **kwargs,
                    )

            # Safely index step weights and aggregate into final meta-loss
            target_step_idx = max(0, self.num_inner_steps - 1)
            meta_loss = self._update_meta_loss(meta_loss, q_step_loss, target_step_idx)

            # Capture finalized buffers
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            return meta_loss, q_metric, updated_task_buffers

        # Vectorize single-task processor across the task batch dimension
        vectorized_processor = vmap(
            process_single_task,
            in_dims=(0, 0, 0, 0, None),
            randomness="different",
        )

        # Reset outer optimizer gradients before chunk accumulation
        if training:
            self.optimizer.zero_grad()

        # Execute manual chunked accumulation engine
        total_meta_loss, total_metric, accumulated_buffers = self._accumulate_chunked_step(
            vectorized_processor,
            Xsupport,
            Ysupport,
            Xquery,
            Yquery,
            all_buffers,
            training,
        )

        # Step outer meta-optimizer and copy finalized buffers
        if training:
            self.optimizer.step()
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in accumulated_buffers:
                        buffer_tensor.copy_(accumulated_buffers[name])

        return total_meta_loss, total_metric

    def _init_step_weights(self, num_inner_steps: int, training: bool, **kwargs: Any) -> None:
        """
        Initializes per-step query loss annealing weights if multi-step loss is enabled.
        """
        if training and self.multi_step_loss and num_inner_steps > 0:
            self.step_weights = get_per_step_loss_weights(
                num_inner_steps, kwargs.get("epoch"), kwargs.get("epochs"), self.device
            )

    def _update_meta_loss(
        self,
        current_meta_loss: torch.Tensor,
        q_loss: torch.Tensor,
        inner_step: int,
    ) -> torch.Tensor:
        """
        Accumulates intermediate query loss into current_meta_loss with step weighting.

        Args:
            current_meta_loss (torch.Tensor): Accumulated scalar loss tensor.
            q_loss (torch.Tensor): Calculated query loss for the current step.
            inner_step (int): Current inner loop iteration index.

        Returns:
            torch.Tensor: Updated meta-loss tensor.
        """
        if self.step_weights is not None and self.multi_step_loss:
            return current_meta_loss + (q_loss / self.step_weights[inner_step])
        return current_meta_loss + q_loss