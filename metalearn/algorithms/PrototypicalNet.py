from typing import Dict, Optional, Tuple, Union, Callable, Any
import torch
from torch.func import vmap
from metalearn.loss.base import BaseLoss
from metalearn.algorithms.BaseLearner import MetaOptimizer
from metalearn.model_wrappers.PrototypicalWrapper import ProtoNet_Model


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# Snell, J., Swersky, K., & Zemel, R. (2017).
# "Prototypical Networks for Few-shot Learning."
# Advances in Neural Information Processing Systems (NeurIPS), 2017.
# arXiv:1703.05175.
# ==============================================================================


class PrototypicalNetwork(MetaOptimizer):
    """
    Prototypical Networks Meta-Learning Algorithm.

    Executes a parameter-free, metric-based inner adaptation by computing the mean feature 
    vector (prototype) for each class in the support set and classifying query samples 
    based on their Euclidean/Cosine distance to these prototypes.

    Key Features:
    - Stateless vectorized task computation via `torch.func.vmap`.
    - Memory-efficient manual chunked gradient accumulation ensuring OOM-free execution.
    - Zero computation graph overhead during pure inference/evaluation steps.
    - Robust synchronization and persistence of running buffers (e.g., BatchNorm).
    """

    def __init__(
        self,
        *,
        model: ProtoNet_Model,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        chunk_size: int = 8,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        """
        Initializes the Prototypical Networks meta-optimizer module.

        Args:
            model (ProtoNet_Model): The ProtoNet neural network model wrapper.
            optimizer (torch.optim.Optimizer): The outer-loop meta-optimizer (e.g., Adam).
            loss_fn (BaseLoss): Loss module used for query set evaluation (e.g., CrossEntropy).
            chunk_size (int): Chunk size for memory management during task batch execution.
            device (Optional[torch.device]): Target hardware compute device (CPU or GPU).
            **kwargs: Additional operational keyword arguments.
        """
        # Initialize base MetaOptimizer class
        super().__init__()

        # Store model, optimizer, and loss evaluation modules
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.chunk_size = chunk_size

        # Resolve target device and transfer modules
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def compute_loss(
        self,
        *,
        x_s: torch.Tensor,
        y_s: Union[torch.Tensor, Dict[str, torch.Tensor]],
        x_q: torch.Tensor,
        y_q: Union[torch.Tensor, Dict[str, torch.Tensor]],
        model_states: Dict[str, torch.Tensor],
        training: bool = False,
        **kwargs: Any,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Executes a functional forward pass and evaluates loss and classification metrics.

        Args:
            x_s (torch.Tensor): Support set input feature tensor.
            y_s (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support target label structure.
            x_q (torch.Tensor): Query set input feature tensor.
            y_q (Union[torch.Tensor, Dict[str, torch.Tensor]]): Query target label structure.
            model_states (Dict[str, torch.Tensor]): Complete model parameters and buffers dictionary.
            training (bool): Flag indicating training mode for dropout and normalization layers.
            **kwargs: Additional context arguments.

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
                - ((loss_tensor, metric_tensor), model_forward_output_dictionary)
        """
        # Pack operational context flags to ensure functional_call compatibility
        forward_kwargs = {"training": training, "num_step": 0}

        # Stateless functional execution: passes Query, Support, and Support targets to ProtoNet wrapper
        out_dict = torch.func.functional_call(self.model, model_states, (x_q, x_s, y_s), forward_kwargs)

        # Calculate loss and metric over query set predictions
        loss_and_metric = self.loss_fn(out_dict=out_dict, targets=y_q, model_states=model_states, **kwargs)

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
    ) -> Tuple[float, float, Dict[str, torch.Tensor]]:
        """
        Executes manual chunked gradient accumulation across the task batch.
        Computes backward passes per chunk immediately to free autograd memory while 
        strictly preserving the true mathematical outer batch size.

        Args:
            vectorized_processor (Callable): Function mapped across tasks via torch.func.vmap.
            Xsupport (Union[torch.Tensor, Dict]): Batched support set inputs.
            Ysupport (Union[torch.Tensor, Dict]): Batched support set targets.
            Xquery (Union[torch.Tensor, Dict]): Batched query set inputs.
            Yquery (Union[torch.Tensor, Dict]): Batched query set targets.
            all_buffers (Dict[str, torch.Tensor]): Master model buffers dictionary.
            training (bool): If True, accumulates gradients via backward().

        Returns:
            Tuple[float, float, Dict[str, torch.Tensor]]:
                - total_meta_loss (float): Mean loss across the entire batch.
                - total_metric (float): Mean evaluation metric across the entire batch.
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

            # Scale chunk loss proportionally to global batch size
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
            **kwargs: Context parameters forwarded to loss calculation.

        Returns:
            Tuple[float, float]: Mean meta-loss and mean query evaluation metric.
        """
        # Fetch next task batch from iterator
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

        # Extract parameters and buffers as explicit dictionaries
        model_states = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers):
            """Core execution loop for a single task within vmap."""
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            combined = {**model_states, **task_buffers}

            # 🛡️ EVALUATION FIX: Do not build graph if not training
            if training:
                (loss, metric), out_dict = self.compute_loss(
                    x_s=x_s,
                    y_s=y_s,
                    x_q=x_q,
                    y_q=y_q,
                    model_states=combined,
                    training=True,
                    **kwargs,
                )
            else:
                with torch.no_grad():
                    (loss, metric), out_dict = self.compute_loss(
                        x_s=x_s,
                        y_s=y_s,
                        x_q=x_q,
                        y_q=y_q,
                        model_states=combined,
                        training=False,
                        **kwargs,
                    )

            # Safely detach buffers before escaping vmap closure
            raw_buffers = out_dict.get("buffers", task_buffers)
            updated_task_buffers = {k: v.detach() for k, v in raw_buffers.items()}

            return loss, metric, updated_task_buffers

        # Vectorize single-task processing over the task batch dimension
        vectorized_processor = vmap(
            process_single_task,
            in_dims=(0, 0, 0, 0, None),
            randomness="different",
        )

        # Reset outer optimizer gradients before accumulation
        if training:
            self.optimizer.zero_grad()

        # Execute manual chunked gradient accumulation engine
        total_meta_loss, total_metric, accumulated_buffers = self._accumulate_chunked_step(
            vectorized_processor,
            Xsupport,
            Ysupport,
            Xquery,
            Yquery,
            all_buffers,
            training,
        )

        # Step optimizer and update running buffers
        if training:
            self.optimizer.step()
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in accumulated_buffers:
                        buffer_tensor.copy_(accumulated_buffers[name])

        return total_meta_loss, total_metric

    def adapt_and_update(
        self,
        Xsupport: torch.Tensor,
        Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """
        Deployment behavior for Prototypical Networks.
        Extracts prototypes from the support set via vmap, updates running buffers, 
        and registers aggregated prototypes in persistent deployed buffers for inference.

        Args:
            Xsupport (torch.Tensor): Support set inputs tensor of shape `(num_tasks, K, ...)`.
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support target label structure.
            **kwargs: Context arguments forwarded to prototype computation.
        """
        # Device transfer helper
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())

        def _adapt_single_task(x_s, y_s, global_buffers):
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            states = {**params, **task_buffers}
            forward_kwargs = {"training": False, "num_step": 0}

            # Extract representations via functional execution of backbone
            feat_s = torch.func.functional_call(self.model.backbone, states, (x_s,), forward_kwargs)
            feat_s_flat = feat_s.flatten(start_dim=1) if feat_s.dim() > 2 else feat_s

            labels_s = y_s["labels"] if isinstance(y_s, dict) else y_s
            samples_mask_s = y_s.get("samples_mask", None) if isinstance(y_s, dict) else None

            # Compute class prototypes and capture updated center buffers
            prototypes, class_mask, center_bufs = self.model.center_head.compute_class_centers(
                features=feat_s_flat,
                labels=labels_s,
                samples_mask=samples_mask_s,
                task_buffers=task_buffers,
                prefix="center_head.",
                training=False,
                **kwargs,
            )

            task_buffers.update(center_bufs)
            updated_task_buffers = {k: v.detach() for k, v in task_buffers.items()}

            return prototypes, class_mask, updated_task_buffers

        # Vectorize deployment adaptation across task batch dimension
        vectorized_adapt = vmap(
            _adapt_single_task,
            in_dims=(0, 0, None),
            chunk_size=self.chunk_size,
            randomness="different",
        )

        with torch.no_grad():
            self.model.eval()
            batched_prototypes, batched_masks, batched_buffers = vectorized_adapt(
                Xsupport, Ysupport, all_buffers
            )

            # Aggregate prototypes and active class masks across tasks
            avg_prototypes = batched_prototypes.mean(dim=0)
            combined_class_mask = batched_masks.any(dim=0)

            # Register deployed prototypes onto model for standalone deployment
            self.model.deployed_prototypes = avg_prototypes
            self.model.deployed_class_mask = combined_class_mask

            # Synchronize running buffers respecting data types
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

        print("✅ ProtoNet: Prototypes and running buffers securely computed and registered for deployment.")