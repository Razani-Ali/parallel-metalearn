from typing import Dict, Optional, Tuple
import torch
from metalearn.loss.base import BaseLoss
from .BaseLearner import MetaOptimizer
from torch.func import vmap
from ..model_wrappers.PrototypicalWrapper import ProtoNet_Model


class PrototypicalNetwork(MetaOptimizer):
    """
    Prototypical Networks Meta-Learning Algorithm.

    Executes purely functional, parameter-free inner adaptation via prototype 
    extraction. Highly accelerated through vectorized task processing using `vmap`.
    """

    def __init__(
        self,
        *,
        model: ProtoNet_Model,
        optimizer: torch.optim.Optimizer,
        loss_fn: BaseLoss,
        chunk_size: int = 8,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initializes the Prototypical Networks meta-optimizer.

        Args:
            model (ProtoNet_Model): The ProtoNet wrapper model.
            optimizer (torch.optim.Optimizer): The outer-loop meta-optimizer (e.g., Adam).
            loss_fn (BaseLoss): Loss module used for query evaluation (e.g., CrossEntropy).
            chunk_size (int): Chunk size for vmap memory management.
            device (Optional[torch.device]): Target compute device.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.chunk_size = chunk_size

        # Device assignment
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def compute_loss(
        self, 
        *, 
        x_s: torch.Tensor,
        y_s: Dict[str, torch.Tensor],
        x_q: torch.Tensor, 
        y_q: Dict[str, torch.Tensor], 
        model_states: Dict[str, torch.Tensor],
        training: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a functional forward pass and evaluates the loss.
        """
        forward_kwargs = {"training": training, "num_step": 0}
        
        # Trigger functional execution (Processes both Support and Query inherently)
        out_dict = torch.func.functional_call(self.model, model_states, (x_q, x_s, y_s), forward_kwargs)
        
        # Calculate cross-entropy loss over query logits
        return self.loss_fn(out_dict=out_dict, targets=y_q, model_states=model_states, **kwargs), out_dict

    def step(
        self, 
        task_itrator, 
        training: bool = True, 
        **kwargs
    ) -> Tuple[float, float]:
        """
        Executes a single meta-training or meta-validation step over a batch of tasks.
        """
        # Fetch batch
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

        # Extract parameters for functional call
        model_states = dict(self.model.named_parameters())
        model_buffers = dict(self.model.named_buffers())

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers):
            """Core execution logic for a single task."""
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            # ProtoNet calculates prototypes and query loss directly in one step!
            combined = {**model_states, **task_buffers}
            (loss, metric), out_dict = self.compute_loss(
                x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q, 
                model_states=combined, training=training, **kwargs
            )
            updated_task_buffers = out_dict.get("buffers", task_buffers)

            return loss, metric, updated_task_buffers

        # Vectorize single task execution over the batch dimension
        vectorized_processor = vmap(
            process_single_task, in_dims=(0, 0, 0, 0, None),
            randomness="different", chunk_size=self.chunk_size
        )

        # Launch parallelized computation
        meta_losses, metrics, batched_buffers = vectorized_processor(
            Xsupport, Ysupport, Xquery, Yquery, model_buffers)

        # Backpropagation and optimization
        if training:
            self.optimizer.zero_grad()
            meta_losses.mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in batched_buffers:
                        mean_buf = batched_buffers[name].mean(dim=0)
                        buffer_tensor.copy_(mean_buf)

        return (
            meta_losses.mean().item(),
            metrics.mean().item(),
        )

    def adapt_and_update(self, Xsupport: torch.Tensor, Ysupport: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Deployment behavior for Prototypical Networks.
        Extracts prototypes from the support set via vmap, updates running buffers, 
        and saves aggregated prototypes in persistent deployed buffers for inference.
        """
        Xsupport = Xsupport.to(self.device)
        Ysupport = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in Ysupport.items()
        }

        params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())

        def _adapt_single_task(x_s, y_s, global_buffers):
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            states = {**params, **task_buffers}
            forward_kwargs = {"training": False, "num_step": 0}

            # Forward pass through backbone using functional_call
            feat_s = torch.func.functional_call(self.model.backbone, states, (x_s,), forward_kwargs)
            feat_s_flat = feat_s.flatten(start_dim=-1) if feat_s.dim() > 2 else feat_s

            # Compute task prototypes and class validity mask
            labels_s = y_s["labels"]
            samples_mask_s = y_s.get("samples_mask", None)

            prototypes, class_mask = self.model.center_head.compute_class_centers(
                features=feat_s_flat,
                labels=labels_s,
                samples_mask=samples_mask_s,
                training=False,
                **kwargs
            )

            # Retrieve updated model buffers (including running_prototypes if enabled)
            out_buffers = dict(self.model.named_buffers())

            return prototypes, class_mask, out_buffers

        # Vectorize adaptation across task batch dimension
        vectorized_adapt = vmap(_adapt_single_task, in_dims=(0, 0, None), chunk_size=self.chunk_size)

        with torch.no_grad():
            self.model.eval()
            batched_prototypes, batched_masks, batched_buffers = vectorized_adapt(
                Xsupport, Ysupport, all_buffers
            )

            # Average computed prototypes and aggregate active class mask across tasks
            avg_prototypes = batched_prototypes.mean(dim=0)
            combined_class_mask = batched_masks.any(dim=0)

            # Securely register prototypes and active masks for inference deployment
            self.model.deployed_prototypes = avg_prototypes
            self.model.deployed_class_mask = combined_class_mask

            # Sync running buffers (running_prototypes, BatchNorm stats, etc.) back to model
            for name, buffer_tensor in self.model.named_buffers():
                if name in batched_buffers:
                    buffer_tensor.copy_(batched_buffers[name].mean(dim=0))

            print("✅ ProtoNet: Prototypes and running buffers securely computed and registered for deployment.")