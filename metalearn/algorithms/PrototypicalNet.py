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
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a functional forward pass and evaluates the loss.
        """
        forward_kwargs = kwargs.get("kwargs_to_forward", {})
        
        # Trigger functional execution (Processes both Support and Query inherently)
        out_dict = torch.func.functional_call(self.model, model_states, (x_q, x_s, y_s), forward_kwargs)
        
        # Calculate cross-entropy loss over query logits
        return self.loss_fn(out_dict=out_dict, targets=y_q, model_states=model_states, **kwargs)

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
        all_states = {**model_states, **model_buffers}

        def process_single_task(x_s, y_s, x_q, y_q):
            """Core execution logic for a single task."""
            # ProtoNet calculates prototypes and query loss directly in one step!
            loss, metric = self.compute_loss(
                x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q, 
                model_states=all_states, **kwargs
            )
            return loss, metric

        # Vectorize single task execution over the batch dimension
        vectorized_processor = vmap(
            process_single_task, in_dims=(0, 0, 0, 0),
            randomness="different", chunk_size=self.chunk_size
        )

        # Launch parallelized computation
        meta_losses, metrics = vectorized_processor(Xsupport, Ysupport, Xquery, Yquery)

        # Backpropagation and optimization
        if training:
            self.optimizer.zero_grad()
            meta_losses.mean().backward()
            self.optimizer.step()

        return (
            meta_losses.mean().item(),
            metrics.mean().item(),
        )

    def adapt_and_update(self, Xsupport: torch.Tensor, Ysupport: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Deployment behavior for Prototypical Networks.
        Extracts prototypes from the support set and saves them in a persistent buffer 
        for future inference/query predictions.
        """
        Xsupport = Xsupport.to(self.device)
        Ysupport = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in Ysupport.items()}
        
        # Note: True episodic evaluation during meta-validation occurs in step(training=False).
        # This method is purely for finalizing the model into a standard deployed classifier.
        with torch.no_grad():
            self.model.eval()
            feat_s = self.model.backbone(Xsupport)
            prototypes, class_mask = self.model.center_head.compute_class_centers(
                features=feat_s, 
                labels=Ysupport["labels"], 
                samples_mask=Ysupport.get("samples_mask", None)
            )
            
            # Register computed prototypes directly onto the model for inference use
            self.model.deployed_prototypes = prototypes
            self.model.deployed_class_mask = class_mask
            print("✅ Prototypes securely computed and registered for inference deployment.")