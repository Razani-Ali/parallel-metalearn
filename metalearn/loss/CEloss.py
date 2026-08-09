from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from metalearn.loss.base import BaseLoss
from .categorical_accuracy import CategoricalAccuracy


class CrossEntropy(BaseLoss):
    """
    Standard Categorical Cross-Entropy Loss module with integrated accuracy metric evaluation.
    
    Supports masked evaluation to handle padded tasks and variable batch sizes cleanly 
    within vectorized PyTorch vmap pipelines.
    """
    def __init__(self, metric_fn: Optional[callable] = None):
        """
        Initializes the Categorical CrossEntropy loss class.

        Args:
            metric_fn (Optional[callable]): Custom evaluation metric function. Defaults to CategoricalAccuracy.
        """
        # Call base loss module constructor
        super().__init__()
        # Initialize default PyTorch unweighted CrossEntropyLoss instance
        self.ce_loss_fn = nn.CrossEntropyLoss()
        # Assign custom metric module or default to CategoricalAccuracy
        self.metric_fn = metric_fn or CategoricalAccuracy

    def forward(
        self,
        out_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model_states: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes categorical cross-entropy loss and classification metric value.

        Args:
            out_dict (Dict[str, torch.Tensor]): Model predictions containing logits tensor under 'logits' key.
            targets (torch.Tensor or Dict): Targets dictionary containing target labels tensor under 'labels'.
            model_states (Optional[Dict[str, torch.Tensor]]): Functional model state dict (optional).
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing scalar Loss and Metric tensors.
        """
        # Ensure model output dictionary contains necessary classification logits
        if "logits" not in out_dict:
            raise KeyError("The 'out_dict' must contain the 'logits' key.")

        # Ensure target object contains ground-truth label tensor
        if "labels" not in targets:
            raise KeyError("The 'targets' must contain the 'labels' key.")
        
        # Extract raw network logits from output dictionary
        logits = out_dict["logits"]
        # Extract target label tensor from target structure
        labels = targets["labels"]

        # Check whether target dictionary provides a binary mask for padded/dummy samples
        if "samples_mask" in targets:
            # Cast binary samples mask to float32 tensor for multiplication
            mask = targets["samples_mask"].to(torch.float32)

            # Compute unreduced cross-entropy loss for each sample individually
            raw_loss = F.cross_entropy(logits, labels, reduction='none')
            # Zero out padded sample losses using mask and normalize by total valid samples
            loss = (raw_loss * mask).sum() / (mask.sum() + 1e-8)

        else:
            # Assign None to mask when no dummy padding is present
            mask = None
            # Compute standard reduction cross-entropy loss over unpadded samples
            loss = self.ce_loss_fn(logits, labels)

        # Compute accuracy or custom metric using predicted logits, true labels, and sample mask
        metric = self.metric_fn(logits, labels, mask)

        # Ensure computed metric is returned as a valid PyTorch tensor on correct device
        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric, device=loss.device, dtype=torch.float32)

        # Return computed loss and evaluation metric tuple
        return loss, metric