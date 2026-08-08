from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
from metalearn.loss.base import BaseLoss
from .categorical_accuracy import CategoricalAccuracy


class CrossEntropy(BaseLoss):
    """
    Standard Categorical Cross-Entropy Loss with Accuracy metric calculation.
    """
    def __init__(self, metric_fn: Optional[callable] = None):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.metric_fn = metric_fn or CategoricalAccuracy

    def forward(
        self,
        out_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model_states: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if "logits" not in out_dict:
            raise KeyError("The 'out_dict' must contain the 'logits' key.")

        if "labels" not in targets:
            raise KeyError("The 'targets' must contain the 'labels' key.")
        
        logits = out_dict["logits"]
        labels = targets["labels"]

        if "samples_mask" in targets:
            mask = targets["samples_mask"]
            logits, labels = logits[mask], labels[mask]

        loss = self.ce_loss_fn(logits, labels)

        metric = self.metric_fn(logits, labels)

        if not isinstance(metric, torch.Tensor):
            metric = torch.tensor(metric, device=loss.device, dtype=torch.float32)

        return loss, metric