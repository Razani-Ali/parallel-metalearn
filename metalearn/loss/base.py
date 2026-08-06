from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """
    Abstract Base Class for all Loss functions in the framework.
    All custom loss classes must inherit from this class to ensure
    signature and output consistency across different algorithms (e.g., MAML).
    """

    @abstractmethod
    def forward(
        self,
        out_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        model_states: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates loss and evaluation metric.

        Args:
            out_dict: Dictionary containing model outputs (e.g., {"logits": tensor, "features": tensor}).
            targets: Dictionary containing ground truth targets (e.g., {"labels": tensor}).
            model_states: Dict of parameter tensors (e.g., fast_weights/combined_params in meta-learning).
            **kwargs: Additional contextual arguments (e.g., inner_step, training, epoch).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (loss, metric), both as scalar Tensors.
        """
        pass