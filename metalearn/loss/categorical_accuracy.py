import torch
import torch.nn as nn


class CategoricalAccuracy(nn.Module):
    """
    Vectorized Categorical Accuracy Metric compatible with PyTorch vmap.
    
    Computes the ratio of correctly predicted target classes.
    Designed without dynamic shapes, loop iterations, or non-functional mutations 
    to ensure pure compatibility with torch.func.vmap and multi-task batching.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Predicted class logits of shape (..., num_classes).
            targets (torch.Tensor): Ground truth target indices of shape (...).

        Returns:
            torch.Tensor: Scalar accuracy value (or tensor of accuracies if vmapped over tasks).
        """
        # 1. Extract predicted class indices along the final dimension (num_classes)
        preds = torch.argmax(logits, dim=-1)

        # 2. Check element-wise equality between predictions and target indices
        correct_predictions = torch.eq(preds, targets).to(torch.float32)
        correct_predictions = correct_predictions * mask if mask else correct_predictions
        # 3. Compute mean accuracy across all target instances
        accuracy = torch.mean(correct_predictions)

        return accuracy
