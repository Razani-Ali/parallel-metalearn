import torch
import torch.nn as nn

# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# This implementation incorporates Squeeze-and-Excitation module formulation proposed in:
#
# [1] X. Yang, L. Zhang, and J. Wang, "Task-Agnostic Generalized Meta-learning 
#     Based on MAML for Few-Shot Bearing Fault Diagnosis," in Image and Graphics 
#     (ICIG 2023), Lecture Notes in Computer Science, vol. 14355, 
#     Springer, Cham, 2023, pp. 118–129. 
#     DOI: 10.1007/978-3-031-46305-1_10.
#
# ==============================================================================

class SqueezeAndExcitation1D(nn.Module):
    """
    Squeeze-and-Excitation (SE) Module for 1D feature maps.
    Based on Equations (3) & (4) in the TAGML paper.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        # Global spatial/temporal average pooling
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        
        reduced_channels = max(1, channels // reduction_ratio)
        
        # Excitation block: W1 (down-projection) -> ReLU -> W2 (up-projection) -> Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, length)
        b, c, _ = x.size()
        
        # Squeeze operation: (b, c, l) -> (b, c, 1) -> (b, c)
        q = self.squeeze(x).view(b, c)
        
        # Excitation operation: (b, c) -> (b, c) -> (b, c, 1)
        z = self.excitation(q).view(b, c, 1)
        
        # Channel-wise recalibration (broadcast multiply)
        return x * z