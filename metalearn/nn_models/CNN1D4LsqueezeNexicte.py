import torch
import torch.nn as nn
from metalearn.nn_models.basic_layers import SqueezeAndExcitation1D
from metalearn.nn_models.basic_layers.BatchNormalization import BatchNorm

# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# This implementation incorporates 1D-CNN backbones with Squeeze-and-Excitation 
#   channel attention modules proposed in:
#
# [1] X. Yang, L. Zhang, and J. Wang, "Task-Agnostic Generalized Meta-learning 
#     Based on MAML for Few-Shot Bearing Fault Diagnosis," in Image and Graphics 
#     (ICIG 2023), Lecture Notes in Computer Science, vol. 14355, 
#     Springer, Cham, 2023, pp. 118–129. 
#     DOI: 10.1007/978-3-031-46305-1_10.
#
# ==============================================================================

class ConvBlock1D(nn.Module):
    """Standard 1D Convolutional Block: Conv1D -> BatchNorm1d -> ReLU -> MaxPool1d"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, UseBatchNormalization: bool = True,
                 use_per_step_stats=False,
                 track_running_stats=False,
                 affine=True,
                 max_inner_steps=1, **kwargs):
        
        super().__init__()

        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding)

        self.bn = BatchNorm(out_channels, use_per_step_stats=use_per_step_stats,
                            track_running_stats=track_running_stats, affine=affine,
                            max_inner_steps=max_inner_steps) if UseBatchNormalization else None

        self.relu = nn.ReLU(inplace=False)

        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x, **kwargs)

        x = self.relu(x)

        return self.pooling(x)



class TAGML_Backbone(nn.Module):
    """
    Backbone architecture conforming to Figure 1 in the TAGML paper:
    Encoder (ConvBlocks) -> SE Module -> Flatten.
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_channels: int = 64, 
        num_blocks: int = 4, 
        se_reduction: int = 16,
        **block_kwargs
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        c_in = in_channels
        for _ in range(num_blocks):
            self.blocks.append(ConvBlock1D(c_in, hidden_channels, **block_kwargs))
            c_in = hidden_channels
        
        # SE Attention Module (Transforms Feature Map 1 to Feature Map 2)
        self.se_module = SqueezeAndExcitation1D(channels=hidden_channels, reduction_ratio=se_reduction)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input shape: (Batch, Channels, Signal_Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Encoder forward: yields Feature map 1
        for block in self.blocks:
            x = block(x, **kwargs)
        
        # SE module forward: yields Feature map 2
        x2 = self.se_module(x)
        
        # Flatten representation
        features = torch.flatten(x2, start_dim=1)
        return features