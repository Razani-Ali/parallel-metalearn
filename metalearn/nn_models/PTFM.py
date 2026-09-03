# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# This module implements the Prototypical Time-Frequency Mixer (PTFM) feature 
# extraction backbone and baseline architectures proposed by Wang et al. in:
#
# [1] D. Wang, T. Wang, and X. Wang, "Few-Shot Fault Diagnosis for Industrial 
#     Robot Transmission Systems via a Prototypical Time-Frequency Mixer," 
#     IEEE Access, vol. 13, pp. 178045-178059, 2025. 
#     DOI: 10.1109/ACCESS.2025.3620386.
#
# Official Open-Source Repository:
#     https://github.com/DannieW727/fewshot-robot-fault.git
#
# LICENSE NOTICE (CC BY 4.0):
# The original work and publication are licensed under the Creative Commons 
# Attribution 4.0 International License (CC BY 4.0). Under this license, 
# reproduction and modification are permitted provided appropriate credit is given.
#
# FRAMEWORK INTEGRATION NOTICE:
# This code adapts the model definitions from the official repository to fit our 
# modular Meta-Learning framework (specifically integrating with our functional 
# PrototypicalNetwork optimizer, vectorized vmap engine, and ProtoNet_Model wrapper).
# Redundant standalone loss and evaluation loops have been decoupled to rely on 
# our centralized distance and loss calculators.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Feedforward network with GELU activation used within MLP-Mixer blocks."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MixerBlock(nn.Module):
    """
    MLP-Mixer Block performing sequential token and channel mixing.
    Matches the official repository structure.
    """
    def __init__(self, num_tokens: int, dim: int, token_mlp_dim: int = 64, channel_mlp_dim: int = 64):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens)
        )

        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Token mixing: (B, tokens, dim) -> (B, dim, tokens)
        y = x.permute(0, 2, 1)
        y = self.token_norm(y)
        y = self.token_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y

        # Channel mixing
        y = self.channel_norm(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class PTFM_Backbone(nn.Module):
    """
    Dual-branch Time-Frequency Mixer Backbone for Prototypical Networks.
    Processes 1D vibration signals into compact latent embeddings.
    """
    def __init__(self, input_length: int = 1250, dim: int = 128, mixer_tokens: int = 8):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.mixer_tokens = mixer_tokens

        # Time-domain projection branch
        self.temporal_proj = nn.Linear(input_length, dim)

        # Frequency-domain (rFFT) projection branch
        self.fft_proj = nn.Linear(input_length // 2 + 1, dim)
        
        # Token expansion: maps 2 domain tokens into T mixer tokens
        self.fusion_proj = nn.Linear(2, mixer_tokens)

        # Cross-domain token-channel mixer
        self.mixer = MixerBlock(
            num_tokens=mixer_tokens, 
            dim=dim, 
            token_mlp_dim=64, 
            channel_mlp_dim=64
        )

        # Flatten & projection to final latent embedding
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Accommodate (Batch, 1, Length) and (Batch, Length) shapes
        if x.dim() == 3:
            x = x.squeeze(1)

        # Temporal branch projection
        h_time = self.temporal_proj(x)

        # Spectral feature extraction and projection
        fft_feat = torch.fft.rfft(x, dim=1).abs()
        h_fft = self.fft_proj(fft_feat)

        # Modality concatenation & token expansion
        h_stack = torch.stack([h_time, h_fft], dim=1)
        h_fused = self.fusion_proj(h_stack.permute(0, 2, 1)).permute(0, 2, 1)

        # Interaction & output embedding
        h_mixed = self.mixer(h_fused)
        out = self.output_proj(h_mixed)
        return out


# ==============================================================================
# ABLATION & BASELINE BACKBONES (FROM OFFICIAL BENCHMARK)
# ==============================================================================

class PTFM_NoFFT_Backbone(nn.Module):
    """Ablation Variant A1: Removes FFT branch, retaining only temporal projection."""
    def __init__(self, input_length: int = 1250, dim: int = 128, mixer_tokens: int = 10):
        super().__init__()
        self.input_length = input_length
        self.dim = dim
        self.mixer_tokens = mixer_tokens

        self.temporal_proj = nn.Linear(input_length, dim)
        self.token_proj = nn.Linear(dim, dim)
        self.mixer = MixerBlock(num_tokens=mixer_tokens, dim=dim, token_mlp_dim=64, channel_mlp_dim=64)
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        h_time = self.temporal_proj(x)
        h_repeat = h_time.unsqueeze(1).repeat(1, self.mixer_tokens, 1)
        h_proj = self.token_proj(h_repeat)
        h_mixed = self.mixer(h_proj)
        return self.output_proj(h_mixed)


class SimpleConvBlock(nn.Module):
    """Convolutional block replacing MLP-Mixer in Ablation Variant A2."""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


class PTFM_NoMixer_Backbone(nn.Module):
    """Ablation Variant A2: Replaces MixerBlock with SimpleConvBlock."""
    def __init__(self, input_length: int = 1250, dim: int = 128, mixer_tokens: int = 10):
        super().__init__()
        self.temporal_proj = nn.Linear(input_length, dim)
        self.fft_proj = nn.Linear(input_length // 2 + 1, dim)
        self.fusion_proj = nn.Linear(2, mixer_tokens)
        self.conv_block = SimpleConvBlock(dim=dim)
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mixer_tokens * dim, dim)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        h_time = self.temporal_proj(x)
        h_fft = self.fft_proj(torch.fft.rfft(x, dim=1).abs())
        h_stack = torch.stack([h_time, h_fft], dim=1)
        h_fused = self.fusion_proj(h_stack.permute(0, 2, 1)).permute(0, 2, 1)
        h_conv = self.conv_block(h_fused)
        return self.output_proj(h_conv)


class CNN_Backbone(nn.Module):
    """Standard 1D-CNN baseline encoder."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x).squeeze(-1)