import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductSelfAttention(nn.Module):
    """
    Scaled Dot-Product Self-Attention module.
    
    Computes query, key, and value linear projections and applies scaled dot-product 
    attention mechanism. Fully stateless and compatible with functional execution.
    """

    def __init__(self, hidden_dim: int, **kwargs):
        """
        Initializes the DotProductSelfAttention module.

        Args:
            hidden_dim (int): Dimensionality of input representations and projections.
        """
        # Call parent constructor
        super(DotProductSelfAttention, self).__init__()
        
        # Linear projection layer for Query
        self.query = nn.Linear(hidden_dim, hidden_dim)
        # Linear projection layer for Key
        self.key = nn.Linear(hidden_dim, hidden_dim)
        # Linear projection layer for Value
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor scalar (sqrt(d_k)) to stabilize gradients
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for Scaled Dot-Product Self-Attention.

        Args:
            x (torch.Tensor): Input sequence tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            torch.Tensor: Attended sequence representations of shape [batch_size, seq_len, hidden_dim].
        """
        # Project input tensor into Query representations: [batch_size, seq_len, hidden_dim]
        q = self.query(x)
        # Project input tensor into Key representations: [batch_size, seq_len, hidden_dim]
        k = self.key(x)
        # Project input tensor into Value representations: [batch_size, seq_len, hidden_dim]
        v = self.value(x)

        # Compute raw attention scores via matrix multiplication: Q * K^T
        # Shape: [batch_size, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Convert attention scores to probabilities via Softmax over last dimension
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute weighted sum over Value representations: Softmax(Q * K^T / scale) * V
        # Shape: [batch_size, seq_len, hidden_dim]
        out = torch.matmul(attn_probs, v)

        return out