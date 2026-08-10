import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UniGRU(nn.Module):
    """
    Unidirectional Gated Recurrent Unit (GRU).
    
    A purely functional implementation optimized for PyTorch 2.0+ (torch.compile, vmap).
    Uses torch.unbind to iterate over the time dimension efficiently.
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        """
        Initializes the Unidirectional GRU.

        Args:
            input_dim (int): Number of features in the input sequence.
            hidden_dim (int): Number of features in the hidden state.
        """
        # Initialize parent nn.Module
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input-to-hidden weights (Reset, Update, New gates) -> [3 * hidden, input]
        self.w_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        # Hidden-to-hidden weights -> [3 * hidden, hidden]
        self.w_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        
        # Biases -> [3 * hidden]
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_dim))

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Xavier and Orthogonal initialization."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w_ih)
            nn.init.orthogonal_(self.w_hh)
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            # Set update gate bias to 1.0 to prevent early gradient vanishing
            self.bias_ih[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, hx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Unidirectional GRU.

        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len, input_dim].
            hx (torch.Tensor, optional): Initial hidden state [1, batch_size, hidden_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence output and final hidden state.
        """
        batch_size = x.size(0)
        
        # Create zero initial hidden state if not provided
        if hx is None:
            hx = torch.zeros(1, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
            
        h_t = hx[0]

        # Pre-compute all input projections across the entire sequence length
        x_gates_all = F.linear(x, self.w_ih, self.bias_ih)
        outputs = []
        
        # Iterate over sequence steps without memory copies using torch.unbind
        for gi in torch.unbind(x_gates_all, dim=1):
            # Compute hidden-to-hidden projections for the current step
            gh = F.linear(h_t, self.w_hh, self.bias_hh)

            # Split gates into Reset (r), Update (i), and New (n) components
            i_r, i_i, i_n = gi.chunk(3, dim=1)
            h_r, h_i, h_n = gh.chunk(3, dim=1)

            # Apply activations
            reset_gate = torch.sigmoid(i_r + h_r)
            update_gate = torch.sigmoid(i_i + h_i)
            new_gate = torch.tanh(i_n + reset_gate * h_n)

            # Update the hidden state
            h_t = (1.0 - update_gate) * new_gate + update_gate * h_t
            outputs.append(h_t)

        # Stack outputs to reform the time dimension: [batch_size, seq_len, hidden_dim]
        out_seq = torch.stack(outputs, dim=1)
        
        return out_seq, h_t.unsqueeze(0)


class ParallelBiGRU(nn.Module):
    """
    Parallel Bidirectional Gated Recurrent Unit (BiGRU).
    
    Processes Forward and Backward directions simultaneously within a single sequence loop
    using batched matrix multiplications, doubling the processing speed on GPUs.
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        """
        Initializes the Parallel Bidirectional GRU.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Index 0: Forward, Index 1: Backward -> Shape: [2, 3 * hidden, input]
        self.w_ih = nn.Parameter(torch.empty(2, 3 * hidden_dim, input_dim))
        # Shape: [2, 3 * hidden, hidden]
        self.w_hh = nn.Parameter(torch.empty(2, 3 * hidden_dim, hidden_dim))
        
        # Biases -> Shape: [2, 3 * hidden]
        self.bias_ih = nn.Parameter(torch.empty(2, 3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(2, 3 * hidden_dim))

        self._init_weights()

    def _init_weights(self):
        """Applies independent initializations to both directions."""
        with torch.no_grad():
            for d in range(2):
                nn.init.xavier_uniform_(self.w_ih[d])
                nn.init.orthogonal_(self.w_hh[d])
                nn.init.zeros_(self.bias_ih[d])
                nn.init.zeros_(self.bias_hh[d])
                self.bias_ih[d, self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, hx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simultaneous BiGRU Forward Pass.

        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len, input_dim].
            hx (torch.Tensor, optional): Initial hidden states [2, batch_size, hidden_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Concatenated outputs and final states.
        """
        batch_size = x.size(0)

        # Initialize bidirectional hidden states
        if hx is None:
            hx = torch.zeros(2, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)

        # Stack normal sequence and flipped sequence along the direction dimension (dim=0)
        # Shape: [2, batch_size, seq_len, input_dim]
        x_parallel = torch.stack([x, torch.flip(x, dims=[1])], dim=0)

        # Parallel linear projection for both directions using Einsum
        # Shape: [2, batch_size, seq_len, 3 * hidden_dim]
        x_gates_parallel = torch.einsum('d b t i, d g i -> d b t g', x_parallel, self.w_ih) + self.bias_ih.unsqueeze(1).unsqueeze(1)

        h_t = hx
        outputs = []

        # Iterate over sequence steps processing BOTH directions simultaneously
        for gi in torch.unbind(x_gates_parallel, dim=2):
            # Parallel hidden projection for both directions -> [2, batch_size, 3 * hidden]
            gh = torch.einsum('d b h, d g h -> d b g', h_t, self.w_hh) + self.bias_hh.unsqueeze(1)

            # Chunk gates -> Shapes: [2, batch_size, hidden_dim]
            i_r, i_i, i_n = gi.chunk(3, dim=-1)
            h_r, h_i, h_n = gh.chunk(3, dim=-1)

            # Compute activations concurrently
            reset_gate = torch.sigmoid(i_r + h_r)
            update_gate = torch.sigmoid(i_i + h_i)
            new_gate = torch.tanh(i_n + reset_gate * h_n)

            # Update hidden states concurrently
            h_t = (1.0 - update_gate) * new_gate + update_gate * h_t
            outputs.append(h_t)

        # Stack temporal outputs -> [2, batch_size, seq_len, hidden_dim]
        out_parallel = torch.stack(outputs, dim=2)

        # Separate Forward and Backward features and restore sequence alignment for backward
        out_fwd = out_parallel[0]
        out_bwd = torch.flip(out_parallel[1], dims=[1])

        # Concatenate directional features -> [batch_size, seq_len, 2 * hidden_dim]
        out_final = torch.cat([out_fwd, out_bwd], dim=-1)

        return out_final, h_t