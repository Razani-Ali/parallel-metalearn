import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class UniLSTM(nn.Module):
    """
    Unidirectional Long Short-Term Memory (LSTM) module.
    
    Functional implementation utilizing torch.unbind to map recurrences efficiently
    without mutating in-place tensors.
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        """
        Initializes the Unidirectional LSTM.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Weights -> Shape: [4 * hidden, input] (Input, Forget, Cell, Output gates)
        self.w_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        # Weights -> Shape: [4 * hidden, hidden]
        self.w_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        
        # Biases -> Shape: [4 * hidden]
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))

        self._init_weights()

    def _init_weights(self):
        """Applies Xavier and Orthogonal initialization techniques."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.w_ih)
            nn.init.orthogonal_(self.w_hh)
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            # Set forget gate bias to 1.0 to preserve early-stage gradients
            self.bias_ih[self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the Unidirectional LSTM.

        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len, input_dim].
            hx (Tuple, optional): Tuple containing initial (h_0, c_0) states.

        Returns:
            Tuple: Sequence output and final state tuple (h_final, c_final).
        """
        batch_size = x.size(0)
        
        # Initialize zero states if no previous states are provided
        if hx is None:
            h_0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
            c_0 = torch.zeros(1, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        else:
            h_0, c_0 = hx
            
        h_t, c_t = h_0[0], c_0[0]

        # Pre-compute all input-to-hidden projections
        x_gates_all = F.linear(x, self.w_ih, self.bias_ih)
        outputs = []
        
        # Sequence recurrence loop
        for gi in torch.unbind(x_gates_all, dim=1):
            # Compute hidden-to-hidden projection
            gh = F.linear(h_t, self.w_hh, self.bias_hh)
            
            # Split into input, forget, cell candidate, and output gates
            i, f, g, o = (gi + gh).chunk(4, dim=1)

            # Apply nonlinearities
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Update cell and hidden states
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            
            outputs.append(h_t)

        # Stack outputs -> [batch_size, seq_len, hidden_dim]
        out_seq = torch.stack(outputs, dim=1)
        
        return out_seq, (h_t.unsqueeze(0), c_t.unsqueeze(0))


class ParallelBiLSTM(nn.Module):
    """
    Parallel Bidirectional Long Short-Term Memory (BiLSTM).
    
    Fuses forward and backward recurrent step executions into a single batched sequence loop
    utilizing torch.einsum for fully parallelized hardware execution.
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        """
        Initializes the Parallel Bidirectional LSTM.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Index 0: Forward, Index 1: Backward -> Shape: [2, 4 * hidden, input]
        self.w_ih = nn.Parameter(torch.empty(2, 4 * hidden_dim, input_dim))
        # Shape: [2, 4 * hidden, hidden]
        self.w_hh = nn.Parameter(torch.empty(2, 4 * hidden_dim, hidden_dim))
        
        # Biases -> Shape: [2, 4 * hidden]
        self.bias_ih = nn.Parameter(torch.empty(2, 4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(2, 4 * hidden_dim))

        self._init_weights()

    def _init_weights(self):
        """Initializes directional parameters independently."""
        with torch.no_grad():
            for d in range(2):
                nn.init.xavier_uniform_(self.w_ih[d])
                nn.init.orthogonal_(self.w_hh[d])
                nn.init.zeros_(self.bias_ih[d])
                nn.init.zeros_(self.bias_hh[d])
                # Set forget gate bias to 1.0 for both directions
                self.bias_ih[d, self.hidden_dim:2 * self.hidden_dim].fill_(1.0)

    def forward(self, x: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simultaneous BiLSTM Forward Pass.

        Args:
            x (torch.Tensor): Input sequence [batch_size, seq_len, input_dim].
            hx (Tuple, optional): Tuple containing initial (h_0, c_0) states. Shape of each: [2, batch, hidden].

        Returns:
            Tuple: Concatenated outputs and final state tuple (h_final, c_final).
        """
        batch_size = x.size(0)

        # Initialize directional state tensors
        if hx is None:
            h_0 = torch.zeros(2, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
            c_0 = torch.zeros(2, batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        else:
            h_0, c_0 = hx

        # Stack normal and flipped sequences -> [2, batch_size, seq_len, input_dim]
        x_parallel = torch.stack([x, torch.flip(x, dims=[1])], dim=0)

        # Compute batched linear projections for both directions
        # Shape: [2, batch_size, seq_len, 4 * hidden_dim]
        x_gates_parallel = torch.einsum('d b t i, d g i -> d b t g', x_parallel, self.w_ih) + self.bias_ih.unsqueeze(1).unsqueeze(1)

        h_t, c_t = h_0, c_0
        outputs = []

        # Iterate over sequence steps processing BOTH directions concurrently
        for gi in torch.unbind(x_gates_parallel, dim=2):
            # Parallel hidden projection -> [2, batch_size, 4 * hidden_dim]
            gh = torch.einsum('d b h, d g h -> d b g', h_t, self.w_hh) + self.bias_hh.unsqueeze(1)

            # Chunk combined gates -> Shapes: [2, batch_size, hidden_dim]
            i, f, g, o = (gi + gh).chunk(4, dim=-1)

            # Apply nonlinearities concurrently for both directions
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Update cells and hidden states concurrently
            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            
            outputs.append(h_t)

        # Stack temporal outputs -> [2, batch_size, seq_len, hidden_dim]
        out_parallel = torch.stack(outputs, dim=2)

        # Separate outputs and reverse the backward sequence to restore alignment
        out_fwd = out_parallel[0]
        out_bwd = torch.flip(out_parallel[1], dims=[1])

        # Concatenate along the features dimension -> [batch_size, seq_len, 2 * hidden_dim]
        out_final = torch.cat([out_fwd, out_bwd], dim=-1)

        return out_final, (h_t, c_t)