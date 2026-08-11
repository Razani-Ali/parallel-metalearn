import torch
import torch.nn as nn
from typing import Optional


class BatchNorm(nn.Module):
    """
    Custom Dynamic N-Dimensional Batch Normalization Layer with Per-Step Statistics Support.

    Supports arbitrary input tensor dimensions ([N, C], [N, C, L], [N, C, H, W], [N, C, D, H, W])
    matching the standard PyTorch behavior of BatchNorm1d, BatchNorm2d, and BatchNorm3d.
    Fully compatible with PyTorch `torch.func.vmap`, `functional_call`, and MAML execution loops.

    Optionally tracks per-inner-step running statistics (`running_mean`, `running_var`) 
    when `use_per_step_stats=True` for step-aware meta-learning (e.g., MAML++).
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        use_per_step_stats: bool = False,
        max_inner_steps: int = 5,
    ):
        """
        Initializes the dynamic BatchNorm module.

        Args:
            num_features (int): Number of channels / features (C) expected in input tensor.
            eps (float): Small constant added to denominator for numerical stability.
            momentum (float): Value used for the running mean and variance computation.
            affine (bool): If True, enables learnable affine parameters (gamma weight and beta bias).
            track_running_stats (bool): If True, tracks running mean and running variance.
            use_per_step_stats (bool): If True, allocates separate running statistics buffers per inner step.
            max_inner_steps (int): Maximum number of inner adaptation steps threshold.
        """
        # Call parent nn.Module constructor
        super().__init__()

        # Store feature / channel dimension size
        self.num_features = num_features
        # Store numerical stability epsilon
        self.eps = eps
        # Store running statistics momentum factor
        self.momentum = momentum
        # Store boolean flag for learnable affine parameters
        self.affine = affine
        # Store boolean flag for tracking running statistics
        self.track_running_stats = track_running_stats
        # Store boolean flag for per-step running statistics (MAML++)
        self.use_per_step_stats = use_per_step_stats
        # Store maximum capacity of inner adaptation steps
        self.max_inner_steps = max_inner_steps

        # Register learnable affine parameters (gamma weight and beta bias) if enabled
        if self.affine:
            # Register gamma scale parameter initialized to ones
            self.weight = nn.Parameter(torch.ones(num_features))
            # Register beta shift parameter initialized to zeros
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            # Register weight as None when affine transformation is disabled
            self.register_parameter("weight", None)
            # Register bias as None when affine transformation is disabled
            self.register_parameter("bias", None)

        # Register running mean and variance buffers if tracking is enabled
        if self.track_running_stats:
            if self.use_per_step_stats:
                # Register 2D buffer for per-step running mean: [max_inner_steps, num_features]
                self.register_buffer("running_mean", torch.zeros(max_inner_steps, num_features))
                # Register 2D buffer for per-step running variance: [max_inner_steps, num_features]
                self.register_buffer("running_var", torch.ones(max_inner_steps, num_features))
            else:
                # Register 1D buffer for standard running mean: [num_features]
                self.register_buffer("running_mean", torch.zeros(num_features))
                # Register 1D buffer for standard running variance: [num_features]
                self.register_buffer("running_var", torch.ones(num_features))
        else:
            # Register buffers as None if tracking is disabled
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Performs forward pass of dynamic N-dimensional Batch Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C], [N, C, L], [N, C, H, W], etc.
            **kwargs: Operational context flags such as 'training' and 'inner_step' / 'num_step'.

        Returns:
            torch.Tensor: Normalized output tensor with identical shape as input x.
        """
        # Determine operational training mode from kwargs or module attribute
        training = kwargs.get("training", self.training)

        # Retrieve current inner step index from context dictionary (defaults to step 0)
        num_step = kwargs.get("inner_step", kwargs.get("num_step", 0))

        # Ensure step index is represented as a Tensor on target device
        if isinstance(num_step, int):
            num_step = torch.tensor(num_step, device=x.device)
            # Clamp step index to maximum inner steps boundary
            step_idx = torch.clamp(num_step, max=self.max_inner_steps - 1)
        else:
            step_idx = torch.clamp(num_step, max=self.max_inner_steps - 1)

        # Extract total dimensional rank of input tensor
        ndim = x.dim()

        # Validate minimum required rank for batch normalization
        if ndim < 2:
            raise ValueError(f"Expected input tensor to have at least 2 dimensions [N, C], got {ndim}D.")

        # Dynamically determine reduction dimensions and broadcasting shape
        if ndim == 2:
            # For 2D tensor [N, C], reduce over batch dimension 0
            reduce_dims = [0]
            # Construct 2D broadcast shape: [1, C]
            broadcast_shape = [1, self.num_features]
        else:
            # For N-D tensor [N, C, d1, d2, ...], reduce over batch 0 and spatial/temporal dims (2, 3, ...)
            reduce_dims = [0] + list(range(2, ndim))
            # Construct N-D broadcast shape: [1, C, 1, 1, ...]
            broadcast_shape = [1, self.num_features] + [1] * (ndim - 2)

        # Calculate dynamic batch statistics or use running statistics
        if training or not self.track_running_stats:
            # Compute current batch mean across reduce_dims
            batch_mean = x.mean(dim=reduce_dims)
            # Compute current batch variance across reduce_dims (unbiased=False matches PyTorch default)
            batch_var = x.var(dim=reduce_dims, unbiased=False)

            # Update running statistics out-of-place if tracking is enabled
            if self.track_running_stats:
                # Slice relevant running statistics view based on per-step configuration
                if self.use_per_step_stats:
                    r_mean = self.running_mean[step_idx]
                    r_var = self.running_var[step_idx]
                else:
                    r_mean = self.running_mean
                    r_var = self.running_var

                # Compute exponential moving average update step for mean
                new_mean = (1 - self.momentum) * r_proto_mean if 'r_proto_mean' in locals() else (1 - self.momentum) * r_mean + self.momentum * batch_mean.detach()
                # Compute exponential moving average update step for variance
                new_var = (1 - self.momentum) * r_var + self.momentum * batch_var.detach()

                # Perform out-of-place buffer update to maintain functional vmap compatibility
                if self.use_per_step_stats:
                    # Construct boolean step mask for non-inplace update: [max_inner_steps, 1]
                    step_mask = (torch.arange(self.max_inner_steps, device=x.device) == step_idx).view(-1, 1)
                    # Update running_mean tensor slice out-of-place via torch.where
                    self.running_mean = torch.where(step_mask, new_mean.unsqueeze(0), self.running_mean)
                    # Update running_var tensor slice out-of-place via torch.where
                    self.running_var = torch.where(step_mask, new_var.unsqueeze(0), self.running_var)
                else:
                    # Directly assign updated 1D running_mean buffer
                    self.running_mean = new_mean
                    # Directly assign updated 1D running_var buffer
                    self.running_var = new_var

            # Assign batch statistics for normalization in training mode
            mean = batch_mean
            var = batch_var
        else:
            # Evaluation mode: assign stored running statistics
            if self.use_per_step_stats:
                mean = self.running_mean[step_idx]
                var = self.running_var[step_idx]
            else:
                mean = self.running_mean
                var = self.running_var

        # Reshape mean tensor to broadcast shape [1, C, 1, 1, ...]
        mean_bc = mean.view(broadcast_shape)
        # Reshape variance tensor to broadcast shape [1, C, 1, 1, ...]
        var_bc = var.view(broadcast_shape)

        # Normalize input tensor using standard batch normalization equation
        x_norm = (x - mean_bc) / torch.sqrt(var_bc + self.eps)

        # Apply learnable affine transformation if enabled
        if self.affine:
            # Reshape gamma weight tensor to broadcast shape
            weight_bc = self.weight.view(broadcast_shape)
            # Reshape beta bias tensor to broadcast shape
            bias_bc = self.bias.view(broadcast_shape)
            # Scale and shift normalized tensor
            x_norm = x_norm * weight_bc + bias_bc

        # Return normalized output tensor
        return x_norm