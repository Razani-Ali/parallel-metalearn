import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """
    Dynamic-Shape, Purely Functional Batch Normalization Module.

    Supports N-dimensional input tensors (1D, 2D, 3D, 4D, or N-D signals, time-series, 
    and images) without hardcoded dimension indexing. Fully compatible with PyTorch's 
    functional execution paradigm (`torch.func.vmap` and `functional_call`) by enforcing 
    out-of-place buffer updates to avoid in-place tensor mutation errors.

    Additionally supports per-step learnable parameters (gamma/beta) and running 
    statistics buffers to mitigate distribution shift across inner-adaptation steps 
    in meta-learning algorithms (e.g., MAML++).
    """

    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1,
        track_running_stats: bool = False, 
        use_per_step_stats: bool = False,
        max_inner_steps: int = 5, 
        learn_gamma: bool = True, 
        learn_beta: bool = True, 
        **kwargs
    ):
        """
        Initializes the custom BatchNorm module.

        Args:
            num_features (int): Number of channels/features C in the input tensor.
            eps (float): Small epsilon value added to denominator for numerical stability.
            momentum (float): Momentum factor for exponential moving average updates of running stats.
            track_running_stats (bool): If True, tracks persistent running mean and variance.
            use_per_step_stats (bool): If True, allocates independent parameters/buffers per adaptation step.
            max_inner_steps (int): Upper bound on inner-loop adaptation steps.
            learn_gamma (bool): If True, enables gradient tracking for scale parameter (weight).
            learn_beta (bool): If True, enables gradient tracking for shift parameter (bias).
            **kwargs: Extra unused keyword arguments for pipeline flexibility.
        """
        # Call parent PyTorch nn.Module constructor
        super().__init__()
        
        # Store fundamental normalization configuration attributes
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats  
        self.use_per_step_stats = use_per_step_stats
        self.max_inner_steps = max_inner_steps
        
        # Store learnability boolean flags
        self.learnable_gamma = learn_gamma
        self.learnable_beta = learn_beta

        # Configure parameters and persistent buffers depending on per-step mode
        if self.use_per_step_stats:
            # Instantiate 2D affine parameters of shape (max_inner_steps, num_features)
            self.weight = nn.Parameter(
                torch.ones(max_inner_steps, num_features), 
                requires_grad=self.learnable_gamma
            )
            self.bias = nn.Parameter(
                torch.zeros(max_inner_steps, num_features), 
                requires_grad=self.learnable_beta
            )
            # Register 2D persistent running statistics buffers of shape (max_inner_steps, num_features)
            self.register_buffer('running_mean', torch.zeros(max_inner_steps, num_features))
            self.register_buffer('running_var', torch.ones(max_inner_steps, num_features))
        else:
            # Instantiate standard 1D affine parameters of shape (num_features,)
            self.weight = nn.Parameter(
                torch.ones(num_features), 
                requires_grad=self.learnable_gamma
            )
            self.bias = nn.Parameter(
                torch.zeros(num_features), 
                requires_grad=self.learnable_beta
            )
            # Register standard 1D persistent running statistics buffers of shape (num_features,)
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """
        Executes functional forward pass for Batch Normalization.

        Args:
            x (torch.Tensor): Input tensor of arbitrary dimensionality (e.g., [N, L], [N, C, L], [N, C, H, W]).
            kwargs (dict): Operational context container holding runtime flags like 'inner_step' and 'training'.

        Returns:
            torch.Tensor: Normalized, scaled, and shifted output tensor matching input shape.
        """
        # Extract adaptation inner-step index from context dictionary (defaults to step 0)
        num_step = kwargs.get('inner_step', 0)
        
        # Ensure step index is represented as a Tensor on the target compute device
        if isinstance(num_step, int):
            num_step = torch.tensor(num_step, device=x.device)
            # Clamp index to maximum inner steps boundary to avoid tensor indexing out-of-bounds
            step_idx = torch.clamp(num_step, max=self.max_inner_steps - 1)
        else:
            step_idx = num_step

        # Extract training mode boolean flag from context dictionary (defaults to True)
        training = kwargs.get('training', True)

        # 1. Dynamically infer reduction dimensions and broadcast shape based on input tensor rank
        ndim = x.dim()
        if ndim == 1:
            # Handle 1D vector case: reduce along dim 0, broadcast along dim 0
            reduce_dims = [0]
            broadcast_shape = [1]
        else:
            # Reduce across batch dimension (0) and spatial/temporal dimensions (2, 3, ...), preserving channel (1)
            reduce_dims = [0] + list(range(2, ndim))
            # Construct broadcast shape corresponding to input rank: [1, C, 1, 1, ...]
            broadcast_shape = [1, self.num_features] + [1] * (ndim - 2)

        # Select corresponding parameter and buffer slices based on per-step configuration
        if self.use_per_step_stats:
            weight, bias = self.weight[step_idx], self.bias[step_idx]
            r_mean, r_var = self.running_mean[step_idx], self.running_var[step_idx]
        else:
            weight, bias = self.weight, self.bias
            r_mean, r_var = self.running_mean, self.running_var

        # Compute batch statistics or utilize tracked running statistics
        if not self.track_running_stats:
            # Compute statistics directly from the current input batch (Transductive/vmap mode)
            mean = x.mean(dim=reduce_dims)
            var = x.var(dim=reduce_dims, unbiased=False)
            mean_to_use, var_to_use = mean, var
        else:
            if training:
                # Compute statistics for current training batch
                batch_mean = x.mean(dim=reduce_dims)
                batch_var = x.var(dim=reduce_dims, unbiased=False)
                
                # Out-of-place running average updates (creates new tensor instances to avoid in-place vmap mutations)
                new_mean = (1 - self.momentum) * r_mean + self.momentum * batch_mean.detach()
                new_var = (1 - self.momentum) * r_var + self.momentum * batch_var.detach()

                # Re-bind updated statistics buffers without modifying underlying memory buffers in-place
                if self.use_per_step_stats:
                    # Clone buffer tensor and replace the row slice for the active inner step
                    updated_rm = self.running_mean.clone()
                    updated_rm[step_idx] = new_mean
                    self.running_mean = updated_rm

                    updated_rv = self.running_var.clone()
                    updated_rv[step_idx] = new_var
                    self.running_var = updated_rv
                else:
                    # Directly assign newly created 1D mean and variance tensors
                    self.running_mean = new_mean
                    self.running_var = new_var

                # Use current batch statistics during training execution
                mean_to_use, var_to_use = batch_mean, batch_var
            else:
                # Use tracked running statistics during evaluation execution
                mean_to_use, var_to_use = r_mean, r_var

        # 2. Reshape scale, shift, mean, and variance tensors to enable correct tensor broadcasting
        w = weight.view(broadcast_shape)
        b = bias.view(broadcast_shape)
        m = mean_to_use.view(broadcast_shape)
        v = var_to_use.view(broadcast_shape)

        # Apply standardized Batch Normalization formula: (x - mean) / sqrt(var + eps) * weight + bias
        x_norm = (x - m) / torch.sqrt(v + self.eps)
        return w * x_norm + b