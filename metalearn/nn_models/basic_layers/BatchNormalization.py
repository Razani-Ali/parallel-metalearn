import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The concept of per-step Batch Normalization parameters (maintaining distinct
# gamma, beta, running_mean, and running_var for each inner-adaptation step)
# is adapted from MAML++:
#
# [1] A. Antoniou, H. Edwards, and A. Storkey, "How to train your MAML," 
#     in Int. Conf. Learn. Representations (ICLR), 2019. 
#     arXiv:1810.09502.
# ==============================================================================


class BatchNorm(nn.Module):
    """
    Custom Batch Normalization module designed for meta-learning and vmap compatibility.

    Supports per-step learnable affine parameters (gamma/beta) and running statistics 
    to handle task-specific distribution shifts across inner-loop adaptation steps.
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
            num_features (int): Number of channels/features C in the input tensor (N, C, ...).
            eps (float): Value added to the denominator for numerical stability.
            momentum (float): Value used for the running_mean and running_var computation.
            track_running_stats (bool): If True, tracks running statistics. Set to False for vmap compatibility.
            use_per_step_stats (bool): If True, maintains separate affine parameters and buffers per inner step.
            max_inner_steps (int): Maximum number of inner-loop adaptation steps.
            learn_gamma (bool): If True, gamma (weight) is a learnable parameter.
            learn_beta (bool): If True, beta (bias) is a learnable parameter.
            **kwargs: Additional optional keyword arguments.
        """
        super().__init__()
        
        # Store basic normalization configuration parameters
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats  # Set to False to prevent in-place buffer mutation during vmap
        self.use_per_step_stats = use_per_step_stats
        self.max_inner_steps = max_inner_steps
        
        # Flags indicating parameter learnability
        self.learnable_gamma = learn_gamma
        self.learnable_beta = learn_beta

        # Initialize parameters and buffers based on per-step configuration
        if self.use_per_step_stats:
            # Instantiate 2D parameter tensors with shape (max_inner_steps, num_features)
            self.weight = nn.Parameter(torch.ones(max_inner_steps, num_features), requires_grad=self.learnable_gamma)
            self.bias = nn.Parameter(torch.zeros(max_inner_steps, num_features), requires_grad=self.learnable_beta)
            # Register 2D persistent running statistics buffers for each step
            self.register_buffer('running_mean', torch.zeros(max_inner_steps, num_features))
            self.register_buffer('running_var', torch.ones(max_inner_steps, num_features))
        else:
            # Instantiate standard 1D parameter tensors with shape (num_features,)
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=self.learnable_gamma)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=self.learnable_beta)
            # Register standard 1D persistent running statistics buffers
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """
        Performs the forward pass for Batch Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, ...) or (N, C).
            kwargs (dict): Context dictionary containing runtime flags like 'num_step' and 'training'.

        Returns:
            torch.Tensor: Batch-normalized output tensor.
        """
        # Extract current inner-loop adaptation step index from kwargs (default to 0)
        num_step = kwargs.get('num_step', 0)
        if isinstance(num_step, int):
            # Convert integer index to tensor placed on the same device as the input
            num_step = torch.tensor(num_step, device=x.device)

        # Extract training mode flag from kwargs (default to True)
        training = kwargs.get('training', True)

        # Retrieve weight (gamma) and bias (beta) according to the step mode
        if self.use_per_step_stats:
            weight, bias = self.weight[num_step], self.bias[num_step]
        else:
            weight, bias = self.weight, self.bias

        # Handle batch normalization execution path based on tracking configuration
        if not self.track_running_stats:
            # Calculate batch statistics directly from current batch data (vmap-friendly path)
            return F.batch_norm(
                x, None, None, weight, bias, 
                True,  # Force usage of current batch statistics
                self.momentum, self.eps
            )
        else:
            # Extract corresponding running mean and variance when tracking is enabled
            if self.use_per_step_stats:
                r_mean, r_var = self.running_mean[num_step], self.running_var[num_step]
            else:
                r_mean, r_var = self.running_mean, self.running_var

            # Execute functional batch normalization using tracked running statistics
            return F.batch_norm(
                x, r_mean, r_var, weight, bias, 
                training, self.momentum, self.eps
            )