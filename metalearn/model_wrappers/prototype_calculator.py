import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class BasePrototype(nn.Module, ABC):
    """
    Abstract Base Class for all Prototype / Centroid Extractor modules.
    
    Enforces a standardized `compute_class_centers` interface supporting
    masking (samples_mask), PyTorch functional execution, and torch.func.vmap.
    """

    def __init__(self, max_classes: int, detach_prototypes: bool = False):
        """
        Initializes the base prototype calculator interface.

        Args:
            max_classes (int): Upper bound on total target classes (ways) per task batch.
            detach_prototypes (bool): Whether to detach computed prototypes from autograd graph.
        """
        super().__init__()
        self.max_classes = max_classes
        self.detach_prototypes = detach_prototypes

    @abstractmethod
    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method for computing class centroids/prototypes.

        Args:
            features (torch.Tensor): Extracted sample representations of shape [num_samples, latent_dim].
            labels (torch.Tensor): Ground-truth class index tensor of shape [num_samples].
            samples_mask (Optional[torch.Tensor]): Binary validity mask tensor for padded samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - centroids: Mean feature representations of shape [max_classes, latent_dim].
                - mask: Boolean tensor of shape [max_classes] indicating active classes.
        """
        pass

class SimplePrototype(BasePrototype):
    """
    Non-parametric, Class-Agnostic Centroid/Prototype Extractor Module with Per-Step Statistics.

    Computes class centers (prototypes) across feature tensors using vectorized matrix
    operations while respecting sample validity masks (samples_mask). Fully differentiable
    and compatible with PyTorch `vmap`, `functional_call`, and MAML execution loops.

    Optionally maintains exponential moving averages (`running_prototypes`) per inner adaptation 
    step when `keep_running_Prototype=True` and `use_per_step_stats=True`, properly avoiding 
    zero-bias on newly initialized classes and correctly ignoring masked out samples.
    """

    def __init__(
        self, 
        max_classes: int, 
        latent_dim: int, 
        keep_running_Prototype: bool = False, 
        use_per_step_stats: bool = False,
        max_inner_steps: int = 5,
        momentum: float = 0.1, 
        detach_prototypes: bool = False
    ):
        """
        Initializes the SimplePrototype module.

        Args:
            max_classes (int): Upper bound on total target classes (ways) per task batch.
            latent_dim (int): Dimensionality of extracted feature representations.
            keep_running_Prototype (bool): If True, tracks running prototype moving averages.
            use_per_step_stats (bool): If True, allocates distinct running prototype buffers per inner step.
            max_inner_steps (int): Maximum number of inner adaptation steps.
            momentum (float): Momentum factor for exponential moving average updates.
            detach_prototypes (bool): Whether to detach output prototypes from computational graph.
        """
        # Initialize PyTorch parent module
        super().__init__(max_classes=max_classes, detach_prototypes=detach_prototypes)
        
        # Store configuration parameters
        self.latent_dim = latent_dim
        self.keep_running_Prototype = keep_running_Prototype
        self.use_per_step_stats = use_per_step_stats
        self.max_inner_steps = max_inner_steps
        self.momentum = momentum

        # Register persistent buffers if running prototype tracking is enabled
        if self.keep_running_Prototype:
            if self.use_per_step_stats:
                # Register 3D buffer to store running feature prototypes per inner step [max_inner_steps, max_classes, latent_dim]
                self.register_buffer("running_prototypes", torch.zeros(max_inner_steps, max_classes, latent_dim))
                # Register 2D boolean buffer tracking class initialization per inner step [max_inner_steps, max_classes]
                self.register_buffer("initialized_classes", torch.zeros(max_inner_steps, max_classes, dtype=torch.bool))
            else:
                # Register 2D buffer to store running feature prototypes [max_classes, latent_dim]
                self.register_buffer("running_prototypes", torch.zeros(max_classes, latent_dim))
                # Register 1D boolean buffer tracking initialized classes [max_classes]
                self.register_buffer("initialized_classes", torch.zeros(max_classes, dtype=torch.bool))

    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates mean feature centroids and active class presence masks while 
        ignoring padded/dummy samples via samples_mask. Fully tracks gradients.

        Args:
            features (torch.Tensor): Extracted sample representations of shape [num_samples, latent_dim].
            labels (torch.Tensor): Ground-truth class index tensor of shape [num_samples].
            samples_mask (Optional[torch.Tensor]): Binary/boolean validity mask of shape [num_samples].
            **kwargs: Operational context flags such as 'inner_step' and 'training'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - centroids: Mean feature representations of shape [max_classes, latent_dim].
                - mask: Boolean tensor of shape [max_classes] indicating active classes.
        """
        # Extract adaptation inner-step index from context dictionary (defaults to step 0)
        num_step = kwargs.get('inner_step', 0)
        if isinstance(num_step, int):
            num_step = torch.tensor(num_step, device=features.device)
            # Clamp index to maximum inner steps boundary to avoid tensor indexing out-of-bounds
            step_idx = torch.clamp(num_step, max=self.max_inner_steps - 1)
        else:
            step_idx = num_step

        # Extract training mode flag from kwargs (defaults to True)
        training = kwargs.get('training', True)

        # 1. Generate static one-hot encoding matrix over dataset max_classes spectrum
        one_hot = F.one_hot(labels, num_classes=self.max_classes).to(features.dtype)

        # 2. Filter out dummy/padded samples using samples_mask if provided
        if samples_mask is not None:
            # Cast mask to feature floating-point dtype and expand dimension for broadcasting
            mask_expanded = samples_mask.to(features.dtype).unsqueeze(-1)  # Shape: [num_samples, 1]
            # Zero out one-hot assignments for invalid padded samples
            valid_one_hot = one_hot * mask_expanded
        else:
            valid_one_hot = one_hot

        # 3. Compute valid sample count per class category: shape [max_classes, 1]
        class_counts = valid_one_hot.sum(dim=0, keepdim=True).T

        # 4. Aggregate feature vectors for each class via matrix multiplication: shape [max_classes, latent_dim]
        class_sums = torch.matmul(valid_one_hot.T, features)

        # 5. Perform safe element-wise division (clamp prevents division-by-zero for absent classes)
        centroids = class_sums / class_counts.clamp(min=1)

        # 6. Construct boolean validity mask (True for present classes with valid counts > 0)
        mask = (class_counts.squeeze(-1) > 0)

        # Handle exponential moving average (EMA) update logic if running prototypes tracking is enabled
        if self.keep_running_Prototype:
            # Slice current step's buffers if per-step mode is active
            if self.use_per_step_stats:
                r_proto = self.running_prototypes[step_idx]
                r_init = self.initialized_classes[step_idx]
            else:
                r_proto = self.running_prototypes
                r_init = self.initialized_classes

            if training:
                # Expand active class validity mask for vector broadcasting
                valid_mask = mask.unsqueeze(-1).to(features.dtype)
                # Expand initialization boolean tracking mask for vector broadcasting
                is_init = r_init.unsqueeze(-1).to(features.dtype)

                # Calculate standard exponential moving average (EMA) update step
                updated_existing = (1 - self.momentum) * r_proto + self.momentum * centroids.detach()

                # Avoid zero-initialization bias: assign raw centroid directly if class is uninitialized
                new_val_for_valid = is_init * updated_existing + (1 - is_init) * centroids.detach()

                # Apply out-of-place update solely to classes present in current batch mask
                new_running_prototypes = valid_mask * new_val_for_valid + (1 - valid_mask) * r_proto

                # Update state buffers out-of-place for vmap compatibility
                if self.use_per_step_stats:
                    # Clone buffer tensor and replace the slice for the active inner step
                    updated_rp = self.running_prototypes.clone()
                    updated_rp[step_idx] = new_running_prototypes
                    self.running_prototypes = updated_rp

                    updated_ic = self.initialized_classes.clone()
                    updated_ic[step_idx] = r_init | mask
                    self.initialized_classes = updated_ic
                else:
                    # Directly re-bind newly computed 2D/1D tensors
                    self.running_prototypes = new_running_prototypes
                    self.initialized_classes = r_init | mask

                # In training mode, return current batch centroids and batch presence mask
                out_centroids = centroids
                out_mask = mask
            else:
                # In evaluation/inference mode, return stored running prototypes and initialization mask
                out_centroids = r_proto
                out_mask = r_init
        else:
            # Standard non-running prototype mode
            out_centroids = centroids
            out_mask = mask

        # Detach outputs from autograd graph if detach_prototypes flag is enabled
        if self.detach_prototypes:
            return out_centroids.detach(), out_mask.detach()

        return out_centroids, out_mask


# ==============================================================================
# DEVELOPER NOTE: EXTENDING MaskedClassCenterHead FOR PARAMETRIC METRIC-LEARNING
# ==============================================================================
# To convert this non-parametric prototype extractor into a fully learnable module
# (e.g., for ProtoMAML or learnable Attention-based Centroids), follow these strategies:
#
# 1. Weighted Average / Attention-based Prototypes:
#    Replace uniform average division (`class_sums / class_counts`) with learnable
#    sample attention weights (Soft-Centroids):
#
#        # Learnable Linear Projection for Sample Weighting
#        self.sample_scorer = nn.Linear(latent_dim, 1)
#        weights = F.softmax(self.sample_scorer(features), dim=0) # shape [num_samples, 1]
#        weighted_one_hot = one_hot * weights
#        centroids = torch.matmul(weighted_one_hot.T, features)
#
# 2. Feature Adapter / Projection Layer:
#    Inject a learnable metric space transformation before computing class means:
#
#        self.proj = nn.Linear(latent_dim, proj_dim, bias=False)
#        projected_feats = self.proj(features)
#        # Compute centroids using projected_feats
#
# 3. Learnable Class Offsets & Temperature Scaling:
#    Register learnable class bias vectors or a log-temperature scalar parameter:
#
#        self.class_offsets = nn.Parameter(torch.zeros(max_classes, latent_dim))
#        self.temperature = nn.Parameter(torch.tensor(1.0))
#        centroids = raw_centroids + self.class_offsets
#
# Note: For functional MAML compatibility (vmap), ensure any new parameters added
# to this module are properly exposed via `model.get_fast_weights()` in your Wrapper!
# ==============================================================================