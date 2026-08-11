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
    Non-parametric Centroid / Prototype Extractor Module supporting Exponential Moving Average (EMA).

    Calculates mean feature vectors (prototypes) per class using one-hot matrix aggregations 
    while respecting sample validity masks (`samples_mask`). Fully differentiable and 
    compatible with PyTorch `vmap` and `functional_call` through out-of-place buffer updates.

    When `keep_running_Prototype` is enabled, maintains a persistent running moving average 
    of class prototypes across training iterations (`running_prototypes`).
    """

    def __init__(
        self, 
        max_classes: int, 
        latent_dim: int = 64, 
        keep_running_Prototype: bool = False, 
        momentum: float = 0.1, 
        detach_prototypes: bool = False
    ):
        """
        Initializes the SimplePrototype module.

        Args:
            max_classes (int): Upper bound on total target classes (ways) per task batch.
            latent_dim (int): Dimensionality of feature representation vectors.
            keep_running_Prototype (bool): If True, tracks running moving average of class prototypes.
            momentum (float): Momentum value used for running prototype EMA updates.
            detach_prototypes (bool): If True, detaches returned prototypes from autograd graph.
        """
        # Call parent BasePrototype constructor
        super().__init__(max_classes=max_classes, detach_prototypes=detach_prototypes)
        
        # Store structural attributes
        self.max_classes = max_classes
        self.latent_dim = latent_dim
        self.keep_running_Prototype = keep_running_Prototype
        self.momentum = momentum

        # Register persistent buffers if running prototype tracking is active
        if self.keep_running_Prototype:
            # Register 2D buffer storing running average class prototypes: [max_classes, latent_dim]
            self.register_buffer("running_prototypes", torch.zeros(max_classes, latent_dim))
            # Register 1D boolean buffer tracking which classes have been initialized at least once
            self.register_buffer("initialized_classes", torch.zeros(max_classes, dtype=torch.bool))

    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates class feature centroids and active class presence masks.
        Optionally updates running prototype buffers when enabled.

        Args:
            features (torch.Tensor): Sample representations of shape [num_samples, latent_dim].
            labels (torch.Tensor): Target class indices of shape [num_samples].
            samples_mask (Optional[torch.Tensor]): Boolean mask indicating valid non-padded samples.
            **kwargs: Extra operational flags including 'training' mode boolean.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - out_centroids: Class centroids tensor of shape [max_classes, latent_dim].
                - out_mask: Boolean validity mask of shape [max_classes].
        """
        # Extract runtime training state from kwargs (defaults to True)
        training = kwargs.get('training', True)

        # 1. Convert integer labels to static one-hot assignment matrix: [num_samples, max_classes]
        one_hot = F.one_hot(labels, num_classes=self.max_classes).to(features.dtype)

        # 2. Apply sample validity mask if provided to zero out padded dummy samples
        if samples_mask is not None:
            # Expand sample mask shape for broadcasting: [num_samples, 1]
            mask_expanded = samples_mask.to(features.dtype).unsqueeze(-1)
            # Mask out one-hot assignments for invalid samples
            valid_one_hot = one_hot * mask_expanded
        else:
            valid_one_hot = one_hot

        # 3. Compute count of valid samples per class category: [max_classes, 1]
        class_counts = valid_one_hot.sum(dim=0, keepdim=True).T

        # 4. Sum feature vectors per class via matrix multiplication: [max_classes, latent_dim]
        class_sums = torch.matmul(valid_one_hot.T, features)

        # 5. Compute mean feature centroids safely avoiding division by zero
        centroids = class_sums / class_counts.clamp(min=1)

        # 6. Generate active class presence mask (True for classes present with count > 0)
        mask = (class_counts.squeeze(-1) > 0)

        # 7. Manage running prototype moving average updates if tracking is enabled
        if self.keep_running_Prototype:
            if training:
                # Prepare 2D masks for broadcasting across latent dimension: [max_classes, 1]
                valid_mask_2d = mask.unsqueeze(-1).to(features.dtype)
                is_init_2d = self.initialized_classes.unsqueeze(-1).to(features.dtype)
                
                # Compute standard Exponential Moving Average (EMA) for existing prototypes
                updated_existing = (1.0 - self.momentum) * self.running_prototypes + self.momentum * centroids.detach()
                
                # If a class is seen for the first time, initialize directly with centroid (avoiding zero-bias)
                new_val_for_valid = is_init_2d * updated_existing + (1.0 - is_init_2d) * centroids.detach()
                
                # Apply updates ONLY to classes present in current batch to prevent updating absent classes
                new_running_prototypes = valid_mask_2d * new_val_for_valid + (1.0 - valid_mask_2d) * self.running_prototypes
                
                # Out-of-place re-binding to support vmap functional tracing
                self.running_prototypes = new_running_prototypes
                self.initialized_classes = self.initialized_classes | mask
                
                # Output current batch centroids during training
                out_centroids = centroids
                out_mask = mask
            else:
                # During evaluation/inference mode, return running moving average prototypes
                out_centroids = self.running_prototypes
                out_mask = self.initialized_classes
        else:
            # Standard episodic mode without running prototype tracking
            out_centroids = centroids
            out_mask = mask

        # Detach prototypes from computational graph if requested
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