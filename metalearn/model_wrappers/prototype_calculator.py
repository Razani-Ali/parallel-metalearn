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
    Non-parametric, Class-Agnostic Centroid/Prototype Extractor Module.
    
    Computes class centers (prototypes) across feature tensors using vectorized matrix
    operations while respecting sample validity masks (samples_mask). Fully differentiable
    and compatible with PyTorch vmap, functional_call, and MAML execution loops.
    """

    def __init__(self, max_classes: int, detach_prototypes: bool = False):
        """
        Initializes the MaskedClassCenterHead module.

        Args:
            max_classes (int): Upper bound on total target classes (ways) per task batch.
            detach_prototypes (bool): Whether to detach prototypes tensor from computational graph or not
        """
        # Initialize PyTorch parent module
        super().__init__()
        # Store fixed maximum class count for static one-hot tensor dimensions
        self.max_classes = max_classes

        self.detach_prototypes = detach_prototypes

    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates mean feature centroids and active class presence masks while 
        ignoring padded/dummy samples via samples_mask. Fully tracks gradients.

        Args:
            features (torch.Tensor): Extracted sample representations of shape 
                                    [num_samples, latent_dim].
            labels (torch.Tensor): Ground-truth class index tensor of shape 
                                  [num_samples].
            samples_mask (Optional[torch.Tensor]): Binary/boolean validity mask 
                                                  of shape [num_samples].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - centroids: Mean feature representations of shape [max_classes, latent_dim].
                - mask: Boolean tensor of shape [max_classes] indicating active classes.
        """
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

        # Return computed class prototypes and binary presence mask tuple
        if self.detach_prototypes:
            return centroids.detach(), mask.detach()

        return centroids, mask


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