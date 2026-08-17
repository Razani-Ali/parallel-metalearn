import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
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
        # Call parent nn.Module constructor
        super().__init__()
        # Store maximum capacity of target classes
        self.max_classes = max_classes
        # Store flag for detaching prototypes from autograd computational graph
        self.detach_prototypes = detach_prototypes

    @abstractmethod
    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Abstract method for computing class centroids/prototypes.
        """
        pass


class SimplePrototype(BasePrototype):
    """
    Non-parametric, Class-Agnostic Centroid/Prototype Extractor Module with Per-Step Statistics.

    Computes class centers (prototypes) across feature tensors using vectorized matrix
    operations while respecting sample validity masks (samples_mask). Fully differentiable
    and compatible with PyTorch `torch.func.vmap`, `functional_call`, and MAML execution loops.

    Optionally maintains exponential moving averages (`running_prototypes`) per inner adaptation 
    step when `keep_running_Prototype=True` and `use_per_step_stats=True`, properly avoiding 
    zero-bias on newly initialized classes and correctly ignoring masked out samples.
    Purely functional: returns updated buffer states in a dictionary without modifying 
    `self` in-place, eliminating `vmap` tensor escape errors.
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
        # Initialize base prototype abstract parent class
        super().__init__(max_classes=max_classes, detach_prototypes=detach_prototypes)
        
        # Store latent dimensionality of input feature embeddings
        self.latent_dim = latent_dim
        # Store boolean flag enabling exponential moving average tracking for prototypes
        self.keep_running_Prototype = keep_running_Prototype
        # Store boolean flag enabling step-specific prototype buffers (for MAML++)
        self.use_per_step_stats = use_per_step_stats
        # Store maximum inner-loop step count threshold
        self.max_inner_steps = max_inner_steps
        # Store exponential smoothing momentum factor
        self.momentum = momentum

        # Register state buffers if prototype moving average tracking is enabled
        if self.keep_running_Prototype:
            if self.use_per_step_stats:
                # Register 3D persistent buffer: [max_inner_steps, max_classes, latent_dim]
                self.register_buffer(
                    "running_prototypes", 
                    torch.zeros(max_inner_steps, max_classes, latent_dim)
                )
                # Register 2D boolean initialization tracker: [max_inner_steps, max_classes]
                self.register_buffer(
                    "initialized_classes", 
                    torch.zeros(max_inner_steps, max_classes, dtype=torch.bool)
                )
            else:
                # Register 2D persistent buffer: [max_classes, latent_dim]
                self.register_buffer(
                    "running_prototypes", 
                    torch.zeros(max_classes, latent_dim)
                )
                # Register 1D boolean initialization tracker: [max_classes]
                self.register_buffer(
                    "initialized_classes", 
                    torch.zeros(max_classes, dtype=torch.bool)
                )

    def compute_class_centers(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None,
        task_buffers: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "center_head.",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates mean feature centroids and active class presence masks while 
        ignoring padded/dummy samples via samples_mask. Fully tracks gradients.

        Args:
            features (torch.Tensor): Extracted sample representations of shape [num_samples, latent_dim].
            labels (torch.Tensor): Ground-truth class index tensor of shape [num_samples].
            samples_mask (Optional[torch.Tensor]): Binary/boolean validity mask of shape [num_samples].
            task_buffers (Optional[Dict[str, torch.Tensor]]): Current task state buffers dictionary.
            prefix (str): Buffer key name prefix for model state mapping.
            **kwargs: Operational context flags such as 'inner_step' and 'training'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
                - centroids: Mean feature representations of shape [max_classes, latent_dim].
                - mask: Boolean tensor of shape [max_classes] indicating active classes.
                - updated_buffers: Dictionary containing state buffers updated purely out-of-place.
        """
        num_samples = features.shape[0]
        device = features.device
        num_step = kwargs.get('inner_step', 0)

        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        else:
            features = features
        
        if isinstance(num_step, int):
            step_idx = min(num_step, self.max_inner_steps - 1)
        else:
            step_idx = num_step

        # Zero-Shot Evaluation
        if num_samples == 0:
            buf_key = f"{prefix}running_prototypes"
            if task_buffers is not None and buf_key in task_buffers:
                r_proto_full = task_buffers[buf_key]
            else:
                r_proto_full = self.running_prototypes

            if self.use_per_step_stats:
                centroids = r_proto_full[step_idx]
            else:
                centroids = r_proto_full

            mask = torch.ones(self.max_classes, dtype=torch.bool, device=device)
            return centroids, mask, {}
        
        # Ensure step index is represented as a Tensor on the target compute device
        if isinstance(num_step, int):
            num_step = torch.tensor(num_step, device=features.device)
            # Clamp index to maximum inner steps boundary to prevent indexing out-of-bounds
            step_idx = torch.clamp(num_step, max=self.max_inner_steps - 1)
        else:
            step_idx = num_step

        # Extract training mode boolean flag from context dictionary (defaults to True)
        training = kwargs.get('training', True)

        # Generate class index tensor spanning full max_classes range: [max_classes]
        class_indices = torch.arange(self.max_classes, device=labels.device)
        
        # Generate static one-hot encoding matrix via pure out-of-place equality broadcast to avoid aten::scatter_
        one_hot = (labels.unsqueeze(-1) == class_indices.unsqueeze(0)).to(features.dtype)

        # Filter out invalid/padded dummy samples using samples_mask if provided
        if samples_mask is not None:
            # Cast mask tensor to feature floating-point dtype and add trailing dimension for broadcasting
            mask_expanded = samples_mask.to(features.dtype).unsqueeze(-1)  # Shape: [num_samples, 1]
            # Zero out one-hot allocations for masked padded samples
            valid_one_hot = one_hot * mask_expanded
        else:
            valid_one_hot = one_hot

        # Calculate valid sample count per class category: shape [max_classes, 1]
        class_counts = valid_one_hot.sum(dim=-2, keepdim=True).mT
        class_counts = torch.clamp(class_counts, min=1.0)

        # Aggregate feature vectors per class category via matrix multiplication: shape [max_classes, latent_dim]
        class_sums = torch.matmul(valid_one_hot.mT, features)

        # Compute mean class centroids using safe element-wise division (clamping prevents zero-division)
        centroids = class_sums / class_counts

        # Construct boolean validity mask indicating present active classes (sample count > 0)
        mask = (class_counts.squeeze(-1) > 0)

        # Initialize dictionary container for updated state buffers
        updated_buffers = {}

        # Handle running prototype exponential moving average logic if enabled
        if self.keep_running_Prototype:
            # Retrieve persistent state buffers from task_buffers dictionary if passed, otherwise fall back to self
            if task_buffers is not None and f"{prefix}running_prototypes" in task_buffers:
                r_proto_full = task_buffers[f"{prefix}running_prototypes"]
                r_init_full = task_buffers[f"{prefix}initialized_classes"]
            else:
                r_proto_full = self.running_prototypes
                r_init_full = self.initialized_classes

            # Slice current inner step's buffer view if per-step mode is active
            if self.use_per_step_stats:
                r_proto = r_proto_full[step_idx]
                r_init = r_init_full[step_idx]
            else:
                r_proto = r_proto_full
                r_init = r_init_full

            if training:
                # Expand active class mask for vector broadcasting along feature dimension
                valid_mask = mask.unsqueeze(-1).to(features.dtype)
                # Expand initialization tracker mask for vector broadcasting along feature dimension
                is_init = r_init.unsqueeze(-1).to(features.dtype)

                # Compute exponential moving average update step using current batch centroids
                updated_existing = (1 - self.momentum) * r_proto + self.momentum * centroids.detach()

                # Avoid zero-initialization bias: assign raw batch centroid directly for newly seen classes
                new_val_for_valid = is_init * updated_existing + (1 - is_init) * centroids.detach()

                # Apply update exclusively to active present classes in current batch mask
                new_running_prototypes = valid_mask * new_val_for_valid + (1 - valid_mask) * r_proto
                
                # Combine class initialization boolean states via bitwise OR
                new_init = r_init | mask

                # Construct new buffer tensors out-of-place to maintain pure functional execution
                if self.use_per_step_stats:
                    # Construct boolean mask identifying the current inner step index slice
                    step_mask = (torch.arange(self.max_inner_steps, device=features.device) == step_idx)
                    # Reshape step mask for 3D prototype broadcasting: [max_inner_steps, 1, 1]
                    step_mask_proto = step_mask.view(-1, 1, 1)
                    # Reshape step mask for 2D initialization broadcasting: [max_inner_steps, 1]
                    step_mask_init = step_mask.view(-1, 1)

                    # Update step slice out-of-place via torch.where selection
                    updated_rp_full = torch.where(step_mask_proto, new_running_prototypes.unsqueeze(0), r_proto_full)
                    updated_ic_full = torch.where(step_mask_init, new_init.unsqueeze(0), r_init_full)
                else:
                    # Direct assignment for standard non-per-step mode
                    updated_rp_full = new_running_prototypes
                    updated_ic_full = new_init

                # Populate updated buffer states inside dictionary (avoiding self mutation)
                updated_buffers[f"{prefix}running_prototypes"] = updated_rp_full
                updated_buffers[f"{prefix}initialized_classes"] = updated_ic_full

                # Assign current batch centroids and batch mask to outputs for training mode
                out_centroids = centroids
                out_mask = mask
            else:
                # Assign tracked running prototypes and initialization mask to outputs for evaluation mode
                out_centroids = r_proto
                out_mask = r_init
        else:
            # Standard non-running prototype mode assignment
            out_centroids = centroids
            out_mask = mask

        # Detach prototype outputs from autograd graph if detach_prototypes flag is active
        if self.detach_prototypes:
            return out_centroids.detach(), out_mask.detach(), updated_buffers

        # Return computed centroids, presence mask, and updated state buffers
        return out_centroids, out_mask, updated_buffers


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