import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from .prototype_calculator import SimplePrototype, BasePrototype
from ..distances.elastics import ElasticDistance
from ..distances.classics import Euclidean, Distance


class ProtoNet_Model(nn.Module):
    """
    Prototypical Network Model Wrapper.
    
    Encapsulates the backbone feature extractor, prototype calculator, and distance 
    metric. Performs a single unified forward pass processing both support and query 
    sets seamlessly. Fully compatible with torch.func.vmap and functional_call.
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        max_classes: int,
        latent_dim: int,
        distance_module: Optional[Union[Distance, ElasticDistance]] = None,
        drop_rate: float = 0.0,
        prototype_class: Optional[BasePrototype] = None,
        **kwargs
    ):
        """
        Initializes the ProtoNet_Model wrapper.

        Args:
            backbone (nn.Module): Feature extractor module.
            max_classes (int): Upper bound on target classes (ways).
            latent_dim (int): Dimensionality of extracted feature representations.
            distance_module (Optional[Union[Distance, ElasticDistance]]): Distance metric module.
            drop_rate (float): Dropout probability.
            prototype_class (Optional[BasePrototype]): Custom prototype calculator.
            **kwargs: Additional operational flags such as 'keep_running_Prototype'.
        """
        super().__init__()
        # Store backbone architecture
        self.backbone = backbone
        self.max_classes = max_classes
        self.latent_dim = latent_dim
        # Default to Euclidean distance if no custom metric module is provided
        self.distance_module = distance_module if distance_module else Euclidean()
        
        # Instantiate prototype extractor with per-step / running capabilities
        self.center_head = prototype_class if prototype_class else SimplePrototype(
            max_classes=max_classes,
            latent_dim=latent_dim,
            keep_running_Prototype=kwargs.get("keep_running_Prototype", False),
            use_per_step_stats=kwargs.get("use_per_step_stats", False),
            max_inner_steps=kwargs.get("max_inner_steps", 5)
        )
        self.drop_rate = drop_rate

        # Register non-persistent buffers for test-time inference deployment
        self.register_buffer("deployed_prototypes", None, persistent=False)
        self.register_buffer("deployed_class_mask", None, persistent=False)

    def forward(
        self, 
        x_q: torch.Tensor, 
        x_s: Optional[torch.Tensor] = None, 
        y_s: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Flexible functional forward pass for Prototypical Networks.
        
        - Episodic Mode (x_s and y_s provided): Computes dynamic prototypes from x_s.
        - Standard Inference Mode (x_s and y_s are None): Uses registered deployed_prototypes.

        Args:
            x_q (torch.Tensor): Query inputs batch [batch_size, ...].
            x_s (Optional[torch.Tensor]): Support set inputs. Defaults to None.
            y_s (Optional[Dict[str, torch.Tensor]]): Support set target dict. Defaults to None.
            **kwargs: Operational flags such as 'inner_step' and 'training'.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing 'logits', 'prototypes', 
                                     'features', and updated 'buffers'.
        """
        # Determine dropout training state
        dropout_training = kwargs.get('training', self.training)
        center_bufs = {}

        # Check if support set contains valid samples for Few-Shot calculation
        has_support_samples = (x_s is not None and y_s is not None and x_s.shape[0] > 0)

        # 1. Extract support set features and calculate prototypes in Episodic Mode
        if has_support_samples:
            # Pass support set through backbone
            feat_s = self.backbone(x_s, **kwargs)

            # Flatten feature representation if dimensional rank > 2
            feat_s_flat = feat_s.flatten(start_dim=-1) if feat_s.dim() > 2 else feat_s

            # Apply dropout regularization on support features
            feat_s_flat = torch.nn.functional.dropout(
                feat_s_flat, 
                p=self.drop_rate, 
                training=dropout_training
            )

            # Unpack ground-truth labels and optional sample validity masks
            labels_s = y_s["labels"]
            samples_mask_s = y_s.get("samples_mask", None)

            # 💡 Correctly unpack 3 values (prototypes, class_mask, center_bufs) to prevent unpacking errors
            prototypes, class_mask, center_bufs = self.center_head.compute_class_centers(
                features=feat_s_flat, 
                labels=labels_s, 
                samples_mask=samples_mask_s,
                prefix="center_head.",
                **kwargs
            )

        else:
            # 🚀 Zero-Shot Mode / Fallback
            if self.deployed_prototypes is not None:
                prototypes = self.deployed_prototypes
                class_mask = self.deployed_class_mask

            else:
                dummy_feats = torch.empty((0, self.latent_dim), device=x_q.device)
                dummy_labels = torch.empty((0,), dtype=torch.long, device=x_q.device)
                
                prototypes, class_mask, center_bufs = self.center_head.compute_class_centers(
                    features=dummy_feats,
                    labels=dummy_labels,
                    prefix="center_head.",
                    **kwargs
                )

        # 2. Extract query set features
        feat_q = self.backbone(x_q, **kwargs)

        # Flatten query feature representation if dimensional rank > 2
        feat_q_flat = feat_q.flatten(start_dim=-1) if feat_q.dim() > 2 else feat_q

        # Apply dropout regularization on query features
        feat_q_flat = torch.nn.functional.dropout(
            feat_q_flat, 
            p=self.drop_rate, 
            training=dropout_training
        )

        # 3. Calculate classification logits via distance metric module
        logits = self.distance_module(queries=feat_q_flat, prototypes=prototypes, class_mask=class_mask)

        # Collect current model buffers and merge with updated prototype buffers
        current_buffers = dict(self.named_buffers())
        current_buffers.update(center_bufs)

        # Return standardized functional output dictionary
        return {
            "logits": logits,
            "prototypes": prototypes,
            "features": feat_q,
            "buffers": current_buffers
        }