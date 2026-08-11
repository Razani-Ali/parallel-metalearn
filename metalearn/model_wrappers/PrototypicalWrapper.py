import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from .prototype_calculator import SimplePrototype, BasePrototype
from ..distances.elastics import ElasticDistance
from ..distances.classics import Euclidean, Distance
from metalearn.dataset.Scalers import BaseScaler


class ProtoNet_Model(nn.Module):
    """
    Prototypical Network Model Wrapper.
    
    Encapsulates the backbone feature extractor, prototype calculator, and distance 
    metric. Performs a single unified forward pass processing both support and query 
    sets seamlessly. Fully compatible with torch.func.vmap.
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        max_classes: int,
        distance_module: Optional[Union[Distance, ElasticDistance]] = None,
        drop_rate: float = 0.0,
        scaler: Optional[BaseScaler] = None,
        prototype_class: Optional[BasePrototype] = None,
        latent_dim: int = None,
    ):
        """
        Initializes the ProtoNet_Model wrapper.

        Args:
            backbone (nn.Module): Feature extractor module.
            max_classes (int): Upper bound on target classes (ways).
            distance_module (nn.Module): Distance computation module (e.g., EuclideanDistance).
            drop_rate (float): Dropout probability.
            scaler (Optional[nn.Module]): Data Scaler.
            prototype_class (Optional[BasePrototype]): Custom prototype calculator.
            latent_dim (int): Dimensionality of extracted feature vectors.
        """
        super().__init__()
        self.backbone = backbone
        self.max_classes = max_classes
        self.distance_module = distance_module if distance_module else Euclidean()
        
        self.center_head = prototype_class if prototype_class else SimplePrototype(max_classes=max_classes, latent_dim=latent_dim)
        self.drop_rate = drop_rate
        self.scaler = scaler

        # Register non-persistent buffers for inference deployment
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
        Flexible forward pass for Prototypical Networks.
        
        - Episodic Mode (x_s and y_s provided): Computes dynamic prototypes from x_s.
        - Standard Inference Mode (x_s and y_s are None): Uses registered deployed_prototypes.
        
        1. Extracts support features.
        2. Computes prototypes using support labels and samples_mask.
        3. Extracts query features.
        4. Calculates distances/logits between query features and prototypes.

        Args:
            x_q (torch.Tensor): Query inputs or standard input batch [batch_size, ...].
            x_s (Optional[torch.Tensor]): Support set inputs. Defaults to None.
            y_s (Optional[Dict[str, torch.Tensor]]): Support set target dict. Defaults to None.
            **kwargs: Operational flags such as 'inner_step' and 'training'.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'logits', 'prototypes', and 'features'.
        """
        dropout_training = kwargs.get('training', self.training)

        # 1. Extract support set features
        if x_s is not None and y_s is not None:
            # Episodic Meta-Learning Mode: Compute dynamic prototypes from support set
            if self.scaler:
                x_s = self.scaler(x_s, **kwargs)

            feat_s = self.backbone(x_s, **kwargs)

            if feat_s.dim() > 2:
                features_flat = feat_s.flatten(start_dim=-1)
            else:
                features_flat = feat_s

            feat_s_flat = torch.nn.functional.dropout(
                features_flat, 
                p=self.drop_rate, 
                training=dropout_training
            )

            # 2. Compute prototypes using support features and ground-truth labels
            labels_s = y_s["labels"]
            samples_mask_s = y_s.get("samples_mask", None)
            prototypes, class_mask = self.center_head.compute_class_centers(
                features=feat_s_flat, 
                labels=labels_s, 
                samples_mask=samples_mask_s,
                **kwargs
            )

        else:
            # Standard Inference Mode: Use pre-registered prototypes
            if self.deployed_prototypes is None:
                raise RuntimeError(
                    "❌ Deployed prototypes not found! "
                    "Either pass (x_s, y_s) for episodic evaluation or call "
                    "`algorithm.adapt_and_update(x_s, y_s)` prior to standard inference."
                )
            prototypes = self.deployed_prototypes
            class_mask = self.deployed_class_mask

        # 3. Extract query set features
        if self.scaler:
            x_q = self.scaler(x_q, **kwargs)

        feat_q = self.backbone(x_q, **kwargs)

        if feat_q.dim() > 2:
            features_flat = feat_q.flatten(start_dim=-1)
        else:
            features_flat = feat_q

        feat_q_flat = torch.nn.functional.dropout(
            features_flat, 
            p=self.drop_rate, 
            training=dropout_training
        )

        # 4. Compute logits using the integrated distance metric module
        logits = self.distance_module(queries=feat_q_flat, prototypes=prototypes, class_mask=class_mask)

        # Return standardized output dictionary compatible with BaseLoss
        current_buffers = dict(self.named_buffers())
        return {
            "logits": logits,
            "prototypes": prototypes,
            "features": feat_q,
            "buffers": current_buffers
        }