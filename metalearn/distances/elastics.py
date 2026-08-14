from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Elastic factors for each class as suggested in paper:
# Meta-learning with elastic prototypical network for fault transfer diagnosis
# of bearings under unstable speeds 
# https://doi.org/10.1016/j.ress.2024.110001


class ElasticDistance(nn.Module, ABC):
    """
    Abstract Base Class for Elastic Distance Metrics with class-level learnable scaling.
    """

    def __init__(self, class_num: int, eps: float = 1e-6):
        super().__init__()
        assert class_num is not None, "class_num must be provided"
        self.class_num = class_num
        self.scaling = nn.Parameter(torch.ones(class_num))
        self.eps = eps

    @abstractmethod
    def forward(
        self,
        queries: torch.Tensor,
        prototypes: torch.Tensor,
        class_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            queries (torch.Tensor): Query embeddings of shape (N, D).
            prototypes (torch.Tensor): Prototype embeddings of shape (C, D).
            class_mask (Optional[torch.Tensor]): Boolean mask indicating active classes (C,).

        Returns:
            torch.Tensor: Classification logits of shape (N, C).
        """
        pass

    def _apply_elastic_scaling_and_mask(
        self, 
        raw_distances: torch.Tensor, 
        class_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Applies per-class elastic scaling factors and masks absent classes."""
        num_proto = raw_distances.shape[1]
        
        # Clamp scaling factors to prevent division by zero
        class_scale = torch.clamp(self.scaling[:num_proto], min=self.eps)  # Shape: (C,)
        
        # Apply class elasticity factor: d_elastic = d / scale
        elastic_dists = raw_distances / class_scale[None, :]  # Shape: (N, C)
        
        # Convert distances to logits (negative distance)
        logits = -elastic_dists
        
        # Zero-out absent classes
        if class_mask is not None:
            logits = logits.masked_fill(~class_mask.unsqueeze(0), float('-inf'))
            
        return logits


class ElasticEuclidean(ElasticDistance):
    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=2.0) ** 2
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticManhattan(ElasticDistance):
    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=1.0)
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticMahalanobis(ElasticDistance):
    def __init__(self, class_num: int, in_dim: int, eps: float = 1e-6):
        super().__init__(class_num=class_num, eps=eps)
        self.L = nn.Parameter(torch.eye(in_dim))

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_proj = torch.matmul(queries, self.L.T)
        y_proj = torch.matmul(prototypes, self.L.T)

        dists = torch.cdist(x_proj, y_proj, p=2.0) ** 2
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticDiagonalL2(ElasticDistance):
    def __init__(self, class_num: int, in_dim: int, eps: float = 1e-6):
        super().__init__(class_num=class_num, eps=eps)
        self.L = nn.Parameter(torch.ones(in_dim))

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        scaling = torch.clamp(self.L, min=self.eps)
        x_scaled = queries * scaling[None, :]
        y_scaled = prototypes * scaling[None, :]

        dists = torch.cdist(x_scaled, y_scaled, p=2.0) ** 2
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticMinkowski(ElasticDistance):
    def __init__(self, class_num: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__(class_num=class_num, eps=eps)
        self.p = p

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=self.p) ** self.p
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticChebyshevDistance(ElasticDistance):
    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        diff = torch.abs(queries.unsqueeze(1) - prototypes.unsqueeze(0))
        dists = torch.max(diff, dim=-1).values
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticCosine(ElasticDistance):
    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_norm = F.normalize(queries, p=2, dim=-1, eps=self.eps)
        y_norm = F.normalize(prototypes, p=2, dim=-1, eps=self.eps)

        sim = torch.matmul(x_norm, y_norm.T)
        dists = 1.0 - sim
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticMinkowskiMahalanobis(ElasticDistance):
    def __init__(self, class_num: int, in_dim: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__(class_num=class_num, eps=eps)
        self.L = nn.Parameter(torch.eye(in_dim))
        self.p = p

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_proj = torch.matmul(queries, self.L.T)
        y_proj = torch.matmul(prototypes, self.L.T)

        dists = torch.cdist(x_proj, y_proj, p=self.p) ** self.p
        return self._apply_elastic_scaling_and_mask(dists, class_mask)


class ElasticDiagonalMinkowski(ElasticDistance):
    def __init__(self, class_num: int, in_dim: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__(class_num=class_num, eps=eps)
        self.L = nn.Parameter(torch.ones(in_dim))
        self.p = p

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        scaling = torch.clamp(self.L, min=self.eps)
        x_scaled = queries * scaling[None, :]
        y_scaled = prototypes * scaling[None, :]

        dists = torch.cdist(x_scaled, y_scaled, p=self.p) ** self.p
        return self._apply_elastic_scaling_and_mask(dists, class_mask)