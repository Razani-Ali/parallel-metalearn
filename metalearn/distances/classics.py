from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Distance(nn.Module, ABC):
    """
    Abstract Base Class for all Distance Metrics in Prototypical Networks.
    Converts distances into classification logits (-distance) and safely masks absent classes.
    """

    def __init__(self):
        super().__init__()

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
            queries (torch.Tensor): Query feature embeddings of shape (N, D).
            prototypes (torch.Tensor): Prototype feature embeddings of shape (C, D).
            class_mask (Optional[torch.Tensor]): Boolean presence mask of shape (C,).

        Returns:
            torch.Tensor: Logits tensor (negative distance) of shape (N, C).
        """
        pass

    def _mask_logits(self, logits: torch.Tensor, class_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper method to fill absent classes with -inf."""
        if class_mask is not None:
            logits = logits.masked_fill(~class_mask.unsqueeze(0), float('-inf'))
        return logits


class Euclidean(Distance):
    """Standard Squared Euclidean Distance metric."""

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=2.0) ** 2
        logits = -dists
        return self._mask_logits(logits, class_mask)


class Manhattan(Distance):
    """Manhattan (L1) Distance metric."""

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=1.0)
        logits = -dists
        return self._mask_logits(logits, class_mask)


class Mahalanobis(Distance):
    """Mahalanobis Distance metric with learnable linear transformation tensor L."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.L = nn.Parameter(torch.eye(in_dim))

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_proj = torch.matmul(queries, self.L.T)
        y_proj = torch.matmul(prototypes, self.L.T)

        dists = torch.cdist(x_proj, y_proj, p=2.0) ** 2
        logits = -dists
        return self._mask_logits(logits, class_mask)


class DiagonalL2(Distance):
    """Weighted L2 Distance metric with learnable diagonal dimension scaling."""

    def __init__(self, in_dim: int, eps: float = 1e-6):
        super().__init__()
        self.L = nn.Parameter(torch.ones(in_dim))
        self.eps = eps

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        scaling = torch.clamp(self.L, min=self.eps)
        x_scaled = queries * scaling[None, :]
        y_scaled = prototypes * scaling[None, :]

        dists = torch.cdist(x_scaled, y_scaled, p=2.0) ** 2
        logits = -dists
        return self._mask_logits(logits, class_mask)


class Minkowski(Distance):
    """Minkowski (L_p) Distance metric."""

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = p

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        dists = torch.cdist(queries, prototypes, p=self.p) ** self.p
        logits = -dists
        return self._mask_logits(logits, class_mask)


class ChebyshevDistance(Distance):
    """Chebyshev (Infinity-norm) Distance metric."""

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # ||x - y||_inf = max(|x_i - y_i|)
        diff = torch.abs(queries.unsqueeze(1) - prototypes.unsqueeze(0))  # (N, C, D)
        dists = torch.max(diff, dim=-1).values  # (N, C)
        logits = -dists
        return self._mask_logits(logits, class_mask)


class CosineDistance(Distance):
    """Cosine Distance metric."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_norm = F.normalize(queries, p=2, dim=-1, eps=self.eps)
        y_norm = F.normalize(prototypes, p=2, dim=-1, eps=self.eps)

        sim = torch.matmul(x_norm, y_norm.T)
        dists = 1.0 - sim
        logits = -dists
        return self._mask_logits(logits, class_mask)


class MinkowskiMahalanobis(Distance):
    """Minkowski-Mahalanobis Distance metric."""

    def __init__(self, in_dim: int, p: float = 3.0):
        super().__init__()
        self.L = nn.Parameter(torch.eye(in_dim))
        self.p = p

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x_proj = torch.matmul(queries, self.L.T)
        y_proj = torch.matmul(prototypes, self.L.T)

        dists = torch.cdist(x_proj, y_proj, p=self.p) ** self.p
        logits = -dists
        return self._mask_logits(logits, class_mask)


class DiagonalMinkowski(Distance):
    """Diagonal-scaled Minkowski Distance metric."""

    def __init__(self, in_dim: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.L = nn.Parameter(torch.ones(in_dim))
        self.p = p
        self.eps = eps

    def forward(self, queries: torch.Tensor, prototypes: torch.Tensor, class_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        scaling = torch.clamp(self.L, min=self.eps)
        x_scaled = queries * scaling[None, :]
        y_scaled = prototypes * scaling[None, :]

        dists = torch.cdist(x_scaled, y_scaled, p=self.p) ** self.p
        logits = -dists
        return self._mask_logits(logits, class_mask)