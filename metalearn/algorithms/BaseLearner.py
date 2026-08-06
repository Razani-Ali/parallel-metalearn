from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional
import torch
import torch.nn as nn


class MetaOptimizer(nn.Module, ABC):
    """
    Abstract Base Class for all Meta-Learning Algorithms (e.g., MAML, ProtoNet).
    
    Establishes a unified interface for meta-training/evaluation steps 
    and test-time adaptation/deployment.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def step(
        self, 
        task_loader: Any, 
        training: bool = True, 
        **kwargs: Any
    ) -> Tuple[float, float]:
        """
        Executes a single meta-training or meta-validation step over a batch of tasks.

        Args:
            task_loader: Dataloader/Iterable yielding batches of tasks 
                         (typically yielding Xs, Ys, Xq, Yq).
            training (bool): If True, computes meta-gradients and updates outer model parameters.
            **kwargs: Additional contextual arguments (e.g., epoch, inner_steps override).

        Returns:
            Tuple[float, float]: A tuple containing:
                - mean_meta_loss (float): Average meta-loss value across the batch of tasks.
                - mean_metric (float): Average evaluation metric (e.g., accuracy) across tasks.
        """
        pass

    @abstractmethod
    def adapt_and_update(
        self, 
        Xsupport: torch.Tensor, 
        Ysupport: torch.Tensor, 
        **kwargs: Any
    ) -> None:
        """
        Adapts parameters on the provided support set and permanently updates the internal model.
        Intended exclusively for test-time adaptation, fine-tuning, or deployment.

        Args:
            Xsupport (torch.Tensor): Support set inputs of shape (num_tasks, K_support, ...).
            Ysupport (torch.Tensor): Support set targets of shape (num_tasks, K_support, ...).
            **kwargs: Additional context arguments (e.g., custom inner adaptation steps).
        """
        pass