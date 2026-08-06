from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Tuple, Any, Iterator
import torch
from torch import nn


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The inner-loop optimizer parameterization incorporates concepts from:
#
# 1. Base Learnable Inner Learning Rate Framework:
#    [1] Z. Li, F. Zhou, F. Chen, and H. Li, "Meta-SGD: Learning to Learn 
#        Quickly for Few-Shot Learning," arXiv preprint arXiv:1707.09835, 2017.
#
# 2. Per-Step & Per-Layer Learnable Inner Learning Rates (LSLR):
#    [2] A. Antoniou, H. Edwards, and A. Storkey, "How to train your MAML," 
#        in Int. Conf. Learn. Representations (ICLR), 2019. arXiv:1810.09502.
# ==============================================================================


class BaseInnerOptimizer(nn.Module, ABC):
    """
    Abstract Base Class for Task-Level (Inner-Loop) Optimizers in Meta-Learning.

    Manages per-layer and per-step meta-parameters (e.g., learnable inner learning rates)
    and provides a PyTree-compatible interface for updating fast weights.
    """

    def __init__(
        self, 
        initial_fast_weights: OrderedDict,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        **kwargs: Any
    ):
        """
        Initializes the base inner optimizer.

        Args:
            initial_fast_weights (OrderedDict): Dictionary of initial model fast weights.
            max_inner_steps (int): Maximum number of adaptation steps in the inner loop.
            per_layer (bool): If True, assigns distinct learning rates per layer/parameter.
            per_step (bool): If True, assigns distinct learning rates per adaptation step.
            **kwargs: Additional optional keyword arguments.
        """
        super().__init__()
        self.max_inner_steps = max_inner_steps
        self.per_layer = per_layer
        self.per_step = per_step
        # Extract parameter names to structure per-layer parameter dictionaries
        self.param_names = list(initial_fast_weights.keys())

    def _make_meta_param(self, init_val: float, learnable: bool) -> nn.ParameterDict:
        """
        Helper method to instantiate per-layer/per-step meta-parameters.

        Args:
            init_val (float): Initial value for the parameters.
            learnable (bool): Whether gradients should be computed for these parameters.

        Returns:
            nn.ParameterDict: Encapsulated meta-parameters.
        """
        params = nn.ParameterDict()
        
        if self.per_layer:
            # Create a unique parameter entry for each layer in the network
            for p_name in self.param_names:
                clean_name = p_name.replace(".", "_")
                size = (self.max_inner_steps,) if self.per_step else (1,)
                params[clean_name] = nn.Parameter(torch.full(size, init_val), requires_grad=learnable)
        else:
            # Create a single global parameter shared across all layers
            size = (self.max_inner_steps,) if self.per_step else (1,)
            params["global"] = nn.Parameter(torch.full(size, init_val), requires_grad=learnable)
            
        return params

    def _get_lr_tree(self, fast_weights: Dict[str, torch.Tensor], step: int) -> Dict[str, torch.Tensor]:
        """
        Constructs a PyTree of learning rates matching the exact structure of fast_weights.

        Args:
            fast_weights (Dict[str, torch.Tensor]): Model parameters dictionary.
            step (int): Current inner loop iteration index.

        Returns:
            Dict[str, torch.Tensor]: A learning rate PyTree with matching keys.
        """
        # Clamp step index to maximum inner steps boundary to prevent out-of-bounds indexing
        step_idx = min(step, self.max_inner_steps - 1) if self.per_step else 0
        
        if self.per_layer:
            # Map corresponding layer-specific learning rates to parameter keys
            return {
                name: self.lrs[name.replace(".", "_")][step_idx]
                for name in fast_weights.keys()
            }
        else:
            # Broadcast the global learning rate to all parameter keys
            lr_val = self.lrs["global"][step_idx]
            return {name: lr_val for name in fast_weights.keys()}

    @abstractmethod
    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        first_order: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Abstract forward pass to update fast weights based on computed gradients.
        Must be overridden by subclasses.
        """
        pass

    def inner_lr_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Yields all learnable meta-parameters managing inner learning rates.

        Yields:
            Tuple[str, nn.Parameter]: Named parameter tuples for outer optimizer registration.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param