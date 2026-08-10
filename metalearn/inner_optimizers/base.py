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

    Manages per-layer and per-step meta-parameters (e.g., learnable inner learning rates,
    momentum factors, and beta scalars) and provides a PyTree-compatible interface 
    for updating fast weights without in-place mutations.
    """

    def __init__(
        self, 
        initial_fast_weights: OrderedDict,
        first_order: bool = False,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        **kwargs: Any
    ):
        """
        Initializes the base inner optimizer interface.

        Args:
            initial_fast_weights (OrderedDict): Dictionary of initial model fast weights.
            first_order (bool): If True, breaks computational graph for first-order MAML.
            max_inner_steps (int): Maximum number of adaptation steps in the inner loop.
            per_layer (bool): If True, assigns distinct hyper-parameters per layer parameter.
            per_step (bool): If True, assigns distinct hyper-parameters per adaptation step.
            **kwargs: Additional keyword arguments.
        """
        # Call parent constructors for nn.Module and ABC
        super().__init__()
        # Store inner step upper bound
        self.max_inner_steps = max_inner_steps
        # Store first-order graph truncation flag
        self.first_order = first_order
        # Store per-layer parameterization flag
        self.per_layer = per_layer
        # Store per-step parameterization flag
        self.per_step = per_step
        # Extract parameter names to structure layer-wise parameter dictionaries
        self.param_names = list(initial_fast_weights.keys())

    def _make_meta_param(
        self, 
        init_val: float, 
        learnable: bool, 
        param_name: str = "global"
    ) -> nn.ParameterDict:
        """
        Instantiates per-layer/per-step meta-parameters for any scalar hyper-parameter.

        Args:
            init_val (float): Initial value for the hyper-parameter tensor.
            learnable (bool): Whether gradients should be tracked for outer meta-updates.
            param_name (str): Identifier prefix for the parameter dictionary keys.

        Returns:
            nn.ParameterDict: Encapsulated meta-parameters dictionary.
        """
        # Initialize parameter container dictionary
        params = nn.ParameterDict()
        
        # Check if layer-specific hyper-parameters are requested
        if self.per_layer:
            # Create a unique parameter tensor for each layer in the model
            for p_name in self.param_names:
                # Sanitize layer string name for PyTorch ParameterDict registration
                clean_name = f"{param_name}_{p_name.replace('.', '_')}"
                # Determine tensor shape based on per-step configuration
                size = (self.max_inner_steps,) if self.per_step else (1,)
                # Register parameter tensor initialized with init_val
                params[clean_name] = nn.Parameter(torch.full(size, init_val), requires_grad=learnable)
        else:
            # Create a single global parameter shared across all neural layers
            size = (self.max_inner_steps,) if self.per_step else (1,)
            # Register single global hyper-parameter tensor
            params[f"{param_name}_global"] = nn.Parameter(torch.full(size, init_val), requires_grad=learnable)
            
        return params

    def _get_param_tree(
        self, 
        param_dict: nn.ParameterDict, 
        fast_weights: Dict[str, torch.Tensor], 
        step: int,
        param_name: str = "global"
    ) -> Dict[str, torch.Tensor]:
        """
        Constructs a PyTree of hyper-parameters matching the structure of fast_weights.

        Args:
            param_dict (nn.ParameterDict): Source meta-parameter dictionary container.
            fast_weights (Dict[str, torch.Tensor]): Model parameters dictionary structure.
            step (int): Current inner loop adaptation step index.
            param_name (str): Identifier prefix used during parameter instantiation.

        Returns:
            Dict[str, torch.Tensor]: Hyper-parameter PyTree matching fast_weights keys.
        """
        # Clamp step index to boundary to prevent index out-of-bounds errors
        step_idx = min(step, self.max_inner_steps - 1) if self.per_step else 0
        
        # Check if parameterization is per-layer
        if self.per_layer:
            # Map layer-specific parameter tensor at current step index
            return {
                name: param_dict[f"{param_name}_{name.replace('.', '_')}"][step_idx]
                for name in fast_weights.keys()
            }
        else:
            # Broadcast global parameter value across all layer keys
            val = param_dict[f"{param_name}_global"][step_idx]
            return {name: val for name in fast_weights.keys()}

    @abstractmethod
    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Abstract forward pass to update fast weights based on computed gradients.

        Args:
            fast_weights (Dict[str, torch.Tensor]): Current fast weights dictionary.
            gradients (Dict[str, torch.Tensor]): Gradients corresponding to fast weights.
            state (Dict): Optimizer state container (e.g., momentum, square averages).
            step (int): Current inner adaptation step index.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict]: Tuple of updated fast weights and new state dict.
        """
        pass

    def inner_lr_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Yields all learnable meta-parameters managing inner hyper-parameters.

        Yields:
            Tuple[str, nn.Parameter]: Named parameter tuples for outer optimizer registration.
        """
        # Iterate over registered named parameters in the module
        for name, param in self.named_parameters():
            # Filter parameters requiring gradient computation
            if param.requires_grad:
                yield name, param