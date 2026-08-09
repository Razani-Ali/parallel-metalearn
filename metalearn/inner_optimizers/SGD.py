from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.utils._pytree as pytree
from .base import BaseInnerOptimizer


class InnerSGD(BaseInnerOptimizer):
    """
    Inner-loop Stochastic Gradient Descent (SGD) optimizer for functional meta-learning.
    
    Handles task-specific fast-weight updates across PyTree structures, supporting 
    first-order approximations, learnable per-step or per-layer learning rates, 
    and vectorization via torch.func.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.01,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_lr: bool = True, 
        **kwargs
    ):
        """
        Initializes the InnerSGD optimizer with meta-learnable or fixed learning rate parameters.

        Args:
            initial_fast_weights (OrderedDict): Dictionary of parameters (fast weights) to optimize.
            first_order (bool): If True, detaches gradients to compute first-order MAML updates.
            inner_lr (float): Initial fast-weight learning rate value.
            max_inner_steps (int): Maximum number of adaptation steps in the inner loop.
            per_layer (bool): If True, assigns a separate learning rate tensor per layer parameter.
            per_step (bool): If True, allocates dedicated learning rates for each adaptation step.
            learn_lr (bool): If True, registers learning rates as meta-learnable parameters.
            **kwargs: Additional keyword arguments passed to BaseInnerOptimizer.
        """
        # Call the parent optimizer initialization with model configuration and metadata
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register or instantiate the fast-weight learning rate structure
        self.lrs = self._make_meta_param(inner_lr, learn_lr)

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """
        Initializes state containers for stateless SGD optimization.

        Args:
            fast_weights (Dict[str, torch.Tensor]): Current fast weights of the model.

        Returns:
            Dict: Empty dictionary as standard SGD maintains no historical momentum states.
        """
        # Return an empty dictionary because SGD does not track momentum or second moments
        return {}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Executes a single functional inner-loop SGD update step over task parameters.

        Args:
            fast_weights (Dict[str, torch.Tensor]): Dictionary of current task parameter tensors.
            gradients (Dict[str, torch.Tensor]): Gradients corresponding to fast_weights.
            state (Dict): Optimizer state container (unused for SGD).
            step (int): Current inner-loop gradient step index.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict]: Updated fast weights dictionary and empty state dictionary.
        """
        # Check if first-order MAML is enabled to prevent higher-order derivative tracking
        if self.first_order:
            # Detach gradient tensors from autograd graph to ignore second-order terms
            grads_tree = pytree.tree_map(lambda g: g.detach(), gradients)
        else:
            # Keep original gradient computation graph for full second-order MAML
            grads_tree = gradients

        # Retrieve structured learning rate PyTree corresponding to current step and layer configuration
        lr_tree = self._get_lr_tree(fast_weights, step)

        # Apply gradient descent update element-wise across nested parameter dictionaries
        new_weights = pytree.tree_map(
            lambda p, g, lr: p - lr * g,
            fast_weights,
            grads_tree,
            lr_tree
        )

        # Return updated parameters along with an empty optimizer state dictionary
        return new_weights, {}