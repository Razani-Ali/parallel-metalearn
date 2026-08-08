from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.utils._pytree as pytree
from .base import BaseInnerOptimizer


class InnerSGD(BaseInnerOptimizer):
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
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        self.lrs = self._make_meta_param(inner_lr, learn_lr)

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        return {}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        
        if self.first_order:
            grads_tree = pytree.tree_map(lambda g: g.detach(), gradients)
        else:
            grads_tree = gradients

        lr_tree = self._get_lr_tree(fast_weights, step)

        new_weights = pytree.tree_map(
            lambda p, g, lr: p - lr * g,
            fast_weights,
            grads_tree,
            lr_tree
        )

        return new_weights, {}