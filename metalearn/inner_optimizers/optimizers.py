from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.utils._pytree as pytree
from metalearn.inner_optimizers.base import BaseInnerOptimizer


class InnerSGD(BaseInnerOptimizer):
    """
    Inner-loop Stochastic Gradient Descent (SGD) supporting per-layer/per-step learning rates.
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
        """Initializes the InnerSGD optimizer with learnable/fixed learning rate parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learnable learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_lr, "lr")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes state containers for stateless SGD optimization."""
        # SGD maintains no historical momentum states
        return {}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional SGD update step across fast weights."""
        # Detach gradients if first-order approximation is enabled
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Extract current step learning rate tree
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")

        # Apply standard gradient descent formula: p_next = p - lr * g
        new_weights = pytree.tree_map(lambda p, g, lr: p - lr * g, fast_weights, grads_tree, lr_tree)
        # Return updated fast weights and empty state dictionary
        return new_weights, {}


class InnerSGDMomentum(BaseInnerOptimizer):
    """
    Inner-loop SGD with Momentum and optional Nesterov acceleration.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerSGDMomentum with per-layer/per-step momentum parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register momentum factor meta-parameters
        self.momentums = self._make_meta_param(momentum, learn_params, "mom")
        # Register dampening factor meta-parameters
        self.dampenings = self._make_meta_param(dampening, learn_params, "damp")
        # Store Nesterov acceleration boolean flag
        self.nesterov = nesterov

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes momentum buffer state tensors with zeros matching fast_weights shapes."""
        return {"momentum_buffer": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional SGD with Momentum update step."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for the current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        mom_tree = self._get_param_tree(self.momentums, fast_weights, step, "mom")
        damp_tree = self._get_param_tree(self.dampenings, fast_weights, step, "damp")
        # Retrieve previous momentum buffer tree
        buf_tree = state["momentum_buffer"]

        # Inner mathematical update closure per tensor leaf
        def update_fn(p, g, lr, mom, damp, buf):
            # Calculate updated momentum buffer: buf_next = mom * buf + (1 - damp) * g
            buf_next = mom * buf + (1 - damp) * g
            # Calculate effective gradient based on Nesterov configuration
            g_eff = g + mom * buf_next if self.nesterov else buf_next
            # Apply weight update: p_next = p - lr * g_eff
            p_next = p - lr * g_eff
            return p_next, buf_next

        # Execute leaf update closure over PyTree leaves
        results = pytree.tree_map(update_fn, fast_weights, grads_tree, lr_tree, mom_tree, damp_tree, buf_tree)

        # Unpack updated parameters and new momentum state buffer
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {"momentum_buffer": pytree.tree_map(lambda x: x[1], results)}
        return new_weights, new_state


class InnerRMSprop(BaseInnerOptimizer):
    """
    Inner-loop RMSprop optimizer with per-layer/per-step learnable hyper-parameters.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerRMSprop with learnable decay rate (alpha) and epsilon parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register exponential smoothing factor (alpha) meta-parameters
        self.alphas = self._make_meta_param(alpha, learn_params, "alpha")
        # Register numerical stability (eps) meta-parameters
        self.epss = self._make_meta_param(eps, learn_params, "eps")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes mean square gradient average buffer tensors with zeros."""
        return {"square_avg": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional RMSprop update step."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for the current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        alpha_tree = self._get_param_tree(self.alphas, fast_weights, step, "alpha")
        eps_tree = self._get_param_tree(self.epss, fast_weights, step, "eps")
        # Retrieve previous moving average square buffer tree
        sq_avg_tree = state["square_avg"]

        # Inner mathematical update closure
        def update_fn(p, g, lr, alpha, eps, sq_avg):
            # Update exponential moving average of squared gradients
            sq_avg_next = alpha * sq_avg + (1 - alpha) * (g ** 2)
            # Update parameter tensor using RMSprop formula
            p_next = p - lr * g / (torch.sqrt(sq_avg_next) + eps)
            return p_next, sq_avg_next

        # Execute PyTree map across parameter structures
        results = pytree.tree_map(update_fn, fast_weights, grads_tree, lr_tree, alpha_tree, eps_tree, sq_avg_tree)

        # Unpack updated weights and state
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {"square_avg": pytree.tree_map(lambda x: x[1], results)}
        return new_weights, new_state


class InnerAdagrad(BaseInnerOptimizer):
    """
    Inner-loop Adagrad optimizer with per-layer/per-step hyper-parameters.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.01,
        eps: float = 1e-10,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerAdagrad with learnable learning rate and epsilon parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register epsilon meta-parameters
        self.epss = self._make_meta_param(eps, learn_params, "eps")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes accumulated squared gradients buffer state tensors with zeros."""
        return {"sum_squares": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)}

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional Adagrad update step."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for the current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        eps_tree = self._get_param_tree(self.epss, fast_weights, step, "eps")
        # Retrieve previous sum of squares buffer tree
        sum_sq_tree = state["sum_squares"]

        # Inner mathematical update closure
        def update_fn(p, g, lr, eps, sum_sq):
            # Accumulate squared gradients: sum_sq_next = sum_sq + g^2
            sum_sq_next = sum_sq + (g ** 2)
            # Apply Adagrad parameter update formula
            p_next = p - lr * g / (torch.sqrt(sum_sq_next) + eps)
            return p_next, sum_sq_next

        # Execute PyTree map operation
        results = pytree.tree_map(update_fn, fast_weights, grads_tree, lr_tree, eps_tree, sum_sq_tree)

        # Unpack updated parameters and new state
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {"sum_squares": pytree.tree_map(lambda x: x[1], results)}
        return new_weights, new_state


class InnerAdadelta(BaseInnerOptimizer):
    """
    Inner-loop Adadelta optimizer with per-layer/per-step hyper-parameters.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerAdadelta with learnable decay rate (rho) and scaling parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters (typically 1.0 for standard Adadelta)
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register decay factor (rho) meta-parameters
        self.rhos = self._make_meta_param(rho, learn_params, "rho")
        # Register epsilon meta-parameters
        self.epss = self._make_meta_param(eps, learn_params, "eps")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes squared gradient average and accumulated delta state buffers with zeros."""
        return {
            "sq_avg": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights),
            "acc_delta": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)
        }

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional Adadelta update step."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        rho_tree = self._get_param_tree(self.rhos, fast_weights, step, "rho")
        eps_tree = self._get_param_tree(self.epss, fast_weights, step, "eps")
        # Retrieve previous state buffer trees
        sq_avg_tree = state["sq_avg"]
        acc_delta_tree = state["acc_delta"]

        # Inner mathematical update closure
        def update_fn(p, g, lr, rho, eps, sq_avg, acc_delta):
            # Update running average of squared gradients
            sq_avg_next = rho * sq_avg + (1 - rho) * (g ** 2)
            # Compute standard deviation denominator
            std = torch.sqrt(sq_avg_next + eps)
            # Compute step delta tensor
            delta = (torch.sqrt(acc_delta + eps) / std) * g
            # Update running average of squared parameter updates
            acc_delta_next = rho * acc_delta + (1 - rho) * (delta ** 2)
            # Apply weight update: p_next = p - lr * delta
            p_next = p - lr * delta
            return p_next, sq_avg_next, acc_delta_next

        # Execute PyTree map
        results = pytree.tree_map(update_fn, fast_weights, grads_tree, lr_tree, rho_tree, eps_tree, sq_avg_tree, acc_delta_tree)

        # Unpack updated parameters and new state buffers
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {
            "sq_avg": pytree.tree_map(lambda x: x[1], results),
            "acc_delta": pytree.tree_map(lambda x: x[2], results)
        }
        return new_weights, new_state


class InnerAdam(BaseInnerOptimizer):
    """
    Inner-loop Adam optimizer with per-layer/per-step learnable hyper-parameters.
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerAdam with learnable beta1, beta2, and epsilon meta-parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register first moment decay factor (beta1) meta-parameters
        self.beta1s = self._make_meta_param(beta1, learn_params, "b1")
        # Register second moment decay factor (beta2) meta-parameters
        self.beta2s = self._make_meta_param(beta2, learn_params, "b2")
        # Register numerical stability factor (eps) meta-parameters
        self.epss = self._make_meta_param(eps, learn_params, "eps")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes first moment (m) and second moment (v) buffer state tensors with zeros."""
        return {
            "exp_avg": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights),
            "exp_avg_sq": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)
        }

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional Adam update step with bias correction."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for the current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        b1_tree = self._get_param_tree(self.beta1s, fast_weights, step, "b1")
        b2_tree = self._get_param_tree(self.beta2s, fast_weights, step, "b2")
        eps_tree = self._get_param_tree(self.epss, fast_weights, step, "eps")
        # Retrieve previous moment state buffer trees
        exp_avg_tree = state["exp_avg"]
        exp_avg_sq_tree = state["exp_avg_sq"]
        # Calculate bias correction step iteration integer (1-indexed)
        t = step + 1

        # Inner mathematical update closure
        def update_fn(p, g, lr, b1, b2, eps, m, v):
            # Update biased first moment estimate: m = b1 * m + (1 - b1) * g
            m_next = b1 * m + (1 - b1) * g
            # Update biased second raw moment estimate: v = b2 * v + (1 - b2) * g^2
            v_next = b2 * v + (1 - b2) * (g ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m_next / (1 - (b1 ** t))
            # Compute bias-corrected second raw moment estimate
            v_hat = v_next / (1 - (b2 ** t))
            
            # Apply Adam parameter update formula
            p_next = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
            return p_next, m_next, v_next

        # Execute PyTree map
        results = pytree.tree_map(
            update_fn, fast_weights, grads_tree, lr_tree, b1_tree, b2_tree, eps_tree, exp_avg_tree, exp_avg_sq_tree
        )

        # Unpack updated weights and state buffers
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {
            "exp_avg": pytree.tree_map(lambda x: x[1], results),
            "exp_avg_sq": pytree.tree_map(lambda x: x[2], results)
        }
        return new_weights, new_state


class InnerAdamax(BaseInnerOptimizer):
    """
    Inner-loop Adamax optimizer (variant of Adam based on the infinity norm).
    """
    def __init__(
        self, 
        initial_fast_weights: OrderedDict, 
        first_order: bool = False,
        inner_lr: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_inner_steps: int = 5, 
        per_layer: bool = False, 
        per_step: bool = False,
        learn_params: bool = True, 
        **kwargs
    ):
        """Initializes InnerAdamax with learnable beta1, beta2, and infinity norm parameters."""
        # Initialize parent BaseInnerOptimizer
        super().__init__(initial_fast_weights, first_order, max_inner_steps, per_layer, per_step, **kwargs)
        # Register learning rate meta-parameters
        self.lrs = self._make_meta_param(inner_lr, learn_params, "lr")
        # Register beta1 meta-parameters
        self.beta1s = self._make_meta_param(beta1, learn_params, "b1")
        # Register beta2 meta-parameters
        self.beta2s = self._make_meta_param(beta2, learn_params, "b2")
        # Register epsilon meta-parameters
        self.epss = self._make_meta_param(eps, learn_params, "eps")

    def init_state(self, fast_weights: Dict[str, torch.Tensor]) -> Dict:
        """Initializes first moment (m) and infinity norm (u) state buffer tensors with zeros."""
        return {
            "exp_avg": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights),
            "exp_inf": pytree.tree_map(lambda p: torch.zeros_like(p), fast_weights)
        }

    def forward(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        gradients: Dict[str, torch.Tensor], 
        state: Dict, 
        step: int, 
        training: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Executes a functional Adamax update step."""
        # Handle first-order gradient detachment
        should_detach = self.first_order or not training
        grads_tree = pytree.tree_map(lambda g: g.detach(), gradients) if should_detach else gradients
        # Retrieve hyper-parameter PyTrees for current step
        lr_tree = self._get_param_tree(self.lrs, fast_weights, step, "lr")
        b1_tree = self._get_param_tree(self.beta1s, fast_weights, step, "b1")
        b2_tree = self._get_param_tree(self.beta2s, fast_weights, step, "b2")
        eps_tree = self._get_param_tree(self.epss, fast_weights, step, "eps")
        # Retrieve previous state trees
        exp_avg_tree = state["exp_avg"]
        exp_inf_tree = state["exp_inf"]
        # Calculate bias correction step iteration index
        t = step + 1

        # Inner mathematical update closure
        def update_fn(p, g, lr, b1, b2, eps, m, u):
            # Update biased first moment estimate
            m_next = b1 * m + (1 - b1) * g
            # Update exponentially weighted infinity norm estimate
            u_next = torch.max(b2 * u, torch.abs(g))
            
            # Compute bias-corrected first moment estimate
            m_hat = m_next / (1 - (b1 ** t))
            # Apply Adamax update formula
            p_next = p - lr * m_hat / (u_next + eps)
            return p_next, m_next, u_next

        # Execute PyTree map
        results = pytree.tree_map(
            update_fn, fast_weights, grads_tree, lr_tree, b1_tree, b2_tree, eps_tree, exp_avg_tree, exp_inf_tree
        )

        # Unpack updated parameters and new state buffers
        new_weights = pytree.tree_map(lambda x: x[0], results)
        new_state = {
            "exp_avg": pytree.tree_map(lambda x: x[1], results),
            "exp_inf": pytree.tree_map(lambda x: x[2], results)
        }
        return new_weights, new_state