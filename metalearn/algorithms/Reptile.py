from collections import OrderedDict
from typing import Dict, Optional, Tuple
import torch
from metalearn.loss.base import BaseLoss
from .BaseLearner import MetaOptimizer
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from torch.func import grad, vmap
from metalearn.model_wrappers.MAMLWrapper import MAML_Model
import torch.utils._pytree as pytree
from metalearn.loss import LabelEncoder


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The Reptile optimization strategy implemented here is based on:
#
# Nichol, A., Achiam, J., & Schulman, J. (2018).
# "On First-Order Meta-Learning Algorithms." 
# arXiv preprint arXiv:1803.02999.
# ==============================================================================


class Reptile(MetaOptimizer):
    """
    Reptile Meta-Learning Algorithm.

    A first-order meta-learning optimization algorithm. Instead of differentiating
    through the inner-loop optimization path like MAML, Reptile adapts parameters 
    using the inner optimizer and updates the global meta-parameters towards the 
    adapted task-specific weights using weight deltas (W_new - W_old).
    
    Features fully vectorized task processing via `torch.func.vmap`.
    """

    def __init__(
        self,
        *,
        model: MAML_Model,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: BaseInnerOptimizer,
        support_loss_fn: BaseLoss,
        query_loss_fn: Optional[BaseLoss] = None,
        encoder: LabelEncoder = None,
        inner_steps: int = 5,
        chunk_size: int = 8,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initializes the Reptile meta-optimizer module.

        Args:
            model (MAML_Model): The base neural network model.
            optimizer (torch.optim.Optimizer): The outer-loop optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): The task-level inner-loop optimizer.
            support_loss_fn (BaseLoss): Loss module used during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module used for query evaluation.
            encoder (LabelEncoder): One-Hot Encoder for categorical labels
            inner_steps (int): Number of gradient adaptation steps in the inner loop.
            chunk_size (int): Chunk size for vmap memory management.
            device (Optional[torch.device]): Target compute device.
        """
        super().__init__()
        # Store primary components
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer
        
        # Loss function setup
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn
        
        self.num_inner_steps = inner_steps
        self.chunk_size = chunk_size
        self.encoder = encoder

        # Device assignment and module transfer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner-loop parameters to outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for n, p in inner_params]})

        # Functional gradient calculation wrapper for inner loop steps
        # Reptile just needs first-order gradients for the inner updates
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0)

    def _extract_model_states(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """Extracts static parameters, buffers, and initial fast weights."""
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())
        
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())
        
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})
        
        return static_params, all_buffers, initial_fast_weights

    def _inner_loss_fn(
        self, 
        fast_weights: Dict[str, torch.Tensor], 
        static_params: Dict[str, torch.Tensor], 
        buffers: Dict[str, torch.Tensor], 
        x_s: torch.Tensor, 
        y_s: Dict[str, torch.Tensor], 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes support loss for a single inner-step adaptation."""
        combined_params = {**fast_weights, **static_params, **buffers}
        loss, _ = self.compute_loss(
            X=x_s, Y=y_s, model_states=combined_params, loss_module=self.support_loss_fn, **kwargs
        )
        return loss

    def compute_loss(
        self, 
        *, 
        X: torch.Tensor, 
        Y: Dict[str, torch.Tensor], 
        model_states: Dict[str, torch.Tensor], 
        loss_module: BaseLoss, 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs functional forward pass and computes evaluation metrics."""
        forward_kwargs = kwargs.get("kwargs_to_forward", {})
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)
        return loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs)

    # =========================================================================
    # 1. Main Training/Evaluation Step 
    # =========================================================================
    def step(
        self, 
        task_itrator, 
        training: bool = True, 
        **kwargs
    ) -> Tuple[float, float]:
        """
        Executes a single Reptile meta-training step.
        Adapts weights on the support set, computes weight deltas, and directly 
        injects them into the optimizer gradients.
        """
        Xs, Ys, Xq, Yq = next(task_itrator)

        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        Xsupport = to_device(Xs, self.device)
        Ysupport = to_device(Ys, self.device)
        Xquery = to_device(Xq, self.device)
        Yquery = to_device(Yq, self.device)

        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q):
            """Core execution loop for a single task within vmap."""
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)

            if isinstance(y_s, dict) and "labels" in y_s and isinstance(y_q, dict) and "labels" in y_q and self.encoder is not None:
                y_s_enc, y_q_enc = self.encoder(y_s["labels"], y_q["labels"])
                
                y_s = {**y_s, "labels": y_s_enc}
                y_q = {**y_q, "labels": y_q_enc}

            # 1. Execute inner adaptation steps (Standard SGD towards task manifold)
            for inner_step in range(self.num_inner_steps):
                grads = self.inner_step_fn(
                    fast_weights, static_params, all_buffers,
                    x_s, y_s, inner_step=inner_step,
                    training=True, **kwargs
                )
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads,
                    state=opt_state, step=inner_step
                )

            # 2. Compute Reptile Deltas: (W_new - W_old)
            # This directly replaces the computational graph backpropagation of MAML
            param_deltas = pytree.tree_map(lambda w_new, w_old: w_new - w_old, fast_weights, initial_fast_weights)

            # 3. Final Query Evaluation (For logging/metrics only)
            combined = {**fast_weights, **static_params, **all_buffers}
            q_loss, q_metric = self.compute_loss(
                X=x_q, Y=y_q, model_states=combined,
                loss_module=self.query_loss_fn,
                inner_step=self.num_inner_steps,
                training=training, **kwargs
            )

            return param_deltas, q_loss, q_metric

        # Parallelize single task processing across the task batch using vmap
        vectorized_processor = vmap(
            process_single_task, in_dims=(0, 0, 0, 0),
            randomness="different", chunk_size=self.chunk_size
        )

        # Retrieve vectorized weight deltas and query metrics
        vectorized_deltas, query_losses, query_metrics = vectorized_processor(
            Xsupport, Ysupport, Xquery, Yquery
        )

        # Outer loop meta-optimization step (First-Order Injection)
        if training:
            self.optimizer.zero_grad()
            
            for name, param in self.model.named_parameters():
                if name in vectorized_deltas:
                    # Average the deltas across the task batch (dim=0)
                    avg_delta = vectorized_deltas[name].mean(dim=0)
                    # Reptile Update Rule: Theta = Theta + lr * Delta
                    # Adam/SGD do: Theta = Theta - lr * grad
                    # Therefore, we set grad = -Delta
                    param.grad = -avg_delta.detach()
            
            self.optimizer.step()

        return query_losses.mean().item(), query_metrics.mean().item()

    # =========================================================================
    # 2. Deployment / Inference Logic
    # =========================================================================
    def adapt_and_update(self, Xsupport: torch.Tensor, Ysupport: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Adapts parameters on the provided support set and permanently updates `self.model`.
        Intended exclusively for test-time adaptation or deployment.
        """
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def _adapt_single_task(x_s, y_s):
            """Performs inner-loop adaptation for a single task."""
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            
            for inner_step in range(self.num_inner_steps):
                grads = self.inner_step_fn(
                    fast_weights, static_params, all_buffers, x_s, y_s, inner_step=inner_step, training=False, **kwargs
                )
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads, state=opt_state, step=inner_step
                )
            return fast_weights

        # Vectorize task adaptation across the task batch dimension
        vectorized_adapt = vmap(_adapt_single_task, in_dims=(0, 0), chunk_size=self.chunk_size)
        fast_weights_batched = vectorized_adapt(Xsupport.to(self.device), Ysupport.to(self.device))

        # Permanently copy adapted weights back to base model parameters (averaged across tasks)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in fast_weights_batched:
                    param.copy_(fast_weights_batched[name].mean(dim=0))