from collections import OrderedDict
from typing import Dict, Optional, Tuple
import torch
from metalearn.loss.base import BaseLoss
from .BaseLearner import MetaOptimizer
from .MetaUtils import get_per_step_loss_weights
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from torch.func import grad, vmap
from metalearn.model_wrappers.MAMLWrapper import MAML_Model


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
#
# [1] ProtoMAML:
#     Triantafillou, E., Zhu, T., Dumoulin, V., Lamblin, P., Evci, U., Xu, K.,
#     Goroshin, R., Gelada, C., Swersky, K., Manzagol, P. A., & Larochelle, H. (2019).
#     "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples."
#     International Conference on Learning Representations (ICLR), 2020.
#     arXiv:1903.03096.
#
# The Multi-Step Loss (MSL) optimization and per step learning rates:
# [2] A. Antoniou, H. Edwards, and A. Storkey, "How to train your MAML," 
#     in Int. Conf. Learn. Representations (ICLR), 2019. 
#     arXiv:1810.09502.
#
# MetaSGD (Learnable lr for each layer)
# [3] Z. Li, F. Zhou, F. Chen, and H. Li, "Meta-SGD: Learning to Learn 
#        Quickly for Few-Shot Learning," arXiv preprint arXiv:1707.09835, 2017.
#
# [4] BOIL (Body Only Inner Loop):
#     Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2020).
#     "BOIL: Towards Representation Change for Few-shot Learning."
#     International Conference on Learning Representations (ICLR), 2021.
#     arXiv:2008.08882.
# ==============================================================================


class ProtoMAMLv2(MetaOptimizer):
    """
    ProtoMAML (Version 2) - Pure Prototypical Inner Loop.

    In this version:
    1. The classification head is NEVER updated via Gradient Descent (BaseInnerOptimizer).
    2. Before EVERY loss calculation (Support or Query), the prototypes are dynamically 
       recalculated from the support set using the current backbone state.
    3. Gradients flow perfectly through the prototype calculation down to the backbone.
    """

    def __init__(
        self,
        *,
        model: MAML_Model,
        optimizer: torch.optim.Optimizer,
        inner_optimizer: BaseInnerOptimizer,
        support_loss_fn: BaseLoss,
        query_loss_fn: Optional[BaseLoss] = None,
        inner_steps: int = 1,
        multi_step_loss: bool = False,
        chunk_size: int = 80,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initializes the ProtoMAML_v2 meta-optimizer module.

        Args:
            model (MAML_Model): The ProtoMAML wrapper neural network model.
            optimizer (torch.optim.Optimizer): The outer-loop optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): The task-level inner-loop optimizer.
            support_loss_fn (BaseLoss): Loss module used during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module used for query evaluation.
            inner_steps (int): Number of gradient adaptation steps for the BACKBONE.
            multi_step_loss (bool): Whether to compute weighted query losses across all steps.
            chunk_size (int): Chunk size for vmap memory management.
            device (Optional[torch.device]): Target compute device.
        """
        # Call parent MetaOptimizer initialization
        super().__init__()
        
        # Store essential modules
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer
        
        # Configure loss functions
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn
        
        # Store optimization hyperparameters
        self.num_inner_steps = inner_steps
        self.multi_step_loss = multi_step_loss
        self.chunk_size = chunk_size

        # Resolve compute device and transfer modules
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner learning rates to the outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for n, p in inner_params]})

        # Initialize multi-step weight buffer
        self.step_weights = None

        # Wrap inner loss function with autograd to compute gradients w.r.t fast_weights
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0)

    def _extract_model_states(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """
        Extracts states while strictly filtering out classification head parameters.
        This guarantees the InnerSGD optimizer NEVER updates the head via gradients.

        Returns:
            Tuple[Dict, Dict, OrderedDict]: (static_params, all_buffers, initial_fast_weights)
        """
        # Retrieve all model components
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())
        
        # Extract trainable keys specified by the model wrapper
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())
        
        # FORCE FILTER: Remove any head weights from fast adaptation tracking
        trainable_keys = [k for k in trainable_keys if not k.startswith("head.")]
        
        # Separate static parameters and dynamically adapting weights
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})
        
        # Return properly segregated parameter dictionaries
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
        """
        Calculates support loss. Dynamically constructs prototype head weights 
        BEFORE calculating the loss to ensure complete autograd tracking.

        Args:
            fast_weights (Dict): Current backbone parameters.
            x_s (torch.Tensor): Support set features.
            y_s (Dict): Support set targets.

        Returns:
            torch.Tensor: Evaluated support loss.
        """
        # 1. DYNAMIC PROTOTYPE COMPUTATION
        # Calculate W and b from support set features using current backbone weights
        proto_weights = self.model.initialize_head_weights(x_s, y_s, fast_weights)

        # 2. Combine dynamically generated head with backbone and static components
        combined_params = {**proto_weights, **static_params, **buffers}
        
        # 3. Evaluate support loss through the newly generated parameters
        loss, _ = self.compute_loss(
            X=x_s, Y=y_s, model_states=combined_params, loss_module=self.support_loss_fn, **kwargs
        )
        
        # Return scalar loss tensor linked to the backbone via prototypes
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
        """
        Executes functional forward pass and computes metric evaluations.
        """
        # Extract isolated kwargs required for the forward pass
        forward_kwargs = kwargs.get("kwargs_to_forward", {})
        
        # Trigger stateless functional execution
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)
        
        # Return loss and performance metrics
        return loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs)

    # =========================================================================
    # Training Step
    # =========================================================================
    def step(
        self, 
        task_itrator, 
        training: bool = True, 
        **kwargs
    ) -> Tuple[float, float]:
        """
        Executes a complete vectorized meta-training step.
        """
        # Prepare multi-step loss scalars
        self._init_step_weights(self.num_inner_steps, training, **kwargs)

        # Fetch and unpack task batch
        Xs, Ys, Xq, Yq = next(task_itrator)

        # Utility function to migrate batch elements to compute device
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        # Transfer datasets to device
        Xsupport = to_device(Xs, self.device)
        Ysupport = to_device(Ys, self.device)
        Xquery = to_device(Xq, self.device)
        Yquery = to_device(Yq, self.device)

        # Retrieve static parameters and strictly backbone fast weights
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q):
            """Internal closure defining per-task execution logic."""
            # Initialize fast weights (Backbone ONLY)
            fast_weights = OrderedDict(initial_fast_weights)
            # Initialize inner optimizer states
            opt_state = self.inner_optimizer.init_state(fast_weights)
            # Initialize meta-loss accumulator
            meta_loss = torch.tensor(0.0, device=self.device)

            # INNER LOOP ADAPTATION (Backbone representation learning)
            for inner_step in range(self.num_inner_steps):
                # 1. Compute gradients for backbone. 
                # (Head is dynamically computed inside inner_step_fn before loss eval)
                grads = self.inner_step_fn(
                    fast_weights, static_params, all_buffers,
                    x_s, y_s, inner_step=inner_step,
                    training=True, **kwargs
                )
                
                # 2. Update backbone fast weights
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads,
                    state=opt_state, step=inner_step
                )

                # 3. Intermediate Query Evaluation for Multi-Step Loss
                last_step = (inner_step == self.num_inner_steps - 1)
                if self.multi_step_loss and not last_step:
                    # Dynamically calculate prototypes using the intermediately adapted backbone
                    proto_weights = self.model.initialize_head_weights(x_s, y_s, fast_weights)
                    combined = {**proto_weights, **static_params, **all_buffers}
                    
                    # Evaluate query loss
                    q_step_loss, _ = self.compute_loss(
                        X=x_q, Y=y_q, model_states=combined,
                        loss_module=self.query_loss_fn, inner_step=inner_step,
                        training=True, **kwargs
                    )
                    meta_loss = self._update_meta_loss(meta_loss, q_step_loss, inner_step)

            # FINAL QUERY EVALUATION
            # Recompute prototype head weights using the fully adapted backbone
            proto_weights = self.model.initialize_head_weights(x_s, y_s, fast_weights)
            combined = {**proto_weights, **static_params, **all_buffers}
            
            # Evaluate final meta-objective
            q_step_loss, q_metric = self.compute_loss(
                X=x_q, Y=y_q, model_states=combined,
                loss_module=self.query_loss_fn,
                inner_step=self.num_inner_steps,
                training=training, **kwargs
            )
            
            # Accumulate final loss
            target_step_idx = max(0, self.num_inner_steps - 1)
            meta_loss = self._update_meta_loss(meta_loss, q_step_loss, target_step_idx)

            # Return scalar loss and metric for task
            return meta_loss, q_metric

        # Vectorize execution over batch dimension
        vectorized_processor = vmap(
            process_single_task, in_dims=(0, 0, 0, 0),
            randomness="different", chunk_size=self.chunk_size
        )

        # Launch parallelized computation
        meta_losses, metrics = vectorized_processor(
            Xsupport, Ysupport, Xquery, Yquery
        )

        # Compute outer gradients and optimize meta-parameters
        if training:
            self.optimizer.zero_grad()
            meta_losses.mean().backward()
            self.optimizer.step()

        # Return mean values for logging
        return (
            meta_losses.mean().item(),
            metrics.mean().item(),
        )

    # =========================================================================
    # Deployment / Adaptation Step
    # =========================================================================
    def adapt_and_update(self, Xsupport: torch.Tensor, Ysupport: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Adapts backbone parameters to the support set and permanently binds 
        the computed prototypes as the final inference classification head.
        """
        # Retrieve strictly backbone initial parameters
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def _adapt_single_task(x_s, y_s):
            # Adapt Backbone via SGD
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            
            for inner_step in range(self.num_inner_steps):
                grads = self.inner_step_fn(
                    fast_weights, static_params, all_buffers, x_s, y_s, inner_step=inner_step, training=False, **kwargs
                )
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads, state=opt_state, step=inner_step
                )
            
            # After full backbone adaptation, compute final deterministic prototype head
            final_weights = self.model.initialize_head_weights(x_s, y_s, fast_weights)
            return final_weights

        # Vectorize adaptation 
        vectorized_adapt = vmap(_adapt_single_task, in_dims=(0, 0), chunk_size=self.chunk_size)
        fast_weights_batched = vectorized_adapt(Xsupport.to(self.device), Ysupport.to(self.device))

        # Permanently inject adapted backbone and computed head back to base model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in fast_weights_batched:
                    param.copy_(fast_weights_batched[name].mean(dim=0))

    def _init_step_weights(self, num_inner_steps: int, training: bool, **kwargs) -> None:
        """Calculates multi-step loss scalar weights."""
        if training and self.multi_step_loss and num_inner_steps > 0:
            self.step_weights = get_per_step_loss_weights(
                num_inner_steps, kwargs.get("epoch"), kwargs.get("epochs"), self.device
            )

    def _update_meta_loss(
        self,
        current_meta_loss: torch.Tensor,
        q_loss: torch.Tensor,
        inner_step: int,
    ) -> torch.Tensor:
        """Applies loss weighting."""
        if self.step_weights is not None and self.multi_step_loss:
            return current_meta_loss + (q_loss / self.step_weights[inner_step])
        return current_meta_loss + q_loss