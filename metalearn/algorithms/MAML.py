from collections import OrderedDict
from typing import Dict, Optional, Tuple
import torch
from metalearn.loss.base import BaseLoss
from .BaseLearner import MetaOptimizer
from .MetaUtils import get_per_step_loss_weights
from metalearn.inner_optimizers.base import BaseInnerOptimizer
from torch.func import grad, vmap
from metalearn.model_wrappers.MAMLWrapper import MAML_Model
from metalearn.loss import LabelEncoder


# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The Multi-Step Loss (MSL) optimization strategy and per-step gradient loss 
# aggregation in this implementation are based on the MAML++ architecture:
#
# [1] A. Antoniou, H. Edwards, and A. Storkey, "How to train your MAML," 
#     in Int. Conf. Learn. Representations (ICLR), 2019. 
#     arXiv:1810.09502.
#
# MAML original paper:
# [2] C. Finn, P. Abbeel and S. Levine,
#       "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
#       2019, arXiv:1703.03400.
#
# MetaSGD can be implemented using same class (per layer learnable lr + only one task)
# [3] Z. Li, F. Zhou, F. Chen, and H. Li, "Meta-SGD: Learning to Learn 
#        Quickly for Few-Shot Learning," arXiv preprint arXiv:1707.09835, 2017.
#
# [4] ANIL (Almost No Inner Loop):
#     Raghu, A., Raghu, M., Bengio, S., & Vinyals, O. (2019).
#     "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML."
#     International Conference on Learning Representations (ICLR), 2020.
#     arXiv:1909.09157.
#
# [5] BOIL (Body Only Inner Loop):
#     Oh, J., Yoo, H., Kim, C., & Yun, S. Y. (2020).
#     "BOIL: Towards Representation Change for Few-shot Learning."
#     International Conference on Learning Representations (ICLR), 2021.
#     arXiv:2008.08882.
# ==============================================================================


class MAML(MetaOptimizer):
    """
    Model-Agnostic Meta-Learning (MAML) implementation with modular design.

    Features:
    - Decoupled deployment/inference adaptation logic (`adapt_and_update`).
    - Encapsulated functional forward passes and loss evaluation.
    - Support for multi-step loss weighting and custom inner optimizers.
    - Fully vectorized task processing via `torch.func.vmap`.
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
        inner_steps: int = 1,
        multi_step_loss: bool = False,
        chunk_size: int = 80,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initializes the MAML meta-optimizer module.

        Args:
            model (MAML_Model): The base neural network model.
            optimizer (torch.optim.Optimizer): The outer-loop optimizer (e.g., Adam).
            inner_optimizer (BaseInnerOptimizer): The task-level inner-loop optimizer.
            support_loss_fn (BaseLoss): Loss module used during support set adaptation.
            query_loss_fn (Optional[BaseLoss]): Loss module used for query evaluation (defaults to support_loss_fn).
            encoder (LabelEncoder): One-Hot Encoder for categorical labels
            inner_steps (int): Number of gradient adaptation steps in the inner loop.
            multi_step_loss (bool): Whether to compute weighted query losses across all inner steps.
            chunk_size (int): Chunk size for vmap memory management during parallel task execution.
            device (Optional[torch.device]): Target compute device (CPU or GPU).
        """
        super().__init__()
        # Store primary components and hyper-parameters
        self.model = model
        self.optimizer = optimizer
        self.inner_optimizer = inner_optimizer
        
        # Loss function setup (fallback to support loss if query loss is not specified)
        self.support_loss_fn = support_loss_fn
        self.query_loss_fn = query_loss_fn or support_loss_fn
        self.encoder = encoder
        
        self.num_inner_steps = inner_steps
        self.multi_step_loss = multi_step_loss
        self.chunk_size = chunk_size

        # Device assignment and module transfer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.inner_optimizer.to(self.device)
        self.support_loss_fn.to(self.device)
        if self.query_loss_fn is not self.support_loss_fn:
            self.query_loss_fn.to(self.device)

        # Register learnable inner-loop parameters (e.g., per-layer learning rates) to outer optimizer
        inner_params = list(self.inner_optimizer.inner_lr_parameters())
        if inner_params:
            self.optimizer.add_param_group({"params": [p for n, p in inner_params]})

        # Buffer for multi-step loss weights
        self.step_weights = None

        # Functional gradient calculation wrapper for inner loop steps
        self.inner_step_fn = grad(self._inner_loss_fn, argnums=0)

    def _inner_loss_fn(
        self, 
        fast_weights: Dict, 
        static_params: Dict, 
        buffers: Dict, 
        x_s: torch.Tensor, 
        y_s: torch.Tensor, 
        inner_step: int = 0,
        training: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes support loss for a single inner-step adaptation.
        
        Returns:
            torch.Tensor: Loss value.
        """
        # Combine current fast weights with static parameters and model buffers
        combined_params = {**fast_weights, **static_params, **buffers}
        
        # Calculate support loss using functional evaluation
        (loss, _), _= self.compute_loss(
            X=x_s, Y=y_s,
            model_states=combined_params,
            loss_module=self.support_loss_fn,
            inner_step=inner_step,
            training=training,
            **kwargs
        )
        return loss

    def compute_loss(
        self, 
        *, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        model_states: Dict[str, torch.Tensor], 
        loss_module: BaseLoss,
        inner_step: int = 0,
        training: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a functional forward pass and computes loss/metrics using the given loss module.

        Args:
            X (torch.Tensor): Input batch tensor.
            Y (torch.Tensor): Target labels tensor.
            model_states (Dict[str, torch.Tensor]): Complete state dict of model parameters and buffers.
            loss_module (BaseLoss): Loss function module to evaluate predictions.
            inner_step (int): current inner gradient step
            training (bool): whether if you train or validate a model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Calculated scalar loss and evaluation metric.
            Dict: Model forward returned output
        """
        # Isolate forwarding keyword arguments to prevent PyTorch functional_call argument errors
        forward_kwargs = {"training": training, "num_step": inner_step}
        
        # Perform functional execution of the neural network
        out_dict = torch.func.functional_call(self.model, model_states, (X,), forward_kwargs)
        
        # Evaluate loss and metric via encapsulated BaseLoss instance
        return loss_module(out_dict=out_dict, targets=Y, model_states=model_states, **kwargs), out_dict

    def _extract_model_states(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], OrderedDict]:
        """
        Extracts static parameters, buffers, and initial fast weights from the model.

        Returns:
            Tuple[Dict, Dict, OrderedDict]: (static_params, all_buffers, initial_fast_weights)
        """
        # Extract all named parameters and buffers as dictionaries
        all_params = dict(self.model.named_parameters())
        all_buffers = dict(self.model.named_buffers())
        
        # Identify trainable keys meant for fast adaptation
        trainable_keys = list(OrderedDict(self.model.get_fast_weights()).keys())
        
        # Separate static parameters from fast-adapting parameters
        static_params = {k: v for k, v in all_params.items() if k not in trainable_keys}
        initial_fast_weights = OrderedDict({k: all_params[k] for k in trainable_keys})
        
        return static_params, all_buffers, initial_fast_weights

    # =========================================================================
    # 1. Deployment / Inference Logic (Decoupled from Training Loop)
    # =========================================================================
    def adapt_and_update(self, Xsupport: torch.Tensor, Ysupport: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> None:
        """
        Adapts parameters and buffers on the provided support set and permanently updates `self.model`.
        Intended exclusively for test-time adaptation or deployment in MAML.

        Args:
            Xsupport (torch.Tensor): Support set inputs tensor of shape (num_tasks, K, ...).
            Ysupport (Union[torch.Tensor, Dict[str, torch.Tensor]]): Support set target labels or dictionary.
            **kwargs: Operational arguments passed down to loss modules and model forward pass.
        """
        # Helper function to recursively transfer tensors or dictionaries to target compute device
        def to_device(obj, device):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obj.items()}
            return obj

        # Transfer support dataset inputs and labels to the designated device
        Xsupport = to_device(Xsupport, self.device)
        Ysupport = to_device(Ysupport, self.device)

        # Extract current model state split into static parameters, persistent buffers, and initial fast weights
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        # Define single-task adaptation closure compatible with torch.func.vmap
        def _adapt_single_task(x_s, y_s, global_buffers):
            # Create an isolated clone of model buffers for this specific task to enable vmap parallelization
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            
            # Initialize task-specific fast weights dictionary
            fast_weights = OrderedDict(initial_fast_weights)
            
            # Initialize task-specific inner optimizer state
            opt_state = self.inner_optimizer.init_state(fast_weights)
            
            # Execute inner adaptation steps on support set
            for inner_step in range(self.num_inner_steps):
                # Compute support gradients w.r.t fast_weights using functional autograd
                grads = self.inner_step_fn(
                    fast_weights, static_params, task_buffers, 
                    x_s, y_s, inner_step=inner_step, training=True, **kwargs
                )
                # Update task-specific fast weights via the inner optimizer
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads, state=opt_state, step=inner_step
                )

            # Combine adapted fast weights with static parameters and updated task buffers
            combined = {**fast_weights, **static_params, **task_buffers}
            
            # Execute final functional pass to capture updated task buffers (e.g., BatchNorm, running_prototypes)
            _, out_dict = self.compute_loss(
                X=x_s, Y=y_s, model_states=combined,
                loss_module=self.support_loss_fn,
                inner_step=self.num_inner_steps,
                training=False, **kwargs
            )
            
            # Extract updated buffers dictionary from forward output, falling back to task_buffers if empty
            updated_task_buffers = out_dict.get("buffers", task_buffers)

            # Return adapted weights and updated task buffers for batch aggregation
            return fast_weights, updated_task_buffers

        # Vectorize single-task adaptation execution across the task batch dimension (dim=0 for inputs, None for global_buffers)
        vectorized_adapt = vmap(_adapt_single_task, in_dims=(0, 0, None), chunk_size=self.chunk_size)
        
        # Execute vectorized adaptation across all support tasks simultaneously
        fast_weights_batched, batched_buffers = vectorized_adapt(Xsupport, Ysupport, all_buffers)

        # Permanently copy adapted parameters and task-averaged buffers back to the base model
        with torch.no_grad():
            # Update trainable model parameters with mean adapted values across tasks
            for name, param in self.model.named_parameters():
                if name in fast_weights_batched:
                    param.copy_(fast_weights_batched[name].mean(dim=0))
            
            # Update persistent model buffers with mean updated values across tasks
            for name, buffer_tensor in self.model.named_buffers():
                if name in batched_buffers:
                    buffer_tensor.copy_(batched_buffers[name].mean(dim=0))

        print("✅ MAML: Model parameters and running buffers successfully adapted and updated.")

    # =========================================================================
    # 2. Main Training/Evaluation Step (Clean & Single Purpose)
    # =========================================================================
    def step(
        self, 
        task_itrator, 
        training: bool = True, 
        **kwargs
    ) -> Tuple[float, float]:
        """
        Executes a single meta-training or meta-validation step over a batch of tasks.

        Args:
            task_itrator: Iterable dataloader yielding (Xs, Ys, Xq, Yq) task batches.
            training (bool): If True, computes gradients and updates outer model parameters.

        Returns:
            Tuple[float, float]: Mean meta-loss and mean query evaluation metric.
        """
        # Initialize multi-step loss weights if enabled
        self._init_step_weights(self.num_inner_steps, training, **kwargs)

        # Retrieve next batch of tasks and transfer to device
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

        # Extract fixed/trainable model states
        static_params, all_buffers, initial_fast_weights = self._extract_model_states()

        def process_single_task(x_s, y_s, x_q, y_q, global_buffers):
            """Core execution loop for a single task within vmap."""
            task_buffers = {k: v.clone() for k, v in global_buffers.items()}
            fast_weights = OrderedDict(initial_fast_weights)
            opt_state = self.inner_optimizer.init_state(fast_weights)
            meta_loss = torch.tensor(0.0, device=self.device)

            if isinstance(y_s, dict) and "labels" in y_s and isinstance(y_q, dict) and "labels" in y_q and self.encoder is not None:
                y_s_enc, y_q_enc = self.encoder(y_s["labels"], y_q["labels"])
                
                y_s = {**y_s, "labels": y_s_enc}
                y_q = {**y_q, "labels": y_q_enc}

            # Execute inner adaptation steps
            for inner_step in range(self.num_inner_steps):
                # Calculate support gradients
                grads = self.inner_step_fn(
                    fast_weights, static_params, task_buffers,
                    x_s, y_s, inner_step=inner_step,
                    training=True, **kwargs
                )
                
                # Update task-specific fast weights via inner optimizer
                fast_weights, opt_state = self.inner_optimizer(
                    fast_weights=fast_weights, gradients=grads,
                    state=opt_state, step=inner_step
                )

                # Accumulate multi-step query loss for intermediate adaptation steps
                last_step = (inner_step == self.num_inner_steps - 1)
                if self.multi_step_loss and not last_step:
                    combined = {**fast_weights, **static_params, **task_buffers}
                    (q_step_loss, _), _ = self.compute_loss(
                        X=x_q, Y=y_q, model_states=combined,
                        loss_module=self.query_loss_fn, inner_step=inner_step,
                        training=True, **kwargs
                    )
                    meta_loss = self._update_meta_loss(meta_loss, q_step_loss, inner_step)

            # Final Query Loss & Metric Evaluation on adapted weights
            combined = {**fast_weights, **static_params, **task_buffers}
            (q_step_loss, q_metric), out_dict = self.compute_loss(
                X=x_q, Y=y_q, model_states=combined,
                loss_module=self.query_loss_fn,
                inner_step=self.num_inner_steps,
                training=training, **kwargs
            )
            
            # Safely index step weights (prevents out-of-bounds error if num_inner_steps is 0)
            target_step_idx = max(0, self.num_inner_steps - 1)
            meta_loss = self._update_meta_loss(meta_loss, q_step_loss, target_step_idx)

            updated_task_buffers = out_dict.get("buffers", task_buffers)

            return meta_loss, q_metric, updated_task_buffers

        # Parallelize single task processing across the task batch using vmap
        vectorized_processor = vmap(
            process_single_task, in_dims=(0, 0, 0, 0, None),
            randomness="different", chunk_size=self.chunk_size
        )

        # Process all tasks simultaneously
        meta_losses, metrics, batched_buffers = vectorized_processor(
            Xsupport, Ysupport, Xquery, Yquery, all_buffers
        )

        # Outer loop meta-optimization step
        if training:
            self.optimizer.zero_grad()
            meta_losses.mean().backward()
            self.optimizer.step()
            with torch.no_grad():
                for name, buffer_tensor in self.model.named_buffers():
                    if name in batched_buffers:
                        mean_buf = batched_buffers[name].mean(dim=0)
                        buffer_tensor.copy_(mean_buf)

        # Return scalar metric values
        return (
            meta_losses.mean().item(),
            metrics.mean().item(),
        )

    def _init_step_weights(self, num_inner_steps: int, training: bool, **kwargs) -> None:
        """
        Calculates per-step loss weights if multi-step loss is enabled during training.
        """
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
        """
        Accumulates query loss into current_meta_loss with step weighting if configured.

        Args:
            current_meta_loss (torch.Tensor): Accumulated scalar loss tensor.
            q_loss (torch.Tensor): Calculated query loss for the current step.
            inner_step (int): Current inner loop iteration index.

        Returns:
            torch.Tensor: Updated meta-loss tensor.
        """
        if self.step_weights is not None and self.multi_step_loss:
            return current_meta_loss + (q_loss / self.step_weights[inner_step])
        return current_meta_loss + q_loss