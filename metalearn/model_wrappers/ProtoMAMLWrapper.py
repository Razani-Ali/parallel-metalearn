import torch
import torch.nn as nn
from typing import Dict, Optional, Union, List
from .prototype_calculator import SimplePrototype, BasePrototype
from collections import OrderedDict
from metalearn.dataset.Scalers import BaseScaler



class ProtoMAML_Model(nn.Module):
    """
    ProtoMAML Model Wrapper.
    
    Provides a standard forward pass compatible with `functional_call`. 
    Exposes a helper method to dynamically overwrite the initial classifier 
    weights (head.weight, head.bias) using class prototypes before the inner loop,
    fully preserving backpropagation gradients through support set prototypes.
    """

    def __init__(
        self, 
        backbone: nn.Module, 
        latent_dim: int,
        max_classes: int,
        scaler: Optional[BaseScaler] = None,
        drop_rate: float = 0.0,
        prototype_class: Optional[BasePrototype] = None,
        fast_weights_names: Union[str, List[str], None] = None,
        **kwargs,
    ):
        """
        Initializes the ProtoMAML_Model wrapper.

        Args:
            backbone (nn.Module): Feature extractor module outputting [batch, latent_dim].
            latent_dim (int): Dimensionality of extracted feature vectors.
            max_classes (int): Upper bound on target classes (ways).
            scaler (Optional[nn.Module]): Data Scaler.
            drop_rate (float): Dropout probability.
            prototype_class (Optional[BasePrototype]): custom prototype claculator
            fast_weights_names (Optional[Union[List[str], str]]): 
                - If None: Adapts all trainable parameters (standard).
                - If "BIOL": Adapts only parameters in the `backbone` module.
                - If List[str]: Adapts specific parameters matching the listed names.
            **kwargs: Additional keyword arguments.
        """
        # Initialize PyTorch parent module
        super().__init__()
        # Store foundational feature extraction backbone
        self.scaler = scaler
        self.backbone = backbone
        
        # Instantiate non-parametric class center calculator
        self.center_head = prototype_class if prototype_class else SimplePrototype(max_classes=max_classes, latent_dim=latent_dim)
        
        # Initialize standard linear head (overwritten dynamically during functional execution)
        self.head = nn.Linear(latent_dim, max_classes)
        # Initialize regularization dropout layer
        self.drop_rate = drop_rate

        # Initialize Fast Weights
        self.fast_weights = fast_weights_names

    def initialize_head_weights(
        self, 
        x_s: torch.Tensor, 
        y_s: Dict[str, torch.Tensor], 
        current_fast_weights: Dict[str, torch.Tensor],
        task_buffers: Dict[str, torch.Tensor] = None,
        inner_step: int = 0,
        training: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates ProtoMAML initialization for the classification head (W^(0), b^(0))
        directly from support set prototypes while maintaining full autograd history.
        Gradients backpropagate smoothly through prototypes into the backbone.

        Args:
            x_s (torch.Tensor): Support set input feature tensor.
            y_s (Dict[str, torch.Tensor]): Target dictionary containing 'labels' and optional 'samples_mask'.
            current_fast_weights (Dict[str, torch.Tensor]): Current fast weights dictionary.
            task_buffers (Optional[Dict[str, torch.Tensor]]): Current task state buffers dictionary.
            inner_step (int): current inner gradient step
            training (bool): whether if you train or validate a model

        Returns:
            Dict[str, torch.Tensor]: Updated fast weights dictionary with prototype-initialized head.
        """
        # 1. Isolate backbone parameters for isolated functional evaluation
        all_task_states = {**current_fast_weights, **(task_buffers or {})}
        backbone_states = {
            k.removeprefix("backbone."): v 
            for k, v in all_task_states.items() 
            if k.startswith("backbone.")
        }

        forward_kwargs = {"training": training, "num_step": inner_step}
        
        # 2. Extract support features using functional_call over current backbone weights
        if self.scaler:
            x_s = self.scaler(x_s)

        features = torch.func.functional_call(self.backbone, backbone_states, (x_s,), forward_kwargs)

        if features.dim() > 2:
            features_flat = features.flatten(start_dim=-1)
        else:
            features_flat = features

        # 3. Extract target labels and optional samples validity mask from target dict
        labels = y_s["labels"]
        samples_mask = y_s.get("samples_mask", None)

        # 4. Compute masked prototypes with full gradient tracking back to features
        centroids, mask, center_bufs = self.center_head.compute_class_centers(
            features=features_flat, 
            labels=labels, 
            samples_mask=samples_mask,
            task_buffers=task_buffers,
            prefix="center_head.",
        )

        task_buffers.update(center_bufs)
        # 5. Convert prototypes to linear classifier weights via ProtoMAML equations:
        # W_k = 2 * c_k
        w_init = 2.0 * centroids  # Shape: [max_classes, latent_dim]
        # b_k = - ||c_k||^2
        b_init = -torch.sum(centroids ** 2, dim=-1)  # Shape: [max_classes]

        # 6. Construct updated fast weights dictionary replacing head weights
        proto_weights = {**current_fast_weights}
        proto_weights["head.weight"] = w_init
        proto_weights["head.bias"] = b_init

        # Return updated fast weights containing prototype-based head parameters
        return proto_weights

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Standard parameter-free forward pass evaluating classification logits.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Operational flags such as 'inner_step' and 'training'.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing 'logits' and 'features'.
        """
        # Pass inputs through feature extraction backbone
        if self.scaler:
            x = self.scaler(x, **kwargs)
            
        features = self.backbone(x, **kwargs)
        dropout_training = kwargs.get('training', self.training)

        if features.dim() > 2:
            features_flat = features.flatten(start_dim=-1)
        else:
            features_flat = features

        if self.drop_rate > 0.0:
            features_flat = torch.nn.functional.dropout(
                features_flat, 
                p=self.drop_rate, 
                training=dropout_training
            )

        # Evaluate logits via classification linear head
        logits = self.head(features_flat)

        # Return formatted prediction dictionary
        current_buffers = dict(self.named_buffers())
        return {
            "logits": logits,
            "features": features,
            "buffers": current_buffers
        }

    def get_fast_weights(self, **kwargs) -> OrderedDict:
        """
        Extracts parameter tensors designated for fast adaptation in the inner loop.

        Returns:
            OrderedDict: Dictionary of parameters to be updated by the inner optimizer.
        """

        # Case 1: BOIL - Only adapt parameters belonging to the body or backbone
        if isinstance(self.fast_weights, str) and self.fast_weights.upper() == "BOIL":
            return OrderedDict(
                (name, param)
                for name, param in self.named_parameters()
                if param.requires_grad and name.startswith("backbone.")
            )

        # Case 2: Specific whitelist of parameter names provided
        if isinstance(self.fast_weights, list):
            return OrderedDict(
                (name, param)
                for name, param in self.named_parameters()
                if param.requires_grad and (
                    name in self.fast_weights or 
                    name.replace("backbone.", "").replace("head.", "") in self.fast_weights
                )
            )

        # Case 3: Standard MAML - Adapt all trainable parameters
        elif self.fast_weights is None:
            return OrderedDict(
                (name, param)
                for name, param in self.named_parameters()
                if param.requires_grad
            )

        # Raise exception for unrecognized fast parameter specifications
        else:
            raise ValueError(f"Unknown fast weights specification: {self.fast_weights}")