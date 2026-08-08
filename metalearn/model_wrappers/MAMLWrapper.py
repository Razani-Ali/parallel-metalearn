import torch
from torch import nn
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from metalearn.dataset.Scalers import BaseScaler

# ==============================================================================
# ACKNOWLEDGEMENT & CITATION:
# The concept of ANIL (Almost No Inner Loop), where inner-loop adaptation is 
# restricted solely to the task head parameters while keeping the backbone feature 
# extractor frozen during inner steps, is introduced by Raghu et al.:
#
# [1] A. Raghu, M. Raghu, S. Bengio, and O. Vinyals, "Rapid Learning or Feature 
#     Reuse? Towards Understanding the Effectiveness of MAML," in Int. Conf. 
#     Learn. Representations (ICLR), 2020. arXiv:1909.09157.
# ==============================================================================


class MAML_Model(nn.Module):
    """
    Model Wrapper for Functional Meta-Learning Execution (MAML / ANIL).

    Encapsulates a feature extractor backbone and a classification head. 
    Designed for functional forward passes via `torch.func.functional_call` 
    and supports flexible fast parameter extraction strategies.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module = None,
        scaler: Optional[BaseScaler] = None,
        drop_rate: float = 0.0,
        fast_weights_names: Optional[Union[List[str], str]] = None,
        **kwargs
    ):
        """
        Initializes the MAML Wrapper.

        Args:
            backbone (nn.Module): Feature extractor network.
            head (nn.Module): Task classifier head.
            scaler (Optional[nn.Module]): Data Scaler.
            drop_rate (float): Dropout probability applied before the head layer.
            fast_weights_names (Optional[Union[List[str], str]]): 
                - If None: Adapts all trainable parameters (standard MAML).
                - If "ANIL": Adapts only parameters in the `head` module.
                - If List[str]: Adapts specific parameters matching the listed names.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Store backbone network and classification head
        self.backbone = backbone
        self.head = head
        self.scaler = scaler
        self.drop_rate = drop_rate
        self.fast_weights = fast_weights_names

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Standard forward pass compliant with the unified functional loss signature.

        Args:
            x (torch.Tensor): Input feature/image batch tensor.
            **kwargs: Operational flags such as 'return_features' or 'training'.

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]: 
                Dictionary containing logits (and optionally features) or raw tensor output.
        """
        # 1. Forward pass through backbone feature extractor
        if self.scaler:
            x = self.scaler(x)
        features = self.backbone(x)

        # 2. Apply dropout regularizer prior to classification head
        dropout_training = kwargs.get('training', self.training)

        if features.dim() > 2:
            features_flat = features.flatten(start_dim=-1)
        else:
            features_flat = features

        features_flat = torch.nn.functional.dropout(
            features_flat, 
            p=self.drop_rate, 
            training=dropout_training
        )

        # 3. Compute final logits through head module
        logits = self.head(features_flat)

        # Return standardized output dictionary
        return {"logits": logits, "features": features}

    def get_fast_weights(self, **kwargs) -> OrderedDict:
        """
        Extracts parameter tensors designated for fast adaptation in the inner loop.

        Returns:
            OrderedDict: Dictionary of parameters to be updated by the inner optimizer.
        """
        # Case 1: ANIL strategy - Only adapt parameters belonging to the classifier head
        if isinstance(self.fast_weights, str) and self.fast_weights.upper() == "ANIL":
            return OrderedDict(
                (name, param)
                for name, param in self.named_parameters()
                if param.requires_grad and name.startswith("head.")
            )

        # Case 2: Specific whitelist of parameter names provided
        elif isinstance(self.fast_weights, list):
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


# 📝 Developer Note: Extending MAML for New Tasks (e.g., Semantic Segmentation)
# Architectural Flexibility Notice

# The refactored MAML pipeline decouples task-specific evaluation by delegating
# loss and metric computation entirely to loss_fn via a unified dictionary interface.

# If you wish to extend MAML to tasks beyond standard classification
# (such as Semantic Segmentation, Autoencoders/Reconstruction, or Multi-task Learning),
# follow these simple steps without modifying the core MAML class:

# Update Model Forward Output:

# In your model wrapper (or custom nn.Module), return all necessary task prediction tensors
# (e.g., features, decoder_output, segmentation_mask) inside the return dictionary:

# Python
# def forward(self, x, **kwargs):
#     features = self.backbone(x)
#     seg_mask = self.decoder(features)  # Decoder output for Segmentation

#     # Return dictionary with all needed outputs
#     return {
#         "logits": seg_mask,
#         "decoder_out": seg_mask,
#         "features": features
#     }
# Process Outputs in Custom BaseLoss:

# Implement a dedicated subclass of BaseLoss that retrieves the required keys from out_dict:

# Python
# class SegmentationLoss(BaseLoss):
#     def forward(self, out_dict, targets, model_states=None, **kwargs):
#         # Retrieve decoder output directly from the standardized dictionary
#         seg_preds = out_dict["decoder_out"]

#         # Compute task-specific loss (e.g., Dice Loss / CrossEntropy)
#         loss = self.criterion(seg_preds, targets)
#         metric = self.compute_mIoU(seg_preds, targets)

#         return loss, metric
# Key Takeaway: The MAML execution loop operates entirely on out_dict and targets.
# You do not need to touch MAML.py, step(), or vmap logic when introducing
# new network architectures or output heads.