import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any


class LabelEncoder(nn.Module):
    """
    Class-Agnostic Label Encoder supporting Dictionary Targets and PyTorch vmap.
    
    If targets are passed as Dictionaries containing a 'labels' key:
    - Automatically encodes the 'labels' tensor across Support and Query sets consistently.
    - Preserves all other key-value pairs (e.g., domain IDs, regression targets) untouched.
    """

    def __init__(
        self, 
        num_classes: int, 
        max_n_way: Optional[int] = None, 
        shuffle: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_n_way = max_n_way
        self.shuffle = shuffle
        
        # Spectrum buffer: [0, 1, ..., num_classes - 1]
        self.register_buffer("class_spectrum", torch.arange(num_classes), persistent=False)

    def generate_mapping(self, y_support: torch.Tensor) -> torch.Tensor:
        """Generates mapping table strictly based on present classes in support labels."""
        device = y_support.device
        class_spectrum = self.class_spectrum.to(device)

        one_hot_supp = (y_support.unsqueeze(-1) == class_spectrum).to(torch.float32)
        flat_one_hot = one_hot_supp.reshape(-1, self.num_classes)
        class_presence = (flat_one_hot.sum(dim=0) > 0).to(torch.int64)

        class_ranks = torch.cumsum(class_presence, dim=0) - 1

        if self.shuffle:
            if self.max_n_way is not None:
                slot_scores = torch.rand(self.max_n_way, device=y_support.device, dtype=torch.float32)
                available_slots = torch.argsort(slot_scores)
                slots_mapped = available_slots[class_ranks.clamp(min=0)]
            else:
                slots_mapped = class_ranks

            shuffle_scores = torch.rand_like(class_spectrum, dtype=torch.float32)
            perm_indices = torch.argsort(shuffle_scores)
            
            shuffled_slots = slots_mapped[perm_indices]
            inv_perm = torch.argsort(perm_indices)
            mapping_table = shuffled_slots[inv_perm]
        else:
            mapping_table = class_ranks

        mapping_table = torch.where(
            class_presence > 0, 
            mapping_table, 
            torch.tensor(-1, device=y_support.device, dtype=torch.int64)
        )

        return mapping_table

    def apply_mapping(self, targets: torch.Tensor, mapping_table: torch.Tensor) -> torch.Tensor:
        """Applies pre-computed mapping table to target labels tensor."""
        device = targets.device
        class_spectrum = self.class_spectrum.to(device)

        one_hot = (targets.unsqueeze(-1) == class_spectrum).to(torch.float32)
        encoded_targets = torch.matmul(one_hot, mapping_table.to(one_hot.dtype)).to(torch.int64)

        return encoded_targets

    def _encode_tensor_pair(
        self, 
        y_supp_tensor: torch.Tensor, 
        y_query_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core encoding routine for support and query label tensors."""
        mapping_table = self.generate_mapping(y_supp_tensor)
        enc_supp = self.apply_mapping(y_supp_tensor, mapping_table)
        enc_query = self.apply_mapping(y_query_tensor, mapping_table)
        return enc_supp, enc_query

    def forward(
        self, 
        y_support: Union[torch.Tensor, Dict[str, Any]], 
        y_query: Union[torch.Tensor, Dict[str, Any]]
    ) -> Tuple[Union[torch.Tensor, Dict[str, Any]], Union[torch.Tensor, Dict[str, Any]]]:
        """
        Encodes Support and Query targets, automatically handling Dictionary target structures.
        """

        if isinstance(y_support, torch.Tensor) and isinstance(y_query, torch.Tensor):
            return self._encode_tensor_pair(y_support, y_query)

        else:
            raise TypeError("❌ Invalid Target Type: Expected torch.Tensor or Dict containing 'labels' key.")