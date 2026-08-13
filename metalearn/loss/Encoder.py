import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any


class LabelEncoder(nn.Module):
    """
    Class-Agnostic Label Encoder supporting Dictionary Targets, samples_mask, Zero-Shot Fallback, and PyTorch vmap.

    This module re-indexes dataset-wide class integer IDs (e.g., 0 to num_classes-1) 
    into relative task-local class indices (0 to way-1).

    Key Features:
    - Dict & Tensor Inputs: Automatically detects 'labels' and 'samples_mask' keys inside target dictionaries.
    - samples_mask Integration: Ignores padded/dummy samples during class presence determination and forces their output label to -1.
    - Zero-Shot Fallback: If support set is empty (support_shot = 0), dynamically falls back to query set labels to construct mapping tables.
    - vmap Compatible: Uses pure tensor operations (torch.where, matmul, cumsum) avoiding Python control flow and .item() calls.
    """

    def __init__(
        self, 
        num_classes: int, 
        max_n_way: Optional[int] = None, 
        shuffle: bool = True
    ):
        """
        Initializes the LabelEncoder module.

        Args:
            num_classes (int): Total number of unique base classes across the dataset.
            max_n_way (Optional[int]): Maximum number of target slots/ways per task.
            shuffle (bool): If True, randomly permutes task class integer mappings.
        """
        # Call parent PyTorch module constructor
        super().__init__()
        # Store dataset total unique class count
        self.num_classes = num_classes
        # Store upper limit for task ways
        self.max_n_way = max_n_way
        # Store shuffling configuration boolean flag
        self.shuffle = shuffle
        
        # Register persistent spectrum buffer containing indices: [0, 1, ..., num_classes - 1]
        self.register_buffer("class_spectrum", torch.arange(num_classes), persistent=False)

    def generate_mapping(
        self, 
        y_support: torch.Tensor, 
        y_query: Optional[torch.Tensor] = None,
        supp_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates class mapping table based on support set classes, with automatic 
        fallback to query set classes during Zero-Shot evaluation (empty support set),
        while strictly ignoring padded samples flagged by samples_mask.

        Args:
            y_support (torch.Tensor): Raw support label tensor.
            y_query (Optional[torch.Tensor]): Raw query label tensor for Zero-Shot fallback.
            supp_mask (Optional[torch.Tensor]): Boolean/binary validity mask for support set.
            query_mask (Optional[torch.Tensor]): Boolean/binary validity mask for query set.

        Returns:
            torch.Tensor: Lookup table mapping dataset class IDs to task-local indices [0, way-1].
        """
        # Extract compute device from support label tensor
        device = y_support.device
        # Align class spectrum buffer with target device
        class_spectrum = self.class_spectrum.to(device)

        # 1. Compute class presence for support set via broadcasting
        one_hot_supp = (y_support.unsqueeze(-1) == class_spectrum).to(torch.float32)
        
        # Apply support validity mask if provided (zeroes out padded sample votes)
        if supp_mask is not None:
            # Cast mask to float and expand trailing dimension for elementwise multiplication
            mask_s_exp = supp_mask.to(torch.float32).unsqueeze(-1)
            # Filter out padded samples from one-hot encoding
            one_hot_supp = one_hot_supp * mask_s_exp

        # Flatten sample dimensions for dataset-wide class accumulation
        flat_supp = one_hot_supp.reshape(-1, self.num_classes)
        # Determine active presence boolean vector for support set
        supp_presence = (flat_supp.sum(dim=0) > 0).to(torch.int64)

        # 2. Compute class presence for query set (Zero-Shot Fallback path)
        if y_query is not None and y_query.numel() > 0:
            # Compute one-hot broadcast for query labels
            one_hot_query = (y_query.unsqueeze(-1) == class_spectrum).to(torch.float32)
            
            # Apply query validity mask if provided
            if query_mask is not None:
                # Cast mask to float and expand trailing dimension
                mask_q_exp = query_mask.to(torch.float32).unsqueeze(-1)
                # Filter out padded samples from query one-hot encoding
                one_hot_query = one_hot_query * mask_q_exp

            # Flatten sample dimensions for query class accumulation
            flat_query = one_hot_query.reshape(-1, self.num_classes)
            # Determine active presence boolean vector for query set
            query_presence = (flat_query.sum(dim=0) > 0).to(torch.int64)
        else:
            # Fallback zero presence vector when query tensor is unavailable
            query_presence = torch.zeros_like(supp_presence)

        # 3. Pure tensor conditional selection for Zero-Shot fallback (vmap safe)
        has_supp = (supp_presence.sum() > 0).to(torch.int64)
        # Use support presence if available; otherwise fallback to query presence
        class_presence = torch.where(has_supp > 0, supp_presence, query_presence)

        # Calculate 0-indexed relative task ranks for present classes
        class_ranks = torch.cumsum(class_presence, dim=0) - 1

        # Process random class slot permutation if shuffle is enabled
        if self.shuffle:
            if self.max_n_way is not None:
                # Generate uniform random scores for slot assignment
                slot_scores = torch.rand(self.max_n_way, device=device, dtype=torch.float32)
                # Sort indices based on random scores to draw permutation
                available_slots = torch.argsort(slot_scores)
                # Map cumulative class ranks to randomized slot indices
                slots_mapped = available_slots[class_ranks.clamp(min=0)]
            else:
                # Use standard sequential class ranks if max_n_way bound is absent
                slots_mapped = class_ranks

            # Generate random permutation scores across total class spectrum
            shuffle_scores = torch.rand_like(class_spectrum, dtype=torch.float32)
            # Obtain sorting indices for spectrum permutation
            perm_indices = torch.argsort(shuffle_scores)
            
            # Apply permutation mapping to slot allocations
            shuffled_slots = slots_mapped[perm_indices]
            # Calculate inverse permutation indices to restore spectrum order
            inv_perm = torch.argsort(perm_indices)
            # Construct final randomized lookup mapping table
            mapping_table = shuffled_slots[inv_perm]
        else:
            # Use deterministic cumulative ranks when shuffle is False
            mapping_table = class_ranks

        # Mask out absent/unseen classes with -1 index
        mapping_table = torch.where(
            class_presence > 0, 
            mapping_table, 
            torch.tensor(-1, device=device, dtype=torch.int64)
        )

        # Return computed lookup table linking dataset IDs to task-local indices
        return mapping_table

    def apply_mapping(
        self, 
        targets: torch.Tensor, 
        mapping_table: torch.Tensor,
        samples_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies pre-computed mapping table to raw target labels tensor,
        forcing padded/masked samples to -1.

        Args:
            targets (torch.Tensor): Raw target labels tensor.
            mapping_table (torch.Tensor): Pre-computed lookup mapping table.
            samples_mask (Optional[torch.Tensor]): Boolean/binary validity mask.

        Returns:
            torch.Tensor: Encoded target labels tensor mapped to task slot range [0, way-1].
        """
        # Retrieve compute device from target tensor
        device = targets.device
        # Align class spectrum buffer with compute device
        class_spectrum = self.class_spectrum.to(device)

        # Broadcast targets to one-hot representation over dataset class spectrum
        one_hot = (targets.unsqueeze(-1) == class_spectrum).to(torch.float32)
        # Multiply one-hot matrix by lookup mapping table to obtain re-indexed target IDs
        encoded_targets = torch.matmul(one_hot, mapping_table.to(one_hot.dtype)).to(torch.int64)

        # If validity mask is provided, force invalid/padded samples to -1 label
        if samples_mask is not None:
            # Convert mask to boolean tensor for logical indexing
            bool_mask = samples_mask.to(torch.bool)
            # Replace invalid padded locations with -1
            encoded_targets = torch.where(
                bool_mask, 
                encoded_targets, 
                torch.tensor(-1, device=device, dtype=torch.int64)
            )

        # Return re-mapped target tensor
        return encoded_targets

    def _encode_tensor_pair(
        self, 
        y_supp_tensor: torch.Tensor, 
        y_query_tensor: torch.Tensor,
        supp_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core encoding routine processing support and query label tensors.

        Args:
            y_supp_tensor (torch.Tensor): Raw support set labels tensor.
            y_query_tensor (torch.Tensor): Raw query set labels tensor.
            supp_mask (Optional[torch.Tensor]): Support set validity mask tensor.
            query_mask (Optional[torch.Tensor]): Query set validity mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded support and query set label tensors.
        """
        # Generate unified class mapping table with Zero-Shot fallback and mask filtering
        mapping_table = self.generate_mapping(
            y_support=y_supp_tensor, 
            y_query=y_query_tensor,
            supp_mask=supp_mask,
            query_mask=query_mask
        )
        
        # Apply lookup table mapping to support labels
        enc_supp = self.apply_mapping(y_supp_tensor, mapping_table, samples_mask=supp_mask)
        # Apply identical lookup table mapping to query labels
        enc_query = self.apply_mapping(y_query_tensor, mapping_table, samples_mask=query_mask)
        
        # Return mapped support and query label pair
        return enc_supp, enc_query

    def forward(
        self, 
        y_support: Union[torch.Tensor, Dict[str, Any]], 
        y_query: Union[torch.Tensor, Dict[str, Any]]
    ) -> Tuple[Union[torch.Tensor, Dict[str, Any]], Union[torch.Tensor, Dict[str, Any]]]:
        """
        Encodes Support and Query targets, handling Dictionary target structures and Tensors.

        Args:
            y_support (Union[torch.Tensor, Dict[str, Any]]): Support target tensor or dictionary.
            y_query (Union[torch.Tensor, Dict[str, Any]]): Query target tensor or dictionary.

        Returns:
            Tuple[Union[torch.Tensor, Dict[str, Any]], Union[torch.Tensor, Dict[str, Any]]]: Encoded targets.
        """
        # Scenario A: Target arguments are passed as Dictionaries containing 'labels'
        if isinstance(y_support, dict) and isinstance(y_query, dict):
            # Extract support label tensor
            y_supp_tensor = y_support["labels"]
            # Extract query label tensor
            y_query_tensor = y_query["labels"]
            
            # Extract optional support sample validity mask
            supp_mask = y_support.get("samples_mask", None)
            # Extract optional query sample validity mask
            query_mask = y_query.get("samples_mask", None)
            
            # Execute core encoding pair routine
            enc_supp, enc_query = self._encode_tensor_pair(
                y_supp_tensor=y_supp_tensor, 
                y_query_tensor=y_query_tensor,
                supp_mask=supp_mask,
                query_mask=query_mask
            )
            
            # Reconstruct support dictionary preserving all auxiliary keys (e.g. domain IDs)
            out_support = {**y_support, "labels": enc_supp}
            # Reconstruct query dictionary preserving all auxiliary keys
            out_query = {**y_query, "labels": enc_query}
            
            # Return updated dictionary pair
            return out_support, out_query

        # Scenario B: Target arguments are passed directly as PyTorch Tensors
        elif isinstance(y_support, torch.Tensor) and isinstance(y_query, torch.Tensor):
            # Execute core encoding pair routine without masks
            return self._encode_tensor_pair(y_support, y_query)

        # Scenario C: Unsupported target type
        else:
            # Raise exception for invalid type signature
            raise TypeError("❌ Invalid Target Type: Expected torch.Tensor or Dict containing 'labels' key.")