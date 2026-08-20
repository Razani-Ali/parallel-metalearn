import torch
from typing import List, Tuple, Optional, Dict, Union, Any
import random
import numpy as np
from torch.utils.data import Dataset


class DecoupledFewShotSampler:
    """
    Generic and Standalone Few-Shot Sampler for Meta-Learning Tasks.

    Decouples raw string dataset labels from integer-indexed targets (0, 1, ..., N-1) 
    required by classification loss functions (e.g., CrossEntropyLoss). Operates 
    on arbitrary-rank NumPy arrays where the first axis indexes samples.
    """

    def __init__(
        self, 
        X_base: np.ndarray, 
        Y_base: np.ndarray, 
        numeric_to_string: Dict[int, str]
    ):
        """
        Initializes the FewShotSampler instance.

        Args:
            X_base (np.ndarray): Foundational feature tensor of shape [Total_Samples, ...].
            Y_base (np.ndarray): Master 1D array of original string (or object) labels.
            numeric_to_string (Dict[int, str]): Mapping translating numeric target IDs 
                                                to string category labels in Y_base 
                                                (e.g., {0: 'class_a', 1: 'class_b'}).
        """
        # Store foundational feature tensor and target labels
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        
        # Build category-to-row-index mapping and calculate class capacities
        self.indices_by_class = self._map_instances_to_classes()
        self.label_frequencies = self._calculate_label_frequencies()

    def _map_instances_to_classes(self) -> Dict[str, List[int]]:
        """
        Maps each unique string category label to its corresponding row indices.

        Returns:
            Dict[str, List[int]]: Dictionary mapping string labels to index lists.
        """
        mapping: Dict[str, List[int]] = {}
        for i, label in enumerate(self.Y):
            # Convert label to string to ensure safe hash mapping
            label_str = str(label)
            if label_str not in mapping:
                mapping[label_str] = []
            mapping[label_str].append(int(i))
        return mapping

    def _calculate_label_frequencies(self) -> Dict[str, int]:
        """
        Calculates available sample count per category.

        Returns:
            Dict[str, int]: Dictionary mapping string labels to sample capacities.
        """
        return {
            label: len(indices) for label, indices in self.indices_by_class.items()
        }

    def reset_seed(self, seed: int) -> None:
        """
        Resets the internal pseudo-random number generator state for reproducibility.

        Args:
            seed (int): Random seed integer.
        """
        random.seed(seed)

    def _validate_inputs(
        self, 
        target_numeric_classes: Tuple[int, ...], 
        samples_per_class: Tuple[int, ...]
    ) -> None:
        """
        Validates target class IDs and requested sample counts against database bounds.

        Args:
            target_numeric_classes (Tuple[int, ...]): Requested target numeric IDs.
            samples_per_class (Tuple[int, ...]): Requested sample count per class ID.
        """
        # Ensure dimensions match between class IDs and requested sample counts
        if len(target_numeric_classes) != len(samples_per_class):
            raise ValueError("❌ Shape Mismatch: Length of target classes must match samples per class!")

        # Validate existence and capacity for each requested class
        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            if num_label not in self.numeric_to_string:
                raise KeyError(f"❌ Map Error: Numeric class ID '{num_label}' is missing from numeric_to_string map!")
            
            string_name = str(self.numeric_to_string[num_label])
            if string_name not in self.indices_by_class:
                raise KeyError(f"❌ Label Error: Target class label '{string_name}' was not found in dataset!")
                
            available_count = self.label_frequencies[string_name]
            if available_count < required_samples:
                raise ValueError(
                    f"❌ Capacity Exceeded: Class '{string_name}' has {available_count} samples, "
                    f"which is less than the requested {required_samples} samples!"
                )

    def _sample_single_class(
        self, 
        num_label: int, 
        required_samples: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Samples a specified number of instances randomly for a single numeric class ID.

        Args:
            num_label (int): Target numeric class ID.
            required_samples (int): Number of instances to sample.

        Returns:
            Tuple[List[np.ndarray], List[int]]: Lists of sampled X arrays and integer Y labels.
        """
        string_name = str(self.numeric_to_string[num_label])
        available_indices = self.indices_by_class[string_name]
        
        # Draw random row indices without replacement
        chosen_indices = random.sample(available_indices, required_samples)

        # Extract features and assign numeric target integer ID
        sampled_x = [self.X[idx] for idx in chosen_indices]
        sampled_y = [num_label] * required_samples

        return sampled_x, sampled_y

    def _post_process_and_shuffle(
        self, 
        s_x: List[np.ndarray], 
        s_y: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates, shuffles, and formats sampled features and target labels.

        Args:
            s_x (List[np.ndarray]): List of sampled feature matrices.
            s_y (List[int]): List of sampled integer class IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shuffled feature tensor (X) and 1D label array (Y).
        """
        # Pair features and labels for synchronized shuffling
        combined = list(zip(s_x, s_y))
        random.shuffle(combined)

        # Stack into NumPy production arrays
        X_final = np.array([item[0] for item in combined])
        Y_final = np.array([item[1] for item in combined], dtype=np.int64)
        
        return X_final, Y_final

    def sample(
        self, 
        target_numeric_classes: Tuple[int, ...], 
        samples_per_class: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orchestrates the sampling pipeline to extract N-way K-shot task batches.

        Args:
            target_numeric_classes (Tuple[int, ...]): Tuple of target class integer IDs (e.g., (0, 1, 2)).
            samples_per_class (Tuple[int, ...]): Parallel tuple of sample counts per ID (e.g., (5, 5, 5)).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X_final: Task feature tensor of shape [Total_Sampled, ...].
                - Y_final: 1D NumPy int64 array of numeric target labels [Total_Sampled].
        """
        # Validate input parameters and dataset limits
        self._validate_inputs(target_numeric_classes, samples_per_class)

        all_sampled_x: List[np.ndarray] = []
        all_sampled_y: List[int] = []

        # Sample instances class by class
        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            x_c, y_c = self._sample_single_class(num_label, required_samples)
            all_sampled_x.extend(x_c)
            all_sampled_y.extend(y_c)

        # Shuffle and return formatted output arrays
        return self._post_process_and_shuffle(all_sampled_x, all_sampled_y)


# ==============================================================================
# DEVELOPER NOTE: EXTENDING FOR MULTITASK LEARNING (REGRESSION / AUXILIARY TARGETS)
# ==============================================================================
# To support Multi-Task Learning (e.g., joint Classification and Regression),
# you can easily inject an auxiliary targets array (e.g., continuous metrics,
# auxiliary labels) synchronized 1-to-1 with the first axis of X_base and Y_base.
#
# Implementation Steps:
# 1. Update __init__: Accept `reg_base: Optional[np.ndarray] = None` and store `self.reg = reg_base`.
# 2. Update `_sample_single_class`:
#    When indexing chosen samples via `chosen_indices`, slice `self.reg` directly:
#        if self.reg is not None:
#            sampled_reg.append(self.reg[idx])
# 3. Update `_post_process_and_shuffle`:
#    Include `sampled_reg` in `zip(s_x, s_y, s_reg)` before shuffling to guarantee
#    synchronous shuffling across all modalities, then format into numpy array:
#        Reg_final = np.array([item[2] for item in combined])
# 4. Update `sample`: Return `(X_final, Y_final, Reg_final)` tuple.
# ==============================================================================


class DecoupledMetaTaskDataset(Dataset):
    """
    A PyTorch Dataset that generates single Few-Shot Learning tasks.
    
    Task batching (stacking across the task dimension) is delegated 
    to PyTorch's DataLoader via the `batch_size` argument.
    """

    def __init__(
        self,
        support_sampler: Any,
        query_sampler: Any,
        way: Union[int, Tuple[int, int]] = 5,
        support_shot: Union[int, Tuple[int, int]] = 1,
        query_shot: Union[int, Tuple[int, int]] = 15,
        max_classes: Optional[int] = None,
        tasks_pool: Optional[List[Tuple[int, ...]]] = None,
        available_classes: Optional[List[int]] = None,
    ):
        """
        Initializes the MetaTaskDataset.

        Args:
            support_sampler (Any): Instance of FewShotSampler for support sets.
            query_sampler (Any): Instance of FewShotSampler for query sets.
            way (Union[int, Tuple[int, int]]): Number of classes per task (fixed int or (min, max) tuple).
            support_shot (Union[int, Tuple[int, int]]): Support samples per class (fixed or tuple).
            query_shot (Union[int, Tuple[int, int]]): Query samples per class (fixed or tuple).
            max_classes (Optional[int]): Maximum available classes if tasks_pool is None.
            tasks_pool (Optional[List[Tuple[int, ...]]]): Predefined list of class ID tuples.
            available_classes (Optional[List[int]]): List of Available Classes (Default: list(range(N Ways)))
        """
        self.support_sampler = support_sampler
        self.query_sampler = query_sampler
        self.tasks_pool = tasks_pool
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot
        self.imbalanced_shot = type(support_shot) == tuple or type(query_shot) == tuple

        # Set up available classes for random task generation if no predefined pool is provided
        # Set up available classes for random task generation if no predefined pool is provided
        if self.tasks_pool is not None:
            self.available_classes = []
        elif available_classes is not None:
            self.available_classes = list(available_classes)
        elif max_classes is not None:
            self.available_classes = list(range(max_classes))
        else:
            raise ValueError("Either tasks_pool, available_classes, or max_classes must be provided.")
        
        # Determine upper bounds for static Tensor padding (critical for vmap compatibility)
        self.max_way = self.way[1] if isinstance(self.way, tuple) else int(self.way)
        self.max_s_shot = self.support_shot[1] if isinstance(self.support_shot, tuple) else int(self.support_shot)
        self.max_q_shot = self.query_shot[1] if isinstance(self.query_shot, tuple) else int(self.query_shot)

        self.max_support_samples = self.max_way * self.max_s_shot
        self.max_query_samples = self.max_way * self.max_q_shot
        

    def __len__(self) -> int:
        """
        Returns an arbitrarily large number to keep the DataLoader yielding tasks.
        """
        return int(416667)

    def _resolve_param_value(self, param: Union[int, Tuple[int, int]]) -> int:
        """
        Resolves a single parameter value that might be a fixed int or a random range (min, max).
        """
        if isinstance(param, tuple) and len(param) == 2:
            return random.randint(param[0], param[1])
        return int(param)

    def _resolve_shot_per_class(
        self, 
        shot_param: Union[int, Tuple[int, int]], 
        num_classes: int
    ) -> Tuple[int, ...]:
        """
        Resolves shot counts for all target classes in a task.

        Args:
            shot_param: Fixed integer or (min, max) range tuple.
            num_classes (int): Number of target classes in the task.

        Returns:
            Tuple[int, ...]: A tuple containing sample counts for each class.
        """
        # Scenario A: Imbalanced sampling (each class draws its own random shot count from the range)
        if self.imbalanced_shot and isinstance(shot_param, tuple) and len(shot_param) == 2:
            return tuple(random.randint(shot_param[0], shot_param[1]) for _ in range(num_classes))

        # Scenario B: Balanced sampling (a single random or fixed shot count applied uniformly to all classes)
        single_shot_val = self._resolve_param_value(shot_param)
        return (single_shot_val,) * num_classes

    def _pad_and_mask(
        self, 
        x_arr: np.ndarray, 
        y_arr: np.ndarray, 
        target_max_samples: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Pads feature and label arrays to target_max_samples using dummy data 
        and constructs a boolean mask tensor (1.0 for valid, 0.0 for dummy).
        """
        num_real_samples = len(x_arr)
        
        # 1. Convert real sampled numpy arrays to PyTorch Tensors
        x_tensor = torch.from_numpy(x_arr)
        y_tensor = torch.from_numpy(y_arr)

        # 2. Check if padding is needed
        if self.imbalanced_shot:
            pad_count = target_max_samples - num_real_samples

            # Pad X with zeros along sample dimension
            x_pad_shape = (pad_count,) + x_tensor.shape[1:]
            x_dummy = torch.zeros(x_pad_shape, dtype=x_tensor.dtype)
            x_final = torch.cat([x_tensor, x_dummy], dim=0)

            # Pad Y labels with -1 (or 0)
            y_dummy = torch.full((pad_count,), fill_value=-1, dtype=y_tensor.dtype)
            y_final = torch.cat([y_tensor, y_dummy], dim=0)

            # Construct binary mask (1.0 for real data, 0.0 for padding)
            mask = torch.cat([
                torch.ones(num_real_samples, dtype=torch.bool),
                torch.zeros(pad_count, dtype=torch.bool)
            ], dim=0)

            y_dict = {
                "labels": y_final,
                "samples_mask": mask
            }

        else:
            x_final = x_tensor
            y_final = y_tensor

            y_dict = {
                "labels": y_final,
            }

        return x_final, y_dict

    def _generate_task_classes(self, current_way: int) -> Tuple[int, ...]:
        """
        Generates or selects a combination of class IDs for a single task.
        """
        if self.tasks_pool is not None:
            # Filter pool to match current way
            valid_tasks = [task for task in self.tasks_pool if len(task) == current_way]
            if not valid_tasks:
                raise ValueError(f"No tasks found in tasks_pool matching way={current_way}")
            return random.choice(valid_tasks)

        if len(self.available_classes) < current_way:
            raise ValueError(f"available_classes has {len(self.available_classes)} classes, but way={current_way}")

        # Sample random combination from available classes
        return tuple(random.sample(self.available_classes, current_way))


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a SINGLE task.
        
        Returns:
            Tuple of 4 tensors (x_support, y_support, x_query, y_query) for one task.
        """
        # Determine number of classes for this task
        current_way = self._resolve_param_value(self.way)
        target_classes = self._generate_task_classes(current_way)

        # Resolve exact support and query sample counts per class
        s_samples_per_class = self._resolve_shot_per_class(self.support_shot, current_way)
        q_samples_per_class = self._resolve_shot_per_class(self.query_shot, current_way)

        # Sample raw numpy instances
        if self.support_sampler is None or sum(s_samples_per_class) == 0:
            # Construct empty dummy support tensors matching signal shape
            sample_x, _ = self.query_sampler.sample(target_classes[:1], (1,))
            feature_shape = sample_x.shape[1:]
            
            x_s_raw = np.empty((0, *feature_shape), dtype=sample_x.dtype)
            y_s_raw = np.empty((0,), dtype=np.int64)

        else:
            x_s_raw, y_s_raw, *_ = self.support_sampler.sample(target_classes, s_samples_per_class)
            
        x_q_raw, y_q_raw, *_ = self.query_sampler.sample(target_classes, q_samples_per_class)

        # Pad and construct mask dictionaries
        x_s, y_s_dict = self._pad_and_mask(x_s_raw, y_s_raw, self.max_support_samples)
        x_q, y_q_dict = self._pad_and_mask(x_q_raw, y_q_raw, self.max_query_samples)

        return x_s, y_s_dict, x_q, y_q_dict

    def reset_rng(self, seed: int) -> None:
        """
        Resets random seeds across Python, NumPy, PyTorch, and internal samplers.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.support_sampler.reset_seed(seed)
        self.query_sampler.reset_seed(seed)


# ==============================================================================
# DEVELOPER NOTE: EXTENDING MetaTaskDataset FOR MULTITASK / SEGMENTATION
# ==============================================================================
# If your downstream task requires auxiliary target tensors synchronized with X and Y
# (e.g., Semantic Segmentation masks, bounding boxes, continuous target matrices),
# follow these 3 simple modifications in MetaTaskDataset:
#
# 1. Update `_sample_single_task`:
#    Unpack the extra metadata or auxiliary tensors returned by `FewShotSampler` 
#    instead of discarding them with `*_`:
#
#        # Before: x_s, y_s, *_ = self.support_sampler.sample(...)
#        # After:  x_s, y_s, mask_s = self.support_sampler.sample(...)
#        #         x_q, y_q, mask_q = self.query_sampler.sample(...)
#
# 2. Convert Auxiliary Outputs to PyTorch Tensors:
#    Convert the sampled auxiliary arrays alongside X and Y in `_sample_single_task`:
#
#        return (
#            torch.from_numpy(x_s),
#            torch.from_numpy(y_s),
#            torch.from_numpy(mask_s),  # Auxiliary Support Tensor (e.g., Segmentation Mask)
#            torch.from_numpy(x_q),
#            torch.from_numpy(y_q),
#            torch.from_numpy(mask_q)   # Auxiliary Query Tensor
#        )
#
# 3. Update `__getitem__` Return Signature:
#    Return the expanded tuple of 6 tensors:
#    `return self._sample_single_task(target_classes, s_samples_per_class, q_samples_per_class)`
#
# Note: PyTorch DataLoader will automatically handle batching (stacking across dim 0)
# for all 6 returned tensors without requiring any changes to batching logic.
# ==============================================================================



# ==============================================================================
# DEVELOPER NOTE: EXTENDING SAMPLER & DATASET FOR DOMAIN ADAPTATION (DA)
# ==============================================================================
# To pass domain labels/tensors directly to your DomainAdaptiveLoss:
#
# 1. FewShotSampler:
#    Accept an optional `domain_base: Optional[np.ndarray] = None` in __init__.
#    Include domain arrays in the synchronous `random.shuffle` step to preserve 
#    1-to-1 alignment with X and Y.
#
# 2. MetaTaskDataset:
#    Unpack domain outputs from both support_sampler and query_sampler:
#        x_s, y_s, d_s = self.support_sampler.sample(...)
#        x_q, y_q, d_q = self.query_sampler.sample(...)
#
#    Return 6 tensors from `__getitem__`:
#        return x_s, y_s, d_s, x_q, y_q, d_q
#
# 3. Do NOT modify MAML.py or InnerSGD.
#
# 4. Ensure your model wrapper returns feature maps in `out_dict["features"]`.
#
# 5. Create a custom `BaseLoss` subclass that computes both classification loss 
#    and domain alignment loss (e.g., MMD distance between source/target features):
#
#    total_loss = classification_loss + lambda_da * domain_alignment_loss
#
# 6. Pass this custom loss module as `query_loss_fn` or `support_loss_fn` to MAML.
# ==============================================================================
