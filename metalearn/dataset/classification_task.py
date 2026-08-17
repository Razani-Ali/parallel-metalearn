import random
import itertools
from typing import List, Tuple, Optional, Dict, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# 1. SYNCHRONIZED / POOLED FEW-SHOT SAMPLER (JOINT & DISJOINT SAMPLING)
# ==============================================================================

class FewShotSampler:
    """
    Unified Few-Shot Sampler for Meta-Learning Tasks.

    Performs simultaneous and strictly disjoint sampling of Support and Query sets 
    from a shared pool of class instances within each episode/task. Translates raw 
    string dataset labels to integer-indexed classification targets (0, 1, ..., N-1).
    Gracefully handles zero-shot support requests by returning empty tensors with 
    preserved feature dimensionality.
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
                                                to string category labels in Y_base.
        """
        # Store foundational feature tensor and target labels
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        
        # Build category-to-row-index mapping and calculate available instance counts
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
            # Convert label to string to ensure consistent hash mapping
            label_str = str(label)
            if label_str not in mapping:
                mapping[label_str] = []
            mapping[label_str].append(int(i))
        return mapping

    def _calculate_label_frequencies(self) -> Dict[str, int]:
        """
        Calculates available sample count per category.

        Returns:
            Dict[str, int]: Dictionary mapping string labels to total instance capacities.
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
        support_samples_per_class: Tuple[int, ...],
        query_samples_per_class: Tuple[int, ...]
    ) -> None:
        """
        Validates target class IDs and verifies that joint sample counts do not exceed capacity.

        Args:
            target_numeric_classes (Tuple[int, ...]): Requested target numeric IDs.
            support_samples_per_class (Tuple[int, ...]): Support sample counts per class ID.
            query_samples_per_class (Tuple[int, ...]): Query sample counts per class ID.
        """
        # Ensure dimensions match between class IDs and requested sample counts
        if not (len(target_numeric_classes) == len(support_samples_per_class) == len(query_samples_per_class)):
            raise ValueError("❌ Shape Mismatch: Length of target classes, support counts, and query counts must match!")

        # Validate existence and total joint capacity for each requested class
        for num_label, s_count, q_count in zip(target_numeric_classes, support_samples_per_class, query_samples_per_class):
            if num_label not in self.numeric_to_string:
                raise KeyError(f"❌ Map Error: Numeric class ID '{num_label}' is missing from numeric_to_string map!")
            
            string_name = str(self.numeric_to_string[num_label])
            if string_name not in self.indices_by_class:
                raise KeyError(f"❌ Label Error: Target class label '{string_name}' was not found in dataset!")
                
            available_count = self.label_frequencies[string_name]
            total_requested = s_count + q_count
            if available_count < total_requested:
                raise ValueError(
                    f"❌ Capacity Exceeded: Class '{string_name}' has {available_count} samples, "
                    f"which is less than the requested joint count of {total_requested} (Support: {s_count} + Query: {q_count})!"
                )

    def _sample_single_class(
        self, 
        num_label: int, 
        s_count: int,
        q_count: int
    ) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
        """
        Jointly samples disjoint Support and Query instances without replacement for a single class.

        Args:
            num_label (int): Target numeric class ID.
            s_count (int): Number of support instances to draw (can be 0).
            q_count (int): Number of query instances to draw.

        Returns:
            Tuple containing:
                - Support feature arrays
                - Support integer labels
                - Query feature arrays
                - Query integer labels
        """
        string_name = str(self.numeric_to_string[num_label])
        available_indices = self.indices_by_class[string_name]
        
        # Draw total required indices without replacement (guarantees intra-task disjointness)
        total_needed = s_count + q_count
        chosen_indices = random.sample(available_indices, total_needed)

        # Partition indices into distinct Support and Query subsets
        support_indices = chosen_indices[:s_count]
        query_indices = chosen_indices[s_count:total_needed]

        # Extract features and assign numeric target integer ID for support
        s_x = [self.X[idx] for idx in support_indices]
        s_y = [num_label] * s_count

        # Extract features and assign numeric target integer ID for query
        q_x = [self.X[idx] for idx in query_indices]
        q_y = [num_label] * q_count

        return s_x, s_y, q_x, q_y

    def _post_process_and_shuffle(
        self, 
        s_x: List[np.ndarray], 
        s_y: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates, shuffles, and formats sampled features and target labels into NumPy arrays.
        Safely returns empty arrays with exact feature dimensions if s_x is empty.

        Args:
            s_x (List[np.ndarray]): List of sampled feature matrices.
            s_y (List[int]): List of sampled integer class IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shuffled feature tensor (X) and 1D label array (Y).
        """
        if len(s_x) == 0:
            # Handle empty partition safely while preserving feature shape and datatypes
            feature_shape = self.X.shape[1:]
            return np.empty((0, *feature_shape), dtype=self.X.dtype), np.empty((0,), dtype=np.int64)

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
        support_samples_per_class: Tuple[int, ...],
        query_samples_per_class: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Orchestrates joint task sampling, outputting disjoint Support and Query batches.

        Args:
            target_numeric_classes (Tuple[int, ...]): Tuple of target class integer IDs (e.g., (0, 1, 2)).
            support_samples_per_class (Tuple[int, ...]): Support sample counts per ID (e.g., (5, 5, 5) or (0, 0, 0)).
            query_samples_per_class (Tuple[int, ...]): Query sample counts per ID (e.g., (4, 4, 4)).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - X_s_final: Support feature tensor [Total_Support_Samples, ...].
                - Y_s_final: Support label array [Total_Support_Samples].
                - X_q_final: Query feature tensor [Total_Query_Samples, ...].
                - Y_q_final: Query label array [Total_Query_Samples].
        """
        # Validate inputs against bounds and capacities
        self._validate_inputs(target_numeric_classes, support_samples_per_class, query_samples_per_class)

        all_s_x: List[np.ndarray] = []
        all_s_y: List[int] = []
        all_q_x: List[np.ndarray] = []
        all_q_y: List[int] = []

        # Sample disjoint support and query instances class by class
        for num_label, s_count, q_count in zip(target_numeric_classes, support_samples_per_class, query_samples_per_class):
            s_x_c, s_y_c, q_x_c, q_y_c = self._sample_single_class(num_label, s_count, q_count)
            all_s_x.extend(s_x_c)
            all_s_y.extend(s_y_c)
            all_q_x.extend(q_x_c)
            all_q_y.extend(q_y_c)

        # Post-process, shuffle, and format each subset independently
        X_s_final, Y_s_final = self._post_process_and_shuffle(all_s_x, all_s_y)
        X_q_final, Y_q_final = self._post_process_and_shuffle(all_q_x, all_q_y)

        return X_s_final, Y_s_final, X_q_final, Y_q_final


# ==============================================================================
# 2. SYNCHRONIZED META-TASK DATASET
# ==============================================================================

class MetaTaskDataset(Dataset):
    """
    A PyTorch Dataset that generates single Few-Shot Learning tasks using a single 
    synchronized sampler to ensure strictly disjoint support and query partitions.
    
    Fully supports 0-shot support evaluation with dimension-safe empty tensor outputs.
    Task batching across episodes is delegated to PyTorch's DataLoader via `batch_size`.
    """

    def __init__(
        self,
        sampler: Optional[FewShotSampler] = None,
        way: Union[int, Tuple[int, int]] = 5,
        support_shot: Union[int, Tuple[int, int]] = 1,
        query_shot: Union[int, Tuple[int, int]] = 15,
        max_classes: Optional[int] = None,
        tasks_pool: Optional[List[Tuple[int, ...]]] = None,
    ):
        """
        Initializes the MetaTaskDataset.

        Args:
            sampler (Optional[FewShotSampler]): Synchronized sampler managing the shared data pool.
            way (Union[int, Tuple[int, int]]): Number of classes per task (fixed int or (min, max) tuple).
            support_shot (Union[int, Tuple[int, int]]): Support samples per class (fixed or tuple, can be 0).
            query_shot (Union[int, Tuple[int, int]]): Query samples per class (fixed or tuple).
            max_classes (Optional[int]): Maximum available classes if tasks_pool is None.
            tasks_pool (Optional[List[Tuple[int, ...]]]): Predefined list of class ID tuples.
        """
        self.sampler = sampler
        self.tasks_pool = tasks_pool
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot
        self.imbalanced_shot = type(support_shot) == tuple or type(query_shot) == tuple

        # Set up available classes for random task generation if no predefined pool is provided
        if self.tasks_pool is None:
            if max_classes is None:
                raise ValueError("Either tasks_pool or max_classes must be provided.")
            self.available_classes = list(range(max_classes))
        else:
            self.available_classes = []

        # Determine upper bounds for static Tensor padding (critical for vmap compatibility)
        self.max_way = self.way[1] if isinstance(self.way, tuple) else int(self.way)
        self.max_s_shot = self.support_shot[1] if isinstance(self.support_shot, tuple) else int(self.support_shot)
        self.max_q_shot = self.query_shot[1] if isinstance(self.query_shot, tuple) else int(self.query_shot)

        self.max_support_samples = self.max_way * self.max_s_shot
        self.max_query_samples = self.max_way * self.max_q_shot

    def __len__(self) -> int:
        """
        Returns an arbitrarily large number to keep the DataLoader yielding continuous episodic tasks.
        """
        return int(416667)

    def _resolve_param_value(self, param: Union[int, Tuple[int, int]]) -> int:
        """
        Resolves a single parameter value that might be a fixed integer or a random range (min, max).
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
        and constructs a boolean mask tensor (True for valid, False for dummy).
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

            # Pad Y labels with -1
            y_dummy = torch.full((pad_count,), fill_value=-1, dtype=y_tensor.dtype)
            y_final = torch.cat([y_tensor, y_dummy], dim=0)

            # Construct binary mask (True for real data, False for padding)
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
        
        # Sample random combination from available classes
        return tuple(random.sample(self.available_classes, current_way))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generates a SINGLE few-shot task with strictly disjoint support and query partitions.
        Handles zero support samples cleanly with correct tensor dimensions.
        
        Returns:
            Tuple containing:
                - x_s (torch.Tensor): Support features tensor [Max_Support_Samples, ...]
                - y_s_dict (Dict[str, torch.Tensor]): Support target labels and optional mask
                - x_q (torch.Tensor): Query features tensor [Max_Query_Samples, ...]
                - y_q_dict (Dict[str, torch.Tensor]): Query target labels and optional mask
        """
        # Determine number of classes for this task
        current_way = self._resolve_param_value(self.way)
        target_classes = self._generate_task_classes(current_way)

        # Resolve exact support and query sample counts per class
        s_samples_per_class = self._resolve_shot_per_class(self.support_shot, current_way)
        q_samples_per_class = self._resolve_shot_per_class(self.query_shot, current_way)

        # Jointly sample disjoint support and query instances from the single unified sampler
        if self.sampler is None or sum(s_samples_per_class) == 0:
            # Zero support scenario: query only
            if self.sampler is not None:
                # Query samples with 0 support samples requested
                _, _, x_q_raw, y_q_raw = self.sampler.sample(
                    target_classes, 
                    (0,) * current_way, 
                    q_samples_per_class
                )
                feature_shape = self.sampler.X.shape[1:]
                feature_dtype = self.sampler.X.dtype
            else:
                raise ValueError("❌ Error: Sampler must not be None when query shots are requested.")

            # Create an empty support array preserving feature dimensionality
            x_s_raw = np.empty((0, *feature_shape), dtype=feature_dtype)
            y_s_raw = np.empty((0,), dtype=np.int64)

        else:
            x_s_raw, y_s_raw, x_q_raw, y_q_raw = self.sampler.sample(
                target_classes, 
                s_samples_per_class, 
                q_samples_per_class
            )

        # Pad and construct mask dictionaries
        x_s, y_s_dict = self._pad_and_mask(x_s_raw, y_s_raw, self.max_support_samples)
        x_q, y_q_dict = self._pad_and_mask(x_q_raw, y_q_raw, self.max_query_samples)

        return x_s, y_s_dict, x_q, y_q_dict

    def reset_rng(self, seed: int) -> None:
        """
        Resets random seeds across Python, NumPy, PyTorch, and the underlying sampler.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        if self.sampler is not None:
            self.sampler.reset_seed(seed)
