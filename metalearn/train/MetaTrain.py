import os
import time
import tempfile
import torch
from typing import Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
from .Utils import init_history, update_history, setup_logger, format_time
from metalearn.file_manager.FileHandle import remove_folder, safe_copy, replace_with_error
from tqdm.auto import tqdm


class MetaTrain:
    """
    Trainer engine for orchestrating Meta-Learning algorithms.

    Provides a decoupled, modular training pipeline supporting checkpointing,
    early stopping, dynamic validation schedules, and test-time adaptation.
    """

    def __init__(
        self,
        TrainLoader: Any,
        algorithm: Any,
        ValLoader: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ):
        """
        Initializes the MetaTrain engine.

        Args:
            TrainLoader (Any): Task dataloader for meta-training.
            algorithm (Any): Meta-learning algorithm instance (subclass of MetaOptimizer).
            ValLoader (Optional[Any]): Task dataloader for meta-validation.
            scheduler (Optional[Any]): Learning rate scheduler for outer optimizer.
        """
        # Store primary dataset loaders and algorithm components
        self.TrainIterator = iter(TrainLoader)
        self.ValIterator = iter(ValLoader)
        self.algorithm = algorithm
        self.model = algorithm.model
        self.optimizer = algorithm.optimizer
        
        # Extract optional inner optimizer if available (e.g., MAML inner optimizer)
        self.inner_optimizer = getattr(algorithm, 'inner_optimizer', None)
        self.scheduler = scheduler
        self.logger = None

    def _save_checkpoint(self, path: str, state_dict: Dict[str, Any]) -> None:
        """
        Saves the training state safely using a temporary file to prevent corruption.

        Args:
            path (str): Target destination path for the checkpoint file.
            state_dict (Dict[str, Any]): Dictionary containing model, optimizer, and history states.
        """
        # Write to temporary file first to ensure atomic write operation
        temp_path = path + ".temp"
        torch.save(state_dict, temp_path)
        # Safely replace destination file with temporary file
        replace_with_error(temp_path, path)

    def _load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Loads and applies saved state dictionary from a checkpoint file.

        Args:
            path (str): Path to the saved checkpoint file.

        Returns:
            Dict[str, Any]: Loaded checkpoint dictionary.
        """
        # Load state dictionary to CPU first
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        device = checkpoint['device']
        
        # Restore base model weights and transfer to device
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device)
        
        # Restore outer optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore inner optimizer state if present
        if self.inner_optimizer and checkpoint.get('inner_optimizer_state'):
            self.inner_optimizer.load_state_dict(checkpoint['inner_optimizer_state'])
            self.inner_optimizer.to(device)
            
        # Restore learning rate scheduler state if present
        if self.scheduler and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        return checkpoint

    def _log_epoch_info(
        self, 
        epoch: int, 
        total_epochs: int, 
        lr: float, 
        updates: Dict[str, Any], 
        time_taken: float, 
        evaluated_val: bool = False
    ) -> None:
        """
        Formats and logs training/validation metrics for the current epoch.

        Args:
            epoch (int): Current outer-loop epoch index.
            total_epochs (int): Total number of planned epochs.
            lr (float): Current learning rate of outer optimizer.
            updates (Dict[str, Any]): Metric values for the epoch.
            time_taken (float): Execution time of training step in seconds.
            evaluated_val (bool): Flag indicating if validation step was executed.
        """
        # Construct header and training summary string
        msg = f"{'#' * 60} Epoch {epoch + 1}/{total_epochs}:\n"
        msg += f"Current Learning Rate: {lr:.6f}\n"
        msg += f"TRAIN - Loss: {updates['train_loss']:.4f} | Metric: {updates['train_metric']:.4f} | Time: {format_time(time_taken)}\n"
        
        # Append validation metrics only if evaluation took place
        if evaluated_val and self.ValIterator:
            msg += f"VAL   - Loss: {updates['val_loss']:.4f} | Metric: {updates['val_metric']:.4f}\n"
            
        self.logger.info(msg)

    def train(
        self,
        epochs: int = 1500,
        check_idx: int = 10,
        log_checkpoint_path: str = "/content/drive/MyDrive/shared_folder/meta",
        temp_path: Optional[str] = None,
        replace_check_point: bool = False,
        patience: Optional[int] = None,
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        Executes the main meta-training loop.

        Args:
            epochs (int): Total number of meta-epochs to train.
            check_idx (int): Interval frequency for validation and checkpointing.
            log_checkpoint_path (str): Directory path to persist logs and checkpoints.
            temp_path (Optional[str]): Working directory for temporary file writes.
            replace_check_point (bool): If True, deletes existing checkpoint directory prior to training.
            patience (Optional[int]): Early stopping epoch threshold without validation improvement.
            **kwargs: Additional hyper-parameters passed to algorithm steps.

        Returns:
            Tuple[Dict[str, Any], float, float]: (training_history, best_val_metric, best_val_loss)
        """
        # --- 1. Setup & Directory Initialization ---
        erl_stp_tresh = kwargs.get("erl_stp_tresh", 1e-4)
        verbose = kwargs.get('verbose', True)
        
        # Set temporary working directory
        temp_path = temp_path or tempfile.gettempdir()
        if replace_check_point and log_checkpoint_path:
            remove_folder(log_checkpoint_path, force=True)

        if log_checkpoint_path:
            os.makedirs(log_checkpoint_path, exist_ok=True)
            
        # Define checkpoint file paths
        chkpt_path = os.path.join(log_checkpoint_path, "checkpoint.pth") if log_checkpoint_path else None
        tmp_chkpt_path = os.path.join(temp_path, "checkpoint.pth")
        
        # Configure logging system
        log_path = os.path.join(log_checkpoint_path, "log.txt") if log_checkpoint_path else None
        self.logger = setup_logger(os.path.join(temp_path, 'log.txt'), verbose=False)

        # Initialize tracking variables
        history = init_history()
        best_val_metric, best_val_loss = -float('inf'), float('inf')
        best_model_state, best_epoch = None, -1
        start_epoch, no_improve_count = 0, 0

        # --- 2. Load Checkpoint (Resume State) ---
        if chkpt_path and os.path.exists(chkpt_path) and not replace_check_point:
            self.logger.info(f"Loading checkpoint from {chkpt_path}")
            ckpt = self._load_checkpoint(chkpt_path)
            
            # Restore saved metrics and progress indices
            history = ckpt['history']
            best_val_metric = ckpt['best_val_metric']
            best_val_loss = ckpt['best_val_loss']
            best_model_state = ckpt['best_model_state']
            best_epoch = ckpt['best_epoch']
            no_improve_count = ckpt['no_improve_count']
            start_epoch = ckpt['last_epoch'] + 1

        # Check early stopping limit upon resume
        if patience and no_improve_count >= patience:
            start_epoch = epochs

        # Progress Bar
        last_val_loss_str = "N/A"
        last_val_acc_str = "N/A"
        pbar = tqdm(range(start_epoch, epochs), leave=True, desc="🚀 Meta-Training", dynamic_ncols=True)

        self.logger.info(f"{'+'*20} Starting Meta-Training Process... {'+'*20}\n")

        for epoch in pbar:
            # Record start time of pure training step
            t_start = time.time()

            # Execute outer training step across task batch
            train_loss, train_metric = self.algorithm.step(
                self.TrainIterator, training=True, epoch=epoch, epochs=epochs-1, **kwargs
            )

            # Calculate pure training latency (excluding evaluation overhead)
            t_end = time.time()
            epoch_time = t_end - t_start

            # Check if current epoch matches evaluation/checkpoint interval
            val_loss, val_metric = None, None
            needs_checkpoint = ((epoch + 1) % check_idx == 0) or (epoch == 0) or (epoch == epochs - 1)
            
            # Execute meta-validation step only at checkpoint steps
            if self.ValIterator and needs_checkpoint:
                val_loss, val_metric = self.algorithm.step(
                    self.ValIterator, training=False, **kwargs
                )
                last_val_loss_str = f"{val_loss:.4f}"
                last_val_acc_str = f"{val_metric * 100:.2f}%"
            
            # Fetch current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Aggregate updates and push to history
            updates = {
                'train_loss': train_loss, 
                'train_metric': train_metric,
                'val_loss': val_loss, 
                'val_metric': val_metric,
                'learning_rate': current_lr, 
                'epoch_elapsed_time': epoch_time,
                'total_elapsed_time': epoch_time  # Only includes pure training duration
            }
            history = update_history(history, updates)
            
            # Write metrics to log file/console
            self._log_epoch_info(
                epoch, epochs, current_lr, updates, epoch_time, 
                evaluated_val=(needs_checkpoint and self.ValIterator is not None)
            )

            # Update progress bar display
            pbar.set_postfix({
                'Tr-Loss': f"{train_loss:.4f}",
                'Tr-Acc': f"{train_metric * 100:.1f}%",
                'Val-Loss': last_val_loss_str,
                'Val-Acc': last_val_acc_str
            })

            # --- 4. Model Selection & Early Stopping ---
            if needs_checkpoint and self.ValIterator:
                # Track best validation metric state
                if val_metric >= best_val_metric:
                    best_val_metric = val_metric
                    best_epoch = epoch
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                # Track best validation loss and manage early stopping counter
                if val_loss <= best_val_loss - erl_stp_tresh:
                    best_val_loss = val_loss
                    no_improve_count = 0
                elif patience:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Step learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if needs_checkpoint and val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- 5. Save Checkpoint State ---
            if chkpt_path and needs_checkpoint:
                self._save_checkpoint(tmp_chkpt_path, {
                    'model_state': self.model.state_dict(),
                    'device': self.algorithm.device,
                    'optimizer_state': self.optimizer.state_dict(),
                    'inner_optimizer_state': self.inner_optimizer.state_dict() if self.inner_optimizer else None,
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                    'history': history,
                    'best_val_metric': best_val_metric,
                    'best_val_loss': best_val_loss,
                    'best_model_state': best_model_state,
                    'no_improve_count': no_improve_count,
                    'best_epoch': best_epoch,
                    'last_epoch': epoch
                })
                # Safely copy files to permanent log folder
                safe_copy(tmp_chkpt_path, chkpt_path)
                safe_copy(os.path.join(temp_path, 'log.txt'), log_path)

            # Clean GPU cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 6. Post-Training Cleanup & Best Weight Restoration ---
        pbar.close()
        if best_model_state and kwargs.get('load_best_model', True):
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"\nLoaded best model from epoch {best_epoch + 1} (Val Metric: {best_val_metric:.4f})")

        self._close_local_logger()
        return history, best_val_metric, best_val_loss

    def meta_test(
        self,
        Loader: Any,
        trials: int = 10,
        **kwargs: Any
    ) -> Tuple[float, float]:
        """
        Runs dedicated meta-testing across multiple random trial tasks.

        Args:
            Loader (Any): Task loader for evaluation tasks.
            trials (int): Number of independent trial iterations to evaluate.
            **kwargs: Additional parameters passed to step function.

        Returns:
            Tuple[float, float]: Average loss and metric values across evaluation trials.
        """

        total_loss, total_metric = 0.0, 0.0
        test_iter = iter(Loader)

        pbar = tqdm(range(trials), desc="🧪 Meta-Testing", leave=False, dynamic_ncols=True)

        for _ in pbar:
            # Run non-training evaluation step
            loss, metric = self.algorithm.step(Loader, training=False, **kwargs)
            total_loss += loss / trials
            total_metric += metric / trials
            
            pbar.set_postfix({'Avg-Loss': f"{total_loss:.4f}", 'Avg-Acc': f"{total_metric * 100:.2f}%"})

        pbar.close()
        return total_loss, total_metric

    def adapt2task(self, Loader: Any, **kwargs: Any) -> None:
        """
        Extracts ش task from Loader and adapts model parameters directly 
        using the algorithm's deployment method (`adapt_and_update`).

        Args:
            Loader (Any): Task loader containing support set samples.
            **kwargs: Additional context parameters for adaptation.
        """
        # Fetch initial batch (Support Set)
        Xs, Ys, _, _ = next(iter(Loader))
        
        # Permanently adapt the base neural network
        self.algorithm.adapt_and_update(
            Xsupport=Xs.to(self.algorithm.device),
            Ysupport=Ys.to(self.algorithm.device),
            **kwargs
        )

    def _close_local_logger(self) -> None:
        """Flushes and releases all active logging handlers to prevent resource leaks."""
        if hasattr(self, 'logger') and self.logger is not None:
            for handler in self.logger.handlers[:]:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)