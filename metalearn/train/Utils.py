import os
import sys
import logging
import logging.handlers
from queue import Queue
from typing import Dict, Any, Optional


class RetryingFileHandler(logging.FileHandler):
    """
    Custom logging FileHandler that attempts stream reconnection if file access drops.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits log record while safely handling stream dropouts.

        Args:
            record (logging.LogRecord): The log record to process.
        """
        try:
            if self.stream is None:
                self.stream = self._open()
            super().emit(record)
        except (OSError, IOError):
            self.close()
            self.stream = None


def setup_logger(log_path: str, verbose: bool = False) -> logging.Logger:
    """
    Configures a thread-safe Queue-based logger with file and console output handlers.

    Args:
        log_path (str): Target file path for saving logs.
        verbose (bool): If True, outputs logs to standard stdout console.

    Returns:
        logging.Logger: Configured active logger instance.
    """
    logger = logging.getLogger("Training")
    logger.setLevel(logging.INFO)
    
    # Reset existing handlers if logger was previously instantiated
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define standard log format
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_queue = Queue(maxsize=-1)
    handlers = []

    # Configure file logging handler if log_path is specified
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f_handler = RetryingFileHandler(log_path, encoding='utf-8')
        f_handler.setFormatter(formatter)
        handlers.append(f_handler)

    # Configure console stdout handler if verbose mode is enabled
    if verbose or not log_path:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    else:
        logger.propagate = False

    # Start non-blocking QueueListener thread to process log messages safely
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    # Attach QueueHandler to logger
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    return logger


def init_history() -> Dict[str, Any]:
    """
    Instantiates an empty training metrics history dictionary.

    Returns:
        Dict[str, Any]: History structure with initialized list buffers.
    """
    return {
        'train_loss': [],
        'train_metric': [],
        'val_loss': [],
        'val_metric': [],
        'learning_rate': [],
        'epoch_elapsed_time': [],
        'total_elapsed_time': 0.0
    }


def update_history(history: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Appends new metric values to the history data structure.
    Strictly follows Single Responsibility Principle (pure data transformation).

    Args:
        history (Dict[str, Any]): Target history dictionary.
        updates (Dict[str, Any]): New key-value metric pairs to insert.

    Returns:
        Dict[str, Any]: Updated history dictionary.
    """
    for key, value in updates.items():
        # Ignore None values (e.g., omitted validation metrics)
        if value is None:
            continue
            
        # Append scalar metrics to list buffers
        if key in history and isinstance(history[key], list):
            history[key].append(value)
        # Accumulate total elapsed training duration
        elif key == 'total_elapsed_time':
            history[key] += value
            
    return history


def format_time(seconds: Optional[float]) -> str:
    """
    Converts time in seconds to a formatted, human-readable string.

    Args:
        seconds (Optional[float]): Time duration in seconds.

    Returns:
        str: Formatted time string (e.g., '12.30 ms', '45.20 s', '2 min 15 s').
    """
    if seconds is None:
        return ''
    if seconds < 1e-2:
        return f"{seconds*1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{int(seconds // 60)} min {int(seconds % 60)} s"
    
    return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60)} min"