"""
Core Downloader Module

This module provides robust, concurrent file downloading, archiving, and copying utilities.
It includes mechanisms for thread-safe operations, atomic file replacements, and 
automatic retry logic for handling network instability or filesystem locks.
"""

import os
import shutil
from typing import Union
import time
from tqdm.auto import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil
import time
from pathlib import Path

def get_temp_path(path: str, suffix: str = "tmp") -> str:
    """
    Generates a temporary file path by appending a suffix before the file extension.
    
    Args:
        path (str): The original target file path.
        suffix (str, optional): The suffix to append. Defaults to "tmp".
        
    Returns:
        str: The newly constructed temporary file path.
    """
    # Split the original path into base name and extension
    base, ext = os.path.splitext(path)
    # Reconstruct the path with the suffix injected before the extension (if it exists)
    return f"{base}_{suffix}{ext}" if ext else f"{path.rstrip('/')}_{suffix}"

def remove_file(file_path: str, force: bool = False) -> bool:
    """
    Safely removes a file from the filesystem.
    
    Args:
        file_path (str): The path to the file to be deleted.
        force (bool, optional): If True, returns True even if the file didn't exist. Defaults to False.
        
    Returns:
        bool: True if the file was successfully removed (or if force=True), False otherwise.
    """
    try:
        # Check if the path exists and is strictly a file
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Attempt to delete the file
            os.remove(file_path)
            return True
        # Return the force flag if the file does not exist
        return force
    except Exception as e:
        # Catch and log any permission or OS errors during deletion
        print(f"⚠️ Warning: Failed to remove file {file_path}: {e}")
        return False

def replace_with_error(src: str, dest: str) -> bool:
    """
    Atomically replaces the destination file with the source file.
    
    Args:
        src (str): The path of the source file (usually a temporary file).
        dest (str): The path of the final destination file.
        
    Returns:
        bool: True if the replacement was successful, False otherwise.
    """
    try:
        # Perform an atomic replace operation (avoids corrupted partial files)
        os.replace(src, dest)
        return True
    except Exception as e:
        # Catch and log errors, such as file locks or permission issues
        print(f"❌ Error replacing {src} with {dest}: {e}")
        return False

def remove_folder(folder_path: Union[str, os.PathLike], force: bool = False) -> bool:
    """
    Safely and recursively removes a directory tree from the filesystem.

    Provides a clean execution wrapper around standard directory tree removal tools.
    It integrates explicit verification checks for pathway existence and node type
    descriptors before launching deletion, isolating permission violations or 
    generic file system exceptions to guarantee system runtime stability.

    Args:
        folder_path (Union[str, os.PathLike]): The pathway locating the target directory 
            tree slated for recursive deletion.
        force (bool, optional): If True, suppresses missing directory exceptions and 
            returns True even if the target folder does not exist. Defaults to False.

    Returns:
        bool: True if the directory tree was successfully deleted (or skipped via force=True),
            False if an OS error, missing permission descriptor, or lock blocked completion.

    Raises:
        FileNotFoundError: If the target path is absent and force is evaluated as False.
        NotADirectoryError: If the designated pathway targets a file descriptor rather 
            than a directory layout container.
    """
    try:
        # Check if the target pathway physically exists on the filesystem disk
        if not os.path.exists(folder_path):
            # If the path is missing but the force flag is active, bypass execution with success
            if force:
                return True
            # Raise an explicit exception if the folder is absent and force is deactivated
            raise FileNotFoundError(f"❌ could not find folder '{folder_path}'")
        
        # Verify that the existing node represents a structural directory, not a generic file link
        if not os.path.isdir(folder_path):
            # Abort operation with a explicit type exception if a file collision occurs
            raise NotADirectoryError(f"🚫 directory '{folder_path}' is not a folder")
        
        # Concurrently clean and recursively purge the entire directory hierarchy layout tree
        shutil.rmtree(folder_path)
        # Return success confirmation after directory tree is wiped
        return True
    
    except PermissionError as e:
        # Intercept, catch, and log access blockages or administrative filesystem privileges
        print(f"🔒 permission denied, error: {e}")
    except Exception as e:
        # Catch, log, and isolate unknown system exceptions to protect application lifecycle
        print(f"⚠️ unknown error: {e}")
        
    # Return failure if any operational exception blockages interrupt execution flow
    return False

"""
Core Downloader & File System Operations Module

This module provides robust, single-responsibility utilities for parallel file 
compressions, thread-safe buffering transfers, and dynamic progress bar management.
"""

# Global locks registry for thread-safe operations
_locks = {}
_master_lock = threading.Lock()

def get_file_lock(file_path: str) -> threading.Lock:
    """Retrieves or creates a thread lock specific to a file/folder path."""
    with _master_lock:
        if file_path not in _locks:
            _locks[file_path] = threading.Lock()
        return _locks[file_path]

def _get_total_bytes(path: Path) -> int:
    """Calculates total payload size of a target file or a directory layout."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


def _stream_file_buffered(p_src: Path, p_dst: Path, chunk_size: int, bar: tqdm):
    """Worker sub-task: Streams a single file using optimized chunk buffers."""
    p_dst.parent.mkdir(parents=True, exist_ok=True)
    temp_dst = p_dst.with_suffix('.tmp')
    
    with open(p_src, 'rb') as fsrc:
        with open(temp_dst, 'wb') as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                bar.update(len(buf))

    shutil.copystat(str(p_src), temp_dst)
    # Reusing your existing atomic file replacement mechanism
    os.replace(temp_dst, p_dst)


def _parallel_dir_copy_engine(src_path: Path, dst_path: Path, max_workers: int, bar: tqdm):
    """Worker sub-task: Copies directory tree structures concurrently across threads."""
    for dirpath, _, _ in os.walk(src_path):
        rel_dir = Path(dirpath).relative_to(src_path)
        (dst_path / rel_dir).mkdir(parents=True, exist_ok=True)
        
    all_files = [Path(dirpath) / f for dirpath, _, filenames in os.walk(src_path) for f in filenames]
    progress_lock = threading.Lock()
            
    def _copy_worker(file_src_path: Path):
        rel_file = file_src_path.relative_to(src_path)
        file_dst_path = dst_path / rel_file
        f_size = file_src_path.stat().st_size
        shutil.copy2(file_src_path, file_dst_path)
        with progress_lock:
            bar.update(f_size)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_copy_worker, all_files)


def safe_copy(src, dst, max_retries=7, chunk_size=128*1024*1024,
              force_sync=False, max_workers=8):
    """
    A simplified single-responsibility core transfer engine.
    Handles thread-safe buffered file streaming OR multi-threaded directory mirroring.

    Args:
        src (str/Path): The source file or directory path.
        dst (str/Path): The destination file or directory path.
        max_retries (int, optional): Number of retry attempts on failure. Defaults to 7.
        chunk_size (int, optional): Size of the read/write buffer in bytes. Defaults to 128MB.
        force_sync (bool, optional): If True, blocks execution until the hardware cache is flushed. Defaults to False.
        max_workers (int, optional): Thread pool size for parallel directory copies. Defaults to 8.
    """
    file_specific_lock = get_file_lock(str(dst))

    def save_it(src_path, dst_path):
        with file_specific_lock:
            p_src = Path(src_path)
            p_dst = Path(dst_path)
            
            is_directory = p_src.is_dir()
            total_bytes = _get_total_bytes(p_src)
            desc_msg = "🗂️ Syncing Directory Layout" if is_directory else "📄 Syncing File"

            for i in range(max_retries):
                try:
                    with tqdm(total=total_bytes, unit='B', unit_scale=True, 
                              unit_divisor=1024, desc=desc_msg, leave=False) as bar:
                        
                        if is_directory:
                            if p_dst.exists():
                                shutil.rmtree(p_dst)
                            _parallel_dir_copy_engine(p_src, p_dst, max_workers, bar)
                        else:
                            _stream_file_buffered(p_src, p_dst, chunk_size, bar)
                            
                    break
                
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"\n⚠️ [I/O Exception Intercepted] Stream broken. Retrying in 10s... ⏳")
                        time.sleep(10)
                    else:
                        print(f"\n💀 Fatal error: Failed to complete copy sequence after {max_retries} attempts: {e}")

    thread = threading.Thread(target=save_it, args=(str(src), str(dst)))
    thread.daemon = True
    thread.start()
    
    if force_sync:
        thread.join()
        if hasattr(os, 'sync'):
            os.sync()
    else:
        pass

    return thread

