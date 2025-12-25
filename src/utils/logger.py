# src/utils/logger.py

import logging
from pathlib import Path
from .paths import LOGS_DIR

def get_logger(log_filename: str, log_subdir: str):
    """
    Create and return a logger instance with file and console handlers.

    Args:
        log_filename (str): Name of the log file.
        log_subdir (str): Sub-directory inside LOGS_DIR to store log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    
    # Ensure the log subdirectory exists; create if not present
    base_dir = LOGS_DIR / log_subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create a unique logger name using subdir and filename
    logger_name = f"{log_subdir}:{log_filename}"
    logger = logging.getLogger(logger_name)

    # set default logging level
    logger.setLevel(logging.INFO)

    # Check if logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        # File handler: write logs to file in UTF-8 encoding
        file_path = base_dir / log_filename
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Log format
            "%Y-%m-%d %H:%M:%S" # Timestamp format
        ))
        logger.addHandler(fh)

        # Console handler: output logs to terminal for real-time monitoring
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

        # Prevent logs from propagating to the root logger
        logger.propagate = False

    return logger
