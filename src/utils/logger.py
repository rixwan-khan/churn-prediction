import logging
from pathlib import Path

# ------------------------------
# Logs folder
# ------------------------------
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # folder agar na ho to create ho jaye

def get_logger(log_filename: str):
    """
    Returns a reusable logger for any module.
    
    Parameters:
    - log_filename: Name of the log file (example: "preprocessing.log")
    
    Returns:
    - logger object
    """

    # Full path to log file
    log_path = LOG_DIR / log_filename
    
    # Create logger with module/file name
    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        # Console handler (optional, uncomment if needed)
        # console_handler = logging.StreamHandler()
        
        # Standardized format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)

    return logger