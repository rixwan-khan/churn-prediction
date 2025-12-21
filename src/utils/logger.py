# src/utils/logger.py

import logging
from pathlib import Path
from .paths import LOGS_DIR

def get_logger(log_filename: str, log_subdir: str):
    base_dir = LOGS_DIR / log_subdir
    base_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f"{log_subdir}:{log_filename}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_path = base_dir / log_filename
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(ch)

        logger.propagate = False

    return logger
