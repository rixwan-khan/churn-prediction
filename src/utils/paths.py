# src/utils/paths.py

from pathlib import Path

# ----------------------------------------
# Project Root
# ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------
# Data Directories
# ----------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEAN_DATA_DIR = DATA_DIR / "clean" # preprocessed file

# ----------------------------------------
# Notebook Directory
# ----------------------------------------
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ----------------------------------------
# Model Output Directories
# ----------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ----------------------------------------
# Convenience Functions
# ----------------------------------------
def raw(path: str):
    """Return full path inside data/raw/"""
    return RAW_DATA_DIR / path


def processed(path: str):
    """Return full path inside data/processed/"""
    return PROCESSED_DATA_DIR / path


def clean(path: str):
    """Return full path inside data/clean/"""
    return CLEAN_DATA_DIR / path


def model(path: str):
    """Return full path inside models/"""
    return MODELS_DIR / path


def report(path: str):
    """Return full path inside reports/"""
    return REPORTS_DIR / path

