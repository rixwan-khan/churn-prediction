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

RAW_DATA_DIR = DATA_DIR / "01_raw"
VALIDATED_DATA_DIR = DATA_DIR / "02_validated"
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"
FEATURED_DATA_DIR = DATA_DIR / "04_featured"
#FINAL_DATA_DIR = DATA_DIR / "05_final"

# ----------------------------------------
# Notebook Directory
# ----------------------------------------
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ----------------------------------------
# Logs Directories
# ----------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"

LOGS_VALIDATION_DIR = LOGS_DIR / "data"
LOGS_INFERENCE_DIR = LOGS_DIR / "inference"
LOGS_SYSTEM_DIR = LOGS_DIR / "system"
LOGS_TRAINING_DIR = LOGS_DIR / "training"

# ----------------------------------------
# Model Output Directories
# ----------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ----------------------------------------
# Convenience helper functions
# ----------------------------------------

def raw(filename: str):
    return RAW_DATA_DIR / filename

def validated(filename: str):
    return VALIDATED_DATA_DIR / filename

def processed(filename: str):
    return PROCESSED_DATA_DIR / filename

def featured(filename: str):
    return FEATURED_DATA_DIR / filename

#def final(filename: str):
#   return FINAL_DATA_DIR / filename

def logs(subdir: str, filename: str):
    mapping = {
        "data": LOGS_VALIDATION_DIR,
        "inference": LOGS_INFERENCE_DIR,
        "system": LOGS_SYSTEM_DIR,
        "training": LOGS_TRAINING_DIR,
    }
    return mapping.get(subdir, LOGS_DIR) / filename

def model(filename: str):
    return MODELS_DIR / filename

def report(filename: str):
    return REPORTS_DIR / filename