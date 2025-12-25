# src/utils/paths.py

from pathlib import Path

# ----------------------------------------
# Project Root

# Define the root directory of the project, two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------
# Data Directories
# ----------------------------------------

# Base data directory
DATA_DIR = PROJECT_ROOT / "data"

# Subdirectories for different stages of data pipeline
RAW_DATA_DIR = DATA_DIR / "01_raw"              # original/raw data
VALIDATED_DATA_DIR = DATA_DIR / "02_validated"  # data after validation
PROCESSED_DATA_DIR = DATA_DIR / "03_processed"  # data after preprocessing
FEATURED_DATA_DIR = DATA_DIR / "04_featured"    # data with engineered features
#FINAL_DATA_DIR = DATA_DIR / "05_final"

# ----------------------------------------
# Notebook Directory
# ----------------------------------------
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"     # All jupyter noteboosk (.ipynb)

# ----------------------------------------
# Logs Directories
# ----------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"               # Base log directory

#  Subdirectories for different types of logs
LOGS_VALIDATION_DIR = LOGS_DIR / "data"        # logs for data validations
LOGS_INFERENCE_DIR = LOGS_DIR / "inference"    # logs for model inference
LOGS_SYSTEM_DIR = LOGS_DIR / "system"          # logs for system level events
LOGS_TRAINING_DIR = LOGS_DIR / "training"      # logs for model trainings

# ----------------------------------------
# Model Output Directories
# ----------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"           # All trained models stored here
REPORTS_DIR = PROJECT_ROOT / "reports"         # All reports, visualizations, or analysis outputs

# ----------------------------------------
# Convenience helper functions
# ----------------------------------------

# Functions to get full file paths in each directory
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
    """
    Return full path for a log file given a subdirectory type and filename.
    
    Args:
        subdir (str): Type of log ('data', 'inference', 'system', 'training')
        filename (str): Log filename
    
    Returns:
        Path: Full path to log file
    """
    mapping = {
        "data": LOGS_VALIDATION_DIR,
        "inference": LOGS_INFERENCE_DIR,
        "system": LOGS_SYSTEM_DIR,
        "training": LOGS_TRAINING_DIR,
    }
    # Default to base logs directory if subdir not found
    return mapping.get(subdir, LOGS_DIR) / filename

def model(filename: str):
    """Return full path for a model file in MODELS_DIR"""
    return MODELS_DIR / filename

def report(filename: str):
    """Return full path for a report file in REPORTS_DIR"""
    return REPORTS_DIR / filename