# src/data/run_preprocessing.py

import os
import pandas as pd

from src.validation.config import (
    RAW_DATA_PATH,
    VALIDATED_DATA_PATH,
    FAILED_ROWS_PATH,
    PROCESSED_DATA_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS
)
from src.validation.validate import validate_data
from src.validation.schema import ChurnSchema
from src.utils.io import load_csv, save_csv

# Ensure output directories exist
os.makedirs(VALIDATED_DATA_PATH.parent, exist_ok=True)
os.makedirs(FAILED_ROWS_PATH.parent, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH.parent, exist_ok=True)

def preprocess_data():
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Validate data using pandera schema
    3. Save validated and failed rows
    4. Save final cleaned dataset
    """
    print(f"Loading raw dataset from: {RAW_DATA_PATH}")
    df = load_csv(RAW_DATA_PATH)

    print("Validating dataset...")
    validated_df = validate_data(df, ChurnSchema)

    # Optionally, additional cleaning steps can go here
    # e.g., handling missing values, type conversion, feature engineering
    # For now, we assume validation already handles this

    print(f"Saving processed dataset to: {PROCESSED_DATA_PATH}")
    save_csv(validated_df, PROCESSED_DATA_PATH)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()
