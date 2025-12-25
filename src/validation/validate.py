# src/validation/validate.py

import os
import warnings
import pandas as pd
from pandera.errors import SchemaErrors

from src.validation.schema import ChurnSchema
from src.validation.config import RAW_DATA_PATH, VALIDATED_DATA_PATH, FAILED_ROWS_PATH
from src.utils.io import load_csv, save_csv
from src.utils.paths import LOGS_VALIDATION_DIR

warnings.simplefilter(action="ignore", category=FutureWarning)


def validate_data(df, schema):

    # Ensure directories exist for saving validated and failed data
    os.makedirs(VALIDATED_DATA_PATH.parent, exist_ok=True)
    os.makedirs(FAILED_ROWS_PATH.parent, exist_ok=True)

    try:
        # Validate dataframe against schema (lazy=True allows reporting all errors at once)
        validated_df = schema.validate(df, lazy=True)
        print("✓ All rows passed validation.")

        # Save all rows as validated and empty dataframe for failed rows
        save_csv(validated_df, VALIDATED_DATA_PATH)
        save_csv(pd.DataFrame(), FAILED_ROWS_PATH)
        return validated_df

    except SchemaErrors as err:
        print("⚠ Validation errors found!")

        # Extract details of failed rows
        error_df = err.failure_cases

        # Select rows that failed validation
        failed_rows = df.iloc[error_df["index"].unique()]
        
        # Select rows that passed validation
        valid_rows = df.drop(error_df["index"].unique())

        # Save valid and failed rows to separate files
        save_csv(valid_rows, VALIDATED_DATA_PATH)
        save_csv(failed_rows, FAILED_ROWS_PATH)

        print("Valid rows saved.")
        print("Failed rows saved.")

        return valid_rows


if __name__ == "__main__":
    
    # Load raw CSV data and run validation
    print(f"Loading: {RAW_DATA_PATH}")
    df = load_csv(RAW_DATA_PATH)
    validate_data(df, ChurnSchema)
