"""
DATA PREPROCESSING PIPELINE
---------------------------------------------------------
Handles:
- Loading raw data
- Type conversion
- Duplicate removal
- Missing value imputation & flags
- Outlier detection & flags
- Optional feature scaling
- Saving fully cleaned dataset
- Logger support for console + log output
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DATA_PATH, CLEAN_DATA_PATH

# ===============================
# 1. LOAD DATA
# ===============================
def load_data(filepath=PROCESSED_DATA_PATH, logger=None):
    """
    Load dataset from CSV or Excel.
    Logs shape of data and prints to console.
    """
    # Detect file type
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    # Log and print number of rows and columns
    msg = f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns"
    print(msg)
    if logger:
        logger.info(msg)

    return df


# ===============================
# 2. REMOVE DUPLICATES
# ===============================
def remove_duplicates(df, subset_cols=None, keep='first', logger=None):
    """
    Remove duplicate rows from the dataset.
    Logs how many duplicates were removed.
    """
    before = df.shape[0]  # Count rows before removing duplicates
    df = df.drop_duplicates(subset=subset_cols, keep=keep)
    after = df.shape[0]   # Count rows after removal
    removed = before - after

    # Log and print
    msg = f"Removed {removed} duplicate rows"
    print(msg)
    if logger:
        logger.info(msg)

    return df


# ===============================
# 3. FIX DATA TYPES
# ===============================
def convert_types(df, numeric_cols, categorical_cols, datetime_cols=None, logger=None):
    """
    Convert columns to appropriate types:
    - numeric_cols -> numeric
    - categorical_cols -> category
    - datetime_cols -> datetime (optional)
    Logs the datatype conversion summary.
    """
    summary = {}

    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        summary[col] = str(df[col].dtype)

    # Convert categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        summary[col] = str(df[col].dtype)

    # Convert datetime columns if provided
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            summary[col] = str(df[col].dtype)

    msg = "Datatype conversion completed"
    print(msg)
    if logger:
        logger.info(msg)
        logger.info(f"Column datatypes after conversion: {summary}")

    return df


# ===============================
# 4. HANDLE MISSING VALUES
# ===============================
def handle_missing(df, numeric_cols, categorical_cols, threshold=0.02, logger=None):
    """
    Handle missing values in dataset.
    - For numeric: fill small missing with median, flag large missing.
    - For categorical: fill small missing with mode, flag large missing.
    Prints and logs a summary table for easier readability.
    """
    print("Handling missing values...")
    missing_summary = {}

    # Numeric columns
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = df[col].isnull().mean() * 100

        if missing_count > 0:
            if missing_pct <= threshold * 100:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Flag column for large missing values
                df[f"was_missing_{col}"] = df[col].isnull().astype(int)
                df[col].fillna(df[col].median(), inplace=True)

        missing_summary[col] = {"Missing Count": missing_count, "Missing %": round(missing_pct, 2)}

    # Categorical columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = df[col].isnull().mean() * 100

        if missing_count > 0:
            if missing_pct <= threshold * 100:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[f"was_missing_{col}"] = df[col].isnull().astype(int)
                df[col].fillna(df[col].mode()[0], inplace=True)

        missing_summary[col] = {"Missing Count": missing_count, "Missing %": round(missing_pct, 2)}

    # Convert summary to readable DataFrame
    df_missing = pd.DataFrame(missing_summary).T
    print("\nMissing Values Summary:")
    print(df_missing)
    if logger:
        logger.info("\nMissing Values Summary:\n" + df_missing.to_string())

    print("Missing value handling completed")
    return df


# ===============================
# 5. OUTLIER DETECTION & FLAGGING
# ===============================
def flag_outliers(df, numeric_cols, z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, logger=None):
    """
    Detect and flag outliers using:
    - Z-score method
    - IQR method
    - Isolation Forest method
    Prints and logs summary tables.
    """
    print("Flagging outliers...")
    outlier_summary = {}

    # Z-score method
    for col in numeric_cols:
        df[f"{col}_zscore"] = zscore(df[col].fillna(df[col].median()))
        df[f"{col}_outlier_z"] = (df[f"{col}_zscore"].abs() > z_thresh).astype(int)
        outlier_summary[col] = {"Z-score outliers": df[f"{col}_outlier_z"].sum()}

    # IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df[f"{col}_outlier_iqr"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        outlier_summary[col]["IQR outliers"] = df[f"{col}_outlier_iqr"].sum()

    # Isolation Forest method
    iso = IsolationForest(contamination=iso_contamination, random_state=42)
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    df["outlier_iso"] = iso.fit_predict(df_numeric)
    df["outlier_iso"] = df["outlier_iso"].map({1: 0, -1: 1})
    outlier_summary["IsolationForest total"] = df["outlier_iso"].sum()

    # Convert summary to readable table
    df_outliers = pd.DataFrame(outlier_summary).T
    print("\nOutlier Summary:")
    print(df_outliers)
    if logger:
        logger.info("\nOutlier Summary:\n" + df_outliers.to_string())

    return df


# ===============================
# 6. OPTIONAL FEATURE SCALING
# ===============================
def scale_features(df, numeric_cols, logger=None):
    """
    Standardize numeric columns to mean=0, std=1
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    msg = "Feature scaling completed"
    print(msg)
    if logger:
        logger.info(msg)

    return df


# ===============================
# 7. SAVE PROCESSED DATA
# ===============================
def save_processed(df, filepath=CLEAN_DATA_PATH, logger=None):
    """
    Save cleaned dataframe to CSV.
    Creates directory if it doesn't exist.
    """
    os.makedirs(filepath.parent, exist_ok=True)
    df.to_csv(filepath, index=False)

    msg = f"Processed dataset saved to: {filepath}"
    print(msg)
    if logger:
        logger.info(msg)


# ===============================
# 8. MASTER PIPELINE
# ===============================
def preprocess_pipeline(input_path=PROCESSED_DATA_PATH, output_path=CLEAN_DATA_PATH,
                        numeric_cols=None, categorical_cols=None, datetime_cols=None,
                        scale=False, logger=None):
    """
    Full preprocessing pipeline in proper sequence:
    1. Load data
    2. Remove duplicates
    3. Convert datatypes
    4. Handle missing values
    5. Flag outliers
    6. Optional scaling
    7. Save cleaned dataset
    """
    print("\n==============================")
    print(" STARTING PREPROCESSING PIPELINE ")
    print("==============================\n")
    if logger:
        logger.info("=== Preprocessing pipeline started ===")

    df = load_data(input_path, logger=logger)
    df = remove_duplicates(df, logger=logger)
    df = convert_types(df, numeric_cols, categorical_cols, datetime_cols, logger=logger)
    df = handle_missing(df, numeric_cols, categorical_cols, logger=logger)
    df = flag_outliers(df, numeric_cols, logger=logger)
    if scale:
        df = scale_features(df, numeric_cols, logger=logger)
    save_processed(df, output_path, logger=logger)

    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("==============================\n")
    if logger:
        logger.info("=== Preprocessing pipeline completed successfully ===")
        logger.info(f"Final cleaned dataset shape: {df.shape}")

    return df
