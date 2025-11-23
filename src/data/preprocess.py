"""
DATA PREPROCESSING PIPELINE
---------------------------
This module handles preprocessing:
- Loading raw data
- Type conversion
- Duplicate removal
- Missing value imputation & flags
- Outlier detection & flags
- Optional feature scaling
- Saving fully cleaned dataset

Designed to be robust, reproducible, and production-ready.
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import RAW_DATA_PATH, CLEAN_DATA_PATH  # project-specific paths


# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------
def load_data(filepath=RAW_DATA_PATH):
    """
    Load dataset from CSV or Excel.
    Default path = RAW_DATA_PATH
    Returns pandas DataFrame.
    """
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -------------------------------------------------------------------
# 2. REMOVE DUPLICATES
# -------------------------------------------------------------------
def remove_duplicates(df, subset_cols=None, keep='first'):
    # row count before checking duplicate
    before = df.shape[0]
    
     # Drop duplicates (all columns if subset_cols=None)
    df = df.drop_duplicates(subset=subset_cols, keep=keep)

    # row count after removing deuplicate
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows")
    return df


# -------------------------------------------------------------------
# 3. FIX DATA TYPES
# -------------------------------------------------------------------
def convert_types(df, numeric_cols, categorical_cols, datetime_cols=None):
    
    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Convert datetime columns if provided
    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    print("Datatype conversion completed")
    return df


# -------------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# -------------------------------------------------------------------
def handle_missing(df, numeric_cols, categorical_cols, threshold=0.02):
    print("Handling missing values...")
    
     # Numeric columns
    for col in numeric_cols:
        missing_pct = df[col].isnull().mean() # fraction of missing values
        if missing_pct <= threshold:
            df[col].fillna(df[col].median(), inplace=True) # fill small missing with median
        else:
            # For large missing, create a flag column and fill
            df[f"was_missing_{col}"] = df[col].isnull().astype(int)
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns
    for col in categorical_cols:
        missing_pct = df[col].isnull().mean()
        if missing_pct <= threshold:
            df[col].fillna(df[col].mode()[0], inplace=True) # fill small missing with mode
        else:
            # For large missing, create a flag column and fill
            df[f"was_missing_{col}"] = df[col].isnull().astype(int)
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("Missing value handling completed")
    return df


# -------------------------------------------------------------------
# 5. OUTLIER DETECTION & FLAGGING
# -------------------------------------------------------------------
def flag_outliers(df, numeric_cols, z_thresh=3, iqr_factor=1.5, iso_contamination=0.01):
    print("Flagging outliers...")

    # Z-score
    for col in numeric_cols:
        df[f"{col}_zscore"] = zscore(df[col].fillna(df[col].median()))
        df[f"{col}_outlier_z"] = (df[f"{col}_zscore"].abs() > z_thresh).astype(int)

    # IQR
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df[f"{col}_outlier_iqr"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

    # Isolation Forest
    iso = IsolationForest(contamination=iso_contamination, random_state=42)
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    df["outlier_iso"] = iso.fit_predict(df_numeric)
    df["outlier_iso"] = df["outlier_iso"].map({1: 0, -1: 1})

    print("Outlier flagging completed")
    return df


# -------------------------------------------------------------------
# 6. OPTIONAL FEATURE SCALING
# -------------------------------------------------------------------
def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Feature scaling completed")
    return df


# -------------------------------------------------------------------
# 7. SAVE PROCESSED DATA
# -------------------------------------------------------------------
def save_processed(df, filepath=CLEAN_DATA_PATH):
    """
    Save preprocessed dataset to clean folder.
    Default path = CLEAN_DATA_PATH
    """
    df.to_csv(filepath, index=False)
    print(f"Processed dataset saved to: {filepath}")


# -------------------------------------------------------------------
# 8. MASTER PIPELINE
# -------------------------------------------------------------------
def preprocess_pipeline(input_path=RAW_DATA_PATH, output_path=CLEAN_DATA_PATH,
                        numeric_cols=None, categorical_cols=None, datetime_cols=None, scale=False):
    """
    Full preprocessing pipeline:
    1. Load dataset
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

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = convert_types(df, numeric_cols, categorical_cols, datetime_cols)
    df = handle_missing(df, numeric_cols, categorical_cols)
    df = flag_outliers(df, numeric_cols)

    if scale:
        df = scale_features(df, numeric_cols)

    save_processed(df, output_path)

    print("\n==============================")
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("==============================\n")

    return df