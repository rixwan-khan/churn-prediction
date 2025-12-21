"""
DATA PREPROCESSING PIPELINE
---------------------------------------------------------
Handles:
- Loading validated data
- Type conversion
- Duplicate removal
- Missing value imputation & flags
- Outlier detection & flags
- Optional feature scaling
- Saving fully cleaned dataset
- Logger support
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -------------------------
# Project internal modules
# -------------------------
from src.utils.paths import VALIDATED_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.utils.io import load_csv, save_csv
from src.validation.validate import validate_dataset
from src.validation.schema import PREPROCESSING_SCHEMA
from src.configs.config import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    DATETIME_COLS,
    SCALE_NUMERIC,
)

logger = get_logger("preprocessing")


# ===============================
# 1. LOAD DATA
# ===============================
def load_data(filepath=VALIDATED_DATA_DIR / "validated_df.csv"):
    df = load_csv(filepath)
    msg = f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns"
    print(msg)
    logger.info(msg)
    return df


# ===============================
# 2. REMOVE DUPLICATES
# ===============================
def remove_duplicates(df, subset_cols=None, keep="first"):
    before = df.shape[0]
    df = df.drop_duplicates(subset=subset_cols, keep=keep)
    removed = before - df.shape[0]

    msg = f"Removed {removed} duplicate rows"
    print(msg)
    logger.info(msg)
    return df


# ===============================
# 3. FIX DATA TYPES
# ===============================
def convert_types(df, numeric_cols, categorical_cols, datetime_cols=None):
    summary = {}

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        summary[col] = str(df[col].dtype)

    for col in categorical_cols:
        df[col] = df[col].astype("category")
        summary[col] = str(df[col].dtype)

    if datetime_cols:
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            summary[col] = str(df[col].dtype)

    logger.info("Datatype conversion completed")
    logger.info(f"Converted dtypes: {summary}")
    print("Datatype conversion completed")

    return df


# ===============================
# 4. HANDLE MISSING VALUES
# ===============================
def handle_missing(df, numeric_cols, categorical_cols, threshold=0.02):
    print("Handling missing values...")
    logger.info("Handling missing values...")

    missing_summary = {}

    # Numeric columns
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        missing_pct = df[col].isnull().mean() * 100

        if missing_count > 0:
            if missing_pct <= threshold * 100:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[f"was_missing_{col}"] = df[col].isnull().astype(int)
                df[col].fillna(df[col].median(), inplace=True)

        missing_summary[col] = {
            "Missing Count": missing_count,
            "Missing %": round(missing_pct, 2),
        }

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

        missing_summary[col] = {
            "Missing Count": missing_count,
            "Missing %": round(missing_pct, 2),
        }

    logger.info("Missing values summary:\n" + str(pd.DataFrame(missing_summary).T))
    print("Missing value handling completed")
    return df


# ===============================
# 5. OUTLIER DETECTION
# ===============================
def flag_outliers(df, numeric_cols, z_thresh=3, iqr_factor=1.5, iso_contamination=0.01):
    print("Flagging outliers...")
    logger.info("Flagging outliers...")

    outlier_summary = {}

    # Z-score
    for col in numeric_cols:
        df[f"{col}_zscore"] = zscore(df[col].fillna(df[col].median()))
        df[f"{col}_outlier_z"] = (df[f"{col}_zscore"].abs() > z_thresh).astype(int)
        outlier_summary[col] = {
            "Z-score outliers": df[f"{col}_outlier_z"].sum()
        }

    # IQR
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df[f"{col}_outlier_iqr"] = (
            (df[col] < lower) | (df[col] > upper)
        ).astype(int)
        outlier_summary[col]["IQR outliers"] = df[f"{col}_outlier_iqr"].sum()

    # Isolation Forest
    iso = IsolationForest(contamination=iso_contamination, random_state=42)
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
    df["outlier_iso"] = iso.fit_predict(df_numeric)
    df["outlier_iso"] = df["outlier_iso"].map({1: 0, -1: 1})
    outlier_summary["IsolationForest total"] = df["outlier_iso"].sum()

    logger.info("Outlier summary:\n" + str(pd.DataFrame(outlier_summary).T))
    print("Outlier flagging completed")
    return df


# ===============================
# 6. OPTIONAL FEATURE SCALING
# ===============================
def scale_features(df, numeric_cols):
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    msg = "Feature scaling completed"
    print(msg)
    logger.info(msg)
    return df


# ===============================
# 7. SAVE CLEANED DATA
# ===============================
def save_processed(df, filepath=PROCESSED_DATA_DIR / "cleaned_data.csv"):
    save_csv(df, filepath)
    msg = f"Processed dataset saved to: {filepath}"
    print(msg)
    logger.info(msg)


# ===============================
# 8. MASTER PIPELINE
# ===============================
def preprocess_pipeline():
    print("\n=== PREPROCESSING PIPELINE STARTED ===")
    logger.info("=== Preprocessing pipeline started ===")

    # 1. Validate using schema
    validate_dataset(VALIDATED_DATA_DIR / "validated_df.csv", PREPROCESSING_SCHEMA)

    df = load_data()
    df = remove_duplicates(df)
    df = convert_types(df, NUMERIC_COLS, CATEGORICAL_COLS, DATETIME_COLS)
    df = handle_missing(df, NUMERIC_COLS, CATEGORICAL_COLS)
    df = flag_outliers(df, NUMERIC_COLS)

    if SCALE_NUMERIC:
        df = scale_features(df, NUMERIC_COLS)

    save_processed(df)

    logger.info("=== Preprocessing completed successfully ===")
    print("=== PREPROCESSING SUCCESSFULLY COMPLETED ===")
    return df
