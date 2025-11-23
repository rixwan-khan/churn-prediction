# src/config.py

from utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLEAN_DATA_DIR

# ---------------------------------------
# Dataset Filenames
# ---------------------------------------
RAW_DATA_FILENAME = "telco-customer-churn-raw.csv"
PROCESSED_DATA_FILENAME = "validated_df.csv"
CLEAN_DATA_FILENAME = "cleaned_telco_churn.csv"

# ---------------------------------------
# Full Paths
# ---------------------------------------
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
CLEAN_DATA_PATH = CLEAN_DATA_DIR / CLEAN_DATA_FILENAME
