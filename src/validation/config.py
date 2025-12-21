# src/validation/config.py

from src.utils.paths import RAW_DATA_DIR, VALIDATED_DATA_DIR, PROCESSED_DATA_DIR, FEATURED_DATA_DIR

# ---------------------------------------
# Dataset Filenames
# ---------------------------------------
RAW_DATA_FILENAME = "telco-customer-churn-raw.csv"
VALIDATED_DATA_FILENAME = "validated_customer_churn.csv"
FAILED_ROWS_FILENAME = "failed_rows.csv"
PROCESSED_DATA_FILENAME = "cleaned_telco_churn.csv"
FEATURED_DATA_FILENAME = "featured_telco_churn.csv"

# ---------------------------------------
# Full Paths
# ---------------------------------------
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
VALIDATED_DATA_PATH = VALIDATED_DATA_DIR / VALIDATED_DATA_FILENAME
FAILED_ROWS_PATH = VALIDATED_DATA_DIR / FAILED_ROWS_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
FEATUREED_DATA_PATH = FEATURED_DATA_DIR / FEATURED_DATA_FILENAME

# ---------------------------------------
# Dataset column types
# ---------------------------------------
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]