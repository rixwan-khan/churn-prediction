# src/validation/config.py

from src.utils.paths import RAW_DATA_DIR, VALIDATED_DATA_DIR, PROCESSED_DATA_DIR, FEATURED_DATA_DIR

# ---------------------------------------
# Dataset Filenames

# Define the names of all key CSV files used in the pipeline
RAW_DATA_FILENAME = "telco-customer-churn-raw.csv"       # Original raw dataset
VALIDATED_DATA_FILENAME = "validated_customer_churn.csv" # Dataset after validation
FAILED_ROWS_FILENAME = "failed_rows.csv"                 # Rows that failed validation
PROCESSED_DATA_FILENAME = "cleaned_telco_churn.csv"      # Cleaned/processed dataset
FEATURED_DATA_FILENAME = "featured_telco_churn.csv"      # Dataset with feature engineering applied


# ---------------------------------------
# Full Paths

# Combine directory paths with filenames for easy file access
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME                    # Path to the original raw dataset
VALIDATED_DATA_PATH = VALIDATED_DATA_DIR / VALIDATED_DATA_FILENAME  # path to the validated dataset after validation
FAILED_ROWS_PATH = VALIDATED_DATA_DIR / FAILED_ROWS_FILENAME        # path to the store rows that failed validation
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME  # path to the cleaned and preprocessed dataset
FEATUREED_DATA_PATH = FEATURED_DATA_DIR / FEATURED_DATA_FILENAME    # path to the dataset with engineered features


# ---------------------------------------
# Dataset column types

# Define which columns are numeric vs categorical for validation and processing
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]