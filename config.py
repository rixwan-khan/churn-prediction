from pathlib import Path
from project_root import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILENAME = "telco-customer-churn-raw.csv"

RAW_DATA_PATH = DATA_DIR / DATA_FILENAME