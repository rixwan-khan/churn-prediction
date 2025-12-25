# src/data/splitted_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# ===== Logger setup =====
LOG_DIR = Path("logs/data")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "splitted_dataset.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===== Default dataset path =====
FEATURED_DATA_PATH = Path("data/featured/featured_telco_churn.csv")

def load_splitted_data(
    data_path: Path = FEATURED_DATA_PATH,
    target: str = 'Churn',
    test_size_val: float = 0.15,
    random_state: int = 42cd
):
    """
    Load featured dataset and return consistent train/val/test splits.

    Args:
        data_path (Path): CSV file path
        target (str): Target column name
        test_size_val (float): Fraction for validation/test (default 0.15 each)
        random_state (int): Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset {data_path} with shape: {df.shape}")

    if target not in df.columns:
        logger.error(f"Target column '{target}' not found in dataset")
        raise ValueError(f"Target column '{target}' not found in dataset")

    X = df.drop(columns=[target])
    y = df[target]

    # --- First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=2*test_size_val, stratify=y, random_state=random_state
    )

    # --- Second split: 50% validation, 50% test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    logger.info(
        f"Split completed | Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ===== Optional quick test =====
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
