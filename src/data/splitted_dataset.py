# src/data/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import DATA_DIR

# -------- Logger setup
logger = get_logger(
    log_filename='data_split.log',
    log_subdir='data'
)

# -------- Dataset path
FEATURED_DATA_PATH = DATA_DIR / '04_featured' / 'featured_telco_churn.csv'


def load_splitted_data(
        data_path: Path = FEATURED_DATA_PATH,
        target: str = 'Churn',
        test_size_val: float = 0.15,
        random_state: int = 42
):
    """
    Summary:
        Loading featured dataset and creating consistent split of train, validaiton and test.

    Args:
        data_path (Path, optional): _description_. Defaults to FEATURED_DATA_PATH.
        target (str, optional): Target column = 'Churn'.
        test_size_val (float, optional): Fraction for validation and test set each.
        random_state (int, optional): Seed for reproducibility.
    """
    logger.info('Starting dataset splitting process')

    if not data_path.exists():
        logger.error(f'Dataset not found at {data_path}')
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f'Dataset loaded from {data_path} | Shape: {df.shape}')

    if target not in df.columns:
        logger.error(f"Target column'{target}' not found")
        raise ValueError(f"Target column '{target}' not found in dataset")

    

    # -------- Separating features and target
    X = df.drop(columns=[target])
    y = df[target]

    # -------- First Split:  70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=2*test_size_val,
        stratify=y,
        random_state=random_state
    )

    # -------- Second Split: Validation 15%, Test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state
    )

    logger.info(
        f"Split completed |"
        f"Train:{X_train.shape},"
        f"Val:{X_val.shape},"
        f"Test: {X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# -------- Entry point
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()

    logger.info('Data splitting moduels executed successfully')


if __name__ == '__main__':
    main()