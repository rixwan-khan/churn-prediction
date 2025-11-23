# src/utils/io.py
import pandas as pd
from .paths import PROCESSED_DATA_DIR
from pathlib import Path

def load_csv(path=PROCESSED_DATA_DIR):
    """Load CSV safely."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def save_csv(df, path):
    """Save dataframe to CSV safely, create directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
