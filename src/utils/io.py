# src/utils/io.py
import pandas as pd
import os

def load_csv(path):
    """Load CSV safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def save_csv(df, path):
    """Save dataframe to CSV safely, create directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
