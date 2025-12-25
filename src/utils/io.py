# src/utils/io.py

import pandas as pd
from pathlib import Path

def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        path (str or Path): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)       # Ensure path is a Path object
    if not path.exists():   # Check if file exists
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)# Read CSV into DataFrame

def save_csv(df, path):
    """
    Save a pandas DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str or Path): Destination file path.
        
    Notes:
        - Automatically creates parent directories if they don't exist.
        - Does not write row indices (index=False).
    """
    path = Path(path)             # Ensure path is a Path object
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)  # Save DataFrame to CSV
