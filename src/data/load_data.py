# src/data/load_data.py

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


from config import RAW_DATA_PATH

import numpy as np
import pandas as pd

df = pd.read_csv(RAW_DATA_PATH)

df.head()

df.info()