# src/training/hyperparameter_tunning/hyperparameter_lr.py

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

from src.utils.logger import get_logger
from src.utils.paths import DATA_DIR


# ----- logger setup
logger = get_logger(
    log_filename='hyperparam_tuning_lr.log'
    log_subdir= 'training'
)

# ---- Dataset path
FEATURE_DATA_PATH = DATA_DIR / '04_featured' / 'featured_telco_churn.csv'


