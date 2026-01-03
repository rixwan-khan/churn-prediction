# src/evaluation/threshold_tuning.py

"""
THRESHOLD TUNING & BUSINESS DECISION
----------------------------------------------
Purpose:
- Tune optimal thresholds for each final model using validation set
- Optimize based on both:
    1) ML metric (F1-score)
    2) Business cost minimization
- Save thresholds & analysis for test evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score
from src.utils.logger import get_logger
from src.data.splitted_dataset import load_splitted_data
from src.utils.model_io import load_model
from src.utils.io import save_csv
from src.utils.paths import REPORTS_DIR

# --------logger configuration
logger = get_logger(
    log_filename='threshold_tuning.log',
    log_subdir='evaluation'
)

# --------Business Cost Matrix
"""
Business Assumptions:

TP: Correctly identify churn -> Send retention offer -> Cost = incentive
FP: Incorrectly predict churn -> Offer sent unnecessarily -> Cost = wasted incentive
FN: Missed churn -> Customer leaves -> Cost = lost revenue
TN: Correctly predict stay -> No cost
"""
COST_MATRIX = {
    'TP': -50,
    'FP': -50,
    'FN': -300,
    'TN': 0
}

# ---------- Business Cost Calculator ----------
def calculate_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Convert confusion matrix counts into expected business cost.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (
        tp * COST_MATRIX['TP'] +
        fp * COST_MATRIX['FP'] +
        fn * COST_MATRIX['FN'] +
        tn * COST_MATRIX['TN']
    )
    return total_cost



