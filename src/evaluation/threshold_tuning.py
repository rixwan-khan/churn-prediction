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
    'TP': -50,  # Cost of retention incentive
    'FP': -50,  # Wasted incentive
    'FN': -300, # Lost customer revenue
    'TN': 0
}

# ---------- Business Cost Calculator ----------
def calculate_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Converts confusion matrix values into total business cost.
    Args:
        y_true: Actual churn labels
        y_pred: Predicted churn labels after thresholding
    
    Returns: Total business cost (negative = loss)
    """

    # confusion_matrix returns values in this order:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate total cost using predefined cost matrix
    total_cost = (
        tp * COST_MATRIX['TP'] +
        fp * COST_MATRIX['FP'] +
        fn * COST_MATRIX['FN'] +
        tn * COST_MATRIX['TN']
    )
    return total_cost


# -------- Threshold Optimization
def tune_threshold (model, X_val, y_val):
    """
    Finds the optimal probability threshold for a model.

    Steps:
        - Get churn probabilities from model
        - Try multiple thresholds between 0.01 and 0.99
        - Convert probabilities to predictions
        - Calculate F1-score and business cost for each threshold
    Returns:
        - results_df   : threshold vs metrics
        - best_f1_row  : row with max F1
        - best_cost_row: row with max business cost
    """
    logger.info('Starting threshold optimization')
    churn_probs = model.predict_proba(X_val)[:,1]

    thresholds = np.linspace(0.01, 0.99, 200)
    records = []

    for thresh in thresholds:
        y_pred = (churn_probs >= thresh).astype(int)
        records.append({
            'threshold'    : round(thresh,3),
            'f1_score'     : f1_score(y_val, y_pred),
            'business_cost': calculate_business_cost(y_val, y_pred)
        })
    
    # Convert results into DataFrame
    results_df = pd.DataFrame(records)

    # Find threshold with maximum F1-score
    best_f1_row = results_df.loc[results_df['f1_score'].idxmax()]

    # Find threshold with minimum business loss (maximum cost value)
    best_cost_row = results_df.loc[results_df['business_cost'].idxmax()]

    logger.info(f"Best F1 threshold: {best_f1_row.threshold} | F1-scores: {best_f1_row.f1_score:.4f}")
    logger.info(f"Best business threshold; {best_cost_row.threshold} | Expected cost: {best_cost_row.business_cost}")

    return results_df, best_f1_row, best_cost_row

# -------- Main Execution
def main():
    """
    Orchestrates the full threshold tuning process:
    - Loads validation data
    - Loads trained models
    - Computes PR-AUC
    - Tunes thresholds
    - Saves threshold analysis
    """
    
    logger.info('Threshold Tuning & Business Decision Pipline Started')

    # loading dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()
    logger.info('Loaded train / Val / test datasets')

    # loading final models
    models = {
        'LogisticRegression': load_model('final_logistic_regression'),
        'XGBoost': load_model('final_xgboost')
    }

    for name, model in models.items():
        logger.info(f'Optimizing threshold for {name}')
        val_probs = model.predict_proba(X_val)[:,1]
        pr_auc = average_precision_score(y_val, val_probs)
        logger.info(f"{name} Validation PR-AUC: {pr_auc:.4f}")

        #Tune thresholds
        results_df, best_f1, best_cost = tune_threshold(model, X_val, y_val)

        #Saving threshold complete analysis
        output_path = REPORTS_DIR / f'{name}_threshold_analysis.csv'
        save_csv(results_df, output_path)
        logger.info(f"{name} threshold analysis saved at: {output_path}")
    


if __name__ == "__main__":
    main()

