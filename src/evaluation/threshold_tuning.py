"""
THRESHOLD TUNING & BUSINESS DECISION MODULE
------------------------------------------
Purpose:
- Convert churn probabilities into actionable business decisions
- Select optimal operating threshold using:
    1) ML metric (F1-score)
    2) Business cost minimization
- Create a clear bridge between ML output and retention strategy
"""
import numpy as np
import pandas as pd
from sklearn.metrics import(
    f1_score,
    confusion_matrix,
    average_precision_score
)

from src.utils.logger import get_logger
from src.data.splitted_dataset import load_splitted_data
from src.utils.io import save_csv
from src.utils.paths import REPORTS_DIR

# -------- Logger setup
logger = get_logger(
    log_filename='threshold_tuning.log',
    log_subdir='evaluation'
)

# -------- Business Cost Assumptions
"""
Business Interpretation:

TP (True Positive):
- Customer would churn
- Send retention offer
- Cost = incentive / discount cost

FP (False Positive):
- Customer would not Churn
- Still offer sent
- Cost = wasted incentive

FN (False Negative):
- Customer churn
- No action taken
- Cost = lost revenue (highest impact)

FN (True Negative):
- Customer Stays
- No action
- Cost = 0
"""
COST_MATRIX = {
    'TP': -50,
    'FP': -50,
    'FN': -300,
    'TN': 0
}


# -------- Business Cost Calculator
def calculate_business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Compute expected business cost based on confusion matrix.
    This function converts ML errors into financial impact.
    Args:
        y_true (np.ndarray): ground truth actual values
        y_pred (np.ndarray): model decision

    Returns:
        int: (total expected cost)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = (
        tp * COST_MATRIX['TP']+
        fp * COST_MATRIX['FP']+
        fn * COST_MATRIX['FN']+
        tn * COST_MATRIX['TN']  
    )
    return total_cost

# -------- Threshold Optimization
def tune_threshold(model, X_val, y_val):
    """
    Finding optimal probability threshold using:
    1) F1-score maximization (ML prespective)
    2) Business cost minimization (Business prespective)

    Args:
        model: trained model 
        X_val: Validation set used for threshold evaluaiton
        y_val: True labels corresponding to X_val


    Returns:
        results_df   : contains threshold, F1_score, business_cost for each threshold
        best_f1_row  : Row with threshold with maximum F1_score
        best_cost_row: Rwo with threshold with maximum business cost      
    """
    logger.info('Starting threshold optimizaiton')

    # Step 1: Predict probabilities for the positive class (churn = 1)
    churn_probs = model.predict_proba(X_val)[:,1]

    # Step 2: Define thresholds to evaluate (0.01 to 0.99)
    thresholds = np.linspace(0.01, 0.99, 200)
    results = []

    # Step 3: Iterate over thresholds and calculate metrics
    for threshold in thresholds:
        
        # Convert probabilities to binary predictions using current threshold
        y_pred = (churn_probs >= threshold).astype(int)

        record = {
            'threshold'     : round(threshold, 3),
            'f1_score'      : f1_score(y_val, y_pred),
            'business_cost' : calculate_business_cost(y_val, y_pred) 
        }
        results.append(record)
    
    # Step 4: Aggregate results into DataFrame
    results_df = pd.DataFrame(results)

    # Step 5: Identify threshold that maximizes F1-score
    best_f1_row = results_df.loc[results_df['f1_score'].idxmax()]
    
    # Step 6: Identify threshold that maximizes expected business cost
    best_cost_row = results_df.loc[results_df['business_cost'].idxmax()]

    logger.info(
        f"Best F1 threshold: {best_f1_row.threshold} | "
        f"F1-score: {best_f1_row.f1_score:.4f}"
    )

    logger.info(
        f"Best business threshold: {best_cost_row.threshold} | "
        f"Expected cost: {best_cost_row.business_cost}"
    )

    return results_df, best_f1_row, best_cost_row



# -------- Main Execution

def main():
    logger.info('Threshold Tuning Pipeline Started.')

    #Loading consistent Train / Val / Test split
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()
    logger.info('Loaded train/val/test datasets')

    # Load best tuned model  (PR-AUC based)
    model = load_best_model()
    logger.info(f'Loaded best model: {model.__class__.__name__}')

    # Sanity Check: Validaiton PR_AUC
    val_probs = model.predict_proba(X_val)[:,1]
    pr_auc = average_precision_score(y_val, val_probs)
    logger.info(f'Validation PR-AUC: {pr_auc:.4f}')

    # Threshold tuning
    threshold_df, best_f1, best_cost = tune_threshold(
        model=model,
        X_val=X_val,
        y_val=y_val
    )

    # Save threshold analysis
    output_path = REPORTS_DIR / 'threshold_analysis.csv'
    save_csv(threshold_df, output_path)
    logger.info(f'Threshold analysis saved at: {output_path}')
    logger.info('Threshold tuning Pipeline Completed.')


if __name__=='__main__':
    main()