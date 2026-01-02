# src/training/run_cv.py

"""
RUN CROSS-VALIDATION (MODEL STABILITY CHECK)
--------------------------------------------
Purpose:
- Evaluate model stability using stratified CV
- Fold-wise scaling for linear models (prevent data leakage)
- Compute PR-AUC for each fold
- Compare Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.training.cross_validation import run_cross_validation
from src.utils.logger import get_logger
from src.data.splitted_dataset import load_splitted_data

# ----- Logger initialization
logger = get_logger(
    log_filename='run_cv.log',
    log_subdir='training'
)

def main():
    """ Main execution function for running CV experiments.."""

    logger.info('Starting Cross_Validation experiments...')

    # --- Load pre-split train/validation/test data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()
    logger.info('Loaded pre-split dataset.')

    # --- Combine train + validation for CV
    X_cv = pd.concat([X_train, X_val], axis=0)
    y_cv = pd.concat([y_train, y_val], axis=0)
    logger.info(f'Combined Train + Validation for CV: {X_cv.shape}, {y_cv.shape}')

    results = {}

    #--- CLogistic Regression with fold-wise scaling
    logger.info('Running CV for Logistic Regression (linear model with scaling)')

    lr_pipline = Pipeline([
          ('scaler', StandardScaler()),
          ('lr', LogisticRegression(
                max_iter=500,
                class_weight='balanced',
                random_state=42
          ))
    ])
    
    lr_mean, lr_std = run_cross_validation(
        model=lr_pipline,
        X=X_cv,
        y=y_cv,
        n_splits=5
    )
    results['LogisticRegression'] = (lr_mean, lr_std)
    logger.info(f'LogisticRegression | PR-AUC: {lr_mean:.4f} ± {lr_std:.4f}')


    # --- Random Forest (no scaling needed)
    logger.info('Running CV for Random Forest (tree-based, no scaling)')
    rf = RandomForestClassifier(
        n_estimators=300,      # No. of trees
        random_state=42,       # Reproducibility
        n_jobs=-1              # Use all CPU Cores
    )
    rf_mean, rf_std = run_cross_validation(
        model=rf,
        X=X_cv,
        y=y_cv,
        n_splits=5
    )
    results['RandomForest'] = (rf_mean, rf_std)
    logger.info(f'RandomForest  | PR-AUC: {rf_mean:.4} ± {rf_std:.4f}')


    # --- XGBoost (tree-based, no scaling)
    logger.info('RRunning CV for XGBoost (tree-based, no scaling)')

    scale_pos_weight = (y_cv==0).sum() / (y_cv==1).sum()  # Balancing class

    xgb = XGBClassifier(
        n_estimators = 300,         # no. of boosting rounds
        max_depth = 5,              # Max depth of each tree
        learning_rate = 0.1,        # Step size shrinkage
        scale_pos_weight = scale_pos_weight,
        #use_label_encoder = False,  # Disable depricated encoder
        eval_metric = 'logloss',    # Classification loss function
        random_state = 42,          # Reproducibility
    )
    xgb_mean, xgb_std = run_cross_validation(
        model=xgb,
        X=X_cv,
        y=y_cv,
        n_splits=5
    )
    results['XGBoost'] = (xgb_mean, xgb_std)
    logger.info(f'XGBoost | PR-AUC: {xgb_mean:.4f} ± {xgb_std:.4f}')

    logger.info('Cross-Validation experiments completed successfully')



if __name__ == '__main__':
        main()