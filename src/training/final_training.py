# src/training/final_training.py

"""
FINAL MODEL TRAINING
------------------------------
Purpose:
- Train final models using BEST hyperparameters
- Use TRAIN + VALIDATION data together
- Produce production-ready models

Models:
- Logistic Regression (best)
- XGBoost (best)
"""
import time
import pandas as pd

from src.utils.logger import get_logger
from src.data.splitted_dataset import load_splitted_data
from src.utils.model_io import load_model, save_model

# -------- Logger Configuration
logger = get_logger(
    log_filename='final_training.log',
    log_subdir='training'
)

# -------- Pipline
def run_final_training():
    """
    Final Training Pipline
    """
    logger.info('Final Stage Training Started')

    # ----- loading dataset (train/validation/test)
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()

    # ----- Combining Train + Validation data
    X_final = pd.concat([X_train, X_val], axis=0)
    y_final = pd.concat([y_train, y_val], axis=0)

    logger.info(f'Final training data shape: X={X_final.shape}, y={y_final.shape}')

    #-------- Logistic Regression (Best Hyperparameters)
    logger.info('Loading best Logistic Regression model from hyperparameter tuning step')
    best_lr = load_model('best_LogisticRegression')

    logger.info('Training final Logistic Regression model')
    best_lr.fit(X_final, y_final)

    save_model(best_lr, 'final_logistic_regression')
    logger.info('Final Logistic Regression model saved')

    # -------- XGBoost (Best Hyperparameter)
    logger.info('Loading best XGBoost model from hyperparameter tuning step')
    best_xgb = load_model('best_xgboost')

    logger.info('Training final xgboost model')
    best_xgb.fit(X_final, y_final)

    save_model(best_xgb, 'final_xgboost')
    logger.info('Final xgboost model saved')



if __name__ == '__main__':
    run_final_training()
