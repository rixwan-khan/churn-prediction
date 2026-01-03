

"""
BASELINE MODEL TRAINING
---------------------
Purpose:
- Train multiple ML models on preprocessed & feature-engineered data
- Compare models fairly using VALIDATION set only
- Save trained baseline models and validation metrics

Models:
- Logistic Regression
- Random Forest (with SMOTE)
- XGBoost
"""

#========== IMPORTS ============================

import time
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger    
from src.data.splitted_dataset import load_splitted_data
from src.utils.model_io import save_model
from src.utils.paths import REPORTS_DIR

#========== LOGGER ============================
logger = get_logger(
    'model_training.log',
    log_subdir='training'
    )

#========== HELPER FUNCTIONS ===================


# Train Logistic Regression-------------------
def train_logistic(X_train, y_train):
    """
    Trains Logistic Regression with class imbalance handling.
    """
    model = LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=42
        )
    model.fit(X_train,y_train)
    return model

# Train Random Forest with SMOTE-------------------
def train_rf_smote(X_train, y_train):
    """
    Applies SMOTE to balance classes and trains Random Forest.
    """
    # SMOTE object to balance minority class
    smote = SMOTE(random_state=42)

    # SMOTE on training data and resampled.
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    #RandomForest Classifier initialized
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # model trained on resampled data.
    model.fit(X_resampled, y_resampled)
    return model

# Train XGBoost -------------------
def train_xgb(X_train, y_train):
    """
    Trains XGBoost with scale_pos_weight for imbalance handling.
    """
    
    #calculation imbalanced data ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train ==1).sum()

    model = XGBClassifier(
        n_estimators=300,            # total trees 
        max_depth=5,                 # each tree depth
        learning_rate=0.1,           # each tree contribution
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',      # classification loss function
        scale_pos_weight = scale_pos_weight, # handling class imbalance
        random_state=42              # reproduceable same result.
    )
    model.fit(X_train,y_train)
    return model

# Evaluation of model with threshold-------------------
def evaluate(model, X_val, y_val):
    """
    Evaluates model performance on validation data.
    """

    # Predict probability of positive class (class = 1) for each row
    y_prob = model.predict_proba(X_val)[:,1]

    # Convert probabilities into class labels using given threshold
    # If probability >= threshold → class 1, else class 0
    #y_pred = (y_prob >= threshold).astype(int)

    # Calculate evaluation metrics
    metrics = {
        'ROC_AUC': roc_auc_score(y_val, y_prob),
        'PR_AUC': average_precision_score(y_val, y_prob),
    }
    return metrics


#========== MAIN TRAINING PIPLINE ===================

def run_training_pipeline():
    """
    End-to-end training pipeline.
    """
    pipeline_start = time.perf_counter()
    logger.info('Starting model training pipeline')

    # ---- Load consistent train / val / test splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_splitted_data()

    results = {}

    # Logistic Regression ------------
    logreg      = train_logistic(X_train, y_train)
    logreg_metrics = evaluate(logreg, X_val, y_val)
    
    save_model(logreg, 'baseline_logistic_regression')
    pd.Series(logreg_metrics).to_json(
        REPORTS_DIR / 'baseline_logistic_regression_metrics.json'
    )

    results['LogisticRegression'] = logreg_metrics
    logger.info('Baseline Logistic Regression completed')


    # Random Forest ---------------
    rf          = train_rf_smote(X_train, y_train)
    rf_metrics  = evaluate(rf, X_val, y_val)
    
    save_model(rf, 'baseline_random_forest')
    pd.Series(rf_metrics).to_json(
        REPORTS_DIR / 'baseline_random_forest_metrics.json'
    )
    
    results['random_forest'] = rf_metrics
    logger.info('Baseline Random Forest completed')

    # XGBoost ------------------
    xgb = train_xgb(X_train, y_train)
    xgb_metrics = evaluate(xgb, X_val, y_val)

    save_model(xgb, 'baseline_xgboost')
    pd.Series(xgb_metrics).to_json(
        REPORTS_DIR / 'baseline_xgboost_metrics.json'
    )

    results['xgboost'] = xgb_metrics
    logger.info('Baseline XGBoost completed')

    for model_name, metrics in results.items():
        logger.info(f'{model_name}: results: {metrics}')
    
    print('Stage-1 completed. Baseline models and metrics saved.')


#========== RUN MAIN ===================
if __name__ == '__main__':
    run_training_pipeline()