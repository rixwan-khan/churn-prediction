#src/training/model_training.py

"""
MODEL TRAINING MODULE
---------------------
This module handles model training AFTER PREPROCESSING + FEATURE ENGINEERING.

Responsibilities:
- Load featured dataset
- Train / Validation / Test split
- Handle class imbalance (SMOTE or class weights)
- Train multiple models (LogReg, RF, XGBoost)
- Tune threshold on validation set using PR-AUC
- Evaluate models on test set
- Save model & metrics
"""

#========== IMPORTS ============================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.utils.paths import DATA_DIR
from src.utils.paths import MODELS_DIR, REPORTS_DIR

# Path to featured dataset (after feature engineering)
FEATURED_DATA_PATH = DATA_DIR / "04_featured" / "featured_telco_churn.csv"


#========== LOGGER ============================
from src.utils.logger import get_logger
logger = get_logger('model_training.log', log_subdir='training')



#========== HELPER FUNCTIONS ===================

# Load dataset-------------------
def load_dataset():
    df = pd.read_csv(FEATURED_DATA_PATH)
    logger.info(f'Loaded dataset with shape: {df.shape}')
    return df

# Split dataset: Train / Validation / Test-------------------
#  70% train,  15% Validation, 15% test
def split_data(df, target='Churn'):

    # Separate features (X) and target label (y)
    X = df.drop(columns=[target])
    y = df[target]

    # First split: 70% Train, 30% Temporary
    # stratify=y keeps the class ratio consistent across all splits
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    # Second split on the Temporary set:
    # 30% → 15% Validation + 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    logger.info(f'Train:{X_train.shape}, Validation:{X_val.shape}, Test: {X_test.shape}')
    return X_train, X_val, X_test, y_train, y_val, y_test


# Threshold tuning method: F1-score maximization based on validation PR curve-------------------
def tune_threshold(y_true, y_prob):

    # Calculating precision, recall, and thresholds for all possible cutoffs
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Compute F1-score for each threshold
    # Small value 1e-9 added to avoid division by zero
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    # index of the threshold with the highest F1-score
    best_idx = np.argmax(f1_scores)

    # Select the threshold corresponding to the best F1-score
    best_threshold = thresholds[best_idx]
    return best_threshold

# Train Logistic Regression-------------------
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=300, class_weight='balanced')
    model.fit(X_train,y_train)
    return model

# Train Random Forest with SMOTE-------------------
def train_rf_smote(X_train, y_train):

    # SMOTE object to balance minority class
    sm = SMOTE(random_state=42)

    # SMOTE on training data and resampled.
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    #RandomForest Classifier initialized
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    # model trained on resampled data.
    model.fit(X_resampled, y_resampled)
    return model

# Train XGBoost -------------------
def train_xgb(X_train, y_train):
    
    #calculation imbalanced data ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train ==1).sum()

    model = XGBClassifier(
        n_estimators=300,            # total trees 
        max_depth=5,                 # each tree depth
        learning_rate=0.1,           # each tree contribution
        use_label_encoder=False,     # Disable manual label encoding
        eval_metrics='logloss',      # classification loss function
        scale_pos_weight = scale_pos_weight, # handling class imbalance
        random_state=42              # reproduceable same result.
    )
    model.fit(X_train,y_train)
    return model

# Evaluation of model with threshold-------------------
def evaluate(model, X,y, threshold=0.5):

    # Predict probability of positive class (class = 1) for each row
    y_prob = model.predict_proba(X)[:,1]

    # Convert probabilities into class labels using given threshold
    # If probability >= threshold → class 1, else class 0
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate evaluation metrics
    metrics = {
        'ROC_AUC': roc_auc_score(y, y_prob),
        'PR_AUC': average_precision_score(y,y_prob),
        'F1': f1_score(y,y_pred),
        'ConfusionMatrix': confusion_matrix(y, y_pred).tolist(),
        'Threshold': threshold
    }
    return metrics


# Saving model and metrics -------------------
def save_artifacts(model, metrics, name):
    """
    Saving trained model and evaluation metrics.

    Models  -> MODELS_DIR
    Metrics -> REPORTS_DIR
    """
    # Check directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    #Saving Model
    model_path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model:{model_path}")

    #Saving Metrics
    metrics_path = REPORTS_DIR / f"{name}_metrics.json"
    pd.Series(metrics).to_json(metrics_path)
    logger.info(f"Saved metrics:{metrics_path}")


#========== MAIN TRAINING PIPLINE ===================

def run_training():
    
    #Load dataset
    df = load_dataset()

    #Splitting dataset into train / validation / test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    results = {}

    # Logistic Regression ------------
    logreg      = train_logistic(X_train, y_train)
    th_logreg   = tune_threshold(y_val, logreg.predict_proba(X_val)[:,1])
    metrics_log = evaluate(logreg, X_test, y_test, th_logreg)
    save_artifacts(logreg, metrics_log, 'logistic_regression')
    results['LogisticRegression'] = metrics_log

    # Random Forest ---------------
    rf          = train_rf_smote(X_train, y_train)
    th_rf       = tune_threshold(y_val, rf.predict_proba(X_val)[:,1])
    metrics_rf  = evaluate(rf, X_test, y_test, th_rf)
    save_artifacts(rf, metrics_rf, 'rf_smote')
    results['RandomForest'] = metrics_rf

    # XGBoost ------------------
    xgb = train_xgb(X_train, y_train)
    th_xgb = tune_threshold(y_val, xgb.predict_proba(X_val)[:,1])
    metrics_xgb = evaluate(xgb, X_test, y_test, th_xgb)
    save_artifacts(xgb, metrics_xgb, 'xgboost')
    results['XGBoost'] = metrics_xgb

    # logger -------------------
    logger.info('Training completed. Models Evaluated on test set')
    for name, metrics in results.items():
        logger.info(f'{name}: {metrics}')
    
    print('Training completed. Logs updated and models saved in Models dir.')


#========== RUN MAIN ===================
if __name__ == '__main__':
    run_training()