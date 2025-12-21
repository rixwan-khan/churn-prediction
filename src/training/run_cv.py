# src/training/run_cv.py

"""
Executes stratified cross-validation for multiple models 
(Logistic Regression, Random Forest, XGBoost) on the featured churn dataset.

It evaluates model stability using PR-AUC and logs results for reproducibility.

"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from src.training.cross_validation import run_cross_validation
from src.utils.paths import DATA_DIR
from src.utils.logger import get_logger


# ----- Logger initialization
logger = get_logger(
    log_filename='run_cv.log',
    log_subdir='training'
)

# ----- Featured dataset loading
FEATURED_DATA_PATH = DATA_DIR / '04_featured' / 'featured_telco_churn.csv'


def main():
    """ Main execution function for running CV experiments.."""

    logger.info('Starting Cross_Validation experiments...')

    #--- Loading dataset
    df = pd.read_csv(FEATURED_DATA_PATH)
    logger.info(f'Loaded dataset with shape: {df.shape}')

    #--- Features and targets of dataset
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    logger.info('Features & Target of dataset seperated...')



    #--- Cross_validation for Logistic Regression with Feature Scaling.
    logger.info('Running CV for Logistic Regression with feature scaling')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # scaling all features (linear model)

    lr = LogisticRegression(
        max_iter= 500,          # Can increase for convergence
        class_weight='balanced' # handling imbalanced data
    )
    lr_mean, lr_std = run_cross_validation(
        model=lr,
        X=pd.DataFrame(X_scaled, columns=X.columns),
        y=y,
        n_splits=5
    )
    logger.info(f'LogisticRegression | PR-AUC: {lr_mean:.4f} ± {lr_std:.4f}')


    # --- Cross_validation for Random Forest
    logger.info('Running CV for Random Forest')
    rf = RandomForestClassifier(
        n_estimators=300,      # No. of trees
        random_state=42,       # Reproducibility
        n_jobs=-1              # Use all CPU Cores
    )
    rf_mean, rf_std = run_cross_validation(
        model=rf,
        X=X,
        y=y,
        n_splits=5
    )
    logger.info(f'RandomForest  | PR-AUC: {rf_mean:.4} ± {rf_std:.4f}')


    # --- Cross_Validation for XGBoost
    logger.info('Running CV for XGBoost')
    scale_pos_weight = (y==0).sum() / (y==1).sum()  # Balancing class

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
        X=X,
        y=y,
        n_splits=5
    )
    logger.info(f'XGBoost | PR-AUC: {xgb_mean:.4f} ± {xgb_std:.4f}')

    logger.info('Cross-Validation experiments completed successfully')




if __name__ == '__main__':
        main()