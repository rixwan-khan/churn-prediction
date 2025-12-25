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
    log_filename='hyperparam_tuning_lr.log',
    log_subdir= 'training'
)

# ---- Dataset path
FEATURE_DATA_PATH = DATA_DIR / '04_featured' / 'featured_telco_churn.csv'


# ---- Tunning function for logistic regression -----
def tune_logistic_regression(X: pd.DataFrame, y:pd.Series, cv_splits:int=5, n_iter: int=20):
    """
    Summary: Perform cross_validated hyperparameter tunning for logistic regression.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        cv_splits (int, optional): Number of CV folds, Defaults to 5.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 20.
    """

    logger.info('Starting hyperparameter tunning for logistic Regression.')

    # Pipeline ensures scaling is applied correctly inside each CV fold
    pipline = Pipeline(
        [
            # Feature scaling
            ('scaler', StandardScaler()),
            
            # Classificaiton model
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
        ]
    )

    # ----- Hyperparameter search space
    #     - C controls regularization strenght
    #     - solver: affects optimizaiton strategy
    param_dist = {
        'clf__C':uniform(0.01, 10),
        'clf__solver':['lbfgs','liblinear'],
    }

    # ----- Stratified CV maintains churn / non-churn ration in each fold
    skf = StratifiedKFold(
        n_splits=cv_splits, # No. of folds
        shuffle=True,       # Shuffle data before splitting
        random_state=42     # fixed and consistent split
    )

    # ----- RandomizedSearchCV balances performanced and computation cost
    search = RandomizedSearchCV(
        estimator= pipline,              # ML pipline (scaling+model)
        param_distributions=param_dist,  # Parameter ranges to sample from
        n_iter=n_iter,                   # random parameter combinaitons
        scoring='average_precision',     # PA-AUC for imbalanced churn data
        n_jobs=-1,                       # all CPU cores for training
        cv=skf,                          # Stratified CV to preserve class ratio
        verbose=1,                       # Show training progress
        random_state=42                  # consistent and reproducible results
    )

    # ----- Executes CV-based hyperparameter search
    search.fit(X,y)

    # ----- Logging best configuration and performance
    logger.info(f'Best params for Logistic Regression: {search.best_params_}')
    logger.info(f'Best PR-AUC for Logistic Regression: {search.best_score_}')

    return search.best_estimator_



# ----- Entry Point -----
def main():
    """ Loads data and executes LR hyperparameter tuning."""

    logger.info('Loading dataset for Logistic Regression tuning')

    df = pd.read_csv(FEATURE_DATA_PATH)

    # Separate feature and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    tune_logistic_regression(X,y)

    logger.info('Logistic Regression hyperparameter tuning completed successfully.')

if __name__ == '__main__':
    main()
