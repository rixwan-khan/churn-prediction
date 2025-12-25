# src/training/hyperparameter_tunning/hyperparameter_tunning_xgb.py

""" Hyperparameter tunning module for XGBoost classifier. """

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

from src.utils.logger import get_logger
from src.utils.paths import DATA_DIR


# -------- Logger setup
logger = get_logger(
    log_filename='hyperparam_tuning_xgb.log',
    log_subdir='training'
)

# -------- Dataset path
FEATURED_DATA_PATH = DATA_DIR / '04_featured' / 'featured_telco_churn.csv'


# ------- Tunning function for XGBoost Classifier
def tune_xgboost(X: pd.DataFrame,y: pd.Series, cv_splits: int = 5, n_iter: int =20):
    logger.info('Starting hyperparameter tuning for XGBoost')

    # -------- Handle Class imbalance
    #  - scale_pos_weight tell XGBoost to penalize minority class errors more.
    scale_pos_weight = (y==0).sum() / (y==1).sum()

    # -------- Model Pipline
    pipline = Pipeline(
        [
            (
                'clf', XGBClassifier(
                    use_label_encoder = False,
                    eval_metric = 'logloss',
                    random_state = 42
                    )
            )
        ]
    )


    # -------- Hyperparameter search space
    #   - Model complexiy (depth, estimators)
    #   - Learning dynamics (learning rate)
    #   - Overfitting control (subsampling)
    param_dist = {
        'clf__n_estimators': randint(100, 500),     # Number of boosting trees
        'clf__max_depth': randint(3,10),            # Tree depth controls complexity
        'clf__learning_rate': uniform(0.01, 0.3),   # Step size shrinkage
        'clf__subsample': uniform(0.5, 0.5),        # Raw sampling per tree
        'clf__colsample_bytree': uniform(0.5, 0.5), # Feature sampling per tree
        'clf__scale_pos_weight': [scale_pos_weight] # Fixed imbalance correction
    }

    # -------- Cross-validation strategy
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=42
    )

    # -------- Randomized hyperparameter search
    search = RandomizedSearchCV(
        estimator=pipline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='average_precision',
        n_jobs=-1,
        cv=skf,
        verbose=1,
        random_state=42
    )

    # -------- Execute search
    search.fit(X, y)

    # -------- Logging best results
    logger.info(f'Best params for XGBoost: {search.best_params_}')
    logger.info(f'Best PR-AUC for XGBoost: {search.best_score_}')

    return search.best_estimator_


# Main entry point
def main():
    
    logger.info('Loading dataset for XGBoost tuning')

    df = pd.read_csv(FEATURED_DATA_PATH)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    tune_xgboost(X,y)

    logger.info('XGBoost hyperparameter tuning completed successfuly')

if __name__ == '__main__':
    main()