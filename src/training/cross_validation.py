#./src/training/cross_validaiton.py
"""
Cross-validation.

This module contains logic for evaluating model stability using
stratified cross-validation.
"""

from typing import Callable, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score


def run_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    metric_fn: Callable = average_precision_score
) -> Tuple[float, float]:
    """
    Run stratified cross-validation and evaluate model performance.

    Parameters
    ----------
    model :
        Any sklearn-compatible classifier or Pipeline that implements
        fit() and predict_proba().
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    n_splits : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Seed for reproducibility.
    metric_fn : Callable, default=average_precision_score
        Metric function that accepts (y_true, y_pred_proba).

    Returns
    -------
    mean_score : float
        Mean cross-validation score.
    std_score : float
        Standard deviation across folds (stability indicator).
    """

    # Defensive checks to ensure correct input types and model capabilities
    
    # Model must provide probabilities
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba()")

    # Ensure features are DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    # Ensure target is Series   
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    # Creating stratified K-Fold object
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    # List to store metric scores for each fold
    fold_scores: List[float] = []

    # Main CV loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        
        # Select training and validation data for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] # Features
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] # Targets

        # Train model on current fold
        model.fit(X_train, y_train)

        # Predict probabilities (class 1)
        val_preds = model.predict_proba(X_val)[:, 1]

        # Compute metric
        score = metric_fn(y_val, val_preds)
        fold_scores.append(score)

    # Aggregate CV results
    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    return mean_score, std_score
