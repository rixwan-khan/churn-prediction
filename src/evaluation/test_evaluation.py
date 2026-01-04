"""
TEST SET EVALUATION
------------------------------
Purpose:
- Evaluate final trained models on TEST set
- Use thresholds computed in threshold_tuning stage
- Report ML-optimal vs Business-optimal performance
"""

import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.utils.logger import get_logger
from src.data.splitted_dataset import load_splitted_data
from src.utils.model_io import load_model
from src.utils.paths import REPORTS_DIR


# -------- Logger
logger = get_logger(
    log_filename="test_evaluation.log",
    log_subdir="evaluation"
)


# -------- Load Threshold
def load_best_threshold(model_name: str, strategy: str) -> float:
    path = REPORTS_DIR / f"{model_name}_threshold_analysis.csv"
    df = pd.read_csv(path)

    if strategy == "f1":
        threshold = df.loc[df["f1_score"].idxmax(), "threshold"]
    elif strategy == "business":
        threshold = df.loc[df["business_cost"].idxmax(), "threshold"]
    else:
        raise ValueError("Strategy must be 'f1' or 'business'")

    logger.info(f"{model_name} | {strategy.upper()} threshold = {threshold}")
    return float(threshold)


# -------- Evaluation Logic
def evaluate(model, X_test, y_test, threshold: float) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }


# -------- Main
def main():
    logger.info("Test Evaluation Started")

    _, _, X_test, _, _, y_test = load_splitted_data()
    logger.info("Test dataset loaded")

    models = {
        "LogisticRegression": load_model("final_logistic_regression"),
        "XGBoost": load_model("final_xgboost")
    }

    strategies = ["f1", "business"]
    all_results = {}

    for model_name, model in models.items():
        logger.info(f"Evaluating model: {model_name}")
        all_results[model_name] = {}

        for strategy in strategies:
            threshold = load_best_threshold(
                model_name=model_name,
                strategy=strategy
            )

            metrics = evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
                threshold=threshold
            )

            all_results[model_name][strategy] = metrics

            # Save individual strategy metrics
            with open(
                REPORTS_DIR / f"{model_name}_test_metrics_{strategy}.json", "w"
            ) as f:
                json.dump(metrics, f, indent=4)

            logger.info(
                f"{model_name} | {strategy.upper()} test metrics: {metrics}"
            )

    # Save combined report (useful for dashboards / CI)
    with open(REPORTS_DIR / "final_test_evaluation_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info("Test Evaluation Completed")
    print("Stage-5B complete: F1 & Business test evaluations saved.")


if __name__ == "__main__":
    main()
