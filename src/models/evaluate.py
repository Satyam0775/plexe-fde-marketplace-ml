import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from src.utils.logger import get_logger

logger = get_logger("evaluation")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model and return metrics dictionary.
    """

    logger.info("Starting model evaluation...")

    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, prob)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)

    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        "roc_auc": roc_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }