"""
evaluation/metrics.py
Compute and display all required evaluation metrics for
binary intrusion detection classification.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> dict:
    """
    Compute all six evaluation metrics for a binary classifier.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = benign, 1 = attack).
    y_pred : np.ndarray
        Predicted binary labels.
    y_scores : np.ndarray
        Predicted probability scores for the attack class.

    Returns
    -------
    dict with keys: precision, recall, f1_score, roc_auc,
                    pr_auc, mcc, confusion_matrix
    """
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    mcc    = matthews_corrcoef(y_true, y_pred)
    cm     = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc  = average_precision_score(y_true, y_scores)
    except ValueError:
        roc_auc = 0.0
        pr_auc  = 0.0

    return {
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "f1_score":         round(f1, 4),
        "roc_auc":          round(roc_auc, 4),
        "pr_auc":           round(pr_auc, 4),
        "mcc":              round(mcc, 4),
        "confusion_matrix": cm,
    }


def format_metrics_table(model_metrics: dict) -> str:
    """
    Format a metrics dictionary as a readable text table.

    Parameters
    ----------
    model_metrics : dict
        {model_name: metrics_dict} mapping.

    Returns
    -------
    str
        Formatted table string.
    """
    header = (
        f"{'Model':<22} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8} {'MCC':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for model_name, m in model_metrics.items():
        rows.append(
            f"{model_name:<22} {m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f1_score']:>8.4f} {m['roc_auc']:>9.4f} "
            f"{m['pr_auc']:>8.4f} {m['mcc']:>8.4f}"
        )
    return "\n".join(rows)
