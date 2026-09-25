"""
Classification metrics, confusion costs, and majority-class baselines.

What / why
----------
Accuracy is dominated by legitimate traffic (~99.87%). We always report
precision, recall, F1, ROC-AUC, PR-AUC, MCC, FNR, FPR.

Leakage risks
-------------
Thresholds may be chosen on validation; test metrics are computed once
with a frozen threshold.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def confusion_rates(y_true, y_pred) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = max(tp + fn, 1)
    n_neg = max(tn + fp, 1)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR_recall": float(tp / n_pos),
        "FPR": float(fp / n_neg),
        "FNR": float(fn / n_pos),
        "TNR": float(tn / n_neg),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE with equal-width bins on [0, 1]."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def classification_report_dict(
    y_true,
    y_prob,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    rates = confusion_rates(y_true, y_pred)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.sum() else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        **rates,
    }
    if prefix:
        out = {f"{prefix}{k}": v for k, v in out.items()}
    return out


def majority_baseline(y_true) -> dict[str, Any]:
    """Always predict legitimate (class 0). Shows why accuracy misleads."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.zeros(len(y_true), dtype=float)
    return classification_report_dict(y_true, y_prob, threshold=0.5)


def metrics_row(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {"model": name}
    row.update(metrics)
    return row


def metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
