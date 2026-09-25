"""
Cost-sensitive decision threshold.

Cost(t) = C_FN * FN(t) + C_FP * FP(t)

Choose t on validation to minimise cost; freeze; evaluate once on test.

Leakage risks
-------------
Never minimise cost on the test set. The grid is evaluated on val only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.metrics import confusion_matrix


def cost_at_threshold(y_true, y_prob, threshold: float, c_fn: float, c_fp: float) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = c_fn * fn + c_fp * fp
    return {"threshold": float(threshold), "cost": float(cost), "FP": int(fp), "FN": int(fn), "TP": int(tp), "TN": int(tn)}


def tune_threshold(y_val, p_val, c_fn: float, c_fp: float, grid_size: int = 99) -> dict[str, Any]:
    grid = np.linspace(0.01, 0.99, grid_size)
    rows = [cost_at_threshold(y_val, p_val, t, c_fn, c_fp) for t in grid]
    best = min(rows, key=lambda r: (r["cost"], -r["threshold"]))
    return {"best": best, "curve": rows, "C_FN" : c_fn, "C_FP": c_fp}
