"""
Statistical tests and bootstrap confidence intervals.

Use Wilcoxon signed-rank when paired seed-level scores exist.
Friedman when ≥3 methods × ≥3 paired blocks.
Bootstrap CIs when we only have one test set (resample test predictions).

Leakage risks
-------------
Tests compare already-frozen test predictions. Do not re-tune inside the
bootstrap using test labels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score, f1_score


def bootstrap_metric_ci(
    y_true,
    y_prob,
    metric: Callable,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    threshold: float | None = None,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if threshold is None:
            stats.append(float(metric(y_true[idx], y_prob[idx])))
        else:
            y_pred = (y_prob[idx] >= threshold).astype(int)
            stats.append(float(metric(y_true[idx], y_pred)))
    arr = np.asarray(stats)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "std": float(arr.std()), "ci_lo": float(lo), "ci_hi": float(hi)}


def bootstrap_pr_auc_ci(y_true, y_prob, n_boot: int, seed: int) -> dict[str, float]:
    return bootstrap_metric_ci(y_true, y_prob, average_precision_score, n_boot, seed)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 6:
        return {"statistic": float("nan"), "pvalue": float("nan"), "note": "n<6; Wilcoxon not powered"}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(stat), "pvalue": float(p)}


def friedman_methods(score_matrix: np.ndarray) -> dict[str, float]:
    """score_matrix: shape (n_blocks, n_methods)."""
    cols = [score_matrix[:, j] for j in range(score_matrix.shape[1])]
    stat, p = friedmanchisquare(*cols)
    return {"statistic": float(stat), "pvalue": float(p)}


def js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 30) -> float:
    """Jensen–Shannon divergence between two 1-D samples (feature or score)."""
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    if hi <= lo:
        return 0.0
    hp, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
    hq, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)
    hp = hp / max(hp.sum(), 1e-12)
    hq = hq / max(hq.sum(), 1e-12)
    m = 0.5 * (hp + hq)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return float(0.5 * _kl(hp, m) + 0.5 * _kl(hq, m))
