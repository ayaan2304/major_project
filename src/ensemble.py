"""
Soft voting, weighted voting, optional stacking.

What / why
----------
The original study reports a voting ensemble. We add validation-tuned
weights (never test-tuned). Stacking uses a logistic meta-learner on
validation probabilities (holdout stacking — the meta-learner never
sees test labels during fit).

Leakage risks
-------------
- Base models trained on train.
- Voting weights / stacker fitted on validation probabilities vs val labels.
- Test used only for the final frozen evaluation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression


def predict_proba_positive(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba
    return proba[:, 1]


def collect_probas(models: Mapping[str, Any], X) -> dict[str, np.ndarray]:
    return {name: predict_proba_positive(m, X) for name, m in models.items()}


def soft_vote(probas: Mapping[str, np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list(probas.values()))
    return stacked.mean(axis=0)


def weighted_vote(probas: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    names = list(probas)
    w = np.array([weights[n] for n in names], dtype=float)
    w = w / w.sum()
    stacked = np.vstack([probas[n] for n in names])
    return np.average(stacked, axis=0, weights=w)


def weights_from_val_pr_auc(y_val, val_probas: Mapping[str, np.ndarray]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score

    w = {}
    for name, p in val_probas.items():
        try:
            w[name] = float(max(average_precision_score(y_val, p), 1e-6))
        except ValueError:
            w[name] = 1e-6
    return w


def fit_stacker(val_probas: Mapping[str, np.ndarray], y_val, seed: int) -> LogisticRegression:
    X_meta = np.column_stack([val_probas[n] for n in sorted(val_probas)])
    clf = LogisticRegression(max_iter=400, random_state=seed, class_weight="balanced")
    clf.fit(X_meta, y_val)
    clf._meta_names = sorted(val_probas)  # type: ignore[attr-defined]
    return clf


def stacker_predict(stacker: LogisticRegression, probas: Mapping[str, np.ndarray]) -> np.ndarray:
    names = list(stacker._meta_names)  # type: ignore[attr-defined]
    X_meta = np.column_stack([probas[n] for n in names])
    return stacker.predict_proba(X_meta)[:, 1]
