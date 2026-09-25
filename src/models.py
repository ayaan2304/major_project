"""
Supervised estimators used in baseline and EP-IAIL ensembles.

What / why
----------
The original study mixes trees and a neural net. We keep that mix and
add CatBoost / LogReg for a stronger, still CPU-feasible ensemble.

Leakage risks
-------------
Factories return unfitted models. Fitting happens on train (or
resampled train) only. Class-weight variants must not be combined
with oversampling unless an ablation explicitly asks for it.

Approximation
-------------
MLP uses a documented sample cap (`models.mlp.max_samples`) because a
sklearn MLP on 5M rows is not CPU-practical.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils import get_logger

logger = get_logger("models")


def _scale_pos_weight(y) -> float:
    y = np.asarray(y).astype(int)
    n_pos = max(int(y.sum()), 1)
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos)


def make_logistic_regression(cfg: Mapping[str, Any], seed: int, class_weight: str | None = "balanced"):
    p = dict(cfg["models"]["logistic_regression"])
    cw = p.pop("class_weight", class_weight)
    return LogisticRegression(random_state=seed, class_weight=cw, **p)


def make_decision_tree(cfg: Mapping[str, Any], seed: int, class_weight: str | None = "balanced"):
    p = dict(cfg["models"]["decision_tree"])
    cw = p.pop("class_weight", class_weight)
    return DecisionTreeClassifier(random_state=seed, class_weight=cw, **p)


def make_random_forest(cfg: Mapping[str, Any], seed: int, class_weight: str | None = "balanced"):
    p = dict(cfg["models"]["random_forest"])
    cw = p.pop("class_weight", class_weight)
    p.pop("n_jobs", None)
    return RandomForestClassifier(
        random_state=seed, class_weight=cw, n_jobs=cfg["compute"]["n_jobs"], **p
    )


def make_xgboost(cfg: Mapping[str, Any], seed: int, y_train=None, use_spw: bool = True):
    p = dict(cfg["models"]["xgboost"])
    spw = _scale_pos_weight(y_train) if (use_spw and y_train is not None) else 1.0
    return XGBClassifier(random_state=seed, scale_pos_weight=spw, **p)


def make_lightgbm(cfg: Mapping[str, Any], seed: int, y_train=None, use_spw: bool = True):
    p = dict(cfg["models"]["lightgbm"])
    spw = _scale_pos_weight(y_train) if (use_spw and y_train is not None) else 1.0
    return LGBMClassifier(random_state=seed, scale_pos_weight=spw, verbose=-1, **p)


def make_catboost(cfg: Mapping[str, Any], seed: int, y_train=None, use_spw: bool = True):
    from catboost import CatBoostClassifier

    p = dict(cfg["models"]["catboost"])
    spw = _scale_pos_weight(y_train) if (use_spw and y_train is not None) else 1.0
    return CatBoostClassifier(random_seed=seed, scale_pos_weight=spw, **p)


def make_mlp(cfg: Mapping[str, Any], seed: int):
    p = dict(cfg["models"]["mlp"])
    p.pop("max_samples", None)
    return MLPClassifier(random_state=seed, **p)


def fit_mlp_maybe_capped(model, X, y, cfg: Mapping[str, Any], seed: int):
    """Fit MLP on a stratified cap if configured. Documented approximation."""
    cap = cfg["models"]["mlp"].get("max_samples")
    if cap is None or len(X) <= int(cap):
        model.fit(X, y)
        return model, {"mlp_rows_used": int(len(X)), "mlp_capped": False}
    rng = np.random.RandomState(seed)
    yv = np.asarray(y).astype(int)
    pos = np.where(yv == 1)[0]
    neg = np.where(yv == 0)[0]
    n_pos = len(pos)
    n_neg = min(len(neg), int(cap) - n_pos)
    if n_neg < 1:
        n_neg = min(len(neg), int(cap) // 2)
        pos = rng.choice(pos, size=min(n_pos, int(cap) - n_neg), replace=False)
    neg_s = rng.choice(neg, size=max(n_neg, 1), replace=False)
    idx = np.concatenate([pos, neg_s])
    rng.shuffle(idx)
    X_use = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    y_use = yv[idx]
    model.fit(X_use, y_use)
    logger.info("MLP fitted on %s rows (cap=%s) — documented approximation", f"{len(idx):,}", cap)
    return model, {"mlp_rows_used": int(len(idx)), "mlp_capped": True}


def baseline_model_zoo(cfg: Mapping[str, Any], seed: int, y_train) -> dict[str, Any]:
    """Original-style members plus LogReg (needed for a linear reference)."""
    return {
        "logistic_regression": make_logistic_regression(cfg, seed),
        "decision_tree": make_decision_tree(cfg, seed),
        "random_forest": make_random_forest(cfg, seed),
        "xgboost": make_xgboost(cfg, seed, y_train=y_train, use_spw=True),
        "lightgbm": make_lightgbm(cfg, seed, y_train=y_train, use_spw=True),
        "mlp": make_mlp(cfg, seed),
    }
