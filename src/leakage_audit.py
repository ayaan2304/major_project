"""
Automated leakage audit.

What / why
----------
The novelty (SHAP prototypes, SMOTE, thresholds) is easy to leak if
anyone fits on the full table. This module asserts split disjointness,
time order, and that synthetic/prototype artifacts were built from train.

Failure mode
------------
If an assertion fires, the experiment must stop — not "continue with a warning".
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    pass


def assert_disjoint_indices(a, b, name_a: str, name_b: str) -> None:
    sa, sb = set(map(int, a)), set(map(int, b))
    inter = sa & sb
    if inter:
        raise LeakageError(f"{name_a} and {name_b} share {len(inter)} indices (e.g. {next(iter(inter))})")


def assert_temporal_order(train_steps, val_steps, test_steps) -> None:
    if len(train_steps) and len(val_steps) and np.max(train_steps) > np.min(val_steps):
        raise LeakageError("Train steps extend into validation (future leakage).")
    if len(val_steps) and len(test_steps) and np.max(val_steps) > np.min(test_steps):
        raise LeakageError("Validation steps extend into test (future leakage).")
    if len(train_steps) and len(test_steps) and np.max(train_steps) > np.min(test_steps):
        raise LeakageError("Train steps extend into test (future leakage).")


def assert_no_resampled_eval(split_name: str, was_resampled: bool) -> None:
    if split_name in {"val", "test", "validation"} and was_resampled:
        raise LeakageError(f"{split_name} was resampled — forbidden.")


def audit_temporal_pipeline(artifacts: Mapping[str, Any]) -> dict[str, bool]:
    """Run the standard assertion bundle. Returns a pass dict if all succeed."""
    assert_disjoint_indices(artifacts["train_index"], artifacts["val_index"], "train", "val")
    assert_disjoint_indices(artifacts["train_index"], artifacts["test_index"], "train", "test")
    assert_disjoint_indices(artifacts["val_index"], artifacts["test_index"], "val", "test")
    assert_temporal_order(
        np.asarray(artifacts["train_steps"]),
        np.asarray(artifacts["val_steps"]),
        np.asarray(artifacts["test_steps"]),
    )
    if artifacts.get("prototype_source") not in {None, "train_fraud"}:
        raise LeakageError("SHAP prototype must be built from train fraud only.")
    if artifacts.get("synthetic_from") not in {None, "train"}:
        raise LeakageError("Synthetic samples must come from train only.")
    if artifacts.get("thresholds_from") not in {None, "train", "val", "train_val"}:
        raise LeakageError("Thresholds must be chosen on train/val, never test.")
    return {"leakage_audit_passed": True}


def subsample_train_keep_all_fraud(
    X: pd.DataFrame,
    y: pd.Series,
    legit_cap: int | None,
    seed: int,
    keep_all_fraud: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """CPU approximation: keep every fraud row, cap legitimate rows.

    This is NOT silent: the returned meta must be persisted in metrics JSON.
    Validation and test are never passed through this function.
    """
    yv = y.astype(int)
    meta = {"keep_all_fraud": keep_all_fraud, "legit_cap": legit_cap, "n_before": int(len(X))}
    if legit_cap is None or (yv == 0).sum() <= int(legit_cap):
        meta.update({"n_after": int(len(X)), "applied": False})
        return X, y, meta
    rng = np.random.RandomState(seed)
    pos = np.where(yv.to_numpy() == 1)[0]
    neg = np.where(yv.to_numpy() == 0)[0]
    neg_s = rng.choice(neg, size=int(legit_cap), replace=False)
    idx = np.concatenate([pos, neg_s])
    rng.shuffle(idx)
    meta.update({"n_after": int(len(idx)), "applied": True, "fraud_kept": int(len(pos))})
    return X.iloc[idx], yv.iloc[idx], meta
