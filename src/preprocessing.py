"""
Original-style preprocessing (Phase 1 reproduction).

What / why
----------
The given baseline drops identifiers and `step`/`type`, then uses a
stratified random split. We reproduce that pipeline *exactly* so later
gains are not confused with a different feature set.

Inputs: full DataFrame, config.
Outputs: X_train, X_test, y_train, y_test, feature_names, split metadata.

Leakage risks
-------------
- Do not fit scalers on train+test. Optional scaling is train-only.
- Do not oversample before splitting.
- Stratified shuffle is NOT a temporal split; Phase 2 replaces it.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger("preprocessing")


def original_feature_matrix(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the original study's unused columns; return X, y."""
    target = cfg["data"]["target"]
    drop_cols = list(cfg["data"]["original_drop_cols"])
    y = df[target].astype(int)
    X = df.drop(columns=drop_cols + [target])
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Stratified 80/20 split matching the original-style protocol."""
    spec = cfg["split"]["original"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=float(spec["test_size"]),
        stratify=y if spec.get("stratified", True) else None,
        random_state=seed,
    )
    meta = {
        "protocol": "stratified_random",
        "test_size": spec["test_size"],
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "fraud_train": int(y_tr.sum()),
        "fraud_test": int(y_te.sum()),
        "feature_names": list(X.columns),
        "train_index": X_tr.index.astype(int).tolist()[:50],  # sample for audit, full saved separately
        "note": "Full index arrays are written to processed split files.",
    }
    return {
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "meta": meta,
    }


def fit_scaler_train_only(X_train: pd.DataFrame) -> StandardScaler:
    """StandardScaler fitted on TRAIN only. Apply later with `.transform`."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
