"""
Imbalance handling: class weights, random oversampling, SMOTE, Borderline-SMOTE.

What / why
----------
Fraud is ~0.13% of PaySim. Unweighted training can collapse to the
majority class. This module applies *train-only* resampling.

EP-IAIL does not live here; it *uses* SMOTE candidates then filters
them in `explanation_filter.py`.

Leakage risks
-------------
Never resample validation or test. Callers must pass training matrices.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler, SMOTE

from src.utils import get_logger

logger = get_logger("imbalance")


def class_counts(y: np.ndarray | pd.Series) -> dict[str, int]:
    y = np.asarray(y).astype(int)
    return {"n": int(len(y)), "fraud": int(y.sum()), "legit": int((y == 0).sum())}


def random_oversample(X, y, cfg: Mapping[str, Any], seed: int):
    strat = cfg["imbalance"]["random_oversample_strategy"]
    ros = RandomOverSampler(sampling_strategy=strat, random_state=seed)
    return ros.fit_resample(X, y)


def smote_oversample(X, y, cfg: Mapping[str, Any], seed: int):
    spec = cfg["imbalance"]["smote"]
    k = _safe_k(y, spec["k_neighbors"])
    sm = SMOTE(sampling_strategy=spec["sampling_strategy"], k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


def borderline_smote_oversample(X, y, cfg: Mapping[str, Any], seed: int):
    spec = cfg["imbalance"]["borderline_smote"]
    k = _safe_k(y, spec["k_neighbors"])
    sm = BorderlineSMOTE(
        sampling_strategy=spec["sampling_strategy"],
        k_neighbors=k,
        kind=spec.get("kind", "borderline-1"),
        random_state=seed,
    )
    return sm.fit_resample(X, y)


def generate_synthetic_minority(
    X: pd.DataFrame,
    y: pd.Series,
    n_candidates: int,
    k_neighbors: int,
    seed: int,
    method: str = "borderline_smote",
) -> pd.DataFrame:
    """Return *only* the newly generated fraud rows (not the original train).

    SMOTE interpolates minority neighbors in feature space. The raw
    candidates are statistically plausible interpolations but may not
    be explanation-consistent — that is EP-IAIL's job.
    """
    y = pd.Series(np.asarray(y).astype(int), index=getattr(y, "index", None))
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    target_pos = min(n_neg, n_pos + int(n_candidates))
    if target_pos <= n_pos:
        logger.warning("n_candidates produced no extras; returning empty frame")
        return pd.DataFrame(columns=X.columns)
    k = _safe_k(y, k_neighbors)
    try:
        estimator = {
            "smote": SMOTE(sampling_strategy={1: target_pos}, k_neighbors=k, random_state=seed),
            "borderline_smote": BorderlineSMOTE(
                sampling_strategy={1: target_pos}, k_neighbors=k, random_state=seed, kind="borderline-1"
            ),
        }[method]
        X_res, y_res = estimator.fit_resample(X, y)
    except ValueError as exc:
        logger.warning("%s failed (%s); falling back to vanilla SMOTE.", method, exc)
        estimator = SMOTE(sampling_strategy={1: target_pos}, k_neighbors=k, random_state=seed)
        X_res, y_res = estimator.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = np.asarray(y_res).astype(int)
    # imblearn concatenates original rows first, then synthetics.
    synthetic = X_res.iloc[len(X) :]
    synthetic = synthetic.loc[y_res[len(X) :] == 1].copy()
    logger.info("Generated %s synthetic fraud candidates via %s", f"{len(synthetic):,}", method)
    return synthetic.reset_index(drop=True)


def _safe_k(y, requested: int) -> int:
    n_pos = int(np.asarray(y).astype(int).sum())
    k = min(int(requested), max(n_pos - 1, 1))
    return max(k, 1)
