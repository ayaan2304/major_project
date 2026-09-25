"""
Explanation-preserving filter for synthetic fraud (EP-IAIL).

What / why
----------
Accept a SMOTE candidate iff it is (1) explanation-similar to the
genuine-fraud SHAP prototype and (2) close to the genuine-fraud
cloud in feature space.

Filtering rule (plain English)
------------------------------
Keep x* if similarity(SHAP(x*), proto) >= t_sim AND
distance(x*, fraud_cloud) <= t_plaus.

Thresholds are percentiles of *genuine train fraud* scores (and a
small validation grid over those percentiles). Test is never used.

Leakage risks
-------------
- Candidates from train SMOTE only.
- Distances/covariance estimated on train fraud features only.
- Percentile grid evaluated on validation predictions of a model
  trained on train+filtered-synth — validation labels guide the
  choice of t_sim/t_plaus, then freeze for test.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.linalg import pinv

from src.shap_prototype import combined_similarity, tree_shap_values
from src.utils import get_logger

logger = get_logger("explanation_filter")


def mahalanobis_distance(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    delta = X - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta) + 1e-12)


def euclidean_distance(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - mean, axis=1)


def fit_plausibility(X_fraud: np.ndarray, metric: str) -> dict[str, Any]:
    mean = X_fraud.mean(axis=0)
    if metric == "euclidean":
        return {"metric": metric, "mean": mean, "inv_cov": None}
    cov = np.cov(X_fraud, rowvar=False)
    # Shrink slightly so singular covariance (duplicate features) does not explode.
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = pinv(cov)
    return {"metric": metric, "mean": mean, "inv_cov": inv_cov}


def score_plausibility(X: np.ndarray, fitted: Mapping[str, Any]) -> np.ndarray:
    if fitted["metric"] == "euclidean":
        return euclidean_distance(X, fitted["mean"])
    return mahalanobis_distance(X, fitted["mean"], fitted["inv_cov"])


def genuine_score_reference(
    shap_fraud: np.ndarray,
    X_fraud: np.ndarray,
    proto: np.ndarray,
    plaus_fitted: Mapping[str, Any],
    cosine_weight: float,
    magnitude_weight: float,
) -> dict[str, np.ndarray]:
    """Distributions of genuine train-fraud similarity and plausibility."""
    sim = combined_similarity(shap_fraud, proto, cosine_weight, magnitude_weight)
    dist = score_plausibility(X_fraud, plaus_fitted)
    return {"similarity": sim, "plausibility": dist}


def thresholds_from_percentiles(
    genuine: Mapping[str, np.ndarray],
    sim_pct: float,
    plaus_pct: float,
) -> dict[str, float]:
    """t_sim = low percentile of genuine similarity (must be at least this aligned).

    t_plaus = high percentile of genuine distance (must not be farther than this).
    """
    t_sim = float(np.percentile(genuine["similarity"], sim_pct))
    t_plaus = float(np.percentile(genuine["plausibility"], plaus_pct))
    return {"t_sim": t_sim, "t_plaus": t_plaus, "sim_pct": sim_pct, "plaus_pct": plaus_pct}


def apply_filter(
    candidate_sim: np.ndarray,
    candidate_plaus: np.ndarray,
    t_sim: float,
    t_plaus: float,
) -> np.ndarray:
    return (candidate_sim >= t_sim) & (candidate_plaus <= t_plaus)


def filter_candidates(
    candidates: pd.DataFrame,
    model,
    background,
    proto: np.ndarray,
    plaus_fitted: Mapping[str, Any],
    t_sim: float,
    t_plaus: float,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Score and filter synthetic rows. Returns accepted frame + audit stats."""
    ep = cfg["ep_iail"]
    sv = tree_shap_values(model, candidates, background, cfg["compute"]["shap_batch_size"])
    sim = combined_similarity(sv, proto, ep["cosine_weight"], ep["magnitude_weight"])
    Xn = candidates.to_numpy(dtype=float)
    dist = score_plausibility(Xn, plaus_fitted)
    mask = apply_filter(sim, dist, t_sim, t_plaus)
    accepted = candidates.loc[mask].reset_index(drop=True)
    stats = {
        "n_candidates": int(len(candidates)),
        "n_accepted": int(mask.sum()),
        "n_rejected": int((~mask).sum()),
        "acceptance_rate": float(mask.mean()) if len(mask) else 0.0,
        "t_sim": float(t_sim),
        "t_plaus": float(t_plaus),
        "candidate_similarity": sim,
        "candidate_plausibility": dist,
        "accepted_mask": mask,
        "accepted_shap": sv[mask] if mask.any() else np.zeros((0, sv.shape[1])),
        "rejected_shap": sv[~mask] if (~mask).any() else np.zeros((0, sv.shape[1])),
    }
    logger.info(
        "EP-IAIL filter: %s generated, %s accepted (%.2f%%)",
        stats["n_candidates"],
        stats["n_accepted"],
        100 * stats["acceptance_rate"],
    )
    return {"accepted": accepted, "stats": stats}
