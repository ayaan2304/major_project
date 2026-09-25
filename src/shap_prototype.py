"""
Provisional model + SHAP fraud prototype (core of EP-IAIL).

What / why
----------
Standard SMOTE asks only "does this point sit between fraud neighbors?"
EP-IAIL additionally asks "does a tree model *explain* this point like
it explains genuine fraud?"

We use TreeSHAP (not KernelSHAP) because:
- the provisional model is a GBDT (exact TreeSHAP);
- KernelSHAP on 6M rows is computationally unjustifiable;
- TreeSHAP is consistent for tree ensembles.

Approximation (explicit)
------------------------
Background set is a train subsample of size `shap_background_n`.
SHAP for candidates is computed in batches. Background/explain sets
are drawn from TRAIN only.

Prototype strategies compared on a train-fraud holdout (never val/test):
mean, median, trimmed mean of SHAP vectors. We pick the strategy that
maximises mean cosine similarity of held-out genuine fraud to the
prototype estimated on the complement.

Leakage risks
-------------
- Fit provisional model on train (optionally resampled train) only.
- Prototype from genuine TRAIN fraud SHAP only.
- Do not compute prototype from synthetic points.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import shap
from scipy.stats import trim_mean
from sklearn.metrics.pairwise import cosine_similarity

from src.models import make_lightgbm
from src.utils import get_logger

logger = get_logger("shap_prototype")


def fit_provisional_tree(X_train, y_train, cfg: Mapping[str, Any], seed: int):
    """LightGBM is the default TreeSHAP source: fast, native trees, strong on tabular fraud."""
    model = make_lightgbm(cfg, seed, y_train=y_train, use_spw=True)
    model.fit(X_train, y_train)
    return model


def _as_ndarray(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def tree_shap_values(
    model,
    X,
    background,
    batch_size: int,
) -> np.ndarray:
    """Return SHAP values for the positive class, shape (n_samples, n_features).

    Prefer interventional TreeSHAP with a train background. If the installed
    SHAP/model pair rejects probability output, fall back to path-dependent
    TreeSHAP and log that approximation.
    """
    try:
        explainer = shap.TreeExplainer(
            model,
            data=_as_ndarray(background),
            feature_perturbation="interventional",
            model_output="probability",
        )
    except Exception as exc:  # noqa: BLE001 — environment-dependent SHAP API
        logger.warning("Interventional probability TreeSHAP unavailable (%s); using path-dependent TreeSHAP.", exc)
        explainer = shap.TreeExplainer(model)
    Xn = _as_ndarray(X)
    chunks = []
    for start in range(0, len(Xn), batch_size):
        sl = Xn[start : start + batch_size]
        sv = explainer.shap_values(sl)
        if isinstance(sv, list):
            sv = sv[1]
        chunks.append(np.asarray(sv))
        logger.info("TreeSHAP batch %s–%s / %s", start, start + len(sl), len(Xn))
    out = np.vstack(chunks) if chunks else np.zeros((0, Xn.shape[1]))
    if out.ndim == 3:
        # shap>=0.45 sometimes returns (n, f, n_classes)
        out = out[:, :, -1]
    return out


def sample_background(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(int(n), len(X))
    return X.sample(n=n, random_state=seed)


def build_prototypes(shap_matrix: np.ndarray, proportiontocut: float) -> dict[str, np.ndarray]:
    """Compare mean / median / trimmed-mean SHAP vectors."""
    return {
        "mean": shap_matrix.mean(axis=0),
        "median": np.median(shap_matrix, axis=0),
        "trimmed_mean": trim_mean(shap_matrix, proportiontocut=proportiontocut, axis=0),
    }


def cosine_to_proto(shap_matrix: np.ndarray, proto: np.ndarray) -> np.ndarray:
    proto = proto.reshape(1, -1)
    return cosine_similarity(shap_matrix, proto).ravel()


def select_prototype_strategy(
    shap_fraud: np.ndarray,
    strategies: Sequence[str],
    proportiontocut: float,
    seed: int,
) -> dict[str, Any]:
    """Train-only holdout: 70% of genuine fraud SHAP builds proto, 30% scores it."""
    rng = np.random.RandomState(seed)
    n = len(shap_fraud)
    if n < 20:
        proto_all = build_prototypes(shap_fraud, proportiontocut)
        chosen = "median" if "median" in strategies else strategies[0]
        return {
            "chosen": chosen,
            "scores": {},
            "prototypes": proto_all,
            "note": "Too few fraud SHAP rows for holdout; defaulted.",
        }
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    build_i, hold_i = idx[:cut], idx[cut:]
    prototypes = build_prototypes(shap_fraud[build_i], proportiontocut)
    scores = {}
    for name in strategies:
        sim = cosine_to_proto(shap_fraud[hold_i], prototypes[name])
        scores[name] = float(np.mean(sim))
    chosen = max(scores, key=scores.get)
    # Rebuild chosen prototype on ALL genuine train-fraud SHAP (still train only).
    full = build_prototypes(shap_fraud, proportiontocut)
    logger.info("Prototype holdout cosine means: %s; chosen=%s", scores, chosen)
    return {"chosen": chosen, "scores": scores, "prototypes": full}


def combined_similarity(
    shap_matrix: np.ndarray,
    proto: np.ndarray,
    cosine_weight: float,
    magnitude_weight: float,
) -> np.ndarray:
    """Direction (cosine) plus a magnitude term.

    Cosine ignores vector length, so a tiny SHAP vector can look 'aligned'
    with fraud even if the model barely attributes anything. The magnitude
    term is 1 - | ||s|| - ||p|| | / (||s|| + ||p|| + eps), i.e. relative
    L2 closeness of attribution strength. Both terms lie in roughly [0, 1]
    (cosine can be negative; we clip to [0, 1] for a conservative filter).
    """
    cos = cosine_to_proto(shap_matrix, proto)
    cos_clipped = np.clip(cos, 0.0, 1.0)
    mag_s = np.linalg.norm(shap_matrix, axis=1)
    mag_p = np.linalg.norm(proto)
    mag_term = 1.0 - np.abs(mag_s - mag_p) / (mag_s + mag_p + 1e-12)
    mag_term = np.clip(mag_term, 0.0, 1.0)
    return cosine_weight * cos_clipped + magnitude_weight * mag_term
