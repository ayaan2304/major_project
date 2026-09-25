"""
Global/local SHAP reporting and explanation-fidelity via top-k ablation.

Fidelity (H5)
-------------
For each instance, zero-out the top-k |SHAP| features vs k random
features and measure the drop in predicted fraud probability.
A faithful explanation should hurt more when top-k features are removed.

Leakage risks
-------------
Explain test instances with a model trained on train. Ablation is an
evaluation of explanations, not a training step.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def global_mean_abs_shap(shap_values: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    mag = np.mean(np.abs(shap_values), axis=0)
    return pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mag}).sort_values(
        "mean_abs_shap", ascending=False
    )


def local_top_features(shap_row: np.ndarray, feature_names: Sequence[str], k: int = 8) -> pd.DataFrame:
    order = np.argsort(-np.abs(shap_row))[:k]
    return pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "shap": shap_row[order],
            "abs_shap": np.abs(shap_row[order]),
        }
    )


def ablation_fidelity(
    model,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    ks: Sequence[int],
    seed: int,
) -> pd.DataFrame:
    """Compare probability drop for top-k SHAP vs random-k ablation."""
    rng = np.random.RandomState(seed)
    base = model.predict_proba(X)[:, 1]
    rows = []
    n_feat = X.shape[1]
    names = list(X.columns)
    Xn = X.to_numpy(dtype=float)
    for i in range(len(X)):
        mag_order = np.argsort(-np.abs(shap_values[i]))
        for k in ks:
            k = min(int(k), n_feat)
            top_idx = mag_order[:k]
            rand_idx = rng.choice(n_feat, size=k, replace=False)
            x_top = Xn[i].copy()
            x_rnd = Xn[i].copy()
            x_top[top_idx] = 0.0
            x_rnd[rand_idx] = 0.0
            # Keep as 1-row DataFrame to satisfy feature names in sklearn/xgb.
            p_top = model.predict_proba(pd.DataFrame([x_top], columns=names))[0, 1]
            p_rnd = model.predict_proba(pd.DataFrame([x_rnd], columns=names))[0, 1]
            rows.append(
                {
                    "row": i,
                    "k": k,
                    "base_p": float(base[i]),
                    "drop_top": float(base[i] - p_top),
                    "drop_random": float(base[i] - p_rnd),
                }
            )
    return pd.DataFrame(rows)
