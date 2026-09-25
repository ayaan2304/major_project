"""
Causal behavioral feature engineering.

What / why
----------
Raw PaySim columns are accounting snapshots. Fraud often appears as
account draining, empty destinations, and bursty TRANSFER/CASH_OUT
pairs. We add features that a production system could compute at
transaction time, without using future rows or raw string IDs as
model inputs.

Identifiers (`nameOrig`, `nameDest`) are used only to *group* history,
then dropped.

Leakage risks
-------------
- Sort by `step` (and a stable tie-breaker) before expanding windows.
- Use shift/cumcount — never rolling statistics that include the
  current row's future neighbors.
- Do not use test labels. Features are label-free.
- After engineering, drop raw IDs.

Approximation
-------------
Global causal history on the full file is valid: a test-hour feature
may use earlier train/val events (those events already happened).
It must not use later test events. Groupby + shift enforces this
if the frame is globally sorted by time.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("feature_engineering")

FEATURE_DECISION_LOG: list[dict[str, str]] = []


def _eps(cfg: Mapping[str, Any]) -> float:
    return float(cfg["data"]["eps"])


def add_point_in_time_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Row-local features: no groupby required, no leakage possible."""
    eps = _eps(cfg)
    out = df.copy()
    amt = out["amount"].astype(float)
    old_o = out["oldbalanceOrg"].astype(float)
    new_o = out["newbalanceOrig"].astype(float)
    old_d = out["oldbalanceDest"].astype(float)
    new_d = out["newbalanceDest"].astype(float)

    out["log_amount"] = np.log(amt + eps)
    # How much of the origin balance is being moved.
    out["orig_amount_to_oldbalance"] = amt / (old_o + eps)
    out["dest_amount_to_oldbalance"] = amt / (old_d + eps)
    # Accounting residuals (PaySim often zeros dest for merchants).
    out["orig_balance_error"] = old_o - amt - new_o
    out["dest_balance_error"] = old_d + amt - new_d
    out["orig_drained"] = ((new_o <= eps) & (old_o > eps)).astype(np.int8)
    out["orig_refill_ratio"] = (new_o - old_o) / (old_o + eps)
    out["dest_refill_ratio"] = (new_d - old_d) / (old_d + eps)
    out["zero_orig_before"] = (old_o <= eps).astype(np.int8)
    out["zero_dest_before"] = (old_d <= eps).astype(np.int8)
    out["abnormal_orig_change"] = (out["orig_balance_error"].abs() > 1.0).astype(np.int8)
    # PaySim step is an hour index in a 30-day simulation (max typically 743).
    out["hour_of_day"] = (out["step"] % 24).astype(np.int16)
    out["day_of_month"] = (out["step"] // 24).astype(np.int16)
    return out


def add_causal_account_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per-account history using only earlier transactions (same or previous step).

    Within a step, PaySim order is the file order after a stable sort.
    We sort by step, then original row order, so "previous" is well-defined.
    """
    orig_col = cfg["data"]["orig_col"]
    dest_col = cfg["data"]["dest_col"]
    out = df.sort_values(["step", "_row_id"], kind="mergesort").copy()

    g_o = out.groupby(orig_col, sort=False)
    out["orig_prior_tx_count"] = g_o.cumcount().astype(np.int32)
    out["orig_time_since_prev"] = (out["step"] - g_o["step"].shift(1)).fillna(out["step"])
    out["orig_prev_amount"] = g_o["amount"].shift(1).fillna(0.0)
    # Expanding mean of *previous* amounts: (cumsum - amount) / count.
    out["orig_prior_amount_sum"] = (g_o["amount"].cumsum() - out["amount"]).clip(lower=0.0)
    out["orig_prior_amount_mean"] = out["orig_prior_amount_sum"] / (out["orig_prior_tx_count"] + 1e-12)

    g_d = out.groupby(dest_col, sort=False)
    out["dest_prior_tx_count"] = g_d.cumcount().astype(np.int32)
    out["dest_novelty"] = (out["dest_prior_tx_count"] == 0).astype(np.int8)
    out["dest_time_since_prev"] = (out["step"] - g_d["step"].shift(1)).fillna(out["step"])
    out["dest_prior_amount_sum"] = (g_d["amount"].cumsum() - out["amount"]).clip(lower=0.0)
    out["dest_prior_amount_mean"] = out["dest_prior_amount_sum"] / (out["dest_prior_tx_count"] + 1e-12)
    return out


def add_type_indicators(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """One-hot `type`. Categories are known from PaySim (not fitted on test labels)."""
    known = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    out = df.copy()
    t = out[cfg["data"]["type_col"]].astype(str)
    for cat in known:
        out[f"type_{cat}"] = (t == cat).astype(np.int8)
    return out


def engineer_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Full causal feature pipeline. Drops raw identifiers at the end."""
    logger.info("Engineering features on %s rows", f"{len(df):,}")
    work = df.copy()
    work["_row_id"] = np.arange(len(work), dtype=np.int64)
    work = add_point_in_time_features(work, cfg)
    work = add_causal_account_features(work, cfg)
    work = add_type_indicators(work, cfg)
    drop_ids = [cfg["data"]["orig_col"], cfg["data"]["dest_col"], "_row_id"]
    work = work.drop(columns=drop_ids)
    # Keep `type` dropped after indicators; keep `step` for splitting / drift.
    work = work.drop(columns=[cfg["data"]["type_col"]])
    return work


def model_feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> list[str]:
    """Numeric model inputs: everything except target and split helpers."""
    ban = {cfg["data"]["target"], "split"}
    cols = [c for c in df.columns if c not in ban]
    return cols


def univariate_separation(train: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Train-only AUROC-style rank separation via Mann-Whitney-friendly AUC.

    For each feature, compute P(score_fraud > score_legit) with a cheap
    rank AUC. Also report variance; near-zero-variance features are flagged.
    """
    from sklearn.metrics import roc_auc_score

    target = cfg["data"]["target"]
    y = train[target].to_numpy()
    rows = []
    feat_cols = [c for c in model_feature_columns(train, cfg) if c != cfg["data"]["time_col"]]
    for col in feat_cols:
        x = train[col].to_numpy(dtype=float)
        var = float(np.nanvar(x))
        nzv = var < 1e-12
        try:
            auc = float(roc_auc_score(y, x))
            auc = max(auc, 1.0 - auc)  # orientation-invariant separation
        except ValueError:
            auc = float("nan")
        rows.append({"feature": col, "rank_auc_sep": auc, "variance": var, "near_zero_variance": nzv})
    out = pd.DataFrame(rows).sort_values("rank_auc_sep", ascending=False)
    return out
