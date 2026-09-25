"""
Phase 2 — Temporal split + causal behavioral features.

python experiments/02_temporal_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_full
from src.feature_engineering import engineer_features, model_feature_columns, univariate_separation
from src.leakage_audit import audit_temporal_pipeline
from src.temporal_split import temporal_frames
from src.utils import get_logger, load_config, results_paths, save_json, save_table, set_global_seed
from src.visualization import plot_class_imbalance


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    paths = results_paths(cfg)
    logger = get_logger("phase2", paths["logs"] / "phase2.log", cfg["logging"]["level"])

    df = load_full(cfg)
    feat = engineer_features(df, cfg)
    split = temporal_frames(feat, cfg)
    parts = split["parts"]
    meta = split["meta"]

    audit = audit_temporal_pipeline(
        {
            "train_index": parts["train"].index.to_numpy(),
            "val_index": parts["val"].index.to_numpy(),
            "test_index": parts["test"].index.to_numpy(),
            "train_steps": parts["train"][cfg["data"]["time_col"]].to_numpy(),
            "val_steps": parts["val"][cfg["data"]["time_col"]].to_numpy(),
            "test_steps": parts["test"][cfg["data"]["time_col"]].to_numpy(),
            "prototype_source": "train_fraud",
            "synthetic_from": "train",
            "thresholds_from": "val",
        }
    )
    save_json({**meta, **audit}, paths["metrics"] / "phase2_temporal_split.json")

    feat_cols = model_feature_columns(feat, cfg)
    save_json({"feature_names": feat_cols}, paths["metrics"] / "phase2_feature_list.json")

    # Persist processed splits (parquet is compact enough for reuse in later phases).
    for name, frame in parts.items():
        out = paths["processed"] / f"phase2_{name}.parquet"
        frame.to_parquet(out, index=True)
        logger.info("Wrote %s (%s rows)", out, f"{len(frame):,}")

    sep = univariate_separation(parts["train"], cfg)
    save_table(sep, paths["tables"] / "phase2_univariate_separation.csv")

    # Correlation on a train subsample to keep the matrix tractable (documented).
    rng = np.random.RandomState(seed)
    n_corr = min(80000, len(parts["train"]))
    corr_idx = rng.choice(len(parts["train"]), size=n_corr, replace=False)
    corr_cols = [c for c in feat_cols if c != cfg["data"]["time_col"]]
    corr = parts["train"].iloc[corr_idx][corr_cols].corr()
    corr.to_csv(paths["tables"] / "phase2_correlation_sample.csv")

    decisions = [
        {
            "feature_group": "log_amount / balance ratios / drain flags",
            "decision": "keep",
            "why": "Row-local, no identifiers, capture known PaySim drain-and-cash-out pattern.",
        },
        {
            "feature_group": "orig/dest prior counts, time-since-prev, dest_novelty",
            "decision": "keep",
            "why": "Causal groupby+shift; uses IDs only as keys then drops them.",
        },
        {
            "feature_group": "raw nameOrig/nameDest",
            "decision": "drop",
            "why": "High-cardinality identifiers; leakage and non-generalizable.",
        },
        {
            "feature_group": "near-zero-variance columns",
            "decision": "flag in table; drop if near_zero_variance True",
            "why": "Unstable for trees and linear models.",
        },
        {
            "feature_group": "correlation subsample (80k train rows)",
            "decision": "documented approximation",
            "why": "Full 4M x 30 correlation matrix is redundant for multicollinearity screening.",
        },
    ]
    save_json(decisions, paths["metrics"] / "phase2_feature_decisions.json")

    plot_class_imbalance(
        int((parts["train"][cfg["data"]["target"]] == 0).sum()),
        int(parts["train"][cfg["data"]["target"]].sum()),
        paths["figures"] / "phase2_train_imbalance.png",
        title="Train split class imbalance (temporal)",
    )
    nzv = sep[sep["near_zero_variance"]]
    logger.info("Near-zero-variance features: %s", list(nzv["feature"]))
    print("Phase 2 complete. Split meta:", meta["n"])


if __name__ == "__main__":
    main()
