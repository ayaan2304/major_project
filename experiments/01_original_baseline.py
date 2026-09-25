"""
Phase 1 — Original-style baseline reproduction.

Protocol (given, reproduced — not claimed as novelty)
-----------------------------------------------------
- Drop step, type, nameOrig, nameDest
- Stratified 80/20 split
- LogReg, DT, RF, XGBoost, LightGBM, MLP, soft-voting ensemble
- Metrics including those that reveal accuracy's failure mode

Approximation (explicit)
------------------------
Random Forest is fitted on a train subset that keeps ALL fraud and at
most `compute.legit_cap` legitimate rows. Boosted trees and LogReg use
the full training split. MLP uses `models.mlp.max_samples`.
The test set is never subsampled.

python experiments/01_original_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_full
from src.ensemble import collect_probas, soft_vote
from src.evaluation import classification_report_dict, majority_baseline, metrics_table
from src.leakage_audit import subsample_train_keep_all_fraud
from src.models import baseline_model_zoo
from src.pipeline import eval_model, fit_named_model
from src.preprocessing import original_feature_matrix, stratified_split
from src.utils import get_logger, load_config, results_paths, save_json, save_table, set_global_seed
from src.visualization import plot_class_imbalance, plot_metric_bars


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    paths = results_paths(cfg)
    logger = get_logger("phase1", paths["logs"] / "phase1.log", cfg["logging"]["level"])

    logger.info("Loading full PaySim table for original-style baseline")
    df = load_full(cfg)
    X, y = original_feature_matrix(df, cfg)
    split = stratified_split(X, y, cfg, seed)
    X_tr, X_te = split["X_train"], split["X_test"]
    y_tr, y_te = split["y_train"], split["y_test"]

    # Persist split sizes (not millions of indices as JSON — parquet instead).
    pd.DataFrame({"index": X_tr.index, "split": "train"}).to_csv(
        paths["processed"] / "phase1_train_index_head.csv", index=False
    )
    save_json(split["meta"], paths["metrics"] / "phase1_split_meta.json")
    save_json({"feature_names": list(X.columns)}, paths["metrics"] / "phase1_feature_list.json")

    n_legit = int((y == 0).sum())
    n_fraud = int((y == 1).sum())
    plot_class_imbalance(
        n_legit,
        n_fraud,
        paths["figures"] / "phase1_class_imbalance.png",
        title="PaySim class imbalance (full file)",
    )

    maj = majority_baseline(y_te)
    maj["model"] = "majority_class_baseline"
    logger.info("Majority-class test accuracy=%.6f recall=%.6f", maj["accuracy"], maj["recall"])

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    joblib.dump(scaler, paths["models"] / "phase1_scaler.joblib")

    zoo = baseline_model_zoo(cfg, seed, y_tr)
    rows = [maj]
    val_probas_test = {}

    rf_X, rf_y, rf_meta = subsample_train_keep_all_fraud(
        X_tr, y_tr, cfg["compute"]["legit_cap"], seed, cfg["compute"]["keep_all_fraud"]
    )
    save_json(rf_meta, paths["metrics"] / "phase1_rf_subsample_meta.json")

    scaled_models = {"logistic_regression", "mlp"}
    capped_models = {"random_forest"}

    for name, model in zoo.items():
        if name in scaled_models:
            Xt, Xe, yt = X_tr_s, X_te_s, y_tr
            if name == "mlp":
                Xt, yt = X_tr_s, y_tr
        elif name in capped_models:
            Xt, Xe, yt = rf_X, X_te, rf_y
        else:
            Xt, Xe, yt = X_tr, X_te, y_tr
        model, extra = fit_named_model(name, model, Xt, yt, cfg, seed)
        metrics, proba = eval_model(name, model, Xe if name not in scaled_models else X_te_s, y_te)
        metrics.update({f"train_{k}": v for k, v in extra.items()})
        rows.append(metrics)
        val_probas_test[name] = proba
        joblib.dump(model, paths["models"] / f"phase1_{name}.joblib")
        logger.info("%s F1=%.4f PR-AUC=%.4f recall=%.4f", name, metrics["f1"], metrics["pr_auc"], metrics["recall"])

    vote_p = soft_vote(val_probas_test)
    vote = classification_report_dict(y_te, vote_p)
    vote["model"] = "soft_voting_ensemble"
    rows.append(vote)

    table = metrics_table(rows)
    save_table(table, paths["tables"] / "phase1_original_baseline_metrics.csv")
    save_json(rows, paths["metrics"] / "phase1_original_baseline_metrics.json")
    plot_metric_bars(
        table,
        ["precision", "recall", "f1", "pr_auc", "mcc", "roc_auc"],
        paths["figures"] / "phase1_metrics.png",
        title="Original-style baseline (stratified 80/20) — not accuracy",
    )
    print(table[["model", "accuracy", "precision", "recall", "f1", "pr_auc", "mcc", "FNR"]].to_string(index=False))
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
