"""
Phase 3 — EP-IAIL vs conventional imbalance handling.

Compares, on the temporal protocol:
  none | class_weight | random_oversample | smote | borderline_smote | ep_iail

Thresholds for EP-IAIL are grid-searched on validation PR-AUC of LightGBM.
Test is touched once per method after freeze.

python experiments/03_ep_iail.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import classification_report_dict, metrics_table
from src.explanation_filter import (
    filter_candidates,
    fit_plausibility,
    genuine_score_reference,
    thresholds_from_percentiles,
)
from src.feature_engineering import model_feature_columns
from src.imbalance import (
    borderline_smote_oversample,
    generate_synthetic_minority,
    random_oversample,
    smote_oversample,
)
from src.leakage_audit import subsample_train_keep_all_fraud
from src.models import make_lightgbm
from src.pipeline import eval_model, fit_named_model
from src.shap_prototype import (
    combined_similarity,
    fit_provisional_tree,
    sample_background,
    select_prototype_strategy,
    tree_shap_values,
)
from src.utils import get_logger, load_config, results_paths, save_json, save_table, set_global_seed
from src.visualization import plot_metric_bars, plot_score_hist


def _xy(frame: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg["data"]["target"]
    cols = [c for c in model_feature_columns(frame, cfg) if c != cfg["data"]["time_col"]]
    return frame[cols], frame[target].astype(int)


def _prepare_train(X, y, cfg, seed, method: str):
    """Return (X_fit, y_fit, extra_meta). Resampling is train-only."""
    Xc, yc, cap_meta = subsample_train_keep_all_fraud(
        X, y, cfg["compute"]["legit_cap"], seed, cfg["compute"]["keep_all_fraud"]
    )
    extra = {"cap_meta": cap_meta, "method": method}
    if method == "none":
        model = make_lightgbm(cfg, seed, y_train=yc, use_spw=False)
        return Xc, yc, extra, model
    if method == "class_weight":
        model = make_lightgbm(cfg, seed, y_train=yc, use_spw=True)
        return Xc, yc, extra, model
    if method == "random_oversample":
        Xs, ys = random_oversample(Xc, yc, cfg, seed)
        model = make_lightgbm(cfg, seed, y_train=ys, use_spw=False)
        extra["n_after_resample"] = int(len(ys))
        return Xs, ys, extra, model
    if method == "smote":
        Xs, ys = smote_oversample(Xc, yc, cfg, seed)
        model = make_lightgbm(cfg, seed, y_train=ys, use_spw=False)
        extra["n_after_resample"] = int(len(ys))
        return Xs, ys, extra, model
    if method == "borderline_smote":
        Xs, ys = borderline_smote_oversample(Xc, yc, cfg, seed)
        model = make_lightgbm(cfg, seed, y_train=ys, use_spw=False)
        extra["n_after_resample"] = int(len(ys))
        return Xs, ys, extra, model
    raise ValueError(method)


def run_ep_iail(X_tr, y_tr, X_va, y_va, cfg, seed, logger, paths):
    ep = cfg["ep_iail"]
    Xc, yc, cap_meta = subsample_train_keep_all_fraud(
        X_tr, y_tr, cfg["compute"]["legit_cap"], seed, cfg["compute"]["keep_all_fraud"]
    )
    prov = fit_provisional_tree(Xc, yc, cfg, seed)
    joblib.dump(prov, paths["models"] / "phase3_provisional_lgbm.joblib")

    bg = sample_background(Xc, cfg["compute"]["shap_background_n"], seed)
    fraud_X = Xc.loc[yc.to_numpy() == 1]
    # SHAP on ALL train-fraud (after cap, this is still every fraud kept).
    shap_fraud = tree_shap_values(prov, fraud_X, bg, cfg["compute"]["shap_batch_size"])
    np.save(paths["shap"] / "phase3_train_fraud_shap.npy", shap_fraud)

    proto_info = select_prototype_strategy(
        shap_fraud, ep["prototype_candidates"], ep["trimmed_mean_proportiontocut"], seed
    )
    chosen = proto_info["chosen"]
    proto = proto_info["prototypes"][chosen]
    save_json(
        {"chosen": chosen, "holdout_cosine_means": proto_info["scores"], "prototype": proto.tolist()},
        paths["metrics"] / "phase3_shap_prototype.json",
    )

    plaus = fit_plausibility(fraud_X.to_numpy(dtype=float), ep["plausibility_metric"])
    genuine = genuine_score_reference(
        shap_fraud,
        fraud_X.to_numpy(dtype=float),
        proto,
        plaus,
        ep["cosine_weight"],
        ep["magnitude_weight"],
    )
    plot_score_hist(
        genuine["similarity"],
        paths["figures"] / "phase3_genuine_similarity.png",
        "Genuine train-fraud explanation similarity",
        "combined similarity",
    )
    plot_score_hist(
        genuine["plausibility"],
        paths["figures"] / "phase3_genuine_plausibility.png",
        "Genuine train-fraud plausibility distance",
        ep["plausibility_metric"],
    )

    candidates = generate_synthetic_minority(
        Xc, yc, ep["n_candidates"], ep["k_neighbors"], seed, method=ep["generator"]
    )
    joblib.dump({"columns": list(Xc.columns)}, paths["models"] / "phase3_feature_columns.joblib")

    # Score every candidate once (TreeSHAP is the expensive step), then only
    # re-threshold. Recomputing SHAP inside the percentile grid would leak no
    # data but would multiply runtime by |grid|.
    scored = filter_candidates(
        candidates,
        prov,
        bg,
        proto,
        plaus,
        t_sim=float(np.min(genuine["similarity"])),
        t_plaus=float(np.max(genuine["plausibility"])),
        cfg=cfg,
    )
    cand_sim = scored["stats"]["candidate_similarity"]
    cand_plaus = scored["stats"]["candidate_plausibility"]

    best = None
    grid_rows = []
    from src.explanation_filter import apply_filter

    for sim_pct in ep["similarity_percentile_grid"]:
        for plaus_pct in ep["plausibility_percentile_grid"]:
            thr = thresholds_from_percentiles(genuine, sim_pct, plaus_pct)
            mask = apply_filter(cand_sim, cand_plaus, thr["t_sim"], thr["t_plaus"])
            accepted = candidates.loc[mask].reset_index(drop=True)
            X_fit = pd.concat([Xc, accepted], axis=0, ignore_index=True)
            y_fit = pd.concat(
                [yc.reset_index(drop=True), pd.Series(np.ones(len(accepted), dtype=int))],
                ignore_index=True,
            )
            model = make_lightgbm(cfg, seed, y_train=y_fit, use_spw=False)
            model.fit(X_fit, y_fit)
            p_va = model.predict_proba(X_va)[:, 1]
            m = classification_report_dict(y_va, p_va)
            row = {
                "sim_pct": sim_pct,
                "plaus_pct": plaus_pct,
                "t_sim": thr["t_sim"],
                "t_plaus": thr["t_plaus"],
                "n_accepted": int(mask.sum()),
                "acceptance_rate": float(mask.mean()) if len(mask) else 0.0,
                "val_pr_auc": m["pr_auc"],
                "val_f1": m["f1"],
                "val_recall": m["recall"],
            }
            grid_rows.append(row)
            logger.info("Grid sim_p=%s plaus_p=%s -> val PR-AUC=%.4f accepted=%s", sim_pct, plaus_pct, m["pr_auc"], row["n_accepted"])
            if best is None or row["val_pr_auc"] > best["val_pr_auc"]:
                filt_stats = {
                    "n_candidates": int(len(candidates)),
                    "n_accepted": row["n_accepted"],
                    "n_rejected": int((~mask).sum()),
                    "acceptance_rate": row["acceptance_rate"],
                    "t_sim": thr["t_sim"],
                    "t_plaus": thr["t_plaus"],
                    "candidate_similarity": cand_sim,
                    "candidate_plausibility": cand_plaus,
                    "accepted_mask": mask,
                }
                best = {**row, "model": model, "X_fit": X_fit, "y_fit": y_fit, "filt": {"accepted": accepted, "stats": filt_stats}, "thr": thr}

    save_table(pd.DataFrame(grid_rows), paths["tables"] / "phase3_ep_iail_threshold_grid.csv")
    # Persist filter distributions for the winning threshold.
    stats = best["filt"]["stats"]
    plot_score_hist(
        stats["candidate_similarity"],
        paths["figures"] / "phase3_candidate_similarity.png",
        "Synthetic candidate explanation similarity",
        "combined similarity",
        vline=best["thr"]["t_sim"],
    )
    plot_score_hist(
        stats["candidate_plausibility"],
        paths["figures"] / "phase3_candidate_plausibility.png",
        "Synthetic candidate plausibility distance",
        ep["plausibility_metric"],
        vline=best["thr"]["t_plaus"],
    )
    audit = {
        "n_candidates": stats["n_candidates"],
        "n_accepted": stats["n_accepted"],
        "n_rejected": stats["n_rejected"],
        "acceptance_rate": stats["acceptance_rate"],
        "chosen_prototype": chosen,
        "thresholds": best["thr"],
        "val_pr_auc": best["val_pr_auc"],
        "thresholds_from": "val",
        "synthetic_from": "train",
        "prototype_source": "train_fraud",
        "cap_meta": cap_meta,
    }
    # Drop huge arrays before JSON.
    save_json(audit, paths["metrics"] / "phase3_ep_iail_filter_audit.json")
    np.save(paths["shap"] / "phase3_candidate_similarity.npy", stats["candidate_similarity"])
    np.save(paths["shap"] / "phase3_candidate_plausibility.npy", stats["candidate_plausibility"])
    joblib.dump(best["model"], paths["models"] / "phase3_ep_iail_lgbm.joblib")
    best["X_fit"].to_parquet(paths["processed"] / "phase3_ep_iail_X_fit.parquet", index=False)
    pd.DataFrame({"y": best["y_fit"].astype(int)}).to_parquet(
        paths["processed"] / "phase3_ep_iail_y_fit.parquet", index=False
    )
    return best["model"], audit


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    paths = results_paths(cfg)
    logger = get_logger("phase3", paths["logs"] / "phase3.log", cfg["logging"]["level"])

    train = pd.read_parquet(paths["processed"] / "phase2_train.parquet")
    val = pd.read_parquet(paths["processed"] / "phase2_val.parquet")
    test = pd.read_parquet(paths["processed"] / "phase2_test.parquet")
    X_tr, y_tr = _xy(train, cfg)
    X_va, y_va = _xy(val, cfg)
    X_te, y_te = _xy(test, cfg)

    methods = ["none", "class_weight", "random_oversample", "smote", "borderline_smote"]
    rows = []
    for method in methods:
        logger.info("=== Imbalance method: %s ===", method)
        X_fit, y_fit, extra, model = _prepare_train(X_tr, y_tr, cfg, seed, method)
        model, fit_meta = fit_named_model(method, model, X_fit, y_fit, cfg, seed)
        m_va, _ = eval_model(method, model, X_va, y_va)
        m_te, _ = eval_model(method, model, X_te, y_te)
        row = {"method": method, **{f"val_{k}": v for k, v in m_va.items()}, **{f"test_{k}": v for k, v in m_te.items()}}
        row["n_fit"] = int(len(y_fit))
        rows.append(row)
        joblib.dump(model, paths["models"] / f"phase3_{method}_lgbm.joblib")
        save_json(extra, paths["metrics"] / f"phase3_{method}_meta.json")

    logger.info("=== EP-IAIL ===")
    ep_model, ep_audit = run_ep_iail(X_tr, y_tr, X_va, y_va, cfg, seed, logger, paths)
    m_va, _ = eval_model("ep_iail", ep_model, X_va, y_va)
    m_te, _ = eval_model("ep_iail", ep_model, X_te, y_te)
    rows.append(
        {
            "method": "ep_iail",
            **{f"val_{k}": v for k, v in m_va.items()},
            **{f"test_{k}": v for k, v in m_te.items()},
            "n_fit": "see filter audit",
        }
    )

    table = pd.DataFrame(rows)
    save_table(table, paths["tables"] / "phase3_imbalance_comparison.csv")
    save_json(rows, paths["metrics"] / "phase3_imbalance_comparison.json")
    # Plot test minority metrics
    plot_df = table.rename(columns={"method": "model"})
    plot_metric_bars(
        plot_df,
        ["test_precision", "test_recall", "test_f1", "test_pr_auc", "test_mcc"],
        paths["figures"] / "phase3_imbalance_comparison.png",
        title="Temporal test: imbalance strategies (LightGBM)",
    )
    cols = ["method", "test_precision", "test_recall", "test_f1", "test_pr_auc", "test_mcc", "test_FNR"]
    print(table[cols].to_string(index=False))
    print("Phase 3 complete.")


if __name__ == "__main__":
    main()
