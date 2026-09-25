"""
Phase 4 — Evaluation suite: ensemble, cost, calibration, XAI, fidelity,
stability, temporal drift, statistics.

python experiments/04_evaluation_suite.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import apply_isotonic, fit_isotonic
from src.ensemble import (
    collect_probas,
    fit_stacker,
    soft_vote,
    stacker_predict,
    weighted_vote,
    weights_from_val_pr_auc,
)
from src.evaluation import classification_report_dict, metrics_table
from src.explainability import ablation_fidelity, global_mean_abs_shap, local_top_features
from src.feature_engineering import model_feature_columns
from src.leakage_audit import subsample_train_keep_all_fraud
from src.models import (
    make_catboost,
    make_lightgbm,
    make_logistic_regression,
    make_mlp,
    make_random_forest,
    make_xgboost,
)
from src.pipeline import eval_model, fit_named_model
from src.shap_prototype import sample_background, tree_shap_values
from src.stability import rank_stability
from src.statistical_tests import bootstrap_pr_auc_ci, friedman_methods, js_divergence, paired_wilcoxon
from src.threshold_optimizer import cost_at_threshold, tune_threshold
from src.utils import get_logger, load_config, results_paths, save_json, save_table, set_global_seed
from src.visualization import plot_metric_bars, plot_reliability


def _xy(frame: pd.DataFrame, cfg, drop_behavioral: bool = False):
    target = cfg["data"]["target"]
    time_col = cfg["data"]["time_col"]
    cols = [c for c in model_feature_columns(frame, cfg) if c != time_col]
    if drop_behavioral:
        keep = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            cfg["data"]["flagged_col"],
        ]
        cols = [c for c in keep if c in frame.columns]
    return frame[cols], frame[target].astype(int)


def _cases(y, p, n_each=3, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    out = {}
    mapping = {
        "TP": (y == 1) & (pred == 1),
        "TN": (y == 0) & (pred == 0),
        "FP": (y == 0) & (pred == 1),
        "FN": (y == 1) & (pred == 0),
    }
    for k, m in mapping.items():
        idx = np.where(m)[0]
        if len(idx) == 0:
            out[k] = []
            continue
        take = rng.choice(idx, size=min(n_each, len(idx)), replace=False)
        out[k] = take.tolist()
    # Borderline: probabilities closest to 0.5
    dist = np.abs(p - 0.5)
    out["borderline"] = np.argsort(dist)[:n_each].tolist()
    return out


def main() -> None:
    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)
    paths = results_paths(cfg)
    logger = get_logger("phase4", paths["logs"] / "phase4.log", cfg["logging"]["level"])

    train = pd.read_parquet(paths["processed"] / "phase2_train.parquet")
    val = pd.read_parquet(paths["processed"] / "phase2_val.parquet")
    test = pd.read_parquet(paths["processed"] / "phase2_test.parquet")
    X_tr, y_tr = _xy(train, cfg)
    X_va, y_va = _xy(val, cfg)
    X_te, y_te = _xy(test, cfg)

    X_fit = pd.read_parquet(paths["processed"] / "phase3_ep_iail_X_fit.parquet")
    y_fit = pd.read_parquet(paths["processed"] / "phase3_ep_iail_y_fit.parquet")["y"].astype(int)

    # ---------- Ensemble on EP-IAIL training set ----------
    members = {}
    scaler = StandardScaler().fit(X_fit)
    X_fit_s = pd.DataFrame(scaler.transform(X_fit), columns=X_fit.columns)
    X_va_s = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
    joblib.dump(scaler, paths["models"] / "phase4_ensemble_scaler.joblib")

    builders = {
        "logistic_regression": lambda: make_logistic_regression(cfg, seed, class_weight=None),
        "random_forest": lambda: make_random_forest(cfg, seed, class_weight=None),
        "xgboost": lambda: make_xgboost(cfg, seed, y_train=y_fit, use_spw=False),
        "lightgbm": lambda: make_lightgbm(cfg, seed, y_train=y_fit, use_spw=False),
        "catboost": lambda: make_catboost(cfg, seed, y_train=y_fit, use_spw=False),
        "mlp": lambda: make_mlp(cfg, seed),
    }
    member_rows = []
    val_probas = {}
    test_probas = {}
    for name, builder in builders.items():
        model = builder()
        Xtr = X_fit_s if name in {"logistic_regression", "mlp"} else X_fit
        Xva = X_va_s if name in {"logistic_regression", "mlp"} else X_va
        Xte = X_te_s if name in {"logistic_regression", "mlp"} else X_te
        if name == "random_forest":
            # RF still capped if the EP-IAIL set is huge from SMOTE.
            cap = cfg["compute"]["legit_cap"]
            if len(Xtr) > cap + int(y_fit.sum()):
                Xtr, ytr, _ = subsample_train_keep_all_fraud(Xtr, y_fit, cap, seed, True)
            else:
                ytr = y_fit
        else:
            ytr = y_fit
        model, extra = fit_named_model(name, model, Xtr, ytr, cfg, seed)
        members[name] = model
        m_va, p_va = eval_model(name, model, Xva, y_va)
        m_te, p_te = eval_model(name, model, Xte, y_te)
        val_probas[name] = p_va
        test_probas[name] = p_te
        member_rows.append({"member": name, **{f"val_{k}": v for k, v in m_va.items()}, **{f"test_{k}": v for k, v in m_te.items()}, **extra})
        joblib.dump(model, paths["models"] / f"phase4_member_{name}.joblib")

    w = weights_from_val_pr_auc(y_va, val_probas)
    vote_va = soft_vote(val_probas)
    vote_te = soft_vote(test_probas)
    wvote_va = weighted_vote(val_probas, w)
    wvote_te = weighted_vote(test_probas, w)
    stacker = fit_stacker(val_probas, y_va, seed)
    stack_va = stacker_predict(stacker, val_probas)
    stack_te = stacker_predict(stacker, test_probas)
    joblib.dump(stacker, paths["models"] / "phase4_stacker.joblib")
    save_json(w, paths["metrics"] / "phase4_vote_weights.json")

    ens_rows = []
    for label, pv, pt in [
        ("soft_vote", vote_va, vote_te),
        ("weighted_vote", wvote_va, wvote_te),
        ("stacking", stack_va, stack_te),
    ]:
        ens_rows.append(
            {
                "ensemble": label,
                **{f"val_{k}": v for k, v in classification_report_dict(y_va, pv).items()},
                **{f"test_{k}": v for k, v in classification_report_dict(y_te, pt).items()},
            }
        )
    save_table(pd.DataFrame(member_rows), paths["tables"] / "phase4_ensemble_members.csv")
    save_table(pd.DataFrame(ens_rows), paths["tables"] / "phase4_ensemble_aggregates.csv")

    # Pick the validation-best aggregator for cost/calibration/XAI.
    best_ens = max(ens_rows, key=lambda r: r["val_pr_auc"])["ensemble"]
    p_va = {"soft_vote": vote_va, "weighted_vote": wvote_va, "stacking": stack_va}[best_ens]
    p_te = {"soft_vote": vote_te, "weighted_vote": wvote_te, "stacking": stack_te}[best_ens]
    logger.info("Best aggregator on val PR-AUC: %s", best_ens)

    # ---------- Cost-sensitive thresholds ----------
    cost_rows = []
    frozen = {}
    for scen_name, scen in cfg["cost"]["scenarios"].items():
        tuned = tune_threshold(y_va, p_va, scen["C_FN"], scen["C_FP"], cfg["cost"]["threshold_grid_size"])
        t = tuned["best"]["threshold"]
        frozen[scen_name] = t
        val_cost = tuned["best"]
        test_cost = cost_at_threshold(y_te, p_te, t, scen["C_FN"], scen["C_FP"])
        test_05 = cost_at_threshold(y_te, p_te, 0.5, scen["C_FN"], scen["C_FP"])
        cost_rows.append(
            {
                "scenario": scen_name,
                "C_FN": scen["C_FN"],
                "C_FP": scen["C_FP"],
                "threshold_val_tuned": t,
                "val_cost": val_cost["cost"],
                "test_cost_tuned": test_cost["cost"],
                "test_cost_at_0.5": test_05["cost"],
                "test_FN_tuned": test_cost["FN"],
                "test_FP_tuned": test_cost["FP"],
                "test_metrics_tuned": classification_report_dict(y_te, p_te, threshold=t),
            }
        )
    save_json({"frozen_thresholds": frozen, "rows": cost_rows}, paths["metrics"] / "phase4_cost_sensitive.json")
    # flatten for csv
    flat = []
    for r in cost_rows:
        d = {k: v for k, v in r.items() if k != "test_metrics_tuned"}
        d.update({f"test_{k}": v for k, v in r["test_metrics_tuned"].items()})
        flat.append(d)
    save_table(pd.DataFrame(flat), paths["tables"] / "phase4_cost_sensitive.csv")

    # ---------- Calibration ----------
    iso = fit_isotonic(y_va, p_va)
    p_va_cal = apply_isotonic(iso, p_va)
    p_te_cal = apply_isotonic(iso, p_te)
    cal_rows = [
        {"split": "test_raw", **classification_report_dict(y_te, p_te)},
        {"split": "test_isotonic_fit_on_val", **classification_report_dict(y_te, p_te_cal)},
        {"split": "val_raw", **classification_report_dict(y_va, p_va)},
        {"split": "val_isotonic", **classification_report_dict(y_va, p_va_cal)},
    ]
    save_table(pd.DataFrame(cal_rows), paths["tables"] / "phase4_calibration.csv")
    plot_reliability(y_te, p_te, paths["figures"] / "phase4_reliability_raw.png", title="Test reliability (raw)")
    plot_reliability(y_te, p_te_cal, paths["figures"] / "phase4_reliability_isotonic.png", title="Test reliability (isotonic on val)")
    joblib.dump(iso, paths["models"] / "phase4_isotonic.joblib")

    # ---------- Ablations (LightGBM, temporal test) ----------
    ablation_rows = []
    phase3 = pd.read_csv(paths["tables"] / "phase3_imbalance_comparison.csv")
    for method in ["none", "class_weight", "smote", "ep_iail"]:
        sub = phase3[phase3["method"] == method].iloc[0].to_dict()
        ablation_rows.append({"ablation": f"imbalance={method}", **{k: sub[k] for k in sub if k.startswith("test_")}})

    # drop behavioral features
    Xb_tr, yb_tr = _xy(train, cfg, drop_behavioral=True)
    Xb_te, yb_te = _xy(test, cfg, drop_behavioral=True)
    Xbc, ybc, _ = subsample_train_keep_all_fraud(Xb_tr, yb_tr, cfg["compute"]["legit_cap"], seed, True)
    m_nb = make_lightgbm(cfg, seed, y_train=ybc, use_spw=True)
    m_nb.fit(Xbc, ybc)
    ablation_rows.append({"ablation": "drop_behavioral_features", **{f"test_{k}": v for k, v in eval_model("nb", m_nb, Xb_te, yb_te)[0].items()}})

    # drop cost threshold (already have 0.5 vs tuned in cost table)
    default_scen = cfg["cost"]["default_scenario"]
    t_star = frozen[default_scen]
    ablation_rows.append({"ablation": "threshold_0.5", **{f"test_{k}": v for k, v in classification_report_dict(y_te, p_te, 0.5).items()}})
    ablation_rows.append({"ablation": "threshold_cost_tuned", **{f"test_{k}": v for k, v in classification_report_dict(y_te, p_te, t_star).items()}})
    ablation_rows.append({"ablation": "drop_calibration_raw", **{f"test_{k}": v for k, v in classification_report_dict(y_te, p_te).items()}})
    ablation_rows.append({"ablation": "with_isotonic_calibration", **{f"test_{k}": v for k, v in classification_report_dict(y_te, p_te_cal).items()}})

    # drop temporal: stratified split on engineered features (train+val+test pooled then random) — documented as a contrast, not a leaked production model.
    pooled = pd.concat([train, val, test], axis=0)
    Xp, yp = _xy(pooled, cfg)
    from sklearn.model_selection import train_test_split

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(Xp, yp, test_size=0.2, stratify=yp, random_state=seed)
    Xtrc, ytrc, _ = subsample_train_keep_all_fraud(Xtr_r, ytr_r, cfg["compute"]["legit_cap"], seed, True)
    m_rand = make_lightgbm(cfg, seed, y_train=ytrc, use_spw=True)
    m_rand.fit(Xtrc, ytrc)
    ablation_rows.append({"ablation": "drop_temporal_split_stratified", **{f"test_{k}": v for k, v in eval_model("rand", m_rand, Xte_r, yte_r)[0].items()}})
    save_table(pd.DataFrame(ablation_rows), paths["tables"] / "phase4_ablation.csv")

    # ---------- Explainability (LightGBM EP-IAIL member, train-background, test subsample) ----------
    lgbm = members["lightgbm"]
    bg = sample_background(X_fit, cfg["compute"]["shap_background_n"], seed)
    rng = np.random.RandomState(seed)
    n_exp = min(int(cfg["compute"]["shap_explain_n"]), len(X_te))
    # Stratify explain subsample: all test fraud up to cap + random legit.
    te_pos = np.where(y_te.to_numpy() == 1)[0]
    te_neg = np.where(y_te.to_numpy() == 0)[0]
    n_pos = min(len(te_pos), n_exp // 2 if n_exp > 20 else len(te_pos))
    pos_i = rng.choice(te_pos, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
    n_neg = min(len(te_neg), n_exp - len(pos_i))
    neg_i = rng.choice(te_neg, size=n_neg, replace=False)
    expl_i = np.concatenate([pos_i, neg_i])
    X_exp = X_te.iloc[expl_i]
    y_exp = y_te.iloc[expl_i]
    shap_exp = tree_shap_values(lgbm, X_exp, bg, cfg["compute"]["shap_batch_size"])
    np.save(paths["shap"] / "phase4_test_subset_shap.npy", shap_exp)
    glob = global_mean_abs_shap(shap_exp, list(X_te.columns))
    save_table(glob, paths["tables"] / "phase4_global_shap.csv")

    p_exp = lgbm.predict_proba(X_exp)[:, 1]
    case_idx = _cases(y_exp.to_numpy(), p_exp, n_each=cfg["evaluation"]["local_case_counts"]["TP"], seed=seed)
    local_dump = {}
    for kind, idxs in case_idx.items():
        local_dump[kind] = []
        for i in idxs:
            top = local_top_features(shap_exp[i], list(X_te.columns), k=8)
            local_dump[kind].append(
                {
                    "subset_row": int(i),
                    "y": int(y_exp.iloc[i]),
                    "p": float(p_exp[i]),
                    "top": top.to_dict(orient="records"),
                }
            )
    save_json(local_dump, paths["metrics"] / "phase4_local_shap_cases.json")

    fid = ablation_fidelity(lgbm, X_exp.reset_index(drop=True), shap_exp, cfg["evaluation"]["topk_fidelity"], seed)
    fid_sum = fid.groupby("k")[["drop_top", "drop_random"]].mean().reset_index()
    save_table(fid, paths["tables"] / "phase4_fidelity_per_row.csv")
    save_table(fid_sum, paths["tables"] / "phase4_fidelity_summary.csv")

    # Real vs synthetic explanation consistency (cosine of mean |SHAP| vectors)
    syn_path = paths["shap"] / "phase3_train_fraud_shap.npy"
    genuine_shap = np.load(syn_path)
    # SHAP on accepted synthetics: take a sample from X_fit beyond original train length if possible.
    n_orig = len(X_tr)  # not equal to X_fit; use y==1 rows that are last-accepted — we score a sample of X_fit fraud
    fraud_fit = X_fit.loc[y_fit.to_numpy() == 1]
    n_syn_s = min(400, len(fraud_fit))
    syn_sample = fraud_fit.sample(n=n_syn_s, random_state=seed)
    shap_fit_fraud = tree_shap_values(lgbm, syn_sample, bg, cfg["compute"]["shap_batch_size"])
    # Compare mean SHAP of genuine train fraud (provisional) vs this mix — also compare candidate accepted vs genuine from phase3 files
    from sklearn.metrics.pairwise import cosine_similarity

    g_mean = genuine_shap.mean(axis=0, keepdims=True)
    s_mean = shap_fit_fraud.mean(axis=0, keepdims=True)
    cons = {
        "cosine_mean_shap_genuine_vs_fit_fraud_sample": float(cosine_similarity(g_mean, s_mean)[0, 0]),
        "note": "Fit-fraud sample mixes genuine train fraud and accepted synthetics because they share the EP-IAIL training label.",
    }
    save_json(cons, paths["metrics"] / "phase4_real_vs_synthetic_shap.json")

    # ---------- Stability ----------
    n_runs = int(cfg["compute"]["stability_runs"])
    imps = []
    stab_metrics = []
    explain_freeze = sample_background(X_va, min(400, len(X_va)), seed)
    for r in range(n_runs):
        rs = seed + 17 * (r + 1)
        m = make_lightgbm(cfg, rs, y_train=y_fit, use_spw=False)
        m.fit(X_fit, y_fit)
        sv = tree_shap_values(m, explain_freeze, bg, cfg["compute"]["shap_batch_size"])
        imps.append(np.mean(np.abs(sv), axis=0))
        p = m.predict_proba(X_te)[:, 1]
        stab_metrics.append(classification_report_dict(y_te, p)["pr_auc"])
    stab = rank_stability(imps, topk=10)
    stab["pr_auc_mean"] = float(np.mean(stab_metrics))
    stab["pr_auc_std"] = float(np.std(stab_metrics))
    save_json(stab, paths["metrics"] / "phase4_stability.json")

    # ---------- Temporal drift on test windows ----------
    time_col = cfg["data"]["time_col"]
    n_win = int(cfg["compute"]["temporal_windows"])
    steps = test[time_col].to_numpy()
    qs = np.quantile(steps, np.linspace(0, 1, n_win + 1))
    drift_rows = []
    prev_p = None
    for w_i in range(n_win):
        lo, hi = qs[w_i], qs[w_i + 1]
        mask = (steps >= lo) & (steps <= hi if w_i == n_win - 1 else steps < hi)
        if mask.sum() < 50 or y_te.to_numpy()[mask].sum() == 0:
            drift_rows.append({"window": w_i, "n": int(mask.sum()), "note": "too few fraud"})
            continue
        pw = p_te[mask]
        yw = y_te.to_numpy()[mask]
        met = classification_report_dict(yw, pw)
        js = js_divergence(prev_p, pw) if prev_p is not None else 0.0
        prev_p = pw
        # SHAP ranking on a small window sample
        idx_w = np.where(mask)[0]
        take = rng.choice(idx_w, size=min(200, len(idx_w)), replace=False)
        svw = tree_shap_values(lgbm, X_te.iloc[take], bg, cfg["compute"]["shap_batch_size"])
        ranking = global_mean_abs_shap(svw, list(X_te.columns))["feature"].head(8).tolist()
        drift_rows.append({"window": w_i, "step_lo": float(lo), "step_hi": float(hi), "n": int(mask.sum()), "js_to_prev_scores": js, "top8_shap": ranking, **met})
    save_json(drift_rows, paths["metrics"] / "phase4_temporal_drift.json")
    flat_drift = []
    for r in drift_rows:
        item = {k: v for k, v in r.items() if k != "top8_shap"}
        top = r.get("top8_shap", [])
        item["top8_shap"] = ",".join(top) if isinstance(top, list) else str(top)
        flat_drift.append(item)
    save_table(pd.DataFrame(flat_drift), paths["tables"] / "phase4_temporal_drift.csv")

    # ---------- Bootstrap CIs + tests vs SMOTE ----------
    smote_model = joblib.load(paths["models"] / "phase3_smote_lgbm.joblib")
    p_smote = smote_model.predict_proba(X_te)[:, 1]
    p_ep = joblib.load(paths["models"] / "phase3_ep_iail_lgbm.joblib").predict_proba(X_te)[:, 1]
    ci_ep = bootstrap_pr_auc_ci(y_te, p_ep, cfg["compute"]["bootstrap_ci_samples"], seed)
    ci_sm = bootstrap_pr_auc_ci(y_te, p_smote, cfg["compute"]["bootstrap_ci_samples"], seed)
    # paired bootstrap differences
    rng = np.random.RandomState(seed)
    yv = y_te.to_numpy()
    diffs = []
    n = len(yv)
    from sklearn.metrics import average_precision_score

    for _ in range(cfg["compute"]["bootstrap_ci_samples"]):
        ix = rng.randint(0, n, size=n)
        diffs.append(average_precision_score(yv[ix], p_ep[ix]) - average_precision_score(yv[ix], p_smote[ix]))
    diffs = np.asarray(diffs)
    stats_out = {
        "ep_iail_pr_auc_ci": ci_ep,
        "smote_pr_auc_ci": ci_sm,
        "delta_ep_minus_smote": {
            "mean": float(diffs.mean()),
            "ci_lo": float(np.quantile(diffs, 0.025)),
            "ci_hi": float(np.quantile(diffs, 0.975)),
        },
        "wilcoxon_stability_pr_auc_vs_dummy": "n/a — single test set; bootstrap used instead of Wilcoxon on seeds unless n_runs>=6",
        "friedman_on_phase3_test_metrics": None,
    }
    # Friedman on methods using bootstrap replicates as blocks (documented approximation of paired blocks).
    methods_p = {
        "none": joblib.load(paths["models"] / "phase3_none_lgbm.joblib").predict_proba(X_te)[:, 1],
        "class_weight": joblib.load(paths["models"] / "phase3_class_weight_lgbm.joblib").predict_proba(X_te)[:, 1],
        "smote": p_smote,
        "ep_iail": p_ep,
    }
    block = []
    rng2 = np.random.RandomState(seed + 1)
    n_blocks = 12
    for _ in range(n_blocks):
        ix = rng2.randint(0, n, size=n)
        block.append([average_precision_score(yv[ix], methods_p[m][ix]) for m in methods_p])
    stats_out["friedman_on_phase3_test_metrics"] = friedman_methods(np.asarray(block))
    stats_out["friedman_methods_order"] = list(methods_p)
    stats_out["note"] = "Friedman blocks are bootstrap resamples of the frozen test set, not independent deployments."
    save_json(stats_out, paths["metrics"] / "phase4_statistical_tests.json")

    # Final comparison table request
    final_rows = []
    for _, r in phase3.iterrows():
        final_rows.append(
            {
                "system": r["method"],
                "precision": r["test_precision"],
                "recall": r["test_recall"],
                "f1": r["test_f1"],
                "pr_auc": r["test_pr_auc"],
                "roc_auc": r["test_roc_auc"],
                "mcc": r["test_mcc"],
                "FNR": r["test_FNR"],
                "brier": r["test_brier"],
                "ece": r["test_ece"],
                "inference_seconds": r.get("test_inference_seconds", np.nan),
            }
        )
    final_rows.append(
        {
            "system": f"ep_iail_ensemble_{best_ens}",
            **{k: classification_report_dict(y_te, p_te)[k] for k in ["precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "FNR", "brier", "ece"]},
        }
    )
    save_table(pd.DataFrame(final_rows), paths["tables"] / "phase4_final_comparison.csv")
    logger.info("Phase 4 complete. Best ensemble=%s", best_ens)
    print("Phase 4 complete.")


if __name__ == "__main__":
    main()
