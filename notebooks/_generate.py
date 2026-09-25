"""Generate the six research notebooks (Objective / Why / Method / Code / ...)."""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


BOOT = """import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.utils import load_config, results_paths
cfg = load_config()
paths = results_paths(cfg)
print("Project:", cfg["project"]["title"])
print("Results:", paths["root"])"""

NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# EDA — PaySim mobile money\n\n## Objective\nDescribe the **actual** file: size, types, imbalance, amount and step ranges.\n\n## Why it matters\nFraud rate drives metric choice. Accuracy will look excellent for a majority classifier.\n\n## Method\nChunked scan via `src.data_validation.validate_paysim` (full file). No modelling, no split fitting.\n\n## Research implication\nIf the fraud rate is ~0.13%, PR-AUC/MCC/FNR are the honest headlines."),
        code(BOOT),
        code("""from src.data_validation import validate_paysim
import json
report = validate_paysim(cfg)
print(json.dumps({k: report[k] for k in ["n_rows","class_counts","fraud_rate","type_distribution","step_range"]}, indent=2))"""),
        md("## Output\nQuality JSON under `results/metrics/phase0_data_quality.json` after `experiments/00_setup_validate.py`.\n\n## Interpretation\nCompare `fraud_rate` to a 99.9% accuracy claim: the majority baseline already sits there.\n\n## Failure modes\nFile missing, dtypes changed, or class counts not partitioning `n_rows` (assertion)."),
    ],
    "02_features.ipynb": [
        md("# Behavioral features\n\n## Objective\nBuild **causal** features from step/type/balances without using raw IDs as inputs.\n\n## Why it matters\nRQ2 / H1: does temporal behavior help minority detection?\n\n## Method\n`engineer_features` then `temporal_frames`. Univariate rank-AUC on **train** only.\n\n## Leakage\nGroupby + shift; globally sorted by `step`. No future rows."),
        code(BOOT),
        code("""import pandas as pd
sep = pd.read_csv(paths["tables"] / "phase2_univariate_separation.csv")
sep.head(15)"""),
        md("## Output\n`data/processed/phase2_{train,val,test}.parquet`, univariate CSV.\n\n## Interpretation\nHigh rank-AUC features (drain flags, amount-to-balance, TRANSFER/CASH_OUT) should dominate SHAP later.\n\n## Failure modes\nUnsorted frames would leak future account history — `temporal_split` asserts step order."),
    ],
    "03_baseline.ipynb": [
        md("# Original-style baseline\n\n## Objective\nReproduce the **given protocol** (not the paper's unpublished code): drop columns, stratified 80/20, ensemble members.\n\n## Why it matters\nWithout this, later gains could be from features/splits instead of EP-IAIL.\n\n## Method\n`experiments/01_original_baseline.py`. Label every table **Original-style baseline**."),
        code(BOOT),
        code("""import pandas as pd
df = pd.read_csv(paths["tables"] / "phase1_original_baseline_metrics.csv")
df[["model","accuracy","precision","recall","f1","pr_auc","roc_auc","mcc","FNR"]]"""),
        md("## Interpretation\nMajority baseline accuracy ≈ 1 − π. If an ensemble's accuracy is 99.9% but recall/F1/PR-AUC/MCC are modest, accuracy is the wrong headline.\n\n## Research implication\nPhase 3 must beat **this protocol's minority metrics**, not a 99.9% accuracy number."),
    ],
    "04_ep_iail.ipynb": [
        md("# EP-IAIL\n\n## Objective\nFilter synthetic fraud with SHAP-prototype similarity + plausibility.\n\n## Why it matters\nRQ3–RQ4 / H2–H3: does explanation-preserving filtering help detection **or** only explanations?\n\n## Method\nProvisional LightGBM (TreeSHAP) → prototype (mean/median/trimmed-mean) → Borderline-SMOTE candidates → percentile grid on validation PR-AUC."),
        code(BOOT),
        code("""import json, pandas as pd
print(json.dumps(json.load(open(paths["metrics"]/"phase3_shap_prototype.json")), indent=2)[:2000])
pd.read_csv(paths["tables"]/"phase3_imbalance_comparison.csv")[
    ["method","test_precision","test_recall","test_f1","test_pr_auc","test_mcc","test_FNR"]
]"""),
        md("## Interpretation\nAcceptance rate too high ⇒ filter is idle. Too low ⇒ under-powered oversampling. If EP-IAIL loses to SMOTE on PR-AUC, report it and inspect rejected SHAP histograms.\n\n## Failure modes\nTest used for percentile selection (forbidden); prototype built on test fraud (forbidden)."),
    ],
    "05_xai.ipynb": [
        md("# Explainability, fidelity, stability\n\n## Objective\nGlobal and local SHAP; top-k vs random ablation; rank stability; genuine vs synthetic SHAP cosine.\n\n## Why it matters\nRQ4–RQ6 / H2 / H5. Detection gains that destroy explanation reliability are not the stated novelty."),
        code(BOOT),
        code("""import pandas as pd, json
display(pd.read_csv(paths["tables"]/"phase4_global_shap.csv").head(12))
display(pd.read_csv(paths["tables"]/"phase4_fidelity_summary.csv"))
print(json.load(open(paths["metrics"]/"phase4_stability.json")))"""),
        md("## Interpretation\nIf mean `drop_top` > `drop_random` for k=1,3,5, H5 is supported on this sample. Low Jaccard across seeds means unstable fraud narratives.\n\n## Approximation\nSHAP on a stratified **test subsample** with train background — documented in config."),
    ],
    "06_final.ipynb": [
        md("# Final comparison and hypotheses\n\n## Objective\nRead the frozen test tables and accept/reject H1–H5 honestly.\n\n## Why it matters\nThe report must not be rewritten to make EP-IAIL win."),
        code(BOOT),
        code("""import pandas as pd
from pathlib import Path
print("Final comparison:")
display(pd.read_csv(paths["tables"]/"phase4_final_comparison.csv"))
print("Ablation:")
display(pd.read_csv(paths["tables"]/"phase4_ablation.csv"))
print("Research report exists:", (ROOT/"reports"/"research_report.md").exists())"""),
        md("## Research implication\nDeployment-relevant numbers are **temporal-split** PR-AUC, FNR, cost at validation-tuned thresholds, plus stability and drift — not Phase 1 accuracy."),
    ],
}


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (NB_DIR / name).write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
