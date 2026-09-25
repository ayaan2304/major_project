# Explanation-Preserving Imbalance-Aware Ensemble Learning for Mobile Money Fraud Detection

Research repository for **EP-IAIL**: filter SMOTE-style synthetic fraud so it stays statistically plausible **and** SHAP-consistent with genuine fraud.

## Research question

Can an imbalance-handling strategy be designed so that synthetic minority/fraud examples remain not only statistically plausible but also behaviorally and explainably consistent with genuine fraudulent transactions?

## What is novel vs the given baseline

The original PaySim ensemble study (stratified 80/20, dropped `step`/`type`/`nameOrig`/`nameDest`, voting of trees + NN, global SHAP, headline accuracy/F1/ROC-AUC) is **reproduced in Phase 1**. It is not claimed as novelty.

**EP-IAIL** (Phases 3–4) adds: a train-only SHAP fraud prototype, train-only synthetic candidates, combined explanation-similarity + plausibility filtering with validation-chosen thresholds, temporal evaluation, cost-sensitive operating points, fidelity/stability/drift tests.

The proposed EP-IAIL framework integrates model explainability directly into minority-class data augmentation — synthetic fraud samples are evaluated for behavioral plausibility and SHAP attribution consistency with genuine fraud before entering final training.

## Setup

Python 3.10+ (this workspace used 3.12). CPU-first; GPU is unused.

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

Place PaySim at:

`data/raw/PS_20174392719_1491204439457_log.csv`

All knobs live in `configs/config.yaml` (seed, splits, caps, EP-IAIL percentiles, cost scenarios).

## How to run (phase by phase)

From the repo root (`major_projectt/`):

```bash
python experiments/00_setup_validate.py      # full-file quality report
python experiments/01_original_baseline.py   # original-style baseline
python experiments/02_temporal_features.py   # chronological split + features
python experiments/03_ep_iail.py             # novelty + imbalance comparison
python experiments/04_evaluation_suite.py    # ensemble, cost, XAI, stats
python experiments/05_write_report.py        # reports/research_report.md from artifacts
```

Confirm Phase 1 metrics exist under `results/tables/phase1_original_baseline_metrics.csv` before treating Phase 3 as complete.

Notebooks in `notebooks/` mirror the same story (EDA → features → baseline → EP-IAIL → XAI → final). They import `src/`; they do not re-fit on test.

## Reproducibility and leakage

- Fixed seed in config.
- Scalers, SMOTE, SHAP prototypes, and thresholds are train/val only.
- `src/leakage_audit.py` asserts disjoint splits and chronological order.
- **No fabricated metrics.** `experiments/05_write_report.py` copies numbers from `results/` or writes `NOT_RUN`.

## Documented approximations (not silent)

PaySim has millions of rows. Config `compute.legit_cap` keeps **all fraud** and caps legitimate **training** rows for CPU-bound models. TreeSHAP uses a train background subset. MLP uses `models.mlp.max_samples`. Correlation screening uses 80k train rows. Validation/test distributions are not resampled.

## Limitations

Simulator data; SHAP background is a subset; training prior is altered by the legit cap; bootstrap Friedman blocks are not independent production days. See `reports/research_report.md` after Phase 5.

## Layout

See the tree in the project prompt: `src/`, `experiments/`, `notebooks/`, `results/`, `configs/config.yaml`. Maths: `docs/mathematical_formulation.md`.
