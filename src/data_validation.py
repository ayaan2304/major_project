"""
Data-quality validation for PaySim (Phase 0).

What / why
----------
Research numbers (n, fraud rate, ranges) must be computed from the file,
never copied from the paper. This module streams the CSV and writes a
machine-readable quality report.

Assertions
----------
fraud + legitimate == total; no negative amounts; isFraud in {0,1}.

Leakage risks
-------------
None: this is descriptive only and must not write fitted transformers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_loader import iter_chunks
from src.utils import get_logger, save_json

logger = get_logger("data_validation")


def validate_paysim(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Scan the raw CSV in chunks and return a quality report dict."""
    n_rows = 0
    n_dup_extra = 0
    missing = Counter()
    type_counts: Counter[str] = Counter()
    class_counts = Counter()
    flagged_counts = Counter()
    step_min, step_max = np.inf, -np.inf
    amount_sum = 0.0
    amount_sumsq = 0.0
    amount_min, amount_max = np.inf, -np.inf
    n_neg_amount = 0
    n_neg_balance = 0
    n_impossible_orig = 0  # newbalanceOrig < 0 already counted; also NaN checks
    columns: list[str] | None = None
    dtypes: dict[str, str] | None = None

    seen_hash_chunks = 0  # duplicates across chunks are not fully counted; document
    for i, chunk in enumerate(iter_chunks(cfg)):
        if columns is None:
            columns = list(chunk.columns)
            dtypes = {c: str(chunk[c].dtype) for c in columns}
        n_rows += len(chunk)
        n_dup_extra += int(chunk.duplicated().sum())
        missing.update({c: int(chunk[c].isna().sum()) for c in chunk.columns})
        type_counts.update(chunk["type"].astype(str).value_counts().to_dict())
        class_counts.update(chunk["isFraud"].value_counts().to_dict())
        flagged_counts.update(chunk["isFlaggedFraud"].value_counts().to_dict())
        step_min = min(step_min, int(chunk["step"].min()))
        step_max = max(step_max, int(chunk["step"].max()))
        amt = chunk["amount"].to_numpy()
        amount_sum += float(amt.sum())
        amount_sumsq += float((amt * amt).sum())
        amount_min = min(amount_min, float(amt.min()))
        amount_max = max(amount_max, float(amt.max()))
        n_neg_amount += int((chunk["amount"] < 0).sum())
        bal_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        n_neg_balance += int((chunk[bal_cols] < 0).sum().sum())
        # PaySim can have accounting inconsistencies; count them, do not drop yet.
        orig_delta = chunk["oldbalanceOrg"] - chunk["amount"] - chunk["newbalanceOrig"]
        n_impossible_orig += int((orig_delta.abs() > 1e-2).sum())
        seen_hash_chunks += 1
        if (i + 1) % 8 == 0:
            logger.info("Validated %s rows...", f"{n_rows:,}")

    normalised = {int(k): int(v) for k, v in class_counts.items()}
    n_fraud = normalised.get(1, 0)
    n_legit = normalised.get(0, 0)
    assert n_fraud + n_legit == n_rows, (
        f"Class counts do not partition the table: fraud={n_fraud}, legit={n_legit}, n={n_rows}"
    )
    assert n_neg_amount == 0, f"Negative amounts found: {n_neg_amount}"

    mean_amt = amount_sum / max(n_rows, 1)
    var_amt = amount_sumsq / max(n_rows, 1) - mean_amt**2
    std_amt = float(np.sqrt(max(var_amt, 0.0)))

    report = {
        "n_rows": n_rows,
        "n_columns": len(columns or []),
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": dict(missing),
        "duplicate_rows_within_chunks": n_dup_extra,
        "duplicate_note": (
            "Exact duplicates are counted inside each chunk only. "
            "A full-file duplicate pass is optional and memory-heavy."
        ),
        "class_counts": {"legitimate": n_legit, "fraud": n_fraud},
        "fraud_rate": n_fraud / n_rows if n_rows else None,
        "imbalance_ratio_legit_per_fraud": (n_legit / n_fraud) if n_fraud else None,
        "type_distribution": dict(type_counts),
        "isFlaggedFraud_counts": {str(k): int(v) for k, v in flagged_counts.items()},
        "step_range": {"min": int(step_min), "max": int(step_max)},
        "amount_stats": {
            "min": amount_min,
            "max": amount_max,
            "mean": mean_amt,
            "std": std_amt,
        },
        "n_negative_amounts": n_neg_amount,
        "n_negative_balance_cells": n_neg_balance,
        "n_origin_balance_inconsistencies": n_impossible_orig,
        "assertions_passed": True,
        "chunks_scanned": seen_hash_chunks,
    }
    return report


def write_quality_report(report: dict[str, Any], path) -> None:
    save_json(report, path)
    logger.info("Wrote data-quality report to %s", path)
