"""
Chronological split on PaySim `step` (hour index).

What / why
----------
Random stratified splits leak future fraud patterns into training
(RQ1 / H4). A deployment-realistic protocol trains on earlier hours
and tests on later hours.

Split definition (plain English)
--------------------------------
Sort unique steps. Assign the earliest train_frac of the *step range*
(not row count) to train, the next val_frac to validation, the rest
to test. Using the time axis (not row quantiles) matches "earliest /
middle / latest".

Leakage risks
-------------
- Never shuffle before cutting on step.
- Never put a step into two splits.
- Features that need history may use PAST rows only (see feature_engineering).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("temporal_split")


def step_cutpoints(steps: np.ndarray, train_frac: float, val_frac: float) -> dict[str, float]:
    """Compute inclusive train_max_step and val_max_step on the time axis."""
    smin, smax = int(steps.min()), int(steps.max())
    span = smax - smin
    train_max = smin + span * train_frac
    val_max = smin + span * (train_frac + val_frac)
    return {"step_min": smin, "step_max": smax, "train_max_step": train_max, "val_max_step": val_max}


def assign_temporal_split(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    """Return a Series of labels {train, val, test} aligned to df.index."""
    time_col = cfg["data"]["time_col"]
    spec = cfg["split"]["temporal"]
    cuts = step_cutpoints(df[time_col].to_numpy(), spec["train_frac"], spec["val_frac"])
    step = df[time_col]
    label = pd.Series("test", index=df.index, dtype="string")
    label.loc[step <= cuts["train_max_step"]] = "train"
    label.loc[(step > cuts["train_max_step"]) & (step <= cuts["val_max_step"])] = "val"
    return label


def temporal_frames(df: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Split a feature-complete frame into train/val/test by step."""
    labels = assign_temporal_split(df, cfg)
    parts = {k: df.loc[labels == k].copy() for k in ("train", "val", "test")}
    time_col = cfg["data"]["time_col"]
    meta = {
        "protocol": "chronological_by_step",
        "fractions": dict(cfg["split"]["temporal"]),
        "n": {k: int(len(v)) for k, v in parts.items()},
        "fraud": {k: int(v[cfg["data"]["target"]].sum()) for k, v in parts.items()},
        "step_range": {
            k: {"min": int(v[time_col].min()) if len(v) else None, "max": int(v[time_col].max()) if len(v) else None}
            for k, v in parts.items()
        },
    }
    # Invariant: max(train step) <= min(val step) and max(val) <= min(test)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    if len(tr) and len(va):
        assert tr[time_col].max() <= va[time_col].min(), "Train/val step overlap"
    if len(va) and len(te):
        assert va[time_col].max() <= te[time_col].min(), "Val/test step overlap"
    if len(tr) and len(te):
        assert tr[time_col].max() <= te[time_col].min(), "Train/test step overlap"
    logger.info("Temporal split sizes: %s", meta["n"])
    return {"parts": parts, "labels": labels, "meta": meta}
