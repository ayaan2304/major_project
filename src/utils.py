"""
Shared utilities: paths, config, seeding, logging, JSON/CSV artifacts.

Why this module exists
----------------------
Every experiment must read the same YAML, use the same seed, and write
auditable artifacts. Centralising this prevents silent drift (different
seeds in different scripts) and makes leakage audits possible.

Leakage risks
-------------
None directly, but callers must never pass test-set objects into helpers
that persist "training-only" artifacts.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the single experiment config.

    Inputs: optional path (defaults to configs/config.yaml).
    Outputs: nested dict.
    Leakage notes: config must not contain test-set statistics.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_logger(name: str, log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Create a logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int) -> None:
    """Fix Python, NumPy seeds. Estimators still receive random_state explicitly."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    root = PROJECT_ROOT / cfg["paths"]["results_dir"]
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
        "metrics": root / "metrics",
        "logs": root / "logs",
        "shap": root / "shap",
        "processed": PROJECT_ROOT / cfg["paths"]["processed_dir"],
    }
    for p in paths.values():
        ensure_dir(p)
    return paths


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def raw_csv_path(cfg: Mapping[str, Any]) -> Path:
    return PROJECT_ROOT / cfg["paths"]["raw_csv"]
