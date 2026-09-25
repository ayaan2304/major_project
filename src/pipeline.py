"""
Shared training/evaluation helpers used by experiment scripts.

Keeps experiment files thin: they orchestrate phases; this module
fits models and writes metrics.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.evaluation import classification_report_dict
from src.ensemble import predict_proba_positive
from src.models import fit_mlp_maybe_capped
from src.utils import get_logger

logger = get_logger("pipeline")


def fit_named_model(name: str, model, X, y, cfg: Mapping[str, Any], seed: int):
    t0 = time.perf_counter()
    extra = {}
    if name == "mlp":
        model, extra = fit_mlp_maybe_capped(model, X, y, cfg, seed)
    else:
        model.fit(X, y)
    extra["fit_seconds"] = time.perf_counter() - t0
    logger.info("Fitted %s in %.1fs on %s rows", name, extra["fit_seconds"], f"{len(X):,}")
    return model, extra


def eval_model(name: str, model, X, y, threshold: float = 0.5) -> dict[str, Any]:
    t0 = time.perf_counter()
    proba = predict_proba_positive(model, X)
    infer = time.perf_counter() - t0
    metrics = classification_report_dict(y, proba, threshold=threshold)
    metrics["model"] = name
    metrics["inference_seconds"] = infer
    metrics["n_eval"] = int(len(y))
    return metrics, proba
