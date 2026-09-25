"""
Chunked PaySim loader.

What / why
----------
PaySim is ~6.3M rows. Loading naively can spike RAM; EDA and quality
checks therefore stream the CSV in chunks. Modeling loaders still need
the full frame (or a documented subsample) in memory.

Inputs: config (path, chunksize, dtypes).
Outputs: iterator of DataFrames, or a full DataFrame.

Leakage risks
-------------
Loading does not split. Never fit transformers here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping

import pandas as pd

from src.utils import raw_csv_path

PAYSIM_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def iter_chunks(cfg: Mapping[str, Any], chunksize: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield CSV chunks. Used for Phase 0 quality reports on the full file."""
    path = raw_csv_path(cfg)
    size = int(chunksize or cfg["data"]["chunksize"])
    yield from pd.read_csv(path, chunksize=size, dtype=PAYSIM_DTYPES)


def load_full(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Load the entire PaySim table. Callers must still split before fitting."""
    path = raw_csv_path(cfg)
    return pd.read_csv(path, dtype=PAYSIM_DTYPES)
