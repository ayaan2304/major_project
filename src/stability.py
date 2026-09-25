"""
Explanation stability across retraining runs.

Stability
---------
Retrain the provisional/final tree with different seeds. Rank features
by mean |SHAP| on a frozen train background / frozen explain set.
Report Spearman/Kendall on ranks and Jaccard overlap of top-k.

Leakage risks
-------------
Frozen explain set is drawn from train (or a stored train sample), never
re-chosen using test performance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def rank_stability(importance_runs: list[np.ndarray], topk: int = 10) -> dict[str, float]:
    """importance_runs: list of mean-|SHAP| vectors, same feature order."""
    ranks = [(-imp).argsort().argsort() for imp in importance_runs]
    spears, kends, jacs = [], [], []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            spears.append(float(spearmanr(ranks[i], ranks[j]).statistic))
            kends.append(float(kendalltau(ranks[i], ranks[j]).statistic))
            top_i = set((-importance_runs[i]).argsort()[:topk].tolist())
            top_j = set((-importance_runs[j]).argsort()[:topk].tolist())
            jacs.append(jaccard(top_i, top_j))
    return {
        "spearman_mean": float(np.mean(spears)) if spears else float("nan"),
        "spearman_std": float(np.std(spears)) if spears else float("nan"),
        "kendall_mean": float(np.mean(kends)) if kends else float("nan"),
        "kendall_std": float(np.std(kends)) if kends else float("nan"),
        "jaccard_top_mean": float(np.mean(jacs)) if jacs else float("nan"),
        "jaccard_top_std": float(np.std(jacs)) if jacs else float("nan"),
        "n_pairs": int(len(spears)),
        "topk": int(topk),
    }
