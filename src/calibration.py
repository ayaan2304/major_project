"""
Probability calibration (validation-fit, test-apply).

What / why
----------
Tree ensembles can be poorly calibrated. We report Brier, log-loss, ECE
and, if validation ECE is high, fit isotonic regression on validation
probabilities (prefit CalibratedClassifier-style).

Leakage risks
-------------
Isotonic/Platt must be fit on validation, never test, never the same
rows used to train the base model if we can avoid it. Here base models
train on train; calibrator fits on val.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(y_val, p_val) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso


def apply_isotonic(iso: IsotonicRegression, p) -> np.ndarray:
    import numpy as np

    return np.clip(iso.predict(p), 1e-7, 1 - 1e-7)
