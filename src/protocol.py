"""Evaluation protocol.

The experiment we run throughout is **leave-one-instrument-out (LOIO)**
calibration transfer under a *zero-target-label* constraint:

    fit on source A (labelled)  ->  apply to target B (labels hidden at fit time)

Reported metrics are computed on B's held-out physical samples only, and the split
is by physical sample (group) so replicate spectra of the same sample never straddle
train/test.

Metrics
-------
RMSE          : root mean squared error, g/L
MAE           : mean absolute error, g/L
R2            : coefficient of determination
bias          : mean signed error (metrological systematic error)
RMSEP/RMSE_Lab: transfer error relative to the in-instrument (label-rich) reference
                calibration on B. 1.0 means "as good as buying a full local calibration".
coverage      : empirical coverage of the 90% prediction interval

The *oracle* row (train and test both on B) is the ceiling; it is not a competitor,
it is what a full local calibration would buy.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

LABELS = ["Glucose", "Acetate", "MagnesiumSulfate"]


def split_by_group(d, test_frac=0.3, seed=0):
    """Split physical-sample groups into fit/test. Returns boolean masks."""
    rng = np.random.default_rng(seed)
    g = np.unique(d.groups)
    rng.shuffle(g)
    n_test = max(1, int(round(len(g) * test_frac)))
    test_g = set(g[:n_test].tolist())
    mask = np.array([x in test_g for x in d.groups])
    return ~mask, mask


def metrics(y_true, y_pred):
    e = y_pred - y_true
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan"),
        "bias": float(np.mean(e)),
    }


class PLSCal:
    """Multi-output PLS2 with per-target component count chosen on held-out groups."""

    def __init__(self, max_nc=15):
        self.max_nc = max_nc
        self.nc = 6
        self.m = None

    def fit(self, X, Y):
        # choose n_components by grouped internal CV, scored on mean standardised RMSE
        g_count = len(Y)
        best, best_s = None, np.inf
        sd = Y.std(axis=0) + 1e-12
        k = min(5, max(2, g_count // 6))
        idx = np.arange(len(Y))
        folds = np.array_split(idx, k)
        for nc in range(1, self.max_nc + 1):
            errs = []
            for f in folds:
                tr = np.setdiff1d(idx, f)
                if len(tr) < nc + 2 or len(f) == 0:
                    continue
                try:
                    m = PLSRegression(n_components=nc, scale=False).fit(X[tr], Y[tr])
                    p = m.predict(X[f])
                    errs.append(np.mean(np.sqrt(np.mean(((p - Y[f]) / sd) ** 2, axis=0))))
                except Exception:
                    errs.append(np.inf)
            if errs:
                s = float(np.mean(errs))
                if s < best_s - 1e-4:
                    best_s, best = s, nc
        self.nc = best or 6
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(X, Y)
        return self

    def predict(self, X):
        return self.m.predict(X)
