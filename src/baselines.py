"""Baseline calibration-transfer methods, all under the zero-target-label constraint.

Conventions: X are spectra (n, p) already SNV-normalised and on the canonical grid;
Y are concentrations (n, 3).  A transfer method sees:

    Xs, Ys   labelled source spectra
    Xt_unl   unlabelled target spectra from the same physical samples as a subset of Xs
             (the *standardization set*: paired by construction)
    Xt_tgt   target spectra to be predicted (labels never shown)

The distinction between Xt_unl and Xt_tgt matters: standardization samples are the
extra measurements a lab takes when qualifying a new instrument, and they are cheap
because they need no reference analytics if they come from an existing sample bank.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


# ---------------------------------------------------------------- source-only
class DirectTransfer:
    """Fit on source only, apply to target. The naive baseline."""

    name = "direct"

    def __init__(self, nc=8):
        self.nc = nc

    def fit(self, Xs, Ys, Xt_unl=None, **kw):
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        return self

    def predict(self, X):
        return self.m.predict(X)


# ------------------------------------------------------- classical standards
def _pds_windows(p, window):
    """Non-overlapping sliding windows covering every index exactly once.

    A fixed number of *strided* windows (the textbook PDS layout) leaves
    uncovered gaps whenever p is not an exact multiple, and those gaps were
    silently filled with an identity fallback, which is what made PDS perform
    worse than the naive baseline. Covering the axis completely removes that
    artefact; window is forced odd so each window is centred.
    """
    if window % 2 == 0:
        window += 1
    wins = []
    start = 0
    while start < p:
        b = min(start + window, p)
        a = max(0, b - window)
        wins.append((a, b))
        if b >= p:
            break
        start = b
    return wins


class PDS:
    """Piecewise Direct Standardisation.

    Standard formulation (Bouveresse & Massart): map TARGET spectra back into the
    SOURCE measurement space, then apply the source calibration unchanged.

        Xt_std  ->  Xs_std      via T   (T fits Xt_std @ T ~= Xs_std)
        predict = PLS_source(Xt_tgt @ T)

    The earlier version regressed in the opposite direction and, because the
    source model was then applied to source-space features that were only
    approximately reconstructed, it degraded below the naive baseline.
    """

    name = "PDS"

    def __init__(self, window=7, n_win=40, nc=8, ridge=1e-6):
        self.window, self.n_win, self.nc, self.ridge = window, n_win, nc, ridge

    def _build_T(self, Xt_std, Xs_std):
        """T such that Xt_std @ T ~= Xs_std  (target space -> source space)."""
        q = Xt_std.shape[1]   # target wavelengths (input)
        p = Xs_std.shape[1]   # source wavelengths (output)
        T = np.zeros((q, p))
        wins = _pds_windows(q, self.window)
        for a, b in wins:
            for j in range(a, min(b, p)):
                Z = Xt_std[:, a:b]
                yj = Xs_std[:, j]
                A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
                T[a:b, j] = np.linalg.solve(A, Z.T @ yj)
        # any source wavelength no window reached: nearest-neighbour fallback
        covered = np.abs(T).sum(axis=1) > 0
        if not covered.all():
            src_idx = np.where(covered)[0]
            if len(src_idx):
                for j in np.where(~covered)[0]:
                    nearest = src_idx[np.argmin(np.abs(src_idx - j))]
                    T[nearest, j] = 1.0
        return T

    def fit(self, Xs, Ys, Xt_unl=None, **kw):
        if Xt_unl is None:
            raise ValueError("PDS needs paired unlabelled target spectra")
        Xs_std = kw.get("Xs_std")
        if Xs_std is None:
            Xs_std = Xs
        self.T = self._build_T(Xt_unl, Xs_std)
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        return self

    def predict(self, X):
        return self.m.predict(X @ self.T)


class DS:
    """Direct Standardisation: one global linear map from target to source space.

    Xt_std @ F ~= Xs_std, then predict = PLS_source(Xt_tgt @ F).

    The ridge is RELATIVE to the mean squared singular value of the standardization
    matrix (lam = ridge * trace(Xt'Xt)/p), matching the convention in tccp.py. An
    earlier version used ridge=1e-3 against the raw Gram matrix, which is ~500x
    weaker in effective terms and made DS look far worse than it is.
    """

    name = "DS"

    def __init__(self, nc=8, ridge=1e-3):
        self.nc, self.ridge = nc, ridge

    def fit(self, Xs, Ys, Xt_unl=None, **kw):
        Xs_std = kw.get("Xs_std", Xs)
        q = Xt_unl.shape[1]
        lam = self.ridge * np.trace(Xt_unl.T @ Xt_unl) / q
        A = Xt_unl.T @ Xt_unl + lam * np.eye(q)
        self.F = np.linalg.solve(A, Xt_unl.T @ Xs_std)
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        return self

    def predict(self, X):
        return self.m.predict(X @ self.F)


class MeanCenterTransfer:
    """Subtract the mean-spectrum difference measured on the standardization set."""

    name = "mean-center"

    def __init__(self, nc=8):
        self.nc = nc

    def fit(self, Xs, Ys, Xt_unl=None, **kw):
        Xs_std = kw.get("Xs_std", Xs)
        self.delta = Xs_std.mean(0) - Xt_unl.mean(0)
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        return self

    def predict(self, X):
        return self.m.predict(X + self.delta)


# --------------------------------------------------------------- oracle rows
class InInstrument:
    """Train and test on the target instrument itself: full local calibration."""

    name = "oracle-local"

    def __init__(self, nc=8):
        self.nc = nc

    def fit(self, Xt_fit, Yt_fit, **kw):
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xt_fit, Yt_fit)
        return self

    def predict(self, X):
        return self.m.predict(X)
