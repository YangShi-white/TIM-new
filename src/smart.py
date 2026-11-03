"""Uncertainty-aware transfer with a label-free validation signal.

The problem
-----------
Selecting among candidate transfer operators (ridge strength, number of PLS
components, which source instrument) normally requires target labels.  With zero
target labels, practitioners default to scoring on the standardization samples -
which research-log F6 showed is biased by a median factor 2.06 and uncorrelated
with the true error (r = 0.008).

The estimator used here
-----------------------
For a *pair* of standardization samples i and j, the reference concentrations are
known from the source laboratory.  Therefore the transferred model's prediction can
be compared against a *difference-based* target-free quantity:

    for two standards i, j with reference values y_i, y_j,
    a correct transfer must reproduce the concentration DIFFERENCE:
        f(x_j) - f(x_i)  ~=  y_j - y_i
    and must NOT depend on the unknown absolute offset of the target response.

A model that over-fits the standardization set reproduces y_i for the samples it saw
but fails to reproduce differences between held-out standards.  Scoring on pairwise
differences out-of-fold therefore penalises over-fitting without needing a target
reference value at the prediction time.

The width of the empirical distribution of these difference residuals is also a
direct estimate of the *prediction* error, not the fitting error: this is what makes
it usable as a conformal scale.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def fit_operator(Xt_std, Xs_std, ridge=1e-2):
    U, s, Vt = np.linalg.svd(Xt_std, full_matrices=False)
    lam = ridge * np.sum(s ** 2) / Xt_std.shape[1]
    return (Vt.T * (s / (s ** 2 + lam))) @ (U.T @ Xs_std)


def _affine(p, y):
    A = np.vstack([p, np.ones_like(p)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


def _q_conf(scores, alpha):
    """Finite-sample split-conformal quantile."""
    n = len(scores)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, lvl, method="higher"))


def pairwise_validation(pred, y):
    """Scale-free disagreement between predicted and reference *differences*.

    Returns per-analyte RMSE of (p_i - p_j) - (y_i - y_j) over all pairs, normalised
    by the reference spread so it is comparable across analytes.
    """
    n, m = pred.shape
    if n < 3:
        return np.full(m, np.nan)
    iu, ju = np.triu_indices(n, k=1)
    dp = pred[iu] - pred[ju]
    dy = y[iu] - y[ju]
    return np.sqrt(np.mean((dp - dy) ** 2, axis=0)) / (np.std(y, axis=0) + 1e-12)


class SMART:
    """Source Model Adapted by Residual Transfer.

    Fits a family of candidate transfer operators, selects among them using the
    out-of-fold *pairwise* criterion (no target reference values), then calibrates a
    conformal interval on the same out-of-fold residuals.
    """

    def __init__(self, nc_grid=(4, 6, 8, 10, 12), ridge_grid=(1e-3, 1e-2, 1e-1),
                 alpha=0.10, folds=5, seed=0, interval_mode="split"):
        self.nc_grid = nc_grid
        self.ridge_grid = ridge_grid
        self.alpha = alpha
        self.folds = folds
        self.seed = seed
        self.interval_mode = interval_mode

    def _fit_one(self, Xs, Ys, Xt_std, Xs_std, Ys_std, nc, ridge):
        # clip n_components to what the smaller matrix supports (Kaiser-style
        # single-replicate instruments can leave very few source rows after the
        # row-level double-use exclusion)
        nc = min(nc, Xs.shape[0] - 1, Xs.shape[1])
        nc = max(1, nc)
        F = fit_operator(Xt_std, Xs_std, ridge)
        m = PLSRegression(n_components=nc, scale=False).fit(Xs, Ys)
        pf = m.predict(Xt_std @ F)
        aff = np.array([_affine(pf[:, j], Ys_std[:, j]) for j in range(Ys_std.shape[1])])
        return F, m, aff

    @staticmethod
    def _apply(X, F, m, aff):
        p = m.predict(X @ F)
        return np.stack([aff[j, 0] * p[:, j] + aff[j, 1] for j in range(p.shape[1])], axis=1)

    def _oof(self, Xs, Ys, Xt_std, Xs_std, Ys_std, nc, ridge):
        """Out-of-fold predictions on the standardization samples."""
        n = len(Xt_std)
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), self.folds)
        pred = np.full((n, Ys_std.shape[1]), np.nan)
        for f in folds:
            keep = np.setdiff1d(np.arange(n), f)
            if len(keep) < 6 or len(f) == 0:
                continue
            F, m, aff = self._fit_one(Xs, Ys, Xt_std[keep], Xs_std[keep], Ys_std[keep],
                                      nc, ridge)
            pred[f] = self._apply(Xt_std[f], F, m, aff)
        good = ~np.isnan(pred).any(axis=1)
        return pred[good], Ys_std[good], good

    def fit(self, Xs, Ys, Xt_std, Xs_std, Ys_std, std_src_rows=None):
        """std_src_rows: row indices in Xs/Ys belonging to the standardization
        samples' source spectra. The caller is responsible for having removed them
        (double-use fix, review #3); if passed here they are removed again
        defensively before the PLS is fitted."""
        if std_src_rows is not None and len(std_src_rows):
            Xs_fit = np.delete(Xs, std_src_rows, axis=0)
            Ys_fit = np.delete(Ys, std_src_rows, axis=0)
        else:
            Xs_fit, Ys_fit = Xs, Ys
        best = None
        sweep = []
        for nc in self.nc_grid:
            for ridge in self.ridge_grid:
                if len(Xs_fit) < 8 or len(Xt_std) < 8:
                    continue
                try:
                    pred, ytrue, _ = self._oof(Xs_fit, Ys_fit, Xt_std, Xs_std, Ys_std, nc, ridge)
                except Exception:  # noqa: BLE001
                    continue
                if len(pred) < 6:
                    continue
                score = float(np.mean(pairwise_validation(pred, ytrue)))
                sweep.append({"nc": nc, "ridge": ridge, "pairwise": score})
                if best is None or score < best["pairwise"]:
                    best = {"nc": nc, "ridge": ridge, "pairwise": score,
                            "pred": pred, "ytrue": ytrue}
        if best is None:
            raise RuntimeError("no candidate transfer operator could be fitted")
        self.sweep = sweep
        self.nc, self.ridge = best["nc"], best["ridge"]

        # --- interval calibration ------------------------------------------------
        # Post-fix measurement (research-log Round 18): calibrating the quantile on
        # cross-fold residuals of the standardization set UNDER-covers the target's
        # held-out samples (0.85-0.87 at nominal 0.90) because the standardization
        # population is systematically easier than the test population. The one
        # configuration that achieves nominal coverage is the disjoint SPLIT: fit the
        # operator on one half of the standards, calibrate the quantile on the other
        # half. It costs width but it is the only certifiable mode, so it is the
        # default; the cv+ residuals are retained for the hyper-parameter sweep only.
        resid = np.abs(best["pred"] - best["ytrue"])
        if self.interval_mode == "split":
            n_all = len(Xt_std)
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n_all)
            n_cal = max(5, min(n_all // 2, n_all - 6))
            cal, fitr = perm[:n_cal], perm[n_cal:]
            F2, m2, aff2 = self._fit_one(Xs_fit, Ys_fit, Xt_std[fitr],
                                         Xs_std[fitr], Ys_std[fitr],
                                         self.nc, self.ridge)
            p_cal = self._apply(Xt_std[cal], F2, m2, aff2)
            r = np.abs(p_cal - Ys_std[cal])
            self.q = np.array([_q_conf(r[:, j], self.alpha)
                               for j in range(r.shape[1])])
            self.n_cal = n_cal
        else:  # "cv+" retained for comparison against the split default
            n = len(resid)
            denom = n + 1 - self.folds
            if denom > 0:
                a_eff = min(0.5, self.alpha * (n + 1) / denom)
                self.q = np.quantile(resid, 1 - a_eff, axis=0, method="higher")
            else:
                self.q = np.full(resid.shape[1], np.inf)
            self.n_cal = n

        self.F, self.m, self.aff = self._fit_one(
            Xs_fit, Ys_fit, Xt_std, Xs_std, Ys_std, self.nc, self.ridge)
        return self

    def predict(self, X):
        p = self._apply(X, self.F, self.m, self.aff)
        return p, np.stack([p - self.q, p + self.q], axis=-1)
