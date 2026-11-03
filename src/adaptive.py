"""Locally-adaptive conformal calibration.

A fixed-width interval q is valid on average but says nothing about an individual
sample: every prediction gets the same +/- q. In a process analyzer the interesting
question is not only "on average 90%" but "does the instrument tell me when THIS
measurement is unreliable?" - e.g. near the detection limit, or when the spectrum
sits outside the calibration's validity domain.

This module provides the standard fix: normalised / locally-weighted conformal
(Mondrian or scale-adaptive), where the interval is

    y_hat -/+ q * s(x)

and s(x) is a per-sample difficulty estimate. The quantile q is then computed on
the *normalised* residuals |r|/s, which restores coverage while allowing the width
to vary.

Two difficulty estimators are compared:

  kNN-dist   distance of the target spectrum to its k nearest standardization
             spectra, in latent space. A spectrum unlike anything in the
             standardization set cannot be trusted equally.
  pred-kNN   distance in the *predicted concentration* space to the nearest
             standardization sample. Near the edge of the calibrated range the
             error grows.

Both are label-free.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from smart import fit_operator, _affine


def _conformal_q(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, lvl, method="higher"))


class AdaptiveTCCP:
    def __init__(self, nc=8, ridge=1e-2, alpha=0.10, folds=5, seed=0,
                 difficulty="knn-dist", k=5):
        self.nc, self.ridge, self.alpha = nc, ridge, alpha
        self.folds, self.seed = folds, seed
        self.difficulty, self.k = difficulty, k

    # ------------------------------------------------------------- difficulty
    def _scores(self, X, ref):
        """Per-sample difficulty; always positive and on a comparable scale."""
        if self.difficulty == "pred-knn":
            p = self._apply(X)
            d = np.linalg.norm(p[:, None, :] - ref[None, :, :], axis=2)
            return np.sort(d, axis=1)[:, : self.k].mean(1)
        d = np.linalg.norm(X[:, None, :] - ref[None, :, :], axis=2)
        s = np.sort(d, axis=1)[:, : self.k].mean(1)
        return s + 1e-6

    def _apply(self, X):
        p = self.m.predict(X @ self.F)
        return np.stack([self.aff[j, 0] * p[:, j] + self.aff[j, 1]
                         for j in range(p.shape[1])], axis=1)

    def _fit_core(self, Xs, Ys, Xt_std, Xs_std, Ys_std):
        self.F = fit_operator(Xt_std, Xs_std, self.ridge)
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        pf = self.m.predict(Xt_std @ self.F)
        self.aff = np.array([_affine(pf[:, j], Ys_std[:, j])
                             for j in range(Ys_std.shape[1])])

    def fit(self, Xs, Ys, Xt_std, Xs_std, Ys_std):
        n = len(Xt_std)
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), min(self.folds, max(2, n // 4)))
        pred = np.full((n, Ys_std.shape[1]), np.nan)
        diff = np.full(n, np.nan)
        for f in folds:
            keep = np.setdiff1d(np.arange(n), f)
            if len(keep) < 6 or len(f) == 0:
                continue
            self._fit_core(Xs, Ys, Xt_std[keep], Xs_std[keep], Ys_std[keep])
            pred[f] = self._apply(Xt_std[f])
            if self.difficulty == "pred-knn":
                ref = self._apply(Xt_std[keep])
                diff[f] = self._scores_pred(Xt_std[f], ref)
            else:
                diff[f] = self._scores(Xt_std[f], Xt_std[keep])

        good = ~np.isnan(pred).any(axis=1) & ~np.isnan(diff)
        self._fit_core(Xs, Ys, Xt_std, Xs_std, Ys_std)
        self._ref = (self._apply(Xt_std) if self.difficulty == "pred-knn"
                     else Xt_std)
        self.q_raw = np.array([
            np.quantile(np.abs(pred[good][:, j] - Ys_std[good][:, j]), 1.0)
            for j in range(Ys_std.shape[1])])
        # normalised conformal quantile per analyte, with the non-degenerate CV+
        # correction (see tccp.py: inflate the miscoverage level by
        # (n+1)/(n+1-k) rather than shifting the rank, which saturates)
        s_good = diff[good]
        resid = np.abs(pred[good] - Ys_std[good])
        n_cal = int(good.sum())
        denom = n_cal + 1 - self.folds
        if denom > 0:
            a_eff = min(0.5, self.alpha * (n_cal + 1) / denom)
        else:
            a_eff = None
        if a_eff is None:
            self.q = np.full(resid.shape[1], np.inf)
        else:
            self.q = np.array([float(np.quantile(resid[:, j] / s_good, 1 - a_eff,
                                                method="higher"))
                               for j in range(resid.shape[1])])
        self.n_cal = n_cal
        return self

    def _scores_pred(self, X, ref_pred):
        p = self._apply(X)
        d = np.linalg.norm(p[:, None, :] - ref_pred[None, :, :], axis=2)
        return np.sort(d, axis=1)[:, : self.k].mean(1) + 1e-6

    def predict(self, X):
        p = self._apply(X)
        s = self._scores(X, self._ref)
        # the same scale enters all analytes; keep per-analyte quantiles
        half = self.q[None, :] * s[:, None]
        return p, np.stack([p - half, p + half], axis=-1)
