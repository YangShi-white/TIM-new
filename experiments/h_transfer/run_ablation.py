"""Ablation: which component of the proposed pipeline carries the benefit?

Each row switches ONE component off, keeping the rest identical:

  full        operator + affine correction + cross-conformal interval   (proposed)
  -affine     operator only, no slope/offset correction on standards
  -operator   no transfer at all: source model applied directly
  -cv          pooled conformal quantile (fit and calibrate on same samples)
  -labelsel    hyper-parameters fixed instead of chosen by the pairwise criterion
  window      windowed (PDS-style) operator instead of a global one

Reported per analyte and averaged: error ratio to a full local calibration, and the
90% interval coverage. The point of the table is to show that no single component
carries everything, and that the interval is not an afterthought.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(PROJ, "src"))

import timdata as td
import protocol as pr
import baselines as bl
from smart import SMART, fit_operator, _affine

OUT = os.path.join(PROJ, "data", "results")
ALPHA = 0.10


class Variant:
    def __init__(self, name, use_affine=True, use_operator=True, cv=True,
                 label_select=True, window=False, fixed=(8, 1e-2), seed=0):
        self.name = name
        self.use_affine = use_affine
        self.use_operator = use_operator
        self.cv = cv
        self.label_select = label_select
        self.window = window
        self.fixed = fixed
        self.seed = seed

    def _op(self, Xt_std, Xs_std):
        if not self.use_operator:
            return np.eye(Xt_std.shape[1])
        if not self.window:
            return fit_operator(Xt_std, Xs_std)
        # windowed operator: blocks of 20 wavenumbers (PDS-like)
        p = Xt_std.shape[1]
        F = np.zeros((p, p))
        for a in range(0, p, 20):
            b = min(a + 20, p)
            F[a:b, a:b] = fit_operator(Xt_std[:, a:b], Xs_std[:, a:b])
        return F

    def fit(self, Xs, Ys, Xt_std, Xs_std, Ys_std):
        if self.label_select:
            # SMART now defaults to the split interval; its q IS the split quantile,
            # so `full` inherits the certifiable interval, matching run_headtohead.
            sel = SMART(alpha=ALPHA, seed=self.seed, interval_mode="split").fit(
                Xs, Ys, Xt_std, Xs_std, Ys_std)
            self.nc, self.ridge, q_cv = sel.nc, sel.ridge, sel.q
            self.n_cal = sel.n_cal
        else:
            self.nc, self.ridge = self.fixed
        F = self._op(Xt_std, Xs_std)
        m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        pf = m.predict(Xt_std @ F)
        if self.use_affine:
            aff = np.array([_affine(pf[:, j], Ys_std[:, j]) for j in range(Ys_std.shape[1])])
        else:
            aff = np.array([[1.0, 0.0]] * Ys_std.shape[1])

        def apply(X):
            p = m.predict(X @ F)
            return np.stack([aff[j, 0] * p[:, j] + aff[j, 1]
                             for j in range(p.shape[1])], axis=1)

        if self.label_select:
            self.q = q_cv
        else:
            if self.cv:
                # disjoint split interval, matching SMART's default mode: fit the
                # operator on half the standards, calibrate the quantile on the
                # other half. The only empirically certifiable mode
                # (pooled 0.838 / cv+ 0.853 / split 0.944 at nominal 0.90).
                n = len(Xt_std)
                rng = np.random.default_rng(self.seed)
                perm = rng.permutation(n)
                n_cal = max(5, min(n // 2, n - 6))
                cal, fitr = perm[:n_cal], perm[n_cal:]
                if len(fitr) >= 6:
                    Fk = self._op(Xt_std[fitr], Xs_std[fitr])
                    mk = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
                    pk = mk.predict(Xt_std[fitr] @ Fk)
                    ak = np.array([_affine(pk[:, j], Ys_std[fitr][:, j])
                                   for j in range(Ys_std.shape[1])])
                    pf2 = mk.predict(Xt_std[cal] @ Fk)
                    p_cal = np.stack([ak[j, 0] * pf2[:, j] + ak[j, 1]
                                      for j in range(pf2.shape[1])], axis=1)
                    resid = np.abs(p_cal - Ys_std[cal])
                else:
                    resid = np.abs(apply(Xt_std) - Ys_std)
            else:
                resid = np.abs(apply(Xt_std) - Ys_std)
            n = len(resid)
            lvl = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
            self.q = np.quantile(resid, lvl, axis=0, method="higher")
        self._apply = apply
        return self

    def predict(self, X):
        p = self._apply(X)
        return p, np.stack([p - self.q, p + self.q], axis=-1)


VARIANTS = [
    ("full", dict()),
    ("-affine", dict(use_affine=False)),
    ("-operator", dict(use_operator=False)),
    ("-cv (pooled)", dict(label_select=False, cv=False)),
    ("-labelsel", dict(label_select=False)),
    ("windowed op", dict(window=True)),
]


def run(seeds=(0, 1, 2), min_std=20):
    data = td.load_all()
    names = list(data)
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            src, tgt = data[a], data[b]
            ia, ib = td.paired_unlabeled(src, tgt)
            if len(ia) < min_std:
                continue
            for seed in seeds:
                fm, tm = pr.split_by_group(tgt, 0.3, seed)
                tk = {tuple(np.round(r, 5)) for r in tgt.y[tm]}
                ok = np.array([k not in tk for k in
                               [tuple(np.round(r, 5)) for r in tgt.y[ib]]])
                ia2, ib2 = ia[ok], ib[ok]
                if len(ia2) < min_std:
                    continue
                # double-use fix, ROW-level (see run_headtohead.py for rationale)
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in tk
                                           for r in src.y])
                Xs, Ys = src.X[keep], src.y[keep]
                Xte, Yte = tgt.X[tm], tgt.y[tm]
                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(Xte) - Yte) ** 2, axis=0))
                for nm, kw in VARIANTS:
                    try:
                        m = Variant(nm, seed=seed, **kw).fit(
                            Xs, Ys, tgt.X[ib2], src.X[ia2], src.y[ia2])
                        p, iv = m.predict(Xte)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {nm} {a}->{b} FAILED {type(e).__name__}: {e}")
                        continue
                    for j, lab in enumerate(td.LABELS):
                        e = np.sqrt(np.mean((p[:, j] - Yte[:, j]) ** 2))
                        rows.append({
                            "source": a, "target": b, "seed": seed, "variant": nm,
                            "analyte": lab, "ratio": float(e / loc_err[j]),
                            "coverage": float(np.mean((Yte[:, j] >= iv[:, j, 0]) &
                                                      (Yte[:, j] <= iv[:, j, 1]))),
                            "width": float(np.mean(iv[:, j, 1] - iv[:, j, 0])),
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "ablation.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== error ratio to full local calibration (lower better) ===")
    print(df.groupby("variant")["ratio"].agg(["mean", "median", "std"]).round(3).to_string())
    print("\n=== 90% interval coverage (nominal 0.90) ===")
    print(df.groupby("variant")["coverage"].agg(["mean", "std"]).round(3).to_string())
    print("\nundercoverage (<0.85) rate:")
    print(df.groupby("variant")["coverage"].apply(lambda s: (s < 0.85).mean()).round(3).to_string())
    return df


if __name__ == "__main__":
    run()
