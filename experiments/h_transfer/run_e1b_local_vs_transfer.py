"""Does local calibration on the standards beat transfer on BOTH axes?

E1 showed local PLS trained on the standardization samples (B1) is more accurate
than the transfer at n >= 20. But B1 in that comparison had no uncertainty statement,
so one could still argue transfer's value is the certified interval.

This script closes that gap: give B1 the same jackknife+ conformal layer and compare
on both axes at once.

  B1          local PLS on the standards                (point only)
  B1+j+       local PLS + jackknife+ conformal           (point + interval)
  SMART       transfer + jackknife+ conformal            (point + interval)
  oracle      full local calibration on all target data  (point + interval)

If B1+j+ matches or beats SMART on both accuracy and coverage, then transfer has no
advantage in this setting at all, and the paper must be reframed entirely around the
uncertainty protocol.
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
from smart import fit_operator, _affine, SMART

OUT = os.path.join(PROJ, "data", "results")
ALPHA = 0.10
STD_SIZES = [8, 12, 20, 30]
REPS = 4


def _q(vals, alpha):
    n = len(vals)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(vals, lvl, method="higher"))


def local_pls_jackknife(Xt_std, Ys_std):
    """B1 + jackknife+ : local PLS, with leave-one-standard-out conformal widths."""
    n = len(Ys_std)
    nc = max(1, min(6, n // 4))
    model = PLSRegression(n_components=nc, scale=False).fit(Xt_std, Ys_std)
    loo = np.full((n, Ys_std.shape[1]), np.nan)
    for i in range(n):
        kk = np.setdiff1d(np.arange(n), [i])
        if len(kk) < 4:
            continue
        m = PLSRegression(n_components=max(1, min(nc, len(kk) - 1)),
                          scale=False).fit(Xt_std[kk], Ys_std[kk])
        loo[i] = m.predict(Xt_std[i:i + 1])[0]
    g = ~np.isnan(loo).any(axis=1)
    r = np.abs(loo[g] - Ys_std[g])
    q = np.array([_q(r[:, j], ALPHA) for j in range(Ys_std.shape[1])])
    return model, q


def _transfer_jackknife(Xs, Ys, Xt_std, Xs_std, Ys_std, Xte, nc=8, ridge=1e-2):
    """Transfer operator + affine, with leave-one-standard-out conformal widths -
    exactly the interval style B1+jackknife+ gets, so the two are comparable."""
    n = len(Xt_std)
    F = fit_operator(Xt_std, Xs_std, ridge)
    mdl = PLSRegression(n_components=nc, scale=False).fit(Xs, Ys)
    pf = mdl.predict(Xt_std @ F)
    aff = np.array([_affine(pf[:, j], Ys_std[:, j]) for j in range(Ys_std.shape[1])])

    def apply(X, Fm=None, mm=None, am=None):
        Fx = F if Fm is None else Fm
        mx = mdl if mm is None else mm
        ax = aff if am is None else am
        p = mx.predict(X @ Fx)
        return np.stack([ax[j, 0] * p[:, j] + ax[j, 1] for j in range(p.shape[1])], axis=1)

    loo = np.full((n, Ys_std.shape[1]), np.nan)
    for i in range(n):
        kk = np.setdiff1d(np.arange(n), [i])
        if len(kk) < 8:
            continue
        Fi = fit_operator(Xt_std[kk], Xs_std[kk], ridge)
        pfi = mdl.predict(Xt_std[kk] @ Fi)
        ai = np.array([_affine(pfi[:, j], Ys_std[kk][:, j]) for j in range(Ys_std.shape[1])])
        loo[i] = apply(Xt_std[i:i + 1], Fi, mdl, ai)[0]
    g = ~np.isnan(loo).any(axis=1)
    r = np.abs(loo[g] - Ys_std[g])
    q = np.array([_q(r[:, j], ALPHA) for j in range(Ys_std.shape[1])])
    p = apply(Xte)
    return p, np.stack([p - q, p + q], axis=-1)


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
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in tk
                                           for r in src.y])
                if keep.sum() < 20:
                    continue
                Xs, Ys = src.X[keep], src.y[keep]
                Xte, Yte = tgt.X[tm], tgt.y[tm]
                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(Xte) - Yte) ** 2, axis=0))

                for n in STD_SIZES:
                    if n > len(ia2):
                        continue
                    for rep in range(REPS):
                        rng = np.random.default_rng(seed * 1000 + rep)
                        sel = rng.choice(len(ia2), n, replace=False)
                        Xt_s, Ys_s = tgt.X[ib2[sel]], src.y[ia2[sel]]

                        def rec(method, p, iv=None):
                            for j, lab in enumerate(td.LABELS):
                                e = np.sqrt(np.mean((p[:, j] - Yte[:, j]) ** 2))
                                r = {"source": a, "target": b, "seed": seed,
                                     "method": method, "analyte": lab, "n_std": n,
                                     "rmse": float(e),
                                     "ratio": float(e / loc_err[j]) if loc_err[j] > 0 else np.nan}
                                if iv is not None:
                                    r["coverage"] = float(np.mean(
                                        (Yte[:, j] >= iv[:, j, 0]) & (Yte[:, j] <= iv[:, j, 1])))
                                    r["width"] = float(np.mean(iv[:, j, 1] - iv[:, j, 0]))
                                rows.append(r)

                        # B1 point only
                        try:
                            nc = max(1, min(6, n // 4))
                            m1 = PLSRegression(n_components=nc, scale=False).fit(Xt_s, Ys_s)
                            rec("B1 local PLS", m1.predict(Xte))
                        except Exception:  # noqa: BLE001
                            pass

                        # B1 + jackknife+
                        try:
                            m1, q1 = local_pls_jackknife(Xt_s, Ys_s)
                            p1 = m1.predict(Xte)
                            rec("B1+jackknife+", p1,
                                np.stack([p1 - q1, p1 + q1], axis=-1))
                        except Exception:  # noqa: BLE001
                            pass

                        # Transfer with the SAME interval style as B1+jackknife+, and
                        # fixed hyper-parameters: this comparison is about the interval
                        # layer and the transfer-vs-local question, not about the
                        # hyper-parameter search (which E12 showed equals plain CV).
                        try:
                            p4, iv4 = _transfer_jackknife(
                                Xs, Ys, Xt_s, src.X[ia2[sel]], Ys_s, Xte)
                            rec("SMART", p4, iv4)
                        except Exception:  # noqa: BLE001
                            pass

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "e1b_local_vs_transfer.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== accuracy ratio (median) ===")
    print(df.pivot_table(index="method", columns="n_std", values="ratio",
                         aggfunc="median").round(3).to_string())
    sub = df.dropna(subset=["coverage"])
    if len(sub):
        print("\n=== coverage / width ===")
        print(sub.pivot_table(index="method", columns="n_std",
                              values=["coverage", "width"], aggfunc="mean").round(3).to_string())
    return df


if __name__ == "__main__":
    run()
