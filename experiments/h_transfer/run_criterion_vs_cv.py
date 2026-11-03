"""Is the pairwise selection criterion any better than ordinary cross-validation?

Review concern M12: after the affine correction the pairwise criterion is
algebraically close to sqrt(2)*RMS(out-of-fold residual)/std(y) - i.e. it IS a CV
criterion. The question that decides whether it belongs in the paper is narrower:
given an IDENTICAL nested protocol, does it rank candidates better than plain CV?

Protocol (fully nested, identical folds for both criteria):
  for each (nc, ridge) candidate:
      for each of 5 folds:
          fit operator + PLS + affine on the other folds only
          predict held-out fold
      criterion A (pairwise): RMSE((p_i-p_j)-(y_i-y_j))/sd(y) over out-of-fold
      criterion B (CV)      : RMSE(p - y)/sd(y)                 over out-of-fold
  compare each criterion's Spearman correlation with the TRUE held-out ratio,
  and whether the two criteria pick the same candidate.
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
from smart import fit_operator, _affine

OUT = os.path.join(PROJ, "data", "results")
NC_GRID = (4, 6, 8, 10, 12)
RIDGE_GRID = (1e-3, 1e-2, 1e-1)


def _fit(Xs, Ys, Xt, Xs_std, Ys_std, nc, ridge):
    F = fit_operator(Xt, Xs_std, ridge)
    m = PLSRegression(n_components=min(nc, Xs.shape[0] - 1, Xs.shape[1]),
                      scale=False).fit(Xs, Ys)
    pf = m.predict(Xt @ F)
    aff = np.array([_affine(pf[:, j], Ys_std[:, j]) for j in range(Ys_std.shape[1])])
    return F, m, aff


def _apply(X, F, m, aff):
    p = m.predict(X @ F)
    return np.stack([aff[j, 0] * p[:, j] + aff[j, 1] for j in range(p.shape[1])], axis=1)


def run(seeds=(0,), min_std=20):
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
                Xt_std, Xs_std, Ys_std = tgt.X[ib2], src.X[ia2], src.y[ia2]
                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(tgt.X[tm]) - tgt.y[tm]) ** 2, axis=0))

                n = len(ia2)
                rng = np.random.default_rng(seed)
                folds = np.array_split(rng.permutation(n), min(5, max(2, n // 4)))
                sd = Ys_std.std(axis=0) + 1e-12

                for nc in NC_GRID:
                    for rg in RIDGE_GRID:
                        oof = np.full((n, Ys_std.shape[1]), np.nan)
                        for f in folds:
                            kk = np.setdiff1d(np.arange(n), f)
                            if len(kk) < 6 or len(f) == 0:
                                continue
                            F, m, aff = _fit(Xs, Ys, Xt_std[kk], Xs_std[kk],
                                             Ys_std[kk], nc, rg)
                            oof[f] = _apply(Xt_std[f], F, m, aff)
                        g = ~np.isnan(oof).any(axis=1)
                        if g.sum() < 6:
                            continue
                        P, Y = oof[g], Ys_std[g]
                        # A: pairwise (same out-of-fold predictions)
                        iu, ju = np.triu_indices(len(P), k=1)
                        dp = P[iu] - P[ju]
                        dy = Y[iu] - Y[ju]
                        crit_pair = float(np.mean(np.sqrt(np.mean((dp - dy) ** 2, 0)) / sd))
                        # B: plain CV on the same out-of-fold predictions
                        crit_cv = float(np.mean(np.sqrt(np.mean((P - Y) ** 2, 0)) / sd))

                        F, m, aff = _fit(Xs, Ys, Xt_std, Xs_std, Ys_std, nc, rg)
                        pred = _apply(tgt.X[tm], F, m, aff)
                        true_ratio = float(np.mean(
                            np.sqrt(np.mean((pred - tgt.y[tm]) ** 2, 0)) / loc_err))
                        rows.append({"source": a, "target": b, "seed": seed,
                                     "nc": nc, "ridge": rg,
                                     "crit_pair": crit_pair, "crit_cv": crit_cv,
                                     "true_ratio": true_ratio})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "criterion_vs_cv.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")

    for col in ("crit_pair", "crit_cv"):
        taus, regs = [], []
        for _, g in df.groupby(["source", "target", "seed"]):
            if len(g) < 4:
                continue
            taus.append(g[col].corr(g.true_ratio, method="spearman"))
            pick = g.loc[g[col].idxmin(), "true_ratio"]
            best, worst = g.true_ratio.min(), g.true_ratio.max()
            regs.append((pick - best) / (worst - best + 1e-12))
        taus = np.array([t for t in taus if not np.isnan(t)])
        regs = np.array(regs)
        print(f"{col:10} spearman median {np.median(taus):+.3f} | "
              f"regret median {np.median(regs):.3f} mean {regs.mean():.3f} | groups {len(taus)}")

    agree = tot = 0
    for _, g in df.groupby(["source", "target", "seed"]):
        if len(g) < 4:
            continue
        tot += 1
        a1 = tuple(g.loc[g.crit_pair.idxmin(), ["nc", "ridge"]])
        a2 = tuple(g.loc[g.crit_cv.idxmin(), ["nc", "ridge"]])
        agree += int(a1 == a2)
    print(f"\nsame candidate picked by both criteria: {agree}/{tot} = {agree/max(tot,1):.3f}")
    return df


if __name__ == "__main__":
    run()
