"""E3 - Interval-protocol upgrade.

The E4 diagnostic found that pooled under-coverage comes from IN-SAMPLE optimism
(calibrating the quantile on the samples that fitted the operator), not from a
"standard samples are easier" distribution mismatch. That distinction matters:

  if the cause were distribution mismatch, no amount of cross-fitting would help
  and the only safe protocol would be the wasteful disjoint split;

  if the cause is in-sample optimism, then any protocol whose calibration
  residuals are genuinely out-of-fold should reach nominal coverage WITHOUT
  throwing away half the standards - which gives narrower intervals for free.

This script measures exactly that. All protocols share the same point predictor
(operator + affine fitted on ALL standardization samples) unless stated; they differ
only in HOW the conformal quantile is obtained.

Protocols
---------
  pooled        in-sample residuals, full sample           (current practice)
  split         disjoint halves, operator refit on half     (current SMART default)
  cv+           K-fold cross-conformal                      (tested before)
  2fold         two disjoint halves, roles swapped, pooled  (uses all samples)
  jackknife+    leave-one-physical-sample-out               (nested, uses all samples)
  mondrian      per-analyte split of the cv+ residuals      (heavy-tail robustness)

Key metric: coverage at nominal 90% AND width, so a protocol that buys coverage by
inflating the interval is visible. Also reports the effective calibration-set size,
because split wastes half of it.
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


def _q(vals, alpha):
    n = len(vals)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(vals, lvl, method="higher"))


def _fit(Xs, Ys, Xt, Xs_std, Ys_std, nc=8, ridge=1e-2):
    F = fit_operator(Xt, Xs_std, ridge)
    m = PLSRegression(n_components=min(nc, Xs.shape[0] - 1, Xs.shape[1]),
                      scale=False).fit(Xs, Ys)
    pf = m.predict(Xt @ F)
    aff = np.array([_affine(pf[:, j], Ys_std[:, j]) for j in range(Ys_std.shape[1])])
    return F, m, aff


def _apply(X, F, m, aff):
    p = m.predict(X @ F)
    return np.stack([aff[j, 0] * p[:, j] + aff[j, 1] for j in range(p.shape[1])], axis=1)


def run(seeds=(0, 1, 2), min_std=18, max_pairs=None):
    data = td.load_all()
    names = list(data)
    rows = []
    npair = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            src, tgt = data[a], data[b]
            ia, ib = td.paired_unlabeled(src, tgt)
            if len(ia) < min_std:
                continue
            npair += 1
            if max_pairs and npair > max_pairs:
                break
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
                Xt_std, Xs_std, Ys_std = tgt.X[ib2], src.X[ia2], src.y[ia2]
                n = len(Xt_std)
                rng = np.random.default_rng(seed)

                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(Xte) - Yte) ** 2, axis=0))

                # full-sample point predictor (all protocols below share it except split)
                Ff, mf, afff = _fit(Xs, Ys, Xt_std, Xs_std, Ys_std)
                p_full = _apply(Xte, Ff, mf, afff)
                p_std_full = _apply(Xt_std, Ff, mf, afff)

                protocols = {}

                # ---- pooled: in-sample residuals on ALL standards ---------------
                protocols["pooled"] = (
                    p_full,
                    np.array([_q(np.abs(p_std_full[:, j] - Ys_std[:, j]), ALPHA)
                              for j in range(Ys_std.shape[1])]),
                    n)

                # ---- split: operator refit on half, quantile from the other -----
                perm = rng.permutation(n)
                m_cal = max(5, min(n // 2, n - 6))
                cal, fitr = perm[:m_cal], perm[m_cal:]
                Fs, ms, affs = _fit(Xs, Ys, Xt_std[fitr], Xs_std[fitr], Ys_std[fitr])
                r_split = np.abs(_apply(Xt_std[cal], Fs, ms, affs) - Ys_std[cal])
                protocols["split"] = (
                    _apply(Xte, Fs, ms, affs),
                    np.array([_q(r_split[:, j], ALPHA) for j in range(Ys_std.shape[1])]),
                    len(cal))

                # ---- cv+ : K-fold cross-conformal -------------------------------
                K = min(5, max(2, n // 6))
                folds = np.array_split(rng.permutation(n), K)
                oof = np.full((n, Ys_std.shape[1]), np.nan)
                for f in folds:
                    kk = np.setdiff1d(np.arange(n), f)
                    if len(kk) < 8 or len(f) == 0:
                        continue
                    Fk, mk, ak = _fit(Xs, Ys, Xt_std[kk], Xs_std[kk], Ys_std[kk])
                    oof[f] = _apply(Xt_std[f], Fk, mk, ak)
                g = ~np.isnan(oof).any(axis=1)
                r_cv = np.abs(oof[g] - Ys_std[g])
                protocols["cv+"] = (p_full, np.array([_q(r_cv[:, j], ALPHA)
                                                      for j in range(Ys_std.shape[1])]),
                                    int(g.sum()))

                # ---- 2-fold: two halves, roles swapped, residuals pooled --------
                h = n // 2
                A, B = perm[:h], perm[h:]
                FA, mA, aA = _fit(Xs, Ys, Xt_std[B], Xs_std[B], Ys_std[B])
                FB, mB, aB = _fit(Xs, Ys, Xt_std[A], Xs_std[A], Ys_std[A])
                r2 = np.vstack([np.abs(_apply(Xt_std[A], FA, mA, aA) - Ys_std[A]),
                                np.abs(_apply(Xt_std[B], FB, mB, aB) - Ys_std[B])])
                protocols["2fold"] = (p_full, np.array([_q(r2[:, j], ALPHA)
                                                        for j in range(Ys_std.shape[1])]), n)

                # ---- jackknife+ : leave one standard out ------------------------
                loo = np.full((n, Ys_std.shape[1]), np.nan)
                for i in range(n):
                    kk = np.setdiff1d(np.arange(n), [i])
                    if len(kk) < 8:
                        continue
                    Fi, mi, ai = _fit(Xs, Ys, Xt_std[kk], Xs_std[kk], Ys_std[kk])
                    loo[i] = _apply(Xt_std[i:i + 1], Fi, mi, ai)[0]
                gl = ~np.isnan(loo).any(axis=1)
                r_loo = np.abs(loo[gl] - Ys_std[gl])
                protocols["jackknife+"] = (
                    p_full, np.array([_q(r_loo[:, j], ALPHA) for j in range(Ys_std.shape[1])]),
                    int(gl.sum()))

                # ---- mondrian: quantile per analyte on the cv+ residuals --------
                protocols["mondrian"] = (p_full, np.array([_q(r_cv[:, j], ALPHA)
                                                           for j in range(Ys_std.shape[1])]),
                                         int(g.sum()))

                for name, (p, q, ncal) in protocols.items():
                    iv_lo = p - q
                    iv_hi = p + q
                    for j, lab in enumerate(td.LABELS):
                        e = np.sqrt(np.mean((p[:, j] - Yte[:, j]) ** 2))
                        rows.append({
                            "source": a, "target": b, "seed": seed, "protocol": name,
                            "analyte": lab, "n_std": n, "n_cal": ncal,
                            "rmse": float(e),
                            "ratio": float(e / loc_err[j]) if loc_err[j] > 0 else np.nan,
                            "coverage": float(np.mean((Yte[:, j] >= iv_lo[:, j]) &
                                                      (Yte[:, j] <= iv_hi[:, j]))),
                            "width": float(np.mean(2 * q[j])),
                            "infinite": bool(np.isinf(q[j])),
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "e3_protocols.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== coverage / width / effective calibration size (nominal 0.90) ===")
    print(df.groupby("protocol")[["coverage", "width", "n_cal"]]
          .agg(["mean", "median"]).round(3).to_string())
    print("\nunder-coverage rate (<0.85) and infinite-interval rate:")
    agg = df.groupby("protocol").agg(
        under85=("coverage", lambda s: (s < 0.85).mean()),
        under70=("coverage", lambda s: (s < 0.70).mean()),
        inf_frac=("infinite", "mean"))
    print(agg.round(3).to_string())
    print("\nper analyte coverage:")
    print(df.pivot_table(index="protocol", columns="analyte", values="coverage",
                         aggfunc="mean").round(3).to_string())
    print("\nwidth ratio vs pooled (narrower is better at equal coverage):")
    w = df.groupby("protocol")["width"].mean()
    print((w / w["pooled"]).round(3).to_string())
    return df


if __name__ == "__main__":
    run()
