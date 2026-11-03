"""Diagnostic: what makes a source instrument good to transfer FROM?

Research-log F2 found transfer quality is governed by the *source*, not the target
(Anton532 averages 3.87x local calibration, Timegate 0.92x).  A laboratory picking a
partner instrument needs to know this *without* target labels.

Candidate label-free indicators, all computable at fit time:

  s1  within-source calibration error, estimated by grouped cross-validation
  s2  spectral signal-to-noise of the analyte regression direction
  s3  concentration range / class balance of the standardisation samples
  s4  split-half reproducibility of the transfer operator

We test which ones correlate with the realised transfer ratio.
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
from tccp import TCCP, fit_operator
from baselines import InInstrument

OUT = os.path.join(PROJ, "data", "results")


def source_cv_error(src, nc=8, folds=5):
    g = np.unique(src.groups)
    rng = np.random.default_rng(0)
    rng.shuffle(g)
    errs = []
    for f in np.array_split(g, folds):
        te = np.isin(src.groups, f)
        if te.sum() < 3 or (~te).sum() < 10:
            continue
        m = PLSRegression(n_components=nc, scale=False).fit(src.X[~te], src.y[~te])
        errs.append(np.sqrt(np.mean((m.predict(src.X[te]) - src.y[te]) ** 2, axis=0)))
    return np.mean(errs, axis=0) if errs else np.full(3, np.nan)


def operator_reproducibility(Xs_std, Xt_std, ridge=1e-2, seed=0):
    """Split-half disagreement of the transferred prediction, scale-free."""
    n = len(Xt_std)
    if n < 8:
        return np.nan
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    h = n // 2
    h1, h2 = perm[:h], perm[h:]
    try:
        F1 = fit_operator(Xt_std[h1], Xs_std[h1], ridge)
        F2 = fit_operator(Xt_std[h2], Xs_std[h2], ridge)
    except Exception:  # noqa: BLE001
        return np.nan
    d = np.linalg.norm(Xt_std @ F1 - Xt_std @ F2) / (np.linalg.norm(Xs_std) + 1e-12)
    return float(d)


def run(seeds=(0, 1, 2)):
    data = td.load_all()
    names = list(data)
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            src, tgt = data[a], data[b]
            ia, ib = td.paired_unlabeled(src, tgt)
            if len(ia) < 20:
                continue
            s_cv = source_cv_error(src)
            for seed in seeds:
                fm, tm = pr.split_by_group(tgt, 0.3, seed)
                tk = {tuple(np.round(r, 5)) for r in tgt.y[tm]}
                ok = np.array([k not in tk for k in
                               [tuple(np.round(r, 5)) for r in tgt.y[ib]]])
                ia2, ib2 = ia[ok], ib[ok]
                if len(ia2) < 16:
                    continue
                # double-use fix, ROW-level (see run_headtohead.py for rationale)
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in tk
                                           for r in src.y])
                Xs, Ys = src.X[keep], src.y[keep]
                loc = InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(tgt.X[tm]) - tgt.y[tm]) ** 2, axis=0))
                try:
                    m = TCCP(nc=8, mode="cv+", seed=seed).fit(
                        Xs, Ys, tgt.X[ib2], src.X[ia2], src.y[ia2])
                    p, _ = m.predict(tgt.X[tm])
                except Exception:  # noqa: BLE001
                    continue
                err = np.sqrt(np.mean((p - tgt.y[tm]) ** 2, axis=0))
                rows.append({
                    "source": a, "target": b, "seed": seed,
                    "ratio": float(np.mean(err / loc_err)),
                    "src_cv_err": float(np.mean(s_cv)),
                    "op_repro": operator_reproducibility(src.X[ia2], tgt.X[ib2]),
                    "n_std": len(ia2),
                    "src_range": float(np.mean(src.y[ia2].std(0))),
                })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "source_diagnostic.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)")
    for col in ["src_cv_err", "op_repro", "n_std", "src_range"]:
        print(f"  corr({col:12}, ratio) = {df[col].corr(df['ratio']):+.3f}"
              f"   spearman = {df[col].corr(df['ratio'], method='spearman'):+.3f}")
    print("\nby source instrument:")
    print(df.groupby("source")[["ratio", "src_cv_err", "op_repro"]].mean().round(2).to_string())
    return df


if __name__ == "__main__":
    run()
