"""Full evaluation of locally-adaptive transfer intervals.

Compares three uncertainty modes on identical point predictions:

  fixed      symmetric +/- q, constant for every sample (simple conformal)
  knn-dist   width scaled by distance to the nearest standardization spectra
  pred-knn   width scaled by distance in predicted-concentration space

Reported: coverage at nominal 90%, mean width, and the Spearman correlation between
per-sample half-width and per-sample absolute error. That correlation is the claim:
an interval that varies is only useful if it varies *with the difficulty*.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(PROJ, "src"))

import timdata as td
import protocol as pr
from tccp import TCCP
from adaptive import AdaptiveTCCP

OUT = os.path.join(PROJ, "data", "results")
ALPHA = 0.10


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
                if len(ia2) < min_std or tm.sum() < 8:
                    continue
                # double-use fix, ROW-level (see run_headtohead.py for rationale)
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in tk
                                           for r in src.y])
                Xs, Ys = src.X[keep], src.y[keep]
                Xte, Yte = tgt.X[tm], tgt.y[tm]
                args = (Xs, Ys, tgt.X[ib2], src.X[ia2], src.y[ia2])

                models = [
                    ("fixed", lambda: TCCP(nc=8, alpha=ALPHA, mode="cv+", seed=seed)),
                    ("knn-dist", lambda: AdaptiveTCCP(alpha=ALPHA, seed=seed,
                                                      difficulty="knn-dist")),
                    ("pred-knn", lambda: AdaptiveTCCP(alpha=ALPHA, seed=seed,
                                                      difficulty="pred-knn")),
                ]
                for nm, mk in models:
                    try:
                        m = mk().fit(*args)
                        p, iv = m.predict(Xte)
                    except Exception as e:  # noqa: BLE001
                        print(f"  {nm} {a}->{b}: {type(e).__name__}: {e}")
                        continue
                    half = (iv[:, :, 1] - iv[:, :, 0]) / 2
                    err = np.abs(p - Yte)
                    for j, lab in enumerate(td.LABELS):
                        cov = float(np.mean(err[:, j] <= half[:, j]))
                        rho = (float(pd.Series(half[:, j]).corr(
                            pd.Series(err[:, j]), method="spearman"))
                            if np.std(half[:, j]) > 1e-12 else np.nan)
                        rows.append({
                            "source": a, "target": b, "seed": seed, "mode": nm,
                            "analyte": lab, "coverage": cov, "halfwidth": float(np.mean(half[:, j])),
                            "spearman_width_error": rho,
                            "width_cv": float(np.std(half[:, j]) / (np.mean(half[:, j]) + 1e-12)),
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "adaptive_intervals.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== coverage (nominal 0.90) ===")
    print(df.groupby("mode")["coverage"].agg(["mean", "std"]).round(3).to_string())
    print("\nunder-coverage (<0.85):")
    print(df.groupby("mode")["coverage"].apply(lambda s: (s < 0.85).mean()).round(3).to_string())
    print("\n=== does the width track the error? Spearman(halfwidth, |error|) ===")
    print(df.groupby("mode")["spearman_width_error"].agg(["mean", "median"]).round(3).to_string())
    print("\nfraction of cells with positive correlation:")
    print(df.groupby("mode")["spearman_width_error"]
          .apply(lambda s: (s > 0).mean()).round(3).to_string())
    print("\n=== mean width ===")
    print(df.groupby("mode")["halfwidth"].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    run()
