"""Is the calibrated interval actually informative, or just wide?

A coverage claim is trivial to satisfy by reporting a huge interval. The useful
question is whether the interval width tracks the actual error: when the model is
about to be wrong, does it know?

For every held-out spectrum we compare
  * the half-width of its 90% interval, and
  * its absolute prediction error,
and report the Spearman correlation between them. A well-calibrated uncertainty
estimator should give a positive correlation well above zero.
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
import baselines as bl
from tccp import TCCP
from smart import SMART

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
                if len(ia2) < min_std or tm.sum() < 5:
                    continue
                keep = np.array([tuple(np.round(r, 5)) not in tk for r in src.y])
                Xs, Ys = src.X[keep], src.y[keep]
                Xte, Yte = tgt.X[tm], tgt.y[tm]
                for nm, mk in [("SMART", lambda: SMART(alpha=ALPHA, seed=seed)),
                               ("TCCP-cv+", lambda: TCCP(nc=8, alpha=ALPHA,
                                                         mode="cv+", seed=seed))]:
                    try:
                        m = mk().fit(Xs, Ys, tgt.X[ib2], src.X[ia2], src.y[ia2])
                        p, iv = m.predict(Xte)
                    except Exception:  # noqa: BLE001
                        continue
                    half = (iv[:, :, 1] - iv[:, :, 0]) / 2
                    err = np.abs(p - Yte)
                    for j, lab in enumerate(td.LABELS):
                        if np.std(half[:, j]) < 1e-12:
                            continue
                        rows.append({
                            "source": a, "target": b, "seed": seed,
                            "method": nm, "analyte": lab,
                            "spearman_width_vs_error": float(
                                pd.Series(half[:, j]).corr(
                                    pd.Series(err[:, j]), method="spearman")),
                            "mean_halfwidth": float(np.mean(half[:, j])),
                            "mean_abs_error": float(np.mean(err[:, j])),
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "interval_informativeness.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)")
    print("\nSpearman(width, |error|) - did the interval know?")
    print(df.groupby(["method", "analyte"])["spearman_width_vs_error"]
          .agg(["mean", "median"]).round(3).to_string())
    print("\noverall:")
    print(df.groupby("method")["spearman_width_vs_error"]
          .agg(["mean", "median"]).round(3).to_string())
    return df


if __name__ == "__main__":
    run()
