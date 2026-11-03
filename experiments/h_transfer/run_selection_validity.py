"""Does the label-free selection criterion actually pick good hyper-parameters?

This is the validity check for the algorithmic contribution. If the pairwise
out-of-fold criterion were no better than guesswork, SMART would just be TCCP with
extra compute.

For every instrument pair we sweep (nc, ridge), and for each candidate record
  * the pairwise criterion (computable with zero target labels), and
  * the true held-out error ratio (computable only because this is an offline study).
Then measure how closely the criterion ranks the candidates.
"""

from __future__ import annotations

import os
import sys
import itertools
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(PROJ, "src"))

import timdata as td
import protocol as pr
import baselines as bl
from smart import SMART

OUT = os.path.join(PROJ, "data", "results")
NC = [4, 6, 8, 10, 12]
RIDGE = [1e-3, 1e-2, 1e-1]


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
                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(tgt.X[tm]) - tgt.y[tm]) ** 2, axis=0))
                for nc, rg in itertools.product(NC, RIDGE):
                    if nc >= min(len(Xs), len(ia2)):
                        continue
                    m = SMART(nc_grid=(nc,), ridge_grid=(rg,), seed=seed)
                    try:
                        m.fit(Xs, Ys, tgt.X[ib2], src.X[ia2], src.y[ia2])
                    except Exception:  # noqa: BLE001
                        continue
                    p, _ = m.predict(tgt.X[tm])
                    err = np.sqrt(np.mean((p - tgt.y[tm]) ** 2, axis=0))
                    rows.append({
                        "source": a, "target": b, "seed": seed, "nc": nc, "ridge": rg,
                        "criterion": float(np.mean([s["pairwise"] for s in m.sweep])),
                        "true_ratio": float(np.mean(err / loc_err)),
                    })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "selection_validity.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)")

    # rank correlation between criterion and true ratio, per pair
    tau = []
    for (a, b, s), g in df.groupby(["source", "target", "seed"]):
        if len(g) < 4:
            continue
        tau.append(g["criterion"].corr(g["true_ratio"], method="spearman"))
    tau = np.array([t for t in tau if not np.isnan(t)])
    print(f"\nSpearman(criterion, true ratio) over {len(tau)} pair/seed groups:")
    print(f"  median {np.median(tau):+.3f}   mean {tau.mean():+.3f}   "
          f"frac positive {np.mean(tau > 0):.3f}")

    # regret: true error of criterion-selected candidate vs the true best candidate
    reg = []
    for (a, b, s), g in df.groupby(["source", "target", "seed"]):
        if len(g) < 4:
            continue
        pick = g.loc[g["criterion"].idxmin(), "true_ratio"]
        best = g["true_ratio"].min()
        worst = g["true_ratio"].max()
        reg.append((pick - best) / (worst - best + 1e-12))
    reg = np.array(reg)
    print(f"\nnormalised regret of label-free selection: median {np.median(reg):.3f}, "
          f"mean {reg.mean():.3f}")
    print("(0 = picked the true best hyper-parameters, 1 = picked the worst)")
    return df


if __name__ == "__main__":
    run()
