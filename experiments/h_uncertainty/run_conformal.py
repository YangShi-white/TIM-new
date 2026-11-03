"""Experiment: does transfer-calibrated conformal prediction deliver honest coverage?

For every ordered instrument pair we fit a transferred calibration and then ask what
a metrologist would ask: does the reported 90% interval actually contain the truth
90% of the time on samples the method has never seen?

We compare three calibration regimes, all using the *same* point predictions:
  pooled  - calibrate on the samples used to fit the transfer (current practice)
  split   - disjoint fit/calibration halves
  cv+     - cross-conformal, every standardization sample scored out-of-fold
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

OUT = os.path.join(PROJ, "data", "results")


def run(alpha=0.10, k=32, nc=8, seeds=(0, 1, 2), min_std=20):
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
                test_keys = {tuple(np.round(r, 5)) for r in tgt.y[tm]}
                # standardization samples must exclude anything the model will be scored on
                ok = np.array([tuple(np.round(r, 5)) not in test_keys for r in tgt.y[ib]])
                ia2, ib2 = ia[ok], ib[ok]
                if len(ia2) < min_std:
                    continue
                # double-use fix, ROW-level: remove exactly the source rows that the
                # standardization operator and affine correction see (indices ia2),
                # keeping other replicate spectra of the same physical samples.
                # Removing the whole physical sample would leave most pairs with no
                # source training data at all, since the shared design covers nearly
                # every source spectrum (verified: 235-270 of 270 rows per pair).
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in test_keys
                                           for r in src.y])
                if keep.sum() < 20:
                    continue
                Xs, Ys = src.X[keep], src.y[keep]
                Xt_std, Xs_std, Ys_std = tgt.X[ib2], src.X[ia2], src.y[ia2]

                for mode in ("pooled", "split", "cv+"):
                    try:
                        mdl = TCCP(nc=nc, alpha=alpha, mode=mode, seed=seed).fit(
                            Xs, Ys, Xt_std, Xs_std, Ys_std)
                        p, iv = mdl.predict(tgt.X[tm])
                    except Exception as e:  # noqa: BLE001
                        print(f"  {a}->{b} {mode} FAILED {type(e).__name__}: {e}")
                        continue
                    y = tgt.y[tm]
                    for j, lab in enumerate(td.LABELS):
                        cov = np.mean((y[:, j] >= iv[:, j, 0]) & (y[:, j] <= iv[:, j, 1]))
                        width = np.mean(iv[:, j, 1] - iv[:, j, 0])
                        err = np.abs(p[:, j] - y[:, j])
                        rows.append({
                            "source": a, "target": b, "mode": mode, "analyte": lab,
                            "seed": seed, "alpha": alpha, "n_std": len(ia2),
                            "coverage": float(cov), "width": float(width),
                            "rmse": float(np.sqrt(np.mean((p[:, j] - y[:, j]) ** 2))),
                            "mae": float(np.mean(err)),
                            "n_test": int(tm.sum()),
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"conformal_alpha{alpha}.csv")
    df.to_csv(p, index=False)
    print(f"\nsaved {p} ({len(df)} rows)")
    summ = df.groupby(["mode", "analyte"])[["coverage", "width", "rmse"]].mean().round(3)
    print(summ.to_string())
    print("\nmean coverage by mode:")
    print(df.groupby("mode")["coverage"].agg(["mean", "std", "min"]).round(3).to_string())
    print(f"\nnominal = {1-alpha:.2f}")
    return df


if __name__ == "__main__":
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    run(alpha=a)
