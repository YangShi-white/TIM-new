"""Leave-one-instrument-out transfer experiment.

For every ordered pair (source A, target B) we
  1. split B's *physical samples* into a calibration half and a held-out test half,
  2. take the standardization set = samples common to A and B (paired, unlabelled),
  3. fit a source model on A, transfer it to B using only unlabelled B spectra,
  4. score on B's held-out physical samples against the reference concentrations.

Every method sees exactly the same data and the same split. Results are written to
data/results/loio_<tag>.csv.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(PROJ, "src"))
import timdata as td
import protocol as pr
import baselines as bl

LABELS = td.LABELS
OUT_DIR = os.path.join(td.DATA_ROOT, "..", "results")


def run_pair(src: td.InstData, tgt: td.InstData, methods, test_frac=0.3, seed=0):
    # split target by physical sample
    fit_m, test_m = pr.split_by_group(tgt, test_frac, seed)

    # paired standardization set: samples present in both instruments
    ia, ib = td.paired_unlabeled(src, tgt)
    if len(ia) < 8:
        return []

    # paired standardization set: samples present in both instruments, EXCLUDING
    # any that fall in the target's held-out test half (review #11: the original
    # version used all paired spectra, letting the operator see test-half samples)
    fit_keys = {tuple(np.round(r, 5)) for r in tgt.y[fit_m]}
    ok = np.array([k in fit_keys for k in
                   [tuple(np.round(r, 5)) for r in tgt.y[ib]]])
    ia_f, ib_f = ia[ok], ib[ok]
    if len(ia_f) < 8:
        return []
    Xt_unl = tgt.X[ib_f]
    Xs_std = src.X[ia_f]

    # source training data: drop test-half samples AND the exact standardization
    # rows the operator/affine see (row-level double-use fix)
    test_keys = {tuple(np.round(r, 5)) for r in tgt.y[test_m]}
    rowmask = np.ones(len(src.y), dtype=bool)
    rowmask[ia_f] = False
    src_keep = rowmask & np.array([tuple(np.round(r, 5)) not in test_keys
                                   for r in src.y])
    Xs, Ys = src.X[src_keep], src.y[src_keep]

    # oracle: full local calibration on target's fit half
    oracle = bl.InInstrument(nc=8).fit(tgt.X[fit_m], tgt.y[fit_m])
    oracle_metrics = {lab: pr.metrics(tgt.y[test_m, j], oracle.predict(tgt.X[test_m])[:, j])
                      for j, lab in enumerate(LABELS)}

    rows = []
    for name, mk in methods.items():
        t0 = time.time()
        try:
            mdl = mk()
            mdl.fit(Xs, Ys, Xt_unl=Xt_unl, Xs_std=Xs_std)
            pred = mdl.predict(tgt.X[test_m])
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: FAILED {type(e).__name__}: {e}")
            continue
        for j, lab in enumerate(LABELS):
            m = pr.metrics(tgt.y[test_m, j], pred[:, j])
            o = oracle_metrics[lab]
            rows.append({
                "source": src.name, "target": tgt.name, "method": name,
                "analyte": lab, "n_std": len(ia_f), "n_test": int(test_m.sum()),
                **m,
                "rmse_ratio_local": m["rmse"] / o["rmse"] if o["rmse"] > 0 else np.nan,
                "sec": round(time.time() - t0, 2),
            })
    return rows


def main(tag="v1", seed=0, test_frac=0.3, instruments=None):
    data = td.load_all()
    names = instruments or list(data)
    methods = {
        "direct": lambda: bl.DirectTransfer(nc=8),
        "mean-center": lambda: bl.MeanCenterTransfer(nc=8),
        "DS": lambda: bl.DS(nc=8, ridge=1e-3),
        "PDS": lambda: bl.PDS(window=7, n_win=40, nc=8),
    }
    all_rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            print(f"[{a} -> {b}]", flush=True)
            rows = run_pair(data[a], data[b], methods, test_frac, seed)
            all_rows += rows
            for r in rows:
                print(f"    {r['method']:12} {r['analyte']:16} "
                      f"RMSE={r['rmse']:8.3f} R2={r['r2']:7.3f} "
                      f"ratio={r['rmse_ratio_local']:6.2f}", flush=True)
    df = pd.DataFrame(all_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"loio_{tag}.csv")
    df.to_csv(p, index=False)
    print(f"\nsaved {p} ({len(df)} rows)")
    print("\n=== mean over all pairs ===")
    print(df.groupby(["method", "analyte"])[["rmse", "r2", "rmse_ratio_local"]]
          .mean().round(3).to_string())
    return df


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    main(tag=tag)
