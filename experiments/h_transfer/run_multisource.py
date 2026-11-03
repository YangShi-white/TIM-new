"""Can several source instruments be combined without target labels?

A laboratory usually owns more than one calibrated instrument. If the transfer
operator is fitted to ONE source, it inherits that source's weaknesses (F2). Here we
test whether pooling several sources helps, and - more importantly - whether the
combination weight can be chosen without target labels.

Three fusion strategies:
  best-single   pick the source with the best label-free criterion value (SMART sweep)
  equal-weight  average the transferred predictions of all sources
  criterion-weight  weight each source by 1/criterion^2 (label-free)

The criterion is the same out-of-fold pairwise agreement used inside SMART, so no
target reference value is involved.
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
from smart import SMART

OUT = os.path.join(PROJ, "data", "results")


def run(seeds=(0, 1, 2), min_std=20, max_sources=7):
    data = td.load_all()
    names = list(data)
    rows = []
    for tgt_name in names:
        tgt = data[tgt_name]
        for seed in seeds:
            fm, tm = pr.split_by_group(tgt, 0.3, seed)
            tk = {tuple(np.round(r, 5)) for r in tgt.y[tm]}
            Xte, Yte = tgt.X[tm], tgt.y[tm]
            loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
            loc_err = np.sqrt(np.mean((loc.predict(Xte) - Yte) ** 2, axis=0))

            preds, crits, used = [], [], []
            for src_name in names:
                if src_name == tgt_name:
                    continue
                src = data[src_name]
                ia, ib = td.paired_unlabeled(src, tgt)
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
                try:
                    m = SMART(seed=seed).fit(src.X[keep], src.y[keep],
                                             tgt.X[ib2], src.X[ia2], src.y[ia2])
                    p, _ = m.predict(Xte)
                except Exception:  # noqa: BLE001
                    continue
                c = float(np.mean([s["pairwise"] for s in m.sweep]))
                preds.append(p)
                crits.append(c)
                used.append(src_name)

            if not preds:
                continue
            preds = np.stack(preds)
            crits = np.array(crits)

            def rec(method, p):
                for j, lab in enumerate(td.LABELS):
                    e = np.sqrt(np.mean((p[:, j] - Yte[:, j]) ** 2))
                    rows.append({"target": tgt_name, "seed": seed, "method": method,
                                 "analyte": lab, "ratio": float(e / loc_err[j]),
                                 "n_sources": len(preds),
                                 "sources": ",".join(used)})

            rec("single (criterion-best)", preds[int(np.argmin(crits))])
            rec("equal-weight fusion", preds.mean(0))
            w = 1.0 / (crits ** 2)
            rec("criterion-weight fusion", np.tensordot(w / w.sum(), preds, axes=1))
            # true oracle: the single source with the lowest real error (per analyte),
            # an upper reference that uses target labels and is never a competitor
            errs = np.stack([np.sqrt(np.mean((p - Yte) ** 2, axis=0)) for p in preds])
            for lab_idx, lab in enumerate(td.LABELS):
                best_pred = preds[int(np.argmin(errs[:, lab_idx]))]
                e = np.sqrt(np.mean((best_pred[:, lab_idx] - Yte[:, lab_idx]) ** 2))
                rows.append({"target": tgt_name, "seed": seed, "method": "oracle-best single",
                             "analyte": lab, "ratio": float(e / loc_err[lab_idx]),
                             "n_sources": len(preds), "sources": ",".join(used)})

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "multisource_fusion.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== ratio to full local calibration (lower better) ===")
    print(df.groupby("method")["ratio"].agg(["mean", "median"]).round(3).to_string())
    print("\n=== by target ===")
    print(df.pivot_table(index="target", columns="method", values="ratio",
                         aggfunc="mean").round(2).to_string())
    return df


if __name__ == "__main__":
    run()
