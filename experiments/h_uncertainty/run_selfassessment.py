"""The self-assessment defect, measured under ONE named definition.

Review defect #2: the headline numbers for the self-assessment bias (median 2.06,
worst 10.4, r = 0.008) could not be traced to any script, and different plausible
definitions give materially different values. This script computes the defect under
its single most natural definition and writes the CSV that the paper must quote:

    DEFINITION (self-assessment, as practitioners perform it):
      fit the full transfer (operator + affine correction) on ALL paired
      standardization samples, then score the transferred model on those SAME
      samples. Compare with the same model's error on the held-out target samples.

The source model does NOT see the standardization samples' source spectra
(std_src_rows are excluded — see the double-use fix), so the contrast here isolates
the operator/affine self-assessment, not the PLS leak.

Outputs data/results/selfassessment.csv with per-pair/seed/analyte values, plus a
summary that findings.md quotes verbatim.
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
                # --- exclude std source rows from the PLS training set ----------
                # keep ALL other source rows (test-half keys and std keys removed);
                # some source spectra of std samples are replicates not in ia2, so
                # exclusion is by concentration-tuple, which covers all replicates.
                std_keys = {tuple(np.round(r, 5)) for r in src.y[ia2]}
                keep = np.array([tuple(np.round(r, 5)) not in tk
                                 and tuple(np.round(r, 5)) not in std_keys
                                 for r in src.y])
                if keep.sum() < 20:  # too few rows left to fit a sane PLS
                    continue
                Xs, Ys = src.X[keep], src.y[keep]

                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(tgt.X[tm]) - tgt.y[tm]) ** 2, axis=0))

                F = fit_operator(tgt.X[ib2], src.X[ia2])
                model = PLSRegression(n_components=8, scale=False).fit(Xs, Ys)
                pf = model.predict(tgt.X[ib2] @ F)
                aff = np.array([_affine(pf[:, j], src.y[ia2][:, j])
                                for j in range(Ys.shape[1])])

                def apply(X):
                    p = model.predict(X @ F)
                    return np.stack([aff[j, 0] * p[:, j] + aff[j, 1]
                                     for j in range(p.shape[1])], axis=1)

                p_in = apply(tgt.X[ib2])          # self-assessment, in-sample
                p_out = apply(tgt.X[tm])          # truth, held-out
                for j, lab in enumerate(td.LABELS):
                    e_in = np.sqrt(np.mean((p_in[:, j] - tgt.y[ib2][:, j]) ** 2))
                    e_out = np.sqrt(np.mean((p_out[:, j] - tgt.y[tm][:, j]) ** 2))
                    rows.append({
                        "source": a, "target": b, "seed": seed, "analyte": lab,
                        "e_in": float(e_in), "e_out": float(e_out),
                        "e_local": float(loc_err[j]),
                        "ratio_in": float(e_in / loc_err[j]),
                        "ratio_out": float(e_out / loc_err[j]),
                        "understate_factor": float(e_out / max(e_in, 1e-12)),
                    })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "selfassessment.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")

    uf = df["understate_factor"]
    r = df["ratio_in"].corr(df["ratio_out"])
    rho = df["ratio_in"].corr(df["ratio_out"], method="spearman")
    print("=== SELF-ASSESSMENT DEFECT (single named definition) ===")
    print(f"cells                                   : {len(df)}")
    print(f"median underestimation e_out/e_in       : {uf.median():.2f}")
    print(f"mean underestimation                    : {uf.mean():.2f}")
    print(f"cells underestimating by > 2x           : {(uf > 2).mean():.3f}")
    print(f"worst cell                              : {uf.max():.1f}x")
    print(f"Pearson(ratio_in, ratio_out)            : {r:+.3f}")
    print(f"Spearman(ratio_in, ratio_out)           : {rho:+.3f}")
    print(f"mean coverage-relevant gap              : ratio_in {df.ratio_in.mean():.2f}"
          f" vs ratio_out {df.ratio_out.mean():.2f}")
    return df


if __name__ == "__main__":
    run()
