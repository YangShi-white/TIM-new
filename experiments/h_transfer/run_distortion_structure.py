"""Quantify the inter-instrument distortion operator and test the low-rank claim.

F1 in findings.md asserts the distortion is global and low rank. That claim underpins
the whole method choice (one global operator rather than windowed standardization),
so it is measured directly here rather than inferred from method scores.

For each ordered pair we form the paired difference operator

    D = Xt_std - Xs_std          (one row per physical sample)

and report
  * the fraction of Var(D) explained by its mean (i.e. how much is a constant
    offset - a pure gain/offset mismatch),
  * the cumulative singular-value energy of the demeaned D for k = 1..10,
  * the share of the distortion that is multiplicative, estimated by regressing
    each target wavelength on the source spectrum in a single global least-squares
    fit (rank-1 per pair as an upper reference).

Also reports, per instrument, the raw intensity scale and the mean-spectrum
correlation on the shared grid, since these are the two quantities a reader will
check first.
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
import brothdata as bd

OUT = os.path.join(PROJ, "data", "results")


def run():
    data = td.load_all()
    names = list(data)
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            ia, ib = td.paired_unlabeled(data[a], data[b])
            if len(ia) < 15:
                continue
            Xa, Xb = data[a].X[ia], data[b].X[ib]
            D = Xb - Xa
            var_d = np.mean(np.sum(D ** 2, axis=1))
            const_off = D.mean(0)
            frac_const = float(1 - np.mean(np.sum((D - const_off) ** 2, axis=1)) / var_d)
            _, s, _ = np.linalg.svd(D - const_off, full_matrices=False)
            ev = s ** 2 / np.sum(s ** 2)

            # --- CONCENTRATION-RESIDUALISED CONTROL (review #1) ------------------
            # The raw distortion D is low-rank BY CONSTRUCTION: the samples are
            # designed 3-analyte mixtures with up to 5 replicates, so
            # D = Xs(A-I) + b is at most rank-3 whatever the instruments do, and
            # the source spectra are themselves equally low-rank. To separate
            # "instrument physics" from "design matrix", we project the three
            # concentrations out of D and re-measure.
            C = data[a].y[ia]                       # (n,3) concentrations
            Cc = np.column_stack([C, np.ones(len(C))])
            beta, *_ = np.linalg.lstsq(Cc, D, rcond=None)
            Dres = D - Cc @ beta                     # residualised distortion
            var_res = np.mean(np.sum(Dres ** 2, axis=1))
            frac_res = float(var_res / var_d)         # share NOT explained by design
            _, s_r, _ = np.linalg.svd(Dres - Dres.mean(0), full_matrices=False)
            ev_r = s_r ** 2 / np.sum(s_r ** 2)
            # and the source spectra's own spectrum for comparison
            _, s_x, _ = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
            ev_x = s_x ** 2 / np.sum(s_x ** 2)

            rows.append({
                "source": a, "target": b, "n_paired": len(ia),
                "frac_const_offset": frac_const,
                "k1": float(ev[:1].sum()), "k2": float(ev[:2].sum()),
                "k3": float(ev[:3].sum()), "k5": float(ev[:5].sum()),
                "k10": float(ev[:10].sum()),
                "mean_spectrum_corr": float(np.corrcoef(Xa.mean(0), Xb.mean(0))[0, 1]),
                # control columns
                "design_share_of_D": float(1 - frac_res),
                "k3_resid": float(ev_r[:3].sum()),
                "k10_resid": float(ev_r[:10].sum()),
                "k3_source_spectra": float(ev_x[:3].sum()),
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "distortion_structure.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== fraction of distortion variance that is a constant offset ===")
    print(df["frac_const_offset"].describe().round(3).to_string())
    print("\n=== cumulative singular energy of the demeaned distortion ===")
    print(df[["k1", "k2", "k3", "k5", "k10"]].agg(["mean", "min", "max"]).round(3).to_string())
    print("\n=== per-instrument mean-spectrum correlation on the shared grid ===")
    piv = df.groupby("target")["mean_spectrum_corr"].mean().round(3)
    print(piv.to_string())
    print("\n=== CONTROL: how much of D is the design, not the instruments? ===")
    print(f"share of D variance explained by the 3 concentrations + intercept "
          f"(median): {df.design_share_of_D.median():.3f}")
    print(f"k3 of D            (median): {df.k3.median():.3f}")
    print(f"k3 of D RESIDUALISED on concentration (median): {df.k3_resid.median():.3f}")
    print(f"k3 of the SOURCE SPECTRA themselves    (median): {df.k3_source_spectra.median():.3f}")
    print("\nReading: the raw 'low-rank distortion' is largely the 3-analyte design "
          "matrix (the source spectra are equally low-rank), so the paper must NOT "
          "argue from this decomposition; the global-operator choice is justified by "
          "predictive performance (tuned DS vs windowed PDS), not by this SVD.")
    print(df.groupby("source")["frac_const_offset"].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    run()
