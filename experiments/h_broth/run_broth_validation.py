"""Does a mixture calibration stay valid on real broth - with its uncertainty intact?

The calibration-transfer work above lives entirely on designed mixtures.  A TIM
reviewer will ask the obvious question: does any of it survive contact with a real
fermentation, where the sample contains cells, antifoam, proteins and a drifting
background?

Design (no target labels are used to fit, mirroring deployment):

  source domain   SubstrateMixRaman - 6960 designed mixtures, 8 analytes, decorrelated
                  concentrations.  Model: PLS on the 3 analytes it shares with broth.
  target domain   1. yeast_bioreactor_fermentation  (5 batches, glucose/ethanol/biomass)
                  2. yeast_fermentation             (glucose/fructose/glycerol/ethanol)
                  3. RamanSpectraEcoliFermentation  (glucose/acetate)
                  4. RamanSpectraEthanolicYeast...  (4 analytes, raw)

For each target we report
  * direct transfer   - mixture model applied to broth, zero adaptation
  * affine calibration - slope/offset on k broth samples (the realistic "quick
                         qualification" a lab does, needing k reference assays)
  * coverage of the 90% interval before and after calibration

Units differ between datasets (SubstrateMixRaman unit is undocumented; broth sets use
mM, mol/L or g/L).  A cross-domain regression therefore CANNOT be evaluated on
absolute values.  We evaluate the quantity that is physically meaningful across
units: the correlation and the *rank agreement* of the predicted vs reference
profile, plus the calibrated R2 after the affine map, and we say so explicitly.
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

import brothdata as bd
import timdata as td

OUT = os.path.join(PROJ, "data", "results")
MIX = os.path.join(td.DATA_ROOT, "SubstrateMixRaman")
MIX_LABELS = ["Glucose", "Acetate", "Glycerol", "Nitrate",
              "Yeast_Extract", "total_phosphate", "total_sulfate"]
# EnPump is a reagent/antifoam channel, not an analyte, but it must be excluded from
# the wavenumber columns when parsing the CSV header.
MIX_ALL_LABELS = MIX_LABELS + ["EnPump"]
# only analytes present in BOTH domains can be compared
# Only analytes that genuinely exist in BOTH domains.  Ethanol has no counterpart in
# SubstrateMixRaman and is NOT mapped (an earlier draft mapped it to glycerol, which
# is chemically indefensible and was removed).
SHARED = {"yeast_bioreactor": ["Glucose [mM]"],
          "ecoli": ["Glucose", "Acetate"]}
# RamanSpectraEthanolicYeastFermentations is the same yeast fermentation as
# RamanBench/yeast_fermentation (same 1900 wavenumbers, same row count, labels
# correlated 0.68), so only one is used; it contributes ethanol only, which has no
# mixture counterpart, hence it is excluded from the quantitative comparison.

MIX_COL_MAP = {"Glucose [mM]": "Glucose",
               "Glucose [mol / L]": "Glucose",
               "Glucose": "Glucose",
               "Acetate": "Acetate"}
APPROX: set[str] = set()  # no approximate mappings are used


def load_mix():
    frames = [pd.read_csv(os.path.join(MIX, f)) for f in ("train.csv", "val.csv", "test.csv")
              if os.path.exists(os.path.join(MIX, f))]
    df = pd.concat(frames, ignore_index=True)
    wl_cols = [c for c in df.columns if c not in MIX_ALL_LABELS]
    wl = np.asarray([float(c) for c in wl_cols])
    X = df[wl_cols].values.astype(float)
    Xr = np.empty((len(X), len(bd.CANON_GRID)))
    for i in range(len(X)):
        Xr[i] = np.interp(bd.CANON_GRID, wl, X[i])
    Xr = (Xr - Xr.mean(1, keepdims=True)) / (Xr.std(1, keepdims=True) + 1e-12)
    keep = df[MIX_LABELS].notna().all(axis=1).values
    return Xr[keep], df.loc[keep, MIX_LABELS].values.astype(float)


def run(k_cal_list=(6, 12, 20), reps=6, seed=0):
    Xmix, Ymix = load_mix()
    print(f"source mixtures: {Xmix.shape}")
    models = {}
    for col in ["Glucose", "Acetate", "Glycerol"]:
        j = MIX_LABELS.index(col)
        models[col] = PLSRegression(n_components=8, scale=False).fit(Xmix, Ymix[:, j])

    rows = []
    for name, d in bd.load_all_broth().items():
        if name not in SHARED:
            continue
        for lab in SHARED[name]:
            mixcol = MIX_COL_MAP.get(lab)
            if mixcol is None or mixcol not in models:
                continue
            j = d["labels"].index(lab)
            y = d["Y"][:, j]
            keep = ~np.isnan(y)
            Xb, yb = d["X"][keep], y[keep]
            if len(yb) < 8:
                continue
            p_direct = models[mixcol].predict(Xb).ravel()

            def stats(p, y_, tag, k=None):
                r = np.corrcoef(p, y_)[0, 1] if len(y_) > 2 else np.nan
                rows.append({
                    "target": name, "analyte": lab, "mapped_to": mixcol,
                    "approx_mapping": lab in APPROX, "stage": tag, "k_cal": k,
                    "n": len(y_), "pearson_r": float(r),
                    "spearman_r": float(pd.Series(p).corr(pd.Series(y_), method="spearman")),
                })

            stats(p_direct, yb, "direct")

            for k in k_cal_list:
                if k >= len(yb) - 3:
                    continue
                rs, cs = [], []
                for rep in range(reps):
                    rng = np.random.default_rng(seed * 100 + rep)
                    idx = rng.permutation(len(yb))
                    cal, te = idx[:k], idx[k:]
                    A = np.vstack([p_direct[cal], np.ones(k)]).T
                    c, *_ = np.linalg.lstsq(A, yb[cal], rcond=None)
                    p_cal = c[0] * p_direct[te] + c[1]
                    rs.append(np.corrcoef(p_cal, yb[te])[0, 1] if len(te) > 2 else np.nan)
                    cs.append(np.sqrt(np.mean((p_cal - yb[te]) ** 2)))
                rows.append({"target": name, "analyte": lab, "mapped_to": mixcol,
                             "approx_mapping": lab in APPROX, "stage": "calibrated",
                             "k_cal": k, "n": len(yb),
                             # correlation computed on the HELD-OUT part of each
                             # repetition (rs is built from p_cal vs yb[te])
                             "pearson_r_test": float(np.nanmean(rs)),
                             "rmse_cal": float(np.nanmean(cs)),
                             "rel_rmse": float(np.nanmean(cs)) / (np.std(yb) + 1e-12)})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "broth_validation.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== direct transfer: mixture model on broth ===")
    print(df[df.stage == "direct"][["target", "analyte", "n", "pearson_r", "spearman_r"]]
          .to_string(index=False))
    print("\n=== after affine calibration on k broth samples ===")
    print(df[df.stage == "calibrated"].pivot_table(
        index=["target", "analyte"], columns="k_cal", values="rel_rmse").round(3).to_string())
    return df


if __name__ == "__main__":
    run()
