"""E1 - The missing baselines, and the crossover point where transfer stops paying.

Why this exists
---------------
The paper claimed "transfer beats a full local calibration at a fraction of the
cost", but compared only against methods that DO need a source instrument. It never
compared against the strongest legitimate competitor:

    B1 Local-Std PLS
        the standardization samples exist on the TARGET instrument, and their
        reference values Ys_std are already known from the source laboratory.
        So one can simply train a PLS on those n target spectra and use it.
        This costs no new reference analytics either.

If B1 matches or beats the transfer at a given n, then transfer has no value at that
n, and the honest recommendation is "just calibrate locally on the standards".

Baselines implemented
---------------------
  B1  Local-Std PLS      PLS on (Xt_std, Ys_std) only
  B2  Augmented PLS      source training set + n target standards, sample weight w
  B3  SBC                direct transfer + slope/intercept on the standards
  SMART                  the proposed pipeline
  oracle-local           full local calibration on all available target data

Outputs
-------
  data/results/e1_baselines.csv        per (pair, seed, analyte, n_std, method)
  printed crossover analysis: for each pair/analyte, the smallest n at which
  SMART's error <= B1's error, and the distribution of that crossover.
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
from smart import SMART, fit_operator, _affine

OUT = os.path.join(PROJ, "data", "results")

# n_std values a laboratory could realistically afford. n=8 is the smallest at which
# a split interval is even definable (see A3: m<=8 gives an infinite interval), so
# the accuracy comparison starts at 8 and the interval comparison at 18.
STD_SIZES = [8, 12, 20, 30]
REPS = 6


def _local_std_pls(Xt_std, Ys_std):
    """B1: PLS trained only on the n standardization spectra of the target
    instrument, with their (already known) reference values."""
    n = len(Ys_std)
    nc = max(1, min(6, n // 4))
    return PLSRegression(n_components=nc, scale=False).fit(Xt_std, Ys_std)


def _augmented_pls(Xs, Ys, Xt_std, Xs_std, Ys_std, weight=3):
    """B2: refit the source model with the target standards appended.

    sklearn's PLSRegression has no sample_weight, so the target standards are
    up-weighted by integer row replication (the standard practical substitute).
    """
    X = np.vstack([Xs] + [Xt_std] * int(weight))
    Y = np.vstack([Ys] + [Ys_std] * int(weight))
    nc = 8
    m = PLSRegression(n_components=min(nc, len(Y) - 1), scale=False)
    m.fit(X, Y)
    return m


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
                # row-level double-use exclusion (same rule as every other script)
                rowmask = np.ones(len(src.y), dtype=bool)
                rowmask[ia2] = False
                keep = rowmask & np.array([tuple(np.round(r, 5)) not in tk
                                           for r in src.y])
                if keep.sum() < 20:
                    continue
                Xs, Ys = src.X[keep], src.y[keep]
                Xte, Yte = tgt.X[tm], tgt.y[tm]

                loc = bl.InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(Xte) - Yte) ** 2, axis=0))

                def record(method, p, n_std, iv=None):
                    for j, lab in enumerate(td.LABELS):
                        e = np.sqrt(np.mean((p[:, j] - Yte[:, j]) ** 2))
                        r = {"source": a, "target": b, "seed": seed, "method": method,
                             "analyte": lab, "n_std": n_std, "rmse": float(e),
                             "local_rmse": float(loc_err[j]),
                             "ratio": float(e / loc_err[j]) if loc_err[j] > 0 else np.nan,
                             "n_test": int(tm.sum())}
                        if iv is not None:
                            r["coverage"] = float(np.mean(
                                (Yte[:, j] >= iv[:, j, 0]) & (Yte[:, j] <= iv[:, j, 1])))
                            r["width"] = float(np.mean(iv[:, j, 1] - iv[:, j, 0]))
                        rows.append(r)

                # the source PLS does not depend on the standardization draw, so fit
                # it once per (pair, seed) instead of once per draw
                mdl_src = PLSRegression(n_components=8, scale=False).fit(Xs, Ys)

                for n in STD_SIZES:
                    if n > len(ia2):
                        continue
                    for rep in range(REPS):
                        rng = np.random.default_rng(seed * 1000 + rep)
                        sel = rng.choice(len(ia2), n, replace=False)
                        Xt_s, Xs_s, Ys_s = (tgt.X[ib2[sel]], src.X[ia2[sel]],
                                            src.y[ia2[sel]])

                        # --- B1: local PLS on the standards only ------------------
                        try:
                            m1 = _local_std_pls(Xt_s, Ys_s)
                            record("B1 local-std PLS", m1.predict(Xte), n)
                        except Exception:  # noqa: BLE001
                            pass

                        # --- B2: augmented / updated PLS --------------------------
                        try:
                            m2 = _augmented_pls(Xs, Ys, Xt_s, Xs_s, Ys_s, weight=3.0)
                            record("B2 augmented PLS", m2.predict(Xte), n)
                        except Exception:  # noqa: BLE001
                            pass

                        # --- B3: SBC (direct + slope/intercept) -------------------
                        # Only the OPERATOR depends on the standards; the source PLS
                        # does not. Fitting it via bl.DS refits the whole PLS on Xs
                        # every draw (~1.5 s), which dominated runtime. Compute the
                        # operator directly and reuse the outer PLS fit.
                        try:
                            F3 = fit_operator(Xt_s, Xs_s, ridge=1e-2)
                            p_std = mdl_src.predict(Xt_s @ F3)
                            p_te = mdl_src.predict(Xte @ F3)
                            for j in range(Ys_s.shape[1]):
                                c = _affine(p_std[:, j], Ys_s[:, j])
                                p_te[:, j] = c[0] * p_te[:, j] + c[1]
                            record("B3 SBC", p_te, n)
                        except Exception:  # noqa: BLE001
                            pass

                        # --- SMART (proposed) -------------------------------------
                        try:
                            m4 = SMART(alpha=0.10, seed=rep, interval_mode="split").fit(
                                Xs, Ys, Xt_s, Xs_s, Ys_s)
                            p, iv = m4.predict(Xte)
                            record("SMART", p, n, iv)
                        except Exception:  # noqa: BLE001
                            pass

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "e1_baselines.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")

    print("=== accuracy: ratio to full local calibration (lower is better) ===")
    print(df.pivot_table(index="method", columns="n_std", values="ratio",
                         aggfunc="mean").round(3).to_string())
    print("\nmedian:")
    print(df.pivot_table(index="method", columns="n_std", values="ratio",
                         aggfunc="median").round(3).to_string())

    # ---- crossover analysis -------------------------------------------------
    print("\n=== crossover: smallest n_std at which SMART is not worse than B1 ===")
    piv = df.groupby(["source", "target", "analyte", "n_std", "method"])["rmse"].mean()
    cross = []
    for key, g in piv.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2])
        try:
            s = g.xs("SMART", level="method")
            b1 = g.xs("B1 local-std PLS", level="method")
        except KeyError:
            continue
        both = pd.DataFrame({"smart": s, "b1": b1}).dropna()
        if both.empty:
            continue
        win = both.index[both.smart <= both.b1]
        cross.append({"pair_analyte": key,
                      "n_crossover": int(win.min()) if len(win) else np.inf,
                      "smart_wins_at_all_n": bool(len(win) == len(both))})
    cr = pd.DataFrame(cross)
    if not cr.empty:
        vc = cr["n_crossover"].value_counts().sort_index()
        print(vc.to_string())
        print(f"\npair-analytes where SMART never beats B1: "
              f"{(cr.n_crossover == np.inf).sum()}/{len(cr)}")
        print(f"SMART beats B1 at every tested n: {cr.smart_wins_at_all_n.mean():.3f}")
        cr.to_csv(os.path.join(OUT, "e1_crossover.csv"), index=False)
    return df


if __name__ == "__main__":
    run()
