"""How many transfer standards does a new instrument actually need?

This is the practical metrological question a laboratory asks when buying a Raman
analyzer: "if I already have a validated calibration on instrument A, how many
reference measurements must I make on instrument B before B's results are
trustworthy?"

Protocol
--------
For each ordered pair (A -> B):
  * hold out 30% of B's physical samples, never used for anything;
  * draw n_std standardized samples from the paired pool (paired = same physical
    samples measured on both instruments, so their reference values already exist);
  * fit the operator + affine correction on those n_std samples only;
  * score on the held-out samples.

We report the error relative to a *full local calibration* on B (all its samples),
which is the cost the transfer is trying to avoid.  Repeat over many random
subsets and seeds to get a mean and spread.
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
from baselines import InInstrument


class PointTransfer:
    """TCCP without the conformal layer: operator + affine correction only.

    Used here because the question is about *point accuracy* versus the number of
    standardization samples; skipping cross-conformal calibration makes each fit an
    order of magnitude cheaper, which matters at this grid size.
    """

    def __init__(self, nc=8, ridge=1e-2):
        self.nc, self.ridge = nc, ridge

    def fit(self, Xs, Ys, Xt_std, Xs_std, Ys_std):
        from tccp import fit_operator, _affine_fit
        self.F = fit_operator(Xt_std, Xs_std, self.ridge)
        from sklearn.cross_decomposition import PLSRegression
        self.m = PLSRegression(n_components=self.nc, scale=False).fit(Xs, Ys)
        pf = self.m.predict(Xt_std @ self.F)
        self.aff = np.array([_affine_fit(pf[:, j], Ys_std[:, j])
                             for j in range(Ys_std.shape[1])])
        return self

    def predict(self, X, _=None):
        p = self.m.predict(X @ self.F)
        return np.stack([self.aff[j, 0] * p[:, j] + self.aff[j, 1]
                         for j in range(p.shape[1])], axis=1), None

OUT = os.path.join(PROJ, "data", "results")

STD_SIZES = [4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50]


def run(reps=8, seeds=(0, 1, 2)):
    data = td.load_all()
    names = list(data)
    rows = []
    for a in names:
        for b in names:
            if a == b:
                continue
            src, tgt = data[a], data[b]
            ia, ib = td.paired_unlabeled(src, tgt)
            for seed in seeds:
                fm, tm = pr.split_by_group(tgt, 0.3, seed)
                test_keys = {tuple(np.round(r, 5)) for r in tgt.y[tm]}
                ok = np.array([k not in test_keys for k in
                               [tuple(np.round(r, 5)) for r in tgt.y[ib]]])
                ia2, ib2 = ia[ok], ib[ok]
                if len(ia2) < max(STD_SIZES) // 2:
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

                # reference cost: full local calibration on B's fit half
                loc = InInstrument(nc=8).fit(tgt.X[fm], tgt.y[fm])
                loc_err = np.sqrt(np.mean((loc.predict(tgt.X[tm]) - tgt.y[tm]) ** 2, axis=0))

                for n in STD_SIZES:
                    if n > len(ia2):
                        continue
                    errs = []
                    for r in range(reps):
                        rng = np.random.default_rng(seed * 1000 + r)
                        sel = rng.choice(len(ia2), n, replace=False)
                        try:
                            m = PointTransfer(nc=8).fit(
                                Xs, Ys, tgt.X[ib2[sel]], src.X[ia2[sel]], src.y[ia2[sel]])
                            p, _ = m.predict(tgt.X[tm])
                        except Exception:  # noqa: BLE001
                            continue
                        errs.append(np.sqrt(np.mean((p - tgt.y[tm]) ** 2, axis=0)))
                    if not errs:
                        continue
                    e = np.mean(errs, axis=0)
                    for j, lab in enumerate(td.LABELS):
                        rows.append({
                            "source": a, "target": b, "seed": seed, "n_std": n,
                            "analyte": lab,
                            "rmse": float(e[j]), "local_rmse": float(loc_err[j]),
                            "ratio": float(e[j] / loc_err[j]) if loc_err[j] > 0 else np.nan,
                        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "std_size_curve.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)")

    # --- survivorship control (review #5) ------------------------------------
    # n_std=40/50 are only reachable by a subset of pairs (the ones with more
    # standardization samples), and that subset is systematically easier. Report
    # the headline curve capped at the size ALL pairs can reach, plus the subset
    # restricted to pairs that can supply 40, so the two populations are visible.
    n_by_size = df.groupby("n_std")["ratio"].agg(["mean", "median", "count"])
    print("\n=== full curve (cells per size in 'count'; note the shrinking population) ===")
    print(n_by_size.round(3).to_string())
    big = df[df.n_std.isin([4, 6, 8, 10, 12, 16, 20, 25, 30, 40])]
    pairs40 = big[big.n_std == 40][["source", "target"]].drop_duplicates()
    restricted = big.merge(pairs40, on=["source", "target"])
    print(f"\n=== same curve restricted to the {len(pairs40)} pairs that can supply n_std=40 ===")
    print(restricted.groupby("n_std")["ratio"].agg(["mean", "median", "count"]).round(3).to_string())

    # pair-level binomial CI for the headline "fraction within 1.5x" claim
    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return ((c - h) / d, (c + h) / d)

    for thr in (1.5, 1.25):
        print(f"\n=== fraction of PAIRS (not cells) reaching ratio <= {thr}, with Wilson CI ===")
        for n in [8, 12, 20, 30]:
            sub = df[(df.n_std == n) & (df.seed == 0)]
            pair_ok = sub.groupby(["source", "target"])["ratio"].apply(
                lambda s: (s <= thr).mean() >= 0.5)
            k, nn = int(pair_ok.sum()), len(pair_ok)
            lo, hi = wilson_ci(k, nn)
            print(f"  n_std={n:3}: {k}/{nn} pairs = {k/nn:.3f}  CI [{lo:.3f}, {hi:.3f}]")
    return df


if __name__ == "__main__":
    run()
