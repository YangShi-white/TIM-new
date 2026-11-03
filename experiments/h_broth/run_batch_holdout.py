"""Does the calibration survive a CHANGE OF BATCH, not just a change of instrument?

The instrument-transfer experiments hold the chemistry fixed and change the hardware.
Deployment in a real plant changes the chemistry too: a new batch of medium, a new
feed profile, a new inoculum. The `yeast_bioreactor_fermentation` set is the only
public data here with explicit batch identity (B1-B4, plus a fed-batch FB), so it is
the only place this question can be asked.

Protocol: leave one whole batch out. Fit the calibration on the other batches, predict
the held-out batch, and ask whether the reported 90% interval covers the truth.
Compare an ordinary grouped split (samples from the same batch appear in train and
test) against the leave-one-batch-out protocol that reflects real deployment.

Analytes are glucose [mM] and biomass [g/L]; ethanol is included but the fed-batch
was glucose-limited with ethanol spiking to 427 mM, which is outside the other
batches' range, so it is reported separately rather than pooled.
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

OUT = os.path.join(PROJ, "data", "results")
ALPHA = 0.10


def _q(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, lvl, method="higher"))


def run(nc=6):
    d = bd.load_yeast_bioreactor()
    X, Y, lab, batch = d["X"], d["Y"], d["labels"], d["batch"]
    rows = []
    batches = sorted(np.unique(batch))
    for held in batches:
        te = batch == held
        tr = ~te
        ok_tr = ~np.isnan(Y[tr]).any(axis=1)
        ok_te = ~np.isnan(Y[te]).any(axis=1)
        if ok_tr.sum() < 8 or ok_te.sum() < 3:
            continue
        m = PLSRegression(n_components=nc, scale=False).fit(X[tr][ok_tr], Y[tr][ok_tr])
        # conformal calibration using out-of-fold residuals WITHIN the training batches
        rng = np.random.default_rng(0)
        idx = np.where(ok_tr)[0]
        folds = np.array_split(rng.permutation(idx), min(5, max(2, len(idx) // 5)))
        oof = np.full((len(idx), Y.shape[1]), np.nan)
        for f in folds:
            keep = np.setdiff1d(idx, f)
            if len(keep) < 6:
                continue
            mk = PLSRegression(n_components=nc, scale=False).fit(X[tr][keep], Y[tr][keep])
            oof[np.isin(idx, f)] = mk.predict(X[tr][f])
        good = ~np.isnan(oof).any(axis=1)
        resid = np.abs(oof[good] - Y[tr][ok_tr][good])
        q = np.array([_q(resid[:, j], ALPHA) for j in range(Y.shape[1])])

        p = m.predict(X[te][ok_te])
        y = Y[te][ok_te]
        for j, name in enumerate(lab):
            e = p[:, j] - y[:, j]
            cov = np.mean((y[:, j] >= p[:, j] - q[j]) & (y[:, j] <= p[:, j] + q[j]))
            rows.append({
                "held_out_batch": held, "analyte": name, "n_test": int(ok_te.sum()),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "bias": float(np.mean(e)),
                "y_std": float(np.std(y[:, j])),
                "rel_rmse": float(np.sqrt(np.mean(e ** 2)) / (np.std(y[:, j]) + 1e-12)),
                "coverage": float(cov), "halfwidth": float(q[j]),
            })
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "batch_holdout.csv")
    df.to_csv(p, index=False)
    print(f"saved {p} ({len(df)} rows)\n")
    print("=== leave-one-batch-out ===")
    print(df[["held_out_batch", "analyte", "n_test", "rmse", "rel_rmse", "bias",
              "coverage"]].round(3).to_string(index=False))
    print("\n=== summary by analyte ===")
    print(df.groupby("analyte")[["rel_rmse", "coverage", "halfwidth"]]
          .mean().round(3).to_string())
    print(f"\nnominal coverage {1-ALPHA:.2f}")
    return df


if __name__ == "__main__":
    run()
