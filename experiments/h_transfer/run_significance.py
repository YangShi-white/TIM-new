"""Paired significance tests: is the proposed method reliably better, or lucky?

All methods see identical cells, so the comparison is paired. For each analyte we
test the per-cell difference in error ratio (SMART - competitor) with a paired
Wilcoxon signed-rank test and report the bootstrap 95% CI of the mean difference.
Also tested: coverage difference vs pooled calibration.

Cells are not independent across seeds of the same pair, so a conservative block
bootstrap resamples PAIRS (not individual cells) and is reported alongside.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(_HERE))

OUT = os.path.join(PROJ, "data", "results")


def block_bootstrap_ci(df, col, block="pair", n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    blocks = df[block].unique()
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, len(blocks), replace=True)
        s = pd.concat([df[df[block] == b] for b in pick])
        vals.append(s[col].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    d = pd.read_csv(os.path.join(OUT, "headtohead.csv"))
    d["pair"] = d["source"] + "->" + d["target"]
    out_rows = []

    # Competitors are listed explicitly and MUST match the method labels actually
    # present in headtohead.csv. An earlier version hard-coded ["DS", "PDS", ...]
    # while the report labelled those rows "DS-tuned"/"PDS-tuned", which produced a
    # numbers-vs-labels mismatch (a data-consistency defect caught in review).
    competitors = ["direct", "PDS", "PDS-tuned", "DS", "DS-tuned", "TCCP-cv+"]
    print("=" * 78)
    print("PAIRED COMPARISON OF ERROR RATIO (SMART vs each competitor)")
    print("=" * 78)
    missing = [c for c in competitors if c not in set(d["method"])]
    if missing:
        print(f"WARNING: competitors absent from headtohead.csv: {missing}")
    for comp in competitors:
        if comp not in set(d["method"]):
            continue
        for lab in ["all"] + sorted(d["analyte"].unique()):
            a = d[(d["method"] == "SMART") & ((d["analyte"] == lab) | (lab == "all"))]
            b = d[(d["method"] == comp) & ((d["analyte"] == lab) | (lab == "all"))]
            m = a.merge(b, on=["pair", "seed", "analyte"], suffixes=("_s", "_c"))
            if len(m) < 8:
                continue
            diff = (m["ratio_s"] - m["ratio_c"]).values
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                p = np.nan
            lo, hi = block_bootstrap_ci(
                pd.DataFrame({"pair": m["pair"], "diff": diff}), "diff")
            out_rows.append({"competitor": comp, "analyte": lab, "n": len(diff),
                             "mean_diff": float(diff.mean()),
                             "median_diff": float(np.median(diff)),
                             "p_wilcoxon": float(p), "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(out_rows)
    print(res.round(4).to_string(index=False))

    print()
    print("=" * 78)
    print("PAIRED COMPARISON OF COVERAGE (SMART / cv+ vs pooled)")
    print("=" * 78)
    c = d.dropna(subset=["coverage"])
    cov_rows = []
    for meth in ["SMART", "TCCP-cv+"]:
        for lab in ["all"] + sorted(c["analyte"].unique()):
            a = c[(c["method"] == meth) & ((c["analyte"] == lab) | (lab == "all"))]
            b = c[(c["method"] == "TCCP-pooled") &
                  ((c["analyte"] == lab) | (lab == "all"))]
            m = a.merge(b, on=["pair", "seed", "analyte"], suffixes=("_m", "_p"))
            if len(m) < 8:
                continue
            diff = (m["coverage_m"] - m["coverage_p"]).values
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                p = np.nan
            lo, hi = block_bootstrap_ci(
                pd.DataFrame({"pair": m["pair"], "diff": diff}), "diff")
            cov_rows.append({"method": meth, "analyte": lab, "n": len(diff),
                             "mean_gain": float(diff.mean()),
                             "p_wilcoxon": float(p), "ci_lo": lo, "ci_hi": hi})
    res2 = pd.DataFrame(cov_rows)
    print(res2.round(4).to_string(index=False))

    res.to_csv(os.path.join(OUT, "significance_ratio.csv"), index=False)
    res2.to_csv(os.path.join(OUT, "significance_coverage.csv"), index=False)
    print(f"\nsaved significance_ratio.csv and significance_coverage.csv")


if __name__ == "__main__":
    main()
