"""Real-fermentation data layer (validation, not calibration-transfer training).

These are genuine bioprocess runs, not synthetic mixtures.  They serve one purpose in
the study: test whether a calibration built on designed mixtures remains *valid*,
with stated uncertainty, when the sample matrix is cell-containing broth.

Sources (all CC-BY-4.0, downloaded to data/raw):

  yeast_bioreactor_fermentation  S. cerevisiae, 5 batches (B1-B4 + fed-batch FB),
                                 65 spectra, targets glucose [mM], ethanol [mM],
                                 biomass [g/L].  The only set with an explicit batch
                                 id, so it supports leave-one-batch-out validation.
  yeast_fermentation             online yeast fermentation, 58 spectra,
                                 glucose/fructose/glycerol/ethanol [mol/L].
  RamanSpectraEcoliFermentation  E. coli, 379 spectra, glucose/acetate [g/L],
                                 from Cedex BioHT reference analytics.
  RamanSpectraEthanolicYeastFermentations  34 spectra, 4 analytes [mol/L], RAW
                                 (negative intensities present, no baseline).

Important data facts carried over from verification:
  * S5 (Ralstonia) is excluded: its README states it mixes experimental and
    synthetic data, and the augmentation is jittered so synthetic rows cannot be
    filtered by deduplication.  It cannot carry a quantitative claim.
  * "Missing" labels are NaN, and 0 is a legitimate concentration. Always mask with
    notna(), never with != 0.
  * SubstrateMixRaman analyte units are NOT documented anywhere. Magnitudes are
    consistent with g/L but the paper is silent, so we never mix its labels with
    the broth sets in one regression.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# canonical grid used across the whole study (matches timdata.CANON_GRID)
GRID_LO, GRID_HI, GRID_STEP = 320.0, 1900.0, 4.0
CANON_GRID = np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP)


def _split_columns(df, label_cols):
    wl = [c for c in df.columns if c not in label_cols]
    numeric = []
    for c in wl:
        try:
            float(c)
            numeric.append(c)
        except (TypeError, ValueError):
            continue
    return numeric


def _to_grid(X, wl, grid):
    out = np.empty((X.shape[0], len(grid)))
    for i in range(X.shape[0]):
        out[i] = np.interp(grid, wl, X[i])
    return out


def _snv(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def load_yeast_bioreactor(preprocess=True):
    """5-batch S. cerevisiae set. Returns dict with X, Y, label names, batch ids."""
    p = os.path.join(RAW, "RamanBench", "yeast_bioreactor_fermentation",
                     "train-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    L = ["Glucose [mM]", "Ethanol [mM]", "Biomass [g/L]"]
    wl_cols = _split_columns(df, L + ["_group_id"])
    wl = np.asarray([float(c) for c in wl_cols])
    X = _to_grid(df[wl_cols].values.astype(float), wl, CANON_GRID)
    if preprocess:
        X = _snv(X)
    return {
        "X": X, "Y": df[L].values.astype(float), "labels": L,
        "batch": df["_group_id"].astype(str).values, "name": "yeast_bioreactor",
    }


def load_yeast_fermentation(preprocess=True):
    p = os.path.join(RAW, "RamanBench", "yeast_fermentation",
                     "train-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    L = [c for c in df.columns if "[" in c]
    wl_cols = _split_columns(df, L)
    wl = np.asarray([float(c) for c in wl_cols])
    X = _to_grid(df[wl_cols].values.astype(float), wl, CANON_GRID)
    if preprocess:
        X = _snv(X)
    return {"X": X, "Y": df[L].values.astype(float), "labels": L,
            "batch": np.zeros(len(df), dtype=int), "name": "yeast_fermentation"}


def load_ecoli(preprocess=True):
    """379 spectra across the released train/val/test CSVs."""
    frames = []
    for s in ("train", "val", "test"):
        p = os.path.join(RAW, "RamanSpectraEcoliFermentation", f"{s}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    L = ["Glucose", "Acetate"]
    wl_cols = _split_columns(df, L)
    wl = np.asarray([float(c) for c in wl_cols])
    X = _to_grid(df[wl_cols].values.astype(float), wl, CANON_GRID)
    if preprocess:
        X = _snv(X)
    return {"X": X, "Y": df[L].values.astype(float), "labels": L,
            "batch": np.zeros(len(df), dtype=int), "name": "ecoli"}


def load_ethanolic_yeast(preprocess=True):
    frames = []
    for s in ("train", "val", "test"):
        p = os.path.join(RAW, "RamanSpectraEthanolicYeastFermentations", f"{s}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    L = [c for c in df.columns if "[" in c]
    wl_cols = _split_columns(df, L)
    wl = np.asarray([float(c) for c in wl_cols])
    X = _to_grid(df[wl_cols].values.astype(float), wl, CANON_GRID)
    if preprocess:
        X = _snv(X)
    return {"X": X, "Y": df[L].values.astype(float), "labels": L,
            "batch": np.zeros(len(df), dtype=int), "name": "ethanolic_yeast"}


LOADERS = {
    "yeast_bioreactor": load_yeast_bioreactor,
    "yeast_fermentation": load_yeast_fermentation,
    "ecoli": load_ecoli,
    "ethanolic_yeast": load_ethanolic_yeast,
}


def load_all_broth():
    out = {}
    for k, fn in LOADERS.items():
        try:
            out[k] = fn()
        except FileNotFoundError:
            pass
    return out


if __name__ == "__main__":
    for k, d in load_all_broth().items():
        n_lab = np.sum(~np.isnan(d["Y"]).any(axis=1))
        print(f"{k:20} n={len(d['Y']):4} complete-labels={n_lab:4} "
              f"X={d['X'].shape} labels={d['labels']}")
        if len(np.unique(d["batch"])) > 1:
            print(f"{'':20} batches={dict(zip(*np.unique(d['batch'], return_counts=True)))}")
