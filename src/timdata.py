"""Canonical data layer for the cross-instrument Raman calibration-transfer study.

Design decisions that matter for validity:

* **Grouping.**  Anton532/Anton785 hold 5 replicate spectra per physical sample
  (165 spectra / 33 concentration tuples).  Any random split would leak the same
  physical sample into train and test.  We therefore group by the rounded
  concentration tuple and always split by group.

* **Canonical grid.**  Instruments disagree in range, step and count (486 to 5472
  points; 0.3 to 3.47 cm-1).  Everything is resampled onto one common grid over the
  intersection of all instrument ranges, so that a model fitted on A can be applied
  to B at all.  Resampling is linear interpolation, done on each spectrum
  independently.

* **Preprocessing.**  Standard normal variate (SNV) is applied per spectrum AFTER
  resampling.  It removes multiplicative scatter/antifluid turbidity effects, which
  the dataset authors explicitly varied, but it does NOT remove wavelength-dependent
  instrument response - that is the shift we want the method to handle.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELS = ["Glucose", "Acetate", "MagnesiumSulfate"]
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
INSTRUMENTS = [
    "Anton532", "Anton785", "Kaiser", "Metrohm",
    "MettlerToledo", "Tec5", "Timegate", "Tornado",
]
# 532 nm excitation vs 785 nm excitation; weak prior on which share physics
LASER = {
    "Anton532": 532, "Anton785": 785, "Kaiser": 785, "Metrohm": 785,
    "MettlerToledo": 785, "Tec5": 785, "Timegate": 532, "Tornado": 785,
}

# common grid: span covered by every instrument, at the coarsest reasonable step
GRID_LO, GRID_HI, GRID_STEP = 320.0, 1900.0, 4.0
CANON_GRID = np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP)


@dataclass
class InstData:
    name: str
    X: np.ndarray        # (n, n_grid) resampled + SNV
    y: np.ndarray        # (n, 3)
    groups: np.ndarray   # (n,) int group id per physical sample
    split: np.ndarray    # (n,) 'train'|'val'|'test'
    wl: np.ndarray       # native grid


def _raw_path(inst: str, split: str) -> str:
    return os.path.join(DATA_ROOT, f"RamanSpectraBioprocessAnalystes{inst}", f"{split}.parquet")


def load_raw(inst: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(_raw_path(inst, split))


def native_grid(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c not in LABELS]
    return np.array([float(c) for c in cols])


def spectrum_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c not in LABELS]
    return df[cols].values.astype(np.float64)


def resample(X: np.ndarray, wl_native: np.ndarray, grid: np.ndarray = CANON_GRID) -> np.ndarray:
    """Linear interpolation of each spectrum onto the canonical grid.

    Instruments that do not cover part of the grid get edge-replicated values;
    the shared span is chosen so this only affects a narrow margin.
    """
    out = np.empty((X.shape[0], len(grid)), dtype=np.float64)
    for i in range(X.shape[0]):
        out[i] = np.interp(grid, wl_native, X[i])
    return out


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-12)


def group_ids(y: np.ndarray, decimals: int = 5) -> np.ndarray:
    """Map each spectrum to a physical-sample id via its rounded label tuple."""
    keys = [tuple(np.round(r, decimals)) for r in y]
    uniq = {}
    for k in keys:
        if k not in uniq:
            uniq[k] = len(uniq)
    return np.array([uniq[k] for k in keys], dtype=int)


def load_instrument(inst: str, preprocess: bool = True) -> InstData:
    frames, splits = [], []
    for s in ("train", "val", "test"):
        p = _raw_path(inst, s)
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
            splits += [s] * len(frames[-1])
    if not frames:
        raise FileNotFoundError(inst)
    df = pd.concat(frames, ignore_index=True)
    wl = native_grid(df)
    X = spectrum_matrix(df)
    y = df[LABELS].values.astype(np.float64)
    Xr = resample(X, wl)
    if preprocess:
        Xr = snv(Xr)
    return InstData(inst, Xr, y, group_ids(y), np.array(splits), wl)


def load_all(preprocess: bool = True) -> dict[str, InstData]:
    out = {}
    for inst in INSTRUMENTS:
        try:
            out[inst] = load_instrument(inst, preprocess)
        except FileNotFoundError:
            pass
    return out


def paired_unlabeled(a: InstData, b: InstData) -> tuple[np.ndarray, np.ndarray]:
    """Indices of physical samples measured by BOTH instruments.

    Returns (idx_a, idx_b) aligned so row k of each refers to the same sample.
    Matching is exact on the concentration tuple, which the dataset provides.
    """
    ka = {tuple(np.round(r, 5)): i for i, r in enumerate(a.y)}
    kb = {tuple(np.round(r, 5)): i for i, r in enumerate(b.y)}
    shared = sorted(set(ka) & set(kb))
    ia = np.array([ka[k] for k in shared])
    ib = np.array([kb[k] for k in shared])
    return ia, ib


if __name__ == "__main__":
    data = load_all()
    print(f"{'instrument':16}{'n':>6}{'groups':>8}{'laser':>8}{'native':>9}")
    for k, d in data.items():
        print(f"{k:16}{len(d.y):6}{len(np.unique(d.groups)):8}{LASER[k]:8}{len(d.wl):9}")
    print(f"\ncanonical grid: {CANON_GRID[0]:.0f}-{CANON_GRID[-1]:.0f} @ {GRID_STEP} ({len(CANON_GRID)} pts)")
    names = list(data)
    print("\npaired unlabeled samples:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ia, ib = paired_unlabeled(data[names[i]], data[names[j]])
            if len(ia):
                print(f"  {names[i]:14} {names[j]:14} {len(ia):4}")
