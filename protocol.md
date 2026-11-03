# Experimental protocol (locked)

Everything below was fixed **before** the reported numbers were produced. Deviations
are recorded in `research-log.md` with the reason.

## Data

All datasets are public and CC-BY-4.0 (`data/raw/`).

**Transfer study — 8 Raman instruments, identical physical samples**

| instrument | spectra | wavenumber range | points | step |
|---|---|---|---|---|
| Anton532 | 270 | 300–3500 | 1601 | 2.00 |
| Anton785 | 270 | 300–2300 | 1001 | 2.00 |
| Kaiser | 134 | 300–1941 | 5472 | 0.30 |
| Metrohm | 399 | 301–3349 | 1875 | 1.58 |
| MettlerToledo | 275 | 300–3200 | 2901 | 1.00 |
| Tec5 | 395 | 300–3210 | 2911 | 1.00 |
| Timegate | 133 | 304–1998 | 486 | 3.47 |
| Tornado | 385 | 300–3300 | 3001 | 1.00 |

Analytes: glucose, acetate, magnesium sulfate (g/L), reference by Cedex BioHT.
Sample identity across instruments is recoverable from the exact concentration tuple;
45–130 paired samples per instrument pair.

**Broth validation** — `SubstrateMixRaman` (5614 complete-label mixture spectra),
`RamanBench/yeast_bioreactor_fermentation` (65, 5 batches), `RamanSpectraEcoliFermentation`
(379), `RamanBench/yeast_fermentation` (58).

## Preprocessing

1. Linear interpolation of every spectrum onto a **common grid 320–1900 cm⁻¹ at
   4 cm⁻¹ (396 points)** — the span all eight instruments cover.
2. Standard normal variate (SNV) per spectrum.

Rationale: the grid must be shared or a model fitted on A cannot be applied to B at
all; SNV removes the multiplicative scatter/antifoam effect that the dataset authors
deliberately varied, without removing the wavelength-dependent response that the
transfer method must handle.

## Splitting

**Physical-sample grouping.** Anton532/Anton785 hold 5 replicate spectra per
concentration; Kaiser holds one. Splits are made on the rounded concentration tuple,
so replicates of one sample never straddle train/test. 30% of the target's physical
samples are held out (3 seeds).

**Leakage control.** The source rows corresponding to a held-out target sample are
removed from source training. Standardization samples that fall in the held-out set
are excluded from the standardization set.

## The zero-target-label constraint

At fit time a method sees only:

- `Xs, Ys` — labelled source spectra, **with the standardization samples' source
  spectra removed** (row-level: exactly the rows whose target twins the operator and
  affine correction see are dropped from the PLS training set; other replicate spectra
  of the same physical samples remain — the shared design covers 235–270 of 270 rows,
  so sample-level removal would empty the training set);
- `Xt_std, Xs_std, Ys_std` — paired standardization samples, aligned. `Ys_std` are the
  concentrations **already measured in the source laboratory**, so obtaining them
  requires no reference analytics at the target instrument. This is the economic
  premise of the whole study.
- `Xt_test` — target spectra, labels hidden.

No method ever sees a target-instrument reference value for a predicted sample.

**Two deployment assumptions stated explicitly** (a referee will raise both):
(i) the paired samples must be *physically identical and chemically stable* over the
time between their measurement on A and on B — for broth standards this is a real
chemical assumption (stability, storage, drift) that must be verified per analyte;
(ii) pairing by exact concentration tuple is a property of this dataset, verified at
45–130 pairs per instrument pair, and must be re-verified on any new application.

## Metrics

| metric | definition |
|---|---|
| `ratio` | RMSE / RMSE of a full local calibration on the same split (lower better) |
| `coverage` | fraction of held-out samples inside the nominal 90% interval |
| `under-covering cell` | a (pair, analyte) with coverage < 0.85 |
| `width` | mean interval width, same units as the analyte |
| `rel_rmse` | RMSE / standard deviation of the reference values |

The local calibration is computed on the same 70% fit half, so `ratio = 1` means
"as good as buying a full local calibration".

## Reproducibility

- Python 3.13, numpy 2.5, scipy 1.18, pandas 3.1, scikit-learn 1.9, pyarrow 26.
- Every experiment seeds its RNG explicitly; conformal folds use `default_rng(seed)`.
- Scripts: `experiments/h_transfer/`, `experiments/h_uncertainty/`,
  `experiments/h_broth/`. Outputs: `data/results/*.csv`.
- Two numerical bugs found and fixed during development are documented in
  `research-log.md` (standardization direction; PDS window coverage), because both
  changed the conclusion and a reader should be able to reproduce the correction.

## Amendment log (post-review)

1. Row-level double-use exclusion added everywhere (was: standardization source
   spectra kept in the PLS training set — a leak the hostile review measured).
2. CV+ finite-sample correction implemented as the non-degenerate level inflation
   a_eff = a(n+1)/(n+1-k); the rank form saturates at the maximum residual and
   over-covers (measured 0.97 at nominal 0.90).
3. DS/PDS baselines receive the same hyper-parameter grids and label-free selection
   as the proposed method.
4. The transfer-standard budget is reported with pair-level Wilson CIs and the
   n>=40 rows restricted to the pairs that can supply them.
5. The self-assessment defect is measured by one named script
   (`run_selfassessment.py`); all earlier variants are retired.
6. Every CSV in data/results/ is regenerated from the current tree in one pass
   (`scripts/rerun_all.sh`), eliminating the provenance mismatches the review found.
