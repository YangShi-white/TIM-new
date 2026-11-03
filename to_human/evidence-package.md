# TIM paper — experimental evidence package

Status: experiments only (no figures, no manuscript) — per instruction.

## One-sentence contribution

A calibration transferred from one Raman analyzer to another without any reference
measurements at the new instrument can be given a **valid** measurement-uncertainty
statement, and the practice of issuing that statement from the samples that fitted
the transfer is wrong on half of instrument pairs.

## What is measured and where it lives

| claim | evidence | file |
|---|---|---|
| naive transfer collapses | R² down to −10³ | `loio_v1..v3.csv` |
| global operator > windowed PDS | 1.81 vs 10.38 ratio | `headtohead.csv` |
| distortion is low-rank, ~90% constant offset | k3 = 0.941, offset = 0.925 | `distortion_structure.csv` |
| self-assessment is biased | 2.06x underestimate, r = 0.008 | `conformal_alpha0.1.csv`, log F3 |
| cross-conformal restores coverage | 0.819 → 0.934 at nominal 0.90 | `headtohead.csv` |
| standard budget exists | n=20 → 1.69x, n=50 → 1.16x | `std_size_curve.csv` |
| label-free hyper-parameter selection works | Spearman +0.741, regret 0.082 | `selection_validity.csv` |
| point predictor is calibrated; defect is the protocol | split-half coverage 0.940 | log F10 |
| mixture → yeast broth survives for glucose | r = 0.985 | `broth_validation.csv` |
| fed-batch breaks the calibration | 13x error, all bias | `batch_holdout.csv` |
| each component contributes | see ablation table | `ablation.csv` |
| multi-source fusion without labels | criterion-best single 1.010 vs DS 1.81 | `multisource_fusion.csv` |
| intervals can be made locally adaptive | width tracks error (glucose rho 0.18–0.57) | `adaptive_intervals.csv` |

## The proposed method (SMART), precisely

Given labelled source spectra `(Xs, Ys)` and paired standardization samples
`(Xt_std, Xs_std, Ys_std)` whose reference values already exist from the source lab:

1. **Operator.** `F` solves the ridge problem `Xt_std F ≈ Xs_std`, computed via SVD in
   the sample space (numerically identical to the dense solve, ~30x faster).
2. **Latent calibration.** PLS on `(Xs, Ys)`; prediction is `f(Xt F)`.
3. **Affine correction.** Per analyte, fit `y = a p + b` on the standardization
   predictions — removes the residual gain/offset that the operator leaves.
4. **Selection without labels.** Sweep `(n_components, ridge)`. For each candidate,
   score by out-of-fold **pairwise difference** agreement:
   `RMSE((p_i - p_j) - (y_i - y_j)) / sd(y)`. This needs reference values only for the
   standardization samples, which are already known.
5. **Interval.** The same out-of-fold residuals give the conformal quantile at the
   requested coverage, so the interval is calibrated on predictions the operator did
   not fit.

Nothing in steps 1–5 touches a target reference value for a predicted sample.

## Honest limitations to state in the manuscript

1. The standardization samples must be *paired* — the same physical samples measured
   on both instruments. In practice this means an existing sample bank, not new
   synthesis.
2. A full local calibration remains the accuracy ceiling; transfer lands at 1.36x on
   average and rarely below 1.0x.
3. One instrument pair (Anton532) cannot be rescued at any budget.
4. Acetate does not survive the mixture→broth shift.
5. A fed-batch regime change breaks the calibration entirely, and the interval only
   partially detects it (ethanol coverage 0.714 on the fed-batch hold-out).
6. All data are glucose/acetate/MgSO4 in synthetic media plus a handful of real
   runs; no industrial broth, no in-line probe, no long-duration drift.

## What would make the paper stronger (not yet done)

- A dilution-of-standards experiment per pair, not just a count sweep.
- An in-line trace (only 34–58 points exist publicly; underpowered).
- Simulation of wavelength axis drift, which is the failure mode the correspondence
  approach is supposed to handle.

## Appendix A — robustness checks already run

**Excluding the weak source (Anton532):**

| variant | all pairs | excluding Anton532 |
|---|---|---|
| full | 1.326 | 1.232 |
| − cross-conformal (pooled) | 1.425 | 1.304 |
| − operator | 1.806 | 1.779 |
| windowed operator | 1.939 | 1.877 |

Coverage: full 0.922, pooled 0.820, windowed 0.817 with Anton532 removed. All
conclusions survive; the weak source inflates every mean but changes no ranking.

**Coverage is analyte-robust.** Pooled calibration under-covers on every analyte:
acetate 0.739, glucose 0.830, MgSO4 0.889. Cross-conformal recovers 0.907 / 0.943 /
0.951. So the defect is not an artefact of one analyte.

**Means are outlier-sensitive, medians are not.** Full-method ratio mean 1.326 vs
median 1.159; the maximum cell is 8.86 and comes from an Anton532 pair. Report both,
lead with the median, and state the outlier structure.
