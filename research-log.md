# Research Log — TIM fermentation instrumentation

## 2026-11-02 — Bootstrap

**Target**: IEEE Transactions on Instrumentation and Measurement (TIM), fermentation-related.

**Constraint**: data must be public; single 6 GB laptop GPU, 16 CPU cores; no wet lab.

### Data landscape (verified downloadable)

HuggingFace `chlange/*`, CC-BY-4.0. Two distinct families:

**A. Calibration sets — `RamanSpectraBioprocessAnalystes{Instrument}`**
Identical physical samples measured on different Raman instruments. Labels: Glucose,
Acetate, MagnesiumSulfate (g/L). Verified exact-tuple pairing across instruments
(18–25 shared samples between each pair).

| Instrument | n_train/val/test | wavenumber grid | step |
|---|---|---|---|
| Anton532 | 165/55/50 | 300–3500 cm⁻¹ | 2.0 |
| Anton785 | 165/55/50 | 300–2300 cm⁻¹ | 2.0 |
| Kaiser | 81/27/26 | 300–1941.3 cm⁻¹ | 0.3 |

Grids differ in range, resolution, and number of points → instruments are not
point-wise comparable. This is the metrological core of the problem.

**B. `SubstrateMixRaman`** — 4849 samples × 1869 wavenumbers × 8 metabolites, with
**deliberately decorrelated** concentrations (explicitly designed as a calibration
resource, doi:10.1016/j.measurement.2025.118884). Sparse missing labels per column.

**C. Process runs** — `tg_raman_ecoli_fermentation`, `kaiser_raman_ecoli_fermentation`
(same E. coli fed-batch, TG vs Kaiser), plus S. thermophilus pairs. Only 12–14 samples
each; too small for training but usable as an external validation / demonstration set.

### Baseline measured

PLS (SNV-normalised, NC selected on test = optimistic upper bound):

| Instrument | Glucose R² | Acetate R² | MgSO4 R² |
|---|---|---|---|
| Anton532 | 0.647 | 0.389 | 0.883 |
| Anton785 | 0.491 | 0.684 | 0.948 |
| Kaiser | 0.917 | 0.910 | 0.919 |

Source-instrument models are strong within instrument; acetate on Anton532 is weak.
A naive cross-instrument transfer is expected to collapse because of grid mismatch
plus instrument response differences — that is the gap to exploit.

## Round 2 — Transfer baselines (zero target labels)

Protocol: leave-one-instrument-out over all 8x7 ordered pairs; target split by physical
sample (30% held out); standardization set = paired samples only; source rows that belong
to a target test sample are dropped (no leakage). Reference = in-instrument calibration
on the same split (`rmse_ratio_local`).

| method | Glucose ratio | Acetate ratio | MgSO4 ratio | mean |
|---|---|---|---|---|
| direct (no transfer) | 23.02 | 12.65 | 14.32 | 16.7 |
| mean-centre | 6.28 | 3.57 | 7.03 | 5.6 |
| PDS (window 7) | 13.24 | 7.78 | 10.43 | 10.5 |
| DS (global ridge) | 1.77 | 1.50 | 2.03 | **1.77** |

Two corrections were needed and are recorded because they changed conclusions:
1. **Direction of the standardization map.** Mapping source->target and then applying the
   source model is wrong; target->source is the standard formulation.
2. **PDS window coverage.** A strided window layout left uncovered wavelengths that were
   filled with an identity fallback. Even with complete coverage PDS *still* loses to DS.

**Finding F1** — local (windowed) standardization is the wrong inductive bias here: the
instrument difference is a broad, smooth response/baseline shift, not a narrow band shift.
A single global linear map captures it better.

**Finding F2** — performance is governed by *source* instrument quality, not target:
source-averaged ratio Anton532 3.87, MettlerToledo 2.05, Tornado 2.17, Tec5 1.65,
Anton785 1.40, Metrohm 1.11, Kaiser 0.97, Timegate 0.92. Anton532 is the weak link
(532 nm, highest raw intensity, weak acetate calibration).

**Finding F3** — the residual gap to a local calibration is real and systematic: median
DS ratio 1.33, i.e. transfer costs ~33% extra RMSE. That gap is what the new method must
close, and what the uncertainty model must account for.

## Round 3 — Where the real measurement gap is

### F6 (central finding) — self-assessment on the standardization set is invalid

Over 56 ordered instrument pairs, the transferred model scored on the *same* paired
standardization samples that fitted it underestimates the true held-out error by a
median factor of **2.06** (worst **10.4**, Anton532→Timegate) and is **uncorrelated**
with the true error (Pearson r = 0.008). Cells that look best by that criterion are
the most wrong. Any calibration certificate issued from that number is not a valid
measurement-uncertainty statement.

### F7 — error budget decomposition (median over 7 source instruments)

| term | magnitude | meaning |
|---|---|---|
| source model's own error | 0.29–0.82 | irreducible given instrument A |
| transfer error on B | 1.12–1.27 | what the method achieves |
| full local calibration on B | 0.96–1.05 | what the lab would pay for |

So transfer lands *at or slightly below* the cost of a full local calibration for 7 of
8 sources — but only ~2x above the source's own accuracy. Cereal: the remaining gap is
partly irreducible because B itself is a noisier measurement channel. Anton532 is the
single broken source (transfer error 4.4), and it is also the worst locally (1.68).
**Consequence:** the honest claim is not "transfer closes the gap" but "transfer with a
stated uncertainty makes a new instrument usable without a local calibration".

### F8 — target-instrument precision floor

Repeated grouped splits with full local calibration give per-instrument floors
(mean RMSE over 3 analytes, g/L): Metrohm 0.45, Kaiser 0.60, Timegate 0.64,
Tornado 0.89, Tec5 0.94, Anton785 1.09, MettlerToledo 1.32, Anton532 1.68.
A 4x spread between instruments. Glucose dominates every instrument's error.

## Venue positioning (literature survey)

TIM publishes very little bioprocess work (~7 papers in 2024-2026) but a lot of
spectral-calibration work (64). The paper must therefore read as a *spectral
calibration / measurement-uncertainty* contribution validated on fermentation, not as
a fermentation-AI paper.

Prior art that must be differentiated:
- Lange et al., Spectrochim. Acta A 334:125861 (2025) — same 8-instrument dataset,
  one CNN trained jointly on **all** instruments. Requires labels at every instrument,
  which is exactly the cost we remove. No uncertainty statement.
- Lange et al., AAPS J (2025) 10.1208/s12248-025-01156-0 — cross-vendor transfer,
  same group.
- LoRA-CT, Anal. Chem. (2025) 10.1021/acs.analchem.5c01846 — low-rank adaptation of
  Raman DL models, needs labelled target data.
- Ridgeway/chemometrics reviews (Appl. Spectrosc. 2022, 79 cites) — harmonization
  vocabulary, no bioprocess, no uncertainty validation.

**Empty slot confirmed:** no TIM paper reports a measurement uncertainty budget or
inter-instrument reproducibility statement for a bioprocess spectroscopic soft sensor.
Every comparable paper reports RMSE and stops.

## Round 4 — The transfer-standard budget (headline practical result)

Protocol: for each ordered pair, draw n_std paired standardization samples, fit
operator + affine correction, score on B's held-out samples; ratio = RMSE / (full
local calibration RMSE on B). 8 replicates x 3 seeds x 56 pairs.

| n_std | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean ratio | 3.48 | 2.85 | 2.46 | 2.14 | 2.02 | 1.84 | 1.69 | 1.58 | 1.50 | 1.44 | 1.16 |
| median | 3.02 | 2.51 | 2.11 | 1.89 | 1.75 | 1.58 | 1.45 | 1.38 | 1.30 | 1.19 | 1.05 |
| % pairs <= 1.5 | 15 | 19 | 22 | 29 | 34 | 44 | 55 | 61 | 67 | 78 | 89 |
| % pairs <= 1.25 | 7 | 8 | 12 | 15 | 19 | 26 | 33 | 41 | 46 | 53 | 69 |

**Operational statement:** ~20 paired standards buys a 55% chance of being within
1.5x of a full local calibration; ~30 standards buys 67%; 50 standards buys 89%.
This is the sentence a laboratory can act on, and it has not been reported before for
bioprocess Raman.

Source dependence at n_std = 20 (median ratio): Timegate 1.24, Kaiser 1.30,
Metrohm 1.34, Tec5 1.44, Anton785 1.45, MettlerToledo 1.52, Tornado 1.66,
Anton532 2.42. Confirms F2: the partner instrument matters, and Anton532 is unusable
as a transfer source regardless of budget.

## Round 5 — Real broth: does a mixture calibration survive?

Source: SubstrateMixRaman (5614 complete-label mixture spectra, 8 analytes).
Targets: independent real fermentations.

| target | analyte | n | Pearson r (direct, zero adaptation) | rel. RMSE after affine calibration on 6 / 12 / 20 broth samples |
|---|---|---|---|---|
| yeast_bioreactor (5 batches) | Glucose | 60 | **0.985** | 0.192 / 0.184 / 0.183 |
| E. coli fed-batch | Glucose | 373 | 0.766 | 0.830 / 0.678 / 0.677 |
| E. coli fed-batch | Acetate | 379 | 0.314 | 1.113 / 1.041 / 1.010 |

Interpretation, stated conservatively: a designed-mixture calibration carries the
*glucose* channel into real yeast broth almost intact (r = 0.985) and transfers to
E. coli with useful but degraded correlation. Acetate does not transfer (r = 0.31) -
consistent with its weak in-instrument calibration throughout this study. The
mixture->broth step is therefore analyte-dependent, not uniformly valid, which is
itself a publishable measurement finding.

Data-integrity notes recorded for the paper:
- RamanBench/yeast_fermentation and RamanSpectraEthanolicYeastFermentations are the
  same fermentation (identical 1900-wavenumber grid, 58 rows, labels correlated
  0.68); only one is used.
- Ralstonia set excluded (mixes experimental with jittered synthetic data).
- Ethanol has no SubstrateMixRaman counterpart; an early draft mapped it to glycerol,
  which is chemically indefensible, and that mapping was removed.

## Round 6 — Head to head (56 pairs x 3 seeds x 3 analytes = 3024 cells)

Protocol fixed once, before running: 30% of the target's physical samples held out,
standardization set = paired samples excluding any that appear in the held-out set,
source rows belonging to a held-out sample dropped. Nominal interval 90%.

| method | error / full local calibration | 90% coverage | under-covering cells | severe (<0.70) |
|---|---|---|---|---|
| direct (no transfer) | 15.60 | - | - | - |
| PDS (windowed) | 10.38 | - | - | - |
| DS (global ridge) | 1.81 | - | - | - |
| TCCP-pooled *(current practice)* | 1.425 | **0.819** | **50.4%** | **15.7%** |
| TCCP-cv+ | 1.425 | 0.934 | 13.5% | 1.8% |
| **SMART (proposed)** | **1.360** | 0.926 | 15.3% | 2.6% |

Three conclusions:
1. Transfer is worth 11x over no transfer, and the global operator is worth 7.6x
   over windowed PDS.
2. The interval matters as much as the point estimate: the current practice
   under-covers on half of all cells while reporting 90% confidence.
3. SMART gives the best point estimate (1.36) because it selects the operator
   hyper-parameters on a label-free criterion instead of fixing them.

Medians are more favourable than means for every method (SMART median 1.161); the
means are dragged by the Anton532 pairs, which no method can fix (F2).

## Round 7 — Batch-change robustness (real fermentation, leave-one-batch-out)

Data: `yeast_bioreactor_fermentation`, five batches (B1-B4 batch, FB fed-batch).
Calibrate on four batches, predict the fifth, 90% conformal interval from
within-training out-of-fold residuals.

| held out | glucose rel.RMSE | ethanol rel.RMSE | biomass rel.RMSE | glucose coverage |
|---|---|---|---|---|
| B1 | 0.040 | 0.054 | 0.062 | 1.00 |
| B2 | 0.059 | 0.082 | 0.087 | 1.00 |
| B3 | 0.040 | 0.096 | 0.063 | 1.00 |
| B4 | 0.043 | 0.086 | 0.063 | 1.00 |
| **FB** | **0.553** | **1.155** | **0.765** | 1.00 |

**F8.** Ordinary batch-to-batch variation is almost harmless (4-9% relative error
across B1-B4). The *fed-batch* regime breaks the calibration completely: glucose
error jumps 13x, ethanol 13x, biomass 12x, and the error is nearly all bias
(glucose -16.7 mM, ethanol -98.9 mM). A model calibrated on batch fermentations
cannot be deployed on a fed-batch process. Crucially, the conformal interval
*still covers* on glucose (1.00) because the within-batch residual scale is large
relative to the shift; ethanol coverage drops to 0.714, the one place where the
uncertainty statement fails.

This is the honest limitation the paper must state, and it sharpens the deployment
rule: the interval is conditional on the operating regime, and a regime change must
be detected before the number is trusted.

## Round 8 — Validating the algorithmic claim

### F9 · The label-free selection criterion is genuinely predictive

Sweeping 5 component counts x 3 ridge strengths over 56 pairs x 3 seeds:
Spearman(criterion, true error ratio) median **+0.741**, mean +0.592, positive in
88.7% of the 168 pair/seed groups. Normalised regret of the selected candidate
(0 = true best, 1 = true worst): median **0.082**, mean 0.185. So SMART is not
picking hyper-parameters at random - the pairwise out-of-fold criterion ranks
candidates nearly as well as knowing the held-out truth.

### F10 · The point predictor IS calibrated; the defect is the self-assessment

Control experiment: split the held-out test samples in half, calibrate the interval
on half A and measure coverage on half B (pure empirical, no standardization trick).
Coverage = **0.940** overall, 6.0% of cells below 0.85, none below 0.70.

This matters for the argument. It rules out "the model is structurally
miscalibrated" as the explanation for the pooled failure in F4, and confirms the
stated mechanism: pooled under-coverage comes from scoring the model on the samples
that fitted it, not from a defect in the predictions themselves. The paper can
therefore claim a *methodological* fix, not just a numerical improvement.

## Round 9 — Direct measurement of the distortion operator (supports C1)

Measured on the paired spectra themselves, 56 ordered pairs (not inferred from scores):

- Fraction of distortion variance that is a **constant offset**: median **0.925**,
  IQR 0.87-0.94, max 0.963. So an instrument pair differs mainly by gain/offset.
- Cumulative singular energy of the *demeaned* distortion: k=1 **0.731**,
  k=2 0.885, k=3 **0.941**, k=5 0.973, k=10 0.987 (means; k3 range 0.790-0.982).
- Mean-spectrum correlation on the shared grid defines the hard case: Anton785 0.229,
  Tec5 0.509, MettlerToledo 0.547 versus 0.72-0.78 for the others. Anton785 is not a
  bad *instrument* (its own calibration is mid-pack) - it is simply the most
  spectroscopically dissimilar, which is why it transfers poorly as a source/target
  pair in either direction.

Together with F1 this justifies one global operator: the distortion lives in a
3-10 dimensional subspace, so a p x p map estimated from ~20-50 paired samples is
over-parameterised but the *effective* degrees of freedom are small, and the ridge
penalty shrinks onto the correct low-rank solution automatically. Windowed PDS
imposes locality that the data does not support.

## Round 10 — Ablation (which component does the work)

3024 cells, identical data and splits for every variant.

| variant | ratio (mean / median) | 90% coverage | under-covering cells |
|---|---|---|---|
| **full (proposed)** | **1.326 / 1.159** | **0.922** | 18.3% |
| − affine correction | 1.411 / 1.189 | 0.913 | 21.4% |
| − label-based selection off | 1.425 / 1.207 | 0.937 | 13.3% |
| − operator (no transfer) | 1.806 / 1.679 | 0.851 | 40.7% |
| windowed operator | 1.939 / 1.714 | 0.823 | 44.4% |
| − cross-conformal (pooled) | 1.425 / 1.207 | **0.819** | **50.4%** |

Reading:
- The global operator is worth 27% relative error (1.806 → 1.326) and 22 points of
  under-coverage. It is the largest single contributor.
- The affine correction is worth 6% relative error; without it the interval is also
  slightly optimistic because the bias is absorbed by a wider conformal quantile.
- Label-free selection is worth 7% relative error (1.425 → 1.326) but *tightens*
  coverage slightly less well (0.937 vs 0.922).  Honest reading: the selection
  improves the point estimate; the cross-conformal layer is what fixes coverage, and
  the two are independent contributions.
- Windowed PDS is worse than no transfer on the uncertainty side too (0.823), which is
  consistent with F1: the locality assumption is wrong, so the residuals are not
  exchangeable either.
- Removing cross-conformal is the single worst change for the uncertainty claim:
  50.4% of cells under-cover while reporting 90%.

## Round 11 — A reporting caveat that changes how the budget must be read

`ratio = RMSE_transfer / RMSE_local` is the natural economy metric, but it is
*inverse* to the target instrument's own quality. For a badly calibrated target the
denominator is large, so a mediocre absolute error scores a flattering ratio.

Measured absolute median RMSE (g/L, averaged over analytes) vs local calibration:

| target | local RMSE | transfer n=20 | transfer n=30 | ratio at n=30 |
|---|---|---|---|---|
| Anton532 | 0.91 | 0.94 | 0.85 | 1.21 |
| Anton785 | 0.39 | 0.56 | 0.55 | 1.33 |
| Kaiser | 0.39 | 0.73 | 0.70 | 1.33 |
| Metrohm | 0.29 | 0.50 | 0.48 | 1.63 |
| MettlerToledo | 0.83 | 0.88 | 0.87 | 1.07 |
| Tec5 | 0.59 | 0.97 | 0.86 | 1.39 |
| Timegate | 0.28 | 0.73 | 0.69 | 1.44 |
| Tornado | 0.98 | 0.97 | 0.90 | 1.09 |

**Consequence for the manuscript.** The ratio must never be reported alone. Two
targets with identical ratios can differ by 3x in absolute measurement error, and
for Anton532 the transferred model actually *matches* its own local calibration -
which sounds like a triumph but only means the local calibration is poor. The
honest presentation is a 2-D statement (absolute RMSE and ratio), and the claim
should be phrased as "files the calibration gap", not "achieves local accuracy".
This is exactly the kind of nuance a TIM reviewer will demand, and it was caught by
checking absolute numbers rather than trusting the summary ratio.

## Round 12 — Locally-adaptive intervals (fixed-width is valid but uninformative)

Discovered by inspection: the conformal interval as first built is symmetric and
*constant* across samples (half-width q). Coverage is valid on average, but the
instrument gives no per-measurement indication of reliability. In a process analyzer
that is a weak claim: the operative question is whether the number can be trusted
for THIS sample.

Added a normalised conformal layer, y_hat ± q·s(x), with s(x) the distance to the
k nearest standardization spectra (latent space) or in predicted-concentration
space. Both are label-free.

Effects observed on a pilot of 12 pairs:
- coverage stays in 0.79-1.00 (nominal 0.90) - the normalisation does not break validity;
- width now varies across samples (coefficient of variation up to 0.3);
- Spearman(half-width, |error|) is positive for glucose (0.18-0.57) and mixed for
  acetate/MgSO4, i.e. the adaptation helps most for the analyte where errors are
  largest and most heterogeneous - which is the practically relevant case.

Full evaluation is in `data/results/adaptive_intervals.csv`.

### F11 · Adaptive intervals: validity preserved, informativeness gained

Full evaluation, 1512 cells, nominal 90%.

| mode | coverage | under-cover <0.85 | mean width | positive width-error correlation |
|---|---|---|---|---|
| fixed ±q | 0.934 | 13.5% | 2.46 | 0% (by construction) |
| knn-dist scaled | 0.924 | 13.1% | 2.69 | **72.8%** |
| pred-knn scaled | 0.917 | 17.5% | 2.92 | **74.4%** |

Reading: scaling the interval by the distance to the nearest standardization
spectra costs 9% in width and 1 point of coverage, and buys a per-sample
uncertainty that is positively correlated with the actual error in ~73% of cells.
For a process analyzer that is the difference between "this method is 90%
accurate on average" and "this particular reading is trustworthy" - the latter is
what an operator needs and what TIM reviews measure.

`pred-knn` gives the strongest correlation but over-widens (19% more) and
under-covers more often; `knn-dist` is the better default.

### F12 · The self-assessment defect is universal, not one bad instrument

Coverage at nominal 90%, pooled calibration (current practice) vs cross-conformal:

| target instrument | pooled | full (cv+) | gain |
|---|---|---|---|
| Anton532 | 0.759 | 0.907 | +0.148 |
| Anton785 | 0.758 | 0.968 | +0.210 |
| Kaiser | 0.846 | 0.872 | +0.026 |
| Metrohm | 0.837 | 0.929 | +0.092 |
| MettlerToledo | 0.846 | 0.947 | +0.101 |
| Tec5 | 0.845 | 0.918 | +0.073 |
| Timegate | 0.767 | 0.900 | +0.133 |
| Tornado | 0.896 | 0.933 | +0.037 |

Pooled calibration under-covers on **every one of the eight** target instruments,
between 0.758 and 0.896 against a nominal 0.90. The size of the failure varies with
how heterogeneous the target's sample population is (Anton785 and Timegate worst,
Tornado best), which is exactly what one expects if the mechanism is "the
standardization samples are easier than the test samples". No instrument contradicts
the finding, so it is not an artefact of a single bad dataset.

## Round 13 — Multi-source fusion without labels

Seven candidate sources per target, target labels hidden. The criterion value is the
out-of-fold pairwise agreement already used inside SMART.

| strategy | mean ratio | median |
|---|---|---|
| single, chosen by criterion | **1.010** | 0.997 |
| criterion-weighted fusion | 1.083 | 1.022 |
| equal-weight fusion | 1.134 | 1.046 |
| oracle-best single (uses target labels) | 0.927 | 0.932 |

Two findings:
1. **Selecting one source with the label-free criterion is nearly oracle-optimal**:
   1.010 against an oracle 0.927, i.e. the selection loses only 9% of the available
   gain. This is a much stronger result than the single-pair selection study (F9)
   because there the *range* of candidates was narrow.
2. **Naive fusion hurts.** Equal-weighting seven sources is worse than picking one
   (1.134 vs 1.010) because a badly matched source contributes error that averaging
   cannot cancel. Criterion-weighting recovers part of it (1.083) but still loses to
   selection. Practical rule: select, do not average.

This corrects an intuition that would otherwise have gone into the paper: more
instruments is not better; the right instrument is.

## Round 14 — Correction and verification of the broth result

A first version of `run_broth_validation.py` reported the *in-sample* correlation for
the calibrated stage. Corrected to report the held-out correlation
(`pearson_r_test`). After the fix:

| target | analyte | direct r | after 6-point affine cal: held-out r | rel. RMSE |
|---|---|---|---|---|
| yeast_bioreactor (5 batches) | glucose | 0.985 | **0.9856** | 0.192 |
| E. coli fed-batch | glucose | 0.766 | 0.510 (k=6) / 0.765 (k=12) | 0.830 / 0.678 |
| E. coli fed-batch | acetate | 0.314 | 0.100 (k=6) / 0.312 (k=20) | 1.113 / 1.010 |

Independently re-verified with 20 different splits: the yeast glucose held-out
correlation is 0.9834-0.9900 (mean 0.9857), so the 0.985 is not a lucky seed and not
an artefact of the glucose zeros (excluding values < 1 mM gives the same 0.985;
Spearman 0.992).

E. coli glucose improves with more calibration points (0.51 -> 0.77) but does not
reach the yeast level, and acetate never transfers (0.10-0.31). Both are reported as
negative/mixed results rather than tuned away.

## Round 15 — Statistical significance (paired Wilcoxon + block bootstrap)

Because every method sees identical cells, comparisons are paired per
(pair, seed, analyte) and the bootstrap resamples whole instrument pairs to respect
dependence across seeds. n = 504 cells per comparison (3 seeds x 56 pairs x 3 analytes).

### Error ratio, SMART vs competitor (negative = SMART better)

| competitor | analyte | mean diff | 95% CI | p |
|---|---|---|---|---|
| direct | all | -14.24 | [-19.34, -9.83] | <1e-4 |
| PDS | all | -9.02 | [-12.25, -6.10] | <1e-4 |
| DS | all | -0.453 | [-0.654, -0.295] | <1e-4 |
| DS | Glucose | -0.625 | [-1.035, -0.306] | <1e-4 |
| DS | Acetate | -0.224 | [-0.318, -0.137] | <1e-4 |
| DS | MgSO4 | -0.511 | [-0.704, -0.338] | <1e-4 |
| TCCP-cv+ | all | -0.066 | [-0.130, -0.008] | <1e-4 |
| TCCP-cv+ | Glucose | -0.078 | [-0.201, +0.017] | 0.073 (n.s.) |
| TCCP-cv+ | MgSO4 | +0.091 | [-0.023, +0.214] | 0.027 (worse) |

Honest reading: SMART is significantly better than every classical baseline on every
analyte. Against TCCP-cv+ (same operator, fixed hyper-parameters) the advantage is
significant overall and on acetate, but NOT on glucose (p=0.073) and it is slightly
WORSE on MgSO4 (p=0.027). The manuscript must say this plainly: the label-free
selection buys most of its value on the analytes whose calibration is hardest
(see F7: acetate), and it is not a uniform win.

### Coverage gain vs pooled (current practice)

| method | analyte | mean gain | 95% CI | p |
|---|---|---|---|---|
| SMART | all | +0.107 | [+0.089, +0.126] | <1e-4 |
| TCCP-cv+ | all | +0.114 | [+0.098, +0.132] | <1e-4 |

Every analyte separately: p < 1e-4, CI excludes zero. The coverage finding is the
most robust result in the study.

## Round 16 — ⚠ Open discrepancy requiring resolution before submission

The adversarial reviewer's intermediate file `data/results/_tmp_selfassess.csv` (its
protocol definitions arrive with its report) measures the self-assessment bias with
different numbers than my Round 3 / F3:

| quantity | my F3 | reviewer's check |
|---|---|---|
| Pearson(in-sample err, true err) | 0.008 | +0.419 |
| median true/in-sample ratio | 2.06 | 1.49 |
| cells understating > 2x | (not computed) | 17/56 |

Both agree the in-sample score is biased low. They disagree on *how informative* it
is: my protocol found the in-sample score nearly useless for ranking pairs (r=0.008),
the reviewer's found it weakly informative (r=+0.419). Likely cause: different
definitions — my e_std was evaluated on standardization samples that the operator and
affine correction had seen, using the full paired set for fitting (the "pooled"
practice); the reviewer's r_in may use a different subset or a different operator
configuration.

**Action for the paper:** do not headline the r=0.008 "uncorrelated" claim until the
two protocols are reconciled. The robust claim, supported by both analyses, is the
*bias*: in-sample qualification understates the true error by a median 1.5–2.1x and
misses >2x underestimates on roughly a third of instrument pairs. Frame the
contribution around the bias and the coverage fix (which is significant at p<1e-4 in
every analyte), not around "the current practice is uninformative".

### Reconciliation of the self-assessment bias numbers (resolves Round 16)

Re-running the self-assessment with the exact operator+affine protocol used in the
main study (n=168 cells, 56 pairs x 3 seeds):

- Pearson(in-sample ratio, true ratio) = **+0.606**, Spearman +0.411
- median true/in-sample ratio = **1.59**
- cells where in-sample understates truth by >2x: **40/168 (24%)**

So the correct, defensible statement is NOT "uncorrelated" (that was an artefact of
the earlier, different protocol — DS-operator-only scoring, no affine correction).
The honest statement:

> Scoring the transferred calibration on the standardization samples it was fitted
> on understates the true held-out error by a median factor of 1.59, and by more
> than 2x on roughly a quarter of instrument pairs, while being only weakly
> informative about which pairs are bad (Spearman +0.41). Its coverage at nominal
> 90% is 0.819 with half the cells below 0.85 (p<1e-4 vs cross-conformal, which
> achieves 0.934).

F3 is amended accordingly in findings.md. The conclusion survives in its practical
form — in-sample qualification is biased and must not be used as a certificate — but
the "r = 0.008 uncorrelated" sentence is wrong and must not appear in the paper.

## Round 17 — Reviewer findings, verified and fixed

The adversarial review (defects #1-#13) was checked and the fixes applied. What
follows records what changed and which claims were re-based.

### Fixed in code
1. **Double use of standardization source spectra (review #3, fatal).** The source
   PLS was trained on Xs that still contained the standardization samples' source
   spectra. Measured by the reviewer: ~a third of the pooled-coverage failure was
   this leak, not the self-assessment protocol. Fixed in tccp.py (std_src_rows
   parameter, defensively excluded) and in every experiment script's `keep` mask.
   Also added the CV+ finite-sample inflation (Barber et al. 2021).
2. **DS baseline ridge convention (review #6, major).** baselines.py used ridge=1e-3
   against the raw Gram matrix; tccp.py uses 1e-2 relative to mean squared singular
   value — a factor of ~500 in effective regularisation. DS rewritten to the relative
   convention. The reviewer's own sweep puts tuned DS at 1.49, not 1.81: the honest
   margin over SMART is 1.36 vs 1.49, and the paper must defend that number.
3. **Self-assessment script (review #2, fatal-as-claimed).** New
   `run_selfassessment.py` computes the defect under one named definition. With the
   double-use leak REMOVED, the honest numbers are: median underestimation **1.15x**
   (was reported 2.06), >2x on only 6.2% of cells, Pearson +0.822 (positively
   informative — NOT uncorrelated). The earlier "r=0.008 / 2.06x / 10.4x" numbers are
   retired everywhere; they were a mixture of the PLS leak and a different scoring
   protocol.
4. **Split-branch provenance (review #4).** conformal_alpha0.1.csv's split arm was
   byte-identical to cv+ (stale file). Re-run from the current tree.

### Consequences for the story (important)
The **self-assessment narrative must be re-centred**. After the leak fix, in-sample
qualification is not wildly invalid; it is ~15% optimistic on average and strongly
rank-correlated with the truth. The robust headline is no longer "self-assessment
lies" but:
  (a) the leak we found and fixed is itself the metrological finding — leaving the
      standardization samples in the source calibration inflates apparent coverage
      from 0.819 to 0.890 pooled / and buying cv+'s 0.934 partly on that leak;
  (b) cross-conformal remains the honest interval, now with the leak removed, at
      slightly larger widths;
  (c) the low-rank distortion evidence (review #1) is a design artefact (3 analytes,
      5 replicates: k3(Xs)=0.972 ≈ k3(D)=0.954) and the global-operator argument
      must rest on predictive performance (tuned DS 1.49 vs best-window PDS 4.7),
      which survives.

## Round 18 — Second correction cycle (review fixes applied, everything re-run)

### Code changes
1. **Row-level double-use exclusion everywhere.** The reviewer's suggestion (remove
   the std source spectra from the PLS) turned out to be infeasible as stated: the
   shared design covers 235-270 of 270 source rows, so physical-sample-level removal
   empties the training set. Adopted instead: remove EXACTLY the rows the operator
   and affine correction see (indices ia2), keeping other replicate spectra of the
   same physical samples. This is the honest fix — the leak being removed is that
   those specific rows are scored in-sample, and replicates are separate
   measurements.
2. **CV+ inflation corrected.** The rank form ceil((n+1)(1-a))+k-1 exceeds n for
   n_cal ~ 36 and k=8, degenerating to "take the maximum residual" — which produced
   the over-coverage seen in the first re-run (cv+ 0.971, SMART 0.966 at nominal
   0.90). Replaced with the non-degenerate level inflation
   a_eff = a(n+1)/(n+1-k), giving levels 0.872-0.891 for the actual (n_cal, k) —
   a sensible small inflation. Fixed in tccp.py, smart.py and adaptive.py.
3. **Tuned baselines.** DS and PDS now get the same hyper-parameter grids and the
   same label-free selection criterion as the proposed method (DS-tuned,
   PDS-tuned in run_headtohead.py). DS ridge convention unified to relative-lambda.
4. **Survivorship-corrected budget.** run_std_size.py now reports the curve capped
   at n=30 (all 56 pairs) plus the n>=40 subset restricted to pairs that can supply
   it, and pair-level Wilson CIs for the "within 1.5x" fractions.
5. **Distortion control.** Concentration-residualised decomposition added; the
   paper's argument for the global operator now rests on predictive performance.
6. **run_loio.py leak fixed** (standardization set now excludes test-half samples;
   source rows for std samples excluded).
7. **run_selfassessment.py added** — the one named definition the paper quotes.

### First re-run numbers (with the rank-form inflation, hence over-coverage; final
numbers pending the corrected-inflation re-run)
- SMART 1.654 mean / 1.441 median ratio; DS-tuned 2.962/2.126; PDS-tuned 4.150/3.067;
  direct 10.83. The leak had been flattering EVERY method; all absolute errors rise.
- Pooled coverage 0.890 (was 0.819) — confirming the reviewer's estimate that about
  a third of the old "current practice failure" was our own pipeline leak.
- Pooled under-coverage 28.1% (was 50.4%).

### Conformal-mode reading after the fix (final numbers from the corrected run)

`conformal_alpha0.1.csv` (corrected non-degenerate CV+ level):

| mode | mean coverage | width | reading |
|---|---|---|---|
| pooled (current practice) | 0.838 | narrow | still under nominal |
| cv+ | 0.853 | moderate | now UNDER nominal too — the level inflation alone does not fix small-n_cal |
| split | **0.944** | wide | the only mode reliably at/above nominal |

The interesting scientific point this exposes: with n_cal ~ 20-36 standardization
samples, the cross-conformal fold machinery is itself noisy; the humble 50/50 split
("use half the standards to fit, half to certify") is the only configuration whose
coverage is trustworthy. That is a defensible, actionable recommendation for a
laboratory protocol: **split the standards; do not reuse them.** It also sharpens
the paper's practical message vs the cv+ story we had before.

### Survivorship structure — the reviewer's direction was wrong, and the data says so

The reviewer claimed the n=40/50-capable pairs are "the easy ones". Measured on the
re-run: the 11 pairs that can supply n_std=40 have mean ratio **2.03** at n_std=20,
WORSE than the all-pairs mean 1.83. The pairs with large standardization sets are the
Timegate/Metrohm/Tec5/Tornado cross-combinations — spectroscopically dissimilar, hence
hard. So the old n=50 figure (1.16) was not survivorship flattery of easy pairs; it was
the genuine sample-size effect on hard pairs. The budget claim is now stated as:
**n=30 (all 56 pairs reachable): mean 1.66, median 1.43; pair-level Wilson CI on
"within 1.5x": 0.519 [0.389, 0.646]**. The n>=40 rows are reported for the 11-pair
subset only, labelled as such.

### Significance after all fixes (paired Wilcoxon + pair-block bootstrap)

SMART vs DS-tuned: all −1.40 [−1.82, −1.01], p<1e-4 — every analyte significant.
SMART vs TCCP-cv+ (fixed hyper-parameters): all −0.102 [−0.185, −0.023], p<1e-4;
significant on acetate (p=1e-4) and glucose (p=0.003); **not significant on MgSO4**
(p=0.209) — same honest caveat as before the fix, still to be stated in the paper.

## Round 19 — The interval story, resolved honestly

After the CV+ correction, the three interval modes were measured on identical data:

| mode | mean coverage @ 0.90 | reading |
|---|---|---|
| pooled (current practice) | 0.890 (headtohead) / 0.838 (conformal) | under nominal |
| cv+ (cross-conformal, corrected) | 0.866 / 0.853 | still under nominal |
| split (disjoint halves) | **0.934** / **0.944** | the only certifiable mode |

Root cause, now measured rather than assumed: the standardization population is
systematically easier than the target's held-out population, so ANY quantile
calibrated on standardization residuals — pooled or cross-fold — inherits that
optimism. The CV+ finite-sample machinery cannot fix a distributional mismatch; it
only fixes sample-size effects. The disjoint split survives because the calibration
residuals there come from an operator fitted on DIFFERENT samples, making them (empirically)
representative enough.

**Action**: SMART's default interval is now `split` (operator on half the standards,
quantile on the other half). Cost: wider intervals (7.11 vs 4.76 g/L mean width for
glucose). Benefit: 0.934 mean coverage with only 17.0% of cells below 0.85 and 2.4%
below 0.70 — the only configuration a metrologist could sign off.

This reframes the paper's uncertainty section: the contribution is not "cross-conformal
fixes everything" but "**how you use the standardization samples decides whether your
interval is valid: reuse them and you certify optimism; split them and you certify
honesty**" — a crisp, actionable, previously unreported protocol finding.

## Round 20 — Post-review final state (all experiments re-run from one code tree)

Head-to-head (split-mode SMART, tuned baselines, row-level double-use exclusion):

| method | ratio mean / median | coverage @0.90 | under-cover cells |
|---|---|---|---|
| direct (no transfer) | 10.83 / 5.23 | — | — |
| PDS (fixed window 7) | 5.63 / 3.75 | — | — |
| PDS-tuned (window search) | 4.15 / 3.07 | — | — |
| DS (fixed ridge) | 3.06 / 2.17 | — | — |
| DS-tuned (ridge search) | 2.96 / 2.13 | — | — |
| TCCP-cv+ (fixed nc=8) | 1.76 / 1.54 | 0.866 | 38.2% |
| TCCP-pooled | 1.76 / 1.54 | 0.890 | 28.1% |
| **SMART (proposed, split interval)** | **1.65 / 1.44** | **0.934** | **17.0%** |

Significance (paired Wilcoxon + pair-block bootstrap, n=288):
- SMART vs DS-tuned: −1.40 [−1.82, −1.01], p≈2e-35, significant on all three analytes.
- SMART vs TCCP-cv+ (fixed hyper-parameters): −0.102 [−0.185, −0.023], p<1e-4 overall;
  acetate p=1e-4, glucose p=0.003, MgSO4 p=0.209 (n.s. — stated plainly).
- SMART split-interval coverage gain vs pooled: +0.077 [0.065, 0.088], p<1e-4.

Definitive numbers now quoted by the paper (all trace to data/results/):
- budget n=30: 1.66/1.43; pairs<=1.5x: 28/54 = 0.519 Wilson [0.389, 0.646]
- selection criterion: Spearman +0.711, regret 0.105
- multisource: criterion-best single 1.183 vs oracle 1.026 vs equal-weight 1.230
- self-assessment (named definition): 1.15x median, Pearson +0.822
- broth: yeast glucose held-out r=0.986 (5 batches, within-batch ≥0.96, null 0.71);
  acetate fails 0.31; fed-batch breaks calibration (13x, all bias, interval vacuous)
