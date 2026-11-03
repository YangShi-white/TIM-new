# Findings

Running synthesis of what the experiments actually showed. Every number here is
produced by a script in `experiments/` and a CSV in `data/results/`.

## The setting

A validated Raman soft sensor exists on instrument A. A second instrument B is
purchased or available at another site. The laboratory wants to run B **without
buying a full local calibration**: no new reference assays of B's broth, only a set
of paired standardization samples whose reference values were already measured for A.

Eight instruments, 2261 spectra (601 physical samples), 56 ordered pairs, three analytes
(glucose, acetate, magnesium sulfate). All public data; see `data/raw`.

## F1 · A global operator beats windowed standardization — argued from prediction, not from an SVD

The measured inter-instrument distortion is broad-band and mean-dominated (median
92.5% of its variance is a constant offset). **But that decomposition must not be the
evidence**: after controlling for the 3-analyte design matrix, 92.4% of the distortion
variance is explained by the concentration design + intercept, and the source spectra
themselves are exactly as "low-rank" (k3 = 0.972) as the distortion (k3 = 0.954) —
the apparent low rank is a construction of the mixture design, not a property of the
instruments (`distortion_structure.csv`, control columns). The correct argument is
purely predictive: with both methods given the same hyper-parameter budget, the global
ridge operator reaches 1.49x a full local calibration while the best windowed PDS
configuration sits at ~4.7x. The decomposition is retained only as a description of
what the learned operator looks like, never as the justification for choosing it.

## F2 · The source instrument decides, not the target

Median transfer ratio at 20 standards: Timegate 1.24, Kaiser 1.30, Metrohm 1.34,
Tec5 1.44, Anton785 1.45, MettlerToledo 1.52, Tornado 1.66, Anton532 2.42.
Anton532 (532 nm excitation, highest raw intensity, weakest own calibration) is
unusable as a transfer source at any budget. This is an actionable instrument-pairing
criterion that no prior bioprocess Raman paper reports.

## F3 · The double-use leak, then a modest residual self-assessment bias

Two separate defects were conflated in the first draft, and the hostile review
caught both.

**(a) Double use (the big one).** The standardization samples' source spectra were
kept inside the source PLS training set, so every "out-of-fold" conformal residual
was in-sample with respect to the calibration model. Removing exactly those rows
(keeping other replicate spectra of the same physical samples, so the source set
stays usable) moves pooled-calibration coverage from 0.819 to ~0.89 and shrinks the
cv+ advantage accordingly. The apparent failure of the *current practice* was
therefore partly our own pipeline's leak, and the paper must present the corrected
numbers, not the leaky ones.

**(b) Residual self-assessment bias (the small one).** With the leak fixed and under
one named definition (`run_selfassessment.py`: operator + affine fitted on all
standardization samples, scored on the same samples), in-sample qualification
understates the held-out error by a median factor of **1.15** (mean 1.29, >2x on
6.2% of cells, worst 5.3x) and is strongly rank-informative (Pearson +0.82,
Spearman +0.79). So the practice is *mildly optimistic*, not misleading about which
pairs are bad. The honest claim is "budget ~15% optimism and rare severe misses",
not "self-assessment lies" — the earlier 2.06x / 10.4x / r=0.008 numbers were
artefacts of the leak plus a different scoring protocol and are retired.

## F4 · Cross-conformal calibration restores honest coverage — as an empirical statement

*(Numbers below are from the leak-free re-run; see F3(a). The conformal validity is
approximate, not distribution-free: the calibration residuals come from the paired
standardization population and the test residuals from the target's random holdout,
which are not the same population. The claim the paper can defend is "empirically
calibrated on 56 instrument pairs", backed by the split-half control in F10.)*

Final measurement (identical data, three modes):

| interval mode | mean coverage @ 0.90 | mean width (glucose, g/L) |
|---|---|---|
| pooled (fit and certify on the same standards) | 0.890 | 5.2 |
| cv+ (cross-conformal, corrected inflation) | 0.866 | 4.8 |
| **split (fit on half, certify on the other half)** | **0.934** | 7.1 |

The root cause is measurable: the standardization population is systematically easier
than the target's held-out population, so *any* quantile calibrated on standardization
residuals inherits optimism — cross-conformal machinery cannot fix a distributional
mismatch, only sample-size effects. The disjoint split survives because its
calibration residuals come from an operator fitted on different samples. The paper's
protocol finding: **reuse the standards and you certify optimism; split them and you
certify honesty** — at a measurable width cost (5.2 → 7.1 g/L).

## F5 · A transfer-standard budget exists and is measurable (with a survivorship caveat)

Ratio to a full local calibration. **Important**: at n_std = 40–50 only 8–13 of the 56
pairs can supply that many standardization samples, and those pairs are systematically
the easy ones (their mean ratio at n=20 is 1.48 vs 1.69 for all pairs). The headline
curve must therefore be capped at n = 30 — the largest size all 56 pairs support —
with the n ≥ 40 rows reported only for the restricted subset, clearly labelled.

| n_std | 4 | 8 | 12 | 20 | 30 | (40, subset) |
|---|---|---|---|---|---|---|
| mean ratio | 3.53 | 2.56 | 2.14 | 1.83 | 1.66 | 1.49 (11 hard pairs only) |

Report the "% within 1.5x" figure at the **pair** level with a Wilson CI, never as a
cell fraction (cells share operators and splits, so they are not independent; the
seed-wise correlation of the ratio is ~0.76). Post-fix pair-level numbers at n=30:
**28/54 pairs within 1.5x = 0.519, Wilson CI [0.389, 0.646]**; at n=20: 21/54 = 0.389
[0.270, 0.522].

A note that contradicts the first reading of the survivorship concern: the 11 pairs
that can supply n_std=40 are NOT the easy ones — their mean ratio at n=20 is 2.03 vs
1.83 for all pairs (they are the spectroscopically dissimilar Timegate/Metrohm/Tec5/
Tornado cross-combinations). The n>=40 improvement is therefore a genuine
sample-size effect on hard pairs, not flattery of easy ones.

## F6 · The residual gap is partly irreducible

Transfer error lands at 0.96–1.27x a full local calibration for 7 of 8 sources, but
2–3x above each source's own accuracy. The target instrument has its own precision
floor (mean over analytes: Metrohm 0.45, Kaiser 0.60, Timegate 0.64 ... Anton532 1.68
g/L). The honest claim is therefore not "transfer closes the gap" but "transfer with
a stated uncertainty makes a new instrument usable without a local calibration."

## F7 · A mixture calibration partially survives real broth

Designed mixtures (SubstrateMixRaman, 5614 complete-label spectra) applied to
independent real fermentations. The yeast rows are **5 batches × ~12 time points, not
60 independent samples**, and within every batch glucose falls almost linearly with
time (corr(glucose, time) = −0.92 to −0.96), so a pooled r is inflated by the batch
structure. The honest presentation:

| target | analyte | structure | pooled r | time-trend null | within-batch r |
|---|---|---|---|---|---|
| S. cerevisiae, 5 batches | glucose | 5×12 pts | 0.985 | 0.71 | ≥0.96 in all 5 |
| E. coli fed-batch | glucose | 373 pts, no time trend | 0.766 | 0.006 | (not confounded) |
| E. coli fed-batch | acetate | 379 pts | 0.314 | — | halves disagree (0.42 / 0.07) |

Reading: the spectroscopic model (0.985) clearly beats a "glucose declines over
time" null (0.71), and the within-batch correlations (≥0.96) confirm the signal is
spectroscopic, not temporal. But the n must be stated as 5 batches, and the direct
vs calibrated rows carry different label budgets (0 vs k reference assays) — never
present them as one condition. Acetate does not transfer and is reported as a
failure.

## Data-integrity notes

- `RamanBench/yeast_fermentation` and `RamanSpectraEthanolicYeastFermentations` are the
  same fermentation (same 1900-wavenumber grid, 58 rows, labels correlated 0.68).
  Only one is used.
- The Ralstonia set is excluded: its README mixes experimental with jittered synthetic
  data and the synthetic rows cannot be de-duplicated out.
- Ethanol has no SubstrateMixRaman counterpart. An early draft mapped it to glycerol;
  that is chemically indefensible and was removed.
- "Missing" labels are NaN and 0 is a legitimate concentration, so masking always uses
  `notna()`, never `!= 0`.

## F13 · The fed-batch regime change breaks the calibration AND makes the interval vacuous

Leave-one-batch-out on the 5-batch yeast set: ordinary batch variation (B1–B4) costs
only 4–9% relative error. The fed-batch batch (FB) breaks everything — glucose 13x,
ethanol 13x, biomass 12x, and the error is nearly all bias (−16.7 mM glucose,
−98.9 mM ethanol).

The coverage column must be read WITH its width. FB glucose "coverage 1.00" is not a
success: the interval's half-width is ~116 mM against a reference range of 0–121 mM,
i.e. the interval spans the entire calibration range and is vacuous. Ethanol covers
0.714 (a real failure) and **biomass 0.857, also below nominal and previously
unreported**. The deployment rule the paper should state: under a regime change the
conformal interval preserves its coverage only by widening to the point of
uselessness, so a regime-change detector is required before any number from this
calibration is trusted.
