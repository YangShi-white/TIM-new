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

## F4 · The interval defect is in-sample calibration; several protocols fix it, and the narrowest one is not the split

Two things were separated by measurement (E3, `e3_protocols.csv`, 54 pairs / 2916 cells,
nominal 90%):

| protocol | coverage | width | width vs pooled | effective n_cal |
|---|---|---|---|---|
| pooled (in-sample residuals) | 0.838 | 3.80 | 1.00 | 37 |
| split (disjoint halves) | 0.928 | 7.50 | 1.97 | 18 |
| cv+ (5-fold cross-conformal) | 0.932 | 5.49 | 1.44 | 37 |
| **jackknife+ (leave-one-standard-out)** | **0.926** | **5.32** | **1.40** | 37 |
| 2-fold (roles swapped, pooled residuals) | 0.950 | 6.45 | 1.70 | 37 |
| Mondrian (per-analyte quantiles) | 0.932 | 5.49 | 1.44 | 37 |

Three conclusions:

1. **pooled is uniquely bad** (45.5% of cells under-cover). Every out-of-fold protocol
   reaches roughly nominal coverage. This is the paper's protocol finding.
2. **split is NOT the best replacement.** It wastes half the standards (n_cal 18 vs 37)
   and is therefore the WIDEST workable interval (1.97x pooled). Its earlier apparent
   superiority was the finite-sample ceiling k/(m+1), not a virtue.
3. **jackknife+ dominates split**: same coverage (0.926 vs 0.928), 29% narrower
   (5.32 vs 7.50), and it uses every standard sample.

This also validates the corrected mechanism (F3/E4): the defect was in-sample
optimism, not a "standard samples are easier" distribution mismatch - which is why
cross-fitting rescues the interval without discarding data.

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

**E1 补充（决定性）**：本曲线只与"需要源仪器的迁移方法"比较，**缺少最强合法竞争者**——
标准样本在目标仪器上有光谱、其参考值 `Ys_std` 又已由源实验室测定，因此可以直接用这 n 条样本
在目标仪器上训练本地 PLS（B1），同样不增加任何参考测定。若 B1 在某个 n 上不弱于迁移，
则该 n 处迁移没有价值。交叉点曲线见 `e1_baselines.csv` / `e1_crossover.csv`。

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

## F14 · The label-free selection criterion is algebraically equivalent to ordinary cross-validation

Reviewer concern M12, tested under a fully nested protocol with identical folds
(`criterion_vs_cv.csv`, 54 pairs, 810 cells):

| criterion | Spearman vs truth (median) | normalised regret (median) | picks |
|---|---|---|---|
| pairwise difference | +0.705 | 0.106 | - |
| plain 5-fold CV | +0.707 | 0.106 | - |
| **same candidate picked** | - | - | **54/54 = 1.000** |

They are the same estimator up to a constant. **The criterion must not be presented
as a novel algorithmic contribution**; its legitimate value is limited to (i) being a
practical, verified rule and (ii) applying uniformly to operator hyper-parameters,
source-instrument choice, and fusion weights.

## F15 · Protocol upgrade: jackknife+ replaces the split, and beats it

E3, identical data and point predictions, six protocols, 54 pairs / 2916 cells:

| protocol | coverage | width | with respect to pooled | effective n_cal |
|---|---|---|---|---|
| pooled | 0.838 | 3.80 | 1.00 | 37 |
| split | 0.928 | 7.50 | 1.97 | 18 |
| cv+ | 0.932 | 5.49 | 1.44 | 37 |
| **jackknife+** | **0.926** | **5.32** | **1.40** | 37 |
| 2-fold swapped | 0.950 | 6.45 | 1.70 | 37 |
| Mondrian | 0.932 | 5.49 | 1.44 | 37 |

- pooled is uniquely bad (45.5% of cells under-cover); every out-of-fold protocol
  reaches approximately nominal coverage.
- split is **not** the best replacement: it halves the effective calibration set and
  is therefore the widest workable protocol.
- **jackknife+ matches split's coverage with 29% narrower intervals and uses all
  standards.** This becomes the recommended protocol.
- Mondrian equals cv+ exactly, because the three analytes' residual distributions are
  similar - stratifying by analyte buys nothing here (report as a negative result).

## F16 · DECISIVE NEGATIVE RESULT: transfer does not beat simply calibrating locally on the same standards

The original comparison table omitted the strongest legitimate competitor. The
standardization samples exist on the TARGET instrument and their reference values are
already known, so one can simply train a local PLS on them - no new reference
analytics, and no source instrument at all.

Median ratio to a full local calibration (162 pair x analyte cells, 6 draws):

| n_std | SMART (transfer) | **B1 local PLS on the standards** | B2 augmented PLS | SMART beats B1 |
|---|---|---|---|---|
| 8 | 2.158 | 2.130 | **1.635** | 49.8% |
| 12 | 1.801 | 1.776 | **1.466** | 52.9% |
| 20 | 1.535 | **1.320** | 1.313 | 40.5% |
| 30 | 1.387 | **1.160** | 1.234 | 36.9% |

- At n <= 12, transfer ties local PLS, and **augmented PLS (source + standards) is
  clearly better**.
- At n >= 20, **local PLS wins outright** (transfer wins only 36.9-40.5% of cells).
- Across all tested n, 80.9% of pair-analytes never see transfer beat B1.
- Per analyte: transfer helps only glucose at n<=12 and acetate at n=12-30; it is
  consistently worse for magnesium sulfate.

**The comparison is conservative in transfer's favour**: SMART uses strictly MORE
information (n target spectra + n paired source spectra + the whole source training
set) than B1 (n target spectra + their labels), and still loses.

**What this invalidates:** the economic argument ("a new instrument requires dozens of
reference assays") - because the standards *are* reference assays. If n assays are
going to be done anyway, calibrate locally with them.

**What survives:** the interval protocol finding (F15) is independent of whether
transfer is worth doing - any calibration, including B1, needs a valid uncertainty
statement. Pooled calibration certifies optimism; jackknife+ certifies honestly and
is the narrowest valid option.

**Required repositioning:** the paper can no longer be "transfer beats local
calibration". It must become "when transfer is used, how to certify its uncertainty -
and the crossover analysis that says when transfer is not worth it at all".

**E1b — the transfer has no advantage on either axis.** Giving the local model the same
jackknife+ conformal layer (23292 cells, identical draws, identical interval style):

| n_std | accuracy: B1+jack | accuracy: transfer | coverage: B1+jack | coverage: transfer | width: B1+jack | width: transfer |
|---|---|---|---|---|---|---|
| 8 | 2.050 | 2.033 | 0.877 | 1.000* | 8.03 | **infinite*** |
| 12 | **1.715** | 1.786 | 0.907 | 0.916 | **7.24** | 8.08 |
| 20 | **1.302** | 1.537 | **0.945** | 0.944 | **6.27** | 7.69 |
| 30 | **1.166** | 1.443 | **0.925** | 0.921 | **4.20** | 5.38 |

\* at n=8 the split interval degenerates: after halving, m=4, so the conformal
quantile is unbounded and every transfer interval is infinite, while the local
model still returns a finite interval.

Per-cell, at n=30: the local model is more accurate in 64.2% of cells, has coverage
no worse in 77.6%, and **is both more accurate AND at least as well covered in 52.8%**.

**Verdict: in this setting the transfer has no advantage on accuracy, coverage, or
width.** The transfer claim must be withdrawn entirely. What survives is the
uncertainty-protocol study (F15), which applies to the local calibration just as
much - and is arguably more useful there, since local calibration is what a
laboratory would actually do.
