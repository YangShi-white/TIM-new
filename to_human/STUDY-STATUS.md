# TIM paper — study status

**Date:** 2 November 2026
**Scope:** experiments completed; no figures and no manuscript drafted, per instruction.
**Everything lives in `D:\TIM`.**

---

## 1. What the paper claims

You have eight Raman analyzers measuring the same bioprocess samples. Buying a full
local calibration on every new one is the cost the field wants to avoid. Existing work
(Lange et al., Spectrochim. Acta A 334:125861, 2025) shows a jointly-trained CNN on
all eight instruments works — but only because every instrument has labelled data,
which removes the very cost being avoided.

This study does the harder and more useful version: **transfer a calibration with no
reference measurements at the new instrument at all**, and — the part nobody has done
for bioprocess Raman — attach a **measurement-uncertainty statement that is
statistically valid**.

The headline technical finding is not an accuracy number. It is that the standard way
of qualifying a transferred analyzer is **wrong**: scoring it on the samples that
fitted the transfer understates the true error by a median factor of 1.59 (a quarter
of instrument pairs by more than 2x) and ranks pairs only weakly (Spearman +0.41),
so roughly half the pairs would get a certificate that overstates confidence by
10–20 percentage points (mean coverage 0.819 at nominal 0.90; the fix is significant
at p<1e-4 on every analyte).

*Amendment (research-log Round 16):* an earlier draft claimed the in-sample score was
"uncorrelated" with the truth (r = 0.008). That number was an artefact of a different
scoring protocol (operator-only, no affine correction) and has been corrected. The
bias and the coverage failure stand; the "uncorrelated" sentence does not.

## 2. The four contributions

| # | contribution | key number | evidence file |
|---|---|---|---|
| C1 | A single global low-rank operator beats windowed PDS for cross-instrument Raman transfer, explained by a measured distortion structure | 1.33 vs 1.94 ratio; 92.5% of distortion is a constant offset, 3 components = 94% energy | `ablation.csv`, `distortion_structure.csv` |
| C2 | Self-assessment on the standardization set is invalid; cross-conformal recalibration fixes it | coverage 0.819 → 0.922 at nominal 0.90, under-coverage 50.4% → 18.3% | `headtohead.csv`, `ablation.csv` |
| C3 | A quantified transfer-standard budget, the question practitioners actually ask | 20 standards → 1.69x local calibration; 50 → 1.16x; 55% of cells within 1.5x at n=20 | `std_size_curve.csv` |
| C4 | Hyper-parameters and source instrument can be selected **without target labels** | selection regret 0.082; source choice 1.010 vs oracle 0.927 vs equal-weight fusion 1.134 | `selection_validity.csv`, `multisource_fusion.csv` |

Supporting: locally-adaptive intervals (C5) — width tracks per-sample error in 73% of
cells at a 9% width cost (`adaptive_intervals.csv`).

## 3. What was run

- 56 ordered instrument pairs × 3 seeds = 168 transfer problems per experiment.
- 2261 calibration spectra (601 physical samples), 8 instruments, grids 486–5472 points, 532 and 785 nm.
- Real-broth validation on four independent fermentations (S. cerevisiae ×2,
  E. coli, plus batch-structure analysis over 5 batches).
- ~14,000 experiment cells in total.

## 4. The result that most constrains the story

**F8 / batch hold-out.** Ordinary batch-to-batch variation is nearly harmless (4–9%
relative error across B1–B4 of the yeast bioreactor set). But a **fed-batch** regime
breaks the calibration completely: 13x error, almost all of it bias (−16.7 mM glucose,
−98.9 mM ethanol), and the interval covers poorly for ethanol (0.714).

That is the honest limitation: the method is conditional on the operating regime, and
any regime change must be detected before the number is trusted. It also marks the
natural next paper.

## 5. Known weaknesses (state them, do not hide them)

1. **Ratio reporting is treacherous.** `RMSE / local RMSE` flatters badly calibrated
   targets. Anton532 scores ~1.2 but that is because its own local calibration is
   poor. Always report absolute RMSE alongside (see research-log Round 11).
2. **Pairing requirement.** The standardization samples must be the same physical
   samples on both instruments. That needs a sample bank; it is not free.
3. **Exchangeability caveat.** Conformal validity assumes calibration residuals and
   test residuals are exchangeable. The standardization samples are a *subsample of
   the same sample bank* as the source training set, so this holds only up to the
   pairing assumption. The empirical control (F10: split-half coverage 0.940) supports
   it but does not prove it. An adversarially-minded reviewer will push here, and the
   answer should be "we check it empirically and state the assumption", not "it is
   guaranteed".
4. **Acetate does not transfer to broth** (r = 0.31). Only glucose does, strongly.
5. **No in-line data.** Every public set here is at-line/off-line. An in-line probe
   with fouling and drift is the deployment case and is untested.
6. **Sample counts.** The real-broth sets are 58–379 spectra; the batch study rests on
   65 spectra over 5 batches. Adequate for direction of effect, not for tight RMSEP
   claims. Saying so is cheaper than being caught.

## 6. What I would do next, in priority order

1. **Regime-change detection** — the fed-batch failure is the most interesting result
   in the whole study and deserves its own experiment: does the conformal width rise
   *before* the error does? If yes, that is a deployable alarm.
2. **Dilution-series standards** — the current budget sweep varies *count* but holds
   the standards' span fixed. A real qualification protocol varies the concentration
   span too, and the answer may differ.
3. **Wavelength-axis drift simulation** — the correspondence approach is supposed to
   handle axis drift and this has not been tested at all.
4. **A second analyte family** beyond glucose/acetate/MgSO4 before generalising.

## 7. Files

```
research-log.md        chronological decisions, including two bugs that changed a conclusion
findings.md            synthesised findings F1-F12
protocol.md            the locked experimental protocol
to_human/              this file and the evidence package
src/                   timdata, protocol, baselines, tccp, smart, adaptive, brothdata
experiments/           h_transfer, h_uncertainty, h_broth
data/raw/              all downloaded datasets (CC-BY-4.0)
data/results/          12 result CSVs, every number in this document traces to one
scripts/               the two resumable downloaders
```
