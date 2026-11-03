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

## Amendment 2 (E0) — 口径统一

审查（`TIM_投稿评估与增强方案.md`）指出多处结果表之间口径不一致。本文件与
`to_human/实验报告.md` 已按下表统一，任何引用数字时须注明所属实验。

### 1. 三套"仪器对"集合必须区分

| 记号 | 定义 | 数量 |
|---|---|---|
| 有序对 | 8×7，不要求有足够配对样本 | 56 |
| **可用对（head-to-head / 消融）** | 满足最小标准样本数、行级剔除后源训练集 ≥20 行 | **32**（288 格，3 种子×3 分析物） |
| **预算对（std_size / conformal）** | 满足 n_std ≥ 20，允许更多对进入 | **54**（486 格 / 1458 格） |
| n≥40 可达对 | 配对样本数 ≥40 | 11–23（随档位变化） |

### 2. 两套区间覆盖数字必须注明来源

| 实验 | 文件 | 对数 | pooled | cv+ | split | 宽度增加 |
|---|---|---|---|---|---|---|
| A | `conformal_alpha0.1.csv` | 54 | 0.838 | 0.853 | 0.944 | +98% |
| B | `headtohead.csv` | 32 | 0.890 | 0.866 | 0.934 | +35% |

正文主结果用 B（与精度表同口径）；A 作为补充并在每次引用时标注"54 对口径"。

### 3. 消融与 head-to-head 不是同一批数据

消融 = 54 对 / 486 格（head-to-head 的严格超集，另有 198 格）。因此消融 `full` 的
1.466 与 head-to-head 的 SMART 1.654 不矛盾，差异源于仪器对集合不同。

### 4. `-labelsel` 与 `-cv (pooled)` 两行同值是实现使然

`-labelsel` 变体的 `cv` 参数为默认真值，其区间标定走 pooled 路径，故与 `-cv (pooled)`
在实现上等价。做干净单因素消融须保持选参开启、只改区间协议——该干净版本见
`experiments/h_uncertainty/run_e3_protocols.py` 的协议对照。

### 5. 显著性表的基线标签

`significance_ratio.csv` 的 `competitor` 列**只含未调参基线**
（`DS`、`PDS`、`TCCP-cv+`、`direct`）。调参后基线的差值只能由均值相减得到
（SMART−DS-tuned = −1.308，SMART−PDS-tuned = −2.496），不是配对检验。
`run_significance.py` 已扩展为显式列出全部竞争者并检查其是否存在于 head-to-head。

### 6. 已被证伪并撤回的表述

| 原表述 | 状态 | 依据 |
|---|---|---|
| 自评偏差 2.06x、r=0.008、与真值不相关 | **撤回** | 双重使用泄漏 + 评分协议差异；修复后为 1.15x、r=+0.822 |
| 跨仪器失真是低秩的（3 分量 79–96%） | **不作论据** | 92.4% 可由 3 浓度+截距解释；源谱自身 k3=0.972 ≥ 失真 0.954 |
| 标准样本总体系统性地更容易 | **证伪** | 折外标定残差/测试残差 = 1.148（测试更大），q90 比 1.063 |
| 选参准则是核心算法贡献 | **降级** | 与普通 5 折 CV 完全等价（54/54 同选，Spearman 与遗憾度逐一相同） |
| split 是唯一可靠协议 | **限定** | 0.934–0.944 与有限样本天花板 k/(m+1) 一致；m≤8 时区间无穷 |
