# Code Walkthrough: `experiments/statistical_significance.py`

**Purpose:** Formal **hypothesis testing** of MG-ECFM against BP1–BP4, implemented with NumPy/`math` only (SciPy is used only by unit tests, to cross-check). It reads the drift-evaluation outputs and writes `reports/statistical_significance_results.json`.

**Inputs:** `drift_evaluation_results.csv` (per model × window) and `drift_evaluation_predictions.csv` (per model × transaction).

---

## 1. Tests (docstring)
* **McNemar's test** (paired classification on identical test transactions): `b` = A correct & B wrong, `c` = A wrong & B correct; `χ² = (|b−c|−1)²/(b+c)` with continuity correction, `p = erfc(√(χ²/2))`; the exact binomial p-value is also reported (preferred when `b+c < 25`); odds ratio `b/c` with a Wald 95 % CI (Haldane–Anscombe +0.5 when a count is zero).
* **Wilcoxon signed-rank test** across the 50 windows on per-window PR-AUC differences (undefined windows dropped, zero differences discarded). Statistic `T = min(W+, W−)`; **exact** null distribution when `n ≤ 50` and no ties, else normal approximation with tie correction. Two-sided and one-sided (`H1: A > B`) p-values, rank-biserial correlation and effect size `r = |z|/√n`. F2 and EML are secondary endpoints.
* **Holm–Bonferroni** correction across the four baselines within each test family.
* **Moving-window block bootstrap** (B = 1000): the 50 windows are resampled with replacement and the pooled metric recomputed, preserving within-window dependence. 95 % percentile intervals for PR-AUC, ROC-AUC, F1, F2, Brier, EML of each model and for the paired difference MG-ECFM − baseline.

## 2. Section-by-section (by outline)

### 2.1 Distribution helpers (lines 82–131)
`chi2_sf_df1` (df = 1 survival function via `erfc`), `normal_sf`, `binomial_two_sided_p`, `rankdata_average` (tie-aware ranks), `signed_rank_null_counts(n)` (exact null distribution of the signed-rank statistic by dynamic programming).

### 2.2 McNemar (lines 137–201)
`mcnemar_test(correct_a, correct_b, …)` returns `b`, `c`, χ², asymptotic and exact p, odds ratio and CI. `_normal_quantile` gives z-values for the Wald interval.

### 2.3 Wilcoxon and multiplicity (lines 207–290)
`wilcoxon_signed_rank(x, y, alternative, …)` and `holm_bonferroni(p_values)`.

### 2.4 Bootstrap (lines 295–366)
`pooled_metric(name, y, p, yhat, …)` (PR-AUC, ROC-AUC, F1, F2, Brier, EML pooled over selected windows), `block_bootstrap_ci(statistic, n_blocks, n_resamples, …)`, and `WindowedPredictions` — groups predictions by window so resampling is by window.

### 2.3 Driver (lines 373–500)
`run_statistical_tests(results, predictions, proposed="mg_ecfm", …)` runs every test for each baseline, applies Holm correction, and assembles the JSON (per-model bootstrap CIs, paired differences, McNemar and Wilcoxon tables). `_jsonable`, `main` (CLI: `--results`, `--predictions`, `--output`, `--seed`; returns 2 if inputs are missing).

## 3. Interpretation notes for the paper
* **McNemar compares errors at a fixed threshold**; it is *not* a test for PR-AUC differences. For PR-AUC use the bootstrap CI and the window-level Wilcoxon test (as done here). The McNemar p-values here are < 0.001 against all four baselines simply because millions of transactions make any difference significant — report the effect (odds ratio) too.
* With 50 correlated windows, the Wilcoxon test's independence assumption is approximate.
* In our PaySim-2M results: MG-ECFM vs BP1/BP2/BP3 PR-AUC differences are significant (Wilcoxon, Holm-adjusted p ≤ 0.012); vs BP4 (ARGUS) it is not (p = 0.28).
