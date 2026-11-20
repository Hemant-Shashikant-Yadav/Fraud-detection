# Code Walkthrough: `experiments/compare_with_prior_work.py`

**Purpose:** Computes every number in `reports/COMPARISON_WITH_PRIOR_WORK.md`. It compares MG-ECFM, the three prior-work detectors and BP1–BP4 on the same 28,259 PaySim test transactions.

**Inputs:** `reports/drift_evaluation_predictions.csv` and `reports/prior_work_predictions.csv`.
**Output:** `reports/prior_work_comparison.json`.

---

## 1. Metrics

* **Pooled (`pooled`)**: PR-AUC, ROC-AUC, precision, recall, F1, F2, Brier, recall at 1% and 0.1% FPR, TP/FP/FN, and the share of fraud value blocked at τ.
* **Per window (`per_window`)**: PR-AUC, F1 and recall for each of the 50 windows.

## 2. Monetary policies

The costs come from `configs/project_config.json`.

* `binary_net_loss`: approve/block at the model's own τ.
* `bayes_net_loss`: parameter-free Bayes-optimal 4-action routing, `argmin_a E[L(a) | p, amount]`:
  `E[approve] = p·A`, `E[step-up] = c_step + p·β·A`, `E[hold] = c_hold`, `E[block] = (1 − p)·c_block`.
* `logged_router_net_loss`: the loss of the actions in the predictions file. For MG-ECFM these are the CS-ACRC router actions.
* `budget_router` (per budget ρ in 1%, 2%, 3%, 5%): the budgeted Bayes router applied to each model's test scores in the same way as the pipeline. The routing is window by window, labels are released one window late, and the price starts from a label-free warm start on the window-1 scores.
* `cost_curve` (the equal-friction comparison): the intervention price ν is swept over 400 values with θ = 0. No label is used in any decision. For each intervention-rate cap, the lowest net loss is reported. This is the fair way to compare loss, because a model that intervenes more always loses less.
* `cost_curve_bootstrap`: a paired window-block bootstrap (2,000 resamples) of the loss saving of MG-ECFM against each model at each cap, with a 95% CI and a one-sided p-value.

## 3. Statistical tests

* `block_bootstrap_diff`: paired window-block bootstrap (2,000 resamples of the 50 windows) of the pooled PR-AUC difference. It gives the 95% CI and a one-sided p-value.
* Wilcoxon signed-rank test, one-sided (`H1: MG-ECFM > other`), on per-window PR-AUC and F1. Windows with equal values are dropped. `holm` adjusts the 7 p-values in each metric family.

## 4. Safety checks

The script stops if any model's sorted `tx_id` values differ from the MG-ECFM values. This guarantees that all comparisons are paired.
