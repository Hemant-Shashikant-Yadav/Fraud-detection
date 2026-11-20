# Code Walkthrough: `experiments/run_prior_work_comparison.py`

**Purpose:** Rebuilds the detectors of the earlier Evidence-Gated notebook project and scores them on the **same** PaySim stream, splits and 50 test windows as the Phase-7 drift evaluation. The output lets `compare_with_prior_work.py` compare all models on identical test rows.

**Inputs:** the raw PaySim file (through `prepare_splits("paysim", sample_rows=2_000_000, sample_seed=42)`).
**Outputs:** `reports/prior_work_predictions.csv` (one row per model × test transaction) and `reports/prior_work_fit.json` (features, calibration, threshold, latency).

---

## 1. Models

| Key | Prior-work component | Features | Calibration |
|:---|:---|:---|:---|
| `prior_b1_raw` | Day 6 B1 XGBoost | `step`, `amount`, 5 one-hot types | none |
| `prior_b1_cal` | Day 7 B1 + calibration | the same 7 | Platt on the calibration split |
| `prior_b1_graph` | Day 10 B1 + graph counts | 7 + 6 graph counts | Platt on the calibration split |

All models use `XGBClassifier(scale_pos_weight = n_neg / n_pos, random_state = 42)`. The threshold is F1-optimal on the operating split (`select_operating_threshold`). No test label is used.

## 2. Section-by-section

### 2.1 Graph counting features (`_count_before`, `_recency`, `add_prior_features`)
* Edges are TRANSFER and CASH_OUT rows only.
* `_count_before(key)`: inclusive cumulative edge count per key minus the inclusive count in the same step. The result counts edges at **strictly earlier** steps only, so rows in one step cannot see each other.
* `_recency(key)`: `merge_asof(..., allow_exact_matches=False)` finds the last earlier step with an edge for the key. If there is no history, the value is `9999`.
* `src_dst_pair_seen_before = pair_repeat_before > 0`.

### 2.2 `fit_xgb`
Fits the cost-weighted XGBoost on the train split.

### 2.3 `main`
1. Builds the splits and joins them in time order (the code stops if the steps are not sorted).
2. Adds the prior-work features and maps each test row to its window.
3. Fits each model once, adds Platt calibration where set, selects τ on the operating split and scores the test split.
4. Writes the predictions and the fit report.

## 3. Check against the prior-work report

| Quantity | Report (full test set) | Reproduction (sampled test set) |
|:---|:---:|:---:|
| Raw threshold | 0.9944 | 0.9907 |
| Calibrated threshold | 0.0773 | 0.0475 |
| PR-AUC | 0.3277 | 0.3057 |
| Calibrated recall | 0.2899 | 0.2871 |
