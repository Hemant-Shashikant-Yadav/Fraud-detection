# Code Walkthrough: `src/detection/tabular_engine.py`

**Purpose:** The supervised workhorse: a **cost-sensitive XGBoost** classifier whose raw scores are turned into **calibrated probabilities** (isotonic or Venn-Abers), plus the calibration metrics.

---

## 1. Key ideas
* **Cost sensitivity:** `scale_pos_weight="auto"` becomes `c_FN/c_FP` (25) when cost-sensitive learning is on, otherwise `n_neg/n_pos`. With `amount_weighted`, each fraud row also gets weight `A / mean(A_fraud)` clipped to `[0.25, 10]`, so large frauds dominate the gradient, mirroring the monetary loss.
* **Calibration on a later split:** margins are mapped to `p̂ ∈ [0,1]` on the chronologically later calibration split, never on the training data.
* **Venn-Abers:** gives an *interval* `(p0, p1)`; its width `Δp̂ = p1 − p0` is the epistemic uncertainty used by the router.

## 2. Section-by-section

### 2.1 Imports and `_require_xgboost` (lines 35–59)
XGBoost is imported inside `try/except` so the calibrators stay usable without it; the engine raises a clear `ImportError` otherwise.

### 2.2 `IsotonicCalibrator` (lines 65–88)
Pool-Adjacent-Violators regression (scikit-learn), increasing, clipped to the observed range. `fit` validates alignment; `predict` clips to [0, 1].

### 2.3 `VennAbersCalibrator` (lines 91–192)
Inductive Venn-Abers predictor with **pre-computed rank tables** (O(1) lookup at inference).
* `_iso_value`: fit isotonic regression on `calibration ∪ {(s, y_new)}` and read the value at `s`.
* `fit`: if there are more than `max_unique_scores` (400) distinct scores, quantise them into quantile bins. Then compute `(p0, p1)` for every *tie* (score equal to a calibration point) and every *gap* (score between points, below the first, above the last) by adding the hypothetical label 0 and 1.
* `predict_interval`: searches the rank position, uses the tie table or the gap table, clips to [0,1].
* `predict`: merged point estimate `p = p1 / (1 − p0 + p1)`.

### 2.4 `make_calibrator` (lines 195–200)
Factory: `"isotonic"` or `"venn_abers"` (aliases accepted).

### 2.5 Metrics (lines 206–222)
`brier_score` and `expected_calibration_error` (equal-width bins, default 15).

### 2.6 `TabularEngineConfig` (lines 228–282)
XGBoost parameters (`hist`, depth 6, LR 0.05, `aucpr` eval metric, ...), `n_estimators=600`, `early_stopping_rounds=50`, cost settings (`fn_weight=25`, `fp_weight=1`, `amount_weighted`), calibration method, `val_fraction=0.1`.
`from_hyperparams` merges the `xgboost` block of `hyperparams.json` and auto-selects the device with `resolve_xgb_device()`.

### 2.7 `CalibratedXGBoostEngine` (lines 285–444)
* `resolve_scale_pos_weight` / `sample_weights`: the cost logic above.
* `_params`: finalises XGBoost parameters (seed, thread count, silent).
* `fit` (337–378): validates data (both classes required); if no explicit validation set, takes the **last 10%** of training rows for early stopping (only if both parts contain fraud); trains with `xgb.train`; records the best iteration; switches to `inference_nthread` threads for fast single-row scoring; then calls `calibrate`.
* `calibrate` (380–392): computes margins on the calibration split and fits the chosen calibrator.
* `predict_margin` (395–410): raw log-odds using trees up to the best iteration; uses `inplace_predict` (fast), falling back to a `DMatrix`.
* `predict_uncalibrated`, `predict_proba` (calibrated, clipped), `predict_interval` (Venn-Abers only), `predict_one`.
* `feature_importance`: gain importances mapped back to feature names (`f12` → real name).

## 3. Things to remember
* Because training is cost-weighted, the raw sigmoid is **not** a probability; only the calibrated output is.
* The default calibrator is isotonic; Venn-Abers is only used if configured.
* In our PaySim run the Brier score of MG-ECFM (0.111) was worse than BP1 (0.062) — calibration quality is something to investigate before the paper.
