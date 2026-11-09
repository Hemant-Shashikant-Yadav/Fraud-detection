# Code Walkthrough: `experiments/run_drift_eval.py`

**Purpose:** The **main benchmark**. It trains MG-ECFM and the four baselines (BP1–BP4) on the same chronological data and evaluates all of them on **50 consecutive monitoring windows** (the Menezes & Filho 2025 drift protocol). Its outputs feed every table and figure of the paper, and it also contains the shared building blocks (data splits, metrics, `MGECFMSystem`) used by the ablation, stress test, service and CLI.

**Outputs:** `reports/drift_evaluation_results.csv` (one row per model × window), `drift_evaluation_predictions.csv` (one row per model × test transaction; ~250 MB on PaySim 2M, git-ignored) and `drift_evaluation_manifest.json` (configuration, splits, fit reports, hashes).

---

## 1. Protocol (docstring)
1. A leakage-free causal feature stream is split strictly by time into `train < calibration < operating < test`.
2. Every model is trained once on the past; its binary threshold `τ` is chosen on the **operating** split only (F1-optimal) — never on test data.
3. The test period is cut into **50 step-aligned windows** of roughly equal size.
4. Windows are processed sequentially (*prequential*, "test-then-learn"). Labels of window `j` reach MG-ECFM's online parts (CS-ACRC thresholds, martingale gate) only before window `j + 1 + label_delay` is scored. **Baselines are frozen after training.**
5. Per window and model: PR-AUC (primary), ROC-AUC, precision, recall, F1, F2, Brier, ECE and **Expected Monetary Loss (EML)** under the asymmetric cost matrix.
EML is the mean realised loss per transaction under each model's *own* policy (four-action routing for MG-ECFM, approve/block at `τ` for the baselines); `eml_binary_policy` applies the same binary policy to every model for a like-for-like comparison.

**Data sources:** `auto` (default) resolves to `paysim` and **raises if the raw file is missing** (the guard added this session); `synthetic` must be requested explicitly; `paysim` / `ieee_cis` stream the real files.

## 2. Section-by-section (by outline)

### 2.1 Configuration helpers (lines 128–178)
Small CPU-friendly baseline configs; `load_json`, `load_configs` (hyperparameters + project config), `fast_hyperparams` (reduced capacity), `profile_settings` (`fast` vs `full`).

### 2.2 Data: streams and splits (lines 180–450)
* `EvaluationWindow`, `StreamSplits` (train/calibration/operating/test frames, feature names, windows, source label, canonical stream) with `window_frame` and `summary`.
* `raw_paysim_to_canonical`, `inject_camouflage_drift` (synthetic drift: fraud amounts migrate toward legitimate ones), `build_synthetic_stream`, `build_real_stream` (chunked loader → causal feature engineer).
* `raw_dataset_available` / **`resolve_source`** (no silent synthetic fallback) / `prepare_splits`.
* `partition_windows(steps, n_windows)`: windows never split a step, so a timestamp batch is scored in one window; raises if there are fewer distinct steps than windows.
* `chronological_splits`: splits by **distinct steps** using `temporal_splits` from the project config (or fractions 0.45/0.10/0.10); uses contiguous `iloc` slices (memory-friendly); requires both classes in train and calibration. *Note: on PaySim 2M the configured steps produced an operating split of 186 rows that are all fraud — worth fixing before final results.*
* `candidate_feed`: the contaminated candidate-rule feed (poison rate 10 %) used by the gate.

### 2.3 Metrics (lines 453–522)
`f_beta`, `expected_calibration_error`, `classification_metrics` (PR-AUC, ROC-AUC, precision/recall/F1/F2, Brier, ECE), `select_operating_threshold` (F-score-optimal), `binary_policy_actions`, `make_loss`.

### 2.4 Systems under evaluation (lines 526–860)
* `WindowOutput`: scores, hard decisions, actions, threshold, latency, extras (active rules, λ).
* `PlattCalibrator`: static sigmoid calibration (used by the XGBoost-only ablation).
* **`MGECFMSystem`** (561–775): the full online pipeline, parameterised for the ablations — `gate_mode` (`martingale` / `auto_admit` / `none`), `crc_mode` (`adaptive` / `static`), `routing` (`crc` / `binary`), calibration type, label delay. Key methods:
  * `fit`: submit candidate rules to the gate and replay the training history through it, fit the fusion classifier, replay calibration, then process the **operating split block by block** (score → route → feedback) so the controller and gate enter the test period warmed up, and select `τ` there.
  * `_route` / `_feedback`: four-action routing with rule hits (or binary), CS-ACRC update from labels, gate cycle and encoder sync.
  * `release_feedback` / `score_window`: release due labels, score the window with current λ, queue its feedback for `window + 1 + delay`.
  * `active_rules`, `rule_weights`, `_gate_cycle`, `_replay_history`.
* `BaselineSystem` (778–828): wraps a frozen BP1–BP4 model, trains on train+calibration features, scores windows.
* `clone_system` (830): deep clone so ablation arms can branch from one fitted pipeline.

### 2.5 Evaluator (lines 864–1030)
`DriftEvaluator` (`fit_all`, `window_record`, `evaluate`), `summarise`, `build_systems`, and **`run_drift_evaluation`**, which prepares splits, builds all systems, fits, evaluates the 50 windows, and writes the CSVs/manifest. `_jsonable` converts NumPy types; `main` is the CLI (`--source`, `--profile`, `--n-windows`, `--models`, `--poison-rate`, `--label-delay-windows`, `--max-rows`, ...).

## 3. Things to remember
* This file defines the **evaluation fairness**: same splits, same windows, thresholds chosen on operating data only.
* The PaySim-2M drift evaluation recorded `runtime_seconds ≈ 5137` (~86 min); the deck records ~8 774 s for the whole `full-pipeline` run on an Apple-Silicon Mac (CPU, ~8 GB peak RSS). Time is dominated by baseline training (BP3 ≈ 1 426 s, BP4 ≈ 1 005 s, MG-ECFM ≈ 73 s).
