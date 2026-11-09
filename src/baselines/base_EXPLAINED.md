# Code Walkthrough: `src/baselines/base.py`

**Purpose:** The shared contract and utilities for all four baselines. Anything that must be *identical* across BP1–BP4 for a fair comparison with MG-ECFM lives here.

---

## 1. Why a shared base class
If each baseline did its own scaling, validation and early stopping, differences in the results could come from plumbing instead of the models. The base class fixes: seeding, input checks, heavy-tail scaling, chronological hold-out, cost weights and the latency harness.

Public interface every baseline gets: `fit(X, y)`, `predict_proba(X)` (N×2), `predict(X)`, `measure_latency(X)`.

## 2. Section-by-section

### 2.1 Constants (lines 43–44)
`SEED = 42` (project-wide seed), `EPS = 1e-12` (numerical floor).

### 2.2 Reproducibility: `seed_everything` (lines 50–59)
Seeds Python `random`, NumPy, PyTorch (+ all CUDA devices), sets `PYTHONHASHSEED`, and asks cuDNN for deterministic kernels (`deterministic=True, benchmark=False`).

### 2.3 `config_from_mapping(cls, mapping)` (lines 62–76)
Builds a config dataclass from a dict. **Unknown keys raise `KeyError`** (line 74–75), so a typo like `learnng_rate` fails immediately instead of being ignored.

### 2.4 Input validation (lines 82–112)
* `check_X`: converts to float64, reshapes 1-D to one row, rejects non-2-D, empty, NaN/inf, and wrong feature count; returns a contiguous array.
* `check_y`: requires binary {0,1}, matching length, and **both classes present** (a fraud model cannot train on one class).

### 2.5 `SignedLogStandardScaler` (lines 115–143)
Transform `z = clip((sign(x)·log1p|x| − μ)/σ, −10, 10)`.
* Signed log compresses amounts that span 6+ orders of magnitude but keeps the sign (balance deltas can be negative).
* `fit` learns μ, σ on **training rows only** (no look-ahead); constant columns get σ = 1.
* Output is float32 for PyTorch.

### 2.6 Time-aware helpers (lines 146–183)
* `chronological_holdout(n, fraction)`: returns head/tail indices where the tail is the **last** `fraction` of rows. Rows are time-ordered, so no future row ever trains a model scored on an earlier one. Keeps at least `min_rows` in the head.
* `stratified_tail_ok`: a hold-out is usable for early stopping only if it contains both classes.
* `cost_sensitive_class_weights(y, fn_cost, fp_cost, balance)`: weights `[w0, w1]` = misclassification cost × inverse-frequency balance, then normalised so the mean per-sample weight is 1 (keeps the learning rate comparable).

### 2.7 Small utilities (lines 186–197)
* `two_column_proba(p)`: `[1−p, p]` with clipping to guard round-off.
* `iterate_minibatches(n, batch_size, rng)`: yields shuffled index slices.

### 2.8 `BaselineModel` (abstract class, lines 203–321)
* `__init__` stores seed, config, default `threshold_ = 0.5`, `is_fitted_`, `fit_report_`, and picks the device via `resolve_device()`.
* `fit` (225–238): validate → seed → call subclass `_fit` → record `fit_seconds`, `n_train`, `n_features`, `fraud_rate` in `fit_report_`.
* `predict_proba` (240–245): checks fitted + width, calls subclass `_predict_fraud_proba`, returns 2 columns.
* `predict` (247–250): `proba[:,1] >= threshold_`. The threshold is set later by the experiment on the operating split (never on test data).
* `measure_latency` (252–289): warm-up calls, then times `n_runs` calls on cyclic rows; returns p50/p95/p99/mean/max in ms.
* `get_config`, `count_parameters` (sums trainable parameters of any `nn.Module` attributes).
* Abstract hooks `_fit`, `_predict_fraud_proba` – the only things a subclass must implement.
* Helpers `_check_fitted`, `_rng(offset)`, `_to_tensor`.

### 2.9 `EarlyStopper` (lines 324–347)
Tracks the best validation loss (improvement threshold 1e-6), snapshots the best `state_dict`, and `step()` returns `True` when `patience` epochs pass without improvement. `restore()` loads the best weights back.

### 2.10 `KNNNeighborIndex` (lines 353–393)
Euclidean k-NN graph over the training set, used by the graph baselines (BP2, BP3).
* `fit`: caps `k` at `n_ref−1`, fits scikit-learn `NearestNeighbors`.
* `train_neighbors`: each training node's k nearest **other** nodes; the block at 383–389 removes the node itself even when duplicate rows hide it (drops the farthest instead).
* `query`: links an unseen transaction to its k nearest training nodes, enabling *inductive* scoring (no retraining for a new transaction).
