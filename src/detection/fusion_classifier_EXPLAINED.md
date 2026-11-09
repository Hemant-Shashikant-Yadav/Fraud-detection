# Code Walkthrough: `src/detection/fusion_classifier.py`

**Purpose:** The **Stage-1A fusion scorer** (component C3). It stacks four kinds of evidence into one vector per transaction and turns it into a calibrated fraud probability `p̂`:

`z_t = [ x_clean ‖ h_u ‖ h_v ‖ r_AE ‖ φ_KB ]`

| Block | Meaning | Source |
|---|---|---|
| `x_clean` | leakage-free causal tabular features | Phase 1 |
| `h_u, h_v` | sender / receiver graph embeddings | camouflage-filtered GraphSAGE |
| `r_AE` | autoencoder reconstruction residual | `autoencoder_residual.py` |
| `φ_KB` | summary of knowledge-base rules that fire (5 numbers) | the gate's active rules |

---

## 1. Section-by-section

### 1.1 Docstring and imports (lines 1–55)
States the fusion vector and the three **ablation switches** (`use_relational_embeddings`, `use_ae_residual`, `use_kb_triggers`) read from `hyperparams.json['fusion']`. Training is strictly chronological: `train < validation < calibration`.
`_SCALAR_OPS` maps operator strings (`">"`, `"<="`, ...) to Python functions for the fast rule evaluator.

### 1.2 `load_hyperparams` (lines 58–61)
Reads `configs/hyperparams.json`.

### 1.3 `KnowledgeBaseTriggerEncoder` (lines 67–162)
Encodes "which active rules fired on this event" into a **fixed width** (so the XGBoost input never changes shape when rules are admitted or rolled back).
* `FEATURE_NAMES`: `kb_hit_count`, `kb_any_hit`, `kb_weighted_score`, `kb_max_weight`, `kb_mean_weight`.
* `set_rules(rules, weights)`: installs the current active-rule snapshot; each rule's weight comes from the supplied mapping, else `rule.weight`, else `rule.historical_precision`, else `1.0`.
* `_summarise(hits)`: turns an `n × R` boolean hit matrix into the `n × 5` block (count, any-hit flag, weighted sum, max weight, mean weight of hit rules).
* `rule_hit_matrix` / `transform_frame`: vectorised evaluation on a DataFrame.
* `_eval_condition` / `transform_event`: **single-event fast path** — if a rule exposes `(feature, op, value)` conditions they are evaluated directly on the dict, avoiding DataFrame construction.

### 1.4 `FusionColumns` (lines 168–176)
Names of the `src`, `dst`, `amount`, `step`, `label` columns.

### 1.5 `StackedFusionClassifier.__init__` (lines 182–219)
* Refuses any feature named in `FORBIDDEN_FEATURE_COLUMNS` or `"label"` (raises `LeakageError`).
* Loads hyper-parameters, resolves the three ablation switches (argument overrides config).
* Builds GraphSAGE, autoencoder, XGBoost engine (unless injected) and the KB encoder.

### 1.6 Schema helpers (lines 222–261)
* `fusion_feature_names`: the full ordered column list for the current switches.
* `assemble(...)`: writes all blocks into **one preallocated matrix** (float32 for batch scoring — XGBoost casts to float32 anyway, so scores are bit-identical while memory is halved; this was the memory optimisation in Atharv's commit).

### 1.7 Frame helpers (lines 264–311)
`_x` (feature matrix, with missing-column check), `_graph_args` (feature matrix, amounts, steps, src/dst lists), `_assert_chronological` (each split sorted and strictly after the previous), and `_components(frame)` which runs the graph replay, the AE residual and the KB encoder and assembles `z`.

### 1.8 `fit` (lines 314–362)
1. Check the chronological order of train/validation/calibration.
2. Install rules (optional).
3. Build the training feature arguments **once** and reuse them.
4. Fit the autoencoder (legitimate rows) → fit the graph normaliser and GraphSAGE stream → reset graph memory.
5. Replay the training stream to get `z_train` (recording labels into node memory), then validation and calibration `z`.
6. Fit the cost-sensitive XGBoost on `z_train` and calibrate on `z_cal`.
7. Store a report (history, tabular details, feature count, seconds).

### 1.9 Prediction (lines 365–414)
* `transform` / `predict_proba`: batch scoring of a later chronological frame.
* `score_event(event, commit=True)`: **real-time path** for one transaction with per-component timings (`graph_ms`, `ae_ms`, `kb_ms`, `tabular_ms`); returns calibrated `p_hat`, raw margin, GNN probability and latency. `commit=False` scores without updating node memory.
* `latency_profile(events)`: warm-up, then P50/P95/P99/max latency.

### 1.10 `frame_to_events` (lines 417–419)
DataFrame → list of row dicts for the real-time path.

## 2. Things to remember
* Graph memory is carried forward across splits, so the order of `train → validation → calibration` matters.
* The KB block has constant width, which is what lets the gate change rules without retraining the stacker.
