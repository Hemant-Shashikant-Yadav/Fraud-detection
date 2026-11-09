# Code Walkthrough: `src/orchestrator/service.py`

**Purpose:** The **live MG-ECFM service**. It wraps one fitted pipeline (fusion scorer + martingale knowledge gate + CS-ACRC router) together with the state needed to *use* it: score a new transaction, explain the decision, draw the camouflage graph, show the knowledge base, run post-incident recovery, and save/load everything. The CLI (`main.py`) and the Streamlit dashboard (`app/app.py`) are thin layers on top of this class.

---

## 1. Key design points (docstring, lines 5–24)
* Keeps a **causal feature engineer** replayed over the whole history, so a new transaction is featurised exactly as in the stream (only earlier steps contribute).
* Keeps the **history** of scored transactions with their decisions (used by the batch scorer, the mule-network view and recovery) and a **TD-TPF tracker**.
* Scoring one transaction is a **what-if** evaluation: engineer, graph memory and router are **not advanced** (`commit=False`), so repeated queries are reproducible and never contaminate live state.
* Persistence: pickling, with the in-memory SQLite knowledge store serialised separately.

## 2. Section-by-section

### 2.1 Constants and `utc_clock` (lines 62–72)
Action labels, PaySim→canonical column renames, label aliases, a format version, and `REPLAY_CHUNK_ROWS = 250 000` (memory-bounded replay chunk). `utc_clock` replaces a lambda clock so the store stays picklable.

### 2.2 `ServiceConfig` (lines 78–118)
`source`, `profile` (`demo` / `fast` / `full`), synthetic-stream settings, `n_windows`, `poison_rate`, `label_delay_windows`, `max_rows`, `seed`. `for_profile` gives presets (the demo is a small synthetic stream; fast/full use 400 steps and 50 windows). `profile_hyperparams` returns the published hyper-parameters (`full`) or reduced ones (`fast`/`demo`: fewer trees/epochs).

### 2.3 `MGECFMService.__init__` (lines 138–180)
Stores the system, split summary and canonical stream; replays the causal engineer **in chunks** (only its end state is needed — a memory optimisation); checks the feature schema; records the last step. Builds **explanation references from the training split only**: mean/std of signed-log features (computed in one buffer) and the sorted autoencoder residual distribution; collects verified fraud accounts.

### 2.4 Building (lines 183–278)
* `_feature_config`: from the project config for PaySim/IEEE-CIS.
* `build(config)`: seeds, loads configs, selects the data splits (**`prepare_splits`**, so the no-silent-synthetic guard applies), builds the candidate-rule feed, fits `MGECFMSystem`, **replays the test windows prequentially** and records metrics in a build report.
* `from_evaluated`: wraps a system already run by the drift evaluator.
* `_replay_test`, `_history_metrics`, `_history_frame` (joins decisions with accounts/types).
* `_labelled_accounts`: accounts with **verified** fraud, using only labels released before the last window (respecting label delay): fraud destinations become "mule", sources "victim".

### 2.5 Featurisation (lines 281–318)
`_canonical_rows` builds a one-row canonical frame (`tx_id` negative, label −1). `featurize` uses `engineer.restricted_copy([src, dst])` (exact and O(1) in the number of accounts — Atharv's optimisation) and returns the causal feature event plus timing. The step must be after the last finalised step.

### 2.6 Scoring and explanation (lines 321–416)
* `_fired_rules`: active rules whose conditions hold for the event.
* `score_transaction`: featurise → `score_event(commit=False)` → fired rules → route with a **shallow router copy** (shares live thresholds but not counters) → expected loss per action. Returns probability, risk score, action and label, reason, thresholds, timings, fired rules and flags for known mule/victim accounts; optionally the explanation.
* `_explain`: **TreeSHAP** attributions (XGBoost `pred_contribs`) on the fusion vector, grouped into readable components (sender/receiver embedding, AE residual, KB triggers, each tabular feature), plus **feature deviations** (z-score versus training) and the autoencoder-residual percentile.
* `latency_benchmark`: p50/p95/p99/max authorisation latency over resampled history, with featurisation timed separately.

### 2.7 Batch scoring (lines 438–519)
* `canonicalize_upload`: strips leakage columns (reports which), renames PaySim columns, validates amount/step, extracts labels from several aliases.
* `score_frame`: orders rows chronologically, shifts steps to directly follow the history, computes features **causally within the batch** on a restricted engineer copy, scores with a deep-copied classifier and router (live state untouched), and returns the scored table plus a summary (action counts, ms/transaction, and — if labels exist — metrics and the share of frauds intercepted).

### 2.8 Camouflage graph (lines 522–590)
`explain_graph`: rebuilds the two-hop neighbourhood of a hypothetical transaction on a deep-copied graph, applies the similarity filter, and returns **nodes** (role, hop, flags) and **edges** (similarity, kept/pruned) plus statistics — the data behind the dashboard's camouflage view.

### 2.9 Examples and knowledge base (lines 593–662)
`example_transactions` (a mule-network fraud, a routine merchant payment, a never-seen account), `kb_table` (each candidate rule with status, Subjective-Logic opinion, live e-process wealth and counters), `kb_summary` (with store integrity), `kb_snapshots`.

### 2.10 Recovery (lines 665–708)
Lazily creates the TD-TPF tracker and recovery simulator. `recovery_candidates` lists frauds that were not intercepted and can be traced; `recovery_trace` simulates freeze/block/chargeback for a fraud detected after a given delay and returns the trace, actions and a recovery-vs-delay curve.

### 2.11 Persistence (lines 711–762)
`save` serialises the SQLite store to bytes, temporarily detaches un-picklable members (connection, lock, clock, tracker), writes to a temp file and **atomically replaces** the target (a failed dump never leaves a truncated model). `load` unpickles, restores the store with `deserialize`, and re-attaches the clock. `describe` summarises the service.

## 3. Things to remember
* `models/*.pkl` files come from `save`; they include the online state (thresholds, wealth, memory), not just weights.
* The service is only as good as its build: results in the paper must come from `--source paysim`, not the synthetic demo.
