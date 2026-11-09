# Code Walkthrough: `src/graph/temporal_memory.py`

**Purpose:** Gives every entity (account/card) a small, **persistent state vector** that is updated with each of its transactions and **decays over time**. This is the "memory" that lets the graph model see behaviour history without storing the whole graph.

**Also contains:** `SignedLogStandardizer`, the shared leakage-safe feature scaler used by the graph and the autoencoder.

---

## 1. Idea
On a new event `e_t = (u, v, x, A, τ)` touching node `v`:

`s_v(t) = s_v(t⁻) · exp(−δ·(τ − τ_last)) + W_mem · [x ‖ A]`

* Old state fades exponentially (`δ = 0.05`), the new event adds a learned message.
* O(1) read/write per node via a hash map to a pre-allocated tensor that grows geometrically.
* **Batch causality:** events in the same step do **not** see each other (matches the Phase-1 strict-step contract), so the result is invariant to the order of transactions within a batch.
* `commit` writes the exact closed form of applying all batch events sequentially, so it equals event-by-event updating up to rounding.
* Memory is detached from autograd (TGN-style): no back-propagation through time.

## 2. Section-by-section

### 2.1 `SignedLogStandardizer` (lines 56–126)
`z = (sign(x)·log1p|x| − μ)/σ` with statistics fitted on training rows only. Heavy-tailed amounts (seven orders of magnitude in PaySim) are compressed first. Non-finite inputs become 0; zero-variance columns get σ = 1. Methods: `fit`, `transform` (handles single rows and checks the width), `fit_transform`, `state_dict` / `load_state_dict` (for persistence).

### 2.2 `TemporalMemoryConfig` (lines 132–169)
`feature_dim`, `memory_dim=64`, `decay_delta=0.05`, `message_includes_amount=True`, `max_staleness_steps=720` (state older than this is treated as unseen), `initial_capacity=1024`, `init` scheme. Validated; `from_hyperparams` reads the `temporal_memory` block.

### 2.3 `TemporalNodeMemory.__init__` (lines 183–215)
Creates `W_mem` (a bias-free linear layer) and initialises it deterministically from a seed (Xavier-uniform, orthogonal or normal). Registers three persistent buffers: `memory` (node states), `last_update` (float64 times), `n_updates` (counts). Keeps the key↔row maps and the message scaler.

### 2.4 Registry (lines 218–281)
`n_nodes`, `capacity`, `_ensure_capacity` (doubles the buffers when full — amortised O(1)), `register(keys)` (batch), `register_one` (fast path), `lookup`, `key_of`, `reset_state` (forget nodes, keep weights and scaler).

### 2.5 Messages (lines 284–318)
* `fit_normalizer`: fits the feature scaler and the log-amount mean/scale **on the training split only**.
* `build_message_input(features, amounts)`: scaled `[x ‖ A]` tensor.
* `message`: `W_mem · m`.

### 2.6 Decay and reads (lines 321–354)
`_decay_factor` = `exp(−δ·Δτ)`, zero for never-seen or too-stale nodes; `read` (decayed states, detached); `read_one` (scalar fast path).

### 2.7 Per-event states, commit, update (lines 356–414)
* `event_states`: `read(v, t) + W_mem·m` — has gradient into `W_mem`, uses the **pre-batch** state.
* `commit`: for nodes touched in a batch, finds each node's final time `T_v` (scatter-max), decays the base state to `T_v`, adds each message weighted by `exp(−δ(T_v − τ_i))`, and writes memory, last-update and counts.
* `update`: states then commit. `update_one`: O(1) single-event update used in real-time scoring.

### 2.8 Persistence (lines 417–439)
`export_state` / `load_state`: keys, memory, last-update, counts, scaler and amount statistics — this is the *online state* that must be checkpointed with the model.

## 3. Things to remember
* The decay constant and staleness cut-off are modelling choices; the paper should justify them or run a sensitivity check.
* Scalers are fitted on the training split only, which is part of the leakage-free story.
