# Code Walkthrough: `src/graph/camouflage_graphsage.py`

**Purpose:** The **relational detector** (Stage 2A): a temporal-memory GraphSAGE that produces an embedding for the sender `h_u` and the receiver `h_v` of every transaction, after pruning camouflaged edges. These embeddings are blocks of the fusion vector.

---

## 1. Pipeline per batch (docstring, lines 7–37)
1. **Temporal memory:** each endpoint gets an *event node* whose state is the decayed pre-batch memory plus the event message.
2. **Inductive neighbourhood:** historical neighbours come from a bounded recency index (O(1) insert, O(fan-out) read): the `fanouts[0]` most recent counterparties of `u`/`v`, then `fanouts[l]` of each hop-`l` node (defaults 15, 10). The current edge `u↔v` is included (known at authorisation time). No labels or future information are used as inputs.
3. **Camouflage filter:** each edge scored by `S(u,w)` and pruned (see `similarity_filter.py`).
4. **GraphSAGE** on the filtered neighbourhood: `h_u^(l) = ReLU(W · CONCAT(h_u^(l−1), AGG{h_w^(l−1)}))` with `h^(0) = s(t)`.
5. **Commit:** after scoring, memory and the neighbour index are updated so later transactions see this one; in stream mode commits happen at **step boundaries**.

Training (`fit_stream`) is chronological, one pass per epoch with a **fresh memory**: BCE on `[h_u ‖ h_v]` plus the label-aware similarity loss. Labels are targets only, never inputs.

## 2. Section-by-section

### 2.1 Optional PyTorch Geometric (lines 62–68, 74–87)
If PyG is installed, a `MessagePassing` operator does the aggregation; otherwise the pure-PyTorch fallback is used (numerically equivalent).

### 2.2 `scatter_aggregate` (lines 90–108)
Mean/sum via `index_add`, max via `scatter_reduce`; empty neighbourhoods give zeros.

### 2.3 `FilteredSAGEConv` (lines 111–135)
One SAGE layer: aggregate neighbour states, concatenate with the node's own state, linear layer, ReLU (optional on the last layer).

### 2.4 `TemporalNeighborIndex` (lines 141–188)
Per-node bounded deque of recent counterparties. `insert(a, b)` stores the undirected interaction in both deques (O(1)); `neighbors(node, k)` returns up to `k` **distinct** most-recent counterparties, newest first.

### 2.5 `GraphSAGEConfig` (lines 194–264)
`feature_dim`, `memory_dim=64`, `hidden_dim=128`, `out_dim=64`, `num_layers=2`, `fanouts=(15, 10)`, aggregator, dropout, LR, `epochs=20`, `max_batch_events=1024`, `sim_loss_weight`, `max_pos_weight=100`, latency budget 25 ms, plus nested memory and similarity configs (validated to agree on `memory_dim` and `feature_dim`). `from_hyperparams` builds all three from `hyperparams.json`.

### 2.6 `SubgraphBatch` (lines 267–278)
Container for one batch's computation graph: node states (`2n + H` rows), edges (neighbour → centre), global ids, messages and times.

### 2.7 `CamouflageResistantGraphSAGE` (lines 284–612)
* `__init__`: builds memory, similarity filter, conv stack (dims `memory → hidden … → out`), a linear head on `[h_u ‖ h_v]`, dropout, and the neighbour index; runs under `fork_rng` with a fixed seed.
* `reset_state`, `fit_normalizer`.
* `_register_events`: maps entity keys to rows; missing destinations become `−1`.
* `build_subgraph` (337–405): creates event nodes for sources and destinations, adds edges to historical neighbours (hop 0), the mutual `u↔v` edge, then expands multi-hop neighbours with the per-hop fan-out; computes event states (with gradient into `W_mem`) and decayed historical states; returns a `SubgraphBatch`.
* `propagate` (408–418): camouflage filtering then the conv stack (dropout between layers).
* `forward_subgraph` (420–431): embeddings for the batch's sources/destinations, fraud logit from the head, and filter statistics (edges, kept, kept fraction).
* `commit` / `commit_step` (433–468): write memory and neighbour index after scoring; `commit_step` inserts edges **sorted by (src key, dst key)** so embeddings are invariant to arrival order within a step; frauds are recorded as observed node labels.
* `_similarity_loss` (471–481): label-aware loss on the batch's edges using observed node labels.
* `fit_stream` (483–544): chronological training; each epoch resets state; within a step all mini-batches read the pre-step state and commits happen at the end of the step; loss = weighted BCE (positive weight capped at 100) + `λ·similarity loss`; Adam with weight decay; logs per-epoch history.
* `transform_stream` (547–583): inference over a stream (no gradients), returns `h_src`, `h_dst`, `gnn_proba`, commits each step; labels, if passed, are only recorded for later fine-tuning.
* `embed_event` (586–612): the **single-transaction authorisation fast path** (target < 25 ms), returns embeddings, GNN probability, latency and edge counts.

### 2.8 Helpers (lines 618–637)
`_is_missing` (None/NaN/empty string), `_check_sorted`, `_step_bounds` (index ranges of equal steps).

## 3. Things to remember
* Strict step causality is preserved: events in one step never see each other.
* Training replays the stream from scratch each epoch, which is why the GraphSAGE fit is the most time-consuming part of MG-ECFM.
