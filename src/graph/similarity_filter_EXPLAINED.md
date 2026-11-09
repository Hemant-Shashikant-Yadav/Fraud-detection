# Code Walkthrough: `src/graph/similarity_filter.py`

**Purpose:** The **camouflage filter**. Fraudsters hide by transacting with many benign accounts, so a naive GNN averages fraud signals away. This module learns, from labels, which edges connect *similar* nodes and prunes the rest **before** GraphSAGE message passing.

---

## 1. Idea
For centre node `u` and neighbour `w` with temporal states `s_u`, `s_w`:

`S(u, w) = sigmoid( W_sim · |s_u − s_w| )` (L1, as in CARE-GNN)

Trained **label-aware**: an edge is a positive (target 1) if both endpoints share a class and a negative (target 0) if it crosses the fraud/legit boundary — exactly the camouflage edge a mule creates. Class-balanced weights stop the huge number of benign–benign edges from drowning the signal.

Two pruning modes build the filtered neighbourhood `Ñ(u)`:
* `threshold` – keep `w` iff `S(u, w) ≥ p`;
* `top_p` – keep the `ceil(p·|N(u)|)` most similar neighbours.
`min_neighbors_kept` keeps the best few neighbours of a node whose whole neighbourhood would be pruned.

## 2. Section-by-section

### 2.1 `SimilarityFilterConfig` (lines 43–80)
`memory_dim`, optional `hidden_dim` (0 = linear scorer, > 0 = one-hidden-layer MLP), `threshold_p=0.5`, `mode`, `min_neighbors_kept`, `learning_rate`, `distance` (`l1`/`l2`). Validated in `__post_init__`; `from_hyperparams` reads the `similarity_filter` block.

### 2.2 `FilterResult` (lines 83–89)
NamedTuple: filtered `edge_index` (row 0 = neighbour, row 1 = centre), similarity of **all** input edges, the keep mask, and the kept similarities.

### 2.3 `CamouflageSimilarityFilter.__init__` (lines 92–113)
Builds a linear or MLP scorer with a seeded generator and Xavier-uniform initialisation (bias zero), so results are reproducible.

### 2.4 Scoring (lines 116–131)
`distance` (|Δ| or Δ²), `logits`, `forward` (sigmoid similarity for aligned rows), and `edge_similarity(states, edge_index)` for a whole edge list.

### 2.5 Pruning (lines 134–182)
* `rank_within_centre`: 0-based rank of every edge among those sharing its centre, by descending score — implemented with **two stable sorts** and a cumulative-max trick, no Python loop (O(E log E)).
* `keep_mask`: `threshold` mode compares to `p`; `top_p` mode keeps ranks below `ceil(degree·p)`; then the `min_neighbors_kept` override.
* `filter_edges`: scores all edges, builds the mask on **detached** scores (pruning is not differentiable) and returns the filtered edges plus differentiable similarities; handles empty edge sets.

### 2.6 Training (lines 185–230)
* `label_aware_targets`: same-class indicator.
* `label_aware_loss`: class-balanced binary cross-entropy (each class gets half of the total weight; if only one class exists, uniform weights).
* `fit`: full-batch Adam on a labelled graph snapshot; returns initial and final loss.

### 2.7 `camouflage_report` (lines 232–248)
Diagnostics split by homophilic vs. cross-class edges: share of camouflage edges **pruned**, share of homophilic edges **kept**, overall kept fraction. This is the evidence that the filter does what it claims.

### 2.8 `numpy_states` (lines 251–253)
Float32 tensor view of a NumPy state matrix.

## 3. Things to remember
* The filter is trained using labels, so in the streaming system it must only be trained with **released** (delayed) labels to avoid leakage.
* Pruning decisions use detached scores; gradients flow through the similarity of kept edges only.
