# Code Walkthrough: `src/baselines/care_gnn.py`

**Purpose:** Baseline **BP3**, a re-implementation of CARE-GNN (Dou et al., CIKM 2020): a graph neural network that resists *camouflaged* fraudsters by filtering neighbours with a learned, label-aware similarity and a reinforcement-learning threshold.

**Adaptation made here:** the original needs a pre-built multi-relation graph. This version builds each relation as an *inductive k-NN graph* over one **view** (column group) of the features, so a brand-new transaction can be attached to the graph at scoring time.

---

## 1. Section-by-section

### 1.1 Docstring (lines 1–34)
Lists the paper components and equations being reproduced: multi-relation graph, label-aware similarity (Eq. 2–3), top-p neighbour selector, RL-adapted threshold (Eq. 4–5), relation-aware aggregation (Eq. 6–7) and the combined loss (Eq. 8).

### 1.2 `CAREGNNConfig` (lines 56–85)
Hyper-parameters: `n_relations=3` (or explicit `relation_views` column groups), `k_neighbors=10`, `embed_dim=64`, `n_layers=1`, `epochs=31`, `batch_size=1024`, `learning_rate=0.01`, `lambda_1=2.0` (similarity-loss weight), `lambda_2=1e-3` (L2), initial keep-ratio `p_init=0.5`, RL step `rl_step_tau=0.02`, RL window/stop rule, `undersample_ratio=1.0` (balanced epochs). `__post_init__` validates the ranges.

### 1.3 `RelationThreshold` (lines 91–115) — the RL controller
One per (layer, relation). `p` is the fraction of neighbours kept.
`update(avg_distance)` each epoch:
* record history; ignore if converged or distance is not finite;
* **reward** `+1` if the average neighbour distance did not increase (`prev − current ≥ 0`), else `−1`;
* `p ← clip(p + τ·reward, τ, 1)`;
* **converged** when the last `window` (=10) rewards sum to within ±2 in absolute value (the controller is oscillating around an optimum).

### 1.5 `CARELayer` (lines 121–155)
* `sim_mlp`: linear scorer predicting the label from an embedding; its soft-max fraud probability is the node's "similarity score".
* `intra[r]`: one linear map per relation; `inter`: combines the node with the weighted relation summaries.
* `fraud_score(h)` returns logits and the fraud probability.
* `forward`: for every relation:
  1. L1 distance between the centre node's score and each neighbour's (Eq. 3), computed on detached scores;
  2. keep the `ceil(p·k)` closest neighbours (**top-p selector**);
  3. mean-aggregate kept neighbours, apply relation-specific linear + ReLU;
  4. add `p_r · h_r` (CARE-Weight: relations weighted by their thresholds).
  Then `h = ReLU(inter([h_center ‖ combined]))`. Returns embedding, centre logits (for the similarity loss) and the mean kept distances (for the RL controller).

### 1.6 `CARENetwork` (lines 158–165)
`n_layers` CARE layers plus a 2-class classifier on the final embedding.

### 1.7 `CAREGNN` class (lines 171–339)
* `__init__`: scaler, view/index/table containers, threshold controllers, cached reference tensors.
* `_resolve_views` (192–201): validates explicit views, otherwise splits the columns into `n_relations` contiguous chunks with `np.array_split`.
* `_p`, `relation_thresholds`: current `p` values.
* `_embed` (212–226): **recursive** differentiable embedding of training nodes: layer 0 is the raw scaled features; for each layer it looks up each relation's neighbour table, recursively embeds the neighbours, and calls the layer. It also collects centre logits and distances.
* `_epoch_nodes` (228–233): class-balanced under-sampling — all frauds plus an equal-ish random sample of legitimate nodes, shuffled.
* `_fit` (235–289):
  1. scale features; build one `KNNNeighborIndex` per view and the neighbour tables;
  2. create the network and one `RelationThreshold` per (layer, relation);
  3. each epoch: loss = classifier CE + `λ1 ×` sum of per-layer similarity CE; Adam with weight decay `λ2`;
  4. after the epoch, feed the average neighbour distance per (layer, relation) into the RL controllers;
  5. cache reference embeddings; record thresholds and convergence in `fit_report_`.
* `_materialise_reference` (292–313, no grad): pre-computes every layer's reference embeddings and similarity scores in chunks so scoring a new transaction needs no recursion.
* `_embed_new` (315–324): finds each new row's k nearest *reference* nodes per view and runs the layers using cached reference features/scores — inductive inference in O(k).
* `_predict_fraud_proba` (326–334): chunked soft-max fraud probability.
* `embed` (336–339): exposes final embeddings (used for analysis and plots).

## 2. Things to remember
* Cost: training is expensive (about 1,426 s on the PaySim run vs 73 s for MG-ECFM) because of k-NN graph construction and recursive embedding.
* Default threshold is 0.5; the experiment replaces it with the F1-optimal operating threshold.
