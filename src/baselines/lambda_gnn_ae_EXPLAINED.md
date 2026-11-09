# Code Walkthrough: `src/baselines/lambda_gnn_ae.py`

**Purpose:** Baseline **BP2**, a re-implementation of Alarfaj & Shahzadi (IEEE Access 2025): a **Lambda architecture** that combines a graph neural network (batch layer), an autoencoder (speed layer) and a logistic-regression combiner (serving layer).

---

## 1. Idea in one picture
* **Batch layer:** training transactions form a k-NN similarity graph; a GraphSAGE network learns fraud embeddings on it and its intermediate "views" are stored.
* **Speed layer:** a new transaction is linked to its k nearest stored nodes and embedded with one message-passing step per layer (no retraining), and an autoencoder trained on legitimate traffic gives a reconstruction-error anomaly score.
* **Serving layer:** logistic regression merges GNN log-odds and log reconstruction score, fitted on a chronological hold-out tail.

## 2. Section-by-section

### 2.1 Docstring (lines 1–29)
Formulas: SAGE update `h_v = σ(W_self h_v + W_nbr mean_{u∈N(v)} h_u)`, AE loss `‖x − g(f(x))‖² / d`.

### 2.2 `LambdaGNNAEConfig` (lines 55–88)
Batch-layer settings (`k_neighbors=10`, `gnn_hidden=64`, `gnn_layers=2`, epochs, batch size), speed-layer settings (`ae_hidden`, `ae_latent`, `ae_epochs`), optimisation (LR, weight decay, dropout, costs 25/1, `val_fraction=0.2`, `patience`) and serving settings (`combiner_C`, `threshold`). Validated in `__post_init__`.

### 2.3 `SAGEMeanLayer` (lines 94–106)
`ReLU(W_self h + W_nbr·mean_neighbours)` followed by dropout; the last layer can skip the activation.

### 2.4 `BatchLayerGNN` (lines 109–142)
* Stack of SAGE layers plus a 2-class head.
* `embed_reference(nodes, level, X_ref, nbr_table)` — **recursive, differentiable** embedding for training: level 0 is the features; each level looks up the node's neighbours, embeds them one level lower, averages, and applies the layer.
* `embed_new(x, nb, views)` — speed-layer embedding for unseen rows: uses the **stored views** `views[level]` of the neighbours so each layer is just a neighbour lookup + mean + layer; cost O(k·L).

### 2.5 `SpeedLayerAutoencoder` (lines 148–161)
Undercomplete autoencoder `g(f(x))`; `reconstruction_loss` = per-row mean squared error.

### 2.6 `LambdaGNNAutoencoder` (lines 167–338)
* `__init__`: scaler, k-NN index, placeholders for GNN, AE, combiner, reference tensors, stored views, `ae_scale_`.
* `_materialise_views` (188–201): after training, compute `H^(0..L−1)` for all reference nodes in chunks (eval mode).
* `_gnn_logits_new` (203–211): GNN logits for new rows from stored views.
* `_train_gnn` (213–242): mini-batch Adam with class-weighted CE; after each epoch rebuilds views and validates on the tail (if any) for early stopping; restores the best weights and rebuilds the final views.
* `_train_ae` (245–262): trains the autoencoder on **legitimate** reference rows only; records the median legitimate reconstruction error as a scale.
* `reconstruction_score` (264–269): `log((error + 1e-12) / median_legit_error)` — a normalised anomaly signal.
* `_components` (272–275): returns the 2-column feature matrix `[GNN log-odds, log reconstruction]`.
* `_fit` (277–321): chronological head/tail split (falls back to all rows if the tail lacks both classes); fit scaler on head; build the k-NN graph and neighbour table; train GNN; train AE; fit the combiner on the **tail** (rows the GNN never trained on) or, if none, on the training rows; log graph size, `k`, and combiner coefficients.
* `_predict_fraud_proba` (324–328): scale → k-NN query → components → combiner probability.
* `layer_scores` (330–338): returns the two signals separately for ablation reporting.

## 3. Things to remember
* Inference needs a k-NN query against all reference nodes (0.447 ms/tx in our batch measurement).
* The combiner is class-balanced logistic regression, so output scores are rankings rather than calibrated probabilities.
