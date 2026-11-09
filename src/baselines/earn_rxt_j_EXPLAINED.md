# Code Walkthrough: `src/baselines/earn_rxt_j.py`

**Purpose:** Baseline **BP1**, a re-implementation of Almazroi & Ayub (IEEE Access 2023): SMOTE re-balancing → **EARN** feature extractor (autoencoder + ResNet) → **ResNeXt-GRU** classifier → **Jaya** fine-tuning of the output layer. It is a purely tabular (no graph) deep model.

In our PaySim run BP1 is the fastest at inference (0.025 ms/tx) but trains slowly (~583 s).

---

## 1. Section-by-section

### 1.1 Docstring (lines 1–41)
Explains the four stages and the formulas: SMOTE interpolation `x_new = x_i + u(x_nn − x_i)`, joint EARN loss `MSE(x, x̂) + CE`, the ResNeXt split-transform-merge block, and the Jaya update `θ' = θ + r1(θ_best − |θ|) − r2(θ_worst − |θ|)`.

### 1.2 `EARNRXTJConfig` (lines 70–122)
Grouped hyper-parameters:
* **SMOTE:** `use_smote`, target minority ratio `smote_ratio=0.2`, `smote_k=5`, cap `smote_max_synthetic=200 000`.
* **EARN:** AE sizes (`ae_hidden`, `ae_latent`), ResNet width/blocks/embedding, `earn_epochs`, `recon_weight`.
* **RXT:** token count/channels, `cardinality`, `bottleneck_width`, blocks, GRU size, `rxt_epochs`.
* **Optimisation:** batch size, LR, weight decay, dropout, costs `fn_cost=25 / fp_cost=1`, `val_fraction=0.15`, `patience`.
* **Jaya:** `jaya_population=20`, `jaya_iterations=30`, `jaya_init_scale`.
`__post_init__` rejects invalid values (non-positive ratios/costs, population < 2, ...).

### 1.3 `smote_oversample` (lines 128–151)
For the fraud class only: computes how many synthetic rows are needed (`ratio × n_legit − n_fraud`, capped), finds each fraud's `k` nearest fraud neighbours, then creates `base + gap × (neighbour − base)` with `gap ~ U(0,1)`. Returns the original rows followed by the synthetic rows. If there are < 2 frauds, nothing is generated.

### 1.4 `PreActResidualBlock` (lines 157–171)
`h + W2·relu(BN(W1·relu(BN(h))))` with dropout — a pre-activation residual block for tabular vectors.

### 1.5 `EARNExtractor` (lines 174–203)
Two parallel branches:
* **Autoencoder** encoder (→ tanh latent) and decoder; the squared reconstruction error per row is `recon`.
* **ResNet branch:** input projection → residual blocks → embedding.
The output "ensemble descriptor" is `[z_AE ‖ z_Res ‖ log1p(recon)]`. An auxiliary linear head on `[z_AE ‖ z_Res]` supplies the supervised loss while training the extractor.

### 1.6 `ResNeXtBlock1d` and `ResNeXtGRU` (lines 209–253)
* ResNeXt block: 1×1 reduce → grouped 3-tap convolution (groups = `cardinality`) → 1×1 expand → add shortcut → ReLU.
* `ResNeXtGRU`: linearly projects the descriptor to `T × C` tokens, applies the blocks, then a GRU reads the `T` tokens; the last hidden state (after dropout) feeds the 2-class output layer. `penultimate()` exposes the hidden vector (used by Jaya).

### 1.7 `jaya_minimize` (lines 259–282)
Parameter-free population optimiser. Population is initialised around `theta0` with small noise, **member 0 = θ0 exactly**, so the result can never be worse than the start. Each iteration moves every candidate toward the best and away from the worst using element-wise random weights and accepts the move only if the objective improves (greedy).

### 1.8 `EARNRXTJ` class (lines 288–447)
* `_weighted_ce`: cost-sensitive cross-entropy helper.
* `_train_earn` (306–336) and `_train_rxt` (338–364): mini-batch Adam training loops with early stopping on the validation tail; restore the best weights at the end.
* `_jaya_finetune` (366–389): freezes the network up to the penultimate layer, flattens the output layer `(W, b)` to a vector, and lets Jaya minimise the weighted CE on the validation tail; the best vector is written back.
* `_fit` (391–427): chronological head/tail split → fit the scaler on the head only → (if the tail swallowed all frauds, fall back to all rows) → SMOTE on the head only → class weights → train EARN → extract features → train RXT → Jaya if a usable validation tail exists. Many diagnostics go into `fit_report_`.
* `extract_features` (430–436): public access to `F(x)` for analysis.
* `_predict_fraud_proba` (438–447): scale → EARN → RXT → soft-max, in chunks of 8192.

## 2. Things to remember
* SMOTE and early stopping never touch future data: SMOTE is applied to the head only; the tail is the last rows in time.
* Because the loss is class-weighted and SMOTE is used, raw probabilities are not calibrated (BP1's Brier is nevertheless the best in our table).
