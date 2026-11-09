# Code Walkthrough: `src/detection/autoencoder_residual.py`

**Purpose:** The **unsupervised** anomaly signal. A denoising autoencoder (DAE) learns what *normal* transactions look like; its reconstruction error `r_AE(x) = ‖x − g(f(x))‖₂` is large for unusual transactions and becomes one feature of the fusion classifier.

---

## 1. Idea
* Fit on **legitimate** rows only (by default), so fraud reconstructs badly.
* During training the input is corrupted with Gaussian noise `x̃ = x + N(0, σ²I)` and the network must rebuild the clean `x` (Vincent et al., 2008) — this stops it from learning the identity map.
* Features are standardised with a signed-log scaler fitted on the training split only.
* Early stopping uses the **chronologically last** part of the training rows (no shuffling across time).

## 2. Section-by-section

### 2.1 `AutoencoderConfig` (lines 38–87)
Frozen dataclass: `input_dim`, `hidden_dims=(64, 32)`, `latent_dim=16`, activation (`relu|elu|tanh|gelu`), `input_noise_std=0.1`, `dropout=0.1`, LR, weight decay, `batch_size=2048`, `epochs=30`, `train_on_legitimate_only`, `early_stopping_patience=5`, `val_fraction=0.1`.
`__post_init__` validates; `from_hyperparams` reads the `autoencoder` block of `hyperparams.json`.

### 2.2 `DenoisingAutoencoder` (lines 93–136)
Symmetric MLP built inside `fork_rng` with a fixed seed (so initial weights are reproducible without touching global RNG state).
* **Encoder:** `Linear → act → Dropout` per hidden size, then `Linear` to the latent size.
* **Decoder:** mirrors the encoder in reverse, ending in a `Linear` back to `input_dim`.
* `corrupt(x)` adds the noise; `forward = decode(encode(x))`; `residual(x)` is the per-row L2 error (no corruption).

### 2.3 `AutoencoderResidualEngine` (lines 139–246)
Wraps scaler + model for NumPy inputs.
* `__init__`: creates the scaler (`SignedLogStandardizer`), resolves the device, builds the model.
* `is_fitted`: scaler fitted **and** a best epoch recorded.
* `_tensor`: scale → float32 tensor.
* `fit` (158–217): validates the shape; keeps only legitimate rows if requested (needs ≥ 2); splits the last `val_fraction` as validation (only if ≥ 20 rows); **fits the scaler on the training part only**; trains with Adam + MSE on corrupted mini-batches (shuffled with a seeded generator); evaluates on the validation set each epoch; keeps the best weights (`deepcopy` of the state) and stops after `patience` epochs without improvement; restores the best weights.
* `residual(features)` (219–232): batched (65 536 rows) residuals as a NumPy vector.
* `residual_one` / `latency_ms`: single-row fast path and median latency timing (used in the real-time profile).

## 3. Things to remember
* The residual is *unsupervised*: it can flag novel legitimate behaviour as well as fraud, which is why it is only one input to the supervised fusion model.
* Scaling statistics come from training rows only — no look-ahead.
