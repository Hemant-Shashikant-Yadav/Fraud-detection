# Code Walkthrough: `src/baselines/argus_conformal.py`

**Purpose:** Baseline **BP4**, a re-implementation of ARGUS (Nayak & Bushara, 2026): a hybrid Transformer + state-space network, Monte-Carlo-dropout uncertainty, and *static* split-conformal risk control for selective fraud classification.

**Why it matters for the paper:** it is the strongest baseline (PR-AUC 0.299 vs MG-ECFM 0.286 on PaySim 2M) and the closest relative of MG-ECFM's conformal layer. Its conformal threshold is *static*, which drift breaks; MG-ECFM's CS-ACRC is the adaptive answer.

---

## 1. Section-by-section

### 1.1 Module docstring (lines 1–43)
Describes the architecture (feature tokenizer → repeated Transformer + Gated-Token-Mixing blocks → head) and the conformal rule
`t_hat = sup{ t : (Σ 1{p_i < t} + B) / (n1 + 1) ≤ α }` which bounds the expected false-negative rate by α under exchangeability.

### 1.2 `ARGUSConfig` (lines 68–103)
Dataclass of hyper-parameters: model size (`d_model=32, n_heads=4, n_blocks=2, ssm_state=16`), optimisation (`epochs, batch_size, learning_rate, weight_decay`), costs (`fn_cost=25, fp_cost=1`), data fractions (`val_fraction=0.1`, `cal_fraction=0.2`), early stopping (`patience`), MC dropout (`mc_samples=30`), conformal target (`crc_alpha=0.1`) and the abstention quantile.
`__post_init__` validates: `d_model` divisible by `n_heads`, `0<α<1`, at least one MC sample, fractions valid, `0<dropout<1` (MC dropout needs dropout on).

### 1.3 `FeatureTokenizer` (lines 109–122)
Turns each scalar feature `x_j` into a token `x_j·w_j + b_j` (own weight vector per feature) and prepends a learned `[CLS]` token. Output shape `(batch, d+1, D)`.

### 1.4 `DiagonalSSMKernel` (lines 125–144)
Builds the convolution kernel of a diagonal linear state-space model: `A = −exp(A_log)` (always negative ⇒ stable), `A_bar = exp(Δ·A)`, `B_bar = Δ·B`, kernel `K_t = Σ_n C·B_bar·A_bar^t`. Parameters: `A_log, B, C, log_dt`.

### 1.5 `causal_toeplitz_conv` (lines 147–153)
Applies the kernel as a causal convolution: builds an `L×L` Toeplitz matrix from the lag `i−j` (zero for `j>i`) and contracts it with the sequence by `einsum`.

### 1.6 `GatedTokenMixing` (lines 156–175)
Pre-norm SSM token mixer. Projects to `u` and a `gate`; runs the SSM **forward and backward** over tokens (features have no natural order); sums with a skip term `D·u`, multiplies by `SiLU(gate)`, projects out, and adds a residual connection.

### 1.7 `TransformerBlock` (lines 178–195)
Standard pre-norm multi-head self-attention + GELU feed-forward, with explicit `Dropout` modules (needed so MC dropout can switch them on).

### 1.8 `HybridTransSSM` (lines 198–220)
Stacks `n_blocks` × (Transformer, GTM), then `LayerNorm` on the `[CLS]` token, dropout and a 2-class linear head.
`enable_mc_dropout()` puts the network in eval mode but turns **only dropout layers back to train mode** — the trick behind Bayesian MC dropout.

### 1.9 `crc_fnr_threshold` (lines 226–243)
Implements the conformal risk control rule on the *fraud* calibration scores:
1. sort scores; `budget = α(n1+1) − B` = how many frauds may be missed;
2. if `n1 == 0` or `budget < 0` → infeasible, return `t_hat = 0` (flag everything);
3. else `t_hat = s[m]` where `m = floor(budget)` (at most `m` scores lie strictly below it).
Returns the threshold plus feasibility and counts.

### 1.10 `ARGUSConformal` class (lines 249–372)
* `__init__`: builds config, scaler, placeholders; default threshold = `fallback_threshold`.
* `_split` (265–273): chronological train / validation / calibration slices (the last rows are the calibration tail).
* `_fit` (275–332):
  * scale with a train-only `SignedLogStandardScaler`;
  * class-weighted cross-entropy, AdamW, gradient clipping at 1.0, early stopping on the validation tail when it holds both classes;
  * after training, run MC dropout on the calibration tail, compute `t_hat` from fraud scores only, set it as `threshold_`, and set the **abstain threshold** to the `abstain_quantile` of the MC standard deviations;
  * stores calibration FNR, flag rate and other diagnostics in `fit_report_`.
* `_mc_dropout` (335–357): `T` seeded stochastic passes (RNG forked so global state is untouched). Returns mean probability, std, predictive entropy and BALD mutual information `I = H[p̄] − mean H[p_t]`.
* `_predict_fraud_proba` (359–360): the MC mean is the score.
* `predict_with_uncertainty` (362–365) returns all uncertainty arrays.
* `predict_selective` (367–372): `1` = flag, `0` = approve, `−1` = abstain when the std exceeds the abstain threshold.
* `_binary_entropy` (375–377): clamped entropy helper.

## 2. Things to remember
* The guarantee `E[FNR] ≤ α` assumes exchangeable data; under concept drift it does not hold — the point of the MG-ECFM comparison.
* MC dropout makes scoring `T` times more expensive (latency 0.168 ms per transaction in batch mode in our run).
