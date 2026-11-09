# Code Walkthrough: `src/conformal/adaptive_crc.py`

**Purpose:** The core of the conformal layer — **Cost-Sensitive Online Adaptive Conformal Risk Control (CS-ACRC)**. It keeps three risk thresholds `λ = (λ_low, λ_mid, λ_high)` that turn a risk score into one of four actions, and **adapts them online** from the realised loss of each verified outcome.

| Score range | Action |
|---|---|
| `s < λ_low` | Auto-Approve |
| `λ_low ≤ s < λ_mid` | Step-Up Auth |
| `λ_mid ≤ s < λ_high` | Hold-and-Verify Escrow |
| `s ≥ λ_high` | Immediate Block |

---

## 1. The idea
After each outcome, the normalised loss `ℓ_t = min(L/Z, B)` is compared with a target risk `α`:
`λ_{t+1} = Π[ λ_t + γ(α − ℓ_t) ]`
* loss **above** target → thresholds go **down** → more friction (step-ups, holds, blocks);
* loss **below** target → thresholds go **up** → less friction.
`Π` projects back onto the feasible set so thresholds stay strictly ordered with a minimum gap.

**Guarantee (docstring, lines 33–54):** telescoping the update gives an exact identity, so for *any* sequence (drifting, adversarial, delayed feedback) `|mean(ℓ) − α| ≤ (range + |C_T|)/(γT)`. The *one-sided* bound uses only the "protective" projection displacement. This is a **long-run average** bound, not a per-transaction one — worth stating precisely in the paper.

## 2. Section-by-section

### 2.1 Imports and constants (lines 56–76)
Loads the cost model from `loss_functions.py`; `THRESHOLD_NAMES = ("low","mid","high")`.

### 2.2 Projection (lines 82–114)
* `_pava_nondecreasing`: Pool-Adjacent-Violators algorithm — the L2 projection onto non-decreasing sequences.
* `project_ordered_thresholds(lam, lower, upper, gap)`: projection onto `{lower ≤ λ1, λ_{k+1} − λ_k ≥ gap, λ_K ≤ upper}`. Trick: substitute `μ_k = λ_k − (k−1)·gap`, which turns the gap constraints into plain ordering; then PAVA + clip to the box + shift back. Raises if infeasible.

### 2.3 `CRCStep` (lines 120–133)
Immutable result of one update (loss, normalised loss, thresholds before/after, whether projected/adapted).

### 2.4 `OnlineAdaptiveCRC.__init__` (lines 156–190)
Parameters: `alpha=0.02` (target risk), `gamma=0.005` (step), `lambda_init=(0.20, 0.50, 0.85)`, `lambda_bounds=(0,1)`, `min_gap=0.01`, `warmup`, `mode` (`adaptive` or `static` for the B3 ablation), `loss`, `rolling_window=1000`. Validates everything, then calls `reset()`.

### 2.5 `from_config` and `reset` (lines 193–244)
`from_config` reads `hyperparams.json['conformal_crc']` and the cost block of `project_config.json`. `reset` restores the projected initial thresholds and zeroes all statistics (cumulative losses, projection displacements `_clip` / `_clip_up`, rolling window, history).

### 2.6 Threshold accessors and routing (lines 247–285)
Properties for each λ, `protection_levels` (`θ = 1 − λ`), `is_strictly_monotone`, and `action_for_score(score)` — the threshold rule above.

### 2.7 `update(action, y, amount)` (lines 288–343) — the online step
1. Compute raw loss, normalise by `Z`, cap at `B`.
2. Update running sums and the rolling window.
3. If `mode == "adaptive"` and past warm-up: `raw = λ + γ(α − ℓ)`, project, and accumulate the **projection displacement** (total and upward-only).
4. Record history and return a `CRCStep`.
`update_batch` applies this sequentially over arrays.

### 2.8 Monitoring (lines 353–400)
`rolling_loss`, `rolling_coverage_gap` (rolling loss − α; used by the closed loop as a drift signal), `rolling_fraud_rate`, `cumulative_loss`, and the bound helpers `threshold_ranges`, `epsilon_bound`, `risk_upper_bound`, `window_bound`.

### 2.9 `guarantee()` and `feasibility()` (lines 402–445)
* `guarantee`: reports mean adaptive loss, the coverage gap, `ε_T`, whether the two bounds hold, projection displacements, and the **identity residual** (a numerical check that the telescoping identity holds).
* `feasibility`: if even the protective extreme (blocking/holding every genuine customer) costs more than `α`, the target is unattainable and thresholds pin to their lower bound; a note explains the fix (raise α or Z).

### 2.10 `history`, `summary`, `state_dict`, `load_state_dict` (lines 447–489)
Trajectories for plots (Fig. 3), a summary dict, and snapshot/restore of the *online state* (so a stopped run can resume with its guarantees intact; loading validates monotonicity).

## 3. Things to remember
* In our PaySim run the thresholds mostly rest at bounds because the stream is very cheap relative to the risk budget (docstring: "thresholds resting at the permissive bound simply mean the stream is cheaper than the risk budget").
* `static` mode (frozen λ) is the B3 ablation arm.
