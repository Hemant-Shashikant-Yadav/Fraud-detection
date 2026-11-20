# Code Walkthrough: `src/conformal/budgeted_crc.py`

**Purpose:** The pre-transaction router controller that MG-ECFM uses by default (`conformal_crc.decision_rule = "budgeted_bayes"`). It selects Approve, Step-Up, Hold or Block from the cost matrix and the amount of each transaction. Two online controllers keep the intervention rate at a budget and correct the probabilities under drift.

**Why it replaces the λ-band controller:** On PaySim the old controller (`adaptive_crc.py`) approved 311 of 404 test frauds (loss 213.9M) for three reasons:

1. Its risk target α = 0.02 is above the fraud rate, so "approve everything" was inside the budget.
2. The loss cap `min(L / 10,000, 1)` removed the amount.
3. The 0.01-wide bands left Step-Up and Hold almost empty.

With this controller, the loss is 13.94M and 97.9% of the fraud value is stopped before settlement.

---

## 1. Decision rule

For a calibrated probability `p` and an amount `A`:

| Action | Expected cost |
|:---|:---|
| Approve | `m·q·A` |
| Step-Up | `c_step + β·q·A + ν` |
| Hold | `c_hold + ν` |
| Block | `(1 − q)·c_block + ν` |

* `q = sigmoid(logit(p) + θ)` is the drift-corrected probability.
* `ν = exp(η) − 1 ≥ 0` is the price of one intervention.
* The action is the argmin. With `ν = 0` and `θ = 0`, this is the Bayes-optimal action.
* The cost slopes in `q` are `(m·A, β·A, 0, −c_block)`, so the action never goes down when `q` goes up. A larger `ν` never adds interventions.

## 2. Online controllers (updated only from released feedback)

| Controller | Update | Guarantee |
|:---|:---|:---|
| Intervention budget ρ | `η ← clip(η + γ_b (I_t − ρ), 0, η_max)` | `mean I ≤ ρ + (η_max + C⁺)/(γ_b T)` |
| Drift correction θ | `θ ← clip(θ + γ_c (y_t − q_t), ±θ_max)` | `|mean(y − q)| ≤ (2θ_max + |C|)/(γ_c T)` |

`C` is the clip displacement. The guarantees come from telescoping the updates, as in adaptive conformal inference, so they need no exchangeability assumption and they hold under bounded label delay. `guarantee()` reports them, with the identity residual as a check.

## 3. Section-by-section

### 3.1 Helpers (`_logit`, `_sigmoid`, `BudgetStep`)
Stable logit and sigmoid (p is clipped to [1e-6, 1 − 1e-6]). `BudgetStep` records one feedback update.

### 3.2 `BudgetedBayesCRC.__init__` / `from_config` / `reset`
Reads `hyperparams.json → conformal_crc.budgeted_bayes` and the costs from `project_config.json`. `mode = "static"` freezes ν and θ (used by the ablation arm B3 static).

### 3.3 Decisions
* `expected_losses(p, A)`, `decide(p, A)`: the scalar hot path (pure Python, well under 1 ms).
* `decide_batch(p, A)`: the vectorised version, used in simulations.
* `action_for_score(score)`: the decision at `reference_amount`, for compatibility with `feedback_adapter.py`.
* `thresholds`: the probability cut points (Step-Up, Hold, Block) at `reference_amount`. They are for logging and Figure 3 only. They are cached on `(η, θ)`.

### 3.4 `warm_start(p, A)`
Bisection on η so that the Bayes rule meets the budget on past scores (the calibration split in the pipeline). It reads no labels.

### 3.5 `update` / `update_batch`
The realised loss, the intervention flag, the rolling statistics and the two controller steps. The score `p` is needed for the θ update.

### 3.6 Monitoring and state
`guarantee()`, `summary()`, `history()`, `state_dict()` / `load_state_dict()` (all counters and clip terms are saved, so the certificates survive a reload).

### 3.7 `build_controller(hyperparams, project_config, **overrides)`
A factory. `decision_rule = "threshold"` gives the legacy `OnlineAdaptiveCRC`. `"budgeted_bayes"` gives this class.

## 4. Settings and how they were selected

| Setting | Value | Selection |
|:---|:---:|:---|
| `intervention_budget` ρ | 0.03 | The knee of the loss-against-friction curve on the operating split (steps 576–631) |
| `gamma_budget` γ_b | 0.001 | Operating-split grid {0.0005, 0.001, 0.002}; the result hardly changes |
| `gamma_calibration` γ_c | 0.02 | Operating-split grid {0.001, 0.005, 0.02} |
| `eta_max`, `theta_max` | 20, 4 | Bounds wide enough to never bind in practice |

No test label was used to select any setting.
