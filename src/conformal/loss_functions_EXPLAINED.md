# Code Walkthrough: `src/conformal/loss_functions.py`

**Purpose:** Defines the four routing **actions** and the **asymmetric monetary + customer-friction loss** `L(a, y, A)` that the conformal controller tries to keep small.

---

## 1. The loss matrix (docstring, lines 8–33)
`A` = transaction amount, `y = 1` fraud, `m = undetected_fraud_multiplier` (default 1).

| Action | Legitimate (y=0) | Fraud (y=1) |
|---|---|---|
| APPROVE | 0 | `m·A` (the money is lost) |
| STEP_UP | `c_step` | `c_step + β·A` (β = fraction of fraud that beats the challenge) |
| HOLD | `c_hold` | `c_hold` (intercepted before settlement) |
| BLOCK | `c_block_churn` | 0 |

Hold-and-Verify is the only action whose loss does not depend on `y`: it caps loss at `c_hold`, saving `m·A − c_hold` versus approving a fraud. The controller consumes the **normalised, capped** loss `ℓ = min(L/Z, B)` — capping makes the loss bounded, which the adaptive-conformal guarantee requires.

## 2. Section-by-section

### 2.1 `Action` enum (lines 52–92)
`IntEnum` with `APPROVE=0, STEP_UP=1, HOLD=2, BLOCK=3` (ordered by severity). `.label` gives the human name; `Action.parse(...)` accepts an `Action`, an int code, or a string/alias (`"escrow"`, `"decline"`, `"challenge"`, ...), raising on unknown values.

### 2.2 `CostParameters` (lines 95–144)
Frozen dataclass: `c_step=5, c_hold=25, c_block_churn=150, beta_bypass=0.10, undetected_fraud_multiplier=1, loss_normalizer Z=10000, loss_cap B=1`.
* `__post_init__` validates (finite, non-negative, `β∈[0,1]`, `Z>0`, `B>0`).
* `from_config` reads the `costs` block of `configs/project_config.json`, with overrides.

### 2.3 `asymmetric_loss` (lines 150–174)
Pure scalar function implementing the 8-cell matrix, with input checks (`y ∈ {0,1}`, amount finite and ≥ 0).

### 2.4 `AsymmetricLoss` class (lines 177–260)
Callable bound to a `CostParameters` instance, optimised for the hot path (`__slots__`, no NumPy for scalars, ~1 µs).
* `loss` / `__call__`: raw loss for one transaction.
* `normalized`: `min(L/Z, B)`.
* `matrix(amount)`: the 4×2 loss matrix for an amount.
* `hold_savings(amount)`: loss avoided by holding a fraud instead of approving.
* `batch` / `batch_normalized`: vectorised evaluation using two lookup tables — a fixed-cost table and an amount-coefficient table — so `L = fixed[a, y] + coef[a, y]·A`.
* `expected_losses(p, amount)`: conditional risk of each action given fraud probability `p`.
* `bayes_action(p, amount)`: argmin of expected loss (ties → less severe action).

### 2.5 Helpers (lines 263–276)
`_as_action_array` converts names/ints to validated int codes; `batch_asymmetric_loss` is the functional wrapper.

## 3. Things to remember
* The default costs make BLOCK expensive for genuine customers (150) and APPROVE-on-fraud proportional to the amount, so the optimal action depends strongly on amount; the router's `amount_aware_escalation` uses this.
* The values are **assumptions** (set in `project_config.json`), not estimated from data; the paper should state and justify them.
