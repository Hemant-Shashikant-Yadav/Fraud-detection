# Code Walkthrough: `src/recovery/action_simulator.py`

**Purpose:** Simulates the **post-detection recovery protocol** (Stage 5) on top of a TD-TPF trace: block the victim's instrument, freeze mule wallets, and claw back funds by chargeback or wire recall — all inside the 24-hour settlement window.

---

## 1. Protocol (docstring, lines 8–25)
With `τ_d` = detection time:
1. **Instrument block** at `τ_b = τ_d + block_latency`: follow-on fraudulent debits from the victim after `τ_b` (inside the window) are prevented; follow-on fraud settled before `τ_b` adds to exposure.
2. **Account freeze** at `τ_f = τ_d + freeze_latency`: the trace is evaluated at `τ_f` (money mules move during the reaction time is *followed*, not assumed frozen). Each tainted wallet is frozen for `min(T(v), Balance(v))·freeze_liquid_fraction`. With `freeze_policy="full_account"`, the untainted part is **collateral** charged at `collateral_rate`.
3. **Chargeback / recall** at `τ_c = τ_d + chargeback_latency`: value that reached merchants is disputed (`merchant_chargeback_rate` = 0.70); external wires recalled at `wire_recall_rate`; cash-outs irrecoverable by default.
An action executes only if its time is ≤ `τ0 + W_rec`.

Metrics: `Recovered = frozen + charged_back/recalled + returned_to_victim`; `Recovery Rate = 100·Recovered/A0`; `Exposure = A0 + follow-on fraud attempted`; `Prevented loss = Recovered + follow-on blocked`; `Prevented-Loss Ratio = 100·Prevented/Exposure`; `Net savings = Prevented − operational cost − collateral penalty`.
`mode="expected"` books expected values (deterministic); `mode="stochastic"` draws Bernoulli successes from a seeded generator.

## 2. Section-by-section

### 2.1 Constants (lines 60–68)
Action names (`account_freeze`, `instrument_block`, `chargeback`, `wire_recall`), freeze policies, modes.

### 2.2 `RecoveryPolicy` (lines 71–127)
Frozen dataclass: window 24 h, latencies (freeze 0.5 h, block 0.25 h, chargeback 2 h), success/recovery rates, costs (freeze 10, block 2, chargeback 25), `collateral_rate=0.02`, mode, seed. `__post_init__` validates; `from_config` reads `hyperparams.json['tdtpf']` (and `actions` sub-block).

### 2.3 `RecoveryAction` and `RecoveryOutcome` (lines 130–225)
`RecoveryAction`: one executed/attempted action (target, time, amount targeted/recovered, cost, executed, succeeded, collateral). `RecoveryOutcome`: totals (frozen, charged back, recalled, returned to victim, follow-on exposure/blocked, costs) and **derived metrics** as properties: `recovered`, `recovery_rate_pct`, `exposure`, `prevented_loss`, `prevented_loss_ratio_pct`, `residual_loss`, `net_savings`; `metrics()` and `actions_frame()` export them.

### 2.4 `RecoverySimulator` (lines 228–385)
* `__init__`: requires policy and tracker windows to match; seeds an RNG.
* `_success` / `_realised`: expected value `p·amount` in expected mode, or an all-or-nothing Bernoulli draw in stochastic mode.
* `follow_on_fraud(fraud)`: labelled fraudulent debits from the same victim within the window after `e0`.
* `simulate(fraud, tau_detect)`: runs the three steps in order — (1) block: compute `follow_on_blocked`; (2) freeze: trace at `τ_f`, freeze each liquid mule balance (pay freeze cost, record collateral); (3) chargeback/recall for each exit flow using the per-kind rate; costs charged only for executed actions. If detection is after the window, nothing is executed.
* `simulate_many`: metrics DataFrame for many frauds with per-fraud delays.
* `aggregate`: portfolio recovery rate, prevented-loss ratio, costs, net savings, share within window.
* `recovery_curve(fraud, delays)`: recovery rate / prevented-loss ratio vs detection delay (Fig. 4 of the paper).

## 3. Things to remember
* The recovery curve is essentially flat up to the freeze latency and then falls as money moves through mules and exits — the central trade-off of the time-decay model.
* All rates and latencies are **assumptions** from the config; recovery results are simulation outputs, not measured recoveries.
