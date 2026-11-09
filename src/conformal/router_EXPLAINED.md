# Code Walkthrough: `src/conformal/router.py`

**Purpose:** The **pre-transaction router** (Stage 1B, component C4). It turns the detector's calibrated risk `(p̂, Δp̂, amount, active rules)` into one of four actions using the live CS-ACRC thresholds. It runs for every transaction, so it is pure scalar Python (well under the 1 ms budget).

---

## 1. Decision rule (docstring, lines 16–28)
1. **Conservative score:** `s = clip(p̂ + κ·Δp̂, 0, 1)` — uncertain predictions are pushed up (κ=0.5).
2. **Threshold rule** with the current `λ` from the controller.
3. **Optional amount-aware escalation:** Approve → Step-Up if `s·A·(m − β) > c_step` (approving costs more in expectation than a challenge).
4. **Knowledge-base floor:** if any admitted rule fires, action is at least `rule_floor_action` (default Step-Up).
5. **Epistemic guard:** a Block whose uncertainty `Δp̂ > block_max_uncertainty` (0.25) is downgraded to Hold, whose loss is `c_hold` either way.

## 2. Section-by-section

### 2.1 Reason codes (lines 48–51)
`threshold`, `amount_escalation`, `kb_rule_floor`, `uncertain_block_to_hold` — recorded in each decision for the audit trail and the dashboard.

### 2.2 `RoutingDecision` (lines 54–85)
Slot-based record (cheap to create): action, score, `p_hat`, `delta_p_hat`, amount, thresholds used, reason, number of active rules, latency. Provides `label`, `to_dict`, `__repr__`.

### 2.3 `_count_rules` (lines 88–97)
Accepts `None`, a bool, an int count, or any sized collection of rules and returns the count.

### 2.4 `PreTransactionRouter.__init__` (lines 113–133)
Validates κ and the uncertainty cap, binds a controller (default: a new `OnlineAdaptiveCRC`), parses the rule-floor action, and pre-computes `_c_step` and `_esc_coef = m − β` for speed. Keeps `action_counts`.
`from_config` builds from `hyperparams.json['conformal_crc']['router']`.

### 2.5 `route` (lines 149–189) — the hot path
* validates `p∈[0,1]`, `Δp∈[0,1]`, finite amount;
* computes `s`, reads λ directly from `controller._lam` (a read-only view, no copying);
* picks the action by the thresholds; applies the amount escalation, the rule floor, and the uncertainty guard in that order (each can overwrite the reason);
* updates counters, measures its own latency, returns the `RoutingDecision`.

### 2.6 Convenience entry points (lines 191–218)
* `route_tuple((p, d, amt, rules))`;
* `route_scored_event(score_dict, amount, rules)` — consumes the dict from `StackedFusionClassifier.score_event`, taking `Δp̂` from `delta_p_hat` or from a Venn-Abers interval `p1 − p0`;
* `route_batch(...)` — routes many rows against the *current* thresholds with no feedback between them.

### 2.7 `feedback` (lines 221–223)
Reports the verified label to the controller (`controller.update`), which closes the loop.

### 2.8 Monitoring (lines 226–243)
`action_distribution()` (share per action) and `latency_profile(tuples)` — warm-up, then per-call timing; returns mean/p50/p95/p99/max and whether p99 is under the budget.

## 3. Things to remember
* Routing itself never changes the thresholds; only `feedback`/`update` (delayed labels) does.
* The rule floor is how the knowledge base (martingale-gated rules) influences real decisions.
