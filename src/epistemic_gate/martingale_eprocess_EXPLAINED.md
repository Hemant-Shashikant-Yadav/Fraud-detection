# Code Walkthrough: `src/epistemic_gate/martingale_eprocess.py`

**Purpose:** The statistical heart of the **knowledge-evolution gate**. It decides — with a *provable* false-admission bound — when a candidate fraud rule is good enough to **enter** the active knowledge base and when an active rule has gone bad and must be **rolled back**.

**Core theory:** both decisions are sequential **betting e-processes** (test supermartingales). By Ville's inequality, under the null hypothesis the wealth `M_t` ever reaches `1/α` with probability at most `α`, *at any stopping time* (anytime-valid).

---

## 1. The two tests (docstring, lines 8–26)
| Gate | Null `H0` | Alternative | Indicator `x_i` | Wealth |
|---|---|---|---|---|
| **Admission** (quarantined rule) | Precision ≤ `p0` (0.5) | Precision = `p1` (0.8) | `y_i` (confirmed fraud) | `M^admit = ∏ [y p1/p0 + (1−y)(1−p1)/(1−p0)]` |
| **Rollback** (active rule) | FPR ≤ `q0` (0.2) | FPR = `q1` (0.5) | `1 − y_i` (false alarm) | `M^roll = ∏ [(1−y) q1/q0 + y (1−q1)/(1−q0)]` |

Both are one **Bernoulli likelihood-ratio betting process** with null rate `θ0` and alternative `θ1 > θ0`. For every `θ ≤ θ0` the one-step factor has conditional mean ≤ 1, so `M_t` is a supermartingale for the *composite* null.

**What is and is not guaranteed:** Ville bounds the probability of a *false admission / false rollback* at α under the null (assuming conditionally independent outcomes). It does **not** by itself guarantee "rollback within ≤ 3 cycles"; speed depends on how bad the rule is and how often it fires — which is why the code has exact crossing-probability helpers (below).

## 2. Section-by-section

### 2.1 Constants (lines 59–62)
Allowed betting strategies (`likelihood_ratio`, `plugin`), monitors (`martingale`, `e_detector`), rollback policies (`targeted`, `snapshot`) and screening modes (`strict`, `lenient`, `off`).

### 2.2 Closed-form helpers (lines 68–155)
* `_validate_rates`, `_validate_alpha`: require `0<θ0<θ1<1`, `0<α<1`.
* `betting_factor`, `log_factors`: one-step multiplier `θ1/θ0` (event) or `(1−θ1)/(1−θ0)` (non-event) and its logs.
* `expected_factor(θ)`: `E_θ[f]`, which is ≤ 1 for `θ ≤ θ0` (the supermartingale proof).
* `expected_log_growth`: drift `E[log f]`; positive means the test eventually rejects.
* `min_events_to_cross`: fewest observations (all events) to reach `1/α`: `⌈log(1/α)/log(θ1/θ0)⌉`. For admission: 4 straight hits at the defaults; this is what makes admission need real evidence.
* `crossing_probability(n, θ, θ0, θ1, α)`: **exact** probability the wealth crosses within `n` observations. Because `log M_t` depends only on the event count `k`, the not-yet-crossed probability mass is propagated over `k` with absorption at the boundary (dynamic programme).
* `steps_for_power`: smallest `n` with crossing probability ≥ a target power (doubling then binary search).

### 2.3 `BettingEProcess` (lines 161–259)
* **Betting strategies:** `likelihood_ratio` (fixed `θ1`) or `plugin` (predictable `θ1_t = clip((k+s·m)/(n+s), θ0+δ, 1−δ)` using only past data and a prior mean `m`, e.g. the Subjective-Logic expected probability — validity holds for any predictable `θ1_t > θ0`).
* **Monitors:** `martingale` (plain product, α false-alarm bound) or `e_detector` (CUSUM: `log D_t = max(0, log D_{t−1}) + log f_t`; restarts after good history so detection delay does not grow with the rule's age; guarantee is on average run length `≥ 1/α`).
* State: counts `n`, `n_events`, `log_wealth`, `max_log_wealth`, `crossed`, `crossed_at`. Wealth is kept in **log space** (exponentiated with a 700 cap to avoid overflow).
* `anytime_p_value = min(1, 1/sup M_s)`.
* `update(x)`: log-factor, update wealth (with the CUSUM max for the detector), `max_log_wealth`; returns `True` exactly once at first crossing; raises if already stopped.
* `update_many`, `to_dict` (certificate payload).

### 2.4 Factories and Monte-Carlo (lines 262–296)
`admission_process`, `rollback_process`; `simulate_crossing_rate` — vectorised Monte-Carlo crossing frequency and first-crossing times for either monitor (used in the Ville-bound check of the stress test).

### 2.5 `GateConfig` (lines 302–377)
Defaults: `p0=0.5, p1=0.8, α_admit=0.05, q0=0.2, q1=0.5, α_roll=0.05, max_rollback_cycles=3, max_firings_per_candidate=5000, screening="lenient", max_admissions_per_rule=2`. `__post_init__` validates; `from_config` reads `hyperparams.json['martingale_eprocess']`.
Planning helpers: `admission_min_hits`, `rollback_min_false_alarms`, `rollback_crossing_probability(n_firings, FPR)`, and `firings_per_cycle_for_rollback(FPR, power)` — the firing volume needed so a rule with a given FPR is rolled back within `max_rollback_cycles` with a chosen probability. This is how the "≤ 3 cycles" claim should be certified: **empirically / for a given volume, not as a theorem**.

### 2.6 `RuleRecord`, `CycleReport` (lines 380–409)
Per-rule state (status, opinions, the two e-processes, counters, snapshot id, event log) and a per-cycle report (promoted / rolled back / rejected, head snapshot).

### 2.7 `MartingaleEProcessGate` (lines 412–654)
Lifecycle: `submit → [Subjective-Logic screen] → QUARANTINED → (M_admit ≥ 1/α) → ACTIVE → (M_roll ≥ 1/α) → rollback → QUARANTINED → (budgets exhausted) → REJECTED`. Every promotion/rollback is committed to the `KnowledgeStore` as an immutable snapshot with the e-value certificate.
* `_new_admission`, `_new_rollback`: create processes (the rollback uses the plain likelihood-ratio bet when the CUSUM monitor is chosen).
* `submit_candidate(rule, now)`: fuses the rule's evidence (Subjective Logic), applies the screening mode (`strict` = fusion decision; `lenient` = reject only *confident* disbelief; `off` = admit to buffer), records the candidate as quarantined or rejected in the store.
* `inject_active(rule)`: places a rule directly in the active KB (adversarial-injection experiments) under a fresh rollback monitor.
* `_promote`: remembers the pre-admission snapshot, admits in the store, saves the admission certificate, starts a rollback monitor.
* `_rollback`: `snapshot` policy restores the pre-admission snapshot and also demotes collateral rules admitted since; `targeted` policy reverts just this rule. Saves the certificate; demoted rules return to quarantine (new admission process) or become `rejected` if their admission budget is used; then `_resync_from_store` keeps memory consistent.
* `observe(rule_id, y)`: feeds one verified firing — quarantined rules update the admission process (promotion, or rejection when the firing budget is exhausted); active rules update the rollback process with `1−y`.
* `run_update_cycle(outcomes)`: one deterministic (sorted) validation cycle; after a rollback, the rest of that rule's firings in the cycle are skipped; updates each touched rule's Subjective-Logic opinion with `(hits, false alarms)` and stores the opinions.
* Queries: `quarantine_buffer`, `active_rule_ids`, `status`, `ville_report` (wealth, threshold, anytime p-value, Ville bound), `summary`.

## 3. Things to remember
* In our PaySim ablation the gate ended with **0 active rules** — a result to investigate (candidate quality, thresholds `p0/p1`, or too few verified firings).
* The stress-test table (FAR ≤ 1.6 %, TTR mean ≈ 1.0–1.2 cycles) comes from this class on synthetic candidate feeds.
