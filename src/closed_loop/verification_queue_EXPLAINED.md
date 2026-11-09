# Code Walkthrough: `src/closed_loop/verification_queue.py`

**Purpose:** Simulates **delayed ground truth**. A decision made at time τ only gets its true fraud/legit label after a realistic delay of 1–14 days (customer dispute, bank chargeback, or the dispute window closing). The queue releases labels strictly in chronological order.

**Why it matters:** PaySim has no real dispute timestamps, so the delay is *simulated*; the paper must state this distribution.

---

## 1. Section-by-section

### 1.1 Docstring (lines 1–34)
Separates two information channels:
* **Immediate telemetry** at decision time (step-up pass/fail, hold, block) – *not* a label.
* **Delayed labels** released at `τ + Δτ`, `Δτ ∈ [1, 14]` days, via three sources: `customer_dispute` (log-normal, median 2 d), `bank_chargeback` (log-normal, median 7 d), `maturity` (genuine transactions confirmed when the dispute window ends, or sampled).
A fraud is reported by the customer with probability `dispute_share` (0.6), otherwise by chargeback. 1 PaySim step = 1 hour. Causality rules: a label is never visible early; the clock never moves backwards; ties release in submission order.

### 1.2 Constants (lines 49–63)
Source names, telemetry kinds and the allowed `legit_policy` values (`maturity`, `sampled`).

### 1.3 `LatencyModel` (lines 66–148)
Frozen dataclass of the delay distributions and telemetry rates (`beta_bypass=0.10` = chance a fraudster passes step-up; `legit_step_up_fail_rate=0.02`).
* `__post_init__` validates ranges.
* `from_config` reads `label_delay_days` from `hyperparams.json` (or the project config's `verification` section) and `beta_bypass` from the cost config.
* `min_hours` / `max_hours` convert days to hours.
* `_truncated_lognormal_days` (123–135): inverse-CDF sampling from a log-normal truncated to `[min_days, max_days]` (uniform draw mapped into the allowed CDF range), so no delay falls outside the window.
* `sample(y, rng)` (137–148): for frauds, pick dispute or chargeback and draw a delay; for legitimate rows, either the fixed maximum (`maturity`) or a sampled delay.

### 1.4 `ImmediateTelemetry` and `VerifiedLabel` (lines 151–183)
Frozen records. `VerifiedLabel` carries ids, decision/release times, `y`, amount, action, source, fired rule ids, score, plus `latency_hours/days` properties.

### 1.5 `VerificationQueue` (lines 186–324)
* `__init__`: latency model, two RNG streams (labels, telemetry), a **min-heap** of `(release_time, sequence, label)`, the monotone clock, `_known` release times, and logs.
* Properties: `clock`, `n_pending`, `n_released`, `released`, `next_release_time`.
* `label_available_at`, `is_label_available(tx_id, tau)` – true only if released by `tau` *and* by the clock.
* `_telemetry` (239–249): for `STEP_UP`, passing depends on the truth (fraud passes with `beta_bypass`; legit passes unless it hits the fail rate); other actions map to fixed kinds.
* `submit` (252–282): validates `y`, rejects duplicate ids and decisions earlier than the clock, draws (or accepts a manual) delay, enforces bounds in `strict_bounds` mode, pushes the label on the heap and logs telemetry.
* `release_until(tau_now)` (285–295): refuses to move backwards, then pops every label with release time ≤ `tau_now` in order.
* `drain` (297–301): releases everything left.
* Stats: `latency_stats` (min/max/mean, fraud median, dispute share), `telemetry_stats` (counts and step-up fail rate), `released_frame` (DataFrame).

## 2. Things to remember
* Equal release times keep submission order (the sequence number breaks ties), so runs are deterministic for a given seed.
* The default maturity policy means *legitimate* labels all arrive at day 14, so early labels are fraud-heavy — an important bias to mention when discussing the online thresholds.
