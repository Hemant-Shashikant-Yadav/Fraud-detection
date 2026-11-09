# Code Walkthrough: `src/closed_loop/feedback_adapter.py`

**Purpose:** The "closed loop". It takes verified fraud labels that arrive *late* (1–14 days after a decision) and uses them to (1) recalibrate the conformal thresholds, (2) update the martingale gate that admits or rolls back rules, and (3) decide when to trigger incremental retraining.

**Why it matters:** this is where delayed ground truth enters MG-ECFM without ever leaking into the decision that has already been made.

---

## 1. Section-by-section

### 1.1 Docstring (lines 1–27)
Three consumers of released labels:
1. **CS-ACRC recalibration** — each label `(action, y, amount)` updates the thresholds `λ_{t+1} = Π[λ_t + γ(α − ℓ_t)]`. The adaptive-conformal telescoping guarantee does not depend on *when* a label is observed, so delay keeps the long-run risk bound.
2. **Knowledge-evolution gate** — outcomes of rules that fired are buffered and fed to `run_update_cycle` once per gate interval (admission wealth for quarantined rules, rollback wealth for active rules, Ville threshold `1/α`).
3. **Retrain schedule** — a `RetrainTrigger` is emitted on label volume, risk drift, or a rollback, subject to a cool-down.
Immediate telemetry (step-up pass/fail, blocks) is logged but never treated as a label.

### 1.2 Imports and constants (lines 28–48)
Pulls in `OnlineAdaptiveCRC`, `Action`, and the queue types. Reason codes: `label_volume`, `risk_drift`, `rule_rollback`.

### 1.3 `AdapterConfig` (lines 51–90)
Frozen dataclass: gate interval (6 h), minimum labels/frauds for retraining (500 / 10), cooldown (24 h), drift tolerance (0.01), minimum samples before drift checks (200), retrain-on-rollback flag. `__post_init__` validates. `from_config` reads the `closed_loop` section of `configs/hyperparams.json` and maps its key names onto these fields, with keyword overrides.

### 1.4 Record types (lines 93–133)
* `RetrainTrigger` – time, reasons, counts, rolling loss, rolled-back rule ids.
* `ThresholdUpdate` – audit row for each recalibration (before/after λ).
* `FeedbackStep` – everything that happened in one `advance()` call (released labels, λ before/after, gate report, wealth snapshot, trigger).

### 1.5 `ClosedLoopAdapter` (lines 136–306)
* `__init__`: holds the CRC controller, optional gate, queue, config, callbacks (`on_retrain`, `on_fraud_confirmed`) and counters/buffers (pending rule outcomes, labels since last retrain, rollbacks since last retrain, logs).
* `from_config` (161–169): builds CRC, queue and config from the project JSON files.
* `record_decision` (172–177): submits a decision to the queue (label withheld until its delay elapses).
* `route_and_record` (179–185): routes a score with the **current** λ via `crc.action_for_score`, then records it — so the action is decided before any label is known.
* `_apply_label` (188–200): applies a released label to the CRC (`crc.update`), logs the threshold change, buffers `y` for each rule that fired, updates counters, and fires `on_fraud_confirmed` for frauds.
* `flush_gate` (202–214): sends the buffered outcomes to `gate.run_update_cycle`, records the report and any rollbacks; no-op if there is no gate.
* `advance(tau_now)` (216–229): the main clock tick — release due labels, apply them in release order, run a gate cycle if one is due (or forced), check the retrain rules, and return a `FeedbackStep`.
* `drain` (231–236): end of stream; release everything and force a gate cycle.
* `_check_retrain` (239–262): triggers on volume (enough labels and frauds), drift (rolling loss above α + tolerance after enough samples), or rollback. Cool-down suppresses everything *except* rollbacks. On trigger it logs, resets the counters and calls the callback.
* Views: `wealth(rule_id)` (admission wealth if quarantined, rollback wealth if active), `wealth_snapshot`, `threshold_frame` (DataFrame of all updates, with λ columns per tier), `summary` (counts of labels, gate cycles, rollbacks, promotions, triggers, thresholds, telemetry).

## 2. Things to remember
* Labels are applied in **release order**, which preserves causality.
* The retrain trigger only *signals*; it does not itself retrain anything (the caller supplies `on_retrain`).
