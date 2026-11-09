# Test Walkthrough: `tests/test_phase5_recovery_closedloop.py`

**What it verifies:** Phase 5 — Stage 5 (**TD-TPF recovery**) and Stage 6 (**closed-loop delayed feedback**).
1. TD-TPF computes exact proportional taint across a 3-hop mule chain (hand-computed), stops at `CASH_OUT`/external wires, conserves mass, and builds identical graphs from PaySim, AMLSim and canonical frames.
2. Taint decays as `exp(−κ(τ'−τ))` per lot; `κ = 0` gives pure proportional flow; decay is monotone in κ and time.
3. The simulator reports exact Recovered Amount, Recovery Rate, Prevented-Loss Ratio and net savings inside the 24 h window, follows funds moved during reaction latency, and recovers nothing after the window.
4. The verification queue hides labels until `τ + Δτ` (1–14 days) and releases them chronologically.
5. The closed-loop adapter updates CS-ACRC thresholds and martingale wealth only on release, and schedules retraining.

## Test classes
| Class | What it proves |
|---|---|
| `TestTaintedProportionalFlow` | Three-hop chain hand-computed; commingled clean inflow dilutes the share; **lots decay from their own arrival time**; tracking stops at cash-out and external wire; window and causality; PaySim/AMLSim/canonical frames agree; inflow and opening balances; `max_hops` and `min_taint` pruning; **conservation and bounds on random networks**; `from_config`. |
| `TestTimeDecay` | Single-hop decay is exponential; `κ = 0` = pure proportional flow; monotone in κ and elapsed time; decay telescopes along a full path. |
| `TestRecoverySimulation` | Recovered amount and rate within 24 h; funds moved during reaction latency are followed; nothing recovered outside the window; instrument block and prevented-loss ratio; merchant chargeback; full-account freeze collateral; stochastic mode reproducible; **recovery non-increasing in detection delay**; `simulate_many` and `aggregate`. |
| `TestVerificationQueue` | Latency bounds and sources; labels held and released chronologically; equal release times keep submission order; clock and input guards; immediate telemetry is separate from labels; deterministic with a seed. |
| `TestClosedLoopAdapter` | λ updated **only on release** with the exact ACI step; martingale wealth updated on release; a degraded rule is rolled back and retraining scheduled; gate cycles respect the interval; retrain volume/drift triggers with cool-down; end-to-end delayed stream; `from_config` and deterministic replay. |

## Why it matters
Supports the recovery claims and shows that delayed labels never leak into earlier decisions. The recovery *rates* themselves are simulation results that depend on the configured latencies and success rates.
