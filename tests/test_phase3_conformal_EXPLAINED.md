# Test Walkthrough: `tests/test_phase3_conformal.py`

**What it verifies:** Phase 3 — Stage 1 of MG-ECFM, **CS-ACRC** and pre-transaction routing:
1. the asymmetric loss is exact on all 8 (action × outcome) cells, scalar and vectorised;
2. Hold-and-Verify on a fraud saves `A_t` and costs only `c_hold`;
3. the online update follows `θ_{t+1} = θ_t − γ(α − ℓ)` and thresholds tighten/relax as the fraud rate drifts;
4. empirical loss stays within `α ± ε` (telescoping certificate, one-sided bound, rolling-window bound);
5. `λ_low < λ_mid < λ_high` is strictly preserved by an exact Euclidean projection;
6. routing takes < 1 ms per transaction.

## Test classes
| Class | What it proves |
|---|---|
| `TestLossFunction` | All eight cells; loss matrix; vectorised = scalar; amount dependence; input validation; `from_project_config`; expected loss and Bayes action. |
| `TestHoldAndVerify` | A held fraud costs only the review fee; a portfolio of held frauds; Hold beats Step-Up on a large fraud. |
| `TestAdaptiveDrift` | A single update matches the synopsis equation exactly; thresholds **track fraud-rate drift**; a low base rate relaxes thresholds to the permissive extreme; rolling fraud rate follows drift; `static` mode is frozen; warm-up freezes thresholds; `from_config`. |
| `TestLossBound` | Cumulative loss within `α ± ε`; one-sided risk bound with rare fraud; rolling window within bound after drift; **static thresholds break under drift** (motivating adaptivity); the certificate holds for an **adversarial** sequence; **delayed feedback preserves the certificate**; feasibility report. |
| `TestMonotonicity` | Strict ordering under aggressive updates; projection equals the exact Euclidean projection; unordered initial values are projected; invalid constraints rejected; state round-trip and rollback. |
| `TestRouter` | Threshold regions; epistemic-uncertainty handling (uncertain Block → Hold); KB rule floor; amount-aware escalation; tuple and scored-event inputs; input validation; feedback updates the controller; `from_config`; **sub-millisecond routing** and route+feedback within budget; deterministic replay. |

## Why it matters
These tests are the evidence for the **long-run risk-control** claim — note it is an *average* loss guarantee, not a per-transaction one.
