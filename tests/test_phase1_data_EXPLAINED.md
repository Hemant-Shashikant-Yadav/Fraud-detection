# Test Walkthrough: `tests/test_phase1_data.py`

**What it verifies:** Phase 1 — scaffolding and configuration integrity, **leakage-free ingestion**, strict **temporal causality** of the engineered features, rolling-velocity correctness, chunked-stream exactness, loaders for PaySim / IEEE-CIS / AMLSim, and the adversarial rule-feed generator.

Run: `python tests/test_phase1_data.py` or `python -m pytest tests/test_phase1_data.py -q`.

## Test classes
| Class | What it proves |
|---|---|
| `TestConfigsAndScaffolding` | Seed and temporal-split definitions are consistent; overlapping splits are rejected; `assign_split` works; `hyperparams.json` has all required sections; the scaffolding script builds/verifies the layout. |
| `TestLeakageStripping` | The feature engine **refuses** forbidden balance columns; loaders never emit them; engineered features contain no balance information. |
| `TestTemporalCausality` | Perturbing the *future* never changes earlier features; **within-step permutation invariance**; features are blind to labels; results match a brute-force reference implementation; the first step has empty history; out-of-order input raises. |
| `TestRollingVelocity` | Velocity counts, running moments and ledger-style features equal hand-computed values; `CASH_IN` reverses the flow direction. |
| `TestStreamingExactness` | Chunked streaming gives **exactly** the same features as processing everything at once; the pending buffer holds the last (incomplete) step. |
| `TestRestrictedCopy` | `restricted_copy` (the memory/speed optimisation added in commit `4a15a51`) is **bit-identical** to a full `deepcopy` for single rows and batches. |
| `TestStreamLoaders` | PaySim streaming (including unsorted files), IEEE-CIS sorted by `TransactionDT`, AMLSim v1/v2 schemas, and the `build_loader` + preprocessing pipeline. |
| `TestPerturbationGenerator` | Poisoning rates are honoured, bounds validated, deterministic with a seed, each rule category (genuine, stale, conflicting, poisoned, Sybil) behaves as defined, rules execute on causal features, outcome simulation and serialisation work. |

## Why it matters
These tests are the evidence behind the paper's **leakage-free** claim and the guarantee that streaming equals batch processing.
