# Test Walkthrough: `tests/test_phase4_epistemic_martingale.py`

**What it verifies:** Phase 4 — Stage 3 of MG-ECFM: **Subjective-Logic fusion** and the **anytime-valid martingale e-process gate** with its versioned SQLite knowledge store.
1. Subjective Logic preserves `b + d + u = 1`; trust discount inflates `u`; freshness composes with provenance; cumulative consensus = evidence addition; cluster-aware fusion neutralises Sybil amplification.
2. Admission promotes high-precision rules once `M_t ≥ 1/α_admit` and never promotes useless ones.
3. Rollback removes injected poisoned/degraded rules within ≤ 3 cycles (Monte-Carlo + exact DP certificate **for a certified firing volume**).
4. Ville's inequality: empirical false-admission / false-rollback frequencies stay below α and match the exact DP value.
5. The knowledge store creates snapshots, commits atomically, rolls back, survives injected faults and concurrent writers.

## Test classes
| Class | What it proves |
|---|---|
| `TestSubjectiveLogic` | Evidence mapping matches the synopsis; additivity under every operator; validation; trust discount; time decay and composition; hand-computed consensus; consensus algebra and corroboration; dogmatic/vacuous limits; averaging fusion is idempotent; serialisation; the fusion engine separates feed categories; **cluster-aware fusion counters Sybil amplification**. |
| `TestEProcessMathematics` | Wealth equals the synopsis product; **supermartingale under the composite null**; minimum evidence to cross; exact crossing probability matches Monte-Carlo. |
| `TestAdmissionGate` | Promotes true high-precision rules; rarely promotes useless ones; plug-in betting is valid and powerful; promotion into the active KB comes with a certificate; firing budget rejects undecided candidates; end-to-end false-admission and accuracy on the feed. |
| `TestRollbackGate` | Certified firing volume for 3 cycles; injected poisoned rules rolled back within 3 cycles (at that volume); a degraded admitted rule is rolled back; **martingale delay grows with good history** (motivating the CUSUM e-detector); rollback restores snapshot `V_j`. |
| `TestVilleInequality` | False-admission and false-rollback frequencies bounded by α; the bound holds across sequential gate tests; e-detector average run length. |
| `TestKnowledgeStore` | Schema; snapshots and commits; atomic rollback; **fault injection leaves no partial state**; JSON constraints; thread-safe concurrent writers; active rules feed the router and the fusion encoder. |
| `TestConfigAndDeterminism` | `from_config`; deterministic replay. |

## Why it matters
Evidence for the paper's guarantee statements. Note what is *not* proven here: the "≤ 3 cycles" result holds **at a certified firing volume**, not unconditionally.
