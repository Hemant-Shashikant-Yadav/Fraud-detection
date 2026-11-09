# Test Walkthrough: `tests/test_phase7_experiments.py`

**What it verifies:** Phase 7 — the experimental evaluation suite:
1. **Drift evaluator** — chronological, step-aligned windows; every model steps through windows in order and logs every metric; no information from a window can affect an earlier one; labels reach online components only after the configured delay.
2. **Ablation runner** — the five ablation points (seven arms) activate exactly the intended components; branched arms share one fitted pipeline; the Analyst-Assist layer never changes a decision.
3. **Adversarial stress test** — 5–25 % poisoning sweep; harmful rules rolled back within the certified number of cycles; Ville's inequality holds.
4. **Statistical testing** — McNemar, Wilcoxon (exact and normal), Holm adjustment and block bootstrap checked against hand-computed or SciPy reference values.

## Test classes
| Class | What it proves |
|---|---|
| `TestSourceResolution` *(added this session)* | `auto` and explicit `paysim`/`ieee_cis` **raise** when the raw file is missing; `prepare_splits("auto")` never returns synthetic data; explicit `synthetic` is allowed. |
| `TestDriftWindowsAndMetrics` | Fifty step-aligned windows cover the test split; invalid partitions rejected; chronological split order and classes; camouflage drift only moves *later* fraud; hand-computed classification metrics; the operating threshold maximises F1. |
| `TestDriftEvaluator` | Steps through every window in order; logged metrics valid; predictions cover each test transaction exactly once; **no leakage** from a truncated stream, nor from future features and labels; label-release schedule; `run_drift_evaluation` writes its outputs. |
| `TestAblationComponents` | The assist layer is causal and read-only; recovery accounting rules. |
| `TestAblationRunner` | Five-point matrix written; component switches; static vs adaptive conformal control; full-pipeline recovery; the assist layer is decision-independent. |
| `TestAdversarialStressTest` | Sweep covers all rates and policies; false admission; rollback cycles measured; **Ville's maximal inequality**; exact binomial helpers. |
| `TestStatisticalSignificance` | McNemar hand-computed, degenerate and Haldane cases; χ² survival function; Wilcoxon exact matches a reference, ties/zeros/normal approximation; Holm–Bonferroni; block-bootstrap CI; full report. |

## Why it matters
This is the safeguard behind the benchmark's integrity: the leakage tests and the label-delay tests are what let the paper claim a *prequential, leak-free* evaluation.
