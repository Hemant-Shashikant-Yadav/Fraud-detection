# Test Walkthrough: `tests/test_phase6_baselines.py`

**What it verifies:** Phase 6 — the four re-implemented baselines (BP1 EARN+RXT-J, BP2 Lambda GNN+AE, BP3 CARE-GNN, BP4 ARGUS) share one interface, train on synthetic and Phase-1 causal features, emit valid probabilities, report latency percentiles, and their key mathematical components behave as specified.

## Test classes
| Class | What it proves |
|---|---|
| `TestBaselinesTrainAndPredict` | `fit` returns self; `predict_proba` is a valid distribution (rows sum to 1); `predict` equals thresholded probability; models **learn the fraud signal** on synthetic data; inference is deterministic; training is reproducible with a seed; single-row scoring; fit reports are filled. |
| `TestLatency` | `measure_latency` returns p50/p95/p99/mean/max; default `n_runs` and batched mode; argument validation. |
| `TestPhaseFeatureCompatibility` | The Phase-1 frame has both classes in the training window; baselines train on Phase-1 causal features and on the Phase-2 fusion width; CARE-GNN relation views cover all columns. |
| `TestUnifiedInterface` | Registry and signatures; factory aliases and per-baseline config routing; unfitted-model and invalid-input errors; lists and pandas accepted. |
| `TestComponents` | **SMOTE** interpolates between fraud neighbours; **Jaya** minimises the sphere function; **CRC threshold** hand-computed and **controls expected FNR**; CARE RL threshold policy; CARE top-p keeps the most similar neighbours; SSM kernel and causal convolution; ARGUS MC-dropout and selective prediction; BP2 speed layer flags out-of-distribution rows; k-NN excludes self even with duplicate rows; shared preprocessing helpers. |

## Why it matters
Shows the comparison baselines are *faithful and fair* (same interface, same preprocessing, same seeds) — a likely reviewer question.
