# Test Walkthrough: `tests/test_phase2_graph_detection.py`

**What it verifies:** Phase 2 — temporal memory, the camouflage similarity filter, filtered GraphSAGE, the denoising-autoencoder residual, cost-sensitive calibrated XGBoost (isotonic and **exact** Venn-Abers), the fusion-vector layout, and the **≤ 25 ms** real-time latency budget.

## Test classes
| Class | What it proves |
|---|---|
| `TestTemporalMemory` | Decay follows `exp(−δ·Δτ)`; unseen/stale nodes read zero; **batch commit equals sequential event-by-event updates**; events in a batch do not see each other; gradients reach `W_mem` but not the stored memory; read/update cost is constant; export/load round-trips including the scaler. |
| `TestSimilarityFilter` | `S(u,w) = sigmoid(W|s_u−s_w|)` formula; threshold pruning removes camouflaged edges; label-aware training *breaks* camouflage (cross-class similarity falls); top-p and `min_neighbors_kept` modes; `rank_within_centre`. |
| `TestCamouflageGraphSAGE` | SAGE layer formula; **PyG and the pure-PyTorch fallback agree**; filtered message passing ignores camouflage; stream embeddings are **causal**; within-step permutation invariance; `fit_stream` trains every component; single-event latency under budget. |
| `TestAutoencoderResidual` | The residual equals `‖x − g(f(x))‖₂`; anomalies have larger residuals than normal rows. |
| `TestTabularEngine` | Cost-sensitive weighting; calibrated risk bounded in [0,1] and monotone; calibration improves Brier; Venn-Abers interval; **Venn-Abers matches a brute-force implementation exactly**. |
| `TestFusionClassifier` | Fusion vector layout `[x ‖ h_u ‖ h_v ‖ r_AE ‖ φ_KB]`; KB encoder gives the same result for frame and single-event paths; calibrated risk bounded on a future split; end-to-end and full-size latency under budget; leakage columns and overlapping splits are rejected. |

## Why it matters
Supports the claims on camouflage resistance, causal embeddings, calibration quality and real-time latency.
