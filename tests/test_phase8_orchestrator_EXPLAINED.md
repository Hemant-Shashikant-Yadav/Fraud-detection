# Test Walkthrough: `tests/test_phase8_orchestrator.py`

**What it verifies:** Phase 8 — the central orchestrator, camera-ready artifacts and the dashboard.
1. `python main.py --mode quick-demo` runs end to end as a subprocess, exits 0, and every internal check in its manifest passes (all five models predict every test transaction; significance/ablation/stress outputs exist; single-transaction latency; camouflage graph; batch scoring; TD-TPF trace; service save/load round trip; runtime budget).
2. Prediction files are well formed (probabilities in [0,1], valid actions, one row per model × test transaction, chronological windows).
3. Tables: LaTeX is structurally balanced with the four labelled tables; Markdown holds the same tables; both can be generated from the published `reports/`.
4. Figures: four PDF + PNG pairs, valid signatures, 300-DPI PNG (pHYs chunk), auditable figure data.

## Test classes
| Class | What it proves |
|---|---|
| `TestQuickDemoEndToEnd` | Exit code 0; all internal checks pass; runtime within budget; every model predicts every test transaction; experiment outputs exist; latency profile recorded; a scored upload strips leakage columns. |
| `TestPaperTables` | Quick-demo tables; tables from the published reports; missing inputs raise. |
| `TestPaperFigures` | Quick-demo figures (formats, resolution, figure data). |
| `TestPredictCLI` | `--predict` writes decisions; rejects a missing file; `--help` works. |
| `TestLiveService` | **What-if scoring is side-effect free**; presets cover the dashboard; graph explanation; KB opinions valid; recovery trace. |
| `TestDashboard` | `app/app.py` compiles; renders headlessly (skipped if Streamlit is not installed). |

## Notes
* This test **runs the synthetic quick demo** (~40 s) by design; it does not need the real datasets.
* It exercises `main.py`, `src/orchestrator/service.py`, `reports/generate_paper_*` and `app/app.py` together, so it is the best single smoke test after any change.
