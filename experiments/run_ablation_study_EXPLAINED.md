# Code Walkthrough: `experiments/run_ablation_study.py`

**Purpose:** A systematic **5-point ablation** of MG-ECFM: it switches components on and off to measure what each contributes. All arms use the *same* chronological stream, the *same* 50 windows, the *same* contaminated candidate-rule feed and the same causal label release as the drift evaluation.

**Outputs:** `reports/ablation_study_results.csv` (pooled summary per arm) and `ablation_study_windows.csv` (per window).

---

## 1. The arms (docstring)
| Arm | Configuration |
|---|---|
| `B1_xgboost_only` | Causal tabular features → cost-sensitive XGBoost, static Platt calibration, approve/block at `τ`. No graph, gate or recovery. |
| `B2_graph_gate_off` | XGBoost + camouflage GraphSAGE + DAE + `φ_KB`, evidence gate **OFF** (Auto-Admit: every candidate rule enters the KB), CS-ACRC routing. |
| `B3_static_crc` | Full gated pipeline, thresholds learned on the operating split then **frozen**. |
| `B3_adaptive_crc` | Same fitted pipeline with online adaptive CS-ACRC. |
| `B4_full_mg_ecfm` | Gate ON + adaptive CS-ACRC + **TD-TPF recovery**. |
| `B5_assist_off` / `B5_assist_on` | B4 without / with the Analyst-Assist layer. |

B3-static, B3-adaptive, B4 and B5 **branch from one fitted pipeline** (`clone_system`), so their differences are caused only by the ablated component — which is also why they share PR-AUC/F1 by construction. TD-TPF recovery is post-incident: it changes the *realised monetary outcome* of frauds that slipped through, not any authorisation decision.

## 2. Section-by-section (by outline)

### 2.1 Analyst-Assist layer (lines 126–288)
* `AssistExplanation`: nearest verified cases, neighbour fraud share, firing rules with their Subjective-Logic probabilities, most anomalous features.
* `AnalystAssistLayer`: **retrieval-based** case-evidence assistant. For each transaction routed to Hold-and-Verify it retrieves the `k` (=10) most similar *already-verified* cases (cosine similarity over signed-log standardised features; the index only holds cases whose labels were **released**) and builds an explanation. Methods: `fit`, `_embed`, `ingest` (adds newly released cases), `explain`.
* `AssistedSystem`: wraps `MGECFMSystem` so assist runs **after** the routing decision — it cannot influence the action (the test shows identical decisions with assist on/off).

### 2.2 TD-TPF recovery accounting (lines 291–358)
`recovery_accounting(predictions, canonical, …)`: for frauds not intercepted before settlement, runs the TD-TPF tracer and recovery simulator at the detection delay and returns recovered value, prevented-loss ratio and net loss per arm.

### 2.3 Knowledge-base diagnostics (lines 364–384)
`kb_diagnostics`: active rules at the end, rules ever admitted, the share of admitted rules whose test-period precision is ≤ `p0` ("admitted bad rate", rules with ≥ 5 firings only) and the mean precision of active rules.

### 2.4 Runner (lines 390–600)
* `pooled_metrics`: pooled PR-AUC, F1, F2, Brier, EML, net monetary loss, prevented-loss ratio.
* `triage_accuracy`: on the hold queue, accuracy of score-only triage (`p ≥ 0.5`) versus evidence-assisted triage (average of `p` and the neighbour fraud share), plus queue size.
* `ArmRun` (container), `build_arm_system` (constructs the arm's `MGECFMSystem` configuration).
* `run_ablation_study(...)`: prepares splits (using the same `prepare_splits`, so the **no-silent-synthetic guard** applies), fits the needed pipelines, evaluates each arm window by window, computes accounting and diagnostics, and writes the CSVs.
* `main`: CLI (`--source`, `--profile`, `--n-windows`, `--poison-rate`, `--label-delay-windows`, `--max-rows`, ...).

## 3. What the PaySim 2M results say (for context)
XGBoost-only has the highest PR-AUC (0.336 vs 0.283); the gate-ON arms end with **0 active rules**, so B3/B4/B5 are numerically identical; B2 (gate off, all rules auto-admitted) has the best monetary loss. These results do not yet support the claim that each novel component adds measurable value — a key point for the paper.
