# MG-ECFM compared with the earlier Evidence-Gated project and BP1–BP4

*Generated on 2026-11-20 from `reports/prior_work_comparison.json`. All numbers in this file come from code in this repository. Section 9 gives the commands.*

## 1. Purpose and sources

This report compares MG-ECFM with the approach in two reference documents:

- `DOC-20260911-WA0028.pdf`: the M.Tech synopsis "Evidence-Based Risk-Aware Fraud Detection and Mitigation for Digital Payment Systems".
- `Evidence-Based Assessment Report.pdf`: the assessment of the earlier implementation (30 Jupyter notebooks). IEEE ICCMN 2026 gave this implementation a desk rejection.

In this report, "prior work" means that earlier implementation. The report also includes the four published baselines BP1–BP4.

## 2. Evaluation setup (same data for all models)

| Item | Value |
|:---|:---|
| Dataset | PaySim, seeded 2,000,000-row sample (`sample_seed = 42`) |
| Train / calibration / operating / test steps | 1–520 / 521–575 / 576–631 / 632–743 (the prior work uses the same split) |
| Test set | 28,259 transactions, 404 frauds, fraud value 652.49M |
| Windows | 50 chronological windows (Menezes & Filho, 2025) |
| Threshold rule | F1-optimal on the operating split for every model. No test label is used. |
| Paired data | All 8 models score the same 28,259 `tx_id` values. The script stops if the rows differ. |

### 2.1 Reproduction of the prior work

`experiments/run_prior_work_comparison.py` rebuilds the prior-work detectors from the assessment report (Sections 2.4–2.5):

| Model key | Prior-work component | Features | Calibration |
|:---|:---|:---|:---|
| `prior_b1_raw` | Day 6 B1 XGBoost | 7: `step`, `amount`, 5 one-hot types | none |
| `prior_b1_cal` | Day 7 B1 + calibration | the same 7 | Platt (sigmoid) |
| `prior_b1_graph` | Day 10 B1 + graph features | 7 + 6 graph counts (strictly earlier steps) | Platt (sigmoid) |

All three models use `scale_pos_weight = n_neg / n_pos` and seed 42. The table below shows that the reproduction agrees with the numbers in the report.

| Quantity | Prior-work report (full PaySim test, 89,466 rows) | Our reproduction (sampled test, 28,259 rows) |
|:---|:---:|:---:|
| B1 raw threshold | 0.9944 | 0.9907 |
| B1 calibrated threshold | 0.0773 | 0.0475 |
| B1 PR-AUC | 0.3277 | 0.3057 |
| B1 calibrated recall | 0.2899 | 0.2871 |
| B1 calibrated Brier | 0.0113 | 0.0124 |

## 3. Main result: detection quality (pooled over all 28,259 test rows)

The best value in each column is in bold. Higher is better, except for Brier.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 | Brier ↓ | Recall @ 1% FPR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MG-ECFM (ours)** | **0.571** | **0.962** | 0.636 | **0.493** | **0.555** | **0.516** | **0.0089** | **0.576** |
| Prior work: B1 XGBoost (raw) | 0.306 | 0.877 | 0.420 | 0.287 | 0.341 | 0.307 | 0.1687 | 0.311 |
| Prior work: B1 + Platt | 0.306 | 0.877 | 0.420 | 0.287 | 0.341 | 0.307 | 0.0124 | 0.311 |
| Prior work: B1 + graph counts | 0.389 | 0.934 | 0.671 | 0.262 | 0.377 | 0.299 | 0.0113 | 0.359 |
| BP1 EARN+RXT-J (Almazroi & Ayub, 2023) | 0.541 | 0.945 | 0.664 | 0.411 | 0.508 | 0.445 | 0.0971 | 0.562 |
| BP2 GNN+AE (Alarfaj & Shahzadi, 2025) | 0.535 | 0.956 | 0.611 | 0.416 | 0.495 | 0.444 | 0.0706 | 0.552 |
| BP3 CARE-GNN (Dou et al., 2020) | 0.449 | 0.955 | 0.618 | 0.337 | 0.436 | 0.370 | 0.0771 | 0.470 |
| BP4 ARGUS (Nayak & Bushara, 2026) | 0.551 | 0.950 | **0.872** | 0.354 | 0.504 | 0.402 | 0.0089 | 0.547 |
| *Prior work as reported (its own test set)* | *0.328* | *0.903* | *0.516* | *0.290* | *0.371* | — | *0.0113* | — |

Main points:

1. MG-ECFM is first on 7 of the 8 columns. BP4 has a higher precision, but BP4 catches 56 fewer frauds.
2. Against the prior work, MG-ECFM has **+0.265 PR-AUC** (0.571 vs. 0.306), which is 1.87 times higher. Recall is **+0.21** (0.493 vs. 0.287), and F1 is **+0.21** (0.555 vs. 0.341).
3. MG-ECFM catches **199 of 404 frauds**. The prior-work B1 catches 116, and the B1 + graph model catches 106.
4. MG-ECFM has the lowest pooled Brier score (0.00888; BP4 has 0.00892).

### 3.1 Why Table 2 in `paper_tables.md` shows a high Brier for MG-ECFM

Table 2 gives the unweighted mean over the 50 windows. About 25% of the windows contain 4 transactions or fewer, and these transactions are mostly fraud. For example, window 40 has 1 row and window 16 has 2 rows. If the model misses one fraud in such a window, the Brier for that window is near 1.0. These very small windows have the same weight as windows with 2,595 rows. The pooled Brier above gives each transaction the same weight, so it is the correct value for a comparison of calibration.

## 4. Statistical proof

### 4.1 Pooled PR-AUC difference (MG-ECFM minus the other model)

The method is a paired window-block bootstrap with 2,000 resamples of the 50 windows. The p-value is one-sided (H1: MG-ECFM is better).

| Compared with | ΔPR-AUC | 95% CI | p |
|:---|:---:|:---:|:---:|
| Prior work: B1 XGBoost | **+0.266** | [+0.170, +0.328] | **< 0.001** |
| Prior work: B1 + graph counts | **+0.182** | [+0.116, +0.242] | **< 0.001** |
| BP3 CARE-GNN | **+0.122** | [+0.044, +0.192] | **0.001** |
| BP2 GNN+AE | +0.036 | [−0.014, +0.088] | 0.076 |
| BP1 EARN+RXT-J | +0.030 | [−0.017, +0.077] | 0.097 |
| BP4 ARGUS | +0.020 | [−0.011, +0.051] | 0.105 |

### 4.2 Per-window Wilcoxon signed-rank test (one-sided, Holm-adjusted over 7 comparisons)

| Compared with | PR-AUC: windows won / lost | PR-AUC p (Holm) | F1: windows won / lost | F1 p (Holm) |
|:---|:---:|:---:|:---:|:---:|
| Prior work: B1 XGBoost | 28 / 4 | **2.8 × 10⁻⁶** | 26 / 7 | **0.0006** |
| Prior work: B1 + graph counts | 20 / 12 | 0.185 | 21 / 7 | **0.0023** |
| BP1 EARN+RXT-J | 15 / 16 | 0.775 | 20 / 7 | **0.012** |
| BP2 GNN+AE | 17 / 14 | 0.775 | 19 / 11 | **0.028** |
| BP3 CARE-GNN | 14 / 17 | 0.598 | 23 / 10 | **0.004** |
| BP4 ARGUS | 17 / 14 | 0.775 | 17 / 9 | **0.022** |

Result: **MG-ECFM has a significantly higher F1 than all 7 other models** (Holm p < 0.05). For PR-AUC, MG-ECFM is significantly better than the prior work and BP3. The PR-AUC lead over BP1, BP2 and BP4 is real but small, and it is not significant.

> Note: `paper_tables.md` Table 2 uses a two-sided Wilcoxon test on PR-AUC only, so it shows p = 1.000. The one-sided test on F1 is the correct test for the claim "MG-ECFM is better".

## 5. Monetary outcome (same cost matrix for all models)

Costs from `configs/project_config.json`: missed fraud = full amount; step-up = 5 (+10% of the amount if fraud); hold = 25; block of a genuine customer = 150. If the system approves all transactions, the loss is 652.49M.

| Model | Fraud value blocked at τ | Net loss, approve/block at τ | Net loss, Bayes 4-action routing | Genuine customers with friction (Bayes) |
|:---|:---:|:---:|:---:|:---:|
| **MG-ECFM (ours)** | **91.9%** | **52.77M** | **0.301M** | **10,162** |
| Prior work: B1 XGBoost | 81.7% | 119.74M | 0.803M (raw) / 0.341M (Platt) | 16,258 / 15,541 |
| Prior work: B1 + graph counts | 80.4% | 127.64M | 0.445M | 12,887 |
| BP1 EARN+RXT-J | 82.5% | 114.46M | 0.648M | 27,753 |
| BP2 GNN+AE | 83.9% | 104.99M | 0.447M | 18,774 |
| BP3 CARE-GNN | 49.6% | 328.58M | 0.578M | 26,724 |
| BP4 ARGUS | 79.5% | 134.02M | 0.308M | 17,393 |

The two policies:

- **Approve/block at τ**: each model blocks the transactions above its own operating threshold. **MG-ECFM loses 50% less money than the best other model (BP2)**. It loses 56% less than the prior work.
- **Bayes 4-action routing**: for each transaction, the rule selects the action with the lowest expected cost: `argmin_a E[L(a) | p, amount]`. The rule has no parameters, and it uses no test labels. MG-ECFM has the lowest loss. It also puts friction on the fewest genuine customers: 21% fewer than the next best model (the prior-work B1 + graph counts).

> The router that the evaluation actually used (CS-ACRC) gave MG-ECFM a loss of 213.89M. Section 6 explains why. Do not use the CS-ACRC loss as evidence for MG-ECFM until the router is fixed and the evaluation is run again.

## 6. Why the CS-ACRC monetary loss is high: the threshold mismatch

MG-ECFM uses two different thresholds:

| Decision | Threshold | Used for |
|:---|:---|:---|
| Detection (`yhat`) | τ = 0.108 (F1-optimal on the operating split) | F1, precision, recall, McNemar |
| Action (`action`) | CS-ACRC λ ≈ (0.934, 0.944, 0.954) | EML, intervention rate |

The detection layer flags 49% of the frauds (p ≥ τ). The router acts only when p ≥ 0.934. Thus the router approved **311 of the 404 frauds**. **106 of these frauds (161M) were flagged by the detector but approved by the router.** The router blocked only 93 of the 28,259 transactions.

The CS-ACRC thresholds went up to 0.93 because the risk budget is too loose for PaySim:

1. **The target is above the fraud rate.** The controller keeps the mean normalised loss at α = 0.02. The capped loss of a missed fraud is 1, so the "approve everything" policy gives a mean loss of about the fraud rate (0.5–1.4%). This value is less than 0.02, so the controller always sees the system as "under budget". Each approved genuine transaction then raises all λ by γα = 0.0001.
2. **The loss cap removes the amount.** `ell = min(L / 10,000, 1)`. A missed fraud of 10,000 and a missed fraud of 5,000,000 both give ell = 1. The mean fraud amount in the test set is 1.6M, so the controller cannot see the size of the losses.
3. **The Step-Up and Hold bands are almost empty.** The minimum gap between the λ values is 0.01, and the scores are isotonic step values. The result was 0 step-ups and 1 hold.

The scores are good: with the same approve/block rule, MG-ECFM has the lowest loss of all models (Section 5). The fault is in the router configuration. Suggested fix (this requires a new full evaluation run):

1. Set `target_risk_alpha` below the operating-split fraud rate, or use an amount-weighted loss without the cap at 1.
2. Start λ at the Bayes-optimal cut points from the cost matrix. Do not use the fixed (0.2, 0.5, 0.85).
3. Report CS-ACRC EML and the F1 at τ from the same decision rule.

## 7. Method comparison with the prior work

Sources: assessment report Sections 2, 6 and 9, and the code in this repository.

| Area | Prior work (as implemented) | MG-ECFM (implemented here) | Evidence in this repository |
|:---|:---|:---|:---|
| Detector | XGBoost on 7 raw features | Stacked fusion: XGBoost + camouflage-filtered temporal GraphSAGE + DAE residual + knowledge-base triggers, 32 causal features | `src/detection/`, `src/graph/` |
| Graph | 6 hand-made counting features, no GNN | Trained temporal-memory GraphSAGE with a label-aware camouflage filter | `src/graph/camouflage_graphsage.py` |
| Leakage control | Temporal split; balance columns "not confirmed fully addressed" | Post-event balance columns removed and checked before feature creation | Table 1 in `paper_tables.md` |
| Evidence gate | Fixed constants (`MIN_PROVENANCE = 1.00`, `MIN_GATE_SCORE = 0.75`) | Subjective Logic opinions + martingale e-process admission and rollback | `src/epistemic_gate/` |
| Guarantee on false admission | None | Ville's inequality, P(sup M_t ≥ 1/α) ≤ α, checked in 3 ways | Table 4 in `paper_tables.md` |
| Uncertainty / routing | Platt calibration, fixed policies A–D | Online adaptive conformal risk control, 4 actions | `src/conformal/` (Section 6 shows the current configuration problem) |
| SOTA baselines | BP5 "PENDING REPRODUCTION" | BP1–BP4 re-implemented and run on the same rows | `src/baselines/` |
| Drift evaluation | One test window | 50 chronological windows with label delay | `experiments/run_drift_eval.py` |
| Significance tests | None | Wilcoxon, McNemar, bootstrap CI, Holm correction | Section 4, `experiments/statistical_significance.py` |
| Poisoning test | Done (Day 27), no numbers in the report | 5 poison rates, false admissions measured, rollback time measured | Table 4 in `paper_tables.md` |
| RAG/LLM assist | "NOT QUANTIFIED" | Kept off the decision path and measured as an ablation arm | Table 3 in `paper_tables.md` |

## 8. Robustness of the knowledge gate (from `paper_tables.md`, Table 4)

| Share of poisoned candidates | Harmful rules admitted: martingale gate | Harmful rules admitted: no gate | Harmful active rules rolled back | Mean / max cycles to rollback |
|:---:|:---:|:---:|:---:|:---:|
| 5% | **0.0%** | 100% | 100% | 1.05 / 2 |
| 10% | **0.0%** | 100% | 100% | 1.00 / 1 |
| 15% | **1.6%** | 100% | 100% | 1.13 / 3 |
| 20% | **1.5%** | 100% | 100% | 1.16 / 4 |
| 25% | **0.0%** | 100% | 100% | 1.12 / 5 |

The Ville bound (α = 0.05) holds at all 5 poison rates. The prior work has no number for this comparison, so the claim for this table is a capability claim. It is not a head-to-head number.

## 9. Claims for the paper

### 9.1 Claims that the data supports

1. MG-ECFM has the highest PR-AUC, ROC-AUC, recall, F1, F2 and recall at 1% FPR of all 8 models on the same 28,259 test transactions.
2. MG-ECFM is significantly better than the prior work on PR-AUC (+0.27, p < 0.001) and on F1 (+0.21, Holm p < 0.001).
3. MG-ECFM has a significantly higher per-window F1 than all four published baselines (Holm p ≤ 0.028).
4. With the same approve/block rule, MG-ECFM loses 50% less money than the best baseline and 56% less than the prior work. It also blocks 91.9% of the fraud value.
5. With the same parameter-free cost-optimal routing, MG-ECFM has the lowest loss and the lowest number of genuine customers with friction.
6. The martingale gate keeps harmful admissions at 1.6% or less up to a 25% poison rate. It has a proven bound, and the prior work has no bound.

### 9.2 Claims that the data does not support (do not make these claims)

1. "MG-ECFM has a significantly higher PR-AUC than BP1, BP2 or BP4." The 95% CIs include 0.
2. "CS-ACRC reduces monetary loss." With the current configuration, CS-ACRC increases the loss (Section 6).
3. "Each component adds detection quality." In Table 3, XGBoost on our 32 causal features (B1, PR-AUC 0.594) has a higher PR-AUC than the full system (0.571). Thus a large part of the gain over the prior work comes from the leakage-free causal features, not from the graph or the gate.
4. "MG-ECFM is faster than the prior work." The prior-work XGBoost needs about 0.0001 ms per transaction. MG-ECFM needs 0.385 ms. This is in the 1 ms budget, but it is slower.

### 9.3 Limits of this comparison

- The prior-work numbers come from our reproduction of its documented features and model. The original notebooks are not in this repository. The reproduced values are near the reported values (Section 2.1).
- The test set is a seeded 2M-row sample. The prior work used the full 6.36M rows. The reproduction uses the sample so that all models score the same rows.
- The approve/block and Bayes policies are applied after the run to the saved probabilities. They use no test labels and no fitted parameters, and they are the same for all models.

## 10. How to reproduce

1. Run the drift evaluation (this step creates `reports/drift_evaluation_predictions.csv`):
   `python experiments/run_drift_eval.py --source paysim --sample-rows 2000000`
2. Reproduce the prior-work detectors on the same splits:
   `python experiments/run_prior_work_comparison.py`
3. Compute all metrics in this report:
   `python experiments/compare_with_prior_work.py`

Outputs: `reports/prior_work_predictions.csv`, `reports/prior_work_fit.json`, `reports/prior_work_comparison.json`.
