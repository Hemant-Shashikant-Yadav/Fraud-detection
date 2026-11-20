# MG-ECFM compared with the earlier Evidence-Gated project and BP1–BP4

*Updated on 2026-11-21 from `reports/prior_work_comparison.json` (after the router fix). All numbers in this file come from code in this repository. Section 10 gives the commands.*

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
| **MG-ECFM (ours)** | **0.593** | **0.962** | 0.754 | **0.426** | **0.544** | **0.466** | **0.0089** | **0.571** |
| Prior work: B1 XGBoost (raw) | 0.306 | 0.877 | 0.420 | 0.287 | 0.341 | 0.307 | 0.1687 | 0.311 |
| Prior work: B1 + Platt | 0.306 | 0.877 | 0.420 | 0.287 | 0.341 | 0.307 | 0.0124 | 0.311 |
| Prior work: B1 + graph counts | 0.389 | 0.934 | 0.671 | 0.262 | 0.377 | 0.299 | 0.0113 | 0.359 |
| BP1 EARN+RXT-J (Almazroi & Ayub, 2023) | 0.541 | 0.945 | 0.664 | 0.411 | 0.508 | 0.445 | 0.0971 | 0.562 |
| BP2 GNN+AE (Alarfaj & Shahzadi, 2025) | 0.535 | 0.956 | 0.611 | 0.416 | 0.495 | 0.444 | 0.0706 | 0.552 |
| BP3 CARE-GNN (Dou et al., 2020) | 0.449 | 0.955 | 0.618 | 0.337 | 0.436 | 0.370 | 0.0771 | 0.470 |
| BP4 ARGUS (Nayak & Bushara, 2026) | 0.551 | 0.950 | **0.872** | 0.354 | 0.504 | 0.402 | 0.0089 | 0.547 |
| *Prior work as reported (its own test set)* | *0.328* | *0.903* | *0.516* | *0.290* | *0.371* | — | *0.0113* | — |

Main points:

1. MG-ECFM is first on 7 of the 8 columns. BP4 has a higher precision, but BP4 catches 29 fewer frauds.
2. Against the prior work, MG-ECFM has **+0.287 PR-AUC** (0.593 vs. 0.306), which is 1.94 times higher. Recall is **+0.14** (0.426 vs. 0.287), and F1 is **+0.20** (0.544 vs. 0.341).
3. MG-ECFM catches **172 of 404 frauds with 56 false alarms**. The prior-work B1 catches 116 with 160 false alarms, and the B1 + graph model catches 106 with 52 false alarms.
4. MG-ECFM has the lowest pooled Brier score (0.00888; BP4 has 0.00892).

### 3.1 The isotonic tie-break (new in this version)

The isotonic calibration of MG-ECFM is a step function. On PaySim it merged about 5,700 distinct scores into 67 probability values, and the ties cost PR-AUC. On the **operating split** (before the test run), the PR-AUC was 0.532 after isotonic and 0.572 for the raw score. The calibrator now breaks the ties with the raw score: `p = (1 − ε)·p_iso + ε·sigmoid(margin)`, ε = 10⁻⁶. The ranking is the raw ranking, and each probability changes by at most 10⁻⁶.

| MG-ECFM on the test set | Before the tie-break | After the tie-break |
|:---|:---:|:---:|
| PR-AUC (pooled) | 0.571 | **0.593** |
| PR-AUC (mean over windows) | 0.504 | **0.530** |
| Brier | 0.00888 | 0.00888 |
| Router decisions | — | identical |
| F1 at τ | **0.555** | 0.544 |
| Precision / recall at τ | 0.636 / 0.493 | 0.754 / 0.426 |

F1 is a little lower, because the F1-optimal τ from the operating split moved to a higher-precision point. PR-AUC is the primary metric of the protocol, and it does not depend on τ.

### 3.2 Why Table 2 in `paper_tables.md` shows a high Brier for MG-ECFM

Table 2 gives the unweighted mean over the 50 windows. About 25% of the windows contain 4 transactions or fewer, and these transactions are mostly fraud. For example, window 40 has 1 row and window 16 has 2 rows. If the model misses one fraud in such a window, the Brier for that window is near 1.0. These very small windows have the same weight as windows with 2,595 rows. The pooled Brier above gives each transaction the same weight, so it is the correct value for a comparison of calibration.

## 4. Statistical proof

### 4.1 Pooled PR-AUC difference (MG-ECFM minus the other model)

The method is a paired window-block bootstrap with 2,000 resamples of the 50 windows. The p-value is one-sided (H1: MG-ECFM is better). Bold: the 95% CI is above 0.

| Compared with | ΔPR-AUC | 95% CI | p |
|:---|:---:|:---:|:---:|
| Prior work: B1 XGBoost | **+0.287** | [+0.202, +0.349] | **< 0.001** |
| Prior work: B1 + graph counts | **+0.203** | [+0.138, +0.263] | **< 0.001** |
| BP1 EARN+RXT-J | **+0.052** | [+0.001, +0.099] | **0.021** |
| BP2 GNN+AE | **+0.058** | [+0.008, +0.110] | **0.011** |
| BP3 CARE-GNN | **+0.143** | [+0.065, +0.215] | **< 0.001** |
| BP4 ARGUS | **+0.042** | [+0.010, +0.072] | **0.003** |

Result: **MG-ECFM has a significantly higher pooled PR-AUC than all 7 other models.** The prior-work rows "B1 + Platt" and "B1 (raw)" have the same PR-AUC, because Platt scaling does not change the ranking.

### 4.2 Per-window Wilcoxon signed-rank test (one-sided, Holm-adjusted over 7 comparisons)

| Compared with | PR-AUC: windows won / lost | PR-AUC p (Holm) | F1: windows won / lost | F1 p (Holm) |
|:---|:---:|:---:|:---:|:---:|
| Prior work: B1 XGBoost | 29 / 2 | **6.7 × 10⁻⁷** | 26 / 5 | **0.001** |
| Prior work: B1 + graph counts | 23 / 7 | **0.013** | 21 / 3 | **0.001** |
| BP1 EARN+RXT-J | 17 / 13 | 0.114 | 19 / 5 | **0.047** |
| BP2 GNN+AE | 20 / 10 | 0.114 | 19 / 6 | 0.077 |
| BP3 CARE-GNN | 17 / 14 | 0.104 | 24 / 6 | **0.022** |
| BP4 ARGUS | 21 / 9 | **0.033** | 16 / 7 | **0.047** |

Result: per window, MG-ECFM has a significantly higher F1 than 6 of the 7 models (not BP2, p = 0.077). It has a significantly higher PR-AUC than the prior work and BP4. Against BP1, BP2 and BP3, MG-ECFM wins more windows, but the per-window PR-AUC test is not significant (p ≈ 0.10–0.11).

### 4.3 McNemar test at the operating thresholds

From `reports/statistical_significance_results.json`: MG-ECFM makes significantly fewer errors than BP1 (p = 0.0026), BP2 (p < 10⁻⁵) and BP3 (p < 10⁻⁵). Against BP4 the difference is not significant (p = 0.57).

> Note: `paper_tables.md` Table 2 uses a two-sided Wilcoxon test on PR-AUC only. The one-sided tests above are the correct tests for the claim "MG-ECFM is better".

## 5. Monetary outcome

### 5.1 The router fix

The first version of this report showed a CS-ACRC loss of 213.9M for MG-ECFM. Section 6 explains the cause. The router is now replaced by the **budgeted Bayes CS-ACRC router** (`src/conformal/budgeted_crc.py`):

1. For each transaction, the router selects the action with the lowest expected cost: `argmin_a E[L(a) | q, amount]`. The router uses the amount of each transaction, and it has no fixed λ bands.
2. A price ν for each intervention keeps the long-run intervention rate at the budget ρ = 3%. ν is updated online by dual ascent from the released feedback.
3. An online intercept shift θ corrects the probabilities when the fraud rate drifts. The shift is updated from released labels only.

Both online updates have anytime guarantees of the ACI type. All settings were selected on the operating split (steps 576–631) before the test run: γ_b = 0.001, γ_c = 0.02, ρ = 3%. The price ν starts from a label-free warm start on the calibration scores.

### 5.2 Result of the pipeline (MG-ECFM own policy, 28,259 test transactions)

| | Old router (λ bands) | **New router (budgeted Bayes)** |
|:---|:---:|:---:|
| Net monetary loss | 213.89M | **13.94M** (−93%) |
| Fraud value stopped before settlement (Hold or Block) | 67.2% | **97.9%** |
| Frauds approved (of 404) | 311 | **122** |
| Intervention rate (all transactions) | 0.3% | 4.6% |
| Genuine customers with friction | ~0% | 3.7% |

The new router used 4.6% interventions against a 3% budget. In the test period the fraud rate is 1.43%, almost twice the operating rate (0.78%). The step size γ_b = 0.001 adapts slowly over 28,259 transactions. The budget guarantee is a long-run bound, and for this short stream it is loose.

For reference, the baselines with approve/block at τ lose 105.0M (BP2) to 328.6M (BP3), at 0.6–1.0% interventions. These numbers come from different intervention rates, so they are not an equal-friction comparison. Section 5.3 gives the fair comparison.

### 5.3 Equal-friction comparison (the main monetary result)

A model that intervenes more will lose less money. A fair comparison must therefore use the same intervention rate for all models. For each model, the script sweeps the intervention price ν of the Bayes rule over a fixed grid (no labels are used by any decision). It then reports the lowest net loss at each intervention-rate cap. This is the same idea as reading an ROC curve at a fixed false-positive rate.

**Lowest net loss at the same intervention rate (lower is better):**

| Intervention rate ≤ | **MG-ECFM** | Prior: B1 | Prior: B1 + Platt | Prior: B1 + graph | BP1 | BP2 | BP3 | BP4 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1% | **49.67M** | 131.56M | 109.64M | 73.62M | 67.04M | 66.70M | 70.24M | 61.95M |
| 2% | **31.79M** | 103.52M | 91.80M | 43.22M | 38.33M | 39.12M | 40.19M | 37.95M |
| 3% | **22.22M** | 83.08M | 79.85M | 34.43M | 28.57M | 26.21M | 28.00M | 29.27M |
| 5% | **12.14M** | 68.67M | 71.13M | 20.51M | 22.49M | 17.19M | 18.82M | 16.01M |

**Fraud value stopped at the same intervention rate:**

| Intervention rate ≤ | **MG-ECFM** | Best prior-work variant | Best baseline |
|:---:|:---:|:---:|:---:|
| 1% | **92.4%** | 88.7% | 90.5% (BP4) |
| 2% | **95.1%** | 93.4% | 94.2% (BP4) |
| 3% | **96.6%** | 94.7% | 96.0% (BP2) |
| 5% | **98.1%** | 96.9% | 97.6% (BP4) |

MG-ECFM has the lowest loss at every intervention rate. At the 1% rate it also has the fewest genuine customers with friction (0.39% of genuine transactions; the others have 0.47–0.62%).

### 5.4 Statistical proof of the savings

The method is a paired window-block bootstrap (2,000 resamples of the 50 windows) of the saving "loss of the other model − loss of MG-ECFM". The p-value is one-sided. No multiple-comparison correction is applied.

| Compared with | Saving at 1% | Saving at 2% | Saving at 3% | Saving at 5% |
|:---|:---:|:---:|:---:|:---:|
| Prior: B1 XGBoost | **81.9M** (p < 0.001) | **71.7M** (p < 0.001) | **60.9M** (p < 0.001) | **56.5M** (p < 0.001) |
| Prior: B1 + Platt | **60.0M** (p < 0.001) | **60.0M** (p < 0.001) | **57.6M** (p < 0.001) | **59.0M** (p < 0.001) |
| Prior: B1 + graph counts | **24.0M** (p < 0.001) | **11.4M** (p < 0.001) | **12.2M** (p < 0.001) | **8.4M** (p < 0.001) |
| BP1 EARN+RXT-J | **17.4M** (p < 0.001) | **6.5M** (p = 0.012) | 6.4M (p = 0.034, CI includes 0) | **10.4M** (p < 0.001) |
| BP2 GNN+AE | **17.0M** (p < 0.001) | **7.3M** (p = 0.004) | 4.0M (p = 0.085) | **5.1M** (p = 0.019) |
| BP3 CARE-GNN | **20.6M** (p < 0.001) | **8.4M** (p = 0.015) | **5.8M** (p = 0.011) | **6.7M** (p < 0.001) |
| BP4 ARGUS | **12.3M** (p < 0.001) | **6.2M** (p = 0.015) | **7.1M** (p = 0.004) | 3.9M (p = 0.030, CI includes 0) |

Bold: the 95% CI is above 0. The savings against the prior work are significant at every rate. The savings against all four baselines are significant at 1% and 2%. At 3% and 5%, three comparisons are not significant.

### 5.5 Other policies

- **Approve/block at τ:** MG-ECFM loses 72.30M and blocks 88.9% of the fraud value. The best other model (BP2) loses 104.99M, and the prior-work B1 loses 119.74M. (Before the tie-break, MG-ECFM lost 52.77M; the new τ has a higher precision and a lower recall.)
- **Bayes rule without a budget (ν = 0):** MG-ECFM loses 0.30M, the lowest of all models. But this policy puts friction on 37% of the transactions, so it is not realistic.

## 6. Why the old CS-ACRC loss was high (now fixed)

The old router used two different thresholds:

| Decision | Threshold | Used for |
|:---|:---|:---|
| Detection (`yhat`) | τ = 0.108 (F1-optimal on the operating split) | F1, precision, recall, McNemar |
| Action (`action`) | CS-ACRC λ ≈ (0.934, 0.944, 0.954) | EML, intervention rate |

The router acted only when p ≥ 0.934, so it approved 311 of the 404 frauds. 106 of these frauds (161M) were flagged by the detector. The λ values went up for three reasons:

1. **The risk target was above the fraud rate.** With α = 0.02 and a capped loss of 1 per missed fraud, "approve everything" stays inside the budget (fraud rate 0.5–1.4%). The controller always raised the thresholds.
2. **The loss cap removed the amount.** `ell = min(L / 10,000, 1)` gives the same loss for a missed fraud of 10,000 and one of 5,000,000.
3. **The bands were too narrow.** The minimum gap of 0.01 left Step-Up and Hold almost empty.

The new router removes all three causes. It routes on expected cost with the real amount, and it controls a quantity that binds: the intervention rate. The legacy controller stays in the code (`decision_rule: "threshold"`) for comparison.

## 7. Method comparison with the prior work

Sources: assessment report Sections 2, 6 and 9, and the code in this repository.

| Area | Prior work (as implemented) | MG-ECFM (implemented here) | Evidence in this repository |
|:---|:---|:---|:---|
| Detector | XGBoost on 7 raw features | Stacked fusion: XGBoost + camouflage-filtered temporal GraphSAGE + DAE residual + knowledge-base triggers, 32 causal features | `src/detection/`, `src/graph/` |
| Graph | 6 hand-made counting features, no GNN | Trained temporal-memory GraphSAGE with a label-aware camouflage filter | `src/graph/camouflage_graphsage.py` |
| Leakage control | Temporal split; balance columns "not confirmed fully addressed" | Post-event balance columns removed and checked before feature creation | Table 1 in `paper_tables.md` |
| Evidence gate | Fixed constants (`MIN_PROVENANCE = 1.00`, `MIN_GATE_SCORE = 0.75`) | Subjective Logic opinions + martingale e-process admission and rollback | `src/epistemic_gate/` |
| Guarantee on false admission | None | Ville's inequality, P(sup M_t ≥ 1/α) ≤ α, checked in 3 ways | Table 4 in `paper_tables.md` |
| Routing | Platt calibration, fixed policies A–D | Budgeted Bayes routing on the cost matrix, with an anytime bound on the intervention rate and an online drift correction | `src/conformal/budgeted_crc.py` |
| SOTA baselines | BP5 "PENDING REPRODUCTION" | BP1–BP4 re-implemented and run on the same rows | `src/baselines/` |
| Drift evaluation | One test window | 50 chronological windows with label delay | `experiments/run_drift_eval.py` |
| Significance tests | None | Wilcoxon, McNemar, bootstrap CI, Holm correction | Sections 4 and 5.4 |
| Poisoning test | Done (Day 27), no numbers in the report | 5 poison rates, false admissions measured, rollback time measured | Table 4 in `paper_tables.md` |
| RAG/LLM assist | "NOT QUANTIFIED" | Kept off the decision path and measured as an ablation arm | Table 3 in `paper_tables.md` |

## 8. Robustness of the knowledge gate

### 8.1 Poisoning stress test (from `paper_tables.md`, Table 4)

| Share of poisoned candidates | Harmful rules admitted: martingale gate | Harmful rules admitted: no gate | Harmful active rules rolled back | Mean / max cycles to rollback |
|:---:|:---:|:---:|:---:|:---:|
| 5% | **0.0%** | 100% | 100% | 1.05 / 2 |
| 10% | **0.0%** | 100% | 100% | 1.00 / 1 |
| 15% | **1.6%** | 100% | 100% | 1.13 / 3 |
| 20% | **1.5%** | 100% | 100% | 1.16 / 4 |
| 25% | **0.0%** | 100% | 100% | 1.12 / 5 |

The Ville bound (α = 0.05) holds at all 5 poison rates.

### 8.2 Effect of the gate on customers (from `paper_tables.md`, Table 3)

| Ablation arm | Net loss | Intervention rate |
|:---|:---:|:---:|
| B2: gate OFF (all 200 candidate rules admitted, 10% poisoned) | 65.39M | **99.97%** |
| B4: full MG-ECFM (martingale gate ON) | **13.94M** | **4.6%** |

Without the gate, the admitted rules fire on almost every payment, and the rule floor sends each of these payments to Step-Up. With the gate, the loss is 4.7 times lower and the friction is 22 times lower.

## 9. Claims for the paper

### 9.1 Claims that the data supports

1. MG-ECFM has the highest PR-AUC, ROC-AUC, recall, F1, F2, recall at 1% FPR and the lowest pooled Brier of all 8 models on the same 28,259 test transactions.
2. **MG-ECFM has a significantly higher pooled PR-AUC than all 7 other models** (window-block bootstrap, every 95% CI above 0).
3. Against the prior work, MG-ECFM is significantly better on PR-AUC (+0.29, p < 0.001) and on F1 (+0.20, Holm p = 0.001).
4. Per window, MG-ECFM has a significantly higher F1 than the prior work, BP1, BP3 and BP4 (Holm p ≤ 0.047). It makes significantly fewer errors than BP1, BP2 and BP3 (McNemar).
5. **At equal intervention rates, MG-ECFM has the lowest monetary loss of all 8 models** (1%, 2%, 3% and 5%). The savings are significant against all 7 models at 1% and 2%.
6. Against the prior work, MG-ECFM saves 8–82M (26–83% of the prior-work loss) at every intervention rate (p < 0.001).
7. With the new router, MG-ECFM stops 97.9% of the fraud value before settlement, and it loses 93% less than with the old router.
8. The martingale gate keeps harmful admissions at 1.6% or less up to a 25% poison rate. Without the gate, 99.97% of the payments get friction.

### 9.2 Claims that the data does not support (do not make these claims)

1. "MG-ECFM has a significantly higher per-window PR-AUC than BP1, BP2 and BP3." The Holm-adjusted Wilcoxon p-values are 0.10–0.11. (The pooled PR-AUC difference is significant.)
2. "MG-ECFM has a significantly higher F1 than BP2." Holm p = 0.077.
3. "The monetary savings against the baselines are significant at every intervention rate." At 3% (BP1, BP2) and at 5% (BP4), the CIs include 0.
4. "The router keeps the intervention rate at exactly 3%." It used 4.6% in the test period (Section 5.2).
5. "The adaptive budget lowers the loss." The static variant (price frozen after the warm-up) lost 11.86M at 5.4% interventions. The adaptive part keeps the friction nearer the budget, but at equal friction it does not lower the loss.
6. "The graph and the autoencoder improve detection on PaySim." In the ablation, XGBoost on our 32 causal features alone has a PR-AUC of 0.594, and the full system has 0.593. The full system has a slightly higher F1 (0.544 vs. 0.539). Most of the gain over the prior work comes from the leakage-free causal features and the training design.
7. "The knowledge base and the recovery module improve the result." In this run the gate admitted 0 rules, and B4 (recovery ON) has the same net loss as B3 (recovery OFF).
8. "MG-ECFM is faster than the prior work." The prior-work XGBoost needs about 0.0001 ms per transaction, and MG-ECFM needs about 0.2–0.4 ms.

### 9.3 Limits of this comparison

- The prior-work numbers come from our reproduction of its documented features and model. The original notebooks are not in this repository. The reproduced values are near the reported values (Section 2.1).
- The test set is a seeded 2M-row sample. The prior work used the full 6.36M rows.
- MG-ECFM was trained again for the router run and for the tie-break run (CPU). The baselines were not trained again, because neither change affects them. Their predictions are the saved predictions of the first full run.
- The equal-friction table selects ν per model on the test rows to reach each intervention cap. This uses no labels in the decisions, and it is the same for all models, like an ROC curve. The selection of the lowest loss on the curve uses the test labels, the same for all models.

## 10. How to reproduce

1. Run the drift evaluation (this step creates `reports/drift_evaluation_predictions.csv`):
   `MG_ECFM_DEVICE=cpu python experiments/run_drift_eval.py --source paysim --sample-rows 2000000`
2. Run the ablation:
   `MG_ECFM_DEVICE=cpu python experiments/run_ablation_study.py --source paysim --sample-rows 2000000`
3. Reproduce the prior-work detectors on the same splits:
   `python experiments/run_prior_work_comparison.py`
4. Compute all metrics in this report:
   `python experiments/compare_with_prior_work.py`

Outputs: `reports/prior_work_predictions.csv`, `reports/prior_work_fit.json`, `reports/prior_work_comparison.json`.

## 11. Next directions (recommended order)

1. **Make the knowledge base work.** The martingale gate admitted 0 of the candidate rules in the full run, so the knowledge-base and recovery parts add nothing to the result (Section 9.2). Check why the candidate rules fail the admission test on PaySim (rule precision against p₀ = 0.5). Then generate candidate rules from the training data, so that genuine rules can be admitted.
2. **Show the value of the graph on a relational dataset.** On PaySim the graph adds no ranking quality, because most PaySim senders appear only once. Run the same protocol on AMLSim or IEEE-CIS, where account history is richer.
3. **Tighter budget tracking.** The router used 4.6% interventions against a 3% budget. Test a larger γ_b, or a per-window normalised update, on the operating split. Report the trade-off.
4. **Run on the full 6.36M-row PaySim log**, the same data size as the prior work, after items 1–3 are stable on the 2M sample.
5. **Fix the old partition test** (`tests/test_phase7_experiments.py::test_partition_rejects_invalid_inputs`). The code limits the number of windows to the number of steps, but the test expects an error. This failure is older than this work.
