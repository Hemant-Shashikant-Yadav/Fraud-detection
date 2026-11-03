# Martingale-Gated Epistemic-Conformal Fraud Mitigation (MG-ECFM)

> **A Statistically Certified Framework for Risk-Aware Prevention, Relational Detection, and Reversible Knowledge Evolution in Digital Payment Systems**

* **Researcher:** Shradha Vijay Kolekar (URN: 1025230008)
* **Guide:** Dr. Anisa B. Shikalgar, Professor, Dept. of Computer Science & Engineering
* **Institution:** Annasaheb Dange College of Engineering & Technology, Ashta, Sangli

M.Tech dissertation codebase and companion to the IEEE journal manuscript (IEEE Access / IEEE TDSC format).
Every module is executable, seeded (`SEED = 42`) and covered by a phase test suite.

> 💡 **First time running this project?** See the step-by-step [First-Time User Guide](FIRST_TIME_USER_GUIDE.md) to set up your environment, download all real datasets with a single command, train models, and reproduce paper results from scratch.

---

## Contents

1. [What MG-ECFM does](#1-what-mg-ecfm-does)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Quick start (60 seconds)](#4-quick-start-60-seconds)
5. [Command-line interface (`main.py`)](#5-command-line-interface-mainpy)
6. [Interactive dashboard](#6-interactive-dashboard)
7. [Reproducing the paper](#7-reproducing-the-paper)
8. [Camera-ready tables and figures](#8-camera-ready-tables-and-figures)
9. [Batch scoring a CSV](#9-batch-scoring-a-csv)
10. [Datasets](#10-datasets)
11. [Tests](#11-tests)
12. [Results snapshot](#12-results-snapshot)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What MG-ECFM does

A payment authorisation pipeline that must decide within a few milliseconds, under concept drift, adversarial
knowledge poisoning, camouflaged mule networks and labels that arrive 1-14 days late.

| Stage | Component | Module |
| :--- | :--- | :--- |
| Causal ledger | Leakage-free streaming features: post-event PaySim balance columns stripped at ingestion; every feature of `e_t` uses only events with step `< tau_t` | `src/data/` |
| Relational detection | Temporal-memory GraphSAGE with an l1 label-aware similarity filter that prunes camouflaged neighbour edges; denoising-autoencoder residual; cost-sensitive calibrated XGBoost fusion | `src/graph/`, `src/detection/` |
| Risk-aware prevention | Cost-Sensitive Adaptive Conformal Risk Control (CS-ACRC): online thresholds `lambda_{t+1} = lambda_t - gamma (alpha - L_t)` route every transaction to **Approve / Step-Up / Hold / Block** | `src/conformal/` |
| Certified knowledge evolution | Subjective-Logic opinions `omega = (b, d, u, a)` with provenance discounting and freshness decay; SAVI betting e-processes admit rules (`H0: precision <= p0`) and roll them back (`H0: FPR <= q0`) at Ville's threshold `1/alpha`; versioned SQLite store with atomic snapshots | `src/epistemic_gate/` |
| Post-incident recovery | Time-Decaying Tainted Proportional Flow (TD-TPF) traces stolen funds through mule wallets inside the 24 h settlement window; freeze / block / chargeback simulation | `src/recovery/` |
| Closed loop | Delayed verification queue (1-14 days) feeding threshold recalibration and rollback | `src/closed_loop/` |
| Baselines | BP1 EARN+RXT-J, BP2 GNN+AE, BP3 CARE-GNN, BP4 ARGUS (hybrid transformer + CRC) | `src/baselines/` |
| Orchestration | Live service (what-if scoring, explanation, batch scoring, recovery, persistence) used by the CLI and dashboard | `src/orchestrator/` |

---

## 2. Repository structure

```
Fraud detection new/
├── main.py                          # Phase 8 - unified CLI (quick demo, train, experiments, predict, UI)
├── app/
│   └── app.py                       # Phase 8 - Streamlit dashboard (5 tabs)
├── configs/
│   ├── project_config.json          # datasets, leakage contract, splits, costs, perturbation feed
│   └── hyperparams.json             # XGBoost, DAE, GraphSAGE, CS-ACRC, Subjective Logic, e-process, TD-TPF
├── src/
│   ├── data/                        # Phase 1 - loaders, causal preprocessor, adversarial rule feeds
│   ├── graph/                       # Phase 2 - temporal memory, similarity filter, camouflage GraphSAGE
│   ├── detection/                   # Phase 2 - calibrated XGBoost, DAE residual, stacked fusion
│   ├── conformal/                   # Phase 3 - asymmetric loss, online CS-ACRC, pre-transaction router
│   ├── epistemic_gate/              # Phase 4 - subjective logic, martingale e-processes, knowledge store
│   ├── recovery/                    # Phase 5 - TD-TPF tracker, recovery action simulator
│   ├── closed_loop/                 # Phase 5 - verification queue, feedback adapter
│   ├── baselines/                   # Phase 6 - BP1-BP4
│   └── orchestrator/                # Phase 8 - live MG-ECFM service
├── experiments/                     # Phase 7 - drift eval, ablation, stress test, significance tests
├── reports/
│   ├── generate_paper_tables.py     # Phase 8 - Tables 1-4 -> paper_tables.tex / paper_tables.md
│   ├── generate_paper_figures.py    # Phase 8 - Figures 1-4 -> figures/*.pdf + 300-DPI *.png
│   ├── *.csv / *.json               # Phase 7 experiment outputs (inputs of the generators)
│   └── quick_demo/                  # everything produced by `python main.py --mode quick-demo`
├── models/                          # saved live services (*.pkl) created by train / quick-demo
├── tests/                           # test_phase1_data.py ... test_phase8_orchestrator.py
├── scripts/                         # run_phaseN_tests.bat runners, diagram / deck generators
└── requirements.txt
```

---

## 3. Installation

Python 3.10-3.13 (Windows, Linux or macOS). From the repository root:

```powershell
cd "D:\Coding\Project\Fraud detection new"
python -m venv venv
.\venv\Scripts\Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install --upgrade pip

# PyTorch Installation:
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU (CUDA 12.1+ on Windows / Linux):
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# For macOS Apple Silicon (M1/M2/M3/M4):
# pip install torch   (Apple Silicon Metal MPS acceleration is supported out of the box)

pip install -r requirements.txt
```

`torch_geometric` and `scipy` are optional (pure-PyTorch aggregation and the built-in statistics are used
without them). `streamlit` is only needed for the dashboard.

---

## 4. Quick start (60 seconds)

```powershell
python main.py --mode quick-demo
```

On a small synthetic PaySim-schema stream (≈ 8.8 k transactions, 10 drift windows) this runs, in order:

1. the causal feature stream and chronological `train < calibration < operating < test` split;
2. MG-ECFM **and** BP1-BP4 through the prequential drift protocol;
3. McNemar / Wilcoxon / block-bootstrap significance tests;
4. the B1-B5 ablation study and the 5 %-25 % poisoning sweep with Ville's-bound checks;
5. the live service: single-transaction authorisation with TreeSHAP reasons, a 200-transaction latency
   profile, the camouflage graph, batch CSV scoring, a TD-TPF recovery trace and a save / load round trip;
6. the camera-ready tables and figures.

Every step is checked; the console ends with `quick demo finished in XX.Xs - N/N checks passed` and the full
record is in `reports/quick_demo/quick_demo_manifest.json`. The demo uses reduced model capacity and is a
**smoke test** — the numbers for the paper come from Section 7.

---

## 5. Command-line interface (`main.py`)

| Command | What it does | Main outputs |
| :--- | :--- | :--- |
| `python main.py --mode quick-demo` | End-to-end smoke run with checks (Section 4) | `reports/quick_demo/`, `models/quick_demo_service.pkl` |
| `python main.py --mode train [--profile fast\|full\|demo] [--source auto\|synthetic\|paysim]` | Fit the live service on the training split, replay the test period, profile latency, save | `models/mg_ecfm_service.pkl` (+ `.json` report) |
| `python main.py --mode drift-eval [--profile fast\|full] [--n-windows 50]` | 50-window chronological drift benchmark, MG-ECFM vs BP1-BP4 | `reports/drift_evaluation_*.csv/json` |
| `python main.py --mode ablation` | B1-B5 ablation (seven arms) | `reports/ablation_study_*.csv` |
| `python main.py --mode stress-test` | Poisoning sweep 5 %-25 %, time-to-rollback, Ville's inequality | `reports/adversarial_*.csv` |
| `python main.py --mode full-pipeline` | drift-eval → significance → ablation → stress-test → tables + figures | all of the above |
| `python main.py --mode paper-artifacts [--reports-dir reports]` | Regenerate tables and figures from existing CSVs | `reports/paper_tables.*`, `reports/figures/` |
| `python main.py --predict FILE.csv [--output OUT.csv] [--model PKL]` | Score a transaction CSV | `FILE_scored.csv` |
| `python main.py --ui [--port 8501] [--model PKL]` | Launch the Streamlit dashboard | browser at `http://localhost:8501` |

Common options:
* `--max-rows N`: Limit streaming ingestion to first $N$ rows (e.g. `--max-rows 500000` for memory-safe fast training in ~38s).
* `--device {auto,cuda,mps,cpu}`: Hardware target (auto-detects NVIDIA CUDA on Windows/Linux or Apple Silicon MPS on macOS).
* `--seed 42`: Fix RNG seeds across NumPy, PyTorch, and Python.
* `--profile {fast,full,demo}`: Select training epochs, estimators, and graph fanouts.

```powershell
# Recommended memory-safe training on real PaySim data (~38 seconds):
python main.py --mode train --source paysim --profile fast --max-rows 500000

# Explicitly select GPU compute target:
python main.py --mode train --source paysim --profile fast --max-rows 500000 --device cuda   # NVIDIA
python main.py --mode train --source paysim --profile fast --max-rows 500000 --device mps    # Apple Silicon
```

---

## 6. Interactive dashboard

```powershell
python main.py --ui                  # equivalent: streamlit run app/app.py
```

The dashboard loads, in order, `$MG_ECFM_SERVICE`, `models/mg_ecfm_service.pkl` (from `--mode train`) or
`models/quick_demo_service.pkl` (from `--mode quick-demo`); if none exists it builds the demo service once.
The sidebar shows the live CS-ACRC thresholds, the binary operating threshold, the active rule count and a
**latency profiler** (p50 / p95 / p99 of the authorisation path against the 5 ms budget).

| Tab | For the examiner |
| :--- | :--- |
| **⚡ Live Scoring** | Pick a scenario (verified mule-network transfer, routine merchant payment, large transfer to an unseen account) or type any sender / receiver / amount / type. Shows the action badge (Approve / Step-Up / Hold / Block), the routing reason, the authorisation latency, a risk dial whose bands are the live CS-ACRC thresholds, the TreeSHAP attribution of the fusion vector (graph embeddings, DAE residual, KB triggers, causal features), fired knowledge-base rules with their opinion-weighted precision, and the expected loss of every action. Scoring is *what-if*: the live state is never modified. |
| **🕸 Mule Network** | The two-hop neighbourhood of the same transaction. Grey edges were kept, dashed red edges were **pruned** by the label-aware similarity filter before GraphSAGE aggregation; red-ringed accounts are verified fraud counterparties; squares are merchants. Edge and account tables list the similarity scores. |
| **📄 Batch Scorer** | Upload a CSV (or score the built-in sample), see the action distribution and download the decisions. Leakage columns are stripped automatically. |
| **📚 Knowledge Base** | Every candidate rule with its gate status (active / quarantined / rejected), opinion `(b, d, u, a)`, expected probability, live admission / rollback wealth, a belief-uncertainty map and the versioned snapshot history with an integrity check. |
| **🧭 Recovery Trace** | Choose a fraud that MG-ECFM did not intercept and a time-to-detection; TD-TPF traces the tainted funds, simulates freeze / instrument block / chargeback and plots recovery against detection delay with the 24 h settlement window. |

The latency shown is the authorisation path (graph embedding, DAE residual, `phi_KB`, calibrated XGBoost and
routing). Causal featurisation of the incoming event happens before it and is reported separately; the
explanation (TreeSHAP) is computed after the decision and is not part of the timed path.

---

## 7. Reproducing the paper

All runs are deterministic (`SEED = 42`, chronological splits, fixed perturbation feed).

```powershell
# 1. Verify the installation (Phases 1-8, writes reports\phase8_test_log.txt)
scripts\run_phase8_tests.bat

# 2. Run every experiment and rebuild the camera-ready artifacts
python main.py --mode full-pipeline                 # fast profile, synthetic drift stream (minutes)
python main.py --mode full-pipeline --profile full  # published model capacity

# 3. On the real PaySim log (place the CSV first, see Section 10)
python main.py --mode train --source paysim --profile fast --max-rows 500000
python main.py --mode full-pipeline --source paysim --max-rows 2000000
```

`full-pipeline` is equivalent to running, in order, `experiments/run_drift_eval.py`,
`experiments/statistical_significance.py`, `experiments/run_ablation_study.py`,
`experiments/run_adversarial_stress_test.py`, `reports/generate_paper_tables.py` and
`reports/generate_paper_figures.py`.

---

## 8. Camera-ready tables and figures

```powershell
python reports\generate_paper_tables.py     # -> reports\paper_tables.tex, reports\paper_tables.md
python reports\generate_paper_figures.py    # -> reports\figures\fig1..fig4 (.pdf + 300-DPI .png)
```

| Artifact | Content |
| :--- | :--- |
| Table 1 | Dataset summary and causal-ledger contract (stripped columns, entity keys, chronological split) |
| Table 2 | MG-ECFM vs BP1-BP4: PR-AUC, F1, Brier, expected monetary loss, latency; Holm-adjusted Wilcoxon and McNemar p-values |
| Table 3 | B1-B5 ablation matrix (component switches, ranking, calibration, monetary loss, prevented loss, KB size) |
| Table 4 | Poisoning 5 %-25 %: false-admission rate per policy, rollback rate and time-to-rollback, Ville's bound (exact, Monte-Carlo with 95 % CI, in-gate) |
| Fig. 1 | PR-AUC trajectories over the chronological drift windows |
| Fig. 2 | Martingale wealth of rollback (injected rules) and admission (quarantined candidates) e-processes with stopping times at `1/alpha` |
| Fig. 3 | CS-ACRC thresholds `lambda_low / lambda_mid / lambda_high` under drift, with fraud and intervention rates |
| Fig. 4 | TD-TPF recovery rate vs time-to-detection (kappa sensitivity) and prevented-loss ratio |

LaTeX: `\input{paper_tables.tex}` in an IEEEtran manuscript with `\usepackage{amsmath,amssymb,graphicx}`; wide
tables use `table*` and scale to `\linewidth`. Figures are sized for the IEEE column (3.5 in) and page
(7.16 in) widths with embedded TrueType fonts; the data behind Figs. 2 and 4 is saved in
`figures/figure_data/` so every plotted point can be audited. Every number in the tables is read from the
experiment CSV/JSON files — nothing is typed in by hand.

---

## 9. Batch scoring a CSV

Accepted schemas (column order does not matter, extra columns are carried through):

* **PaySim:** `step, type, amount, nameOrig, nameDest[, isFraud]` — the post-event balance columns
  (`oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest`) and `isFlaggedFraud` are dropped;
* **canonical:** `step, tx_type, amount, src, dst[, label]`.

```powershell
python main.py --predict data\my_transactions.csv
```

The output adds `mg_ecfm_step, p_hat, risk_score, action, action_label, route_reason, kb_rules_fired,
binary_flag`. Rows are processed in step order after the service's history (steps are shifted so the
batch directly follows it), features are computed causally inside the batch, and the live service is not
modified. If the file contains labels, PR-AUC / F1 / Brier are printed.

---

## 10. Datasets

| Dataset | Source | Rows | Place at |
| :--- | :--- | :---: | :--- |
| PaySim | Lopez-Rojas et al. mobile-money simulator | 6,362,620 (8,213 frauds) | `data/raw/paysim/PS_20174392719_1491204439457_log.csv` |
| IEEE-CIS | Vesta / IEEE CIS fraud detection | 590,540 | `data/raw/ieee_cis/train_transaction.csv`, `train_identity.csv` |
| AMLSim | IBM AML simulator | configurable | `data/raw/amlsim/tx.csv`, `accounts.csv` |

`--source auto` (the default) resolves to PaySim and **stops with an error if the raw file is missing**;
it never falls back to synthetic data silently, so a run reported as real data cannot be a synthetic one.
To run without a download, pass `--source synthetic` explicitly: the schema-faithful synthetic PaySim
generator with gradual camouflage drift (fraud amounts migrate towards the legitimate amount distribution in
the later part of the stream). Synthetic results are for smoke tests only, not for paper numbers.
`--mode quick-demo` always uses synthetic data.

---

## 11. Tests

```powershell
scripts\run_phase8_tests.bat            # Phase 8 + Phase 1-7 regression -> reports\phase8_test_log.txt
scripts\run_phase8_quick.bat            # Phase 8 only
scripts\run_phase8_tests.bat artifacts  # also rebuild reports\paper_tables.* and reports\figures\
python tests\test_phase8_orchestrator.py
python -m pytest tests -q
```

`tests/test_phase8_orchestrator.py` runs `python main.py --mode quick-demo` as a subprocess and verifies:
exit code and every internal check; predictions of all five models for every test transaction; experiment
outputs; the 5 ms p95 authorisation budget; leakage stripping in batch scoring; LaTeX structure and Markdown
tables (also from the published `reports/` CSVs); four PDF/PNG figures at 300 DPI with auditable data; the
`--predict` CLI; side-effect-free what-if scoring, TreeSHAP additivity, KB opinion validity
(`b + d + u = 1`), TD-TPF conservation; and a headless render of the dashboard through `streamlit.testing`.

| Suite | Scope |
| :--- | :--- |
| `test_phase1_data.py` | causal featurisation, leakage refusal, permutation invariance, streaming exactness |
| `test_phase2_graph_detection.py` | temporal memory, similarity filter, GraphSAGE, DAE, calibrated fusion |
| `test_phase3_conformal.py` | asymmetric loss, CS-ACRC updates and guarantees, router |
| `test_phase4_epistemic_martingale.py` | subjective logic, e-processes, Ville's bound, knowledge store |
| `test_phase5_recovery_closedloop.py` | TD-TPF, recovery simulator, verification queue, feedback |
| `test_phase6_baselines.py` | BP1-BP4 |
| `test_phase7_experiments.py` | drift protocol, ablation, stress test, statistics |
| `test_phase8_orchestrator.py` | CLI, quick demo, artifacts, live service, dashboard |

---

## 12. Results snapshot

From the Phase-7 run in `reports/` (synthetic PaySim stream, drift strength 0.5, 50 windows, fast profile);
regenerate with `python main.py --mode full-pipeline`. Mean over windows:

| Model | PR-AUC | F1 | Brier | EML |
| :--- | :---: | :---: | :---: | :---: |
| **MG-ECFM** | 0.553 | 0.425 | **0.0353** | **572.9** |
| BP1 EARN+RXT-J | 0.537 | 0.363 | 0.1060 | 1909.4 |
| BP2 GNN+AE | **0.585** | 0.458 | 0.0578 | 1236.8 |
| BP3 CARE-GNN | 0.572 | 0.446 | 0.0534 | 1388.0 |
| BP4 ARGUS | 0.495 | **0.507** | 0.2604 | 801.0 |

On this run MG-ECFM has the best calibration and the lowest expected monetary loss (its four-action
CS-ACRC routing), while BP2 and BP3 rank slightly higher on PR-AUC (BP2's advantage is significant after
Holm correction, p = 0.029). The martingale gate admitted no adversarial (poisoned / Sybil) rule at any
poisoning rate, rolled back every injected harmful rule within 5 update cycles, and Ville's bound held
in all three checks at every rate (Table 4). Treat these figures as the current state of the experiments,
not final paper claims; re-run on the real datasets before submission.

---

## 13. Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `ModuleNotFoundError: torch` / `xgboost` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install -r requirements.txt` |
| `streamlit is not installed` | `pip install streamlit` |
| Dashboard shows the demo model | Run `python main.py --mode train` (saves `models/mg_ecfm_service.pkl`) or pass `--model` |
| `upload is missing columns` | Use the PaySim or canonical column names of Section 9 |
| Quick demo over 60 s | First run includes PyTorch start-up and JIT warm-up; the stage timings are in `reports/quick_demo/quick_demo_manifest.json` |
| `torch.jit.script is deprecated` warning | Harmless (emitted by a baseline on recent PyTorch) |

Phase 1 implementation details: [`PHASE1_IMPLEMENTATION_AND_CODE_WALKTHROUGH.md`](PHASE1_IMPLEMENTATION_AND_CODE_WALKTHROUGH.md).
