# MG-ECFM: Complete Beginner & First-Time User Guide
===================================================

> **A Step-by-Step Walkthrough to Set Up, Ingest Real Datasets, Train Models, Reproduce Research Results, and Launch the Interactive Dashboard from Scratch.**

---

## Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step 1: Clone and Open the Repository](#step-1-clone-and-open-the-repository)
3. [Step 2: Python Virtual Environment Setup](#step-2-python-virtual-environment-setup)
4. [Step 3: Install Required Dependencies](#step-3-install-required-dependencies)
5. [Step 4: Configure Kaggle API Credentials](#step-4-configure-kaggle-api-credentials)
6. [Step 5: Download & Extract All Datasets in One Command](#step-5-download--extract-all-datasets-in-one-command)
7. [Step 6: Quick Smoke Test (Sanity Check)](#step-6-quick-smoke-test-sanity-check)
8. [Step 7: Train the Production Fraud Mitigation Service](#step-7-train-the-production-fraud-mitigation-service)
9. [Step 8: Reproduce the Full Research Paper Results](#step-8-reproduce-the-full-research-paper-results)
10. [Step 9: Launch the Interactive Streamlit Dashboard](#step-9-launch-the-interactive-streamlit-dashboard)
11. [Step 10: Run Batch Inference on Custom CSVs](#step-10-run-batch-inference-on-custom-csvs)
12. [Step 11: Cross-Dataset Generalization (IEEE-CIS)](#step-11-cross-dataset-generalization-ieee-cis)
13. [Troubleshooting & Common Questions](#troubleshooting--common-questions)

---

## 1. Prerequisites & System Requirements

Before starting, ensure your system has:
* **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+ recommended).
* **Python**: Python 3.10, 3.11, 3.12, or 3.13 (64-bit).
* **RAM**:
  * Minimum: 8 GB
  * Recommended: 16 GB+ (PaySim has 6.36 million records).
* **Disk Space**: At least 4 GB free space for datasets and generated reports.
* **Kaggle Account**: Required for downloading PaySim and IEEE-CIS raw files.

Check your Python version:
```powershell
python --version
# Output should be: Python 3.10.x, 3.11.x, 3.12.x, or 3.13.x
```

---

## Step 1: Clone and Open the Repository

Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:

```powershell
cd "path\to\Fraud detection new"
```

Verify your workspace directory:
```powershell
# Windows PowerShell
Get-ChildItem -Name

# Linux / macOS
ls
```
You should see files such as `main.py`, `requirements.txt`, `configs/`, `src/`, `data/`, etc.

---

## Step 2: Python Virtual Environment Setup

Isolating dependencies inside a virtual environment prevents conflicts with existing Python packages.

### Windows (PowerShell)
```powershell
# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. If PowerShell scripts are restricted, allow execution for your user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Activate the environment
.\venv\Scripts\Activate.ps1
```

### Windows (Command Prompt `cmd.exe`)
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Linux / macOS (Bash / Zsh)
```bash
python3 -m venv venv
source venv/bin/activate
```

*(Once activated, your terminal prompt will be prefixed with `(venv)`).*

---

## Step 3: Install Required Dependencies

1. **Upgrade `pip`**:
   ```powershell
   python -m pip install --upgrade pip
   ```

2. **Install PyTorch** (CPU build recommended for universal compatibility):
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   *(If you have a dedicated NVIDIA GPU and CUDA installed, install the matching CUDA wheel from [pytorch.org](https://pytorch.org)).*

3. **Install Project Requirements**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Verify Environment Integrity**:
   ```powershell
   python -c "import torch, xgboost, streamlit, pandas, numpy; print('Environment is ready!')"
   ```
   If it prints `Environment is ready!`, all core dependencies are installed.

---

## Step 4: Configure Kaggle API Credentials

The project relies on public benchmark datasets hosted on Kaggle:
- **PaySim** (Mobile money fraud simulation: 6.36 million transactions)
- **IEEE-CIS** (Vesta e-commerce card-not-present fraud detection)

### 4.1 Obtain Your Kaggle Token
1. Go to [kaggle.com](https://www.kaggle.com/) and sign in.
2. Click your profile picture (top right) $\rightarrow$ **Settings** (or visit [kaggle.com/settings](https://www.kaggle.com/settings)).
3. Scroll down to the **API** section and click **Create New Token**.
4. You will receive an API token (starts with `KGAT_...`) or download a `kaggle.json` file.

### 4.2 Save Credentials in `.env`
In the project root folder, create a file named `.env` (or copy `.env.example` to `.env`):

```env
# If you have the new token:
KAGGLE_API_TOKEN=KGAT_your_token_here

# OR if you have a legacy kaggle.json:
# KAGGLE_USERNAME=your_kaggle_username
# KAGGLE_KEY=your_kaggle_key
```

### 4.3 Accept IEEE-CIS Competition Rules *(Crucial Step)*
Kaggle requires users to agree to competition terms before allowing data downloads:
1. Open [kaggle.com/competitions/ieee-fraud-detection/rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) in your browser.
2. Click **"I Understand and Accept"**.
*(If this step is skipped, Kaggle will return a `403 Forbidden` error when downloading IEEE-CIS).*

---

## Step 5: Download & Extract All Datasets in One Command

We provide a dedicated download manager that automates downloading, extracting, and verifying all datasets:

```powershell
python scripts/download_datasets.py
```

### What this command does:
1. Automatically reads your `KAGGLE_API_TOKEN` from `.env`.
2. Downloads and unzips **PaySim** (`ealaxi/paysim1`) into `data/raw/paysim/`.
3. Downloads and unzips **IEEE-CIS** (`ieee-fraud-detection`) into `data/raw/ieee_cis/`.
4. Copies or fetches **AMLSim** (`tx.csv` and `accounts.csv`) into `data/raw/amlsim/`.
5. Skips any file that is already downloaded and healthy.
6. Prints a status audit table:

```
==============================================================================
 MG-ECFM Raw Datasets Status Audit
==============================================================================
Dataset      Filename                            Status     Size (MB)   
------------------------------------------------------------------------------
paysim       PS_20174392719_1491204439457_log.csv OK         470.67      
ieee_cis     train_transaction.csv               OK         651.69      
ieee_cis     train_identity.csv                  OK         25.30       
amlsim       tx.csv                              OK           0.01      
amlsim       accounts.csv                        OK           0.01      
==============================================================================
```

---

## Step 6: Quick Smoke Test (Sanity Check)

Before executing multi-million-row runs, verify that the end-to-end framework works properly:

```powershell
python main.py --mode quick-demo
```

* **Runtime**: ~45–60 seconds.
* **What happens**: Executes an end-to-end smoke evaluation through all 8 stages on a small stream: causal feature engineering, baseline evaluations (BP1–BP4), statistical significance, ablation, adversarial stress test, and camera-ready report generation.
* **Expected Output**:
  ```
  quick demo finished in 48.2s - 10/10 checks passed
  ```
* **Output location**: Artifacts are stored in `reports/quick_demo/`.

---

## Step 7: Train the Production Fraud Mitigation Service

Train the live MG-ECFM service on the real temporal training split of PaySim (steps 1–520).

### Recommended Verification Profile (~38 seconds, Memory-Safe):
For first-time runs and systems with 8 GB or 16 GB RAM, train on the first 500,000 temporal transactions:
```powershell
python main.py --mode train --source paysim --profile fast --max-rows 500000
```
* **Why use `--max-rows 500000`**: PaySim has 6.36 million rows; streaming the entire dataset into memory can exceed 8 GB RAM. The 500,000-row slice preserves strict temporal ordering, exercises all relational graph construction and autoencoder updates, and generates a production-grade 315 MB service artifact in just ~38 seconds without risk of memory exhaustion.

### Full-Scale Training (All 6.36 Million Rows, ~2–4 minutes):
```powershell
python main.py --mode train --source paysim --profile fast
```

### Full Profile (Maximum Publication-Grade Capacity):
```powershell
python main.py --mode train --source paysim --profile full
```

### Hardware Acceleration (NVIDIA CUDA & Apple Silicon MPS):
The pipeline automatically routes neural networks (GraphSAGE, Residual Autoencoder, Baselines BP1–BP4) and tabular models to your dedicated GPU:
* **Auto-detect (Default)**: Automatically selects CUDA on NVIDIA systems, MPS on Apple Silicon Macs, or CPU.
* **Explicit Selection**:
  ```powershell
  # Force NVIDIA CUDA (Windows / Linux)
  python main.py --mode train --source paysim --profile fast --max-rows 500000 --device cuda

  # Force Apple Silicon GPU (macOS M1/M2/M3/M4)
  python main.py --mode train --source paysim --profile fast --max-rows 500000 --device mps

  # Force CPU
  python main.py --mode train --source paysim --profile fast --max-rows 500000 --device cpu
  ```
  *(You can also set the environment variable `MG_ECFM_DEVICE=cuda` or `MG_ECFM_DEVICE=mps`)*.

### What this produces:
* Persists the live trained model into `models/mg_ecfm_service.pkl` (~315 MB).
* Produces a companion summary report in `models/mg_ecfm_service.json`.
* Measures and prints real-time authorization latency ($p50 \approx 3.0\text{ ms}$, $p95 < 5.0\text{ ms}$).

---

## Step 8: Reproduce the Full Research Paper Results

To re-run the complete benchmark protocol and regenerate all camera-ready tables and figures:

```powershell
python main.py --mode full-pipeline --source paysim --profile fast
```
*(For exhaustive published capacity, use `--profile full`)*.

### What this single command executes in order:
1. **Chronological Drift Evaluation (50 Windows)**:
   Evaluates MG-ECFM alongside 4 competitive state-of-the-art baselines under prequential test streaming:
   - **BP1**: EARN-RxT-J (SMOTE + ResNet-Transformer + Jaya)
   - **BP2**: STAN-GNN (Spatio-Temporal Graph Neural Net + DAE)
   - **BP3**: HGT-CFD (Heterogeneous Graph Transformer)
   - **BP4**: Mamba-ECF (Selective State Space Model)
2. **Statistical Significance Testing**:
   Computes McNemar's paired test, Wilcoxon signed-rank test across windows, and block-bootstrap confidence intervals ($p < 0.05$).
3. **5-Point Component Ablation (B1–B5)**:
   Measures the incremental value of XGBoost tabular detection, GraphSAGE relational embeddings, DAE residual anomaly detection, Martingale e-process gate, and CS-ACRC risk control.
4. **Adversarial Poisoning Stress Tests**:
   Evaluates resilience across 5%, 10%, 15%, 20%, and 25% adversarial rule poisoning rates, verifying Ville's boundary condition.
5. **Camera-Ready Publication Artifacts**:
   - LaTeX tables: `reports/paper_tables.tex`
   - Markdown summary: `reports/paper_tables.md`
   - High-resolution figures (PDF + 300 DPI PNG): `reports/figures/fig1..fig4`

### How to verify your results are from real data:
Open `reports/drift_evaluation_manifest.json` and verify the `splits.source` property:
```json
"source": "paysim"
```
*(If it says `paysim`, your numbers are certified from real PaySim data and valid for publication).*

---

## Step 9: Launch the Interactive Streamlit Dashboard

Explore the trained system with a web-based dashboard:

```powershell
python main.py --ui
```
*(Or directly: `streamlit run app/app.py`)*

Open your browser at **`http://localhost:8501`**.

### Dashboard Tour:
* **⚡ Live Scoring Tab**: Enter custom transaction parameters (amount, sender, receiver, transaction type) or select predefined attack scenarios. View real-time decision badges (`APPROVE`, `STEP_UP`, `HOLD`, `BLOCK`), latency breakdown ($< 5\text{ ms}$), TreeSHAP feature attributions, and fired rules.
* **🕸 Mule Network Tab**: Inspect the 2-hop relational neighbourhood. Dashed red edges illustrate edges **pruned** by the label-aware similarity filter to prevent camouflage poisoning.
* **📄 Batch Scorer Tab**: Upload any transaction CSV file, score all rows in streaming batches with post-event leakage columns stripped automatically, and export the decisions.
* **📚 Knowledge Base Tab**: Real-time inspection of active, quarantined, and rejected detection rules, Subjective Logic opinions $(b, d, u, a)$, and rollback wealth trajectories.
* **🧭 Recovery Trace Tab**: Simulate post-incident recovery using Time-Decaying Tainted Proportional Flow (TD-TPF) to trace stolen funds through mule networks inside the 24-hour settlement window.

---

## Step 10: Run Batch Inference on Custom CSVs

Score any transaction file directly from the command line:

```powershell
python main.py --predict path/to/transactions.csv
```

* Automatically strips post-event leakage columns (e.g. `oldbalanceOrg`, `newbalanceOrig`) to preserve real-world validity.
* Saves `transactions_scored.csv` containing fraud probabilities $\hat{p}$, action recommendations, and explanations.

---

## Step 11: Cross-Dataset Generalization (IEEE-CIS)

To evaluate how MG-ECFM generalizes from mobile money (PaySim) to card-not-present e-commerce (IEEE-CIS):

```powershell
python main.py --mode full-pipeline --source ieee_cis --profile fast
```
The loader streams `train_transaction.csv` and `train_identity.csv` in `TransactionDT` chronological order.

---

## Troubleshooting & Common Questions

### Q1: `kaggle: 403 Client Error: Forbidden`
* **Cause**: You have not accepted the competition rules on Kaggle.
* **Solution**: Go to [kaggle.com/competitions/ieee-fraud-detection/rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) and click **"I Understand and Accept"**, then re-run `python scripts/download_datasets.py`.

### Q2: PowerShell says `running scripts is disabled on this system`
* **Cause**: Default Windows execution policy prevents running unapproved scripts.
* **Solution**: In PowerShell, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Q3: `MemoryError` or Out-of-Memory during PaySim training
* **Cause**: PaySim has 6.36 million rows; full in-memory arrays can exceed 8 GB RAM.
* **Solution**: Use `--profile fast` or set `--max-rows`:
  ```powershell
  python main.py --mode train --source paysim --profile fast
  # Or test with a smaller slice:
  python main.py --mode train --source paysim --profile fast --max-rows 500000
  ```

### Q4: How do I know if results were generated from real or synthetic data?
* **Solution**: Inspect `reports/drift_evaluation_manifest.json`:
  * `"source": "paysim"` $\rightarrow$ Certified from real PaySim data.
  * `"source": "synthetic_paysim(drift=0.5)"` $\rightarrow$ Synthetic generator.

### Q5: Can I run this on a GPU or an Apple Silicon Mac?
* **Solution**: Yes, cross-platform hardware acceleration is fully supported out of the box:
  * **Windows / Linux (NVIDIA CUDA)**: Install the PyTorch CUDA build:
    ```powershell
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
    XGBoost and PyTorch neural networks will automatically use your dedicated NVIDIA GPU.
  * **macOS (Apple Silicon M1/M2/M3/M4)**: Standard PyTorch from `pip install -r requirements.txt` automatically supports Metal Performance Shaders (`mps`). PyTorch tensors and modules are dispatched directly to the Apple Silicon GPU.
  * **Explicit Device Flag**: You can verify or force the compute target using `--device`:
    ```powershell
    python main.py --mode train --source paysim --profile fast --max-rows 500000 --device auto
    ```
    When `main.py` launches, it prints a hardware diagnostics banner detailing the active device name and memory.

