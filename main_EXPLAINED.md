# Code Walkthrough: `main.py`

**Purpose:** The **single command-line entry point** of the project. Every stage — quick demo, training the live service, the 50-window benchmark, ablation, stress test, paper tables/figures, batch scoring and the dashboard — is one `--mode` or flag.

```bash
python main.py --mode quick-demo                       # ~40 s smoke run on synthetic data
python main.py --mode train --source paysim --profile full
python main.py --mode full-pipeline --source paysim --profile full
python main.py --predict some_transactions.csv
python main.py --ui
```

---

## 1. Section-by-section

### 1.1 Docstring and imports (lines 1–73)
Documents the modes (`quick-demo`, `train`, `drift-eval`, `ablation`, `stress-test`, `full-pipeline`, `paper-artifacts`) and the `--predict` / `--ui` flags. Defines `ROOT`, adds it to `sys.path`, and sets constants: `SEED=42`, output directories (`reports/`, `models/`, `reports/quick_demo/`), service paths (`models/mg_ecfm_service.pkl` for training, `models/quick_demo_service.pkl` for the demo), `APP_PATH`, and the 60 s quick-demo budget.

### 1.2 Quick-demo configuration (lines ~26–45)
`QUICK_STREAM` (160 steps, 900 customers, 3 % fraud, drift 0.5), `QUICK_WINDOWS = 10`, and `QUICK_BASELINE_CONFIGS` (same architectures with far fewer epochs). The published numbers come from the full modes, **not** from the quick demo.

### 1.3 Helpers (lines 75–106)
`_jsonable` (reuses the drift-eval converter), `_print_header`, and `StageTimer` — wall-clock bookkeeping that prints `[stage] … done in X s` and stores per-stage seconds.

### 1.4 `make_sample_transactions` (lines 109–116)
Writes a small PaySim-schema CSV (including the leakage columns, which the scorer strips) used by the demo and the batch scorer.

### 1.5 `run_quick_demo` (lines 119–266)
Runs the whole pipeline on a small **synthetic** stream with timed stages: causal stream + splits → fit MG-ECFM + BP1–BP4 → prequential drift evaluation → statistical significance → B1–B5 ablation → adversarial stress test → live service (real-time scoring, latency profile on 200 transactions, camouflage graph, batch CSV, TD-TPF trace) → save/load round trip → paper tables and figures. A `check(name, ok, detail)` helper records pass/fail checks (e.g. "predictions[mg_ecfm]: n scored, mean PR-AUC"); everything is written to `reports/quick_demo/` with a manifest (`quick_demo_manifest.json`) that records runtime, stage seconds and `all_passed`.

### 1.6 Train / predict / artifacts (lines 272–349)
* `run_train(profile, source, output, seed, max_rows)`: builds `ServiceConfig`, fits the live `MGECFMService`, runs a 200-transaction latency benchmark, saves the pickle and a JSON report.
* `load_or_build_service`: uses a trained service if present (`--model`, default, or the quick-demo one), otherwise builds a demo service on the fly and caches it.
* `run_predict`: scores a CSV (PaySim or canonical schema) and writes `<name>_scored.csv`.
* `run_paper_artifacts(reports_dir)`: regenerates `paper_tables.{tex,md}` and `figures/`.
* `run_full_pipeline(args)`: drift-eval → significance → ablation → stress test → paper artifacts (5 steps, each with a header and timing).

### 1.7 Dashboard launcher (lines 352–366)
`launch_ui(port, model_path)`: checks that Streamlit is installed, passes the model path through `MG_ECFM_SERVICE`, and runs `streamlit run app/app.py`.

### 1.8 CLI (lines 372–460)
* `MODES` and `build_parser()`: arguments `--mode`, `--predict`, `--ui`, `--output`, `--model`, `--profile (demo|fast|full)`, `--source (auto|synthetic|paysim|ieee_cis)`, `--n-windows`, `--n-steps`, `--drift-strength`, `--max-rows`, `--reports-dir`, `--port`, `--seed`.
* `main(argv)`: dispatches each mode. `--mode train` resolves the data source with **`resolve_source`** (so `auto` raises if PaySim is missing instead of silently using synthetic data); all other experiment modes get the same guard through `prepare_splits`. `KeyboardInterrupt` → exit 130; any other exception prints a traceback and returns 1 so batch runners/CI see failures.

## 2. Things to remember
* `--source auto` ⇒ PaySim, or an error. Use `--source synthetic` only for smoke tests.
* `--profile full` uses the published hyper-parameters; `fast` is the reduced profile used for the PaySim-2M run reported so far (`profile: fast` in the manifest).
