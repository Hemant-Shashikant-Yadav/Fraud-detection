# Code Walkthrough: Windows batch scripts (`run_phase1.bat` and `scripts/*.bat`)

These small `.bat` files are **convenience wrappers** for Windows (Command Prompt). They share one pattern, so they are explained together instead of one file each.

## Common pattern
```bat
@echo off
cd /d "%~dp0\.."                       :: go to the project root, whatever folder you launch from
if not exist reports mkdir reports     :: make sure the log folder exists
set LOG=reports\phaseN_..._log.txt     :: every script writes a log here
set PY=python
where python >nul 2>&1 || set PY=py    :: fall back to the Windows "py" launcher
%PY% <command> >> %LOG% 2>&1           :: run, appending stdout+stderr to the log
echo EXIT_CODE=%ERRORLEVEL% >> %LOG%   :: record the exit code
```
Logs are written to `reports/phaseN_*_log.txt`; the earlier logs in `reports/` came from these scripts.

## File by file
| File | What it does |
|---|---|
| `run_phase1.bat` | Runs `scripts\run_phase1_demo.py` (the Phase-1 live demo) and prints `[SUCCESS]` or `[ERROR]`, then `pause`s. Header carries the candidate/guide information. |
| `scripts\install_phase2_deps.bat` | Installs CPU PyTorch, XGBoost ≥ 2, scikit-learn ≥ 1.3 and `torch_geometric`, logs to `phase2_install_log.txt`, then calls `run_phase2_tests.bat`. |
| `scripts\install_phase8_deps.bat` | Installs `streamlit ≥ 1.32` and `matplotlib ≥ 3.7`, prints their versions into `phase8_install_log.txt`. |
| `scripts\run_phase2_tests.bat` | Logs the environment (Python, numpy, pandas, sklearn, torch, xgboost, torch_geometric versions), runs `tests\test_phase2_graph_detection.py`, then the Phase-1 regression; log `phase2_test_log.txt`. |
| `scripts\run_phase3_tests.bat` | Phase-3 (conformal) tests plus regression of earlier phases. |
| `scripts\run_phase4_tests.bat` | Phase-4 (epistemic gate / martingale) tests, then Phase 3, 2 and 1 regressions (each with its own `PHASEn_EXIT_CODE`). |
| `scripts\run_phase5_tests.bat` | Phase-5 (recovery and closed loop) tests plus regression. |
| `scripts\run_phase6_tests.bat` | Phase-6 (baselines BP1–BP4) tests, then Phase-5 … regression; sets `OVERALL=1` if any step fails. |
| `scripts\run_phase7_tests.bat` | Phase-7 tests (drift evaluation, ablation, adversarial stress, significance; SciPy used only to cross-check) plus Phase 1–6 regression. With the argument `experiments` it then runs the four Phase-7 experiments. Sets `PYTHONIOENCODING=utf-8`. |
| `scripts\run_phase7_experiments.bat` | Runs, in order: `experiments\run_drift_eval.py %*`, `statistical_significance.py`, `run_ablation_study.py %*`, `run_adversarial_stress_test.py`; any extra arguments (e.g. `--source paysim --max-rows 2000000`) are forwarded to the drift and ablation scripts; writes `phase7_experiments_log.txt`; exits with `OVERALL`. **With the no-silent-synthetic guard, running it without arguments now needs the PaySim file — pass `--source synthetic` for a smoke run.** |
| `scripts\run_phase8_tests.bat` | Logs the environment, runs `tests\test_phase8_orchestrator.py` (quick demo, tables, figures, CLI, service, dashboard), prints the quick-demo checks from its manifest, optionally rebuilds `paper_tables.*` and figures (argument `artifacts`), then regression of Phases 1–7; summarises `TOTAL_TESTS_RUN` / `TOTAL_SKIPPED` from the log. |
| `scripts\run_phase8_quick.bat` | One line: calls `run_phase8_tests.bat quick` (Phase-8 tests only). |

## Notes
* They are **Windows-only** conveniences; on macOS/Linux run the same Python commands directly (e.g. `python tests/test_phase8_orchestrator.py`).
* Logs, not console output, are the record — look in `reports/`.
