@echo off
REM MG-ECFM Phase 7 experiments: 50-window drift evaluation, significance tests,
REM 5-point ablation and adversarial stress test. Writes reports\phase7_experiments_log.txt
setlocal
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase7_experiments_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
set PYTHONIOENCODING=utf-8
set OVERALL=0
echo started %DATE% %TIME% > %LOG%
echo === drift evaluation (50 windows) === >> %LOG%
%PY% experiments\run_drift_eval.py %* >> %LOG% 2>&1
if errorlevel 1 set OVERALL=1
echo === statistical significance === >> %LOG%
%PY% experiments\statistical_significance.py >> %LOG% 2>&1
if errorlevel 1 set OVERALL=1
echo === ablation study === >> %LOG%
%PY% experiments\run_ablation_study.py %* >> %LOG% 2>&1
if errorlevel 1 set OVERALL=1
echo === adversarial stress test === >> %LOG%
%PY% experiments\run_adversarial_stress_test.py >> %LOG% 2>&1
if errorlevel 1 set OVERALL=1
echo OVERALL_EXIT_CODE=%OVERALL% >> %LOG%
echo finished %DATE% %TIME% >> %LOG%
echo DONE >> %LOG%
exit /b %OVERALL%
