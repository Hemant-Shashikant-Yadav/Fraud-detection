@echo off
REM MG-ECFM Phase 8 test runner: orchestrator quick demo, camera-ready artifacts, dashboard
REM (+ Phase 1-7 regression). Writes reports\phase8_test_log.txt
REM Usage:  scripts\run_phase8_tests.bat            (Phase 8 tests + Phase 1-7 regression)
REM         scripts\run_phase8_tests.bat quick      (Phase 8 tests only)
REM         scripts\run_phase8_tests.bat artifacts  (also rebuild reports\paper_tables.* and reports\figures\)
setlocal
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase8_test_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
set PYTHONIOENCODING=utf-8
set OVERALL=0
set RC1=skipped
set RC2=skipped
set RC3=skipped
set RC4=skipped
set RC5=skipped
set RC6=skipped
set RC7=skipped
set RCA=skipped

echo === environment === > %LOG%
echo started %DATE% %TIME% >> %LOG%
%PY% --version >> %LOG% 2>&1
%PY% -c "import sys; print(sys.executable)" >> %LOG% 2>&1
%PY% -c "import numpy, pandas, sklearn, matplotlib; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'matplotlib', matplotlib.__version__)" >> %LOG% 2>&1
%PY% -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())" >> %LOG% 2>&1
%PY% -c "import xgboost; print('xgboost', xgboost.__version__)" >> %LOG% 2>&1
%PY% -c "import streamlit; print('streamlit', streamlit.__version__)" >> %LOG% 2>&1 || echo streamlit NOT INSTALLED - dashboard render test will be skipped: pip install streamlit >> %LOG%

echo === phase 8 tests (quick demo end-to-end, tables, figures, CLI, live service, dashboard) === >> %LOG%
%PY% tests\test_phase8_orchestrator.py >> %LOG% 2>&1
set RC8=%ERRORLEVEL%
echo PHASE8_EXIT_CODE=%RC8% >> %LOG%
if not "%RC8%"=="0" set OVERALL=1

echo === quick demo manifest checks === >> %LOG%
%PY% -c "import json; m=json.load(open(r'reports\quick_demo\quick_demo_manifest.json',encoding='utf-8')); [print(('PASS' if c['passed'] else 'FAIL'), '|', c['check'], '|', c['detail']) for c in m['checks']]; print('QUICK_DEMO_RUNTIME_S', round(m['runtime_seconds'],1)); print('QUICK_DEMO_ALL_PASSED', m['all_passed'])" >> %LOG% 2>&1

if /I "%~1"=="quick" goto summary

for %%P in (7 6 5 4 3 2 1) do call :regress %%P

if /I not "%~1"=="artifacts" goto summary
echo === camera-ready artifacts from reports\ === >> %LOG%
%PY% main.py --mode paper-artifacts >> %LOG% 2>&1
set RCA=%ERRORLEVEL%
if not "%RCA%"=="0" set OVERALL=1

:summary
echo === summary === >> %LOG%
%PY% -c "import re,io; t=io.open(r'%LOG%',encoding='utf-8',errors='replace').read(); n=[int(x) for x in re.findall(r'^Ran (\d+) tests? in', t, re.M)]; s=[int(x) for x in re.findall(r'skipped=(\d+)', t)]; print('RUN_COUNTS', n); print('TOTAL_TESTS_RUN', sum(n)); print('TOTAL_SKIPPED', sum(s))" >> %LOG% 2>&1
echo P8=%RC8% P7=%RC7% P6=%RC6% P5=%RC5% P4=%RC4% P3=%RC3% P2=%RC2% P1=%RC1% ARTIFACTS=%RCA% >> %LOG%
echo OVERALL_EXIT_CODE=%OVERALL% >> %LOG%
echo finished %DATE% %TIME% >> %LOG%
echo DONE >> %LOG%
type %LOG% | findstr /C:"EXIT_CODE" /C:"Ran " /C:"TOTAL_" /C:"FAILED" /C:"OK" /C:"QUICK_DEMO"
exit /b %OVERALL%

:regress
set FILE=
if "%1"=="7" set FILE=tests\test_phase7_experiments.py
if "%1"=="6" set FILE=tests\test_phase6_baselines.py
if "%1"=="5" set FILE=tests\test_phase5_recovery_closedloop.py
if "%1"=="4" set FILE=tests\test_phase4_epistemic_martingale.py
if "%1"=="3" set FILE=tests\test_phase3_conformal.py
if "%1"=="2" set FILE=tests\test_phase2_graph_detection.py
if "%1"=="1" set FILE=tests\test_phase1_data.py
echo === phase %1 regression === >> %LOG%
%PY% %FILE% >> %LOG% 2>&1
set RC%1=%ERRORLEVEL%
echo PHASE%1_EXIT_CODE=%ERRORLEVEL% >> %LOG%
if errorlevel 1 set OVERALL=1
goto :eof
