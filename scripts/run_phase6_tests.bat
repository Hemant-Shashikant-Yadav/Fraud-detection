@echo off
REM MG-ECFM Phase 6 test runner (+ Phase 1-5 regression): writes reports\phase6_test_log.txt
setlocal
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase6_test_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
set OVERALL=0

echo === environment === > %LOG%
echo started %DATE% %TIME% >> %LOG%
%PY% --version >> %LOG% 2>&1
%PY% -c "import sys; print(sys.executable)" >> %LOG% 2>&1
%PY% -c "import numpy, pandas, sklearn; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)" >> %LOG% 2>&1
%PY% -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads())" >> %LOG% 2>&1

echo === phase 6 tests (baselines BP1-BP4) === >> %LOG%
%PY% tests\test_phase6_baselines.py >> %LOG% 2>&1
set RC6=%ERRORLEVEL%
echo PHASE6_EXIT_CODE=%RC6% >> %LOG%
if not "%RC6%"=="0" set OVERALL=1

echo === phase 5 regression === >> %LOG%
%PY% tests\test_phase5_recovery_closedloop.py >> %LOG% 2>&1
set RC5=%ERRORLEVEL%
echo PHASE5_EXIT_CODE=%RC5% >> %LOG%
if not "%RC5%"=="0" set OVERALL=1

echo === phase 4 regression === >> %LOG%
%PY% tests\test_phase4_epistemic_martingale.py >> %LOG% 2>&1
set RC4=%ERRORLEVEL%
echo PHASE4_EXIT_CODE=%RC4% >> %LOG%
if not "%RC4%"=="0" set OVERALL=1

echo === phase 3 regression === >> %LOG%
%PY% tests\test_phase3_conformal.py >> %LOG% 2>&1
set RC3=%ERRORLEVEL%
echo PHASE3_EXIT_CODE=%RC3% >> %LOG%
if not "%RC3%"=="0" set OVERALL=1

echo === phase 2 regression === >> %LOG%
%PY% tests\test_phase2_graph_detection.py >> %LOG% 2>&1
set RC2=%ERRORLEVEL%
echo PHASE2_EXIT_CODE=%RC2% >> %LOG%
if not "%RC2%"=="0" set OVERALL=1

echo === phase 1 regression === >> %LOG%
%PY% tests\test_phase1_data.py >> %LOG% 2>&1
set RC1=%ERRORLEVEL%
echo PHASE1_EXIT_CODE=%RC1% >> %LOG%
if not "%RC1%"=="0" set OVERALL=1

echo === summary === >> %LOG%
findstr /B /C:"Ran " %LOG% >> "%TEMP%\mg_ecfm_phase6_summary.txt"
type "%TEMP%\mg_ecfm_phase6_summary.txt" >> %LOG%
del "%TEMP%\mg_ecfm_phase6_summary.txt" >nul 2>&1
echo P6=%RC6% P5=%RC5% P4=%RC4% P3=%RC3% P2=%RC2% P1=%RC1% >> %LOG%
echo OVERALL_EXIT_CODE=%OVERALL% >> %LOG%
echo finished %DATE% %TIME% >> %LOG%
echo DONE >> %LOG%
type %LOG% | findstr /C:"EXIT_CODE" /C:"Ran "
exit /b %OVERALL%
