@echo off
REM MG-ECFM Phase 3 test runner (+ Phase 1/2 regression): writes reports\phase3_test_log.txt
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase3_test_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
echo === environment === > %LOG%
%PY% --version >> %LOG% 2>&1
%PY% -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" >> %LOG% 2>&1
echo === phase 3 tests === >> %LOG%
%PY% tests\test_phase3_conformal.py >> %LOG% 2>&1
echo PHASE3_EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo === phase 2 regression === >> %LOG%
%PY% tests\test_phase2_graph_detection.py >> %LOG% 2>&1
echo PHASE2_EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo === phase 1 regression === >> %LOG%
%PY% tests\test_phase1_data.py >> %LOG% 2>&1
echo PHASE1_EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo DONE >> %LOG%
