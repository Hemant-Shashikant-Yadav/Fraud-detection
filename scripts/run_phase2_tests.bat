@echo off
REM MG-ECFM Phase 2 test runner: writes reports\phase2_test_log.txt
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase2_test_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
echo === environment === > %LOG%
%PY% --version >> %LOG% 2>&1
%PY% -c "import sys; print(sys.executable)" >> %LOG% 2>&1
%PY% -c "import numpy, pandas, sklearn; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__)" >> %LOG% 2>&1
%PY% -c "import torch; print('torch', torch.__version__)" >> %LOG% 2>&1
%PY% -c "import xgboost; print('xgboost', xgboost.__version__)" >> %LOG% 2>&1
%PY% -c "import torch_geometric; print('torch_geometric', torch_geometric.__version__)" >> %LOG% 2>&1
echo === phase 2 tests === >> %LOG%
%PY% tests\test_phase2_graph_detection.py >> %LOG% 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo === phase 1 regression === >> %LOG%
%PY% tests\test_phase1_data.py >> %LOG% 2>&1
echo PHASE1_EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo DONE >> %LOG%
