@echo off
REM MG-ECFM Phase 2 dependencies (CPU PyTorch, XGBoost, PyTorch Geometric), then runs the Phase 2 tests.
cd /d "%~dp0\.."
if not exist reports mkdir reports
set LOG=reports\phase2_install_log.txt
set PY=python
where python >nul 2>&1 || set PY=py
echo === install === > %LOG%
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu >> %LOG% 2>&1
%PY% -m pip install "xgboost>=2.0" "scikit-learn>=1.3" torch_geometric >> %LOG% 2>&1
echo INSTALL_EXIT_CODE=%ERRORLEVEL% >> %LOG%
echo INSTALL_DONE >> %LOG%
call "%~dp0run_phase2_tests.bat"
