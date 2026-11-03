@echo off
REM MG-ECFM Phase 8 dependencies (dashboard + figures); log: reports\phase8_install_log.txt
setlocal
cd /d "%~dp0\.."
if not exist reports mkdir reports
set PY=python
where python >nul 2>&1 || set PY=py
echo started %DATE% %TIME% > reports\phase8_install_log.txt
%PY% -m pip install "streamlit>=1.32" "matplotlib>=3.7" >> reports\phase8_install_log.txt 2>&1
echo PIP_EXIT_CODE=%ERRORLEVEL% >> reports\phase8_install_log.txt
%PY% -c "import streamlit, matplotlib; print('streamlit', streamlit.__version__, 'matplotlib', matplotlib.__version__)" >> reports\phase8_install_log.txt 2>&1
echo DONE >> reports\phase8_install_log.txt
