@echo off
REM ============================================================================
REM MG-ECFM Phase 1 Live Execution & Demonstration Script
REM Candidate: Shradha Vijay Kolekar (URN: 1025230008)
REM Guide: Dr. Anisa B. Shikalgar | ADCET Ashta
REM ============================================================================

echo [INFO] Starting MG-ECFM Phase 1 Live Execution...
python scripts\run_phase1_demo.py
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Phase 1 Execution Completed Successfully.
) else (
    echo [ERROR] Phase 1 Execution Encountered an Issue.
)
pause

