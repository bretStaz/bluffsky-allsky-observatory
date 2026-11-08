@echo off
cd /d "%~dp0"
if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" allsky_workflow_guard.py --once
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    "%~dp0..\venv\Scripts\python.exe" allsky_workflow_guard.py --once
) else (
    python allsky_workflow_guard.py --once
)
