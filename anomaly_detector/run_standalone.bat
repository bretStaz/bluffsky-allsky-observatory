@echo off
cd /d "%~dp0"
if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" detector.py --daemon
) else if exist "%~dp0..\venv\Scripts\python.exe" (
    "%~dp0..\venv\Scripts\python.exe" detector.py --daemon
) else (
    python detector.py --daemon
)
