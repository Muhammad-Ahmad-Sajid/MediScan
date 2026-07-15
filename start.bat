@echo off
title Bone Fracture Diagnostics Launcher
color 0B
cd /d "%~dp0"

:: 1. Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found in this folder.
    echo Please make sure you are in the project root folder: %~dp0
    pause
    exit /b
)

:: 2. Activate Virtual Environment & Run launcher
call venv\Scripts\activate.bat
python launcher.py
pause
