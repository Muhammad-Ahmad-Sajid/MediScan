@echo off
title Bone Fracture Diagnostics Server
echo =====================================================================
echo Starting Bone Fracture Detection & Prognosis Dashboard Backend Server...
echo =====================================================================
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Python virtual environment (venv) was not found!
    echo Please make sure you are in the correct directory.
    pause
    exit /b
)
call venv\Scripts\activate.bat
python main.py
pause
