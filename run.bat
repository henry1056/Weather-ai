@echo off
setlocal enabledelayedexpansion
title Weather-AI Studio Launcher

echo ============================================================
echo               Weather-AI Studio Launcher
echo ============================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ from https://www.python.org/ and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: 2. Set up virtual environment if not present
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Virtual environment created successfully.
)

:: 3. Activate virtual environment
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: 4. Install / verify dependencies
echo [*] Checking and installing dependencies from requirements.txt...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install required packages.
    pause
    exit /b 1
)

:: 5. Launch the Streamlit GUI
echo.
echo ============================================================
echo [*] Starting Weather-AI Studio GUI on Streamlit...
echo [*] Press Ctrl+C in this terminal to stop the application.
echo ============================================================
echo.

streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Streamlit exited with an error code: %errorlevel%
)

pause
