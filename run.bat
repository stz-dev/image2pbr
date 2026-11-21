@echo off
REM Image2PBR launcher for Windows: creates a virtual environment on first run, then starts the app.
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt || (pause & exit /b 1)
) else (
    call .venv\Scripts\activate.bat
)
python app.py
pause
