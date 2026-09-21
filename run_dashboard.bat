@echo off
REM Double-click this or run from cmd.exe - no activation needed (uses venv Python).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv missing. Run: python -m venv venv
    echo Then: python -m pip install --upgrade pip
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)
venv\Scripts\python.exe -m streamlit run soc_dashboard.py
if errorlevel 1 pause
