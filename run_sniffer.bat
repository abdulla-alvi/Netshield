@echo off
REM Live sniffer: needs Npcap + usually "Run as administrator" in cmd.
REM Optional args:   run_sniffer.bat --iface "Wi-Fi"     run_sniffer.bat --bpf "ip and tcp"
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv missing.
    pause
    exit /b 1
)
venv\Scripts\python.exe live_sniffer.py %*
if errorlevel 1 pause
