@echo off
REM Fixes common PyTorch WinError 1114 (c10.dll) on Windows by repairing VC++ runtime, then verifies import.
echo [1/3] Repairing Microsoft Visual C++ 2015-2022 (x64)... Accept UAC if asked.
winget install --id Microsoft.VCRedist.2015+.x64 -e --accept-package-agreements --accept-source-agreements
echo.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo ERROR: venv not found. Create venv first.
  pause & exit /b 1
)
echo [2/3] Reinstalling CPU-only PyTorch wheel (optional refresh)...
venv\Scripts\pip.exe install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
echo.
echo [3/3] Testing import...
venv\Scripts\python.exe -c "import torch; print('PyTorch OK:', torch.__version__)"
if errorlevel 1 (
  echo FAILED. Reboot PC, then run this script again.
) else (
  echo You can run: venv\Scripts\python.exe live_sniffer.py
)
pause
