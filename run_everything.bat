@echo off
REM Idempotent DB + model registration + health check (correct path even when launched from elsewhere)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_everything.ps1"
pause
