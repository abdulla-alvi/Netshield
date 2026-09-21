@echo off
REM Moves to project folder automatically (fixes "wrong path" when cmd is elsewhere).
REM Double-click this file, or run:  setup_database.bat
REM Optional:                           setup_database.bat "your_mysql_password"

cd /d "%~dp0"
if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_database.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_database.ps1" -PlainPassword "%~1"
)
if errorlevel 1 pause
exit /b %errorlevel%
