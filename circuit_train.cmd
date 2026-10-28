@echo off
chcp 65001 >nul
cd /d "%~dp0"
where pwsh >nul 2>nul
if %errorlevel%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0circuit_train.ps1"
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0circuit_train.ps1"
)
echo.
echo Log: %~dp0results\circuit_train.log
pause
