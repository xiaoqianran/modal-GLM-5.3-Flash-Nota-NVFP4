@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"

echo Stopping deployed Modal app: %APP_NAME%
echo.

uv run modal app stop "%APP_NAME%" --yes
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo [OK] Modal app stopped.
) else (
  echo [ERROR] Failed to stop Modal app. Exit code: %RC%
)

pause
exit /b %RC%
