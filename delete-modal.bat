@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"

echo Stopping deployed Modal app and all functions: %APP_NAME%
echo.

uv run modal app stop "%APP_NAME%" --yes
set "RC=%ERRORLEVEL%"

rem Migration cleanup: old releases used a second standalone cache app.
uv run modal app stop "glm53-cache-backup" --yes >nul 2>&1

echo.
if "%RC%"=="0" (
  echo [OK] Modal app stopped.
  echo [OK] B300 serving and CPU cache worker share the same app lifecycle.
) else (
  echo [ERROR] Failed to stop Modal app. Exit code: %RC%
)

pause
exit /b %RC%
