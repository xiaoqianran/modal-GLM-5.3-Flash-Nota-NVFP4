@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"
set "URL=https://zhiyuqqq--glm53-flash-nota-b300-serve.modal.run/v1/models"

echo Starting or waking deployed B300 service...
echo %URL%
echo.

curl.exe --fail --show-error --silent --max-time 1800 "%URL%"
set "RC=%ERRORLEVEL%"

echo.
echo.
if "%RC%"=="0" (
  echo [OK] Service is ready.
  echo [INFO] B300 scaledown window: 30 minutes after the last request.
) else (
  echo [ERROR] Failed to start or reach deployed service. Exit code: %RC%
  echo [INFO] Check deployment and logs with:
  echo        uv run modal app logs %APP_NAME%
)

pause
exit /b %RC%
