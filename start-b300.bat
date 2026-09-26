@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"

set "MODAL_WORKSPACE="
for /f "usebackq delims=" %%A in (`uv run modal profile current 2^>nul`) do set "MODAL_WORKSPACE=%%A"
if not defined MODAL_WORKSPACE goto :profile_error

set "PUBLIC_BASE_URL=https://%MODAL_WORKSPACE%--%APP_NAME%-vllmserver-serve.modal.run"
set "OPENAI_BASE_URL=%PUBLIC_BASE_URL%/v1"
set "MODELS_URL=%OPENAI_BASE_URL%/models"

echo [B300 START] Waking deployed B300 container...
echo [INFO] App: %APP_NAME%
echo [INFO] Endpoint: %MODELS_URL%
echo.

rem Any request to the deployed web endpoint causes Modal to allocate/wake VllmServer.
rem /v1/models is intentionally cheap and waits until the server is API-ready.
curl.exe --fail --show-error --silent --max-time 1800 "%MODELS_URL%"
set "RC=%ERRORLEVEL%"

echo.
echo.
if "%RC%"=="0" (
  echo [OK] B300 is active and the API is ready.
  echo [INFO] It will scale down automatically after 30 minutes without requests.
) else (
  echo [ERROR] Failed to wake or reach B300. Exit code: %RC%
  echo [INFO] Check logs with:
  echo        uv run modal app logs %APP_NAME%
)

pause
exit /b %RC%

:profile_error
echo.
echo [ERROR] Could not determine the active Modal profile.
pause
exit /b 1
