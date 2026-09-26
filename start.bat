@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"
set "MODAL_WORKSPACE="
for /f "tokens=3" %%A in ('uv run modal profile current ^| findstr /b /c:"Active profile:"') do set "MODAL_WORKSPACE=%%A"
if not defined MODAL_WORKSPACE goto :profile_error

set "PUBLIC_BASE_URL=https://%MODAL_WORKSPACE%--%APP_NAME%-serve.modal.run"
set "OPENAI_BASE_URL=%PUBLIC_BASE_URL%/v1"
set "MODELS_URL=%OPENAI_BASE_URL%/models"
set "CHAT_URL=%OPENAI_BASE_URL%/chat/completions"

echo Starting or waking deployed B300 service...
echo.
echo [PUBLIC API]
echo Base URL:        %PUBLIC_BASE_URL%
echo OpenAI Base URL: %OPENAI_BASE_URL%
echo Models:          %MODELS_URL%
echo Chat:            %CHAT_URL%
echo.

curl.exe --fail --show-error --silent --max-time 1800 "%MODELS_URL%"
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

:profile_error
echo.
echo [ERROR] Could not determine the active Modal profile.
pause
exit /b 1
