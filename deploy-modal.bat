@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".env" goto :env_error
findstr /r /c:"^HF_TOKEN=hf_.*" ".env" >nul || goto :token_error

echo [1/3] Sync Hugging Face secret...
uv run modal secret create --from-dotenv .env --force huggingface || goto :failed

echo.
echo [2/3] Ensure model is cached in Modal Volume...
uv run modal run app.py::step_001_download || goto :failed

echo.
echo [3/3] Deploy single-B300 service...
uv run modal deploy app.py || goto :failed

echo.
echo [OK] Deployment completed.
echo Run start.bat to start or wake the deployed service.
pause
exit /b 0

:env_error
echo.
echo [ERROR] .env was not found in:
echo %CD%
pause
exit /b 1

:token_error
echo.
echo [ERROR] Put HF_TOKEN=hf_xxx in .env first.
pause
exit /b 1

:failed
set "RC=%ERRORLEVEL%"
echo.
echo [ERROR] Deployment failed. Exit code: %RC%
pause
exit /b %RC%
