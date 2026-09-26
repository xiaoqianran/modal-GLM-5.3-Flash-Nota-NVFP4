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
set "GLM53_MODAL_WORKSPACE=%MODAL_WORKSPACE%"
set "PUBLIC_BASE_URL=https://%MODAL_WORKSPACE%--%APP_NAME%-serve.modal.run"
set "OPENAI_BASE_URL=%PUBLIC_BASE_URL%/v1"

if not exist ".env" goto :env_error
findstr /r /c:"^HF_TOKEN=hf_.*" ".env" >nul || goto :hf_token_error
findstr /r /c:"^GITHUB_TOKEN=..*" ".env" >nul || goto :github_token_error

set "HF_SECRET_FILE=%TEMP%\glm53-hf-%RANDOM%-%RANDOM%.env"
set "GH_SECRET_FILE=%TEMP%\glm53-gh-%RANDOM%-%RANDOM%.env"
findstr /b "HF_TOKEN=" ".env" > "%HF_SECRET_FILE%"
findstr /b "GITHUB_TOKEN=" ".env" > "%GH_SECRET_FILE%"

echo [1/7] Sync Hugging Face secret...
uv run modal secret create --from-dotenv "%HF_SECRET_FILE%" --force huggingface || goto :failed

echo.
echo [2/7] Sync GitHub secret...
uv run modal secret create --from-dotenv "%GH_SECRET_FILE%" --force github || goto :failed

del /q "%HF_SECRET_FILE%" 2>nul
del /q "%GH_SECRET_FILE%" 2>nul

echo.
echo [3/7] Ensure model is cached in Modal Volume...
uv run modal run app.py::step_001_download || goto :failed

echo.
echo [4/7] Restore runtime caches: Modal Volume first, GitHub Release fallback...
uv run modal run app.py::step_009_restore_runtime_caches || echo [WARN] Runtime cache restore failed; GPU startup will regenerate missing cache.

echo.
echo [5/7] Compact Triton and TorchInductor seeds into single archives...
uv run modal run app.py::step_013_compact_staged_archives || echo [WARN] Staged cache compaction failed; deployment will continue with fallback staging.

echo.
echo [6/7] Ensure runtime caches are backed up to GitHub Release...
uv run modal run app.py::step_010_publish_runtime_caches || echo [WARN] GitHub cache backup failed; deployment will continue.

echo.
echo [7/7] Deploy single-B300 service...
uv run modal deploy app.py || goto :failed

echo.
echo [OK] Deployment completed.
echo.
echo [PUBLIC API]
echo Base URL:        %PUBLIC_BASE_URL%
echo OpenAI Base URL: %OPENAI_BASE_URL%
echo Models:          %OPENAI_BASE_URL%/models
echo Chat:            %OPENAI_BASE_URL%/chat/completions
echo.
echo Run start.bat to start or wake the deployed service.
pause
exit /b 0

:profile_error
echo.
echo [ERROR] Could not determine the active Modal profile.
pause
exit /b 1

:env_error
echo.
echo [ERROR] .env was not found in:
echo %CD%
pause
exit /b 1

:hf_token_error
echo.
echo [ERROR] Put HF_TOKEN=hf_xxx in .env first.
pause
exit /b 1

:github_token_error
echo.
echo [ERROR] Put GITHUB_TOKEN=github_xxx in .env first.
pause
exit /b 1

:failed
set "RC=%ERRORLEVEL%"
if defined HF_SECRET_FILE del /q "%HF_SECRET_FILE%" 2>nul
if defined GH_SECRET_FILE del /q "%GH_SECRET_FILE%" 2>nul
echo.
echo [ERROR] Deployment failed. Exit code: %RC%
pause
exit /b %RC%
