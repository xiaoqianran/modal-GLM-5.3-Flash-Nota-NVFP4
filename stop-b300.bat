@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
chcp 65001 >nul

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_NAME=glm53-flash-nota-b300"
set "APP_ID="
set "FOUND_B300=0"
set "STOP_FAILED=0"

set "APP_JSON=%TEMP%\glm53-apps-%RANDOM%-%RANDOM%.json"
set "CONTAINER_JSON=%TEMP%\glm53-containers-%RANDOM%-%RANDOM%.json"

echo [B300 STOP] Resolving deployed app...
uv run modal app list --json > "%APP_JSON%"
if errorlevel 1 goto :modal_error

for /f "usebackq delims=" %%A in (`uv run python -c "import json; apps=json.load(open(r'%APP_JSON%', encoding='utf-8')); print(next((x.get('app_id','') for x in apps if x.get('description')=='%APP_NAME%' and x.get('state')=='deployed'), ''))"`) do set "APP_ID=%%A"

if not defined APP_ID (
  echo [ERROR] No deployed app named %APP_NAME% was found.
  goto :cleanup_error
)

echo [INFO] App ID: %APP_ID%
echo [INFO] CPU cache workers will NOT be stopped.
echo.

uv run modal container list --app-id "%APP_ID%" --json > "%CONTAINER_JSON%"
if errorlevel 1 goto :modal_error

for /f "usebackq delims=" %%C in (`uv run python -c "import json; rows=json.load(open(r'%CONTAINER_JSON%', encoding='utf-8')); [print(x.get('container_id','')) for x in rows if x.get('container_id')]"`) do (
  call :inspect_and_stop "%%C"
)

del /q "%APP_JSON%" "%CONTAINER_JSON%" >nul 2>&1

if "%FOUND_B300%"=="0" (
  echo [OK] No running B300 container was found. The deployment remains active.
  pause
  exit /b 0
)

if not "%STOP_FAILED%"=="0" (
  echo [ERROR] One or more B300 containers could not be stopped.
  pause
  exit /b 1
)

echo.
echo [OK] Running B300 container stopped.
echo [OK] Modal deployment is still deployed.
echo [OK] CPU backup worker / schedule remains available.
pause
exit /b 0

:inspect_and_stop
set "CID=%~1"
set "GPU_INFO_FILE=%TEMP%\glm53-gpu-%RANDOM%-%RANDOM%.txt"

rem CPU cache containers have no NVIDIA GPU. Only stop a container that
rem successfully reports an NVIDIA B300 via nvidia-smi.
uv run modal container exec --no-pty "%CID%" nvidia-smi -L > "%GPU_INFO_FILE%" 2>&1
if errorlevel 1 (
  del /q "%GPU_INFO_FILE%" >nul 2>&1
  exit /b 0
)

findstr /i /c:"B300" "%GPU_INFO_FILE%" >nul
if errorlevel 1 (
  echo [SKIP] Container %CID% has a GPU, but it is not identified as B300.
  type "%GPU_INFO_FILE%"
  del /q "%GPU_INFO_FILE%" >nul 2>&1
  exit /b 0
)

set "FOUND_B300=1"
echo [FOUND] B300 container: %CID%
type "%GPU_INFO_FILE%"
del /q "%GPU_INFO_FILE%" >nul 2>&1

echo [STOP] Terminating B300 container %CID% only...
uv run modal container stop "%CID%" --graceful --yes
if errorlevel 1 set "STOP_FAILED=1"
exit /b 0

:modal_error
echo [ERROR] Failed to query Modal.
goto :cleanup_error

:cleanup_error
del /q "%APP_JSON%" "%CONTAINER_JSON%" >nul 2>&1
pause
exit /b 1
