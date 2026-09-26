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
set "VERIFY_FAILED=0"

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

rem Kill only the local wake-up curl started by start-b300.bat.
rem Otherwise Modal may see the still-open HTTP input and immediately
rem reschedule it onto a fresh B300 after we terminate the current one.
echo [LOCAL] Cancelling any pending start-b300 wake request...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ps = Get-CimInstance Win32_Process | Where-Object { $_.Name -ieq 'curl.exe' -and $_.CommandLine -like '*%APP_NAME%*v1/models*' }; foreach ($p in $ps) { Write-Host ('[LOCAL] Killing curl PID ' + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"

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

call :verify_b300_gone
if not "%VERIFY_FAILED%"=="0" (
  echo [ERROR] A B300 container is still alive after stop attempts.
  echo [ERROR] Deployment was NOT stopped. Inspect the remaining Modal input/request.
  pause
  exit /b 1
)

echo.
echo [OK] No live B300 container remains.
echo [OK] Modal deployment is still deployed.
echo [OK] CPU backup worker / schedule remains available.
pause
exit /b 0

:inspect_and_stop
set "CID=%~1"
set "GPU_INFO_FILE=%TEMP%\glm53-gpu-%RANDOM%-%RANDOM%.txt"
set "CONTAINER_LOG_FILE=%TEMP%\glm53-container-%RANDOM%-%RANDOM%.log"
set "IS_B300=0"

rem CPU cache containers have no NVIDIA GPU. Only stop a container that
rem successfully reports an NVIDIA B300 via nvidia-smi.
uv run modal container exec --no-pty "%CID%" -- nvidia-smi -L > "%GPU_INFO_FILE%" 2>&1
if not errorlevel 1 (
  findstr /i /c:"B300" "%GPU_INFO_FILE%" >nul
  if not errorlevel 1 set "IS_B300=1"
)

rem If exec is temporarily unavailable during startup, fall back to runtime
rem logs so the B300 serving container can still be identified.
if "%IS_B300%"=="0" (
  uv run modal container logs "%CID%" --tail 200 > "%CONTAINER_LOG_FILE%" 2>&1
  if not errorlevel 1 (
    findstr /i /c:"[RUNTIME_START]" /c:"[003_VLLM_PROCESS_START]" /c:"[007_API_READY]" "%CONTAINER_LOG_FILE%" >nul
    if not errorlevel 1 set "IS_B300=1"
  )
)

if "%IS_B300%"=="0" (
  del /q "%GPU_INFO_FILE%" "%CONTAINER_LOG_FILE%" >nul 2>&1
  exit /b 0
)

set "FOUND_B300=1"
echo [FOUND] B300 container: %CID%
if exist "%GPU_INFO_FILE%" type "%GPU_INFO_FILE%"
del /q "%GPU_INFO_FILE%" "%CONTAINER_LOG_FILE%" >nul 2>&1

echo [STOP] Terminating B300 container %CID% only...
uv run modal container stop "%CID%" --yes
if errorlevel 1 set "STOP_FAILED=1"
exit /b 0

:verify_b300_gone
set "VERIFY_ATTEMPT=0"

:verify_loop
set /a VERIFY_ATTEMPT+=1
set "LIVE_B300=0"
set "VERIFY_JSON=%TEMP%\glm53-verify-%RANDOM%-%RANDOM%.json"

rem Give Modal control-plane state a short moment to settle.
>nul 2>&1 ping 127.0.0.1 -n 3

uv run modal container list --app-id "%APP_ID%" --json > "%VERIFY_JSON%"
if errorlevel 1 (
  del /q "%VERIFY_JSON%" >nul 2>&1
  set "VERIFY_FAILED=1"
  exit /b 0
)

for /f "usebackq delims=" %%C in (`uv run python -c "import json; rows=json.load(open(r'%VERIFY_JSON%', encoding='utf-8')); [print(x.get('container_id','')) for x in rows if x.get('container_id')]" `) do (
  call :verify_one_container "%%C"
)
del /q "%VERIFY_JSON%" >nul 2>&1

if "%LIVE_B300%"=="0" (
  echo [VERIFY] No live B300 found.
  exit /b 0
)

if %VERIFY_ATTEMPT% GEQ 5 (
  set "VERIFY_FAILED=1"
  exit /b 0
)

echo [VERIFY] B300 still alive; retrying stop ^(%VERIFY_ATTEMPT%/5^)...
goto :verify_loop

:verify_one_container
set "VERIFY_CID=%~1"
set "VERIFY_GPU=%TEMP%\glm53-verify-gpu-%RANDOM%-%RANDOM%.txt"

uv run modal container exec --no-pty "%VERIFY_CID%" -- nvidia-smi -L > "%VERIFY_GPU%" 2>&1
if errorlevel 1 (
  rem A stopped/detached container may linger briefly in container list.
  rem If exec is impossible, it is not an actively usable B300.
  del /q "%VERIFY_GPU%" >nul 2>&1
  exit /b 0
)

findstr /i /c:"B300" "%VERIFY_GPU%" >nul
if errorlevel 1 (
  del /q "%VERIFY_GPU%" >nul 2>&1
  exit /b 0
)

set "LIVE_B300=1"
echo [VERIFY] Live B300: %VERIFY_CID%
type "%VERIFY_GPU%"
del /q "%VERIFY_GPU%" >nul 2>&1
uv run modal container stop "%VERIFY_CID%" --yes >nul 2>&1
exit /b 0

:modal_error
echo [ERROR] Failed to query Modal.
goto :cleanup_error

:cleanup_error
del /q "%APP_JSON%" "%CONTAINER_JSON%" >nul 2>&1
pause
exit /b 1
