@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "NO_PROXY=127.0.0.1,localhost,::1,0.0.0.0"
set "no_proxy=%NO_PROXY%"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto RUN
echo [ERROR] .venv not found.
echo          python -m venv .venv
echo          .venv\Scripts\python.exe -m pip install -r requirements.txt
exit /b 1
:RUN
".venv\Scripts\python.exe" -u live.py %*
endlocal
