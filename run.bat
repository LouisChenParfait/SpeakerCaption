@echo off
setlocal
cd /d "%~dp0"
rem Keep Python output UTF-8 without calling chcp: chcp resets the console
rem handle and can swallow stdout when the script is launched from PowerShell.
rem If Chinese text looks garbled in a cmd window, run "chcp 65001" first.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%~1"=="" (
  echo SpeakerCaption - offline speaker diarized transcription
  echo Usage:   run.bat ^<audio-file^> [options]
  echo Example: run.bat podcast.wav --num-speakers 2 --language zh
  echo Example: run.bat meeting.m4a --num-speakers 6 --asr confucius
  echo Full options: python diarize.py --help   or see README.md
  exit /b 1
)
"%PY%" diarize.py %*
endlocal
