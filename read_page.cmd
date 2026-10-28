@echo off
rem Read a scanned page with every reader this computer has: drag a PDF / JPG / PNG onto this file.
rem Hebrew in context, Uni-MuMER and the English VLM are used when downloaded (gpu_eval.cmd downloads them).
rem The result: results\<name>_p1_read.png (+ .txt, with the derivation checks) - it opens by itself.
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  echo Drag a scanned page ^(PDF, JPG or PNG^) onto read_page.cmd
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing - run gpu_eval.cmd or setup.cmd first
  pause
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
set PYTHONWARNINGS=ignore
if exist "models\dictalm-3.0-1.7b-base\model.safetensors" set EXAM_GRADER_HEBREW_CONTEXT=1
if exist "models\uni-mumer-qwen3vl-2b\model.safetensors" set EXAM_GRADER_FORMULA2=mumer
set Q=models\qwen3-vl-8b\model-0000
if exist "%Q%1-of-00004.safetensors" if exist "%Q%2-of-00004.safetensors" if exist "%Q%3-of-00004.safetensors" if exist "%Q%4-of-00004.safetensors" set EXAM_GRADER_ENGLISH_VLM=1
".venv\Scripts\python.exe" read_page.py "%~1" --check
for %%F in ("%~1") do set "NAME=%%~nF"
if exist "results\%NAME%_p1_read.png" start "" "results\%NAME%_p1_read.png"
pause
