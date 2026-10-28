@echo off
rem Read every student's exam (not the *_checked copies) with the exam-grader readers - fully local, on this computer's
rem GPU; with more than one GPU (as nvidia-smi lists them) one reader per GPU, each on its share of the students, the
rem others in windows of their own. Progress is printed as it goes and kept in the results folder's progress.log. It can
rem be stopped and run again: the students already read are skipped. Results: %EXAMS%\grader_runs (outside this folder)
chcp 65001 >nul
setlocal EnableDelayedExpansion
set "PROJ=%~dp0"
set "EXAMS=C:\Users\benariet\Desktop\EXAM_TEST_FOR_TAL"
set "IN=%EXAMS%\Stud_exam\Stud_exam"
if not exist "%IN%\1.pdf" set "IN=%EXAMS%\Stud exam\Stud_exam"
if not exist "%IN%\1.pdf" set "IN=%EXAMS%\Stud exam\Stud exam"
set "OUT=%EXAMS%\grader_runs"
cd /d "%PROJ%"
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing in %PROJ%
  pause
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
set PYTHONWARNINGS=ignore
if exist "models\dictalm-3.0-1.7b-base\model.safetensors" set EXAM_GRADER_HEBREW_CONTEXT=1
if exist "models\uni-mumer-qwen3vl-2b\model.safetensors" set EXAM_GRADER_FORMULA2=mumer
echo Reading the exams in !IN!
echo Results: !OUT!   (log: !OUT!\progress.log)
echo.
set NGPU=1
for /f %%c in ('nvidia-smi -L 2^>nul ^| find /c "GPU "') do set NGPU=%%c
if "!NGPU!"=="0" set NGPU=1
if !NGPU! GTR 1 (
  echo !NGPU! GPUs: one reader on each, the others in windows of their own
  set /a LAST=!NGPU!-1
  for /l %%k in (1,1,!LAST!) do (
    start "exam reader - GPU %%k" cmd /c "set CUDA_VISIBLE_DEVICES=%%k&& .venv\Scripts\python.exe tools\read_exams.py "!IN!" --out "!OUT!" --shard %%k/!NGPU! & pause"
  )
  set CUDA_VISIBLE_DEVICES=0
  ".venv\Scripts\python.exe" tools\read_exams.py "!IN!" --out "!OUT!" --shard 0/!NGPU!
) else (
  ".venv\Scripts\python.exe" tools\read_exams.py "!IN!" --out "!OUT!"
)
echo.
echo Finished - you can close this window.
pause
