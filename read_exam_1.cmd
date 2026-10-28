@echo off
rem Read ONE student's exam - exam 1 (1.pdf, not the _checked copy) - with the exam-grader readers, fully local, on
rem this computer's GPU. Progress is printed as it goes (each page, then each heavy reader with the time left) and kept
rem in the results folder's progress.log. Results (outside this project folder - students' work stays with the exams):
rem   %EXAMS%\grader_runs\1.json, and a picture + text per page in %EXAMS%\grader_runs\read\
chcp 65001 >nul
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
if not exist "%IN%\1.pdf" (
  echo The exams were not found under %EXAMS%
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
echo Reading exam 1 from %IN%
echo Results: %OUT%   (log: %OUT%\progress.log)
echo.
".venv\Scripts\python.exe" tools\read_exams.py "%IN%" --out "%OUT%" --only 1
echo.
echo Finished - you can close this window.
pause
