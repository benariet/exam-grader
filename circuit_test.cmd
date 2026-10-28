@echo off
rem Can the local vision model read a hand-drawn circuit? Qwen3-VL-8B (downloaded by gpu_eval.cmd) reads
rem data\my_scans\tal_circuit.pdf into a netlist: 4-bit on the GPU, then full precision on the CPU.
rem Everything printed also goes to results\circuit_test.log; the numbers to results\circuit_test.json.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing - run gpu_eval.cmd first
  pause
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" tools\circuit_test.py %*
echo.
echo Log: %~dp0results\circuit_test.log
pause
