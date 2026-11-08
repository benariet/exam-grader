#!/bin/bash
# Read each line of the exams' answers whole, with the lines before it as context (tools/reread_lines.py), on a GPU
# server (Slurm):
#
#   sbatch --export=ALL,EXAMS="1 3 5",IN=$HOME/eg-runs/in,READ=$HOME/eg-runs/read,OUT=$HOME/eg-runs/lines tools/server_reread_lines.sbatch
#
# IN holds <n>.pdf, READ the exams' merged readings <n>.json; results: OUT/<n>.json (judge with tools/recheck_exams.py).
# VIEWS: each math line read at this many scales of its image, the reading most like the others kept (default 3 -
# 6.11, the 8 graded exams: 51 steps checked right against 44, 25 values agreeing with the solution against 22; three
# times the reading time, 4-10 minutes an exam on one GPU; VIEWS=1: one look)
#SBATCH -J eg-lines
#SBATCH -p shared_a6000,shared_a5000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o eg-lines.%j.log
set -u
cd "${PROJ:-$HOME/exam-grader}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"
for n in $EXAMS; do
  .venv/bin/python tools/reread_lines.py "$IN/$n.pdf" "$READ/$n.json" "$OUT/$n.json" --views "${VIEWS:-3}" || echo "FAILED $n"
done
echo "ALL DONE"
