#!/bin/bash
# Read exams' answers as blocks (tools/read_blocks.py: each sub-question's whole answer read at once by Qwen3-VL-8B)
# on a GPU server (Slurm) - one model load for all the exams named:
#
#   sbatch --export=ALL,EXAMS="5 9",IN=$HOME/eg-runs/in,OUT=$HOME/eg-runs/blocks tools/server_read_blocks.sbatch
#
# IN holds <n>.pdf; results: OUT/<n>_blocks.json and OUT/<n>.json (the same in read_exams.py's form, for
# tools/recheck_exams.py). PAGES (optional): only these pages, e.g. PAGES=9,11
#SBATCH -J eg-blocks
#SBATCH -p shared_a6000,shared_a5000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o eg-blocks.%j.log
set -u
cd "${PROJ:-$HOME/exam-grader}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"
for n in $EXAMS; do
  .venv/bin/python tools/read_blocks.py "$IN/$n.pdf" "$OUT/${n}_blocks.json" --as-reading "$OUT/$n.json" \
    ${PAGES:+--pages "$PAGES"} || echo "FAILED $n"
done
echo "ALL DONE"
