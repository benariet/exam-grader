#!/bin/bash
# Read one exam on a GPU server (Slurm): the ordinary readers (tools/read_exams.py), then Qwen3-VL-8B over its math
# pieces (tools/reread_math_vlm.py) merged with Uni-MuMER's readings. Judge the result with tools/recheck_exams.py
# where the solution and the grades sheet are.
#
#   sbatch --export=EXAM=6,IN=$HOME/eg-runs/in,OUT=$HOME/eg-runs/out tools/server_read_exam.sbatch
#
# IN holds <EXAM>.pdf; results: OUT/<EXAM>.json (ordinary), OUT/vlm/<EXAM>.json, OUT/merged/<EXAM>.json (judge this one)
#SBATCH -J eg-read
#SBATCH -p shared_a6000,shared_a5000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o eg-read.%j.log
set -u
cd "${PROJ:-$HOME/exam-grader}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} EXAM_GRADER_THREADS=${SLURM_CPUS_PER_TASK:-8}
[ -f models/dictalm-3.0-1.7b-base/model.safetensors ] && export EXAM_GRADER_HEBREW_CONTEXT=1
[ -f models/uni-mumer-qwen3vl-2b/model.safetensors ] && export EXAM_GRADER_FORMULA2=mumer
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT/vlm" "$OUT/merged"
.venv/bin/python tools/read_exams.py "$IN" --out "$OUT" --only "$EXAM" || { echo "FAILED reading"; exit 1; }
.venv/bin/python tools/reread_math_vlm.py "$IN/$EXAM.pdf" "$OUT/$EXAM.json" "$OUT/vlm/$EXAM.json" || { echo "FAILED vlm"; exit 1; }
.venv/bin/python tools/reread_math_vlm.py --merge "$OUT/vlm/$EXAM.json" "$OUT/merged/$EXAM.json" || { echo "FAILED merge"; exit 1; }
echo "ALL DONE $EXAM"
