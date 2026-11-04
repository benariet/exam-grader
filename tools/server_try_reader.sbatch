#!/bin/bash
# Try another vision-language model as the reader (4.11: is a stronger model than Qwen3-VL-8B the way to read more?)
# on a GPU server (Slurm), in two phases - the math pieces of every exam first (the merge holds each exam's readings
# against the others'), then the merge and the lines:
#
#   sbatch --export=ALL,PHASE=pieces,MODEL=models/qwen3.6-27b,EXAMS="1 3 5 7",IN=$HOME/eg-runs/in,BASE=$HOME/eg-runs/base,OUT=$HOME/eg-runs/q27 tools/server_try_reader.sbatch
#   sbatch --dependency=afterok:<both pieces jobs> --export=ALL,PHASE=lines,... tools/server_try_reader.sbatch
#   (PHASE=all: both in one job - one model load)
#
# IN holds <n>.pdf, BASE each exam's reading as tools/read_exams.py left it - or as tools/reread_math_vlm.py did
# (another VLM's readings: Uni-MuMER's are put back first). Results: OUT/vlm/<n>.json (the pieces read again),
# OUT/read/<n>.json (merged), OUT/lines/<n>.json (the lines read whole) - judge with tools/recheck_exams.py.
# Two GPUs: a model too big for one is spread over both (in 8 bits where both are too small - recog/hebrew_vlm.py) -
# or, with SERVE=<model dir>, served by vLLM in the job (fast: FP8, many requests at once; VLLM: how to run its python -
# by default the cluster's Apptainer image of it, /truenas/sif_images/qwen36_vllm_latest.sif, vLLM 0.20).
#SBATCH -J eg-try
#SBATCH -p shared_a6000,shared_a5000
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -o eg-try.%j.log
set -u
cd "${PROJ:-$HOME/exam-grader}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
mkdir -p "$OUT/vlm" "$OUT/read" "$OUT/lines"
if [ -n "${SERVE:-}" ]; then
  # SERVE: a model for vLLM - served on this machine over the job's GPUs, the tools pointed at
  # it (recog/remote_vlm.py): FP8 weights run on Ampere cards through vLLM's Marlin kernels
  export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost      # (the server's web proxy: not for this)
  PORT=$((8000 + SLURM_JOB_ID % 1000))
  NG=$(nvidia-smi -L | wc -l)
  VLLM=${VLLM:-apptainer exec --nv /truenas/sif_images/qwen36_vllm_latest.sif python3}   # (the cluster's vLLM image)
  ($VLLM -m vllm.entrypoints.openai.api_server --model "$SERVE" \
      --served-model-name reader --host 127.0.0.1 --port $PORT --tensor-parallel-size $NG --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 1}' --max-num-seqs 32 > "$OUT/vllm.$SLURM_JOB_ID.log" 2>&1) &
  for i in $(seq 1 120); do curl -s "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 10; done
  curl -s "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM did not start"; tail -n 30 "$OUT/vllm.$SLURM_JOB_ID.log"; exit 1; }
  echo "vLLM serving $SERVE on $NG GPUs after $((i * 10)) s"
  MODEL=http://127.0.0.1:$PORT/v1
fi
if [ "$PHASE" = pieces ] || [ "$PHASE" = all ]; then
  for n in $EXAMS; do
    .venv/bin/python tools/reread_math_vlm.py "$IN/$n.pdf" "$BASE/$n.json" "$OUT/vlm/$n.json" --model "$MODEL" || echo "FAILED $n"
  done
fi
if [ "$PHASE" = lines ] || [ "$PHASE" = all ]; then            # (all: the pieces of every exam first, then the lines)
  for n in $EXAMS; do
    .venv/bin/python tools/reread_math_vlm.py --merge "$OUT/vlm/$n.json" "$OUT/read/$n.json" "$OUT"/vlm/*.json &&
    .venv/bin/python tools/reread_lines.py "$IN/$n.pdf" "$OUT/read/$n.json" "$OUT/lines/$n.json" --model "$MODEL" || echo "FAILED $n"
  done
fi
echo "ALL DONE"
