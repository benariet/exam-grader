#!/bin/bash
# Read one exam on a GPU server (Slurm), its pages shared among SHARES processes at once (the first reading runs on the
# CPU, page by page: 30 pages took about 1.5 hours in one process), then Qwen3-VL-8B over its math pieces, merged with
# Uni-MuMER's readings and checked against the other exams already read (a VLM's made-up line repeats across exams).
#
#   sbatch --export=ALL,EXAM=6,IN=$HOME/eg-runs/in,OUT=$HOME/eg-runs/out tools/server_read_exam_parallel.sbatch
#
# IN holds <EXAM>.pdf; results: OUT/<EXAM>.json, OUT/vlm/<EXAM>.json, OUT/merged/<EXAM>.json (judge this one)
# SKIP: pages never read (default 27,29 - this course's formula sheets, the same printed page in every copy; SKIP= reads all)
# More CPUs read faster (the first reading is on the CPU): sbatch -c 48 --mem=160G -p shared_a6000 --export=ALL,SHARES=6,...
# (each share loads the heavy readers on the one GPU in turn: about 6 GB each - 6 shares on a 48 GB card)
#SBATCH -J eg-read
#SBATCH -p shared_a6000,shared_a5000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o eg-read.%j.log
set -u
cd "${PROJ:-$HOME/exam-grader}"
SHARES=${SHARES:-4}
SKIP=${SKIP-27,29}
BOOKLET=${BOOKLET-booklet/moedC}       # the printed booklet's templates (tools/build_booklet.py) - BOOKLET= reads the print too
T=$(( ${SLURM_CPUS_PER_TASK:-8} / SHARES )); [ "$T" -lt 1 ] && T=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T EXAM_GRADER_THREADS=$T
[ -f models/dictalm-3.0-1.7b-base/model.safetensors ] && export EXAM_GRADER_HEBREW_CONTEXT=1
[ -f models/uni-mumer-qwen3vl-2b/model.safetensors ] && export EXAM_GRADER_FORMULA2=mumer
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT/vlm" "$OUT/merged" "$OUT/parts"
N=$(.venv/bin/python -c "import sys, pypdfium2 as p; print(len(p.PdfDocument(sys.argv[1])))" "$IN/$EXAM.pdf") || { echo "FAILED page count"; exit 1; }
STEP=$(( (N + SHARES - 1) / SHARES ))
echo "$EXAM: $N pages, $SHARES shares of $STEP, $T threads each"
pids=()
for ((a = 1; a <= N; a += STEP)); do
  b=$(( a + STEP - 1 )); [ "$b" -gt "$N" ] && b=$N
  .venv/bin/python tools/read_exams.py "$IN" --out "$OUT/parts" --only "$EXAM" --pages "$a-$b" ${SKIP:+--skip-pages "$SKIP"} \
    $([ -n "$BOOKLET" ] && [ -f "$BOOKLET/booklet.json" ] && echo --booklet "$BOOKLET") \
    > "$OUT/parts/$EXAM.p$a-$b.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
grep -h "DONE\|FAILED" "$OUT/parts/$EXAM".p*-*.log
[ "$fail" = 0 ] || { echo "FAILED reading a share"; exit 1; }
.venv/bin/python tools/join_pages.py "$OUT/parts" "$EXAM" && cp "$OUT/parts/$EXAM.json" "$OUT/$EXAM.json" || { echo "FAILED join"; exit 1; }
.venv/bin/python tools/reread_math_vlm.py "$IN/$EXAM.pdf" "$OUT/$EXAM.json" "$OUT/vlm/$EXAM.json" || { echo "FAILED vlm"; exit 1; }
.venv/bin/python tools/reread_math_vlm.py --merge "$OUT/vlm/$EXAM.json" "$OUT/merged/$EXAM.json" "$OUT"/vlm/*.json || { echo "FAILED merge"; exit 1; }
echo "ALL DONE $EXAM"
