"""End-to-end evaluation on REAL handwritten formulas — recognizer + grader, fully local.

Images + ground-truth LaTeX: data/handwritten_formulas (70 human-written samples from
linxy/LaTeX_OCR, human_handwrite test split). Recognizer: Pix2Text MFR on ONNX Runtime.

Two scenarios per image:
  A) the reference answer IS the ground truth   -> student is right (expect full marks)
  B) the reference is a different expression    -> student is wrong (expect 0)
Reports how many grades were auto-accepted, how many of those were correct, and how many
went to a human. Results: results/formula_eval.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from grader import Criterion, Question, grade_question  # noqa: E402
from recog import readings_from_candidates  # noqa: E402
from recog.formula import FormulaRecognizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
ap.add_argument("--threshold", type=float, default=0.85, help="auto-accept confidence threshold")
args = ap.parse_args()

DATA = os.path.join(HERE, "data", "handwritten_formulas")
labels = {os.path.basename(k): v for k, v in json.load(open(os.path.join(DATA, "labels.json"), encoding="utf-8")).items()}
files = sorted(labels)[: args.limit or None]


def norm(s):
    return s.replace(" ", "")


rec = FormulaRecognizer()
t0 = time.time()
readings_cache, rows, exact = {}, [], 0
for i, f in enumerate(files):
    cands = rec.recognize(os.path.join(DATA, f), n_best=3)
    readings_cache[f] = [{"text": c.text, "logp": round(c.logp, 4), "min_token_p": round(c.min_token_p, 4)} for c in cands]
    gt = labels[f]
    exact += norm(cands[0].text) == norm(gt)
    readings = readings_from_candidates(cands)
    other = next(labels[g] for g in sorted(labels)[i + 1:] + sorted(labels)[:i] if norm(labels[g]) != norm(gt))
    for scenario, ref, truth in (("A:correct", gt, 1.0), ("B:wrong", other, 0.0)):
        q = Question(f, "", [Criterion("f", "final_formula", 1, ref)])
        r = grade_question(q, readings, confidence_threshold=args.threshold)
        rows.append({"file": f, "scenario": scenario, "route": r.route, "pred": r.most_likely_score,
                     "truth": truth, "conf": r.confidence, "read": cands[0].text, "gt": gt,
                     "status": r.per_reading[0]["criteria"][0]["status"]})
    if i % 10 == 0:
        print(f"  [{i + 1}/{len(files)}] {time.time() - t0:.0f}s", flush=True)

secs = (time.time() - t0) / max(1, len(files))
print(f"\nrecognition exact match (top-1): {exact}/{len(files)} = {exact / max(1, len(files)):.0%}   ({secs:.2f}s per image)\n")
summary = {"n_images": len(files), "exact_top1": exact, "secs_per_image": round(secs, 3), "scenarios": {}}
for scenario in ("A:correct", "B:wrong"):
    rs = [r for r in rows if r["scenario"] == scenario]
    auto = [r for r in rs if r["route"] == "auto"]
    ok = [r for r in auto if r["pred"] == r["truth"]]
    summary["scenarios"][scenario] = {"n": len(rs), "auto": len(auto), "auto_correct": len(ok),
                                      "auto_wrong": len(auto) - len(ok), "review": len(rs) - len(auto)}
    print(f"{scenario:10} n={len(rs)}  auto={len(auto)} ({len(auto) / max(1, len(rs)):.0%})  "
          f"auto&correct={len(ok)}  auto&WRONG={len(auto) - len(ok)}  review={len(rs) - len(auto)}")
    print("   top-reading statuses:", dict(Counter(r["status"] for r in rs)))

bad = [r for r in rows if r["route"] == "auto" and r["pred"] != r["truth"]]
print(f"\nauto-accepted but graded WRONG: {len(bad)}")
for r in bad:
    print(f"  {r['file']} {r['scenario']} conf={r['conf']:.2f}\n     read: {r['read']}\n     true: {r['gt']}")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"summary": summary, "rows": rows, "readings": readings_cache},
          open(os.path.join(HERE, "results", "formula_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwrote results/formula_eval.json")
