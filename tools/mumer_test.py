"""Uni-MuMER (phxember/Uni-MuMER-Qwen3-VL-2B: Qwen3-VL-2B fine-tuned on handwritten math, Apache-2.0,
CROHME 2014 83 % exact) against the formula reader in use (Pix2Text MFR) on the 70 real handwritten
formulas (data/handwritten_formulas) and on extra crops (Tal's page).

    python tools/mumer_test.py --n 30 --extra crop1.png=a^{2}+b^{2}=c^{2} ...

Exact = the same LaTeX without spaces; equivalent = grader.symbolic says the same formula.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402

from grader.symbolic import compare_formula  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=70)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
from recog.mumer import MumerReader  # noqa: E402

rd = MumerReader(threads=2)


def read(path):
    return rd.recognize(path)[0].text


norm = lambda s: s.replace(" ", "")  # noqa: E731
D = os.path.join(HERE, "data", "handwritten_formulas")
labels = {os.path.basename(k): v for k, v in json.load(open(os.path.join(D, "labels.json"), encoding="utf-8")).items()}
prev = json.load(open(os.path.join(HERE, "results", "formula_eval.json"), encoding="utf-8")).get("readings", {})
rows, t0 = [], time.time()
for f in sorted(labels)[: a.n]:
    gt, hyp = labels[f], read(os.path.join(D, f))
    p2t = prev.get(f, [{}])[0].get("text", "")
    eq = lambda x: compare_formula(x, gt).status == "equivalent"  # noqa: E731
    rows.append({"file": f, "gt": gt, "mumer": hyp, "pix2text": p2t, "mumer_exact": norm(hyp) == norm(gt),
                 "pix2text_exact": norm(p2t) == norm(gt), "mumer_equiv": eq(hyp), "pix2text_equiv": eq(p2t)})
    print(f"{f}: gt {gt}\n   mumer   : {hyp}\n   pix2text: {p2t}", flush=True)
n = max(1, len(rows))
summary = {k: f"{sum(r[k] for r in rows)}/{len(rows)}" for k in ("mumer_exact", "pix2text_exact", "mumer_equiv", "pix2text_equiv")}
summary["secs_per_formula"] = round((time.time() - t0) / n, 1)
print(json.dumps(summary, indent=1))
for e in a.extra:
    path, truth = e.split("=", 1) if "=" in e and os.path.isfile(e.split("=", 1)[0]) else (e, "")
    print(f"{os.path.basename(path)}: {read(path)!r}   (written: {truth!r})", flush=True)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(HERE, "results", "formula_eval_mumer.json"), "w",
                                                   encoding="utf-8"), ensure_ascii=False, indent=1)
