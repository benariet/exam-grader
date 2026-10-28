"""Evaluate the mixed Hebrew + math pipeline on data/mixed_eval (100 lines of real handwriting:
Hebrew words from test-split writers + MathWriting valid/test formulas, formula at the start,
middle or end of the line).

Development sets (rules were tuned on them): mixed_eval, mixed_dev_en, mixed_eval_en.
Test sets (built from lines and formulas no development set used): mixed_eval3 (Hebrew + math) and
mixed_eval_en2 (Hebrew + English + math) - report those; mixed_eval2 was the Hebrew + math test set
until the English rules were measured on it.

Measures
  layout   : is every part found with the right kind (Hebrew / math) and nothing merged wrongly?
             Judged by ink: each true part's segment holds >= 90 % of that part's ink pixels and
             <= 10 % of any other part's. (The first version compared boxes with the pasted image
             widths, which include blank margins - correct splits were counted as wrong. Its number
             is kept as layout_correct_old_box_rule for comparison.)
  hebrew   : CER of the Hebrew parts (when the layout is right)
  math     : formula read exactly / equivalent (when the layout is right)
Results: results/mixed_eval.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from grader.symbolic import compare_formula  # noqa: E402
from recog.formula import FormulaRecognizer  # noqa: E402
from recog.hebrew import HebrewRecognizer  # noqa: E402
from recog.hebrew_text import clean  # noqa: E402
from recog.mixed_reader import EnsembleClassifier, MixedReader, default_classifier  # noqa: E402
from recog.segment import ScriptClassifier  # noqa: E402

import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--cls", action="append", default=[], help="classifier folder(s); several = ensemble")
ap.add_argument("--no-cross-check", action="store_true")
ap.add_argument("--no-join", action="store_true", help="do not re-join formulas split into several chunks")
ap.add_argument("--tag", default="")
ap.add_argument("--data", default="mixed_eval", help="folder under data/ (tests: mixed_eval3, mixed_eval_en2)")
ap.add_argument("--no-english", action="store_true", help="Hebrew / math only (no English reader)")
args = ap.parse_args()

D = os.path.join(HERE, "data", args.data)
truth = json.load(open(os.path.join(D, "truth.json"), encoding="utf-8"))
clf = (EnsembleClassifier(args.cls) if len(args.cls) > 1 else ScriptClassifier(args.cls[0])) if args.cls else default_classifier()
reader = MixedReader(clf=clf, join_split_formulas=not args.no_join, english=not args.no_english)
KINDS = ["hebrew", "english", "math"]
confusion = {t: {p: 0 for p in KINDS + ["none"]} for t in KINDS}       # true part -> kind it was read as


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


rows, t0 = [], time.time()
for it in truth:
    page = Image.open(os.path.join(D, it["file"])).convert("L")
    _, pred = reader.read(page, cross_check=not args.no_cross_check)    # segments in reading order
    ink = np.asarray(page) < 128
    spans = [t["x"] for t in it["segments"]]

    def ink_between(a, b):
        a, b = max(0, int(a)), min(ink.shape[1], int(b))
        return int(ink[:, a:b].sum()) if b > a else 0
    totals = [max(1, ink_between(*sp)) for sp in spans]

    def share(i, box):                    # fraction of true part i's ink inside the predicted box
        return ink_between(max(box[0], spans[i][0]), min(box[2], spans[i][1])) / totals[i]
    # layout is right if predicted segments map 1:1 (in order) onto the true parts with the same kinds
    ok = len(pred) == len(it["segments"]) and all(
        p.kind == t["kind"] and share(i, p.box) >= 0.9 and all(share(j, p.box) <= 0.1 for j in range(len(spans)) if j != i)
        for i, (p, t) in enumerate(zip(pred, it["segments"])))
    ok_old = len(pred) == len(it["segments"]) and all(
        p.kind == t["kind"] and overlap((p.box[0], p.box[2]), t["x"]) >= 0.8 * (t["x"][1] - t["x"][0])
        for p, t in zip(pred, it["segments"]))
    for i, t in enumerate(it["segments"]):        # which kind took (most of) each true part's ink
        best = max(pred, key=lambda p: share(i, p.box), default=None)
        confusion[t["kind"]][best.kind if best is not None and share(i, best.box) >= 0.5 else "none"] += 1
    flagged = any(p.p_kind < 0.8 for p in pred)                          # would go to human review
    row = {"file": it["file"], "position": it["position"], "layout_ok": ok, "layout_ok_old": ok_old, "flagged": flagged,
           "pred_boxes": [[int(p.box[0]), int(p.box[2])] for p in pred],
           "pred": [(p.kind, round(p.p_kind, 2), getattr(p, "note", "")) for p in pred],
           "true": [t["kind"] for t in it["segments"]], "parts": []}
    if ok:
        for p, t in zip(pred, it["segments"]):
            if p.kind in ("hebrew", "english"):
                ref, hyp = (clean(t["text"]), clean(p.nbest[0][0])) if p.kind == "hebrew" else (t["text"], p.nbest[0][0])
                row["parts"].append({"kind": p.kind, "ref": ref, "hyp": hyp,
                                     "cer": Levenshtein.distance(ref, hyp) / max(1, len(ref))})
            else:
                class _C: pass
                c = _C(); c.text = p.nbest[0][0]
                exact = c.text.replace(" ", "") == t["text"].replace(" ", "")
                try:
                    equiv = compare_formula(c.text, t["text"]).status == "equivalent"
                except Exception:  # noqa: BLE001
                    equiv = False
                row["parts"].append({"kind": "math", "ref": t["text"], "hyp": c.text, "exact": exact, "equivalent": exact or equiv})
    rows.append(row)

n = len(rows)
lay = [r for r in rows if r["layout_ok"]]
hparts = [p for r in lay for p in r["parts"] if p["kind"] == "hebrew"]
mparts = [p for r in lay for p in r["parts"] if p["kind"] == "math"]
summary = {
    "data": args.data,
    "lines": n,
    "layout_correct": round(len(lay) / n, 3),
    "layout_correct_old_box_rule": round(sum(r["layout_ok_old"] for r in rows) / n, 3),
    "layout_correct_by_position": {pos: round(sum(r["layout_ok"] for r in rows if r["position"] == pos) /
                                         max(1, sum(r["position"] == pos for r in rows)), 3)
                                   for pos in sorted({r["position"] for r in rows})},
    "parts_read_as": confusion,
    "english_CER_(lines_split_right)": round(sum(p["cer"] for r in lay for p in r["parts"] if p["kind"] == "english") /
                                              max(1, sum(p["kind"] == "english" for r in lay for p in r["parts"])), 4),
    "hebrew_CER": round(sum(p["cer"] for p in hparts) / max(1, len(hparts)), 4),
    "math_exact": round(sum(p["exact"] for p in mparts) / max(1, len(mparts)), 3),
    "math_exact_or_equivalent": round(sum(p["equivalent"] for p in mparts) / max(1, len(mparts)), 3),
    "flagged_for_review": round(sum(r["flagged"] for r in rows) / n, 3),
    "layout_wrong_and_NOT_flagged": round(sum((not r["layout_ok"]) and (not r["flagged"]) for r in rows) / n, 3),
    "layout_right_but_flagged": round(sum(r["layout_ok"] and r["flagged"] for r in rows) / n, 3),
    "secs_per_line": round((time.time() - t0) / n, 2),
}
print(json.dumps(summary, ensure_ascii=False, indent=1))
bad = [r for r in rows if not r["layout_ok"]][:8]
for r in bad:
    print(f"  layout miss {r['file']} ({r['position']}): true {r['true']}  pred {r['pred']}")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", f"{args.data}{('_' + args.tag) if args.tag else ''}.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", os.path.relpath(out, HERE))
