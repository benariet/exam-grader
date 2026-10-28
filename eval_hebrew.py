"""Hebrew handwriting recognition on writer-independent test lines - fully local.

Data: data/hebrew_lines (300 lines sampled from the test split of cyttic/trocr-hebrew-matan;
its writers never appear in the training data). Default recognizer: the trained CRNN
(models/hebrew-crnn). --vlm evaluates a local vision-language model instead.

Besides CER/WER it checks the point that matters for grading: does the recognizer's own
confidence separate good readings from bad ones?  Results: results/hebrew_eval_<tag>.json
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.hebrew_text import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300, help="number of test lines")
ap.add_argument("--model", default=None, help="model folder (default: models/hebrew-crnn)")
ap.add_argument("--vlm", action="store_true", help="--model is a VLM folder (e.g. models/qwen3-vl-2b)")
ap.add_argument("--adapter", default=None, help="LoRA adapter folder (VLM only)")
ap.add_argument("--beams", type=int, default=3)
ap.add_argument("--tag", default=None)
ap.add_argument("--no-lm", action="store_true", help="CRNN only, without the Hebrew word LM")
ap.add_argument("--tta", action="store_true", help="read every line at three widths and pool (recog/hebrew.py)")
args = ap.parse_args()

DATA = os.path.join(HERE, "data", "hebrew_lines")
items = json.load(open(os.path.join(DATA, "labels.json"), encoding="utf-8"))[: args.n]

if args.vlm:
    from recog.hebrew_vlm import HebrewRecognizer
    rec = HebrewRecognizer(args.model, adapter_dir=args.adapter)
else:
    from recog.hebrew import HebrewRecognizer
    rec = HebrewRecognizer(args.model) if args.model else HebrewRecognizer()
name = os.path.basename(os.path.normpath(args.model or getattr(rec, "model_dir", "hebrew-crnn")))
print(f"recognizer={name}  device={rec.device}  lines={len(items)}", flush=True)


def cer(ref, hyp):
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def wer(ref, hyp):
    return Levenshtein.distance(ref.split(), hyp.split()) / max(1, len(ref.split()))


rows, t0 = [], time.time()
for k, it in enumerate(items):
    t = time.time()
    cands = (rec.recognize(os.path.join(DATA, it["file"]), n_best=args.beams, use_lm=not args.no_lm, tta=args.tta) if not args.vlm
             else rec.recognize(os.path.join(DATA, it["file"]), n_best=args.beams))
    ref, hyp = clean(it["text"]), clean(cands[0].text)
    m = max(c.logp for c in cands)
    p_top = math.exp(cands[0].logp - m) / sum(math.exp(c.logp - m) for c in cands)
    ac = getattr(cands[0], "acoustic_logp", None)
    per_char = math.exp((ac if ac is not None else cands[0].logp) / max(1, len(hyp)))   # image-only, per char
    rows.append({"file": it["file"], "writer": it.get("writer"), "ref": ref, "hyp": hyp,
                 "cer": cer(ref, hyp), "wer": wer(ref, hyp), "p_top": p_top, "conf": per_char,
                 "nbest": [{"text": c.text, "logp": round(c.logp, 3)} for c in cands],
                 "secs": time.time() - t})
    if k < 3 or (k + 1) % 50 == 0:
        print(f"[{k + 1}/{len(items)}] CER={rows[-1]['cer']:.2f} conf={per_char:.2f}\n    ref: {ref}\n    hyp: {hyp}", flush=True)


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


# the test writers never appear in training, but many writers copied the same texts: report lines whose
# text also occurs in the training data separately (the fair number for free-form answers is 'unseen')
seen_texts = set()
for fn in ("matan_train.parquet", "human_train.parquet"):
    fp = os.path.join(HERE, "data", "train", fn)
    if os.path.isfile(fp):
        import pandas as pd
        seen_texts |= {clean(t) for t in pd.read_parquet(fp, columns=["text"])["text"]}
unseen = [r for r in rows if r["ref"] not in seen_texts]

by_conf = sorted(rows, key=lambda r: -r["conf"])
q = max(1, len(rows) // 4)
summary = {
    "recognizer": name, "word_lm": (not args.no_lm and getattr(rec, "lm", None) is not None), "device": rec.device, "n_lines": len(rows), "n_writers": len({r["writer"] for r in rows}),
    "CER": round(mean(r["cer"] for r in rows), 4),
    "CER_median": round(sorted(r["cer"] for r in rows)[len(rows) // 2], 4),
    "WER": round(mean(r["wer"] for r in rows), 4),
    "lines_exact": round(mean(r["ref"] == r["hyp"] for r in rows), 4),
    "secs_per_line": round(mean(r["secs"] for r in rows), 3),
    # confidence vs. error: if confidence is informative, the top quarter is far cleaner than the bottom quarter
    "CER_lines_with_text_not_in_training": round(mean(r["cer"] for r in unseen), 4) if seen_texts else None,
    "n_lines_with_text_not_in_training": len(unseen) if seen_texts else None,
    "CER_top_quarter_by_confidence": round(mean(r["cer"] for r in by_conf[:q]), 4),
    "CER_bottom_quarter_by_confidence": round(mean(r["cer"] for r in by_conf[-q:]), 4),
}
print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
tag = args.tag or (name + ("+lora" if args.adapter else "") + ("-nolm" if args.no_lm else ""))
out = os.path.join(HERE, "results", f"hebrew_eval_{tag}.json")
json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {os.path.relpath(out, HERE)}")
