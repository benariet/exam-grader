"""Does reading Hebrew lines in context (recog/hebrew_context.py: the handwriting model + a Hebrew
language model) beat the reader in use? Settings are chosen on held-out TRAINING writers (--dev),
the test lines are measured once with the chosen setting.

    python tools/hebrew_context_eval.py --dev --n 120 --alphas 0.25,0.5,0.75,1.0
    python tools/hebrew_context_eval.py --n 300 --alphas 0.5            # the 300 test lines, once

The baseline is the reader in use (word model, five widths). Writes results/hebrew_context_<tag>.json.
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.hebrew import HebrewRecognizer  # noqa: E402
from recog.hebrew_context import HebrewContext, HebrewLM  # noqa: E402
from recog.hebrew_text import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dev", action="store_true", help="held-out training writers (as tools/hebrew_tta.py --dev)")
ap.add_argument("--n", type=int, default=120)
ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
ap.add_argument("--prune", type=float, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
alphas = [float(x) for x in a.alphas.split(",")]

if a.dev:
    import pandas as pd
    df = pd.read_parquet(os.path.join(HERE, "data", "train", "matan_train.parquet"))
    ws = sorted(df["writer"].unique())
    val_writers = set(random.Random(0).sample(ws, max(1, int(len(ws) * 0.05))))
    dev = df[df["writer"].isin(val_writers)].sample(frac=1, random_state=1).head(a.n)
    items = [{"img": Image.open(io.BytesIO(r["image"]["bytes"])), "text": r["text"]} for _, r in dev.iterrows()]
else:
    d = os.path.join(HERE, "data", "hebrew_lines")
    items = [{"img": Image.open(os.path.join(d, it["file"])), "text": it["text"]}
             for it in json.load(open(os.path.join(d, "labels.json"), encoding="utf-8"))[: a.n]]

rec = HebrewRecognizer()
ctx = HebrewContext(rec=rec, lm=HebrewLM(threads=a.threads))
if a.prune is not None:
    ctx.prune = a.prune
err = {k: 0 for k in ["base"] + alphas}
werr = {k: 0 for k in ["base"] + alphas}
chars = words = 0
rows = []
t0 = time.time()
for i, it in enumerate(items):
    img, truth = it["img"].convert("L"), clean(it["text"])
    prepared = ctx.start(img)
    base = prepared[1][0]
    acs = {}
    row = {"truth": truth, "base": base}
    tw = truth.split()
    for key, text in [("base", base)] + [(al, ctx.read(img, alpha=al, prepared=prepared, acs=acs)[0].text) for al in alphas]:
        e = Levenshtein.distance(clean(text), truth)
        err[key] += e
        werr[key] += Levenshtein.distance(clean(text).split(), tw)
        row[str(key)] = text
    chars += len(truth)
    words += len(tw)
    row["truth_lm_per_char"] = ctx.lm.logp([truth])[0] / max(1, len(truth))
    rows.append(row)
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(items)} lines, {(time.time() - t0) / (i + 1):.1f} s/line: " +
              ", ".join(f"{k}: CER {err[k] / chars:.2%}" for k in err), flush=True)

summary = {"set": "dev (held-out training writers)" if a.dev else "test (300 lines, unseen writers)", "lines": len(items),
           "prune": ctx.prune, "max_edits": ctx.max_edits,
           "CER": {str(k): round(err[k] / chars, 4) for k in err}, "WER": {str(k): round(werr[k] / words, 4) for k in werr},
           "truth_lm_logp_per_char": round(sum(r["truth_lm_per_char"] for r in rows) / len(rows), 3),
           "secs_per_line": round((time.time() - t0) / len(items), 1)}
print(json.dumps(summary, ensure_ascii=False, indent=1))
tag = a.tag or ("dev" if a.dev else "test")
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"summary": summary, "rows": rows}, open(os.path.join(HERE, "results", f"hebrew_context_{tag}.json"), "w",
                                                   encoding="utf-8"), ensure_ascii=False, indent=1)
