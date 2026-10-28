"""English handwriting recognition on held-out lines - fully local.

Data: data/english_lines (IAM test lines, Aachen split: writers TrOCR was not trained on) - built by
prepare_data.py --english, or any parquet with image/text columns via --parquet.
Reports CER / WER and whether the reader's confidence separates good readings from bad ones.
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.english import EnglishRecognizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", default=os.path.join(HERE, "data", "english_lines", "iam_test.parquet"))
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--model", default=None)
ap.add_argument("--tag", default="", help="results/english_eval_<tag>.json (default: english_eval.json)")
args = ap.parse_args()

import pandas as pd  # noqa: E402
df = pd.read_parquet(args.parquet)
rows_idx = random.Random(0).sample(range(len(df)), min(args.n, len(df)))
rec = EnglishRecognizer(args.model) if args.model else EnglishRecognizer()


def norm(t):
    # IAM transcriptions put spaces around punctuation ("it 's", "song ,"); compare without them
    return " ".join(t.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ;", ";").replace(" :", ":")
                    .replace(" !", "!").replace(" ?", "?").split())


rows, t0 = [], time.time()
for k, i in enumerate(rows_idx):
    r = df.iloc[i]
    img = Image.open(io.BytesIO(r["image"]["bytes"]))
    c = rec.recognize(img, n_best=3)
    ref, hyp = norm(r["text"]), norm(c[0].text)
    m = max(x.logp for x in c)
    rows.append({"ref": ref, "hyp": hyp, "cer": Levenshtein.distance(ref, hyp) / max(1, len(ref)),
                 "wer": Levenshtein.distance(ref.split(), hyp.split()) / max(1, len(ref.split())),
                 "conf": math.exp(c[0].logp / max(1, len(c[0].text))), "min_p": c[0].min_token_p})
    if k < 3 or (k + 1) % 50 == 0:
        print(f"[{k + 1}] CER {rows[-1]['cer']:.2f}\n   ref: {ref}\n   hyp: {hyp}", flush=True)
by = sorted(rows, key=lambda r: -r["conf"])
q = max(1, len(rows) // 4)
mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
summary = {"lines": len(rows), "CER": round(mean([r["cer"] for r in rows]), 4), "WER": round(mean([r["wer"] for r in rows]), 4),
           "lines_exact": round(mean([r["ref"] == r["hyp"] for r in rows]), 3),
           "CER_top_quarter_by_confidence": round(mean([r["cer"] for r in by[:q]]), 4),
           "CER_bottom_quarter_by_confidence": round(mean([r["cer"] for r in by[-q:]]), 4),
           "secs_per_line": round((time.time() - t0) / len(rows), 2)}
print(json.dumps(summary, indent=1))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"summary": summary, "rows": rows},
          open(os.path.join(HERE, "results", f"english_eval{'_' + args.tag if args.tag else ''}.json"), "w"), indent=1)
