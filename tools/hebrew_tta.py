"""Does reading every Hebrew line at several widths (test-time augmentation) and pooling the n-best
lists help? Measured on the 300 test lines (data/hebrew_lines), against one look.

    python tools/hebrew_tta.py --model models/hebrew-crnn-cpu
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.hebrew import HebrewRecognizer  # noqa: E402
from recog.hebrew_text import clean  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--scales", default="1.0,0.85,1.15")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--dev", action="store_true", help="the training data's held-out writers instead of the test lines "
                "(the same 5 %% of matan writers train_hebrew.py keeps out) - for choosing settings")
a = ap.parse_args()
rec = HebrewRecognizer(a.model) if a.model else HebrewRecognizer()
scales = [float(s) for s in a.scales.split(",")]
data = os.path.join(HERE, "data", "hebrew_lines")
if a.dev:
    import io
    import random
    import pandas as pd
    df = pd.read_parquet(os.path.join(HERE, "data", "train", "matan_train.parquet"))
    ws = sorted(df["writer"].unique())
    val_writers = set(random.Random(0).sample(ws, max(1, int(len(ws) * 0.05))))
    dev = df[df["writer"].isin(val_writers)].sample(frac=1, random_state=1).head(a.n)
    items = [{"img": Image.open(io.BytesIO(r["image"]["bytes"])), "text": r["text"]} for _, r in dev.iterrows()]
else:
    items = json.load(open(os.path.join(data, "labels.json"), encoding="utf-8"))[: a.n]


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


tot = {"one": 0, "pool": 0, "vote": 0}
chars = 0
for it in items:
    img = it["img"] if "img" in it else Image.open(os.path.join(data, it["file"]))
    ref = clean(it["text"])
    pools = {}
    tops = []
    for s in scales:
        v = img if s == 1.0 else img.resize((max(16, int(img.width * s)), img.height), Image.BILINEAR)
        cands = rec.recognize(v, n_best=5)
        if not cands:
            continue
        tops.append(clean(cands[0].text))
        for c in cands:
            pools.setdefault(clean(c.text), []).append(c.logp)
    one = tops[0] if tops else ""
    # pooled: a reading's evidence summed over the views (missing in a view: nothing added)
    pooled = max(pools, key=lambda t: lse(pools[t]) + math.log(len(pools[t]))) if pools else ""
    vote = max(set(tops), key=lambda t: (tops.count(t), t == one)) if tops else ""
    for k, h in (("one", one), ("pool", pooled), ("vote", vote)):
        tot[k] += Levenshtein.distance(ref, h)
    chars += len(ref)
print({k: round(v / chars, 4) for k, v in tot.items()}, "scales", scales)
