"""Does reading an English line at several widths and pooling the n-best lists help TrOCR?
Measured on IAM test lines (the same normalisation as eval_english.py).

    python tools/english_tta.py --n 60
"""
import argparse
import io
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.english import EnglishRecognizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--scales", default="1.0,0.85,1.15")
ap.add_argument("--model", default=None)
a = ap.parse_args()
df = pd.read_parquet(os.path.join(HERE, "data", "english_lines", "iam_test.parquet"))
idx = random.Random(0).sample(range(len(df)), a.n)
rec = EnglishRecognizer(a.model) if a.model else EnglishRecognizer()
scales = [float(s) for s in a.scales.split(",")]


def norm(t):
    return " ".join(t.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ;", ";").replace(" :", ":")
                    .replace(" !", "!").replace(" ?", "?").split())


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


e1 = ep = chars = 0
for i in idx:
    r = df.iloc[i]
    img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    ref = norm(r["text"])
    pools, first = {}, None
    for s in scales:
        v = img if s == 1.0 else img.resize((max(16, int(img.width * s)), img.height), Image.BICUBIC)
        cands = rec.recognize(v, n_best=3)
        if first is None:
            first = norm(cands[0].text)
        for c in cands:
            pools.setdefault(norm(c.text), []).append(c.logp)
    pooled = max(pools, key=lambda t: lse(pools[t]) + math.log(len(pools[t])))
    e1 += Levenshtein.distance(ref, first)
    ep += Levenshtein.distance(ref, pooled)
    chars += len(ref)
print(f"one look CER {e1 / chars:.4f}   pooled over {scales}: {ep / chars:.4f}   ({a.n} lines)")
