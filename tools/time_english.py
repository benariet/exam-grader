"""Seconds per line of the English readers, measured side by side on the same IAM test lines.

    python tools/time_english.py --n 20
"""
import argparse
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.english import BASE_DIR, DEFAULT_DIR, EnglishRecognizer  # noqa: E402

df = pd.read_parquet(os.path.join(HERE, "data", "english_lines", "iam_test.parquet"))
rows = [df.iloc[i] for i in range(a.n)]
imgs = [Image.open(io.BytesIO(r["image"]["bytes"])) for r in rows]
for name, d in (("small", DEFAULT_DIR), ("base", BASE_DIR)):
    if not os.path.isfile(os.path.join(d, "encoder_model.onnx")):
        print(name, "not installed")
        continue
    t0 = time.time()
    rec = EnglishRecognizer(d)
    load = time.time() - t0
    rec.recognize(imgs[0], n_best=3)                     # warm-up
    t0 = time.time()
    hyps = [rec.recognize(im, n_best=3)[0].text for im in imgs]
    per = (time.time() - t0) / len(imgs)
    cer = sum(Levenshtein.distance(r["text"], h) for r, h in zip(rows, hyps)) / sum(len(r["text"]) for r in rows)
    print(f"{name}: {per:.2f} s per line (model load {load:.1f} s), CER on these {len(imgs)} lines {cer:.1%}", flush=True)
