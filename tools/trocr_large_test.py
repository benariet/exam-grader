"""TrOCR-large-handwritten (microsoft, 558M parameters, PyTorch) on the same IAM test lines as
eval_english.py and on Tal's three English lines - is the biggest TrOCR worth it over base / small?

    python tools/trocr_large_test.py --n 60
"""
import argparse
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--beams", type=int, default=4)
ap.add_argument("--extra", nargs="*", default=[], help="more line images: path=truth")
a = ap.parse_args()
torch.set_num_threads(2)
proc = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").eval()


def norm(t):
    return " ".join(t.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ;", ";").replace(" :", ":")
                    .replace(" !", "!").replace(" ?", "?").split())


def read(img):
    px = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(px, num_beams=a.beams, max_new_tokens=64)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


df = pd.read_parquet(os.path.join(HERE, "data", "english_lines", "iam_test.parquet"))
idx = random.Random(0).sample(range(len(df)), min(150, len(df)))[: a.n]      # the first n of eval_english.py's lines
rows, t0 = [], time.time()
for i in idx:
    r = df.iloc[i]
    ref, hyp = norm(r["text"]), norm(read(Image.open(io.BytesIO(r["image"]["bytes"]))))
    rows.append({"ref": ref, "hyp": hyp, "cer": Levenshtein.distance(ref, hyp) / max(1, len(ref))})
mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
out = {"lines": len(rows), "CER_large": round(mean([r["cer"] for r in rows]), 4), "secs_per_line": round((time.time() - t0) / max(1, len(rows)), 2)}
for tag, f in (("small", "english_eval.json"), ("base", "english_eval_base_encfp32.json")):
    p = os.path.join(HERE, "results", f)
    if os.path.isfile(p):
        old = json.load(open(p))["rows"][: len(rows)]
        if [x["ref"] for x in old] == [x["ref"] for x in rows]:
            out[f"CER_{tag}_same_lines"] = round(mean([x["cer"] for x in old]), 4)
print(json.dumps(out, indent=1))
for e in a.extra:
    path, truth = e.split("=", 1)
    hyp = read(Image.open(path))
    print(f"{os.path.basename(path)}: {hyp!r}   (truth {truth!r}, CER {Levenshtein.distance(truth, hyp) / len(truth):.0%})")
json.dump({"summary": out, "rows": rows}, open(os.path.join(HERE, "results", "english_eval_trocr_large.json"), "w"), indent=1)
