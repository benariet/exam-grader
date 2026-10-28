"""heb-ocr as a second opinion for the Hebrew lines our reader is least sure of: the confidence threshold
is chosen on held-out TRAINING writers (dev), then applied once to the test lines.

    python tools/hebrew_second_opinion.py      # needs ext/heb-ocr + mishkefet-v1 (see tools_eval_hebocr.py)
"""
import io
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.hebrew import HebrewRecognizer  # noqa: E402
from recog.hebrew_text import clean  # noqa: E402

CODE, WEIGHTS = "/home/claude/ext/heb-ocr", "/home/claude/ext/mishkefet-v1"
OUT = os.path.join(HERE, "results", "hebrew_second_opinion.json")


def dev_lines(n=300):
    df = pd.read_parquet(os.path.join(HERE, "data", "train", "matan_train.parquet"))
    ws = sorted(df["writer"].unique())
    val_writers = set(random.Random(0).sample(ws, max(1, int(len(ws) * 0.05))))
    dev = df[df["writer"].isin(val_writers)].sample(frac=1, random_state=1).head(n)
    from PIL import Image
    return [(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"), clean(r["text"])) for _, r in dev.iterrows()]


def hebocr_reader():
    import pickle
    from collections import defaultdict
    import torch
    sys.path.insert(0, CODE)
    import hebocr.recognize as R
    from hebocr.lm import CharNGramLM

    class Safe(pickle.Unpickler):
        OK = {("builtins", n) for n in ("dict", "list", "tuple", "set", "frozenset", "int", "float", "str", "bool")}
        OK |= {("collections", "defaultdict"), ("collections", "Counter"), ("collections", "OrderedDict")}

        def find_class(self, m, n):
            if (m, n) in self.OK:
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"refused {m}.{n}")
    real = torch.load
    R.torch.load = lambda f, map_location=None, weights_only=False: real(f, map_location=map_location, weights_only=True)
    d = Safe(open(os.path.join(WEIGHTS, "hebrew_char6.pkl"), "rb")).load()
    lm = CharNGramLM(order=d["order"], backoff=d["backoff"])
    lm.total, lm.counts = d["total"], [defaultdict(int, c) for c in d["counts"]]
    torch.set_num_threads(1)
    return R.Recognizer(os.path.join(WEIGHTS, "mishkefet-v1.pt"), device="cpu", lm=lm, lm_weight=0.4, beam_width=12, tta=3)


def main():
    if os.path.isfile(OUT):
        dev = json.load(open(OUT, encoding="utf-8"))["dev"]
    else:
        items = dev_lines()
        ours = HebrewRecognizer()
        dev = []
        for img, ref in items:
            c = ours.recognize(img, n_best=3, tta=True)[0]
            hyp = clean(c.text)
            dev.append({"ref": ref, "ours": hyp, "conf": math.exp((c.acoustic_logp if c.acoustic_logp is not None else c.logp) / max(1, len(hyp)))})
        rec = hebocr_reader()
        hyps = rec.read_tta([img for img, _ in items], batch_size=8)
        for row, h in zip(dev, hyps):
            row["hebocr"] = clean(h)
        json.dump({"dev": dev}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def cer(rows, th, a="ours", b="hebocr"):
        e = c = 0
        for r in rows:
            hyp = r[b] if r["conf"] < th else r[a]
            e += Levenshtein.distance(r["ref"], hyp)
            c += len(r["ref"])
        return e / c
    confs = sorted(r["conf"] for r in dev)
    cands = [-1.0] + [confs[int(len(confs) * q)] for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)]
    scores = [(cer(dev, th), th) for th in cands]
    best = min(scores)
    print("dev:", " ".join(f"th {th:.3f}: {c:.4f}" for c, th in scores))
    # the test lines: our 5-width readings and heb-ocr's, from the saved evaluations
    ours = json.load(open(os.path.join(HERE, "results", "hebrew_eval_crnn-cpu-tta5.json"), encoding="utf-8"))["rows"]
    heb = {r["file"]: r for r in json.load(open(os.path.join(HERE, "results", "hebrew_eval_hebocr.json"), encoding="utf-8"))["rows"]}
    test = [{"ref": o["ref"], "ours": o["hyp"], "hebocr": heb[o["file"]]["hyp"], "conf": o["conf"]} for o in ours if o["file"] in heb]
    print(f"test, threshold chosen on dev ({best[1]:.3f}): CER {cer(test, best[1]):.4f}  (ours alone {cer(test, -1):.4f})")


if __name__ == "__main__":
    main()
