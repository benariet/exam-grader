"""Tune the word-model weights (lm_weight, word_bonus) on writers held out from training - NOT on the
test lines (data/hebrew_lines) and not on the demo. Writes the chosen values into
models/hebrew-lm/meta.json, where HebrewRecognizer picks them up.

    python tune_hebrew_lm.py [--lines 300]

Choice rule: lowest dev CER; among settings within --tie of it, the smallest lm_weight (a strong word
model is riskier for words missing from the list: technical terms, names, misspellings).
The word list must not contain the handwriting datasets' transcriptions: many writers copied the same
texts, so dev/test words would all be "known" and the weights would come out too strong.
"""
import argparse
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from recog.hebrew import HebrewRecognizer, ctc_prefix_beam_search  # noqa: E402
from recog.hebrew_text import ID2CHAR, clean, to_logical  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-dir", default=os.path.join(HERE, "data", "train"))
ap.add_argument("--lines", type=int, default=300)
ap.add_argument("--tie", type=float, default=0.001, help="CER differences below this count as a tie")
args = ap.parse_args()

# the same held-out writers as train_hebrew.py (5 % of the matan writers, seed 0)
df = pd.read_parquet(os.path.join(args.train_dir, "matan_train.parquet"))
ws = sorted(df["writer"].unique())
val_writers = set(random.Random(0).sample(ws, max(1, int(len(ws) * 0.05))))
dev = df[df["writer"].isin(val_writers)].sample(frac=1, random_state=1).head(args.lines)
print(f"dev: {len(dev)} lines from {dev['writer'].nunique()} held-out writers", flush=True)

rec = HebrewRecognizer()
lm = rec.lm
refs = [clean(t) for t in dev["text"]]
keys = [lm.key(w) for r in refs for w in r.split()]
keys = [k for k in keys if k]
known = sum(k in lm.table for k in keys) / max(1, len(keys))
print(f"word list: {len(lm.table)} words ({len(lm.course_words)} course words); "
      f"{known:.1%} of the dev words are in it", flush=True)
logps = [rec.log_probs(Image.open(io.BytesIO(r["image"]["bytes"]))) for _, r in dev.iterrows()]


def cer_for(lw, wb):
    errs = chars = 0
    for lp, ref in zip(logps, refs):
        ids = ctc_prefix_beam_search(lp, beam=12, n_best=1, lm=lm, lm_weight=lw, word_bonus=wb)[0][0]
        hyp = clean(" ".join(to_logical("".join(ID2CHAR.get(i, "") for i in ids)).split()))
        errs += Levenshtein.distance(ref, hyp)
        chars += len(ref)
    return errs / max(1, chars)


base = cer_for(0.0, 0.0)
print(f"no word model: CER {base:.4f}", flush=True)
results = {}
for lw in (0.15, 0.2, 0.3, 0.45, 0.6):
    for wb in (0.0, 1.5, 3.0):
        results[(lw, wb)] = cer_for(lw, wb)
        print(f"lm_weight {lw:<4} word_bonus {wb:<4} CER {results[(lw, wb)]:.4f}", flush=True)
best = min(results.values())
ties = [k for k, v in results.items() if v <= best + args.tie]
lw, wb = min(ties, key=lambda k: (k[0], results[k]))
chosen = results[(lw, wb)]
print(f"best CER {best:.4f}; chosen: lm_weight={lw} word_bonus={wb} CER {chosen:.4f} (no word model {base:.4f})")
meta_p = os.path.join(HERE, "models", "hebrew-lm", "meta.json")
meta = json.load(open(meta_p, encoding="utf-8"))
meta.update({"lm_weight": lw, "word_bonus": wb,
             "tuned_on": f"{len(dev)} lines of {dev['writer'].nunique()} held-out training writers "
                         f"({known:.0%} of their words in the list), with {len(lm.course_words)} course words",
             "dev_cer_no_lm": round(base, 4), "dev_cer_with_lm": round(chosen, 4),
             "choice": f"lowest dev CER {best:.4f}; within {args.tie} of it the smallest lm_weight"})
meta.pop("test_300_lines_86_unseen_writers", None)         # stale after re-tuning; eval_hebrew.py measures it
json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", meta_p)
