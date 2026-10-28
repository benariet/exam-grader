"""Compare heb-ocr (Mishkefet-v1, github.com/itayinbarr/heb-ocr) with our Hebrew CRNN on our Hebrew test
lines (data/hebrew_lines: 300 lines of cyttic/trocr-hebrew-matan's TEST writers).

    python tools_eval_hebocr.py --code /home/claude/ext/heb-ocr --weights /home/claude/ext/mishkefet-v1

Loading is done safely: the checkpoint with torch.load(weights_only=True) and the language-model
pickles with an unpickler that admits only plain Python containers (the project's own loaders use
pickle / weights_only=False, which would run any code a file carries).

Note on fairness: our CRNN was trained on the same collection's TRAIN writers (in-domain); heb-ocr
was trained on other Hebrew handwriting (Pinkas cursive, BiblIA, synthetic) - this set is out of
domain for it. The ivrit.ai benchmark would be neutral for both, but it is gated (needs an account).
"""
import argparse
import io
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("builtins", n) for n in ("dict", "list", "tuple", "set", "frozenset", "int", "float", "str", "bool", "bytes")}
    ALLOWED |= {("collections", "defaultdict"), ("collections", "Counter"), ("collections", "OrderedDict")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def safe_pickle(path):
    with open(path, "rb") as fh:
        return SafeUnpickler(fh).load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="/home/claude/ext/heb-ocr")
    ap.add_argument("--weights", default="/home/claude/ext/mishkefet-v1")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tta", type=int, default=3)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import torch
    from collections import defaultdict
    from PIL import Image
    from rapidfuzz.distance import Levenshtein
    sys.path.insert(0, args.code)
    import hebocr.recognize as R
    from hebocr.lm import CharNGramLM

    real_load = torch.load
    R.torch.load = lambda f, map_location=None, weights_only=False: real_load(f, map_location=map_location, weights_only=True)
    lm = None
    if not args.no_lm:
        d = safe_pickle(os.path.join(args.weights, "hebrew_char6.pkl"))
        lm = CharNGramLM(order=d["order"], backoff=d["backoff"])
        lm.total = d["total"]
        lm.counts = [defaultdict(int, c) for c in d["counts"]]
    torch.set_num_threads(args.threads or max(1, os.cpu_count() or 1))
    rec = R.Recognizer(os.path.join(args.weights, "mishkefet-v1.pt"), device="cpu", lm=lm, lm_weight=0.4,
                       beam_width=args.beam if lm is not None else 0, tta=args.tta)
    sys.path.insert(0, HERE)
    from recog.hebrew_text import clean

    data = os.path.join(HERE, "data", "hebrew_lines")
    items = json.load(open(os.path.join(data, "labels.json"), encoding="utf-8"))[: args.n]
    imgs = [Image.open(os.path.join(data, it["file"])).convert("RGB") for it in items]
    t0 = time.time()
    hyps = rec.read_tta(imgs, batch_size=8) if args.tta > 1 else rec.read(imgs, batch_size=8)
    secs = (time.time() - t0) / len(items)
    rows = []
    for it, h in zip(items, hyps):
        ref, hyp = clean(it["text"]), clean(h)
        rows.append({"file": it["file"], "ref": ref, "hyp": hyp, "cer": Levenshtein.distance(ref, hyp) / max(1, len(ref)),
                     "wer": Levenshtein.distance(ref.split(), hyp.split()) / max(1, len(ref.split()))})
    import statistics
    summ = {"recognizer": "heb-ocr Mishkefet-v1" + (" (no LM)" if args.no_lm else " + char LM"), "tta": args.tta,
            "n_lines": len(rows),
            "CER": sum(r["cer"] * len(r["ref"]) for r in rows) / sum(len(r["ref"]) for r in rows),
            "CER_median": statistics.median(r["cer"] for r in rows),
            "WER": statistics.mean(r["wer"] for r in rows),
            "lines_exact": sum(r["cer"] == 0 for r in rows) / len(rows), "secs_per_line": secs}
    out = os.path.join(HERE, "results", "hebrew_eval_hebocr" + ("_nolm" if args.no_lm else "") + ".json")
    json.dump({"summary": summ, "rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    for r in rows[:5]:
        print(round(r["cer"], 3), "|", r["ref"], "|", r["hyp"])


if __name__ == "__main__":
    main()
