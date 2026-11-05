"""Hebrew handwriting with no text to it - letters, diaries, notes (the HASH collection of scanned Hebrew handwriting,
github.com/HeOCR/hash: public-domain and CC scans) - read by Mishkefet-v1 and DictaLM (recog/mishkefet.py), and the
lines read surely enough kept as lines to learn from (tools/finetune_mishkefet.py): a model taught by its own sure
readings learns the hands it reads (self-training) - the unsure ones it would only learn its mistakes from.

    python tools/pseudo_label.py <scan images or folders ...> --out data/handwriting/hash [--min-conf 0.95]
        [--min-known 0.75] [--exclude <entry ids>] [--hebrew-lm models/dictalm-3.0-1.7b-base]

A line is kept when Mishkefet is sure of it (its weakest path's confidence), most of its words are Hebrew words (the
word list Mishkefet's language model knows) and it is long enough to be a line of text, not a mark. -> <out>/*.png and
<out>/labels.jsonl ({"image", "text", "kind": "hebrew", "source", "confidence", "known"})"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

IMAGES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def known_share(text: str, wlm) -> float:
    words = re.findall(r"[א-ת]{2,}", text)
    return sum(1 for w in words if wlm.known(w)) / len(words) if words else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-conf", type=float, default=0.95)
    ap.add_argument("--min-known", type=float, default=0.75)
    ap.add_argument("--min-chars", type=int, default=8)
    ap.add_argument("--exclude", default="", help="comma-separated pieces of file names left out (held-out pages)")
    ap.add_argument("--hebrew-lm", default=os.path.join(HERE, "models", "dictalm-3.0-1.7b-base"))
    ap.add_argument("--max-side", type=int, default=2600, help="a bigger scan is scaled down to this")
    args = ap.parse_args()
    from PIL import Image
    from recog import mishkefet
    files = []
    for s in args.src:
        files += sorted(f for f in glob.glob(os.path.join(s, "**", "*"), recursive=True) if f.lower().endswith(IMAGES)) \
            if os.path.isdir(s) else [s]
    skip = [x for x in args.exclude.split(",") if x]
    files = [f for f in files if not any(x in f for x in skip) and "_thumb" not in f]
    os.makedirs(args.out, exist_ok=True)
    rec, wlm = mishkefet.recognizer(), mishkefet.word_lm()
    hlm = mishkefet.HebrewLM(args.hebrew_lm) if args.hebrew_lm and os.path.isdir(args.hebrew_lm) else None
    kept = seen = 0
    with open(os.path.join(args.out, "labels.jsonl"), "a", encoding="utf-8") as out:
        for k, f in enumerate(files, 1):
            img = Image.open(f).convert("RGB")
            if max(img.size) > args.max_side:
                s = args.max_side / max(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)))
            try:
                recs = mishkefet.read_page(img, rec)
            except Exception as e:                                    # noqa: BLE001 (one page: the rest go on)
                print(f"{f}: {type(e).__name__}: {e}", flush=True)
                continue
            sure = [r for r in recs if r["confidence"] >= args.min_conf
                    and len(r["text"].replace(" ", "")) >= args.min_chars]
            if hlm is not None and sure:
                nb = mishkefet.read_nbest(rec, [r["crop"] for r in sure], n=16, beam_width=32)
                for r, t in zip(sure, mishkefet.rescore(nb, hlm, weight=0.3)):
                    r["text"] = t
            stem = re.sub(r"[^\w]+", "_", os.path.splitext(os.path.basename(f))[0])[-60:]
            n = 0
            for j, r in enumerate(sure):
                ks = known_share(r["text"], wlm)
                seen += 1
                if ks < args.min_known:
                    continue
                name = f"{stem}_l{j:02d}.png"
                r["crop"].convert("L").save(os.path.join(args.out, name))
                out.write(json.dumps({"image": name, "text": r["text"].strip(), "kind": "hebrew", "source": f,
                                      "confidence": round(r["confidence"], 3), "known": round(ks, 2)},
                                     ensure_ascii=False) + "\n")
                n += 1
            kept += n
            print(f"{k}/{len(files)} {os.path.basename(f)}: {len(recs)} lines, {n} kept", flush=True)
    print(f"{kept} lines kept of {seen} read surely enough -> {args.out}/labels.jsonl")


if __name__ == "__main__":
    main()
