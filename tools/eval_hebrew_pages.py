"""How well Hebrew handwriting is read: pages whose text is known, read by Mishkefet-v1 (a checkpoint - the original,
or one fine-tuned by tools/finetune_mishkefet.py), the character and word error rates against their text (niqqud,
punctuation and Latin letters left out; the lines joined in reading order).

    python tools/eval_hebrew_pages.py <pages.jsonl> [--checkpoint models/mishkefet-x.pt] [--hebrew-lm ...]
    python tools/eval_hebrew_pages.py <labels.jsonl> --lines [--checkpoint ...]

pages.jsonl: {"image": path, "text": path to its text, or "ref": the text itself, "name": ...} a line.
--lines: lines already cut, with their text (tools/line_sheets.py join, tools/writing_forms.py cut -> labels.jsonl:
{"image", "text"}), each read alone - the CER of the lines, normalized the same way"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def norm(t: str) -> str:
    import unicodedata
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")   # (niqqud)
    return " ".join(re.sub(r"[^א-ת ]", " ", t).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--hebrew-lm", default="")
    ap.add_argument("--min-conf", type=float, default=0.0, help="lines less sure than this left out of the reading")
    ap.add_argument("--max-side", type=int, default=2600)
    ap.add_argument("--lines", action="store_true", help="the file holds lines with their text, not pages")
    args = ap.parse_args()
    from PIL import Image
    from rapidfuzz.distance import Levenshtein
    from recog import mishkefet
    rec = mishkefet.recognizer(checkpoint=args.checkpoint)
    if args.lines:
        base = os.path.dirname(os.path.abspath(args.pages))
        rows = [json.loads(r) for r in open(args.pages, encoding="utf-8")]
        got = rec.read([Image.open(os.path.join(base, r["image"])).convert("L") for r in rows])
        e = sum(Levenshtein.distance(norm(g), norm(r["text"])) for g, r in zip(got, rows))
        n = sum(len(norm(r["text"])) for r in rows)
        print(f"{len(rows)} lines: CER {e / max(1, n):.3f}")
        return
    hlm = mishkefet.HebrewLM(args.hebrew_lm) if args.hebrew_lm and os.path.isdir(args.hebrew_lm) else None
    ce = cn = we = wn = 0
    for row in open(args.pages, encoding="utf-8"):
        p = json.loads(row)
        base = os.path.dirname(os.path.abspath(args.pages))
        ref = norm(p.get("ref") or open(os.path.join(base, p["text"]), encoding="utf-8").read())
        if p["image"].lower().endswith(".pdf"):
            import pypdfium2 as pdfium
            img = pdfium.PdfDocument(os.path.join(base, p["image"]))[0].render(scale=200 / 72).to_pil().convert("RGB")
        else:
            img = Image.open(os.path.join(base, p["image"])).convert("RGB")
        if max(img.size) > args.max_side:
            s = args.max_side / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)))
        recs = [r for r in mishkefet.read_page(img, rec) if r["confidence"] >= args.min_conf]
        if hlm is not None and recs:
            nb = mishkefet.read_nbest(rec, [r["crop"] for r in recs], n=16, beam_width=32)
            for r, t in zip(recs, mishkefet.rescore(nb, hlm, weight=0.3)):
                r["text"] = t
        hyp = norm(" ".join(r["text"] for r in recs))
        c, w = Levenshtein.distance(hyp, ref), Levenshtein.distance(hyp.split(), ref.split())
        ce, cn, we, wn = ce + c, cn + len(ref), we + w, wn + len(ref.split())
        print(f"{p.get('name', p['image'])}: CER {c / max(1, len(ref)):.3f}  WER {w / max(1, len(ref.split())):.3f}"
              f"  ({len(ref.split())} words)", flush=True)
    print(f"ALL: CER {ce / max(1, cn):.3f}  WER {we / max(1, wn):.3f}  ({wn} words)")


if __name__ == "__main__":
    main()
