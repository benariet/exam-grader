"""Read a page of handwriting - Hebrew and formulas - as text (4.11, Tal's notebook page): the lines found and read by
Mishkefet-v1, the Hebrew reader (recog/mishkefet.py), and each also by a vision-language model, the formula reader
(Qwen3-VL - it reads formulas well, Hebrew handwriting not at all: it makes Hebrew up). A line is a formula where the
formula reader's reading holds no Hebrew and some math and the Hebrew reader is not sure of its words (kind()).

    python tools/read_notes.py <scan.pdf | image> [--out page.txt] [--model models/qwen3-vl-8b | http://.../v1]
        [--pages 1,2] [--dpi 200] [--hebrew-model models/mishkefet-tal.pt]

--hebrew-model: Mishkefet fine-tuned on the writer's own lines (tools/writing_forms.py, tools/finetune_mishkefet.py:
5.11, six of Tal's lines learned, the other six read with 12% of the letters wrong instead of 15%).

-> each line: 'H | <Hebrew>', 'M | <LaTeX>' or 'E | <Latin letters>' (and the other reader's reading, to compare, with --both)"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PROMPT = ("This image is one handwritten line from a student's notes in electronics. If it is a formula or a "
          "calculation, transcribe it exactly in LaTeX without $ signs. If it is Hebrew text, answer only HEBREW. "
          "Do not solve, correct or explain.")
_HEB = re.compile(r"[֐-׿]")


_MATHY = re.compile(r"=|\\geq|\\leq|\\approx|\\sim|\\frac|\^|_\{|\\rightarrow")
_LATIN = re.compile(r"[A-Za-z]{2,}(?: [A-Za-z]{1,})*")
SURE_HEBREW = 0.945       # Mishkefet's confidence on a line of Hebrew (Tal's page: 0.95-0.98; on formulas 0.87-0.93)


def kind(heb: str, conf: float, formula: str) -> str:
    """'M' a formula, 'H' Hebrew, 'E' a word in Latin letters ('login'), '' noise (a mark, the binding's holes).
    The formula reader writes Hebrew - made up - over a line of Hebrew, and LaTeX over a formula; the Hebrew reader
    reads a formula as Hebrew words too, but less sure of them"""
    f = formula.strip()
    if "HEBREW" in f.upper() or _HEB.search(f):
        return "H" if heb.strip() else ""
    if _LATIN.fullmatch(f):
        return "E"
    mathy = bool(_MATHY.search(f))
    if mathy and conf < SURE_HEBREW:
        return "M"
    if conf >= 0.85 and len(heb.replace(" ", "")) >= 4:
        return "H"
    return "M" if mathy else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--pages", default="")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--hebrew-lm", default=os.path.join(HERE, "models", "dictalm-3.0-1.7b-base"),
                    help="a Hebrew language model choosing among each Hebrew line's 16 readings ('' - none)")
    ap.add_argument("--lm-weight", type=float, default=0.3)
    ap.add_argument("--hebrew-model", default=None, help="a fine-tuned Mishkefet checkpoint (default: the original)")
    args = ap.parse_args()
    from PIL import Image
    from recog import mishkefet
    from recog.remote_vlm import RemoteReader, is_url
    from tools.read_blocks import tidy_latex
    if args.src.lower().endswith(".pdf"):
        from read_page import load_pages
        pages = load_pages(args.src, args.dpi)
    else:
        pages = [Image.open(args.src)]
    want = {int(p) for p in args.pages.split(",") if p.strip()}
    heb = mishkefet.recognizer(checkpoint=args.hebrew_model)
    if is_url(args.model):
        vlm = RemoteReader(args.model)
    else:
        import torch
        from recog.hebrew_vlm import HebrewRecognizer
        vlm = HebrewRecognizer(model_dir=args.model, dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    hlm = mishkefet.HebrewLM(args.hebrew_lm) if args.hebrew_lm and os.path.isdir(args.hebrew_lm) else None
    out = []
    for pno, img in enumerate(pages, 1):
        if want and pno not in want:
            continue
        clean = mishkefet.no_vertical_rules(img.convert("RGB"))
        out.append(f"== page {pno}")
        before = []                                      # (the Hebrew lines chosen so far: the language model's context)
        for r in mishkefet.read_page(clean, heb):
            formula = tidy_latex(vlm.recognize(r["crop"].convert("RGB"), n_best=1, max_new_tokens=160,
                                               prompt=PROMPT)[0].text)
            if r["box"]["y"] < 0.04 * clean.height:
                continue                                 # (the binding's holes)
            k = kind(r["text"], r["confidence"], formula)
            if not k:
                continue
            if k == "H" and hlm is not None:             # (Tal's page: 24% of the characters wrong -> 21%)
                cands = mishkefet.read_nbest(heb, [r["crop"]], n=16, beam_width=32)
                r["text"] = mishkefet.rescore(cands, hlm, weight=args.lm_weight, context=before)[0]
                before.append(r["text"])
            line = f"{k} | {r['text'].strip() if k == 'H' else formula}"
            if args.both:
                line += f"    [{'formula reader' if k == 'H' else 'Hebrew reader'}: {formula if k == 'H' else r['text']}]"
            out.append(line)
    text = "\n".join(out)
    print(text)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
