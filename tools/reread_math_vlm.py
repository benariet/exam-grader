"""Read an exam's math pieces again with a big vision-language model (Qwen3-VL-8B in 16 bits - a 48 GB GPU) - an
experiment: is a general VLM a better second formula reader than Uni-MuMER on real handwriting?

    python tools/reread_math_vlm.py <exam.pdf> <its tools/read_exams.py json> <out.json> [--model models/qwen3-vl-8b]

Each math piece (its box from the saved reading) is cut from the page rendered again at --dpi (300: twice the
reading's 150, the boxes scaled) and read with the model; the readings replace math2 in a copy of the json (the
Uni-MuMER ones kept as math2_mumer), so tools/recheck_exams.py judges the exam with them unchanged."""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PROMPT_VLM = ("This image is a piece of a student's handwritten exam in electronics (CMOS circuits, logic gates, "
              "delays). Transcribe the handwritten math exactly as written, in LaTeX - do not solve, correct or "
              "complete anything, keep the student's numbers and units. Output only the LaTeX, no $ signs, no words.")


def clean_latex(t: str) -> str:
    """the model's answer -> LaTeX: no code fences, no $ or \\( \\) wrappers, one line"""
    t = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", t.strip())
    t = t.strip()
    for a, b in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if t.startswith(a) and t.endswith(b) and len(t) > len(a) + len(b):
            t = t[len(a):-len(b)].strip()
    return " ".join(t.split())


REPEATS = 3          # one reading for this many pieces or more: made up (but a line on every page, '2025-2026')
_HEB_TEXT = re.compile(r"\\text\s*\{[^{}]*[\u0590-\u05FF][^{}]*\}")
_WORDS = re.compile(r"(?:\b[A-Za-z]{3,}\b[ ,.]+){4,}")


def _trusted(text: str, counts: dict) -> bool:
    """a VLM reading worth putting first: LaTeX, not words; not the same as many other pieces' unless trivially short"""
    if not text or _WORDS.search(re.sub(r"\\text\s*\{[^{}]*\}", " ", text)) or re.search(r"[\u0590-\u05FF]", text):
        return False
    return counts.get(text, 0) < REPEATS or len(text) <= 12


_TAIL = re.compile(r"(?:\\quad|,|;|\s)\s*(?P<v>[A-Za-z])(?:_\{?\w+\}?)?\s*=\s*-?\d\s*$")   # (one digit: 'x = 1')


def _letters(t: str) -> str:
    return re.sub(r"\\[A-Za-z]+", " ", t)                 # (the letters of the math, not of \frac, \text)


def drop_made_up_tail(vlm: str, other: str) -> str:
    """'T = \\frac{1}{5 \\cdot 10^9} = 200 psec x = 1': a letter set to one digit at the end of the VLM's reading, the
    letter found nowhere else in it nor in the other reader's - left out ('b = 3' after 'a = 2' when the other reader
    has 'b' too stays; a longer value, '= 12.075', is never taken for made up)"""
    m = _TAIL.search(vlm)
    if not m or "=" not in vlm[:m.start()]:
        return vlm
    v = m.group("v")
    if re.search(r"(?<![A-Za-z])" + v, _letters(vlm[:m.start()])) or re.search(r"(?<![A-Za-z])" + v, _letters(other)):
        return vlm
    return vlm[:m.start()].rstrip(" ,;")


def merge(data) -> dict:
    """each math piece's math2: the VLM's readings first where trusted (_trusted), then Uni-MuMER's (math2_mumer);
    Uni-MuMER's alone where not -> counts {"vlm": n, "mumer": n}"""
    from collections import Counter
    segs = [s for page in data for segs in page["lines"] for s in segs if s["kind"] == "math" and s.get("math2")]
    counts = Counter(s["math2"][0][0] for s in segs if s.get("math2_mumer"))
    used = {"vlm": 0, "mumer": 0}
    for s in segs:
        if "math2_mumer" not in s:
            continue
        vlm, mumer = s.pop("math2_vlm", None) or s["math2"], s["math2_mumer"]
        s["math2_vlm"] = vlm
        vlm = [(" ".join(_HEB_TEXT.sub(" ", t).split()), lp) for t, lp in vlm]   # (Hebrew words it wrote out: not math)
        vlm = [(drop_made_up_tail(t, mumer[0][0] if mumer else ""), lp) for t, lp in vlm if t] or [("", 0.0)]
        if _trusted(vlm[0][0], counts):
            s["math2"] = vlm + [m for m in mumer if m[0] not in {v[0] for v in vlm}]
            used["vlm"] += 1
        else:
            s["math2"] = mumer
            used["mumer"] += 1
    return used


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--merge":
        data = json.load(open(sys.argv[2], encoding="utf-8"))
        print(merge(data))
        json.dump(data, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False)
        return
    from read_page import load_pages
    from recog.hebrew_vlm import HebrewRecognizer
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("json")
    ap.add_argument("out")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--read-dpi", type=int, default=150, help="the dpi the saved reading was made at")
    ap.add_argument("--n-best", type=int, default=2)
    ap.add_argument("--pad", type=int, default=6, help="pixels around each box (at the reading's dpi)")
    args = ap.parse_args()
    data = json.load(open(args.json, encoding="utf-8"))
    from recog.page import flatten_photo
    imgs = load_pages(args.pdf, args.dpi)
    f = args.dpi / args.read_dpi
    # a page the reading took for a photo (info['photo']) was flattened first - its boxes are in the flattened page: cut
    # those from it, flattened again at the reading's dpi, and enlarged by the same factor
    low = None
    for p, page in enumerate(data):
        if page["info"].get("photo"):
            low = low or load_pages(args.pdf, args.read_dpi)
            flat, _ = flatten_photo(low[p])
            imgs[p] = flat.resize((round(flat.width * f), round(flat.height * f)))
    jobs = [(p, s) for p, page in enumerate(data) for segs in page["lines"] for s in segs if s["kind"] == "math"]
    print(f"{len(jobs)} math pieces on {len(imgs)} pages - loading {args.model}", flush=True)
    t0 = time.time()
    import torch
    rec = HebrewRecognizer(model_dir=args.model, dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None)
    print(f"model ready in {time.time() - t0:.0f} s ({rec.device}, {rec.dtype})", flush=True)
    t1 = time.time()
    for k, (p, s) in enumerate(jobs, 1):
        x0, y0, x1, y1 = s["box"]
        img = imgs[p].convert("L")
        a, b = max(0, int((min(x0, x1) - args.pad) * f)), max(0, int((min(y0, y1) - args.pad) * f))
        c, d = min(img.width, int((max(x0, x1) + args.pad) * f)), min(img.height, int((max(y0, y1) + args.pad) * f))
        if c - a < 4 or d - b < 4:                           # (a box off the page: keep Uni-MuMER's reading)
            continue
        crop = img.crop((a, b, c, d))
        try:
            cands = rec.recognize(crop, n_best=args.n_best, max_new_tokens=160, prompt=PROMPT_VLM)
            got = [(clean_latex(c.text), float(c.logp)) for c in cands if clean_latex(c.text)]
        except Exception as e:                              # noqa: BLE001  (one piece: keep Uni-MuMER's)
            print(f"piece {k}: {type(e).__name__}: {e}", flush=True)
            got = []
        if got:
            if s.get("math2"):
                s["math2_mumer"] = s["math2"]
            s["math2"] = got
        if k % 20 == 0 or k == len(jobs):
            each = (time.time() - t1) / k
            print(f"{k}/{len(jobs)} pieces ({each:.1f} s each, about {each * (len(jobs) - k) / 60:.1f} min left)", flush=True)
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"DONE - saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
