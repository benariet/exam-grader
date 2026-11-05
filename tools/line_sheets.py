"""Lines of Hebrew handwriting to be transcribed by hand: scans (public-domain letters and diaries - tools/
pseudo_label.py's HASH collection - or anyone's pages written for this) cut into their lines by Mishkefet's page mode
(recog/mishkefet.py), each line saved with Mishkefet's reading, and laid out on numbered sheets to read through; the
transcriptions, joined back, are lines to learn from (tools/finetune_mishkefet.py). Mishkefet learned modern Hebrew
handwriting from synthetic lines and 943 real lines of 19th-century cursive - all that was published with its text
(its README); every line transcribed here is real modern handwriting it has never had.

    python tools/line_sheets.py cut <scans or folders ...> --out data/handwriting/senesh [--per-sheet 8]
    python tools/line_sheets.py hint --out data/handwriting/senesh <the writings' published text ...>
    python tools/line_sheets.py auto --out data/handwriting/senesh [--slack 3] [--margin 6]
    python tools/line_sheets.py join --out data/handwriting/senesh <transcriptions.tsv ...>

cut -> <out>/lines/<id>.png, <out>/lines.jsonl ({"id", "source", "half", "box", "reading", "confidence"}) and
<out>/sheets/lines_NNNNN.jpg (each line under its number; the file named by its first line's - a cut resumed
after a stop starts its sheets where it left off). A notebook's two-page spread is cut at its gutter first
(the lines of facing pages would be read across the fold).
hint: where the writing has been published (Project Ben-Yehuda's public-domain texts - Senesh's diary 'בארץ' is
the diary in these notebooks, edited: spelling made modern, a word changed here and there), each line's place in
it found from Mishkefet's reading (the page's lines together first, then line by line, in order) -> <out>/hints.jsonl
({"n", "id", "reading", "hint", "score"}): the published words of the line, for whoever transcribes it to check
against the handwriting - not the transcription itself.
auto: each placed line's transcription made from the published words and checked by the handwriting itself:
Mishkefet's reading, word by word (aligned with the published line), takes the published word where the line's
image supports it about as well (its CTC score within --slack, recog/mishkefet.ctc_logprob) - so a misread word is
mended, and a word the editor changed stays as written; a line is kept when the image supports the result within
--margin of the reading and most of its words are the published ones -> <out>/auto.tsv ('n<TAB>text<TAB>support')
join: a TSV of 'id<TAB>text' (an empty text, or a lone '?', leaves the line out; --exclude: lines of these
pages too - pages held out to test on) -> <out>/labels.jsonl (--name)
({"image": "lines/<id>.png", "text", "kind": "hebrew", "source"})"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

IMAGES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def gutter(img, max_side: int = 1200):
    """where a two-page spread folds: x in the page, or None when it is one page. The fold is the emptiest column
    in the middle 30% of a landscape page, under half as inked as the text beside it"""
    import cv2
    import numpy as np
    if img.width < 1.2 * img.height:
        return None
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    s = min(1.0, max_side / max(g.shape))
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), max(g.shape) / 30.0)
    flat = np.clip(g / np.maximum(bg, 1.0) * 128.0, 0, 255).astype(np.uint8)      # (the fold's shadow out)
    ink = cv2.adaptiveThreshold(flat, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                max(15, (min(flat.shape) // 40) | 1), 12)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(np.float32)
    col = np.convolve(ink.sum(axis=0), np.ones(9) / 9, mode="same")
    w = len(col)
    lo, hi = int(w * 0.35), int(w * 0.65)
    i = lo + int(np.argmin(col[lo:hi]))
    side = max(col[int(w * 0.08):lo].mean(), col[hi:int(w * 0.92)].mean())   # (a facing page may be empty)
    a = b = i                                    # (the middle of the empty run, not its first column)
    while a > lo and col[a - 1] <= col[i] + 0.05 * side:
        a -= 1
    while b < hi - 1 and col[b + 1] <= col[i] + 0.05 * side:
        b += 1
    i = (a + b) // 2
    if side < 0.01 * ink.shape[0]:
        return None                              # (hardly any writing: nothing to tell by)
    return round(i / s) if 0.42 * w <= i <= 0.58 * w and col[i] < 0.45 * side else None


def halves(img):
    """[(name, image)]: the right page first (Hebrew reads from the right), or the page whole"""
    x = gutter(img)
    if x is None:
        return [("", img)]
    return [("R", img.crop((x, 0, img.width, img.height))), ("L", img.crop((0, 0, x, img.height)))]


def sheet(crops: list, start: int, width: int = 1400):
    """one sheet: the crops one under another, each under its number (start, start + 1, ...)"""
    from PIL import Image, ImageDraw
    rows = []
    for k, c in enumerate(crops):
        c = c.convert("L")
        if c.width > width - 70:
            c = c.resize((width - 70, max(1, round(c.height * (width - 70) / c.width))))
        row = Image.new("L", (width, c.height + 14), 255)
        row.paste(c, (width - c.width, 7))                        # (right-aligned: Hebrew starts on the right)
        d = ImageDraw.Draw(row)
        d.text((6, row.height // 2 - 8), str(start + k), fill=0, font_size=26)
        d.line([(0, row.height - 1), (width, row.height - 1)], fill=170)
        rows.append(row)
    out = Image.new("L", (width, sum(r.height for r in rows)), 255)
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def cut(args):
    from PIL import Image
    from recog import mishkefet
    files = []
    for s in args.src:
        files += sorted(f for f in glob.glob(os.path.join(s, "**", "*"), recursive=True) if f.lower().endswith(IMAGES)) \
            if os.path.isdir(s) else [s]
    files = [f for f in files if "thumb" not in os.path.basename(f)]
    os.makedirs(os.path.join(args.out, "lines"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    rec = mishkefet.recognizer()
    index = os.path.join(args.out, "lines.jsonl")
    done = {json.loads(r)["source"] for r in open(index, encoding="utf-8")} if os.path.exists(index) else set()
    n = sum(1 for _ in open(index, encoding="utf-8")) if os.path.exists(index) else 0
    pending = []
    with open(index, "a", encoding="utf-8") as out:
        for f in files:
            if f in done:
                continue
            img = Image.open(f).convert("RGB")
            if max(img.size) > args.max_side:
                s = args.max_side / max(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)))
            stem = re.sub(r"[^\w]+", "_", os.path.splitext(os.path.basename(f))[0])[-50:]
            kept = 0
            for half, page in halves(img):
                recs = mishkefet.read_page(page, rec, turn=False, clean=True)
                mid = sorted(r["box"]["h"] for r in recs)[len(recs) // 2] if recs else 0
                for j, r in enumerate(recs):
                    text = r["text"].strip()
                    if len(text.replace(" ", "")) < args.min_chars or r["crop"].width < 3 * r["crop"].height \
                            or not 0.5 * mid <= r["box"]["h"] <= 2 * mid:
                        continue                  # (a mark, a page number, a stamp; two lines in one, a scrap)
                    lid = f"{stem}{half}_{j:02d}"
                    r["crop"].convert("L").save(os.path.join(args.out, "lines", lid + ".png"))
                    out.write(json.dumps({"id": lid, "n": n, "source": f, "half": half, "box": r["box"],
                                          "reading": text, "confidence": round(r["confidence"], 3)},
                                         ensure_ascii=False) + "\n")
                    pending.append((n, r["crop"]))
                    n += 1
                    kept += 1
            out.flush()
            print(f"{os.path.basename(f)}: {kept} lines", flush=True)
            while len(pending) >= args.per_sheet:
                chunk, pending = pending[:args.per_sheet], pending[args.per_sheet:]
                sheet([c for _, c in chunk], chunk[0][0]).save(
                    os.path.join(args.out, "sheets", f"lines_{chunk[0][0]:05d}.jpg"), quality=88)
        if pending:
            sheet([c for _, c in pending], pending[0][0]).save(
                os.path.join(args.out, "sheets", f"lines_{pending[0][0]:05d}.jpg"), quality=88)
    print(f"{n} lines -> {index}")


def _letters(text: str):
    """the text's Hebrew letters and single spaces (niqqud, punctuation, digits out) and, for each, its place in text"""
    import unicodedata
    out, where, space = [], [], True
    for i, c in enumerate(text):
        if unicodedata.category(c) == "Mn":
            continue
        if "\u05d0" <= c <= "\u05ea":
            out.append(c)
            where.append(i)
            space = False
        elif not space:
            out.append(" ")
            where.append(i)
            space = True
    return "".join(out), where


def place_lines(readings: list, letters: str, min_page: float = 50.0, min_line: float = 55.0) -> list:
    """[(start, end, score) or None] for each of a page's lines (Mishkefet's readings, in order) in `letters`: the
    page found by its lines together, then each line in turn after the one before it"""
    from rapidfuzz import fuzz
    norm = [_letters(r)[0].strip() for r in readings]
    page = " ".join(t for t in norm if t)
    if len(page) < 30:
        return [None] * len(readings)
    a = fuzz.partial_ratio_alignment(page, letters)
    if a is None or a.score < min_page:
        return [None] * len(readings)
    lo, hi = max(0, a.dest_start - 1500), min(len(letters), a.dest_end + 1500)
    cur, out = max(lo, a.dest_start - 100), []          # (from where the page begins; a weak match moves nothing)
    for t in norm:
        if len(t) < 4:
            out.append(None)
            continue
        w0 = max(lo, cur - 200)
        w1 = min(hi, cur + 2 * len(t) + 600)
        b = fuzz.partial_ratio_alignment(t, letters[w0:w1])
        if b is None or b.score < min_line:
            out.append(None)
            continue
        out.append((w0 + b.dest_start, w0 + b.dest_end, round(b.score, 1)))
        cur = w0 + b.dest_end
    return out


def hint(args):
    import unicodedata
    text = "\n".join(open(f, encoding="utf-8").read() for f in args.src)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    letters, where = _letters(text)
    rows = [json.loads(r) for r in open(os.path.join(args.out, "lines.jsonl"), encoding="utf-8")]
    pages = {}
    for d in rows:
        pages.setdefault((d["source"], d.get("half", "")), []).append(d)
    found = 0
    with open(os.path.join(args.out, "hints.jsonl"), "w", encoding="utf-8") as out:
        for key, ds in pages.items():
            for d, p in zip(ds, place_lines([d["reading"] for d in ds], letters)):
                h = ""
                if p:
                    a, b = where[p[0]], where[min(p[1], len(where)) - 1] + 1
                    h = " ".join(text[a:b].split())
                    found += 1
                out.write(json.dumps({"n": d["n"], "id": d["id"], "reading": d["reading"], "hint": h,
                                      "score": p[2] if p else 0}, ensure_ascii=False) + "\n")
    print(f"{found} of {len(rows)} lines placed in the text -> {os.path.join(args.out, 'hints.jsonl')}")


def word_pairs(a: list, b: list) -> list:
    """(i, j): word i of a set against word j of b where the two differ (a Levenshtein alignment of the word lists)"""
    from rapidfuzz.distance import Levenshtein
    return [(op.src_pos, op.dest_pos) for op in Levenshtein.editops(a, b) if op.tag == "replace"]


def guided(reading: str, published: str, score, slack: float = 3.0) -> tuple:
    """reading's words, each set against its published word, the published one taken where score(line) (the image's
    support for a line: a log-probability) drops by no more than slack -> (the line, its support, words taken)"""
    cur = reading.split()
    best = score(" ".join(cur))
    taken = 0
    for i, j in word_pairs(cur, published.split()):
        trial = cur[:i] + [published.split()[j]] + cur[i + 1:]
        s = score(" ".join(trial))
        if s >= best - slack:
            cur, best, taken = trial, s, taken + 1
    return " ".join(cur), best, taken


def auto(args):
    from PIL import Image
    from recog import mishkefet
    rec = mishkefet.recognizer()
    kept = 0
    rows = [json.loads(r) for r in open(os.path.join(args.out, "hints.jsonl"), encoding="utf-8")]
    with open(os.path.join(args.out, "auto.tsv"), "w", encoding="utf-8") as out:
        for d in rows:
            if not d["hint"]:
                continue
            img = Image.open(os.path.join(args.out, "lines", d["id"] + ".png")).convert("L")
            score = lambda t: mishkefet.ctc_logprob(rec, img, t)
            base = score(d["reading"])
            line, sup, taken = guided(d["reading"], d["hint"], score, args.slack)
            words = line.split()
            pub = set(d["hint"].split())
            share = sum(w in pub for w in words) / max(1, len(words))
            if sup >= base - args.margin and share >= args.min_share:
                out.write(f"{d['n']}\t{line}\t{sup - base:.1f}\n")
                kept += 1
    print(f"{kept} lines transcribed -> {os.path.join(args.out, 'auto.tsv')}")


def join(args):
    lines = {}
    for r in open(os.path.join(args.out, "lines.jsonl"), encoding="utf-8"):
        d = json.loads(r)
        lines[d["id"]] = d
        lines[str(d["n"])] = d
    kept, skip = 0, [x for x in args.exclude.split(",") if x]
    with open(os.path.join(args.out, args.name), "w", encoding="utf-8") as out:
        for tsv in args.src:
            for row in open(tsv, encoding="utf-8"):
                if "\t" not in row:
                    continue
                key, text = row.rstrip("\n").split("\t", 1)
                text = " ".join(text.split())
                d = lines.get(key.strip())
                if d is None or text in ("", "?") or any(x in d["source"] for x in skip):
                    continue
                out.write(json.dumps({"image": f"lines/{d['id']}.png", "text": text, "kind": "hebrew",
                                      "source": d["source"]}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"{kept} lines -> {os.path.join(args.out, args.name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cut", "hint", "auto", "join"])
    ap.add_argument("src", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-sheet", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=2600)
    ap.add_argument("--slack", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=6.0)
    ap.add_argument("--min-share", type=float, default=0.6)
    ap.add_argument("--name", default="labels.jsonl", help="join: the file written in --out")
    ap.add_argument("--exclude", default="", help="join: comma-separated pieces of source names left out")
    args = ap.parse_args()
    {"cut": cut, "hint": hint, "auto": auto, "join": join}[args.mode](args)


if __name__ == "__main__":
    main()
