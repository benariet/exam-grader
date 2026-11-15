"""Lines of handwritten math with their text, out of FERMAT (ai4bharat/FERMAT, CC BY 4.0: 2,244 photos of solutions
written out by hand, each with its text) - to teach the formula reader (recog/mumer.py) a school notebook's hand.

The text is a solution's, not its lines' (a paragraph runs over several written lines, an equation chain wraps), so a
line's text is found, not counted: the page cut into its written lines (the ink's rows; a ruled notebook's lines taken
out), each line read by the reader as it is, and the reading matched to the stretch of the solution's text (and the
question's - it is written on the page too) it is closest to; that stretch, as the text has it, is the line's label -
kept when the two agree closely enough (a reading that is mostly right: what the reader misread is what it learns).

    python tools/fermat_lines.py cut --data <the parquet files' folder> --out <dir>          (CPU)
        -> <dir>/lines/<id>_<k>.png, <dir>/lines.jsonl {"id", "img_id", "k", "image"}
    python tools/fermat_lines.py read --out <dir> [--shard 0/4]                               (GPU)
        -> <dir>/read_<shard>.jsonl {"image", "read"}
    python tools/fermat_lines.py pair --data <folder> --out <dir> [--min-score 85] [--held 0.1]
        -> <dir>/train.jsonl, <dir>/held.jsonl {"image", "text", "read", "score"} (held out by question)
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[0-9]|[a-zA-Z]|[^\s]")
DROP = {"\\left", "\\right", "\\,", "\\;", "\\!", "\\quad", "\\qquad", "\\displaystyle", "\\(", "\\)", "\\[", "\\]",
        "$", "\\textcolor", "\\text", "\\mathrm", "\\textbf", "\\mathbf", "\\boxed"}


def tokens(latex: str) -> list:
    """LaTeX -> the reader's tokens (a command whole, a digit or letter each, a sign each): '\\frac{11}{2}' ->
    ['\\frac', '{', '1', '1', '}', '{', '2', '}']; spacing, delimiters and the red marks' wrappers left out"""
    from tools.fermat_check import unmark
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\s*\{([^{}]*)\}", r"\1", unmark(latex or ""))
    return [t for t in TOKEN.findall(s) if t not in DROP]


def ink_of(gray: np.ndarray) -> np.ndarray:
    """a page's strokes: darker than the page's own light by a share of its darkest (lighting evened out), a ruled
    notebook's long faint lines and specks taken out"""
    from scipy import ndimage as ndi
    g = gray.astype(np.float32)
    bg = ndi.maximum_filter(ndi.uniform_filter(g, 15), 25)
    d = bg - g
    ink = d > max(25.0, 0.45 * float(np.percentile(d, 99)))
    w = ink.shape[1]
    ruled = ndi.binary_opening(ndi.binary_dilation(ink, structure=np.ones((3, 1))),
                               structure=np.ones((1, max(40, w // 6))))
    ink &= ~ndi.binary_dilation(ruled, structure=np.ones((3, 1)))
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(ink, lab, index=np.arange(1, n + 1))
        ink &= np.isin(lab, 1 + np.flatnonzero(sizes >= 15))
    return ink


def line_bands(gray: np.ndarray):
    """a page's written lines: [(y0, y1, x0, x1)] - the strokes' rows, a line at each peak of them, cut at the lowest
    row between two peaks (lines whose tails touch are still two), each at most a line's pitch above and below its peak"""
    from scipy import ndimage as ndi
    from scipy.signal import find_peaks
    ink = ink_of(gray)
    h = ink.shape[0]
    raw = ink.sum(axis=1).astype(float)
    prof = ndi.gaussian_filter1d(raw, 4)
    if prof.max() <= 0:
        return []
    peaks, _ = find_peaks(prof, distance=max(18, h // 60), prominence=0.12 * prof.max())
    if len(peaks) == 0:
        return []
    pitch = float(np.median(np.diff(peaks))) if len(peaks) > 1 else h / 4
    cuts = [0] + [int(p + np.argmin(prof[p:q])) for p, q in zip(peaks[:-1], peaks[1:])] + [h]
    out = []
    for i, pk in enumerate(peaks):
        a, b = max(cuts[i], int(pk - pitch)), min(cuts[i + 1], int(pk + pitch))
        rows = np.flatnonzero(raw[a:b] > 0.03 * raw[pk])
        if len(rows) == 0:
            continue
        a, b = a + int(rows[0]), a + int(rows[-1]) + 1
        if b - a < 12:
            continue
        cols = np.flatnonzero(ink[a:b].any(axis=0))
        if len(cols) < 15:
            continue
        out.append((a, b, int(cols[0]), int(cols[-1]) + 1))
    return out


def cut(a):
    import pyarrow.parquet as pq
    from PIL import Image
    os.makedirs(os.path.join(a.out, "lines"), exist_ok=True)
    meta = open(os.path.join(a.out, "lines.jsonl"), "w", encoding="utf-8")
    n_pages = n_lines = 0
    for f in sorted(glob.glob(os.path.join(a.data, "*.parquet"))):
        for r in pq.read_table(f, columns=["new_custom_id", "img_id", "image"]).to_pylist():
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")
            if max(img.size) > a.side:
                img.thumbnail((a.side, a.side))
            gray = np.asarray(img)
            n_pages += 1
            for k, (y0, y1, x0, x1) in enumerate(line_bands(gray)):
                pad = max(6, (y1 - y0) // 5)
                crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
                name = f"{r['new_custom_id']}_{k}.png"
                crop.save(os.path.join(a.out, "lines", name))
                meta.write(json.dumps({"id": r["new_custom_id"], "img_id": r["img_id"], "k": k,
                                       "image": f"lines/{name}"}) + "\n")
                n_lines += 1
    meta.close()
    print(f"{n_pages} pages -> {n_lines} lines", flush=True)


def read(a):
    from PIL import Image
    from recog.mumer import MumerReader
    rows = [json.loads(x) for x in open(os.path.join(a.out, "lines.jsonl"), encoding="utf-8")]
    k, n = (int(x) for x in a.shard.split("/"))
    rows = rows[k::n]
    path = os.path.join(a.out, f"read_{k}.jsonl")
    done = {json.loads(x)["image"] for x in open(path, encoding="utf-8")} if os.path.isfile(path) else set()
    rd = MumerReader(adapter_dir=a.adapter or None)
    with open(path, "a", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            if r["image"] in done:
                continue
            c = rd.recognize(Image.open(os.path.join(a.out, r["image"])), n_best=1, max_new_tokens=160)[0]
            out.write(json.dumps({"image": r["image"], "read": c.text}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                out.flush()
                print(i, len(rows), flush=True)


def match(read_toks: list, text_toks: list):
    """a line's reading against a text's tokens -> (score 0..100, start, end): the stretch of the text the reading is
    closest to (rapidfuzz's best partial alignment, each token one symbol)"""
    from rapidfuzz import fuzz
    code = {}
    sym = lambda ts: "".join(code.setdefault(t, chr(0x4E00 + len(code))) for t in ts)  # noqa: E731
    keep = [i for i, t in enumerate(text_toks) if t not in "{}"]          # (braces left out of the matching: '^2'
    s_text, s_read = sym([text_toks[i] for i in keep]), sym([t for t in read_toks if t not in "{}"])   # and '^ { 2 }')
    if not s_read or not s_text:
        return 0.0, 0, 0
    al = fuzz.partial_ratio_alignment(s_read, s_text)
    if al.dest_end <= al.dest_start:
        return 0.0, 0, 0
    return float(al.score), keep[al.dest_start], keep[al.dest_end - 1] + 1


def held_out(img_id, share: float) -> bool:
    return int(hashlib.md5(str(img_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def pair(a):
    import pyarrow.parquet as pq
    texts = {}
    for f in sorted(glob.glob(os.path.join(a.data, "*.parquet"))):
        for r in pq.read_table(f, columns=["new_custom_id", "orig_q", "pert_a"]).to_pylist():
            texts[r["new_custom_id"]] = (tokens("Question : " + (r["orig_q"] or "")),   # (the page heads its parts
                                         tokens("Answer : " + (r["pert_a"] or "")))  #  so; the texts do not)
    meta = {json.loads(x)["image"]: json.loads(x) for x in open(os.path.join(a.out, "lines.jsonl"), encoding="utf-8")}
    reads = {}
    for p in glob.glob(os.path.join(a.out, "read_*.jsonl")):
        for x in open(p, encoding="utf-8"):
            r = json.loads(x)
            reads[r["image"]] = r["read"]
    train = open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8")
    held = open(os.path.join(a.out, "held.jsonl"), "w", encoding="utf-8")
    import collections
    c = collections.Counter()
    for img, rd in reads.items():
        m = meta[img]
        rt = tokens(rd)
        c["read"] += 1
        if len(rt) < a.min_tokens or not re.search(r"\d", rd):    # (a line with a number in it: what is read here)
            c["too short or no number"] += 1
            continue
        best = (0.0, None, 0, 0)
        for part in texts.get(m["id"], ([], [])):
            sc, s, e = match(rt, part)
            if sc > best[0]:
                best = (sc, part, s, e)
        sc, part, s, e = best
        if part is None or sc < a.min_score or not (0.75 <= (e - s) / max(1, len(rt)) <= 1.33):
            c["no close stretch"] += 1
            continue
        label = " ".join(part[s:e])
        rec = {"image": img, "text": label, "read": " ".join(rt), "score": round(sc, 1), "img_id": m["img_id"]}
        (held if held_out(m["img_id"], a.held) else train).write(json.dumps(rec, ensure_ascii=False) + "\n")
        c["held" if held_out(m["img_id"], a.held) else "train"] += 1
        c["exact already" if label == " ".join(rt) else "to learn"] += 1
    train.close()
    held.close()
    print(dict(c), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cut")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--side", type=int, default=1800)
    r = sub.add_parser("read")
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--adapter", default="")
    q = sub.add_parser("pair")
    q.add_argument("--data", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--min-score", type=float, default=85)
    q.add_argument("--min-tokens", type=int, default=4)
    q.add_argument("--held", type=float, default=0.1)
    a = ap.parse_args()
    {"cut": cut, "read": read, "pair": pair}[a.cmd](a)


if __name__ == "__main__":
    main()
