"""Read each handwritten line whole, in the order of its answer, with the lines before it as context (after the
blocks experiment, 4.11: a whole answer read at once was invented, looped and glued - one line at a time keeps the
answer's order and context and leaves the model no room to go on writing):

    python tools/reread_lines.py <exam.pdf> <its reading json> <out.json> --booklet booklet/moedC [--pages 9,11]

The reading (tools/read_exams.py --booklet, merged with tools/reread_math_vlm.py --merge) gives the page's lines and
their pieces. Each line holding math is cut whole from the page (its print erased, at 300 dpi) and read by
Qwen3-VL-8B, told what the previous lines of the same answer said; that reading becomes the line's one math piece,
first reader, and Uni-MuMER's readings of the line's pieces, side by side, the second reader. Where the two agree a
verdict is sure; where they differ the checks (grader.selfcheck.check_pages: two readers as two experts) weigh both
and are not sure. Hebrew pieces are kept as they were; a line whose math the student crossed out stays out."""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.read_blocks import math_of  # noqa: E402

PROMPT_LINE = (
    "This image is one handwritten line of a student's answer in an electronics exam (CMOS circuits, logic gates, "
    "delays, the body effect), written in Hebrew and math.{context} Transcribe only the line in the image, exactly as "
    "written: the math in LaTeX without $ signs - subscripts as t_{{pHL}}, V_{{DD}}, V_{{Tn}}, powers as 10^{{-12}}, "
    "never Unicode subscripts - and Hebrew words as Hebrew text. Keep the student's numbers, units and mistakes: do "
    "not solve, correct, complete or explain. If the image holds only a mark, a dash, an arrow or part of a drawing, "
    "output an empty line. Output that line only.")
# (without the example names - 4.11, lines5 - fewer lines were made up, but names and the spaces between statements
# were written otherwise and two of the lecturer's five catches were lost)

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋ₙₚₜ", "0123456789+-npt")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def latexify(t: str) -> str:
    """a line read in plain text where the math is LaTeX: Unicode sub- and superscripts as _{..} and ^{..}; names
    written run together as this course writes them - 'tPD1', 'tpHL' (a delay), 'VDD', 'VTn', 'VOHmax' (a voltage) -
    with their subscripts"""
    t = re.sub(r"([₀-₉₊₋ₙₚₜ]+)", lambda m: "_{" + m.group(1).translate(_SUB) + "}", t)
    t = re.sub(r"([⁰-⁹⁺⁻¹²³]+)", lambda m: "^{" + m.group(1).translate(_SUP) + "}", t)
    t = re.sub(r"(?<![A-Za-z\\_])t([pP][A-Za-z]{0,3}\d?)(?![A-Za-z_{])", r"t_{\1}", t)
    t = re.sub(r"(?<![A-Za-z\\_])V([A-Z][A-Za-z]{1,5}|[a-z][A-Za-z]{1,4})(?![A-Za-z_{])", r"V_{\1}", t)
    t = re.sub(r"(?<=\d)(\s*\[?\s*)v(?=\s*\]?\s*(?:$|[,;=)\\]))", r"\1V", t)    # ('-0.496 v', '1.8[v]': volts)
    # a value with its unit, then the next name set equal ('t_{pD1} = 16.1 ps t_{pD2} = 27.6 ps'): two statements
    t = re.sub(r"(\d\s*\[?\s*(?:[munpfkM]\s*|\\mu\s*)?(?:s|sec|V|A|F|Hz|W|\\Omega)\]?)\s+"
               r"(?=[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])\s*=)", r"\1 \\quad ", t)
    return t
CONTEXT = " The lines before it in the same answer read:\n{lines}\n"
KEEP_CONTEXT = 4             # previous lines told to the model


def clean_line(t: str) -> str:
    """a line read with the model's loops taken out: spacing commands written again and again ('\\quad \\quad ...',
    '\\cdot \\cdot ...') once; the line written twice in a row ('a = 1; : a = 1;') once"""
    t = re.sub(r"(?:\s*(?:\\quad|\\qquad|(?<!\\)\\[,; ]|~))+",              # (not '\\': a new line)
               lambda m: r" \quad " if "quad" in m.group(0) else " ", t)  # (a wide space parts two statements: kept)
    t = re.sub(r"(?:\s*\\cdot){2,}", r" \\cdot", t)
    t = re.sub(r"(?:\s*\\ldots|\s*\.\.\.){2,}", r" \\ldots", t)
    t = re.sub(r"\\\\\s*[;,.:]", r"\\\\", t)                      # ('200ps\\;': a new line, no more)
    t = re.sub(r"^(?:\s*\\quad)+|(?:\s*(?:\\\\|\\quad))+\s*$", "", t)
    m = re.match(r"^\s*(.{10,}?)[\s:;,.|]*\1[\s:;,.|]*$", t)
    if m:
        t = m.group(1)
    return " ".join(t.split()).strip(" ,;:|")


_NUM = re.compile(r"\d+(?:\.\d+)?")


def _plain(t: str) -> str:
    """a reading as its letters and digits alone, lower case (LaTeX commands and spacing left out)"""
    t = re.sub(r"\\(?:frac|left|right|text|mathrm|cdot|times|quad|approx|overline|bar|sqrt)", " ", t)
    t = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", t)
    return re.sub(r"[^0-9a-z.]", "", t.lower())


def numbers(t: str) -> list:
    """the numbers of a reading, digits spaced apart ('1 . 1 5', Uni-MuMER's way) put together"""
    t = re.sub(r"(?<=\d)\s+(?=[\d.])|(?<=\.)\s+(?=\d)", "", t)
    return [n.rstrip(".") for n in _NUM.findall(t)]


def made_up(first: str, *seconds: str) -> bool:
    """is the whole-line reading (first) the model writing on, not reading - the line's pieces as read before (each
    of `seconds`: Uni-MuMER's, the vision model's piece by piece) holding little of it? A line with a small formula
    beside Hebrew ('Y_2') read as 't_{pHL} = 8.60 ns, ...'; a dash read as a whole line of values. Two readers that
    differ in a symbol or a digit are not this: the checks weigh them; here the first holds what no other reading
    saw - much longer, its numbers not among theirs"""
    return all(_unsupported(first, s) for s in seconds) if seconds else bool(_plain(first))


def _unsupported(first: str, second: str) -> bool:
    from difflib import SequenceMatcher
    a, b = _plain(first), _plain(second)
    if not a:
        return False
    if not b:
        return len(a) > 6
    sim = SequenceMatcher(None, a, b, autojunk=False).ratio()
    if len(a) > 2.5 * len(b) + 6 and sim < 0.4:
        return True
    n1, n2 = numbers(first), set(numbers(second))
    if n1:
        support = sum(1 for n in n1 if n in n2) / len(n1)
        if support < 0.34 and sim < 0.5:
            return True
    return False


def _first(s, key):
    return ((s.get(key) or [["", 0.0]])[0][0] or "").strip()


def math_lines(page: dict) -> list:
    """the page's lines holding math, top to bottom -> [(line index, its math pieces left to right, the others)];
    pieces the student crossed out (tools/find_crossed.py / read_page's) left out"""
    out = []
    for i, segs in enumerate(page.get("lines", [])):
        maths = [s for s in segs if s["kind"] == "math" and (s.get("crossed") or {}).get("share", 0) < 0.6
                 and not s.get("made_up")]
        if not maths:
            continue
        rest = [s for s in segs if s not in maths and s["kind"] != "math"]
        out.append((i, sorted(maths, key=lambda s: s["box"][0]), rest))
    return sorted(out, key=lambda t: min(s["box"][1] for s in t[1]))


VIEW_SCALES = (1.0, 0.75, 1.33)     # --views: the line's image as cut, smaller, larger


def medoid(texts: list) -> str:
    """of several readings of one line, the one most like the others (letters and digits alone; ties: the first -
    the image as cut): a digit one look misreads the others outvote"""
    from difflib import SequenceMatcher
    if len(texts) < 3:
        return texts[0] if texts else ""
    plain = [_plain(t) for t in texts]
    score = [sum(SequenceMatcher(None, a, b, autojunk=False).ratio() for j, b in enumerate(plain) if j != i)
             for i, a in enumerate(plain)]
    return texts[max(range(len(texts)), key=lambda i: (score[i], -i))]


def read_views(reader, crop, prompt: str, views: int = 1) -> str:
    """a line read at `views` scales of its image (VIEW_SCALES), the reading most like the others kept (medoid)"""
    from PIL import Image
    texts = []
    for sc in VIEW_SCALES[:max(1, views)]:
        img = crop if sc == 1.0 else crop.resize((max(8, round(crop.width * sc)), max(8, round(crop.height * sc))),
                                                  Image.LANCZOS)
        t = reader.read(img, max_new_tokens=160, prompt=prompt).splitlines()
        texts.append(clean_line(" \\\\ ".join(t[:4])))     # (a line of the reading that holds two: both)
    return medoid(texts)


def text_lines(page: dict) -> list:
    """the page's lines with no math piece - read as Hebrew or English only (a formula the layout took for text: its
    numbers in no reading) -> [(line index, its pieces)]; a line all crossed out left out"""
    out = []
    for i, segs in enumerate(page.get("lines", [])):
        live = [s for s in segs if (s.get("crossed") or {}).get("share", 0) < 0.6]
        if live and not any(s["kind"] == "math" for s in segs):
            out.append((i, live))
    return out


def text_line_math(text: str) -> str:
    """a text line read whole -> its math when it states a value ('t_{pHL} = 69.8 ps': a name or an expression set
    equal to something holding a digit), else '' - the rest of such a line (Hebrew words, a lone number, a letter)
    stays as the text readers read it"""
    m = latexify(math_of(text))
    if not m or not re.search(r"(?<![<>!])=", m):
        return ""
    sides = [x for x in re.split(r"(?<![<>!])=", m)]
    return m if any(re.search(r"\d", x) for x in sides[1:]) and re.search(r"[0-9A-Za-z]", sides[0]) else ""


def text_segment(math: str, text: str, segs: list, box) -> dict:
    """a text line's math read whole -> a math piece of its own (one reader: the checks are not sure of it alone)"""
    return {"kind": "math", "box": list(box), "p_kind": min(s.get("p_kind", 1.0) for s in segs),
            "nbest": [[math, 0.0]], "line_text": text, "text_line": True}


def second_text_readings(found: list, reader) -> tuple:
    """the text lines' math (found: [(page, line index, its segment, the line's crop)]) read again by a formula reader
    (Uni-MuMER: the second reader of the math lines) - a value the vision model wrote that this reading does not hold
    (tools/reread_lines.made_up: its numbers in neither) is the model writing, not reading: left out; the others get
    the second reading beside theirs (the checks weigh the two) -> (kept [(page, i, seg)], left out)"""
    kept, dropped = [], 0
    for pno, i, seg, crop in found:
        try:
            second = (reader.recognize(crop, n_best=1) or [None])[0]
            second = second.text if second is not None else ""
        except Exception:                                  # noqa: BLE001  (a reader that fails: no second reading)
            second = ""
        first = seg["nbest"][0][0]
        if made_up(first, second):
            dropped += 1
            continue
        if second and second != first:
            seg["math2"] = [[second, 0.0]]
        kept.append((pno, i, seg))
    return kept, dropped


def second_reading(maths: list) -> str:
    """the line's pieces as Uni-MuMER read them (its own readings, not the merged ones), left to right"""
    return " ".join(t for t in (_first(s, "math2_mumer") or _first(s, "nbest") for s in maths) if t)


def pieces_reading(maths: list) -> str:
    """the line's pieces as read piece by piece and merged (the vision model's reading first), left to right"""
    return " ".join(t for t in (_first(s, "math2") or _first(s, "nbest") for s in maths) if t)


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def hand_page(pdf_high, pdf_low, pno: int, booklet: dict):
    """the page's handwriting alone at the high dpi (print erased) -> a grey PIL image"""
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    from read_page import handwriting_only
    from recog.page import clean_scan
    page, _ = clean_scan(pdf_low[pno - 1])
    hand = None
    if booklet and pno in booklet["pages"]:
        _, hand, _ = handwriting_only(page, booklet["pages"][pno])
    if hand is None:
        hand = np.asarray(page) < 128
    keep = ndi.binary_dilation(hand, iterations=2)
    big = np.asarray(pdf_high[pno - 1].convert("L"))
    mask = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((big.shape[1], big.shape[0]))) > 0
    return Image.fromarray(np.where(mask, big, 255).astype(np.uint8))


def line_segment(text: str, maths: list, box) -> dict:
    """a line read whole -> its one math piece: the whole-line reading first, Uni-MuMER's pieces second (when they
    differ - or none: the checks weigh both, not sure)"""
    first, second = latexify(math_of(text)), second_reading(maths)
    seg = {"kind": "math", "box": list(box), "p_kind": min(s.get("p_kind", 1.0) for s in maths),
           "nbest": [[first or second, 0.0]], "line_text": text, "pieces": [_first(s, "math2") for s in maths]}
    if first and second != first:
        seg["math2"] = [[second, 0.0]]
    return seg


def refilter(lines_data: list, pieces_data: list) -> tuple:
    """a run of this tool judged again without the model (the line readings kept in "line_text"): loops taken out,
    made-up lines given back their pieces (from the reading it started from) -> (data, lines, made up)"""
    n = bad = 0
    for page, orig in zip(lines_data, pieces_data):
        by_i = {i: (maths, rest) for i, maths, rest in math_lines(orig)}
        for i, segs in enumerate(page.get("lines", [])):
            seg = next((s for s in segs if "line_text" in s), None)
            if seg is None or i not in by_i:
                page["lines"][i] = orig["lines"][i]   # (not read whole: its pieces as in the reading given - merged again)
                continue
            maths, rest = by_i[i]
            text = clean_line(seg["line_text"])
            n += 1
            first = latexify(math_of(text))
            if not first or made_up(first, second_reading(maths), pieces_reading(maths)):
                bad += 1
                page["lines"][i] = orig["lines"][i]
            else:
                page["lines"][i] = [line_segment(text, maths, seg["box"])] + rest
    return lines_data, n, bad


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--refilter":       # <lines.json> <the reading it started from> <out>
        data, n, bad = refilter(*(json.load(open(p, encoding="utf-8")) for p in sys.argv[2:4]))
        json.dump(data, open(sys.argv[4], "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{n} lines, {bad} made up: their pieces kept -> {sys.argv[4]}")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("reading")
    ap.add_argument("out")
    ap.add_argument("--booklet", default=os.path.join(HERE, "booklet", "moedC"))
    ap.add_argument("--pages", default="")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--read-dpi", type=int, default=150)
    ap.add_argument("--text-lines", action="store_true", help="the lines read as Hebrew or English only read whole too: "
                    "a value stated in one ('t_{pHL} = 69.8 ps') becomes a math piece of its own")
    ap.add_argument("--views", type=int, default=1, help="each math line read at this many scales of its image "
                    "(1-3: as cut, x0.75, x1.33), the reading most like the others kept")
    ap.add_argument("--text-second", default="mumer", help="the text lines' values read again by Uni-MuMER and kept "
                    "only where its reading holds their numbers ('none': kept as the vision model read them)")
    args = ap.parse_args()
    from read_page import booklet_pages, load_pages
    from tools.read_blocks import BlockReader
    data = json.load(open(args.reading, encoding="utf-8"))
    want = {int(p) for p in re.split(r"[,;:+\s]+", args.pages) if p.strip()}
    book = booklet_pages(args.booklet)
    low, high = load_pages(args.pdf, args.read_dpi), load_pages(args.pdf, args.dpi)
    f = args.dpi / args.read_dpi
    t0 = time.time()
    reader = BlockReader(args.model)
    print(f"model ready in {time.time() - t0:.0f} s", flush=True)
    n_lines = n_agree = n_made_up = n_text = n_text_math = 0
    found = []                  # (the text lines' values waiting for the second reader: page, line, segment, crop)
    for pno, page in enumerate(data, 1):
        if (want and pno not in want) or page.get("info", {}).get("photo"):
            continue
        lines = math_lines(page)
        texts = dict(text_lines(page)) if args.text_lines else {}
        if not lines and not texts:
            continue
        t1 = time.time()
        img = hand_page(high, low, pno, book)
        before = []
        new_lines = {}
        order = [(i, maths, rest) for i, maths, rest in lines] + [(i, None, segs) for i, segs in texts.items()]
        order.sort(key=lambda t: min(s["box"][1] for s in page["lines"][t[0]]))
        for i, maths, rest in order:
            x0, y0, x1, y1 = union([s["box"] for s in page["lines"][i]])
            pad = 6
            crop = img.crop((max(0, int((x0 - pad) * f)), max(0, int((y0 - pad) * f)),
                             min(img.width, int((x1 + pad) * f)), min(img.height, int((y1 + pad) * f))))
            ctx = CONTEXT.format(lines="\n".join(before[-KEEP_CONTEXT:])) if before and maths is not None else ""
            text = read_views(reader, crop, PROMPT_LINE.format(context=ctx), args.views if maths is not None else 1)
            if maths is None:                                   # (a text line: its math, when it states a value)
                m = text_line_math(text)
                n_text += 1
                before.append(text)
                if m:
                    n_text_math += 1
                    seg = text_segment(m, text, rest, (x0, y0, x1, y1))
                    if args.text_second == "none":
                        new_lines[i] = [seg] + page["lines"][i]
                    else:                                       # (Uni-MuMER's reading of it after all the pages)
                        found.append((pno, i, seg, crop))
                continue
            first, second = latexify(math_of(text)), second_reading(maths)
            n_lines += 1
            if not first or made_up(first, second, pieces_reading(maths)):   # (its pieces kept as they were read)
                n_made_up += 1
                new_lines[i] = page["lines"][i]
                before.append(pieces_reading(maths))
                continue
            before.append(text)
            seg = line_segment(text, maths, (x0, y0, x1, y1))
            n_agree += bool(first) and "math2" not in seg
            new_lines[i] = [seg] + rest
        page["lines"] = [new_lines.get(i, segs if not any(s["kind"] == "math" for s in segs) else
                                       [s for s in segs if s["kind"] != "math"])
                         for i, segs in enumerate(page["lines"])]
        print(f"page {pno}: {len(lines)} lines with math, {len(texts)} text lines, {time.time() - t1:.0f} s", flush=True)
    n_text_dropped = 0
    if found:                                                   # (the vision model off the GPU, then Uni-MuMER)
        import gc
        del reader
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        from recog.mumer import MumerReader
        kept, n_text_dropped = second_text_readings(found, MumerReader())
        for pno, i, seg in kept:
            data[pno - 1]["lines"][i] = [seg] + data[pno - 1]["lines"][i]
        n_text_math -= n_text_dropped
    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"DONE {n_lines} lines ({n_agree} read alike by both, {n_made_up} made up: their pieces kept)" +
          (f", {n_text} text lines ({n_text_math} stating a value: a math piece added; {n_text_dropped} more not held "
           f"by the second reader: left out)" if args.text_lines else "") +
          f" in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
