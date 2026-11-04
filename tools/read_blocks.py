"""Read each answer as one block (Tal's idea, 4.11): the whole handwritten answer of a sub-question - every line of
it, in order - read at once by a vision-language model, instead of piece by piece. Each printed page of this
booklet holds one sub-question and the space for its answer, so a page's handwriting (its print erased: the
booklet's template, recog/printed.py) is that answer; a back page written on goes with the page before it.

    python tools/read_blocks.py <exam.pdf> <out.json> --booklet booklet/moedC [--pages 5,7,9] [--model models/qwen3-vl-8b]

Tall answers are cut into strips of whole lines (at blank rows between them, at most about 14 lines a strip) so the
model reads each at a size it reads well; the strips' lines are put back in order. -> out.json: per page
{"page", "strips": [{"box", "text"}], "lines": [text lines]} - and with --as-reading a copy in tools/read_exams.py's
form (each line one math piece, Hebrew left out), which tools/recheck_exams.py judges unchanged."""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PROMPT_BLOCK = (
    "This image is a student's handwritten answer to one part of an exam in electronics (CMOS circuits, logic gates, "
    "delays, the body effect). It is written in Hebrew and math. Transcribe it line by line, top to bottom: one output "
    "line for each handwritten line, in the order written. Write the math in LaTeX without $ signs, Hebrew words as "
    "Hebrew text. Keep the student's numbers, units and mistakes exactly as written: do not solve, correct, complete "
    "or explain anything. Leave out what the student crossed out or scribbled over. Write [?] for a word or symbol "
    "you cannot read. Output only the transcription.")

STRIP_LINES = 14            # handwritten lines at most in one strip the model reads
HEADER, FOOTER = 0.07, 0.95  # page fractions: the printed header and footer bands - no answer there


def line_bands(ink, lh: float) -> list:
    """the rows of handwriting: [(y0, y1)] bands of rows with ink, gaps of less than a third of a letter bridged"""
    rows = ink.any(1)
    out, start, gap = [], None, 0
    for y, r in enumerate(rows):
        if r:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max(2, lh / 3):
                out.append((start, y - gap + 1))
                start, gap = None, 0
    if start is not None:
        out.append((start, len(rows)))
    return [(a, b) for a, b in out if b - a >= max(4, lh / 3) or ink[a:b].sum() > lh * lh]


def strips(bands: list, lh: float, max_lines: int = STRIP_LINES) -> list:
    """bands of handwriting grouped into strips of at most max_lines lines (a band taller than a line counts as
    several) -> [(y0, y1)]"""
    out, cur, n = [], None, 0
    for a, b in bands:
        k = max(1, round((b - a) / (1.6 * lh)))
        if cur is not None and n + k > max_lines:
            out.append(cur)
            cur, n = None, 0
        cur = (a, b) if cur is None else (cur[0], b)
        n += k
    if cur is not None:
        out.append(cur)
    return out


class BlockReader:
    def __init__(self, model_dir: str):
        """model_dir: a downloaded model, or the URL of one vLLM serves in the job (recog/remote_vlm.py)"""
        from recog.remote_vlm import RemoteReader, is_url
        self.remote = RemoteReader(model_dir) if is_url(model_dir) else None
        if self.remote:
            return
        import torch
        from recog.hebrew_vlm import HebrewRecognizer
        self.rec = HebrewRecognizer(model_dir=model_dir,
                                    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None)

    def read(self, img, max_new_tokens: int = 1500, prompt: str = PROMPT_BLOCK) -> str:
        """one strip -> its transcription, its lines kept (greedy: one reading); a line the model keeps writing again
        (a table's borders: '|' 750 times) cut after its third time in a row"""
        if self.remote:
            text = self.remote.read(img, max_new_tokens=max_new_tokens, prompt=prompt)
            text = re.sub(r"^```(?:latex|tex|text)?\s*|\s*```$", "", text.strip())
            return "\n".join(drop_runs([ln.rstrip() for ln in text.splitlines() if ln.strip()]))
        import torch
        from recog.hebrew_vlm import no_thinking
        rec = self.rec
        rec._unpark()
        msgs = [{"role": "user", "content": [{"type": "image", "image": img.convert("RGB")},
                                             {"type": "text", "text": prompt}]}]
        inputs = rec.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              enable_thinking=False, return_tensors="pt").to(rec.model.device)
        if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
            inputs["pixel_values"] = inputs["pixel_values"].to(rec.model.dtype)
        with torch.inference_mode():
            out = rec.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        text = no_thinking(rec.proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        text = re.sub(r"^```(?:latex|tex|text)?\s*|\s*```$", "", text.strip())
        return "\n".join(drop_runs([ln.rstrip() for ln in text.splitlines() if ln.strip()]))


def drop_runs(lines: list, keep: int = 3) -> list:
    """a line written again and again in a row (the model looping) kept `keep` times; a run of very short lines
    ('|', '-', one character each) longer than 6 - the model reading a drawing's strokes - left out"""
    out, run = [], 0
    for i, ln in enumerate(lines):
        run = run + 1 if out and ln == out[-1] else 1
        if run <= keep:
            out.append(ln)
    for period in (2, 3, 4):                       # (a few lines written again and again: kept once)
        i, kept = 0, []
        while i < len(out):
            unit = out[i:i + period]
            k = 1
            while len(unit) == period and out[i + k * period:i + (k + 1) * period] == unit:
                k += 1
            kept.extend(unit if k >= 3 else out[i:i + period * k])
            i += period * k
        out = kept
    res, i = [], 0
    while i < len(out):
        j = i
        while j < len(out) and len(out[j].strip()) <= 1:
            j += 1
        if j - i > 6:
            i = j
            continue
        res.extend(out[i:max(j, i + 1)])
        i = max(j, i + 1)
    return res


def answer_images(pdf: str, booklet: str, pages, dpi: int = 300, read_dpi: int = 150):
    """per page asked: (page, the handwriting alone at dpi - its print erased - cut into strips: [(box, image)])"""
    import numpy as np
    from PIL import Image
    from scipy import ndimage as ndi
    from read_page import booklet_pages, handwriting_only, load_pages
    from recog.page import clean_scan, letter_height
    book = booklet_pages(booklet)
    low = load_pages(pdf, read_dpi)
    high = load_pages(pdf, dpi)
    f = dpi / read_dpi
    for pno in pages:
        page, _ = clean_scan(low[pno - 1])
        hand = None
        if book and pno in book["pages"]:
            _, hand, _ = handwriting_only(page, book["pages"][pno])
        if hand is None:
            hand = np.asarray(page) < 128
        H, W = hand.shape
        hand = hand.copy()
        hand[: int(HEADER * H)] = False
        hand[int(FOOTER * H):] = False
        if hand.sum() < 200:
            yield pno, []
            continue
        lh = letter_height(hand) if hand.sum() > 2000 else 20.0
        keep = ndi.binary_dilation(hand, iterations=2)
        big = np.asarray(high[pno - 1].convert("L"))
        mask = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((big.shape[1], big.shape[0]))) > 0
        clean = np.where(mask, big, 255).astype(np.uint8)          # (the handwriting at 300 dpi, nothing else)
        cols = np.flatnonzero(hand.any(0))
        x0, x1 = max(0, cols[0] - int(lh)), min(W, cols[-1] + int(lh))
        out = []
        for a, b in strips(line_bands(hand, lh), lh):
            a, b = max(0, a - int(0.4 * lh)), min(H, b + int(0.4 * lh))
            box = (int(x0), int(a), int(x1), int(b))
            out.append((box, Image.fromarray(clean[int(a * f):int(b * f), int(x0 * f):int(x1 * f)])))
        yield pno, out


_HEB = re.compile(r"[\u0590-\u05FF][\u0590-\u05FF\"'׳״\-]*")


def tokens_for(box, lh: float = 20.0) -> int:
    """a strip's output budget: about 90 tokens a handwritten line it can hold (a looping model stops there)"""
    return int(120 + 90 * max(1, (box[3] - box[1]) / (1.6 * lh)))


_CMD_END = re.compile(r"\\[A-Za-z]+$")


def tidy_latex(t: str) -> str:
    """a vision model's LaTeX as the checks read it (Qwen3.6, 4.11): math delimiters anywhere ('$$a = 1$$ $$b = 2$$',
    '$x$', '\\[ \\]') a wide space between statements, not text; '\\vert' a bar, '\\qquad' a '\\quad'; a reading spaced
    token by token as handwriting datasets write it ('t _ { p H L } = 1 . 8') put together ('t_{pHL}=1.8' - a command
    keeps the space after it: '\\cdot x')"""
    t = re.sub(r"\$\$|\$|\\\[|\\\]|\\\(|\\\)", r" \\quad ", t)
    t = re.sub(r"\\vert(?![A-Za-z])", "|", t)
    t = re.sub(r"\\qquad(?![A-Za-z])|\\displaystyle(?![A-Za-z])", lambda m: r" \quad " if "qq" in m.group(0) else " ", t)
    toks = t.split()
    if len(toks) >= 6 and sum(1 for x in toks if len(x) == 1) > 0.6 * len(toks):
        out = ""
        for x in toks:
            if out and _CMD_END.search(out) and re.match(r"[A-Za-z]", x):
                out += " "
            out += x
        t = out.replace("\\quad", " \\quad ")
    t = re.sub(r"(?:\s*\\quad(?![A-Za-z])\s*)+", r" \\quad ", t)
    t = re.sub(r"^\s*\\quad\s*|\s*\\quad\s*$", "", t)
    return " ".join(t.split())


def math_of(line: str) -> str:
    """a transcribed line's math: Hebrew words (and \\text{...} with Hebrew) left out, what is left trimmed"""
    t = re.sub(r"\\text\s*\{[^{}]*[\u0590-\u05FF][^{}]*\}", " ", tidy_latex(line))
    t = _HEB.sub(" ", t)
    t = re.sub(r"\[\?\]", " ", t)
    t = re.sub(r"^(?:\s*\\quad(?![A-Za-z]))+|(?:\s*\\quad(?![A-Za-z]))+\s*$", "", " ".join(t.split()))
    t = t.strip(" ,.:;-")
    return t if re.search(r"[0-9A-Za-z\\]", t) else ""


def as_reading(blocks: list, n_pages: int) -> list:
    """the block readings in tools/read_exams.py's form: per page, each line one math piece (boxes: the strip's,
    split evenly among its lines), Hebrew left out - for tools/recheck_exams.py"""
    pages = [{"page": p, "info": {}, "figures": [], "lines": []} for p in range(1, n_pages + 1)]
    for b in blocks:
        rows = []
        for s in b["strips"]:
            lines = [ln for ln in s["text"].splitlines() if ln.strip()]
            x0, y0, x1, y1 = s["box"]
            step = (y1 - y0) / max(1, len(lines))
            for i, ln in enumerate(lines):
                m = math_of(ln)
                if m:
                    box = [x0, int(y0 + i * step), x1, int(y0 + (i + 1) * step)]
                    rows.append([{"kind": "math", "box": box, "p_kind": 1.0, "nbest": [[m, 0.0]], "math2": [[m, 0.0]],
                                  "block_line": ln}])
        pages[b["page"] - 1]["lines"] = rows
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--booklet", default=os.path.join(HERE, "booklet", "moedC"))
    ap.add_argument("--pages", default="", help="e.g. 5,7,9 (default: the odd pages 3-25 and their backs)")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--as-reading", default="", help="also a copy in read_exams.py's form (each line a math piece)")
    args = ap.parse_args()
    import pypdfium2 as pdfium
    n = len(pdfium.PdfDocument(args.pdf))
    pages = [int(p) for p in re.split(r"[,;:+\s]+", args.pages) if p.strip()] or list(range(3, min(n, 26) + 1))
    t0 = time.time()
    reader = BlockReader(args.model)
    print(f"model ready in {time.time() - t0:.0f} s", flush=True)
    blocks = []
    for pno, parts in answer_images(args.pdf, args.booklet, pages):
        t1 = time.time()
        got = [{"box": list(box), "text": reader.read(img, max_new_tokens=tokens_for(box))} for box, img in parts]
        lines = [ln for g in got for ln in g["text"].splitlines()]
        blocks.append({"page": pno, "strips": got, "lines": lines})
        print(f"page {pno}: {len(parts)} strips, {len(lines)} lines, {time.time() - t1:.0f} s", flush=True)
    json.dump(blocks, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if args.as_reading:
        json.dump(as_reading(blocks, n), open(args.as_reading, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"DONE {len(blocks)} pages in {time.time() - t0:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
