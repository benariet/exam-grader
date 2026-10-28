"""Which question each page of a student's scan answers - by the exam's own booklet: a student writes in the printed
booklet, so each scanned page is one of its pages (with handwriting on it), in order; the backs of the sheets, blank or
written on, follow their fronts. Each page is matched to the booklet's pages by its look at a low resolution (the printed
part), in order (a monotone alignment; a page like no booklet page - a cover, an extra sheet - is left out), then the
booklet's pages to questions.

    python tools/exam_questions.py "Moed C QST 1.pdf" "Stud exam\\Stud exam" --questions 2-6,7-9,10-12 --out grader_runs\\pages.json

-> {student: {page: {"booklet": k or null, "question": q or null, "blank": bool}}} - all on this computer, no text read"""
import argparse
import glob
import json
import os
import re

import numpy as np


def _looks(path: str, scale: float = 0.55):
    """each page of a PDF -> (its look: a small normalized grey image, its ink: the share of dark pixels)"""
    import cv2
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    out = []
    for i in range(len(pdf)):
        img = pdf[i].render(scale=scale, grayscale=True).to_numpy()
        if img.ndim == 3:
            img = img[..., 0]
        g = cv2.resize(img, (120, 170), interpolation=cv2.INTER_AREA).astype(np.float32)
        ink = float((img < 160).mean())
        out.append(((g - g.mean()) / (g.std() + 1e-6), ink))
    return out


def align(student: list, booklet: list, gap: float = 0.12, blank_ink: float = 0.003) -> list:
    """-> per student page the booklet page index (0-based) or None: the order-keeping matching of the most likeness,
    a page matched only when it looks like its booklet page more than `gap` (else left out); a blank page (almost no
    ink) is left out of the matching"""
    idx = [i for i, (_, ink) in enumerate(student) if ink >= blank_ink]
    n, m = len(idx), len(booklet)
    sc = np.array([[float((student[i][0] * b[0]).mean()) for b in booklet] for i in idx]) if n else np.zeros((0, m))
    best = np.zeros((n + 1, m + 1))
    move = np.zeros((n + 1, m + 1), dtype=int)                 # 0: page i left out, 1: booklet page j skipped, 2: matched
    for i in range(1, n + 1):
        best[i][0], move[i][0] = best[i - 1][0], 0
    for j in range(1, m + 1):
        best[0][j], move[0][j] = best[0][j - 1], 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(best[i - 1][j], 0), (best[i][j - 1], 1), (best[i - 1][j - 1] + sc[i - 1][j - 1] - gap, 2)]
            best[i][j], move[i][j] = max(opts)
    got = [None] * len(student)
    i, j = n, m
    while i > 0 and j > 0:
        mv = move[i][j]
        if mv == 2:
            got[idx[i - 1]] = j - 1
            i, j = i - 1, j - 1
        elif mv == 0:
            i -= 1
        else:
            j -= 1
    return got


def questions_of(spec: str) -> dict:
    """'2-6,7-9,10-12' -> {booklet page (1-based): question number}"""
    out = {}
    for q, part in enumerate(spec.split(","), 1):
        a, _, b = part.strip().partition("-")
        for k in range(int(a), int(b or a) + 1):
            out[k] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("booklet")
    ap.add_argument("folder")
    ap.add_argument("--questions", required=True, help="the booklet's pages of each question, e.g. 2-6,7-9,10-12")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    qmap = questions_of(args.questions)
    book = _looks(args.booklet)
    res = {}
    pdfs = [p for p in glob.glob(os.path.join(args.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)]
    for pdf in sorted(pdfs, key=lambda p: (int(re.sub(r"\D", "", os.path.basename(p)) or 0), p)):
        name = os.path.splitext(os.path.basename(pdf))[0]
        st = _looks(pdf)
        got = align(st, book)
        pages, last = {}, None
        for k, ((_, ink), b) in enumerate(zip(st, got), 1):
            if b is not None:
                last = b
            bk = b if b is not None else (last if ink >= 0.003 and k > 1 else None)   # (a back written on: its front's)
            pages[k] = {"booklet": None if bk is None else bk + 1, "question": qmap.get(bk + 1) if bk is not None else None,
                        "blank": ink < 0.003, "matched": b is not None}
        res[name] = pages
        n_q = {q: sum(1 for p in pages.values() if p["question"] == q) for q in sorted(set(qmap.values()))}
        print(name, "pages per question:", n_q, "unmatched written pages:",
              [k for k, p in pages.items() if not p["blank"] and not p["matched"]])
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
