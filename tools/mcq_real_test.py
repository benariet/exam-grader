"""Real student handwriting, read with no form: the 50 scanned Data Science exams of
"A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations" (Mendeley Data
sf3kvjwknt, CC BY 4.0). Every student wrote the multiple-choice answers as a list - '1. C', '2. A', ...
The teacher's table (Teacher_manual_marks_Anonymized.csv) holds the letter each student chose: the truth.

The test: find the lines on every page, read each line with the English handwriting reader, keep the
readings that look like '<question 1-20> <letter A-D>', and compare with the teacher's table.

    python tools/mcq_real_test.py --data /home/claude/dl/mendeley_exams --model models/trocr-small-handwritten
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PAT = re.compile(r"^\W*(\d{1,2})\s*[\.,:;\)\-]?\s*\(?([A-Da-d])\)?\W*$")
# the same answer format as a grammar for the reader: it may only read '<1-2 digits><punct.?> <letter A-D>'
GRAMMAR = (r"\s*[1-9][0-9]?\s*[.,:)\-]?\s*[A-Da-d]\s*[.]?\s*", " 0123456789.,:)-ABCDabcd")


def parse(text):
    m = PAT.match(text.strip())
    if not m:
        return None
    q = int(m.group(1))
    return (q, m.group(2).upper()) if 1 <= q <= 20 else None


def render(path, width):
    """the scans are stored at an odd page size (3720 px wide at 150 dpi): render every page to `width`"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    return [pdf[i].render(scale=width / pdf[i].get_width()).to_pil().convert("RGB") for i in range(len(pdf))]


def short_chunks(line_img, page_w, max_share=0.4, col_gap=3.0):
    """the written pieces of a full-width line image that are short ('N. X', not a sentence): ink runs
    split at gaps of more than col_gap line heights (answers written in two columns); a tall thin mark
    (the page's margin rule) is dropped"""
    import numpy as np
    g = np.asarray(line_img.convert("L"))
    ink = g < 160
    cols = np.where(ink.sum(0) > 0)[0]
    if len(cols) == 0:
        return []
    h = ink.shape[0]
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > 0.6 * h:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    runs = [r for r in runs if not (r[1] - r[0] < 0.25 * h and ink[:, r[0]:r[1] + 1].any(1).mean() > 0.8)]
    chunks, cur = [], []
    for r in runs:                                   # group runs closer than col_gap line heights
        if cur and r[0] - cur[-1][1] > col_gap * h:
            chunks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        chunks.append(cur)
    out = []
    for ch in chunks:
        x0, x1 = ch[0][0], ch[-1][1]
        if x1 - x0 > max_share * page_w or x1 - x0 < 0.3 * h:
            continue
        rows = np.where(ink[:, x0:x1 + 1].sum(1) > 0)[0]
        y0, y1 = rows[0], rows[-1]
        pad = max(4, h // 6)
        out.append(line_img.crop((max(0, x0 - pad), max(0, y0 - pad), min(g.shape[1], x1 + pad + 1), min(h, y1 + pad + 1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/claude/dl/mendeley_exams")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "trocr-small-handwritten"))
    ap.add_argument("--tag", default="small")
    ap.add_argument("--students", type=int, default=50)
    ap.add_argument("--only", default="", help="comma-separated student numbers, e.g. 4,5")
    ap.add_argument("--width", type=int, default=1240, help="page width in pixels (A4 at 150 dpi)")
    ap.add_argument("--pages", type=int, default=99, help="read the first N pages of each exam")
    ap.add_argument("--grammar", action="store_true", help="read with the answer format as a grammar")
    ap.add_argument("--min-conf", type=float, default=0.0, help="ignore readings less sure than this")
    a = ap.parse_args()

    from recog.english import EnglishRecognizer
    from recog.page import clean_scan, find_lines
    rec = EnglishRecognizer(a.model)
    truth = {r["Student_ID"]: r for r in csv.DictReader(open(os.path.join(a.data, "Teacher_manual_marks_Anonymized.csv"),
                                                               encoding="utf-8-sig"))}
    out_rows, t0 = [], time.time()
    names = sorted(os.listdir(os.path.join(a.data, "Student_Pdf")), key=lambda n: int(re.sub(r"\D", "", n) or 0))
    if a.only:
        names = [f"Student_{n}.pdf" for n in a.only.split(",")]
    for k, name in enumerate(names[: a.students]):
        sid = name[:-4]
        if sid not in truth:
            continue
        found = {}
        for pno, img in enumerate(render(os.path.join(a.data, "Student_Pdf", name), a.width)[: a.pages]):
            page, info = clean_scan(img)
            for y0, _, line_img in find_lines(page):
                for crop in short_chunks(line_img, page.width):  # the answers are short pieces of lines
                    c = rec.recognize(crop, n_best=3, max_new_tokens=12, grammar=GRAMMAR if a.grammar else None)
                    if not c:
                        continue
                    text = c[0].text
                    p = parse(text)
                    conf = math.exp(c[0].logp / max(1, len(text)))
                    if p and conf >= a.min_conf and (p[0] not in found or conf > found[p[0]]["conf"]):
                        found[p[0]] = {"letter": p[1], "text": text, "conf": conf, "page": pno + 1, "y": y0,
                                       "alt": [x.text for x in c[1:]]}
        t = truth[sid]
        for q in range(1, 21):
            want = (t[str(q)] or "").strip().upper()
            got = found.get(q)
            out_rows.append({"student": sid, "q": q, "truth": want, "read": got["letter"] if got else "",
                             "text": got["text"] if got else "", "conf": got["conf"] if got else 0.0})
        done = [r for r in out_rows if r["student"] == sid]
        print(f"{sid}: found {sum(1 for r in done if r['read'])}/20, right {sum(1 for r in done if r['read'] == r['truth'])}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    n = len(out_rows)
    found = [r for r in out_rows if r["read"]]
    right = [r for r in found if r["read"] == r["truth"]]
    by = sorted(found, key=lambda r: -r["conf"])
    summ = {"reader": a.model, "students": len({r['student'] for r in out_rows}), "answers": n,
            "found": len(found) / max(1, n), "right_of_all": len(right) / max(1, n),
            "right_of_found": len(right) / max(1, len(found)), "secs": time.time() - t0}
    for share in (0.25, 0.5, 0.75):
        top = by[: int(len(by) * share)]
        summ[f"accuracy_most_confident_{int(share * 100)}%"] = sum(r["read"] == r["truth"] for r in top) / max(1, len(top))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"summary": summ, "rows": out_rows}, open(os.path.join(HERE, "results", f"mcq_real_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
