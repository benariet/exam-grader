"""A printed truth table the students fill in (the course's Moed C 1(a): A, B, Vx, C, Y1, Y2 for the eight inputs),
read off the scans and checked against the solution's - on this computer (CPU only: the booklet's template, the cell
reader of the forms).

The table's cells are found on the booklet's template of that page (tools/build_booklet.py: its ruled lines), laid on
each scan (recog/printed.py's map), the print erased; each cell's first character - the logic value, before a
strength a student may add ('0S', '1W') - read by forms/cells.py's reader.

    python tools/truth_table.py "<exams' PDFs>" --booklet booklet/moedC --page 5 --cols 2,4,5 \\
        --key "000;010;110;101;110;101;001;011" --grades "grades_moed C.xlsx" --question 1 --out <folder>
        -> how many cells right / wrong / empty / unsure, beside the points the lecturer took off the question;
           <out>/<n>.table.json per exam and <n>_table_remarks.txt (Hebrew)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

SURE = 0.6              # a reading less likely than this: unsure (left to a human)
ALIASES = {"1": "17", "0": "06"}   # a cell of 0s and 1s: a '7' is a '1' written with a flag, a '6' a '0' with a tail
MIN_INK = 25            # handwriting pixels: fewer, an empty cell


def _lines(profile, need, min_gap=8):
    """the positions where profile >= need, runs merged -> [centre]"""
    idx = np.flatnonzero(profile >= need)
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= 2:
            out[-1].append(i)
        else:
            out.append([i])
    cs = [float(np.mean(r)) for r in out]
    merged = []
    for c in cs:
        if merged and c - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    return merged


def table_cells(template: np.ndarray, rows: int = 9, cols: int = 7):
    """the table's cells on the template (bool, ink True): its ruled lines - a horizontal line spanning most of the
    table, a vertical one most of its height -> {(row, col): (x0, y0, x1, y1)} (row 0: the header), or {} if no
    table of rows x cols is found"""
    from scipy import ndimage as ndi
    H, W = template.shape
    # (a template's ruled lines come out broken - the scans laid on each other a pixel apart: gaps closed first)
    horiz = ndi.binary_closing(template, structure=np.ones((1, 15)))
    vert = ndi.binary_closing(template, structure=np.ones((5, 1)))     # (not 15: a column's printed digits, one
    #                                                                      over the other, would join into a line)
    hs = _lines(horiz.mean(1), 0.45)
    for k in range(len(hs) - rows + 1):                # (rows + 1 lines)
        ys = hs[k:k + rows + 1]
        if len(ys) < rows + 1:
            break
        y0, y1 = int(ys[0]), int(ys[-1])
        band = vert[y0:y1 + 1]
        longest = np.zeros(W)                          # (a ruled line runs the table's height; a printed digit's
        for x in np.flatnonzero(band.mean(0) >= 0.3):  #  stroke, in each row, only its row's)
            d = np.diff(np.r_[0, band[:, x].astype(np.int8), 0])
            longest[x] = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max()
        vs = _lines(longest, 0.35 * band.shape[0])
        if len(vs) == cols + 1:
            return {(i, j): (int(vs[j]), int(ys[i]), int(vs[j + 1]), int(ys[i + 1]))
                    for i in range(rows) for j in range(cols)}
    return {}


def to_scan(box, M, b):
    """a box in the template's frame -> the scan's (recog/printed.py's map: p_ref = M p_img + b, points (row, col))"""
    Mi = np.linalg.inv(M)
    x0, y0, x1, y1 = box
    pts = [Mi @ (np.array([y, x]) - b) for y in (y0, y1) for x in (x0, x1)]
    ys, xs = [p[0] for p in pts], [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def char_groups(hand: np.ndarray):
    """the cell's handwriting (bool) -> its characters' boxes (x0, y0, x1, y1), left to right: pieces one over the
    other are one character; specks and a ruled line's remnant (as long as the cell) left out"""
    from scipy import ndimage as ndi
    lab, n = ndi.label(hand, structure=np.ones((3, 3)))
    if not n:
        return []
    H, W = hand.shape
    boxes = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        size = int((lab[sl] == k + 1).sum())
        w, h = sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
        if size < 6 or (w >= 0.8 * W and h <= 4) or (h >= 0.9 * H and w <= 4):
            continue
        boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop])
    boxes.sort()
    groups = []
    for b in boxes:
        if groups:
            g = groups[-1]
            ov = min(g[2], b[2]) - max(g[0], b[0])
            if ov >= 0.5 * min(b[2] - b[0], g[2] - g[0]):
                groups[-1] = [min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])]
                continue
        groups.append(b)
    hmax = max((g[3] - g[1] for g in groups), default=0)
    groups = [g for g in groups if g[3] - g[1] >= 0.35 * hmax]           # (a dot, a comma: no character)
    narrow = lambda g: g[2] - g[0] <= 0.45 * (g[3] - g[1])              # noqa: E731
    if len(groups) >= 3 and narrow(groups[0]) and narrow(groups[-1]):    # ('(1)': the brackets, no '1's)
        groups = groups[1:-1]
    return [tuple(g) for g in groups]


def read_cells(page, ink, template, cells, want, reader, alphabet="01"):
    """the scan's cells in want [(row, col)] -> {(row, col): (reading or '' (empty) or '?' (none of the alphabet),
    how sure, ink pixels)}: the cell's first character that reads as one of the alphabet's (the logic value - before a
    strength, '1W', or after a parenthesis), as the likelier of them"""
    from recog.printed import fit_map, printed_on
    M, b = fit_map(template, ink)
    printed = printed_on(template, ink)
    hand = ink & ~printed
    gray = np.asarray(page.convert("L"))
    out, crops, owners = {}, [], []
    for key in want:
        x0, y0, x1, y1 = to_scan(cells[key], M, b)
        m = 4                                          # (inside the ruled lines)
        sub = hand[y0 + m:y1 - m, x0 + m:x1 - m]
        n_ink = int(sub.sum())
        groups = char_groups(sub) if n_ink >= MIN_INK else []
        if not groups:
            out[key] = ("", 1.0, n_ink)
            continue
        g = gray[y0 + m:y1 - m, x0 + m:x1 - m]
        for k, (a, c, d, e) in enumerate(groups[:4]):
            s = max(d - a, e - c)
            pad = int(0.25 * s) + 2
            crop = np.full((s + 2 * pad, s + 2 * pad), 255, np.uint8)
            piece = np.where(sub[c:e, a:d], g[c:e, a:d], 255).astype(np.uint8)   # (the character alone, print gone)
            oy, ox = pad + (s - (e - c)) // 2, pad + (s - (d - a)) // 2
            crop[oy:oy + (e - c), ox:ox + (d - a)] = piece
            crops.append(crop)
            owners.append((key, k, n_ink))
    if crops:
        probs = reader.probs(crops)
        # (each value with the digits a hand writes it like: a '1' with a flag reads '7', a '0' with a tail '6')
        idx = [[reader.classes.index(x) for x in ALIASES.get(ch, ch)] for ch in alphabet]
        own = [reader.classes.index(ch) for ch in alphabet]
        best = {}
        for (key, k, n_ink), p in zip(owners, probs):
            q = np.array([p[ix].sum() for ix in idx])
            if key in best and best[key][3] >= 0.5:
                continue                               # (a character of the alphabet before: that one)
            mass = float(q.sum())                      # (how much it looks like one of them at all)
            j = int(np.argmax(q))
            sure = float(q[j] / mass) if mass > 0 else 0.0
            v = alphabet[j] if mass >= 0.5 else "?"
            if v != "?" and p[own[j]] < 0.5 * q[j]:
                v += "~"                               # (read through a look-alike: '7' for a '1' - no sure wrong)
            if key not in best or mass > best[key][3]:
                best[key] = (v, sure * min(1.0, mass / 0.5), n_ink, mass)
        for key, (v, sure, n_ink, _) in best.items():
            out[key] = (v, sure, n_ink)
    return out


def _review_figure(img, boxes, cells_out, name, res, off):
    """one exam's table for review.html: the scan round the cells read, each boxed in its verdict's colour"""
    import base64
    import html
    import io
    from PIL import ImageDraw
    colors = {"right": (20, 160, 60), "wrong": (220, 30, 30), "unsure": (240, 140, 0), "empty": (150, 150, 150)}
    x0 = min(b[0] for b in boxes.values()) - 60
    y0 = min(b[1] for b in boxes.values()) - 60
    x1 = max(b[2] for b in boxes.values()) + 20
    y1 = max(b[3] for b in boxes.values()) + 20
    crop = img.crop((max(0, x0), max(0, y0), x1, y1)).copy()
    d = ImageDraw.Draw(crop)
    for c in cells_out:
        bx = boxes[(c["row"], c["col"])]
        col = colors[c["verdict"]]
        r = (bx[0] - max(0, x0), bx[1] - max(0, y0), bx[2] - max(0, x0), bx[3] - max(0, y0))
        d.rectangle(r, outline=col, width=3)
        d.text((r[2] - 14, r[1] + 2), c["read"] or "-", fill=col)
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    cap = f"מבחן {name}: {res['right']} נכונים, {res['wrong']} שגויים, {res['unsure']} לא בטוחים, {res['empty']} ריקים" + \
        (f" (המרצה הוריד בשאלה כולה {off:g})" if off is not None else "")
    return (f"<figure><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
            f"<figcaption>{html.escape(cap)}</figcaption></figure>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--booklet", required=True)
    ap.add_argument("--page", type=int, required=True, help="the scan's page with the table (the booklet's numbering)")
    ap.add_argument("--cols", default="2,4,5", help="the columns to read (0 = the leftmost)")
    ap.add_argument("--key", required=True, help="the solution's values, a row per ';' in the table's order, a "
                                                 "character per column read: '000;010;...'")
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--ncols", type=int, default=7)
    ap.add_argument("--grades", default="")
    ap.add_argument("--question", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--review", action="store_true", help="<out>/review.html: each exam's table as scanned, each cell "
                    "read boxed - green right, red wrong, orange unsure, gray empty - with its reading (to look over)")
    a = ap.parse_args()
    from PIL import Image
    from forms.cells import CellReader
    from recog.page import clean_scan
    template = np.asarray(Image.open(os.path.join(a.booklet, f"p{a.page}.png")).convert("L")) < 128
    cells = table_cells(template, a.rows, a.ncols)
    if not cells:
        raise SystemExit(f"no {a.rows} x {a.ncols} table found on {a.booklet}/p{a.page}.png")
    cols = [int(c) for c in a.cols.split(",")]
    key = [list(r) for r in a.key.split(";")]
    want = [(r + 1, c) for r in range(len(key)) for c in cols]
    reader = CellReader()
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    os.makedirs(a.out, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    pdfs = sorted((p for p in glob.glob(os.path.join(a.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    table, review = [], []
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        if only and name not in only:
            continue
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf)
        if len(doc) < a.page:
            continue
        img = doc[a.page - 1].render(scale=150 / 72).to_pil().convert("RGB")
        page, _ = clean_scan(img)
        ink = np.asarray(page.convert("L")) < 128
        got = read_cells(page, ink, template, cells, want, reader)
        if a.review:
            from recog.printed import fit_map
            M, b = fit_map(template, ink)
            boxes = {k: to_scan(cells[k], M, b) for k in want}
        res = {"right": 0, "wrong": 0, "empty": 0, "unsure": 0}
        cells_out, remarks = [], []
        for r, row in enumerate(key):
            for c, want_v in zip(cols, row):
                v, p, n_ink = got[(r + 1, c)]
                alike = v.endswith("~")                # (read through a look-alike digit: right when as the key,
                v = v.rstrip("~")                      #  else left to a human)
                verdict = "empty" if v == "" else "unsure" if p < SURE or v not in ("0", "1") else \
                    "right" if v == want_v else "unsure" if alike else "wrong"
                res[verdict] += 1
                cells_out.append({"row": r + 1, "col": c, "read": v, "p": round(p, 3), "verdict": verdict, "key": want_v})
                if verdict == "wrong":
                    remarks.append(f"שורה {r + 1}, עמודה {c + 1}: נכתב {v}, בפתרון {want_v}.")
        json.dump({"exam": name, "cells": cells_out, **res}, open(os.path.join(a.out, f"{name}.table.json"), "w",
                                                                    encoding="utf-8"), ensure_ascii=False, indent=1)
        txt = [f"שאלה {a.question}(א) - טבלת האמת: {res['right']} תאים נכונים, {res['wrong']} שגויים, "
               f"{res['empty']} ריקים, {res['unsure']} לא נקראו בוודאות (לבדיקה ידנית)."] + remarks
        open(os.path.join(a.out, f"{name}_table_remarks.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
        off = marks.get(name, {}).get(a.question) if marks else None
        table.append((res, off))
        if a.review:
            review.append(_review_figure(img, boxes, cells_out, name, res, off))
    if a.review:
        page_html = ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Truth table review</title>",
                     "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}"
                     "figcaption{font-size:14px}</style>",
                     "<h1>טבלת האמת - מה נקרא</h1><p>ירוק: נכון לפי הפתרון. אדום: שגוי. כתום: לא נקרא בוודאות. "
                     "אפור: ריק. בכל תא כתוב מה נקרא.</p>"] + review + ["</html>"]
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(page_html))
    n = len(table)
    tot = {k: sum(r[k] for r, _ in table) for k in ("right", "wrong", "empty", "unsure")}
    print(f"{n} exams, {sum(tot.values())} cells: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
    if marks:
        for lo, hi, what in ((0, 0, "no cell wrong"), (1, 3, "1-3 cells wrong"), (4, 99, "4+ cells wrong")):
            offs = [o for r, o in table if lo <= r["wrong"] <= hi and o is not None]
            if offs:
                print(f"  {what}: {len(offs)} exams, mean points off question {a.question} {sum(offs) / len(offs):.1f}")


if __name__ == "__main__":
    main()
