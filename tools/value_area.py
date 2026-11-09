"""A final value students write in a printed place - the course's Moed C 3: 'DUTY CICLE = ...' at the bottom of the
waveforms' page - read off the scans and checked against the solution's value, on this computer (the formula reader
recog/formula.py; the GPU readers' readings of the cut place, tools/read_crops.py, judged with it when given).

The place is given once per booklet, in the template's pixels (--area); the handwriting there (the print left out,
writing running out of it kept whole) is cut out and read; the value is the last number of the reading
(grader.numeric.parse_numeric: '\\frac{11.5}{200} \\cdot 100\\% = 5.75\\%' -> 5.75 %), a fraction with no unit read as
a share ('0.0575' for 5.75 % with --percent), against the key within --tol.

    python tools/value_area.py "<exams' PDFs>" --booklet booklet/moedC --page 25 --area 80,1500,1100,1612 \\
        --key "5.75%" --percent --name "3 duty cycle" --question 3 --out <folder> [--grades ...] [--review]
        [--crops-only <dir>] [--second readings.json]
        -> right / wrong / unsure / empty per exam; <out>/<n>.value.json, <n>_value_remarks.txt (Hebrew)
"""
import argparse
import glob
import html
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

SURE_P = 0.5


def value_of(text, percent=False):
    """a reading -> its final value (float) or None; with percent, a bare fraction below 1 is a share (x 100)"""
    from grader.numeric import parse_numeric
    try:
        v = parse_numeric(text)
    except Exception:                                  # noqa: BLE001  (not a number)
        return None
    if percent and not (v.unit or "").strip() and 0 < abs(v.value) < 1:
        return v.value * 100
    return float(v.value)


def judge_value(readings, key, tol, percent=False):
    """{reader: [(text, min token p)]} -> (verdict, the reading used, its value): the vision model asked for the final
    value of the place ('vlmfinal') giving the key's - right (it could not have worked it out); two readers whose
    likeliest readings that hold a number agree (within 1 %) decide; else the formula reader's own - right if its
    likeliest number is the key's, wrong if that reading is sure and no other reading is the key's, else unsure"""
    close = lambda v, w, t: v is not None and w is not None and abs(v - w) <= t * max(abs(w), 1e-12)  # noqa: E731
    family = lambda n: "vlm" if n.startswith("vlm") else n                                             # noqa: E731
    tops = {}
    for rd, cands in readings.items():
        for text, p in cands:
            v = value_of(text, percent)
            if v is not None:
                tops[rd] = (text, v, p)
                break
    if "vlmfinal" in tops and close(tops["vlmfinal"][1], key, tol):
        return "right", tops["vlmfinal"][0], tops["vlmfinal"][1]   # (the vision model told only what the place is:
    names = list(tops)                                             #  it cannot work the key out - it read it)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if family(a) == family(b):                 # (the vision model twice - on the cut writing, asked for the
                continue                               #  final value - is one reader)
            if close(tops[a][1], tops[b][1], 0.01):
                return ("right" if close(tops[a][1], key, tol) else "wrong"), tops[a][0], tops[a][1]
    own = readings.get("pix2text", [])
    vals = [value_of(t, percent) for t, _ in own]
    first = next(((t, v, p) for (t, p), v in zip(own, vals) if v is not None), None)
    if first is None:                                  # (the final value the vision model was asked for, if any:
        if "vlmfinal" in tops:                         #  for the human to see, not a verdict)
            return "unsure", tops["vlmfinal"][0], tops["vlmfinal"][1]
        return "unsure", own[0][0] if own else "", None
    if close(first[1], key, tol):
        return "right", first[0], first[1]
    others_right = any(close(v, key, tol) for v in vals) or any(close(x[1], key, tol) for x in tops.values())
    if first[2] >= SURE_P and own and first[0] == own[0][0] and not others_right:
        return "wrong", first[0], first[1]
    return "unsure", first[0], first[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--booklet", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--area", required=True, help="x0,y0,x1,y1 on the booklet's template of the page")
    ap.add_argument("--key", required=True, help="the solution's value, a unit after it if any: '5.75%%'")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--percent", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--question", type=int, default=0)
    ap.add_argument("--grades", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--crops-only", default="")
    ap.add_argument("--second", default="")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    import pypdfium2 as pdfium
    from PIL import Image
    from scipy import ndimage as ndi
    from recog.page import clean_scan
    from recog.printed import fit_map, printed_on
    from tools.expr_box import cleaned_hires, crop_of, soft_hand
    from tools.truth_table import to_scan
    template = np.asarray(Image.open(os.path.join(a.booklet, f"p{a.page}.png")).convert("L")) < 128
    area = tuple(int(v) for v in a.area.split(","))
    key = value_of(a.key, a.percent)
    if key is None:
        raise SystemExit(f"the key {a.key!r} is no number")
    second = json.load(open(a.second, encoding="utf-8")) if a.second else {}
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    for d in (a.crops_only, a.out):
        if d:
            os.makedirs(d, exist_ok=True)
    fr = None
    pdfs = sorted((p for p in glob.glob(os.path.join(a.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    table, review = [], []
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        doc = pdfium.PdfDocument(pdf)
        if len(doc) < a.page:
            continue
        img = doc[a.page - 1].render(scale=150 / 72).to_pil().convert("RGB")
        page, _ = clean_scan(img)
        gray = np.asarray(page.convert("L"))
        ink = gray < 128
        M, b = fit_map(template, ink)
        hand = soft_hand(gray, printed_on(template, ink))
        x0, y0, x1, y1 = to_scan(area, M, b)
        lab, n = ndi.label(ndi.binary_dilation(hand, iterations=1), structure=np.ones((3, 3)))
        lab[~hand] = 0
        keep = []
        for k, sl in enumerate(ndi.find_objects(lab)):
            if sl is None:
                continue
            cy, cx = (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1 and int((lab[sl] == k + 1).sum()) >= 8:
                keep.append(k + 1)
        mask = np.isin(lab, keep) if keep else None
        if mask is not None and int(mask.sum()) < 30:
            mask = None
        crop = crop_of(gray, mask, hi=cleaned_hires(doc, a.page, 300)) if mask is not None else None
        if a.crops_only:
            if crop is not None:
                crop.save(os.path.join(a.crops_only, f"{name}_value.png"))
                hi = cleaned_hires(doc, a.page, 300)       # (the place with its print, for a reader told what it is)
                k = hi.shape[1] / gray.shape[1]
                Image.fromarray(hi[max(0, int((y0 - 15) * k)):int((y1 + 15) * k), max(0, int((x0 - 15) * k)):int((x1 + 15) * k)]
                                ).save(os.path.join(a.crops_only, f"{name}_valuebox.png"))
            continue
        if crop is None:
            v, used, val = "empty", "", None
        else:
            if fr is None:
                from recog.formula import FormulaRecognizer
                fr = FormulaRecognizer()
            readings = {"pix2text": [(c.text, c.min_token_p) for c in fr.recognize(crop, n_best=4)]}
            for rd, cs in second.get(f"{name}_value", {}).items():
                readings[rd] = [(t, 1.0) for t, _ in cs]
            v, used, val = judge_value(readings, key, a.tol, a.percent)
        out = {"exam": name, "verdict": v, "read": used, "value": val, "key": key}
        json.dump(out, open(os.path.join(a.out, f"{name}.value.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        txt = {"right": f"{a.name}: נכתב {val:g} - כמו בפתרון ({a.key})." if val is not None else "",
               "wrong": f"{a.name}: נכתב {val:g}; בפתרון {a.key}." if val is not None else "",
               "unsure": f"{a.name}: לא נקרא בוודאות - לבדיקה ידנית" +
                         (f" (קריאה אפשרית: {val:g}; בפתרון {a.key})." if val is not None else "."),
               "empty": f"{a.name}: לא נמצאה תשובה במקום המיועד."}[v]
        open(os.path.join(a.out, f"{name}_value_remarks.txt"), "w", encoding="utf-8").write(txt + "\n")
        table.append((v, marks.get(name, {}).get(a.question) if marks and a.question else None))
        if a.review:
            import base64
            import io
            q = img.crop((max(0, x0 - 20), max(0, y0 - 20), min(img.width, x1 + 20), min(img.height, y1 + 20)))
            buf = io.BytesIO()
            q.save(buf, "PNG")
            review.append(f"<figure><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
                          f"<figcaption>{html.escape(f'מבחן {name}: {txt}')}</figcaption></figure>")
    if a.review and a.out:
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(
            ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Values review</title>",
             "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}</style>",
             f"<h1>{html.escape(a.name)} - מה נקרא</h1>"] + review + ["</html>"]))
    if table:
        from collections import Counter
        c = Counter(v for v, _ in table)
        print(f"{a.name}: {len(table)} exams - " + ", ".join(f"{k} {c[k]}" for k in ("right", "wrong", "unsure", "empty")))
        if marks and a.question:
            for k in ("right", "wrong", "unsure", "empty"):
                offs = [o for v, o in table if v == k and o is not None]
                if offs:
                    print(f"  {k}: mean points off question {a.question} {sum(offs) / len(offs):.1f} (n={len(offs)})")


if __name__ == "__main__":
    main()
