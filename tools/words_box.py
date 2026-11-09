"""Words students write in a printed answer box - 'Function - / Y1 - / Y2 -', the course's Moed C 1(b): which known
component the circuit is (a full adder) and each output's role (the sum, the carry) - checked against the solution's
words, on this computer; the reading itself by the GPU readers on the server (only the cut rows go there).

The box and its rows are found as tools/expr_box.py finds them (a ruled rectangle, a printed label starting each row);
each row's writing cut out (and the whole box, its labels with it):

    python tools/words_box.py "<exams' PDFs>" --booklet booklet/moedC --page 7 --rows Function,Y1,Y2 --crops-only <dir>
    (server)  python tools/read_crops.py <dir> --out readings.json --readers vlm --prompt words --labels Function,Y1,Y2
    python tools/words_box.py "<exams' PDFs>" --booklet booklet/moedC --page 7 --rows Function,Y1,Y2 \\
        --readings readings.json --key key.json --out <folder> [--grades ... --question 1] [--review]
        key.json: {"Function": {"right": ["full\\\\s*adder", "מחבר\\\\s*מלא"], "right_fuzzy": ["full adder"],
                                "wrong": ["half\\\\s*adder"]}, ...}   (right_fuzzy: words a letter off still count)
        -> per row right (a reading names the solution's), wrong (readings name another, none the solution's), unsure;
           <out>/<n>.words.json and <n>_words_remarks.txt (Hebrew)
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


def _dl(a: str, b: str) -> int:
    """the Damerau-Levenshtein distance (a swap of two letters one edit)"""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def fuzzy_hit(text: str, phrase: str) -> bool:
    """the phrase's words, one after another, in the text, each a letter off at most (words of 4 letters or more -
    'curry', 'caryy' for 'carry'; a shorter one as written)"""
    toks = [t.lower() for t in re.findall(r"[A-Za-z\u0590-\u05ff]+", text)]
    words = phrase.lower().split()
    ok = lambda t, w: t == w or (len(w) >= 4 and _dl(t, w) <= 1)       # noqa: E731
    return any(all(ok(toks[i + k], w) for k, w in enumerate(words)) for i in range(len(toks) - len(words) + 1))


def judge_words(texts, key):
    """the readers' readings of a row [str] and the row's key {'right': [regex], 'wrong': [regex]} -> (verdict, the
    reading that decided): right when a reading names the solution's (and none names a wrong one alone), wrong when
    readings name a wrong one and none the solution's, else unsure"""
    rx = lambda pats: [re.compile(p, re.I) for p in pats]          # noqa: E731
    good, bad = rx(key.get("right", [])), rx(key.get("wrong", []))
    hits_good = [t for t in texts if t and (any(p.search(t) for p in good) or
                                         any(fuzzy_hit(t, f) for f in key.get("right_fuzzy", [])))]   # (misspelt or
    #                                                                                       misread a letter: still it)
    hits_bad = [t for t in texts if t and any(p.search(t) for p in bad) and not any(p.search(t) for p in good)]
    if hits_good and not hits_bad:
        return "right", hits_good[0]
    if hits_bad and not hits_good:
        return "wrong", hits_bad[0]
    return "unsure", (hits_good or hits_bad or [t for t in texts if t] or [""])[0]


def judge_row(got, key):
    """the readers' readings of a row {reader: [[text, logp], ...]} -> (verdict, reading): the row's own reading
    decides ('vlm': the row cut out, nothing round it to guess from), the whole box's ('vlmbox': the labels with it - a
    model may write the likely answer where it cannot read) may only agree; without the row's, the box's verdict is left
    unsure"""
    row = [cs[0][0] for r, cs in got.items() if cs and not r.endswith("box")]
    box = [cs[0][0] for r, cs in got.items() if cs and r.endswith("box")]
    v_row, t_row = judge_words(row, key) if row else ("unsure", "")
    v_box, t_box = judge_words(box, key) if box else ("unsure", "")
    if v_row in ("right", "wrong") and v_box in (v_row, "unsure"):
        return v_row, t_row
    return "unsure", t_row or t_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--booklet", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--rows", required=True, help="the rows' labels, top to bottom: 'Function,Y1,Y2'")
    ap.add_argument("--crops-only", default="")
    ap.add_argument("--readings", default="")
    ap.add_argument("--key", default="")
    ap.add_argument("--grades", default="")
    ap.add_argument("--question", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    import pypdfium2 as pdfium
    from PIL import Image
    from recog.page import clean_scan
    from recog.printed import fit_map, printed_on
    from tools.expr_box import answer_rows, blank_answers, box_view, cleaned_hires, crop_of, row_pieces, soft_hand
    from tools.truth_table import to_scan
    names = [x.strip() for x in a.rows.split(",") if x.strip()]
    template = np.asarray(Image.open(os.path.join(a.booklet, f"p{a.page}.png")).convert("L")) < 128
    rows = answer_rows(template, len(names))
    if len(rows) != len(names):
        raise SystemExit(f"no answer box of {len(names)} rows on {a.booklet}/p{a.page}.png ({len(rows)} rows found)")
    blank = blank_answers(template, rows)
    key = json.load(open(a.key, encoding="utf-8")) if a.key else {}
    readings = json.load(open(a.readings, encoding="utf-8")) if a.readings else {}
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    for d in (a.crops_only, a.out):
        if d:
            os.makedirs(d, exist_ok=True)
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
        rs = []
        for r in rows:
            q = dict(r)
            q["box"] = list(to_scan(r["box"], M, b))
            q["x0"], q["y0"], q["x1"], q["y1"] = to_scan((r["x0"], r["y0"], r["x1"], r["y1"]), M, b)
            rs.append(q)
        masks = row_pieces(soft_hand(gray, printed_on(blank, ink)), rs)
        if a.crops_only:
            hi = cleaned_hires(doc, a.page, 300)
            for nm, mk in zip(names, masks):
                if mk is not None:
                    crop_of(gray, mk, hi=hi).save(os.path.join(a.crops_only, f"{name}_{nm}.png"))
            box_view(hi, gray.shape[1], rs).save(os.path.join(a.crops_only, f"{name}_box.png"))
            continue
        res = {"right": 0, "wrong": 0, "unsure": 0, "empty": 0}
        out, remarks = [], []
        for nm, mk in zip(names, masks):
            got = readings.get(f"{name}_{nm}", {})
            texts = [cs[0][0] for cs in got.values() if cs]
            if mk is None and not texts:
                v, used = "empty", ""
            else:
                v, used = judge_row(got, key.get(nm, {}))
            res[v] += 1
            out.append({"name": nm, "verdict": v, "read": used, "readers": {k: cs[0][0] for k, cs in got.items() if cs}})
            remarks.append({"right": f"{nm}: נכון ({used}).", "wrong": f"{nm}: נכתב '{used}' - שונה מהפתרון.",
                            "unsure": f"{nm}: לא נקרא בוודאות - לבדיקה ידנית.", "empty": f"{nm}: ריק."}[v])
        json.dump({"exam": name, "rows": out, **res}, open(os.path.join(a.out, f"{name}.words.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        open(os.path.join(a.out, f"{name}_words_remarks.txt"), "w", encoding="utf-8").write(
            f"שאלה {a.question}(ב):\n" + "\n".join(remarks) + "\n")
        off = marks.get(name, {}).get(a.question) if marks else None
        table.append((res, off, out))
        if a.review:
            from tools.expr_box import _review_figure
            review.append(_review_figure(img, rs, masks, out, name, off))
    if a.review and a.out:
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(
            ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Words review</title>",
             "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}"
             "figcaption{font-size:14px;white-space:pre-wrap}</style>",
             "<h1>תשובות במילים - מה נקרא</h1><p>ירוק: נכון. אדום: שגוי. כתום: לא נקרא בוודאות.</p>"]
            + review + ["</html>"]))
    if table:
        from collections import Counter
        tot = {k: sum(r[k] for r, _, _ in table) for k in ("right", "wrong", "unsure", "empty")}
        print(f"{len(table)} exams, {sum(tot.values())} rows: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
        for nm in names:
            c = Counter(o["verdict"] for _, _, out in table for o in out if o["name"] == nm)
            print(f"  {nm}: " + ", ".join(f"{k} {c[k]}" for k in ("right", "wrong", "unsure", "empty")))


if __name__ == "__main__":
    main()
