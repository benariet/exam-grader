"""Answers the checks could not decide - words the solution's key does not hold (1(b)'s rows), a final value the readers
did not agree on or written in words (1(f): 'no energy is drawn') - put before a human once, on this computer: a
workbook, an answer a row (the exam, the part, what was read, the writing itself cut out, a column to mark right or
wrong). Marked, its decisions go back to the tools (--decisions): each marked answer decides its exam's part, and each
wording marked alike everywhere decides the same wording in any exam - this Moed's and the next. The workbook and the
decisions hold the students' answers: they stay with the exams, never in the repository.

    python tools/decide_sheet.py make --words <words_box's folder> --words-crops <its crops> \\
        [--value <value_area's folder> --value-crops <its crops> --value-readings <read_crops' readings.json>] ... \\
        --out <xlsx>
    python tools/decide_sheet.py apply <xlsx> --out <decisions.json>      (added to what decisions.json holds)
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MARKS = {"נכון": "right", "שגוי": "wrong"}
HEADER = ["מבחן", "סעיף", "מה נקרא", "מה נכתב (התמונה)", "החלטה", "הערה", "part"]


def norm(text: str) -> str:
    """a reading -> its wording for matching: LaTeX's wrappings, punctuation and spacing gone, lower case ('Full-Adder',
    '\\text{full adder}' -> 'fulladder'); '' when nothing is left or the reader could not read it ('?')"""
    if not text or "?" in text:
        return ""
    t = re.sub(r"\\(?:text|mathrm|textrm|operatorname|mathit|rm)\s*\{([^{}]*)\}", r"\1", text)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    return re.sub(r"[^0-9a-z\u0590-\u05ff]+", "", t.lower())


def load(path: str) -> dict:
    """decisions.json -> {"exams": {part: {exam: verdict}}, "wordings": {part: {wording: verdict}}}"""
    if not path or not os.path.isfile(path):
        return {"exams": {}, "wordings": {}}
    d = json.load(open(path, encoding="utf-8"))
    return {"exams": d.get("exams", {}), "wordings": d.get("wordings", {})}


def decided(dec: dict, part: str, exam: str, text: str):
    """the human's verdict for this exam's part, else for its wording -> (verdict, 'exam' / 'wording') or (None, '')"""
    v = dec.get("exams", {}).get(part, {}).get(str(exam))
    if v:
        return v, "exam"
    w = norm(text)
    v = dec.get("wordings", {}).get(part, {}).get(w) if w else None
    return (v, "wording") if v else (None, "")


def _rows_words(folder, crops):
    """words_box's undecided rows -> [(exam, part, label, reading, image path)]"""
    out = []
    for f in glob.glob(os.path.join(folder, "*.words.json")):
        r = json.load(open(f, encoding="utf-8"))
        for x in r["rows"]:
            if x["verdict"] == "unsure":
                row = x.get("readers", {}).get("vlm") or x.get("read") or ""
                out.append((r["exam"], f"words:{x['name']}", f"1(ב) {x['name']}", row,
                            os.path.join(crops, f"{r['exam']}_{x['name']}.png")))
    return out


def _rows_value(folder, crops, readings):
    """value_area's undecided values -> [(exam, part, label, reading, PIL image)]: the place the vision model found cut
    out of the place's picture (the whole place, smaller, where it found none)"""
    from PIL import Image
    from tools.read_crops import final_crop
    got = json.load(open(readings, encoding="utf-8")) if readings else {}
    out = []
    for f in glob.glob(os.path.join(folder, "*.value.json")):
        r = json.load(open(f, encoding="utf-8"))
        if r["verdict"] != "unsure":
            continue
        n = r["exam"]
        g = got.get(f"{n}_value", {})
        text = (g.get("vlmwords") or g.get("vlmfinal") or [[r.get("read") or ""]])[0][0]
        box = os.path.join(crops, f"{n}_valuebox.png")
        img = None
        if os.path.isfile(box):
            img = Image.open(box).convert("RGB")
            img = final_crop(img, g["where"]) if g.get("where") else img
        out.append((n, f"value:{r.get('name', '')}", r.get("name", ""), text, img))
    return out


def make(a):
    import io

    import openpyxl
    from openpyxl.drawing.image import Image as XImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from PIL import Image
    rows = []
    for folder, crops in zip(a.words, a.words_crops):
        rows += _rows_words(folder, crops)
    for folder, crops, rd in zip(a.value, a.value_crops, a.value_readings + [""] * len(a.value)):
        rows += _rows_value(folder, crops, rd)
    num = lambda n: int(re.sub(r"\D", "", str(n)) or 0)               # noqa: E731
    rows.sort(key=lambda t: (t[1], norm(t[3]), num(t[0])))             # (a part's same wordings one under another)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "החלטות"
    ws.sheet_view.rightToLeft = True
    ws["A1"] = ("תשובות שהבדיקה לא הכריעה. בעמודה 'החלטה' בחר נכון או שגוי (או השאר ריק).\n"
                "כל תשובה שתסמן תוכרע כך במבחן שלה. ניסוח שסימנת בכל מקום אותו דבר יוכרע כך גם במבחנים אחרים.")
    ws["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A1:F1")
    ws.row_dimensions[1].height = 36
    ws.page_setup.orientation = "landscape"            # (printed: the columns on one page's width)
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.print_title_rows = "2:2"
    for c, h in enumerate(HEADER, 1):
        cell = ws.cell(row=2, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    dv = DataValidation(type="list", formula1='"נכון,שגוי,לא ברור"', allow_blank=True)
    ws.add_data_validation(dv)
    for c, w in zip("ABCDEFG", (8, 34, 34, 70, 12, 30, 2)):
        ws.column_dimensions[c].width = w
    ws.column_dimensions["G"].hidden = True             # (the part's id, for apply)
    for i, (exam, part, label, text, img) in enumerate(rows, 3):
        for c, v in zip("ABCFG", (str(exam), label, text, "", part)):
            ws[f"{c}{i}"] = v
            ws[f"{c}{i}"].alignment = Alignment(wrap_text=True, vertical="center")
        dv.add(f"E{i}")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB") if os.path.isfile(img) else None
        h = 30
        if img is not None:
            s = min(1.0, 480 / img.width, 110 / img.height)
            im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
            buf = io.BytesIO()
            im.save(buf, "PNG")
            xi = XImage(buf)
            ws.add_image(xi, f"D{i}")
            h = max(h, im.height * 0.78 + 6)
        ws.row_dimensions[i].height = h
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wb.save(a.out)
    from collections import Counter
    c = Counter(p for _, p, _, _, _ in rows)
    print(f"{a.out}: {len(rows)} answers - " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


def apply(a):
    import openpyxl
    ws = openpyxl.load_workbook(a.sheet).active
    dec = load(a.out)
    seen = {}                                          # (part, wording) -> {verdicts marked}
    n_marked = 0
    for row in ws.iter_rows(min_row=3, values_only=True):
        exam, _, text, _, mark, _, part = (list(row) + [None] * 7)[:7]
        if not exam or not part:
            continue
        v = MARKS.get(str(mark or "").strip())
        w = norm(text or "")
        if w:
            seen.setdefault((part, w), set()).add(v)   # (unmarked or 'not clear' too: a wording marked both ways
        if v:                                          #  or only sometimes decides no other exam)
            dec["exams"].setdefault(part, {})[str(exam)] = v
            n_marked += 1
    n_w = 0
    for (part, w), vs in seen.items():
        if len(vs) == 1 and None not in vs:
            dec["wordings"].setdefault(part, {})[w] = vs.pop()
            n_w += 1
    json.dump(dec, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{a.out}: {n_marked} answers decided, {n_w} wordings that decide others too")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--words", action="append", default=[])
    m.add_argument("--words-crops", action="append", default=[])
    m.add_argument("--value", action="append", default=[])
    m.add_argument("--value-crops", action="append", default=[])
    m.add_argument("--value-readings", action="append", default=[])
    m.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("sheet")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else apply(a)


if __name__ == "__main__":
    main()
