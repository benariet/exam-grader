"""Each exam's checks gathered into one report, and an overview of the whole class - from the folders the course's
tools wrote (all on this computer, beside the exams):

    tools/truth_table.py   <n>.table.json    + <n>_table_remarks.txt
    tools/expr_box.py      <n>.exprs.json    + <n>_expr_remarks.txt
    tools/words_box.py     <n>.words.json    + <n>_words_remarks.txt
    tools/choice_marks.py  <n>.choices.json  + <n>_choice_remarks.txt
    tools/circuit_key.py   circuit_key.json  + <n>_circuit_remarks.txt
    tools/value_area.py    <n>.value.json    + <n>_value_remarks.txt

    python tools/exam_report.py --runs "<folder>" "<folder>" ... --out <folder> [--grades grades.xlsx]
        [--sure "2.2=5,2.3=5"]
        [--xlsx]
        -> <out>/<n>_report.txt (every remark of the exam, in Hebrew, part by part), <out>/summary.csv and
           <out>/summary.html (an exam a row: each part's verdicts; the points surely off - a multiple-choice letter
           wrong where the question is all or nothing (--sure) - beside the lecturer's); with --xlsx
           <out>/by_exam.xlsx: every exam question by question - what was read, the verdict, the points it settles
           (only a multiple-choice letter, all or nothing), why, and what the lecturer took off the question
"""
import argparse
import csv
import glob
import html
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ORDER = ("table", "exprs", "words", "circuit", "choices", "value", "wave", "step")
REMARKS = {"table": "_table_remarks.txt", "exprs": "_expr_remarks.txt", "words": "_words_remarks.txt",
           "circuit": "_circuit_remarks.txt", "choices": "_choice_remarks.txt", "value": "_value_remarks.txt",
           "step": "_step_remarks.txt", "wave": "_wave_remarks.txt"}


def collect(folders):
    """{exam: {part: summary dict}}, {exam: {part: [remark lines]}} from the tools' outputs in folders"""
    summ, rem = defaultdict(dict), defaultdict(dict)
    for d in folders:
        for f in glob.glob(os.path.join(d, "*.table.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]]["table"] = {k: r[k] for k in ("right", "wrong", "unsure", "empty", "strength") if k in r}
        for f in glob.glob(os.path.join(d, "*.exprs.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]]["exprs"] = {x["name"]: x["verdict"] for x in r["rows"]}
        for f in glob.glob(os.path.join(d, "*.words.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]]["words"] = {x["name"]: x["verdict"] for x in r["rows"]}
        for f in glob.glob(os.path.join(d, "*.choices.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]]["choices"] = {x["item"]: (x["verdict"], x.get("chosen", "")) for x in r["items"]}
        for f in glob.glob(os.path.join(d, "*.value.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]].setdefault("value", {})[os.path.basename(d.rstrip("/\\"))] = r["verdict"]
        for f in glob.glob(os.path.join(d, "*.wave.json")):
            r = json.load(open(f, encoding="utf-8"))
            summ[r["exam"]].setdefault("wave", {})[os.path.basename(d.rstrip("/\\"))] = r["verdict"]
        key = os.path.join(d, "circuit_key.json")
        if os.path.isfile(key):
            for q, rows in json.load(open(key, encoding="utf-8")).items():
                for r in rows:
                    summ[r["exam"]]["circuit"] = {f"q{q}": r["verdict"]}
        for part, suffix in REMARKS.items():
            for f in glob.glob(os.path.join(d, "*" + suffix)):
                name = os.path.basename(f)[:-len(suffix)]
                rem[name].setdefault(part, []).extend(x for x in open(f, encoding="utf-8").read().splitlines() if x.strip())
    return summ, rem


STEP = {"right": "כמו בפתרון", "wrong": "שונה מהפתרון", "unsure": "לא נקרא בוודאות", "empty": "לא נמצא"}
HEB = {"right": "נכון", "wrong": "שגוי", "unsure": "לבדיקה ידנית", "unclear": "לא ברור", "none": "לא סומן",
       "empty": "ריק", "other": "פונקציה אחרת", "no drawing": "אין ציור"}


def _q(label: str):
    """'1(א) ...' / '2.2' / 'שאלה 3 - ...' -> the question's number"""
    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else None


def details(folders, sure):
    """{exam: [(question label, what was read, verdict (Hebrew), points settled, why)]} from the tools' outputs - each
    exam's questions part by part, as the per-exam workbook shows them"""
    out = defaultdict(list)
    for d in folders:
        for f in glob.glob(os.path.join(d, "*.table.json")):
            r = json.load(open(f, encoding="utf-8"))
            n = r["exam"]
            rem = os.path.join(d, f"{n}_table_remarks.txt")
            lines = open(rem, encoding="utf-8").read().splitlines()[1:] if os.path.isfile(rem) else []
            read = f"{r['right']} תאים נכונים, {r['wrong']} שגויים, {r['unsure']} לא נקראו בוודאות, {r['empty']} ריקים"
            if r.get("strength"):
                read += f"; חוזק שונה מהפתרון ב-{r['strength']} תאים"
            verdict = "שגוי" if r["wrong"] else "לבדיקה ידנית" if r["unsure"] or r["empty"] or r.get("strength") else "נכון"
            why = " ".join(x.strip() for x in lines if x.strip()) or ("כל התאים שנקראו כמו בפתרון" if not r["wrong"] else "")
            out[n].append(("1(א) טבלת האמת", read, verdict, "", why))
        for f in glob.glob(os.path.join(d, "*.exprs.json")):
            r = json.load(open(f, encoding="utf-8"))
            for x in r["rows"]:
                agreed = "+".join(x.get("agreed") or [])
                why = {"right": "הפונקציה שנקראה שקולה לפתרון",
                       "wrong": f"נקראה פונקציה אחרת ({x.get('function') or ''})",
                       "unsure": "הקוראים לא הסכימו על פונקציה אחת, או שלא נקרא בוודאות",
                       "empty": "לא נמצא כתב בשורה"}.get(x["verdict"], "")
                if agreed:
                    why += f"; קוראים שהסכימו: {agreed}"
                if x.get("matches_table") is False:
                    why += "; לא תואם לעמודה בטבלת האמת של הסטודנט"
                out[r["exam"]].append((f"1(א) ביטוי {x['name']}", x.get("read") or "", HEB.get(x["verdict"], x["verdict"]),
                                       "", why))
        for f in glob.glob(os.path.join(d, "*.words.json")):
            r = json.load(open(f, encoding="utf-8"))
            for x in r["rows"]:
                why = {"right": "נכתבה התשובה שבפתרון (נקראה בשורה עצמה)",
                       "wrong": "נכתבה תשובה אחרת מזו שבפתרון",
                       "unsure": "קריאת השורה לא הכריעה (או שקריאת התיבה כולה לא הסכימה איתה)",
                       "empty": "לא נמצא כתב בשורה"}.get(x["verdict"], "")
                if x.get("decided"):                   # (tools/decide_sheet.py: a human's mark)
                    why = "לפי החלטת הבודק" + (" על אותו ניסוח במבחן אחר" if x["decided"] == "wording" else "")
                out[r["exam"]].append((f"1(ב) {x['name']}", x.get("read") or "", HEB.get(x["verdict"], x["verdict"]),
                                       "", why))
        for f in glob.glob(os.path.join(d, "*.choices.json")):
            r = json.load(open(f, encoding="utf-8"))
            for x in r["items"]:
                pts = sure.get(x["item"])
                points = "" if pts is None else (f"0 מתוך {pts:g} (ודאי)" if x["verdict"] == "wrong" else
                                                 f"{pts:g} מתוך {pts:g} אם ההסבר נכון" if x["verdict"] == "right" else "")
                why = {"right": f"סומנה האות שבפתרון ({x.get('answer', '')}); ההסבר לא נבדק",
                       "wrong": f"סומנה {x.get('chosen', '')}, בפתרון {x.get('answer', '')}; התשובה הכול או כלום",
                       "none": "לא נמצא סימון על התשובות (אולי נכתבה בתיבה)",
                       "unclear": "סומנו כמה תשובות במידה דומה (אולי שינוי תשובה)"}.get(x["verdict"], "")
                read = (f"סומנה {x['chosen']}" if x.get("chosen") else "") + \
                    (f" (מוקפת: {', '.join(x['circled'])})" if x.get("circled") else "")
                out[r["exam"]].append((f"{x['item']} אמריקאית", read, HEB.get(x["verdict"], x["verdict"]), points, why))
        for f in glob.glob(os.path.join(d, "*.value.json")):
            r = json.load(open(f, encoding="utf-8"))
            n = r["exam"]
            rem = os.path.join(d, f"{n}_value_remarks.txt")
            text = open(rem, encoding="utf-8").read().strip() if os.path.isfile(rem) else ""
            if r.get("kind") == "step":                # (a value on the way: where the work went astray)
                rem = os.path.join(d, f"{n}_step_remarks.txt")
                text = open(rem, encoding="utf-8").read().strip() if os.path.isfile(rem) else ""
            label = r.get("name") or (text.split(":")[0] if ":" in text else os.path.basename(d.rstrip("/\\")))
            read = r.get("read") or "" if r.get("value") is None else f"{r['value']:g}"
            verdict = (STEP if r.get("kind") == "step" else HEB).get(r["verdict"], r["verdict"])
            out[n].append((label, read, verdict, "", text.split(":", 1)[-1].strip()))
        for f in glob.glob(os.path.join(d, "*.wave.json")):    # (tools/waveform.py: a drawn waveform's shape)
            r = json.load(open(f, encoding="utf-8"))
            n = r["exam"]
            rem = os.path.join(d, f"{n}_wave_remarks.txt")
            text = open(rem, encoding="utf-8").read().strip() if os.path.isfile(rem) else ""
            out[n].append((r.get("name", ""), "", HEB.get(r["verdict"], r["verdict"]), "", text.split(":", 1)[-1].strip()))
        key = os.path.join(d, "circuit_key.json")
        if os.path.isfile(key):
            for q, rows in json.load(open(key, encoding="utf-8")).items():
                for r in rows:
                    n = r["exam"]
                    funcs = "; ".join(f"{k} = {v}" for k, v in (r.get("functions") or {}).items())
                    read = (f"{r['transistors']} טרנזיסטורים" if r.get("transistors") else "") + (f"; {funcs}" if funcs else "")
                    rem = os.path.join(d, f"{n}_circuit_remarks.txt")
                    why = open(rem, encoding="utf-8").read().strip().split(":", 1)[-1].strip() if os.path.isfile(rem) else ""
                    v = {"none": "לבדיקה ידנית", "other": "פונקציה אחרת", "no drawing": "אין ציור"}.get(r["verdict"])
                    out[n].append((f"{q}.1 המעגל", read, v or HEB.get(r["verdict"], r["verdict"]), "", why))
    order = lambda t: (_q(t[0]) or 99, t[0])           # noqa: E731
    return {n: sorted(v, key=order) for n, v in out.items()}


WRONG = {"שגוי", "פונקציה אחרת"}


def points_view(det, rubric, marks):
    """each exam's questions by the exam's own points (rubric: [{"item": "1(א)", "q": 1, "points": 10, "match": regex
    on a row's label, "mcq": all or nothing}]) -> (rows [exam, question, item, points, status, what the checks say],
    per question [exam, question, its points, the lecturer's, kept, lost for sure, with an error found, open,
    agrees], coverage: the share of the points the checks decided, per exam). An item: right when every check of it
    is; an error found when one is wrong (how much it costs is the lecturer's); open when some are not decided; not
    checked when none is. A value on the way (a step) is no item's check."""
    items, per_q, cover = [], [], {}
    for n in sorted(det, key=lambda x: int(re.sub(r"\D", "", x) or 0)):
        rows = [r for r in det[n] if r[2] not in STEP.values()]
        byq = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])        # (points, kept, lost, error found, open)
        for it in rubric:
            mine = [r for r in rows if re.search(it["match"], r[0])]
            verdicts = [r[2] for r in mine]
            if not mine:
                status = "לא נבדק"
            elif it.get("mcq"):
                status = "נכון (לבדוק את ההסבר)" if verdicts == ["נכון"] else "שגוי - ירד בוודאות" if \
                    verdicts == ["שגוי"] else "פתוח"
            elif any(v in WRONG for v in verdicts):
                status = "נמצאה טעות"
            elif all(v == "נכון" for v in verdicts):
                status = "נכון" + (" (בחלקים שנבדקו)" if it.get("partial") else "")
            else:
                status = "פתוח" + (" (חלק נבדק ונכון)" if "נכון" in verdicts else "")
            p = float(it["points"])
            b = byq[it["q"]]
            b[0] += p
            if status.startswith("נכון"):
                b[1] += p
            elif status.startswith("שגוי"):
                b[2] += p
            elif status == "נמצאה טעות":
                b[3] += p
            else:
                b[4] += p
            items.append([n, it["q"], it["item"], p, status,
                          "; ".join(f"{r[0]}: {r[2]}" for r in mine) if mine else ""])
        done = 0.0
        for q, (pts, kept, lost, err, opn) in sorted(byq.items()):
            off = (marks.get(n) or {}).get(q)
            got = None if off is None else pts - off
            agrees = "" if got is None else ("לבדוק: המרצה נתן יותר ממה שאפשר" if got > pts - lost + 1e-9 else
                                             "לבדוק: נמצאה טעות והמרצה לא הוריד" if err and off == 0 else "מתיישב")
            per_q.append([n, q, pts, "" if got is None else got, kept, lost, err, opn, agrees])
            done += kept + lost + err
        cover[n] = done / max(1e-9, sum(v[0] for v in byq.values()))
    return items, per_q, cover


def write_points(wb, items, per_q, cover):
    """the workbook's points sheets: per question, and per item"""
    from openpyxl.styles import Alignment, Font, PatternFill
    F = "Arial"
    ws = wb.create_sheet("ניקוד לפי שאלה", 1)
    ws.sheet_view.rightToLeft = True
    ws["A1"] = "ניקוד לפי חלוקת הנקודות בטופס: מה הבדיקה אישרה, מה ירד בוודאות, איפה נמצאה טעות ומה פתוח"
    ws["A1"].font = Font(name=F, bold=True, size=12)
    ws["A2"] = (f"הבדיקה הכריעה בממוצע {100 * sum(cover.values()) / max(1, len(cover)):.0f}% מהנקודות "
                "(אושר, ירד בוודאות או נמצאה טעות). 'נמצאה טעות' - כמה יורד על הטעות נשאר להחלטת המרצה.")
    ws["A2"].font = Font(name=F, italic=True, size=9, color="595959")
    head = ["מבחן", "שאלה", "נקודות", "המרצה נתן", "אושר", "ירד בוודאות", "נמצאה טעות", "פתוח / לא נבדק", "בדיקה"]
    for j, h in enumerate(head, 1):
        c = ws.cell(row=4, column=j, value=h)
        c.font, c.fill = Font(name=F, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
    for i, r in enumerate(per_q, 5):
        for j, v in enumerate(r, 1):
            c = ws.cell(row=i, column=j, value=v)
            c.font, c.alignment = Font(name=F, size=10), Alignment(horizontal="right")
        if str(r[-1]).startswith("לבדוק"):
            ws.cell(row=i, column=len(r)).fill = PatternFill("solid", fgColor="FCE4D6")
    for j, w in enumerate([8, 7, 8, 11, 8, 11, 11, 14, 34], 1):
        ws.column_dimensions[chr(64 + j)].width = w
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{4 + len(per_q)}"
    ws2 = wb.create_sheet("ניקוד לפי סעיף", 2)
    ws2.sheet_view.rightToLeft = True
    for j, h in enumerate(["מבחן", "שאלה", "סעיף", "נקודות", "מצב", "הבדיקות"], 1):
        c = ws2.cell(row=1, column=j, value=h)
        c.font, c.fill = Font(name=F, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
    fills = {"נכון": "E2EFDA", "שגוי": "FCE4D6", "נמצאה": "FCE4D6", "פתוח": "FFF2CC", "לא נבדק": "EDEDED"}
    for i, r in enumerate(items, 2):
        for j, v in enumerate(r, 1):
            c = ws2.cell(row=i, column=j, value=v)
            c.font, c.alignment = Font(name=F, size=10), Alignment(horizontal="right", wrap_text=j == 6)
        f = next((v for k, v in fills.items() if str(r[4]).startswith(k)), None)
        if f:
            ws2.cell(row=i, column=5).fill = PatternFill("solid", fgColor=f)
    for j, w in enumerate([8, 7, 10, 8, 26, 90], 1):
        ws2.column_dimensions[chr(64 + j)].width = w
    ws2.freeze_panes = "A2"
    ws2.auto_filter.ref = f"A1:F{1 + len(items)}"


def write_xlsx(path, det, marks, title, rubric=None):
    """the per-exam workbook: every exam's questions in one sheet (to sort and filter), and a sheet per exam"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    F = "Arial"
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head = (Font(name=F, bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864"))
    fills = {"נכון": "E2EFDA", "שגוי": "FCE4D6", "פונקציה אחרת": "FCE4D6", "לבדיקה ידנית": "FFF2CC",
             "לא ברור": "FFF2CC", "ריק": "EDEDED", "לא סומן": "EDEDED", "אין ציור": "EDEDED"}
    cols = ["מבחן", "שאלה / סעיף", "מה נקרא", "פסק", "ניקוד שנקבע אוטומטית", "למה", "המרצה הוריד בשאלה כולה"]
    widths = [8, 24, 44, 14, 22, 70, 14]
    exams = sorted(det, key=lambda n: int(re.sub(r"\D", "", n) or 0))
    wb = Workbook()

    def sheet(ws, rows, with_exam):
        ws.sheet_view.rightToLeft = True
        ws["A1"] = title
        ws["A1"].font = Font(name=F, bold=True, size=13)
        ws["A2"] = ("הבדיקה האוטומטית נותנת פסק לכל סעיף; ניקוד נקבע רק לתשובה אמריקאית (הכול או כלום). "
                    "'לבדיקה ידנית' = לא נבדק. עמודת המרצה: הנקודות שירדו בשאלה כולה, לא בסעיף.")
        ws["A2"].font = Font(name=F, italic=True, size=9, color="595959")
        hs = cols if with_exam else cols[1:]
        for j, h in enumerate(hs, 1):
            c = ws.cell(row=4, column=j, value=h)
            c.font, c.fill, c.border = head[0], head[1], border
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, row in enumerate(rows):
            vals = row if with_exam else row[1:]
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=5 + i, column=j, value=v)
                c.font, c.border = Font(name=F, size=10), border
                c.alignment = Alignment(horizontal="right", vertical="top", wrap_text=True)
            vcol = 4 if with_exam else 3
            fill = fills.get(vals[vcol - 1])
            if fill:
                ws.cell(row=5 + i, column=vcol).fill = PatternFill("solid", fgColor=fill)
        for j, w in enumerate(widths if with_exam else widths[1:], 1):
            ws.column_dimensions[chr(64 + j)].width = w
        ws.freeze_panes = "A5"
        if rows:
            ws.auto_filter.ref = f"A4:{chr(64 + len(hs))}{4 + len(rows)}"

    every = []
    for n in exams:
        for q, read, verdict, points, why in det[n]:
            off = (marks.get(n) or {}).get(_q(q))
            every.append((n, q, read, verdict, points, why, "" if off is None else off))
    ws = wb.active
    ws.title = "כל המבחנים"
    sheet(ws, every, True)
    for n in exams:
        sheet(wb.create_sheet(f"מבחן {n}"[:31]), [r for r in every if r[0] == n], True)
    if rubric:                                         # (the exam's own points: what is decided, what is left)
        write_points(wb, *points_view(det, rubric, marks))
    wb.save(path)


def write_compare(path, n, rows, marks_n, points, sure, title):
    """one exam, question by question: the lecturer's points beside what the checks found in it - each part read, its
    verdict, why - the points they settle, and whether the two agree (a sure deduction more than the lecturer took off
    is a contradiction; parts found wrong with nothing taken off, or points off with every checked part right, are
    pointed out)"""
    by_q = defaultdict(list)
    for r in rows:
        by_q[_q(r[0])].append(r)
    color = {"נכון": "#e2efda", "שגוי": "#fce4d6", "פונקציה אחרת": "#fce4d6", "לבדיקה ידנית": "#fff2cc",
             "לא ברור": "#fff2cc"}
    out = [f"<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>מבחן {html.escape(n)}</title>",
           "<style>body{font-family:Arial,sans-serif;margin:20px;max-width:1100px}table{border-collapse:collapse;width:100%;"
           "margin:6px 0 18px}td,th{border:1px solid #ccc;padding:5px 8px;font-size:13px;vertical-align:top;text-align:right}"
           "th{background:#1f3864;color:#fff}.box{background:#f4f6fa;border:1px solid #d0d7e5;padding:8px 12px;margin:6px 0}"
           "</style>", f"<h1>מבחן {html.escape(n)}</h1><p>{html.escape(title)}</p>"]
    tot_max = tot_lect = 0.0
    for q in sorted(points):
        mx = points[q]
        off = marks_n.get(q)
        parts = by_q.get(q, [])
        settled = sum(pts for item, pts in sure.items() if _q(item) == q and
                      any(r[0].startswith(item) and r[2] == "שגוי" for r in parts))
        wrong = [r for r in parts if r[2] in ("שגוי", "פונקציה אחרת")]
        right = [r for r in parts if r[2] == "נכון"]
        open_ = [r for r in parts if r[2] not in ("שגוי", "פונקציה אחרת", "נכון")]
        if off is not None:
            tot_max += mx
            tot_lect += mx - off
        if off is None:
            agree = "אין ציון מהמרצה לשאלה זו."
        elif settled > off:
            agree = f"סתירה: המערכת בטוחה שירדו לפחות {settled:g} נקודות, והמרצה הוריד {off:g} - לבדוק."
        elif wrong and off == 0:
            agree = "שים לב: המערכת מצאה סעיפים שגויים, אבל המרצה לא הוריד נקודות בשאלה - לבדוק את הקריאה."
        elif off > 0 and not wrong:
            agree = (f"המרצה הוריד {off:g} נקודות, והמערכת לא מצאה שגיאה ודאית בסעיפים שהיא בודקת"
                     + (f" ({len(open_)} סעיפים לא הוכרעו)" if open_ else "") +
                     " - ההורדה בחלקים שהמערכת לא בודקת (הסבר, דרך חישוב, ציור) או בסעיפים שלא הוכרעו.")
        elif off > 0:
            agree = (f"מתיישב: המרצה הוריד {off:g} נקודות, והמערכת מצאה {len(wrong)} סעיפים שגויים"
                     + (f" (מהם {settled:g} נקודות ודאיות)" if settled else "") + ".")
        else:
            agree = "מתיישב: המרצה לא הוריד נקודות, והמערכת לא מצאה שגיאה." if not open_ else \
                f"המרצה לא הוריד נקודות; המערכת לא מצאה שגיאה ({len(open_)} סעיפים לא הוכרעו)."
        lect = "-" if off is None else f"{mx - off:g} מתוך {mx:g} (הוריד {off:g})"
        out.append(f"<h2>שאלה {q}</h2><div class='box'><b>המרצה:</b> {html.escape(lect)}<br>"
                   f"<b>המערכת:</b> {len(right)} נכונים, {len(wrong)} שגויים, {len(open_)} לבדיקה ידנית"
                   + (f"; ירדו בוודאות {settled:g} נקודות" if settled else "") +
                   f"<br><b>השוואה:</b> {html.escape(agree)}</div>")
        out.append("<table><tr><th>סעיף</th><th>מה נקרא</th><th>פסק</th><th>ניקוד שנקבע</th><th>למה</th></tr>")
        for label, read, verdict, pts, why in parts:
            out.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(read)}</td>"
                       f"<td style='background:{color.get(verdict, '#eee')}'>{html.escape(verdict)}</td>"
                       f"<td>{html.escape(pts)}</td><td>{html.escape(why)}</td></tr>")
        out.append("</table>")
    if tot_max:
        out.append(f"<p><b>סה\"כ אצל המרצה:</b> {tot_lect:g} מתוך {tot_max:g} (לפני פקטור).</p>")
    out.append("<p style='font-size:12px;color:#666'>המערכת נותנת פסק לכל סעיף; ניקוד נקבע רק לתשובה אמריקאית "
               "(הכול או כלום). 'לבדיקה ידנית' = לא נבדק. ציון המרצה הוא לשאלה כולה.</p></html>")
    open(path, "w", encoding="utf-8").write("\n".join(out))


def sure_off(s, sure):
    """the points surely off: each multiple-choice item answered with another letter, where it is all or nothing"""
    return sum(pts for item, pts in sure.items() if s.get("choices", {}).get(item, ("",))[0] == "wrong")


def _cell(part, v):
    color = {"right": "#d8f0dc", "wrong": "#f6d5d5", "unsure": "#fbe8c8", "unclear": "#fbe8c8", "none": "#eeeeee",
             "empty": "#eeeeee", "other": "#f6d5d5", "no drawing": "#eeeeee"}
    heb = {"right": "נכון", "wrong": "שגוי", "unsure": "לבדיקה", "unclear": "לא ברור", "none": "לא סומן",
           "empty": "ריק", "other": "פונקציה אחרת", "no drawing": "אין ציור"}
    if isinstance(v, tuple):
        v, letter = v
        txt = heb.get(v, v) + (f" ({letter})" if letter else "")
    else:
        txt = heb.get(v, v)
    return f"<td style='background:{color.get(v, '#fff')}'>{html.escape(txt)}</td>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grades", default="")
    ap.add_argument("--sure", default="", help="multiple-choice items all or nothing, and their points: '2.2=5,2.3=5'")
    ap.add_argument("--xlsx", action="store_true", help="<out>/by_exam.xlsx: every exam question by question")
    ap.add_argument("--title", default="בדיקה אוטומטית - כל מבחן, שאלה אחר שאלה")
    ap.add_argument("--compare", default="", help="exams ('1,3' or 'all'): <out>/compare_<n>.html each - question by "
                    "question, the lecturer's points beside the checks' findings and whether they agree")
    ap.add_argument("--points", default="", help="each question's points, for --compare: '1=50,2=25,3=25'")
    ap.add_argument("--rubric", default="", help="the exam's points per item (JSON file): by_exam.xlsx gets sheets of "
                    "what is decided of them, against the lecturer's")
    a = ap.parse_args()
    summ, rem = collect(a.runs)
    sure = {k.strip(): float(v) for k, v in (x.split("=") for x in a.sure.split(",") if "=" in x)}
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    os.makedirs(a.out, exist_ok=True)
    exams = sorted(set(summ) | set(rem), key=lambda n: int(re.sub(r"\D", "", n) or 0))
    titles = {"table": "שאלה 1(א) - טבלת האמת", "exprs": "שאלה 1(א) - הביטויים", "words": "שאלה 1(ב)",
              "circuit": "שאלה 2.1 - המעגל", "choices": "שאלות אמריקאיות", "value": "ערך סופי",
              "step": "ערכי ביניים (איפה החישוב סטה מהפתרון)", "wave": "צורות האותות"}
    for n in exams:
        lines = [f"מבחן {n} - בדיקה אוטומטית (לעיון המרצה; כל 'לבדיקה ידנית' לא נבדק)", ""]
        for part in ORDER:
            if rem[n].get(part):
                lines += [titles[part] + ":"] + ["  " + x for x in rem[n][part]] + [""]
        off = sure_off(summ.get(n, {}), sure)
        if sure:
            lines.append(f"נקודות שירדו בוודאות (תשובה אמריקאית שגויה, הכול או כלום): {off:g}")
        open(os.path.join(a.out, f"{n}_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    cols = []                                          # (part, key)
    for part in ORDER:
        keys = sorted({k for n in exams for k in summ.get(n, {}).get(part, {})}) if part != "table" else \
            ([k for k in ("right", "wrong", "unsure", "empty", "strength")
              if any(k in summ.get(n, {}).get("table", {}) for n in exams)])
        cols += [(part, k) for k in keys]
    with open(os.path.join(a.out, "summary.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exam"] + [f"{p}:{k}" for p, k in cols] + (["sure off"] if sure else []) +
                   (["lecturer off"] if marks else []))
        for n in exams:
            s = summ.get(n, {})
            row = [n]
            for p, k in cols:
                v = s.get(p, {}).get(k, "")
                row.append(v[0] + (f" {v[1]}" if v[1] else "") if isinstance(v, tuple) else v)
            if sure:
                row.append(sure_off(s, sure))
            if marks:
                row.append(sum(x for x in marks.get(n, {}).values() if x is not None) if n in marks else "")
            w.writerow(row)
    head = "".join(f"<th>{html.escape(titles[p].split(' - ')[0] if p != 'table' else 'טבלה')}<br>{html.escape(k)}</th>"
                   for p, k in cols)
    body = []
    for n in exams:
        s = summ.get(n, {})
        tds = []
        for p, k in cols:
            v = s.get(p, {}).get(k)
            tds.append("<td></td>" if v is None else f"<td>{v}</td>" if p == "table" else _cell(p, v))
        extra = (f"<td>{sure_off(s, sure):g}</td>" if sure else "") + \
            (f"<td>{sum(x for x in marks.get(n, {}).values() if x is not None):g}</td>" if marks and n in marks else
             ("<td></td>" if marks else ""))
        body.append(f"<tr><td><a href='{n}_report.txt'>{html.escape(n)}</a></td>{''.join(tds)}{extra}</tr>")
    open(os.path.join(a.out, "summary.html"), "w", encoding="utf-8").write(
        "<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Class overview</title>"
        "<style>body{font-family:Arial,sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
        "padding:4px 8px;font-size:13px;text-align:center}</style><h1>סיכום הבדיקה האוטומטית</h1>"
        "<p>ירוק: נכון. אדום: שגוי. כתום: לבדיקה ידנית. בעמודות הטבלה: מספר התאים. לחיצה על מספר המבחן: כל ההערות.</p>"
        f"<table><tr><th>מבחן</th>{head}" + ("<th>ירדו בוודאות</th>" if sure else "") +
        ("<th>המרצה הוריד (סה\"כ)</th>" if marks else "") + "</tr>" + "\n".join(body) + "</table></html>")
    if a.compare:
        det = details(a.runs, sure)
        pts = {int(k): float(v) for k, v in (x.split("=") for x in a.points.split(",") if "=" in x)}
        which = exams if a.compare == "all" else [x.strip() for x in a.compare.split(",") if x.strip()]
        for n in which:
            write_compare(os.path.join(a.out, f"compare_{n}.html"), n, det.get(n, []), marks.get(n, {}),
                          pts or {q: 0 for q in sorted({_q(r[0]) for r in det.get(n, [])} - {None})}, sure, a.title)
        print(f"compare: {len(which)} exams")
    if a.xlsx:
        det = details(a.runs, sure)
        rubric = json.load(open(a.rubric, encoding="utf-8")) if a.rubric else None
        write_xlsx(os.path.join(a.out, "by_exam.xlsx"), det, marks, a.title, rubric)
        print(f"by_exam.xlsx: {len(det)} exams, {sum(len(v) for v in det.values())} rows")
        if rubric:
            _, per_q, cover = points_view(det, rubric, marks)
            print(f"points decided: {100 * sum(cover.values()) / max(1, len(cover)):.0f}% on average; questions to "
                  f"look at: {sum(1 for r in per_q if str(r[-1]).startswith('לבדוק'))} of {len(per_q)}")
    print(f"{len(exams)} exams -> {a.out}")


if __name__ == "__main__":
    main()
