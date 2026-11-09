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

ORDER = ("table", "exprs", "words", "circuit", "choices", "value")
REMARKS = {"table": "_table_remarks.txt", "exprs": "_expr_remarks.txt", "words": "_words_remarks.txt",
           "circuit": "_circuit_remarks.txt", "choices": "_choice_remarks.txt", "value": "_value_remarks.txt"}


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
            label = r.get("name") or (text.split(":")[0] if ":" in text else os.path.basename(d.rstrip("/\\")))
            read = "" if r.get("value") is None else f"{r['value']:g}"
            out[n].append((label, read, HEB.get(r["verdict"], r["verdict"]), "", text.split(":", 1)[-1].strip()))
        key = os.path.join(d, "circuit_key.json")
        if os.path.isfile(key):
            for q, rows in json.load(open(key, encoding="utf-8")).items():
                for r in rows:
                    n = r["exam"]
                    funcs = "; ".join(f"{k} = {v}" for k, v in (r.get("functions") or {}).items())
                    read = (f"{r['transistors']} טרנזיסטורים" if r.get("transistors") else "") + (f"; {funcs}" if funcs else "")
                    rem = os.path.join(d, f"{n}_circuit_remarks.txt")
                    why = open(rem, encoding="utf-8").read().strip().split(":", 1)[-1].strip() if os.path.isfile(rem) else ""
                    out[n].append((f"{q}.1 המעגל", read, HEB.get(r["verdict"], r["verdict"]), "", why))
    order = lambda t: (_q(t[0]) or 99, t[0])           # noqa: E731
    return {n: sorted(v, key=order) for n, v in out.items()}


def write_xlsx(path, det, marks, title):
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
    wb.save(path)


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
              "circuit": "שאלה 2.1 - המעגל", "choices": "שאלות אמריקאיות", "value": "ערך סופי"}
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
    if a.xlsx:
        det = details(a.runs, sure)
        write_xlsx(os.path.join(a.out, "by_exam.xlsx"), det, marks, a.title)
        print(f"by_exam.xlsx: {len(det)} exams, {sum(len(v) for v in det.values())} rows")
    print(f"{len(exams)} exams -> {a.out}")


if __name__ == "__main__":
    main()
