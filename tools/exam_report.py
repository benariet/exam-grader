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
        -> <out>/<n>_report.txt (every remark of the exam, in Hebrew, part by part), <out>/summary.csv and
           <out>/summary.html (an exam a row: each part's verdicts; the points surely off - a multiple-choice letter
           wrong where the question is all or nothing (--sure) - beside the lecturer's)
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
    print(f"{len(exams)} exams -> {a.out}")


if __name__ == "__main__":
    main()
