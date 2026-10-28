"""Print an exam as an answer-box form (PDF), with the layout the reader needs.

    python -m forms.make_exam                # all exams of forms/elec101.py -> data/course_elec101/

Per exam: <id>.pdf (the form to print), <id>_key.pdf (the same form with the answers and short
solutions), <id>.json (questions, answers, points, and the template: every cell's box per page in
150-dpi page pixels, the corner markers, the page QR codes).

The form (Hebrew, right to left): four black corner markers per page (the reader straightens the page
by them, scan or phone photo), a QR code per page ('elec101|E1|2': course, exam, page), the student's
ID in 9 cells, and for every question a work area and a final-answer box: one character per cell
(digit, decimal point or minus), left to right, the unit printed after it; a multiple-choice question
takes the option's number in one cell.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from forms import elec101  # noqa: E402
from forms.diagrams import DIAGRAMS  # noqa: E402

DPI = 150
PX = DPI / 96.0                       # CSS px -> 150-dpi page pixels
CELLS = 8                             # cells of a numeric answer
MARKER_MM, MARKER_INSET_MM = 7, 8

CSS = """
@page { size: 210mm 297mm; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Liberation Sans', 'DejaVu Sans', sans-serif; color: #111; }
.page { position: relative; width: 210mm; height: 297mm; page-break-after: always; overflow: hidden; direction: rtl; }
.marker { position: absolute; width: %(m)smm; height: %(m)smm; background: #000; }
.qr { position: absolute; bottom: 6mm; left: 21mm; width: 17mm; height: 17mm; }
.qr img { width: 100%%; height: 100%%; image-rendering: pixelated; }
.foot { position: absolute; bottom: 9mm; right: 20mm; font-size: 9pt; color: #555; }
.foot.l { right: auto; left: 41mm; direction: ltr; }
.content { position: absolute; top: 19mm; bottom: 27mm; left: 17mm; right: 17mm; display: flex; flex-direction: column; gap: 3mm; }
.head { border-bottom: 1.2px solid #222; padding-bottom: 2mm; }
.head h1 { font-size: 15pt; margin: 0 0 1mm 0; }
.head .sub { font-size: 10pt; color: #333; }
.ident { display: flex; gap: 8mm; align-items: center; font-size: 10.5pt; margin-top: 2mm; }
.ident .name { flex: 1; border-bottom: 1px solid #444; height: 7mm; }
.instr { font-size: 9pt; background: #f2f2f2; border-radius: 2mm; padding: 2mm 3mm; line-height: 1.45; }
.guide { margin-top: 0.2mm; display: flex; align-items: center; gap: 2mm; line-height: 1.1; }
.guide svg { height: 5mm; direction: ltr; vertical-align: middle; }
.q { flex: 1; display: flex; flex-direction: column; border: 1px solid #bbb; border-radius: 2mm; padding: 2.5mm 3mm; min-height: 0; }
.qhead { font-weight: bold; font-size: 11pt; }
.qhead .pts { font-weight: normal; color: #444; font-size: 9.5pt; }
.qtext { font-size: 10.5pt; line-height: 1.45; margin: 1mm 0; }
.opts { font-size: 10.5pt; margin: 1mm 0; padding-right: 6mm; }
.opts div { margin: 0.6mm 0; }
.diagram { text-align: center; margin: 1mm 0; direction: ltr; }
.diagram svg { direction: ltr; }
.diagram svg { height: 37mm; width: auto; }
.work { flex: 1; min-height: 15mm; border: 1px dashed #bbb; border-radius: 1.5mm; margin: 1.5mm 0; position: relative; }
.work .hint { position: absolute; top: 1mm; right: 2mm; font-size: 8pt; color: #aaa; }
.work .sol { position: absolute; top: 5mm; right: 3mm; left: 3mm; font-size: 10pt; color: #0b4fb3; direction: rtl; }
.answer { display: flex; align-items: center; gap: 3mm; font-size: 10.5pt; }
.answer .lbl { font-weight: bold; }
.cells { display: flex; gap: 1.6mm; direction: ltr; }
.cell { width: 9mm; height: 11mm; border: 0.35mm solid #777; border-radius: 0.6mm; display: flex; align-items: center; justify-content: center;
        font-size: 18pt; color: #0b4fb3; font-family: 'DejaVu Sans', sans-serif; }
.cell.ex { color: #999; }
bdi { white-space: nowrap; }
.ans { display: flex; align-items: center; gap: 2.5mm; direction: ltr; }
.unit { direction: ltr; font-size: 12pt; min-width: 10mm; }
"""


def qr_data_uri(text: str) -> str:
    import qrcode
    img = qrcode.make(text, box_size=4, border=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_LTR_CH = r"0-9A-Za-zµΩρτ√½∥·⁻⁰¹²³⁴⁵⁶⁷⁸⁹.,=+\-/()°%²"
_LTR_RUN = re.compile(rf"[{_LTR_CH}]+(?:[ ]+[{_LTR_CH}]+)*")


def bidi(text: str) -> str:
    """escape, and isolate every left-to-right run (numbers with units, formulas) so that a Hebrew line
    shows '50 m', '1.72·10⁻⁸ Ω·m' and 'E1 = 10 V' in their own order"""
    out, last = [], 0
    for m in _LTR_RUN.finditer(text):
        a, b = m.span()
        run = text[a:b]
        while run and (run[-1] in ".,: (" or (run[-1] == ")" and run.count("(") < run.count(")"))):
            run, b = run[:-1], b - 1          # sentence punctuation and unmatched parentheses stay Hebrew
        while run and (run[0] in ".,: )" or (run[0] == "(" and run.count("(") > run.count(")"))):
            run, a = run[1:], a + 1
        if not re.search(r"[0-9A-Za-zµΩρτ]", run):
            continue
        out.append(html.escape(text[last:a]) + f'<bdi dir="ltr">{html.escape(run)}</bdi>')
        last = b
    out.append(html.escape(text[last:]))
    return "".join(out)


def cells_html(item: str, n: int, fill: str = "", cls: str = "") -> str:
    fill = list(fill)
    return '<span class="cells">' + "".join(
        f'<span class="cell {cls}" data-item="{item}" data-idx="{i}">{html.escape(fill[i]) if i < len(fill) else ""}</span>'
        for i in range(n)) + "</span>"


def fmt_answer(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if len(s) <= CELLS else f"{x:.3g}"


def question_html(q: dict, num: int, key: bool) -> str:
    parts = [f'<div class="q"><div class="qhead">שאלה {num} <span class="pts">({q["points"]} נקודות)</span></div>',
             f'<div class="qtext">{bidi(q["text"])}</div>']
    if q.get("diagram"):
        parts.append(f'<div class="diagram">{DIAGRAMS[q["diagram"]]()}</div>')
    if q["kind"] == "choice":
        parts.append('<div class="opts">' + "".join(f"<div>{i}. {bidi(o)}</div>"
                                                    for i, o in enumerate(q["options"], 1)) + "</div>")
    sol = f'<div class="sol">פתרון: {bidi(q["solution"])}</div>' if key and q.get("solution") else ""
    hint = "מקום לתשובה המילולית" if q["kind"] == "open" else "מקום לפתרון"
    parts.append(f'<div class="work"><span class="hint">{hint}</span>{sol}</div>')
    if q["kind"] == "number":
        fill = fmt_answer(q["answer"]) if key else ""
        parts.append(f'<div class="answer"><span class="lbl">תשובה סופית:</span><span class="ans">{cells_html(q["id"], CELLS, fill)}'
                     f'<span class="unit">{html.escape(q["unit"])}</span></span></div>')
    elif q["kind"] == "choice":
        fill = str(q["answer"]) if key else ""
        parts.append(f'<div class="answer"><span class="lbl">מספר התשובה הנכונה:</span>{cells_html(q["id"], 1, fill)}</div>')
    parts.append("</div>")
    return "".join(parts)


# how to write the digits (an optional line in the instructions: course "digit_guide": true, or
# python -m forms.make_exam --digit-guide): a plain '1' and a crossed '7' - the European '1' with a long
# flag is the readers' most common confident mistake (read as '7')
_GUIDE_PATHS = {
    "0": "M10,4 C3,4 3,26 10,26 C17,26 17,4 10,4 Z",
    "1": "M10,4 L10,26",
    "2": "M4.5,9 C4.5,2.5 15.5,2.5 15.5,9 C15.5,14 4.5,19 4,26 L16,26",
    "3": "M4.5,6 C9,1.5 16.5,4 15,9.5 C14,13 10,14 8,14 C12,14 16.5,16 16,21 C15.5,27 7,28 4,24.5",
    "4": "M13,26 L13,4 L3.5,19 L17,19",
    "5": "M15.5,4 L6,4 L5,13.5 C9,10.5 16.5,12 16.5,18.5 C16.5,25.5 7.5,27.5 4,24",
    "6": "M14.5,4 C6.5,6 4,14 4,19.5 C4,26 15.5,26.5 15.5,19.5 C15.5,13.5 6,13 4,19.5",
    "7": "M4,4 L16,4 L8,26 M6.5,15 L14.5,15",
    "8": "M10,14 C3.5,12.5 4,4 10,4 C16,4 16.5,12.5 10,14 C3,15.5 3,26 10,26 C17,26 17,15.5 10,14",
    "9": "M15.5,10 C15.5,3 4.5,3 4.5,10 C4.5,16.5 15.5,16.5 15.5,10 L14,26",
}


def digit_guide_svg() -> str:
    parts = [f'<g transform="translate({k * 22},0)"><path d="{d}" fill="none" stroke="#111" stroke-width="2.2" '
             f'stroke-linecap="round" stroke-linejoin="round"/></g>' for k, d in enumerate(_GUIDE_PATHS[str(i)] for i in range(10))]
    return f'<svg viewBox="0 0 218 30" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def exam_html(course: dict, ex: dict, key: bool = False, digit_guide: bool | None = None) -> str:
    if digit_guide is None:
        digit_guide = bool(course.get("digit_guide"))
    qs = {q["id"]: q for q in ex["questions"]}
    order = {q["id"]: i for i, q in enumerate(ex["questions"], 1)}
    m, ins = MARKER_MM, MARKER_INSET_MM
    pages = []
    n_pages = len(ex["pages"])
    for pno, qids in enumerate(ex["pages"], 1):
        body = []
        if pno == 1:
            body.append(
                f'<div class="head"><h1>{html.escape(course["name"])} ({course["name_en"]}) · {html.escape(ex["title"])}</h1>'
                f'<div class="sub">{html.escape(course["semester"])} · תאריך: {ex["date"]} · משך: {ex["minutes"]} דקות · '
                f'{"<b>פתרון</b>" if key else "חומר עזר: מחשבון"}</div>'
                f'<div class="ident"><span>שם:</span><span class="name"></span><span>תעודת זהות:</span>'
                f'{cells_html("id", 9, "" if not key else "")}</div></div>'
                '<div class="instr">כתבו את התשובה הסופית של כל שאלה בתיבת התשובה: <b>תו אחד בכל תא</b> (ספרה, נקודה '
                'עשרונית או סימן מינוס), משמאל לימין, צמוד לתא הראשון. היחידה כבר מודפסת, אין לכתוב אותה. דייקו לשתי '
                'ספרות אחרי הנקודה (אם צריך). לדוגמה: ' + cells_html("example", 5, "-12.5", "ex") + ' &nbsp;בטעות: '
                'העבירו קו על כל התיבה וכתבו את התשובה בבירור לידה (תיבדק ידנית).'
                + ('<div class="guide"><span>כתבו את הספרות כך - <b>1 כקו ישר, 7 עם קו חוצה</b>:</span>'
                   f'<bdi dir="ltr">{digit_guide_svg()}</bdi></div>' if digit_guide else '') + '</div>')
        for qid in qids:
            body.append(question_html(qs[qid], order[qid], key))
        corners = "".join(f'<div class="marker" data-marker="{k}" style="{v}"></div>' for k, v in (
            ("tl", f"top:{ins}mm;left:{ins}mm"), ("tr", f"top:{ins}mm;right:{ins}mm"),
            ("bl", f"bottom:{ins}mm;left:{ins}mm"), ("br", f"bottom:{ins}mm;right:{ins}mm")))
        qr = qr_data_uri(f"{course['id']}|{ex['id']}|{pno}")
        pages.append(f'<div class="page" data-page="{pno}">{corners}<div class="qr" data-qr="{pno}"><img src="{qr}"></div>'
                     f'<div class="foot">{html.escape(course["name"])} · {html.escape(ex["title"])} · עמוד {pno} מתוך {n_pages}</div>'
                     f'<div class="foot l">{course["id"]} {ex["id"]} p{pno}</div>'
                     f'<div class="content">{"".join(body)}</div></div>')
    css = CSS % {"m": m}
    return f'<!doctype html><html lang="he"><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(pages)}</body></html>'


JS_LAYOUT = """() => {
  const out = {pages: [], cells: [], markers: [], qr: []};
  const pages = Array.from(document.querySelectorAll('.page'));
  pages.forEach((pg, i) => {
    const P = pg.getBoundingClientRect();
    out.pages.push({page: i + 1, w: P.width, h: P.height});
    const rel = (el) => { const r = el.getBoundingClientRect(); return [r.left - P.left, r.top - P.top, r.right - P.left, r.bottom - P.top]; };
    pg.querySelectorAll('.cell').forEach(c => out.cells.push({page: i + 1, item: c.dataset.item, idx: +c.dataset.idx, box: rel(c)}));
    pg.querySelectorAll('.marker').forEach(m => out.markers.push({page: i + 1, k: m.dataset.marker, box: rel(m)}));
    pg.querySelectorAll('.qr').forEach(q => out.qr.push({page: i + 1, box: rel(q)}));
  });
  return out;
}"""


def render(html_text: str, pdf_path: str):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        pg = b.new_page()
        pg.set_content(html_text, wait_until="load")
        lay = pg.evaluate(JS_LAYOUT)
        overflow = pg.evaluate("() => Array.from(document.querySelectorAll('.content')).map(c => c.scrollHeight > c.clientHeight + 1)")
        # a question's block must hold its answer row: an overflowing row would sit on the block's border line
        q_over = pg.evaluate("""() => Array.from(document.querySelectorAll('.q')).map(q => {
            const a = q.querySelector('.answer'); if (!a) return 0;
            return Math.round(a.getBoundingClientRect().bottom - q.getBoundingClientRect().bottom + 4); })""")
        pg.pdf(path=pdf_path, width="210mm", height="297mm", print_background=True, prefer_css_page_size=True,
               margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    if any(overflow):
        raise RuntimeError(f"{pdf_path}: content does not fit on page(s) {[i + 1 for i, o in enumerate(overflow) if o]}")
    if any(v > 0 for v in q_over):
        raise RuntimeError(f"{pdf_path}: answer row outside its question block (px over: {q_over})")
    return lay


def to_template(lay: dict) -> dict:
    sc = lambda b: [round(v * PX, 1) for v in b]  # noqa: E731
    tpl = {"dpi": DPI, "page_px": [round(lay["pages"][0]["w"] * PX), round(lay["pages"][0]["h"] * PX)], "pages": {}}
    for pg in lay["pages"]:
        n = pg["page"]
        tpl["pages"][str(n)] = {
            "markers": {m["k"]: [round((m["box"][0] + m["box"][2]) / 2 * PX, 1), round((m["box"][1] + m["box"][3]) / 2 * PX, 1)]
                        for m in lay["markers"] if m["page"] == n},
            "marker_px": round((lay["markers"][0]["box"][2] - lay["markers"][0]["box"][0]) * PX, 1),
            "qr": [sc(q["box"]) for q in lay["qr"] if q["page"] == n][0],
            "items": {}}
        for c in lay["cells"]:
            if c["page"] == n and c["item"] != "example":
                tpl["pages"][str(n)]["items"].setdefault(c["item"], []).append(sc(c["box"]))
    return tpl


def build(course: dict, ex: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{ex['id']}.pdf")
    lay = render(exam_html(course, ex), pdf)
    render(exam_html(course, ex, key=True), os.path.join(out_dir, f"{ex['id']}_key.pdf"))
    spec = {"course": course, **ex, "template": to_template(lay)}
    with open(os.path.join(out_dir, f"{ex['id']}.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
    return spec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "course_elec101"),
                    help="where the PDFs and templates go (a new folder for a trial: the simulated classes of a course "
                         "are tied to its templates)")
    ap.add_argument("--digit-guide", action="store_true", help="print how to write the digits (plain 1, crossed 7)")
    ap.add_argument("--exams", default="", help="only these exams, e.g. E1")
    a = ap.parse_args()
    course = {**elec101.COURSE, **({"digit_guide": True} if a.digit_guide else {})}
    exams = [e for e in elec101.all_exams() if not a.exams or e["id"] in a.exams.split(",")]
    for ex in exams:
        spec = build(course, ex, a.out)
        n = sum(len(v) for p in spec["template"]["pages"].values() for v in p["items"].values())
        print(ex["id"], ex["title"], f"{len(ex['pages'])} pages, {n} cells")
    with open(os.path.join(a.out, "course.json"), "w", encoding="utf-8") as fh:
        json.dump({**course, "exams": [e["id"] for e in exams]}, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
