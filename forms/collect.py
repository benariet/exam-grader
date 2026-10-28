"""Handwriting collection sheets: people COPY printed numbers into answer cells (and printed sentences
onto lines), so every cell and line comes back already labelled - real handwriting of our own
students and colleagues for the cell reader (and the Hebrew / English line readers), without anyone
typing a transcription.

    python -m forms.collect make                 # -> data/collect/sheet_v1..v4.pdf (+ .json templates)
    python -m forms.collect read scans/*.pdf     # -> data/collect/real/<scan>/cells/*.png + labels.jsonl
                                                 #    and lines/*.png + lines.jsonl
    python -m forms.cells train --real data/collect/real ...   # fine-tune with them (see forms/cells.py)

A sheet is two pages with the exam forms' corner markers and a QR code ('collect|v2|1'). Page 1: 16
printed numbers, each with 8 cells to copy it into (one character per cell, from the first cell);
page 2: 8 more numbers, and Hebrew and English sentences to copy onto the line below each. Four
versions with different numbers - every digit, the decimal point and the minus appear often.

Reading checks the copying itself, not the handwriting: a row is kept only when exactly the first
len(number) cells carry ink (a skipped or shifted cell would put every label in that row out of
place); everything else is dropped and counted.
"""
from __future__ import annotations

import html
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from forms.make_exam import CSS, DPI, MARKER_INSET_MM, MARKER_MM, PX, bidi, cells_html, qr_data_uri  # noqa: E402

OUT = os.path.join(HERE, "data", "collect")
COURSE_ID = "collect"
N_CELLS = 8
HEBREW = ["המתח על הנגד שווה למכפלת הזרם בהתנגדות",
          "בחיבור טורי עובר אותו זרם בכל הנגדים",
          "הקבל נטען עד שהזרם במעגל מתאפס",
          "סכום הזרמים הנכנסים לצומת שווה לסכום היוצאים",
          "ההספק המתפתח בנגד גדל עם ריבוע הזרם",
          "מפסק הפחת מנתק כשיש זליגה של זרם לאדמה"]
ENGLISH = ["Power equals voltage times current.",
           "The same current flows through resistors in series.",
           "Kirchhoff's voltage law follows from energy conservation.",
           "A charged capacitor blocks direct current."]

SHEET_CSS = CSS + """
.row { display: flex; align-items: center; gap: 6mm; margin: 1.1mm 0; }
.target { direction: ltr; font-family: 'DejaVu Sans Mono', monospace; font-size: 15pt; width: 34mm; text-align: left;
          background: #f2f2f2; border-radius: 1.5mm; padding: 1mm 2mm; }
.rows { display: flex; flex-direction: column; }
.copy { margin: 1.5mm 0 3mm 0; }
.copy .src { font-size: 12pt; }
.copy .src.en { direction: ltr; text-align: left; }
.copyline { height: 12mm; border-bottom: 0.35mm solid #999; }
"""


def numbers_for(version: int, n: int = 24):
    """numbers to copy: integers, decimals and negatives, every character frequent"""
    rng = np.random.default_rng(1000 + version)
    out = []
    while len(out) < n:
        kind = rng.choice(["int", "dec", "dec", "neg", "small"])
        if kind == "int":
            s = str(int(rng.integers(1, 10)) * 10 ** int(rng.integers(0, 3)) + int(rng.integers(0, 1000)))
        elif kind == "dec":
            s = f"{rng.uniform(0, 999):.{int(rng.integers(1, 4))}f}"
        elif kind == "neg":
            s = f"-{rng.uniform(0, 99):.{int(rng.integers(0, 3))}f}"
        else:
            s = f"0.{int(rng.integers(0, 10))}{int(rng.integers(0, 10))}{int(rng.integers(1, 10))}"
        if len(s) <= N_CELLS and s not in out:
            out.append(s)
    return out


def sheet_html(version: int):
    nums = numbers_for(version)
    m, ins = MARKER_MM, MARKER_INSET_MM
    pages = []
    for pno in (1, 2):
        body = []
        if pno == 1:
            body.append('<div class="head"><h1>דף איסוף כתב יד · גרסה ' + str(version) + '</h1>'
                        '<div class="sub">לאימון הכלי שקורא תשובות בתאים. אין לכתוב שם או פרטים מזהים.</div></div>'
                        '<div class="instr">העתיקו כל מספר מודפס אל התאים שלידו: <b>תו אחד בכל תא</b> (ספרה, נקודה '
                        'עשרונית או מינוס), משמאל לימין, מהתא הראשון, בכתב היד הרגיל שלכם. בעמוד 2 העתיקו גם כל משפט '
                        'אל הקו שמתחתיו. אם טעיתם, אל תתקנו: דלגו לשורה הבאה.</div>')
            rows = nums[:16]
        else:
            rows = nums[16:]
        body.append('<div class="rows">' + "".join(
            f'<div class="row"><span class="target">{html.escape(s)}</span>'
            f'<span class="ans">{cells_html(f"n{(0 if pno == 1 else 16) + i}", N_CELLS)}</span></div>'
            for i, s in enumerate(rows)) + '</div>')
        if pno == 2:
            heb = [HEBREW[(version + k) % len(HEBREW)] for k in range(3)]
            eng = [ENGLISH[(version + k) % len(ENGLISH)] for k in range(2)]
            for k, t in enumerate(heb):
                body.append(f'<div class="copy"><div class="src">{bidi(t)}</div><div class="copyline" data-line="he{k}"></div></div>')
            for k, t in enumerate(eng):
                body.append(f'<div class="copy"><div class="src en">{html.escape(t)}</div><div class="copyline" data-line="en{k}"></div></div>')
        corners = "".join(f'<div class="marker" data-marker="{k}" style="{v}"></div>' for k, v in (
            ("tl", f"top:{ins}mm;left:{ins}mm"), ("tr", f"top:{ins}mm;right:{ins}mm"),
            ("bl", f"bottom:{ins}mm;left:{ins}mm"), ("br", f"bottom:{ins}mm;right:{ins}mm")))
        qr = qr_data_uri(f"{COURSE_ID}|v{version}|{pno}")
        pages.append(f'<div class="page" data-page="{pno}">{corners}<div class="qr"><img src="{qr}"></div>'
                     f'<div class="foot">דף איסוף כתב יד · גרסה {version} · עמוד {pno} מתוך 2</div>'
                     f'<div class="foot l">collect v{version} p{pno}</div><div class="content">{"".join(body)}</div></div>')
    doc = (f'<!doctype html><html lang="he"><head><meta charset="utf-8"><style>{SHEET_CSS % {"m": m}}</style></head>'
           f'<body>{"".join(pages)}</body></html>')
    labels = {f"n{i}": s for i, s in enumerate(nums)}
    labels.update({f"he{k}": HEBREW[(version + k) % len(HEBREW)] for k in range(3)})
    labels.update({f"en{k}": ENGLISH[(version + k) % len(ENGLISH)] for k in range(2)})
    return doc, labels


JS = """() => {
  const out = {pages: [], cells: [], markers: [], qr: [], lines: []};
  Array.from(document.querySelectorAll('.page')).forEach((pg, i) => {
    const P = pg.getBoundingClientRect();
    out.pages.push({page: i + 1, w: P.width, h: P.height});
    const rel = (el) => { const r = el.getBoundingClientRect(); return [r.left - P.left, r.top - P.top, r.right - P.left, r.bottom - P.top]; };
    pg.querySelectorAll('.cell').forEach(c => out.cells.push({page: i + 1, item: c.dataset.item, idx: +c.dataset.idx, box: rel(c)}));
    pg.querySelectorAll('.marker').forEach(m => out.markers.push({page: i + 1, k: m.dataset.marker, box: rel(m)}));
    pg.querySelectorAll('.qr').forEach(q => out.qr.push({page: i + 1, box: rel(q)}));
    pg.querySelectorAll('.copyline').forEach(l => out.lines.push({page: i + 1, item: l.dataset.line, box: rel(l)}));
  });
  return out;
}"""


def make(versions=(1, 2, 3, 4), out=OUT):
    from playwright.sync_api import sync_playwright
    from forms.make_exam import to_template
    os.makedirs(out, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
        for v in versions:
            doc, labels = sheet_html(v)
            pg = b.new_page()
            pg.set_content(doc, wait_until="load")
            lay = pg.evaluate(JS)
            over = pg.evaluate("() => Array.from(document.querySelectorAll('.content')).map(c => c.scrollHeight > c.clientHeight + 1)")
            if any(over):
                raise RuntimeError(f"sheet v{v} overflows")
            pdf = os.path.join(out, f"sheet_v{v}.pdf")
            pg.pdf(path=pdf, width="210mm", height="297mm", print_background=True, prefer_css_page_size=True,
                   margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            pg.close()
            tpl = to_template(lay)
            for ln in lay["lines"]:
                tpl["pages"][str(ln["page"])].setdefault("lines", {})[ln["item"]] = [round(x * PX, 1) for x in ln["box"]]
            with open(os.path.join(out, f"sheet_v{v}.json"), "w", encoding="utf-8") as fh:
                json.dump({"version": v, "labels": labels, "template": tpl}, fh, ensure_ascii=False, indent=1)
            print("sheet", v, {k: labels[k] for k in list(labels)[:6]}, "...")
        b.close()


# ------------------------------------------------------------------ reading filled sheets
def load_sheets(out=OUT):
    sheets = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("sheet_v") and f.endswith(".json"):
            s = json.load(open(os.path.join(out, f), encoding="utf-8"))
            sheets[s["version"]] = s
    return sheets


def identify(img, sheets):
    """-> (version, page, straightened page) or (None, None, None)"""
    import re
    from forms.read_form import read_qr, straighten
    tpl = next(iter(sheets.values()))["template"]
    page = straighten(img, tpl["pages"]["1"], tpl["page_px"])
    if page is None:
        return None, None, None
    for p in (page, np.ascontiguousarray(page[::-1, ::-1])):
        txt, _ = read_qr(p, tpl["pages"]["1"]["qr"])
        m = re.fullmatch(r"collect\|v(\d+)\|(\d)", txt or "")
        if m and int(m.group(1)) in sheets:
            return int(m.group(1)), int(m.group(2)), p
    return None, None, None


def read_filled(images, name: str, out_root=os.path.join(OUT, "real"), sheets=None):
    """cut the labelled cells and lines out of filled sheets (the pages of one person, any order)"""
    from PIL import Image
    from forms.read_form import CELL_SHRINK, cell_crops, refine_row, stray_ink
    sheets = sheets or load_sheets()
    d = os.path.join(out_root, name)
    os.makedirs(os.path.join(d, "cells"), exist_ok=True)
    os.makedirs(os.path.join(d, "lines"), exist_ok=True)
    stats = {"rows": 0, "rows_kept": 0, "cells": 0, "lines": 0, "pages_not_found": 0, "people": 0}
    cell_rows, line_rows = [], []
    person, last_v, seen = -1, None, set()
    for k, img in enumerate(images):
        v, pno, page = identify(img, sheets)
        if v is None:
            stats["pages_not_found"] += 1
            continue
        # one person's sheet = its pages in a row (as a scanner feeds them): a new person begins with a
        # page of another version, or a page this person already has
        if v != last_v or pno in seen:
            person, seen = person + 1, set()
        last_v = v
        seen.add(pno)
        who = f"{name}#{person + 1}"
        sh = sheets[v]
        tpl = sh["template"]["pages"][str(pno)]
        for item, boxes in tpl["items"].items():
            target = sh["labels"][item]
            stats["rows"] += 1
            boxes, _ = refine_row(page, boxes)
            crops = cell_crops(page, boxes, CELL_SHRINK)
            written = [stray_ink(c) >= 15 for c in crops]
            if written != [i < len(target) for i in range(len(crops))]:
                continue                                  # skipped / shifted / extra cells: labels unsafe
            stats["rows_kept"] += 1
            for i, c in enumerate(crops):
                lab = target[i] if i < len(target) else " "
                fn = f"img{k + 1:03d}_v{v}_{item}_{i}.png"      # unique also when many people share one PDF
                Image.fromarray(c).save(os.path.join(d, "cells", fn))
                cell_rows.append({"file": f"cells/{fn}", "label": lab, "person": who})
                stats["cells"] += 1
        for item, box in tpl.get("lines", {}).items():
            x0, y0, x1, y1 = [int(round(t)) for t in box]
            crop = page[max(0, y0 - 6): y1 + 6, x0: x1]
            fn = f"img{k + 1:03d}_v{v}_{item}.png"
            Image.fromarray(crop).save(os.path.join(d, "lines", fn))
            line_rows.append({"file": f"lines/{fn}", "text": sh["labels"][item], "lang": item[:2], "person": who})
            stats["lines"] += 1
    stats["people"] = person + 1
    with open(os.path.join(d, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for r in cell_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "lines.jsonl"), "w", encoding="utf-8") as fh:
        for r in line_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # the copied Hebrew sentences in the Hebrew trainer's format (train_hebrew.py --extra <this file>)
    with open(os.path.join(d, "hebrew_lines.json"), "w", encoding="utf-8") as fh:
        json.dump([{"file": r["file"], "text": r["text"], "person": r["person"]} for r in line_rows if r["lang"] == "he"],
                  fh, ensure_ascii=False, indent=1)
    return stats


def load_real(root=os.path.join(OUT, "real"), people=None, with_people=False):
    """collected cells: (list of crops, list of label indices[, list of person ids]); people: keep only
    these persons ('<scan>#<n>' as read_filled names them)"""
    from PIL import Image
    from forms.cells import CLASSES
    crops, labels, who = [], [], []
    if not os.path.isdir(root):
        return (crops, labels, who) if with_people else (crops, labels)
    for name in sorted(os.listdir(root)):
        f = os.path.join(root, name, "labels.jsonl")
        if not os.path.isfile(f):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            pid = r.get("person", name)
            if people is not None and pid not in people:
                continue
            crops.append(np.asarray(Image.open(os.path.join(root, name, r["file"])).convert("L")))
            labels.append(CLASSES.index(r["label"]))
            who.append(pid)
    return (crops, labels, who) if with_people else (crops, labels)


def simulate_filled(version, seed, digit_source="semeion", out_dir=os.path.join(OUT, "sim"), own=None):
    """a sheet filled by a simulated person (for testing the reading); -> list of page images.
    own: indices (into digit_source) of one real writer's digits - the person writes with them"""
    import subprocess
    from PIL import Image, ImageDraw
    from forms.cells import load_source
    from forms.simulate import Writer, degrade
    rng = np.random.default_rng(seed)
    sheets = load_sheets()
    sh = sheets[version]
    os.makedirs(out_dir, exist_ok=True)
    pre = os.path.join(out_dir, f"_blank_v{version}")
    if not os.path.isfile(pre + "-1.png"):
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", os.path.join(OUT, f"sheet_v{version}.pdf"), pre], check=True)
    digits, labels = load_source(digit_source)
    w = Writer(rng, digits, labels, own=own)
    imgs = []
    for pno in (1, 2):
        page = np.asarray(Image.open(f"{pre}-{pno}.png").convert("RGB")).copy()
        tpl = sh["template"]["pages"][str(pno)]
        for item, boxes in tpl["items"].items():
            s = sh["labels"][item]
            if rng.random() < 0.05:                        # now and then a row copied one cell late
                boxes = boxes[1:]
            for i, ch in enumerate(s[:len(boxes)]):
                w.write(page, boxes[i], ch)
        im = Image.fromarray(page)
        d = ImageDraw.Draw(im)
        for item, box in tpl.get("lines", {}).items():   # a scribble standing in for the copied sentence
            x0, y0, x1, y1 = box
            pts = [(x1 - 20 - k * 14, y1 - 12 - 8 * np.sin(k)) for k in range(int((x1 - x0) / 14) - 4)]
            d.line(pts, fill=w.color, width=3)
        imgs.append(degrade(im, rng, "scan"))
    return imgs


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) > 2 and sys.argv[1] == "read":
        from forms.read_form import load_images
        for p in sys.argv[2:]:
            name = os.path.splitext(os.path.basename(p))[0]
            print(name, read_filled(load_images([p]), name))
    else:
        print(__doc__)
