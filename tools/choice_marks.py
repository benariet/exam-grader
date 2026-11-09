"""The answer students mark in a multiple-choice question printed in the booklet (the course's Moed C 2.2 and 2.3: six
options, א to ו, 'mark one answer only'), found on the scans and checked against the solution's letter - on this
computer (CPU only: the booklet's template, no reader).

The options' printed lines are found on the booklet's template of the question's page - the lines right above the
answer box (its ruled top) - grouped into options by how many lines each takes; laid on each scan (recog/printed.py's
map), the print left out, the handwriting over each option's lines counted: one option with clearly the most (a circle
round its letter, a tick, an underline) is the one chosen; none - nothing marked on the options (the answer may be
written in the box: to a human); two or more alike - unclear (a choice changed, options crossed out: to a human).

    python tools/choice_marks.py "<exams' PDFs>" --booklet booklet/moedC --item "2.2:17:1,1,1,1,1,1:ג" \\
        --item "2.3:19:2,2,2,2,2,2:ד" --question 2 --grades "grades.xlsx" --out <folder> [--review]
        item: name : the booklet's page : lines per option, top to bottom : the solution's letter
        -> how many chose the solution's letter, another, none, unclear (beside the points the lecturer took off the
           question); <out>/<n>.choices.json per exam, <n>_choice_remarks.txt (Hebrew)
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

LETTERS = "אבגדהוזחטי"
MIN_MARK = 30           # handwriting pixels (150 dpi) marking an option: less, no mark
CLEAR = 2.5             # the most marked option: this many times the next, or the choice is unclear


def printed_lines(template: np.ndarray):
    """the template's printed rows, top to bottom -> [(y0, y1, x0, x1, kind)]: kind 'text' (a line of print) or 'rule'
    (a ruled line across most of the page, or a box's sides - a run much taller than text)"""
    prof = template.sum(1)
    out, y0 = [], None
    W = template.shape[1]
    for y, v in enumerate(np.r_[prof, 0]):
        if v > 3 and y0 is None:
            y0 = y
        elif v <= 3 and y0 is not None:
            xs = np.flatnonzero(template[y0:y].any(0))
            h, w = y - y0, xs.max() - xs.min() + 1
            if h >= 6 or w > 0.6 * W:
                kind = "rule" if (h < 12 and w > 0.6 * W) or h > 80 else "text" if h <= 60 else "other"
                out.append((y0, y, int(xs.min()), int(xs.max()), kind))
            y0 = None
    return out


def option_bands(template: np.ndarray, lines_per_option):
    """the options' bands on the template -> [(x0, y0, x1, y1, [its lines' (y0, y1)], its letter's box)] top to bottom:
    the text lines
    right above the first rule below the question (the answer box's top), as many as the options take, grouped; each
    band reaching half way to its neighbours; [] if there are not that many"""
    rows = printed_lines(template)
    n = sum(lines_per_option)
    W = template.shape[1]
    for i, r in enumerate(rows):
        if r[4] == "rule" and i >= n:
            before = rows[:i]
            texts = [t for t in before if t[4] == "text"]
            if len(texts) < n:
                continue
            opts = texts[-n:]
            bands, k = [], 0
            for m in lines_per_option:
                grp = opts[k:k + m]
                k += m
                bands.append([min(g[2] for g in grp), grp[0][0], max(g[3] for g in grp), grp[-1][1],
                              [(g[0], g[1]) for g in grp]])
            for j, b in enumerate(bands):              # (half way to the neighbours; the ends a little out - not
                top = (bands[j - 1][3] + b[1]) // 2 if j else b[1] - 8          # into the box below)
                bot = (b[3] + bands[j + 1][1]) // 2 if j + 1 < len(bands) else min(b[3] + 8, r[0] - 3)
                b[1], b[3] = top, bot
            from scipy import ndimage as ndi
            for b in bands:                            # (each option's letter: the print at its first line's right
                y0, y1 = b[4][0]                       #  end - 'א.')
                lab, _ = ndi.label(template[y0:y1 + 1], structure=np.ones((3, 3)))
                parts = [(sl[1].start, sl[0].start + y0, sl[1].stop, sl[0].stop + y0) for sl in ndi.find_objects(lab)]
                right = max(p[2] for p in parts)
                mine = [p for p in parts if p[0] >= right - 28] or [max(parts, key=lambda p: p[2])]
                b.append((min(p[0] for p in mine), min(p[1] for p in mine), right, max(p[3] for p in mine)))
            for b in bands:                            # (the whole width right of the text: a tick past the letter)
                b[0], b[2] = max(0, b[0] - 40), min(W - 1, b[2] + 120)
            return [tuple(int(v) for v in b[:4]) + (b[4], b[5]) for b in bands]
    return []


def choose(scores, min_mark=MIN_MARK, clear=CLEAR, rings=None):
    """handwriting per option (and which have their letter circled) -> (index or None, 'chosen' | 'none' |
    'unclear'): one letter circled - that one, whatever else is written by the others (a student weighing each option
    marks each); else the option with clearly the most, the others nearly unmarked"""
    if rings is not None and sum(rings) == 1:
        return int(np.flatnonzero(rings)[0]), "chosen"
    order = np.argsort(scores)[::-1]
    best = scores[order[0]]
    if best < min_mark:
        return None, "none"
    second = scores[order[1]] if len(scores) > 1 else 0
    if best >= clear * second and second < min_mark or best >= 2 * clear * second:   # (two options marked - a
        return int(order[0]), "chosen"                 #  choice changed, one crossed out: which is which, a human's)
    return None, "unclear"


def circled(hand: np.ndarray, box, min_px: int = 12) -> bool:
    """a loop drawn round the letter in box (on the scan): handwriting on all four sides of it, at about one distance
    from its middle (an X's or a tick's strokes run out from it, a ring keeps round it)"""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r0 = max(x1 - x0, y1 - y0) / 2
    m = int(max(14, 1.1 * r0))
    X0, Y0 = max(0, int(x0) - m), max(0, int(y0) - m)
    sub = hand[Y0:int(y1) + m, X0:int(x1) + m].copy()
    ly0, lx0 = max(0, int(y0) - 1 - Y0), max(0, int(x0) - 1 - X0)
    sub[ly0:int(y1) + 2 - Y0, lx0:int(x1) + 2 - X0] = False   # (the letter's own place: print, a stroke over it)
    ys, xs = np.nonzero(sub)
    if len(ys) < min_px:
        return False
    dx, dy = xs + X0 - cx, ys + Y0 - cy
    ang = np.arctan2(dy, dx)
    sides = [np.sum(np.abs(np.angle(np.exp(1j * (ang - a)))) < np.pi / 4) for a in (0, np.pi / 2, np.pi, -np.pi / 2)]
    if min(sides) < 3:
        return False
    ex, ey = dx / max(1.0, (x1 - x0) / 2 + 4), dy / max(1.0, (y1 - y0) / 2 + 4)   # (an ellipse round a wide letter)
    r = np.hypot(ex, ey)
    return float(r.std() / max(r.mean(), 1e-6)) < 0.3


def read_choices(page, template, bands):
    """a scan's page (RGB) -> (handwriting pixels marking each option, the bands laid on the scan, the handwriting):
    each mark (a piece of handwriting over the options) is the option's whose line its middle is nearest - a flat one
    (an underline) the line's right above it - so a circle round a letter, taller than a line, counts whole for it"""
    from scipy import ndimage as ndi
    from recog.printed import fit_map, printed_on
    from tools.expr_box import soft_hand
    from tools.truth_table import to_scan
    gray = np.asarray(page.convert("L"))
    ink = gray < 128
    M, b = fit_map(template, ink)
    hand = soft_hand(gray, printed_on(template, ink))
    on_scan = [to_scan(bd[:4], M, b) for bd in bands]
    lines = []                                         # [(option, y0, y1)] on the scan
    for k, bd in enumerate(bands):
        for y0, y1 in bd[4]:                           # (at the lines' right ends, where the letters are: a
            q = to_scan((bd[2] - 120, y0, bd[2] - 120, y1), M, b)   # tilted page's box would be taller)
            lines.append((k, q[1], q[3]))
    lh = float(np.median([y1 - y0 for _, y0, y1 in lines]))
    X0, X1 = min(q[0] for q in on_scan), max(q[2] for q in on_scan)
    Y0, Y1 = on_scan[0][1], on_scan[-1][3]
    region = np.zeros_like(hand)
    region[max(0, Y0):Y1, max(0, X0):X1] = True
    mine = hand & region
    lab, n = ndi.label(ndi.binary_dilation(mine, iterations=4), structure=np.ones((3, 3)))   # (a circle cut where
    lab[~mine] = 0                                                                               #  it crossed print)
    comps = []                                         # [top, bottom, x0, x1, pixels, sum of rows]
    for j, sl in enumerate(ndi.find_objects(lab)):
        if sl is None:
            continue
        m = lab[sl] == j + 1
        size = int(m.sum())
        if size >= 6:
            comps.append([sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, size,
                          float((np.nonzero(m)[0] + sl[0].start).sum())])
    merged = True                                      # (a circle's arcs over and under a line of print - cut where
    while merged:                                      #  it crossed the print - one mark again)
        merged = False
        for ca in comps:
            for cb in comps:
                if ca is cb or cb[0] < ca[1]:
                    continue
                ov = min(ca[3], cb[3]) - max(ca[2], cb[2])
                between = any(ca[1] <= y0 + 6 and cb[0] >= y1 - 6 for _, y0, y1 in lines)
                arc = min(ca[1] - ca[0], cb[1] - cb[0]) <= 0.45 * lh      # (one of them a flat piece - an arc; two
                if arc and ov >= 0.6 * min(ca[3] - ca[2], cb[3] - cb[2]) and \
                        (cb[0] - ca[1] <= 10 or (cb[0] - ca[1] <= 1.3 * lh and between)):   # marks of their own: not)
                    ca[:] = [ca[0], max(ca[1], cb[1]), min(ca[2], cb[2]), max(ca[3], cb[3]), ca[4] + cb[4], ca[5] + cb[5]]
                    comps.remove(cb)
                    merged = True
                    break
            if merged:
                break
    scores = np.zeros(len(bands))
    for top, bot, x0, x1, size, ysum in comps:
        h, w = bot - top, x1 - x0
        cy = ysum / size
        inside = [k for k, y0, y1 in lines if y0 + 2 <= cy <= y1 - 2]
        if h <= 0.45 * lh and w >= 2.5 * h and not inside:     # (an underline: the line above it)
            above = [(cy - y1, k) for k, y0, y1 in lines if y1 <= cy + 3]
            owner = min(above)[1] if above else min((abs(cy - (y0 + y1) / 2), k) for k, y0, y1 in lines)[1]
        else:                                          # (the line it reaches into, or comes nearest)
            mid = (top + bot) / 2
            owner = min((max(0, top - y1, y0 - bot), abs(mid - (y0 + y1) / 2), k) for k, y0, y1 in lines)[2]
        scores[owner] += size
    rings = [circled(hand, to_scan(bd[5], M, b)) for bd in bands]
    return scores, on_scan, hand, rings


def _review_figure(img, on_scan, scores, pick, verdict, item, name):
    import base64
    import html
    import io
    from PIL import ImageDraw
    x0 = max(0, min(b[0] for b in on_scan) - 20)
    y0 = max(0, min(b[1] for b in on_scan) - 20)
    x1 = min(img.width, max(b[2] for b in on_scan) + 20)
    y1 = min(img.height, max(b[3] for b in on_scan) + 20)
    crop = img.crop((x0, y0, x1, y1)).copy()
    d = ImageDraw.Draw(crop)
    for k, bd in enumerate(on_scan):
        col = (20, 160, 60) if k == pick else (150, 150, 150)
        d.rectangle((bd[0] - x0, bd[1] - y0, bd[2] - x0, bd[3] - y0), outline=col, width=3 if k == pick else 1)
        d.text((bd[0] - x0 + 4, bd[1] - y0 + 2), str(int(scores[k])), fill=col)
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    heb = {"right": "נבחרה התשובה הנכונה", "wrong": "נבחרה תשובה אחרת", "none": "לא סומנה תשובה",
           "unclear": "לא ברור מה סומן"}
    cap = f"מבחן {name}, שאלה {item}: {heb[verdict]}" + (f" ({LETTERS[pick]})" if pick is not None else "")
    return (f"<figure><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
            f"<figcaption>{html.escape(cap)}</figcaption></figure>")


def parse_item(s):
    name, page, lines, letter = s.split(":")
    return {"name": name, "page": int(page), "lines": [int(x) for x in lines.split(",")], "answer": letter.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--booklet", required=True)
    ap.add_argument("--item", action="append", required=True, help="name:page:lines per option:the solution's letter")
    ap.add_argument("--question", type=int, default=0, help="the items' question, for the lecturer's points off")
    ap.add_argument("--grades", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    import pypdfium2 as pdfium
    from PIL import Image
    from recog.page import clean_scan
    items = [parse_item(s) for s in a.item]
    for it in items:
        tpl = np.asarray(Image.open(os.path.join(a.booklet, f"p{it['page']}.png")).convert("L")) < 128
        it["template"] = tpl
        it["bands"] = option_bands(tpl, it["lines"])
        if len(it["bands"]) != len(it["lines"]):
            raise SystemExit(f"item {it['name']}: {len(it['lines'])} options not found on p{it['page']}.png")
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
        doc = pdfium.PdfDocument(pdf)
        out, remarks = [], []
        for it in items:
            if len(doc) < it["page"]:
                continue
            img = doc[it["page"] - 1].render(scale=150 / 72).to_pil().convert("RGB")
            page, _ = clean_scan(img, answer_boxes=False)    # (a circle round a letter is no printed box)
            scores, on_scan, _, rings = read_choices(page, it["template"], it["bands"])
            pick, how = choose(scores, rings=rings)
            letter = LETTERS[pick] if pick is not None else ""
            verdict = how if how != "chosen" else ("right" if letter == it["answer"] else "wrong")
            out.append({"item": it["name"], "verdict": verdict, "chosen": letter, "answer": it["answer"],
                        "ink": [int(s) for s in scores], "circled": [LETTERS[k] for k, r in enumerate(rings) if r]})
            remarks.append({"right": f"שאלה {it['name']}: סומנה התשובה הנכונה ({letter}) - יש לבדוק את ההסבר.",
                            "wrong": f"שאלה {it['name']}: סומנה תשובה {letter}; בפתרון {it['answer']}.",
                            "none": f"שאלה {it['name']}: לא נמצא סימון על התשובות - לבדיקה ידנית.",
                            "unclear": f"שאלה {it['name']}: לא ברור איזו תשובה סומנה - לבדיקה ידנית."}[verdict])
            if a.review:
                review.append(_review_figure(img, on_scan, scores, pick, verdict, it["name"], name))
        json.dump({"exam": name, "items": out}, open(os.path.join(a.out, f"{name}.choices.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        open(os.path.join(a.out, f"{name}_choice_remarks.txt"), "w", encoding="utf-8").write("\n".join(remarks) + "\n")
        table.append((out, marks.get(name, {}).get(a.question) if marks and a.question else None))
    if a.review:
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(
            ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Choices review</title>",
             "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}</style>",
             "<h1>שאלות אמריקאיות - מה סומן</h1><p>ירוק: התשובה שנמצאה מסומנת. בכל שורה: כמות הכתב שעליה.</p>"]
            + review + ["</html>"]))
    from collections import Counter
    for it in items:
        c = Counter(o["verdict"] for out, _ in table for o in out if o["item"] == it["name"])
        letters = Counter(o["chosen"] for out, _ in table for o in out if o["item"] == it["name"] and o["chosen"])
        print(f"{it['name']} (the solution's {it['answer']}): {sum(c.values())} exams - " +
              ", ".join(f"{v} {c[v]}" for v in ("right", "wrong", "none", "unclear")) +
              " | chosen: " + " ".join(f"{k}:{v}" for k, v in sorted(letters.items())))
    if marks and a.question:
        for k in range(len(items) + 1):
            offs = [o for out, o in table if sum(x["verdict"] == "wrong" for x in out) == k and o is not None]
            if offs:
                print(f"  {k} item(s) wrong: {len(offs)} exams, mean points off question {a.question} "
                      f"{sum(offs) / len(offs):.1f}")


if __name__ == "__main__":
    main()
