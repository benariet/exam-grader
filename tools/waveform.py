"""Digital waveforms students draw on printed time axes - the course's Moed C 3: Z and Out against the printed clock In
- checked by their shape, on this computer: where the drawing is high, against where the solution's waveform is high.

The booklet's page gives each plot (in the template's pixels: its time axis' x from-to, the y of its V_DD line and of
its 0 axis) and the input's edges (x, rising or falling). A plot's high stretches are the drawing's horizontal strokes
near its V_DD line (the print left out; a stroke along the 0 axis is the axis itself - low is where nothing is high).
Two kinds of waveform:
  inverted  --kind inverted: the input inverted, delayed a little (Z = NOT In, a gate's delay later): each half period
            of the input (its first --settle pixels left out, the delay) must be at the opposite level
  pulses    --kind pulses: low but for a short pulse a little after each rising edge of the input (Out = In AND Z - the
            glitch while Z has not yet fallen): a pulse within --after pixels of each rising edge, none elsewhere

    python tools/waveform.py "<exams' PDFs>" --booklet booklet/moedC --page 25 --plot 216,960,699,962 \\
        --edges 374r,527f,680r,833f --kind inverted --name "3 Z" --question 3 --out <folder> [--grades ...] [--review]
        -> right / wrong / unsure / empty per exam; <out>/<n>.wave.json, <n>_wave_remarks.txt (Hebrew)
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

HIGH, LOW = 0.6, 0.15          # a window's share of high columns to call it high / low


def high_columns(hand, gray, printed, x0, x1, y_vdd, y_zero, run=12):
    """the drawing's columns x0..x1 that are high -> (high, known) bool arrays (x0..x1): a horizontal stroke (at least
    run pixels long) at the V_DD line (a few pixels off it - not the arrows and numbers written above a waveform to mark
    its times, nor a level drawn well below: neither is taken as high); or, on the printed dashed
    V_DD line itself (where the print hides a stroke drawn along it), ink in the gaps between its dashes. A column under
    a dash, with no stroke of its own, takes its neighbours' level when both sides agree, else is not known"""
    from scipy import ndimage as ndi
    top, bottom = int(y_vdd - 9), int(y_vdd + 11)
    band = ndi.binary_dilation(hand[max(0, top):bottom, x0:x1], structure=np.ones((5, 1)))   # (a hand's line is not
    flat = ndi.binary_opening(band, structure=np.ones((1, run))).any(axis=0)                  #  level: a pixel or two)
    line = slice(int(y_vdd - 7), int(y_vdd + 8))
    dash = ndi.binary_dilation(printed[line, x0:x1], structure=np.ones((1, 3))).any(axis=0)
    filled = ~dash & (gray[line, x0:x1] < 175).any(axis=0)       # (a gap between dashes the student's line fills)
    high, known = flat | filled, flat | ~dash
    i = 0
    while i < len(high):                               # (under a dash: its neighbours' level, when they agree)
        if known[i]:
            i += 1
            continue
        j = i
        while j < len(high) and not known[j]:
            j += 1
        if 0 < i and j < len(high) and j - i <= 60 and high[i - 1] == high[j]:     # (a dash: up to 60 px)
            high[i:j], known[i:j] = high[i - 1], True
        i = j
    return high, known


def drawn_range(hand, x0, x1, y_vdd, y_zero):
    """the x-range the student drew in (any handwriting in the plot, its labels' corner left out) -> (from, to) or
    None"""
    sub = hand[int(y_vdd - 14):int(y_zero - 3), x0:x1]
    cols = np.where(sub.sum(axis=0) > 0)[0]
    if len(cols) < 15:
        return None
    return x0 + int(cols.min()), x0 + int(cols.max())


def judge_inverted(high, known, x0, edges, drawn, settle):
    """an inverted copy of the input: each half period after its first edge (settle pixels left out - the delay) at
    the opposite level of the input's -> (verdict, [(from, to, expected, seen)])"""
    wins = []
    for (xa, kind), (xb, _) in zip(edges, edges[1:] + [(None, None)]):
        a, b = xa + settle, (xb if xb is not None else x0 + len(high)) - 5
        if b - a < 20 or drawn is None or a < drawn[0] - 5 or b > drawn[1] + 5:
            continue
        want = "low" if kind == "r" else "high"   # (after a rising edge the input is high: the inverse low)
        k = known[a - x0:b - x0]
        share = float(high[a - x0:b - x0][k].mean()) if k.mean() >= 0.4 else 0.5   # (too little known: not either)
        seen = "high" if share >= HIGH else "low" if share <= LOW else "?"
        wins.append((a, b, want, seen, round(share, 2), round(float(k.mean()), 2)))
    good = sum(1 for w in wins if w[3] == w[2])
    bad = sum(1 for w in wins if w[3] not in (w[2], "?"))
    if drawn is None:
        return "empty", wins
    if bad == 0 and good >= 2 and {w[2] for w in wins if w[3] == w[2]} == {"high", "low"}:   # (both levels seen right:
        return "right", wins                                                                  #  inverted, not copied)
    if len(wins) >= 3 and bad == len(wins):        # (every window the other way: the input copied, not inverted)
        return "wrong", wins
    return "unsure", wins


def judge_pulses(high, x0, edges, drawn, after, widest):
    """low but for a short pulse a little after each rising edge -> (verdict, [(from, to, 'pulse' / 'low', seen)])"""
    if drawn is None:
        return "empty", []
    rises = [x for x, k in edges if k == "r" and drawn[0] - 5 <= x <= drawn[1] - after / 2]
    runs, x = [], 0
    while x < len(high):                               # (the high stretches)
        if high[x]:
            y = x
            while y < len(high) and high[y]:
                y += 1
            runs.append((x0 + x, x0 + y))
            x = y
        x += 1
    wins, near = [], set()
    for r in rises:
        hit = [(a, b) for a, b in runs if a < r + after and b > r]
        near.update(hit)
        ok = len(hit) == 1 and hit[0][1] - hit[0][0] <= widest
        wins.append((r, r + after, "pulse", "pulse" if ok else ("wide" if hit else "low")))
    stray = sum(b - a for a, b in runs if (a, b) not in near)
    if not rises:
        return "unsure", wins
    if all(w[3] == "pulse" for w in wins) and stray <= 10:
        return "right", wins
    if not runs:                                       # (drawn, never high: the glitch missed - Out taken as 0)
        return "wrong", wins
    if all(w[3] == "low" for w in wins) and stray >= 25:     # (pulses, not where they should be)
        return "wrong", wins
    return "unsure", wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--booklet", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--plot", required=True, help="x_from,x_to,y_vdd,y_zero of the plot, on the booklet's template")
    ap.add_argument("--edges", required=True, help="the input's edges on the template: '374r,527f,680r,833f'")
    ap.add_argument("--kind", choices=("inverted", "pulses"), required=True)
    ap.add_argument("--settle", type=int, default=40, help="pixels after an edge left out (the delay drawn)")
    ap.add_argument("--after", type=int, default=110, help="pulses: a pulse must start within this after a rise")
    ap.add_argument("--widest", type=int, default=90, help="pulses: and be at most this wide")
    ap.add_argument("--name", default="")
    ap.add_argument("--question", type=int, default=0)
    ap.add_argument("--grades", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--decisions", default="")
    a = ap.parse_args()
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw
    from recog.page import clean_scan
    from recog.printed import fit_map, from_ref, printed_on
    from tools.expr_box import soft_hand
    from tools.truth_table import to_scan
    template = np.asarray(Image.open(os.path.join(a.booklet, f"p{a.page}.png")).convert("L")) < 128
    px0, px1, py_vdd, py_zero = (int(v) for v in a.plot.split(","))
    edges_t = [(int(e[:-1]), e[-1]) for e in a.edges.split(",")]
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    os.makedirs(a.out, exist_ok=True)
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
        exact = from_ref(template, M, b, ink.shape)     # (the print exactly, not grown: its dashes' gaps)
        x0, y_vdd, x1, y_zero = to_scan((px0, py_vdd, px1, py_zero), M, b)
        edges = sorted((to_scan((x, py_vdd, x, py_vdd), M, b)[0], k) for x, k in edges_t)
        high, known = high_columns(hand, gray, exact, x0, x1, y_vdd, y_zero, 12 if a.kind == "inverted" else 6)
        drawn = drawn_range(hand, x0, x1, y_vdd, y_zero)
        if a.kind == "inverted":
            v, wins = judge_inverted(high, known, x0, edges, drawn, a.settle)
        else:
            v, wins = judge_pulses(high, x0, edges, drawn, a.after, a.widest)
        how = ""
        if v == "unsure" and a.decisions:
            from tools.decide_sheet import decided, load
            hv, how = decided(load(a.decisions), f"wave:{a.name}", name, "")
            v = hv or v
        out = {"exam": name, "name": a.name, "verdict": v, "kind": "wave", "windows": [list(w) for w in wins],
               "high_share": round(float(high.mean()), 3), "known_share": round(float(known.mean()), 3), "drawn": drawn}
        if how:
            out["decided"] = how
        json.dump(out, open(os.path.join(a.out, f"{name}.wave.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        expect = "הפוך ומושהה מעט ביחס ל-In" if a.kind == "inverted" else "נמוך, עם פולס קצר מעט אחרי כל עליה של In"
        txt = {"right": f"{a.name}: צורת האות נכונה ({expect}).",
               "wrong": f"{a.name}: צורת האות שונה מהפתרון (בפתרון: {expect}).",
               "unsure": f"{a.name}: הציור לא הוכרע - לבדיקה ידנית (בפתרון: {expect}).",
               "empty": f"{a.name}: לא נמצא ציור (אולי צויר על הציר עצמו) - לבדיקה ידנית."}[v]
        if how:
            txt = f"{a.name}: {'נכון' if v == 'right' else 'שגוי'} - לפי החלטת הבודק."
        open(os.path.join(a.out, f"{name}_wave_remarks.txt"), "w", encoding="utf-8").write(txt + "\n")
        table.append((v, marks.get(name, {}).get(a.question) if marks and a.question else None))
        if a.review:
            import base64
            import io
            q = img.convert("RGB").crop((max(0, x0 - 30), int(y_vdd - 40), min(img.width, x1 + 10), int(y_zero + 20)))
            d = ImageDraw.Draw(q)
            ox, oy = max(0, x0 - 30), int(y_vdd - 40)
            for i in np.where(high)[0]:                # (what was taken as high: green)
                d.line([(x0 + i - ox, y_vdd - oy - 18), (x0 + i - ox, y_vdd - oy - 12)], fill=(0, 170, 0))
            for wa, wb, want, seen, *_ in wins:        # (each window: what the solution has, what was seen)
                col = (0, 120, 220) if seen in (want, "pulse") else (220, 0, 0)
                d.rectangle([wa - ox, y_zero - oy + 4, wb - ox, y_zero - oy + 10], outline=col, width=2)
            buf = io.BytesIO()
            q.save(buf, "PNG")
            review.append(f"<figure><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
                          f"<figcaption>{html.escape(f'מבחן {name}: {txt}')}</figcaption></figure>")
    if a.review:
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(
            ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Waveforms review</title>",
             "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}</style>",
             f"<h1>{html.escape(a.name)} - מה נקרא</h1><p>ירוק: מה שנקרא כרמה גבוהה. מלבן כחול: חלון שתואם לפתרון; "
             "אדום: לא תואם.</p>"] + review + ["</html>"]))
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
