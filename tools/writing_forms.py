"""Forms to collect handwriting with its text known (4.11: the readers are taught the course's own handwriting - and no
model read Hebrew handwriting at all): each slot of a printed page shows a line - Hebrew as students write it in this
course, its formulas with varied numbers, the two mixed - and a box under it to copy it into by hand. A scanned page
is laid back on its template by the four corner squares, its number read from the row of squares between them, and
each box cut out with the line printed above it as its label - no labelling.

    python tools/writing_forms.py make  --out forms/writing --pages 12 [--seed 1]
        -> forms/writing.pdf (print it) and forms/writing.json (each page's slots: text, kind, box)
    python tools/writing_forms.py cut   forms/writing.json <scan.pdf> --out data/handwriting/<writer> [--dpi 200]
        -> <out>/<page>_<slot>.png and <out>/labels.jsonl ({"image", "text", "kind", "page", "slot"})

Kinds: "hebrew" (a sentence), "math" (LaTeX - printed as the formula, the label is the LaTeX), "mixed" (Hebrew words
then a formula, as an answer's line). A box left empty is left out."""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

A4 = (210.0, 297.0)                 # mm
MARK = 9.0                          # corner square's side, mm
MARGIN = 10.0                       # paper edge to a corner square
BITS = 8                            # the page's number: squares between the top corners, filled = 1
SLOTS = 8                           # lines a page
PROMPT_H, BOX_H, GAP = 8.0, 16.0, 3.5   # mm: the printed line, the box under it, the space after

HEBREW = [
    "הטרנזיסטור נמצא באזור הרוויה", "לכן הטרנזיסטור העליון כבוי", "המתח במוצא עולה עד מתח הספק",
    "במצב זה ה-NMOS כבוי וה-PMOS מוליך", "נשתמש באפקט המצע לחישוב מתח הסף", "זמן ההשהיה גדל כאשר הקיבול גדל",
    "השער מממש את הפונקציה NAND", "המוצא לא מגיע לאפס מלא", "הצומת נטען דרך הטרנזיסטור העליון",
    "לפי טבלת האמת המוצא שווה אחד רק כאשר שתי הכניסות אפס", "נבחר רוחב גדול יותר כדי להקטין את ההשהיה",
    "החישוב לא כולל את קיבול החיווט", "כאשר הכניסה עולה המוצא יורד", "זהו מהפך עם עומס קיבולי",
    "נניח שהזרם קבוע במהלך המעבר", "נחשב את הזרם הממוצע בין שתי הנקודות", "התשובה הסופית מסומנת במסגרת",
    "ראה את החישוב בעמוד הבא", "לא ניתן לחשב ללא נתון נוסף", "הטרנזיסטור עובר מרוויה לאזור הלינארי",
    "המתח בין השער למקור קטן ממתח הסף", "לכן אין זרם במצב היציב", "ההספק הדינמי תלוי בתדר המיתוג",
    "ההספק הסטטי נובע מזרמי זליגה", "שולי הרעש מחושבים מתוך עקום התמסורת", "הגרף מתאר את המתח כפונקציה של הזמן",
    "המעגל הוא מגבר עם עומס פעיל", "בשרשרת מהפכים כדאי להגדיל כל שלב פי ארבעה", "הקיבול של הכניסה גדל עם הרוחב",
    "בדקנו שהטרנזיסטור אכן ברוויה", "יש להתחשב בכך שמתח המקור אינו אפס", "התוצאה קרובה לערך שנמדד",
    "סעיף א", "סעיף ב", "סעיף ג", "סעיף ד", "תשובה", "הסבר", "נתון", "צריך להוכיח", "מש\"ל",
    "נמק את תשובתך", "ההשהיה של השער הראשון", "המאמץ הלוגי של השער", "השהייה כוללת של המסלול",
    "המעגל מתנהג כמחלק מתח", "כאשר השעון עולה הנתון נדגם", "זמן ההחזקה חייב להיות קטן מההשהיה המינימלית",
]

MIXED = [
    ("לכן", "V_{out} = {v1} V"), ("ומכאן", "t_{pHL} = {t1} ns"), ("נציב", "V_{SB} = {v2} V"),
    ("נקבל", "I_{av} = {i1} \\mu A"), ("כאשר", "A = 1, B = 0"), ("ולכן", "|V_{Tp}| = {v3} V"),
    ("בנוסף", "C_L = {c1} pF"), ("סה\"כ", "t_{pd} = {p1} ps"), ("נתון", "V_{DD} = {vdd} V"),
    ("המוצא", "Y = \\overline{A \\cdot B}"), ("לכן", "V_{OH} = V_{DD} - V_{Tn}"), ("במצב זה", "V_{x} = 0"),
]

MATH = [
    "V_{TH} = V_{T0} + \\gamma(\\sqrt{2\\phi_F + V_{SB}} - \\sqrt{2\\phi_F})",
    "V_{Tn} = {a} + {g}(\\sqrt{0.6 + {v2}} - \\sqrt{0.6}) = {v3} V",
    "t_{pHL} = \\frac{C_L \\Delta V}{I_{av}} = \\frac{{c1} pF \\cdot {v1} V}{{i1} \\mu A} = {t1} ns",
    "I_D = \\frac{k_n'}{2} \\frac{W}{L} (V_{GS} - V_{Tn})^2",
    "I_D = {k} \\cdot ({v4} - {v3})^2 = {i1} \\mu A",
    "I_{av} = \\frac{I(t_1) + I(t_2)}{2}",
    "V_{OH,max} = V_{DD} - V_{Tn} = {vdd} - {v3} = {v5} V",
    "t_{pd} = t_{p0}(p + g h) = {p0} ps \\cdot ({pp} + {gg}) = {p1} ps",
    "f_{osc} = \\frac{1}{2 N t_p}",
    "P_{dyn} = \\alpha C_L V_{DD}^2 f",
    "P = {al} \\cdot {c1} pF \\cdot {vdd}^2 \\cdot {f} GHz = {pw} \\mu W",
    "NM_H = V_{OH} - V_{IH} = {v5} - {v1} = {v6} V",
    "Y = \\overline{A \\cdot B} + C",
    "F = \\overline{(A + B) \\cdot C}",
    "F = G B H",
    "\\hat{f} = F^{1/N} = {fh}",
    "T = \\frac{1}{{f} \\cdot 10^9} = {tt} ps",
    "t_{clk \\rightarrow Q} + t_{logic} + t_{setup} \\leq T",
    "\\frac{dV_{out}}{dV_{in}} = -1",
    "V_{SG} - |V_{Tp}| = {v1} - {v3} = {v6} V",
    "V_{SD} > V_{SG} - |V_{Tp}|",
    "R_{eq} = \\frac{3}{4} \\frac{V_{DD}}{I_{DSAT}}",
    "t_p = 0.69 R_{eq} C_L = 0.69 \\cdot {r} k\\Omega \\cdot {c1} pF = {t1} ns",
    "C_{in} = C_{g} \\frac{W}{L} = {c2} fF",
]


def numbers(rng: random.Random) -> dict:
    """the values a formula is written with - varied, as a hand writes digits"""
    r = lambda a, b, d=2: f"{rng.uniform(a, b):.{d}f}"
    return {"v1": r(0.3, 1.8), "v2": r(0.1, 1.8), "v3": r(0.3, 0.8, 3), "v4": r(0.9, 1.8), "v5": r(0.9, 1.6),
            "v6": r(0.1, 0.9), "vdd": rng.choice(["1.8", "1.2", "2.5", "3.3"]), "a": r(0.3, 0.5), "g": r(0.2, 0.6),
            "t1": r(1, 99, 1), "i1": r(1, 60, 2), "c1": rng.choice(["1", "0.5", "2", "10"]), "c2": r(1, 9, 1),
            "k": r(10, 300, 0), "p0": rng.choice(["6.9", "5", "10", "15"]), "pp": rng.choice(["1", "2", "3"]),
            "gg": rng.choice(["1", "4/3", "5/3", "2"]), "p1": r(5, 200, 1), "al": rng.choice(["0.1", "0.25", "0.5"]),
            "f": rng.choice(["1", "2", "5"]), "pw": r(1, 90, 1), "fh": r(2, 6, 2), "tt": rng.choice(["200", "500", "1000"]),
            "r": r(1, 20, 1)}


def fill(t: str, vals: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), t)


def prompts(n: int, rng: random.Random) -> list:
    """n lines to copy: half Hebrew sentences (the reading that fails most), the rest formulas and the two mixed -
    shuffled"""
    out = []
    for k in range(n):
        kind = ("hebrew", "mixed", "hebrew", "math", "hebrew", "mixed", "hebrew", "math", "hebrew", "mixed")[k % 10]
        vals = numbers(rng)
        if kind == "hebrew":
            out.append({"kind": kind, "text": rng.choice(HEBREW)})
        elif kind == "math":
            out.append({"kind": kind, "text": fill(rng.choice(MATH), vals)})
        else:
            w, m = rng.choice(MIXED)
            out.append({"kind": kind, "text": f"{w} {fill(m, vals)}", "words": w, "math": fill(m, vals)})
    rng.shuffle(out)
    return out


_HEB = re.compile(r"[֐-׿\"'\-]+(?:\s+[֐-׿\"'\-]+)*")


def visual(t: str) -> str:
    """Hebrew text in the order a left-to-right renderer draws it: the runs in reverse, the Hebrew ones reversed (a
    number or a Latin word in it stays as written - 'ה-NMOS')"""
    def ltr(r):                                   # (a left-to-right run: the spaces around it change sides)
        core = r.strip()
        return (" " if r[len(r.rstrip()):] else "") + core + (" " if r[:len(r) - len(r.lstrip())] else "") \
            if core else r
    runs, pos = [], 0
    for m in _HEB.finditer(t):
        if m.start() > pos:
            runs.append(ltr(t[pos:m.start()]))
        runs.append(m.group(0)[::-1])
        pos = m.end()
    if pos < len(t):
        runs.append(ltr(t[pos:]))
    return "".join(reversed(runs))


def frame():
    """the corner squares' centres (mm, from the top-left corner of the page): the frame every box is given in"""
    a = MARGIN + MARK / 2
    return [(a, a), (A4[0] - a, a), (A4[0] - a, A4[1] - a), (a, A4[1] - a)]


def layout() -> list:
    """the slots' rectangles (mm): [(the printed line's, the box's)]"""
    top = MARGIN + MARK + 22.0
    out = []
    for k in range(SLOTS):
        y = top + k * (PROMPT_H + BOX_H + GAP)
        out.append(((MARGIN + 6, y, A4[0] - MARGIN - 6, y + PROMPT_H),
                    (MARGIN + 6, y + PROMPT_H, A4[0] - MARGIN - 6, y + PROMPT_H + BOX_H)))
    return out


def make(out: str, pages: int, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    rng = random.Random(seed)
    W, H = A4[0] / 25.4, A4[1] / 25.4
    info = {"paper_mm": A4, "frame_mm": frame(), "bits": BITS, "pages": []}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with PdfPages(out + ".pdf") as pdf:
        for pno in range(1, pages + 1):
            fig = plt.figure(figsize=(W, H))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, A4[0])
            ax.set_ylim(A4[1], 0)
            ax.axis("off")
            for cx, cy in frame():                                    # (the corner squares)
                ax.add_patch(Rectangle((cx - MARK / 2, cy - MARK / 2), MARK, MARK, color="black"))
            x0 = MARGIN + MARK + 8
            step = (A4[0] - 2 * x0) / BITS
            for b in range(BITS):                                     # (the page's number, a bit a square)
                on = (pno >> (BITS - 1 - b)) & 1
                ax.add_patch(Rectangle((x0 + b * step + step / 2 - 2.5, MARGIN + 2), 5, 5,
                                       facecolor="black" if on else "white", edgecolor="black", lw=0.8))
            ax.text(A4[0] - MARGIN - 6, MARGIN + MARK + 11, visual(f"טופס כתב יד - עמוד {pno}"), ha="right",
                    va="center", fontsize=12, family="DejaVu Sans")
            ax.text(A4[0] - MARGIN - 6, MARGIN + MARK + 17,
                    visual("העתיקו כל שורה במסגרת שמתחתיה, בכתב היד הרגיל שלכם, כמו במבחן. אין צורך בדיוק מושלם."),
                    ha="right", va="center", fontsize=8.5, family="DejaVu Sans", color="#444444")
            slots = []
            for (pr, box), p in zip(layout(), prompts(SLOTS, rng)):
                yc = (pr[1] + pr[3]) / 2
                if p["kind"] == "hebrew":
                    ax.text(pr[2], yc, visual(p["text"]), ha="right", va="center", fontsize=12, family="DejaVu Sans")
                elif p["kind"] == "math":
                    ax.text(pr[0], yc, f"${p['text']}$", ha="left", va="center", fontsize=12)
                else:
                    t = ax.text(pr[2], yc, visual(p["words"]), ha="right", va="center", fontsize=12, family="DejaVu Sans")
                    fig.canvas.draw()
                    bb = t.get_window_extent().transformed(ax.transData.inverted())
                    ax.text(min(bb.x0, bb.x1) - 3, yc, f"${p['math']}$", ha="right", va="center", fontsize=12)
                ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                                       edgecolor="#9a9a9a", lw=0.6))
                slots.append({**p, "box_mm": box})
            info["pages"].append({"page": pno, "slots": slots})
            pdf.savefig(fig)
            plt.close(fig)
    json.dump(info, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{pages} pages, {pages * SLOTS} lines -> {out}.pdf, {out}.json")


def find_marks(gray):
    """the four corner squares on a scanned page -> their centres (px), top-left, top-right, bottom-right,
    bottom-left; None if not all four are found"""
    import cv2
    import numpy as np
    h, w = gray.shape
    bw = (gray < 110).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    side = min(w, h) * MARK / A4[0]                                   # (a square's side in px, about)
    cands = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if 0.55 * side < ww < 1.6 * side and 0.55 * side < hh < 1.6 * side and area > 0.75 * ww * hh:
            cands.append(tuple(cent[i]))
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    out = []
    for cx, cy in corners:
        near = [c for c in cands if abs(c[0] - cx) < 0.3 * w and abs(c[1] - cy) < 0.3 * h]
        if not near:
            return None
        out.append(min(near, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2))
    return out


def cut(form: str, scan: str, out: str, dpi: int = 200):
    import cv2
    import numpy as np
    from read_page import load_pages
    info = json.load(open(form, encoding="utf-8"))
    by_page = {p["page"]: p for p in info["pages"]}
    px = dpi / 25.4
    dst = np.float32([(x * px, y * px) for x, y in info["frame_mm"]])
    size = (int(A4[0] * px), int(A4[1] * px))
    os.makedirs(out, exist_ok=True)
    n = 0
    with open(os.path.join(out, "labels.jsonl"), "a", encoding="utf-8") as lab:
        for k, img in enumerate(load_pages(scan, dpi), 1):
            gray = np.asarray(img.convert("L"))
            marks = find_marks(gray)
            if marks is None:
                print(f"scan page {k}: the corner squares not found - skipped")
                continue
            M = cv2.getPerspectiveTransform(np.float32(marks), dst)
            flat = cv2.warpPerspective(gray, M, size, borderValue=255)
            x0 = MARGIN + MARK + 8
            step = (A4[0] - 2 * x0) / BITS
            code = 0
            for b in range(BITS):                                      # (the page's number)
                cx, cy = (x0 + b * step + step / 2) * px, (MARGIN + 4.5) * px
                r = 1.5 * px
                cell = flat[int(cy - r):int(cy + r), int(cx - r):int(cx + r)]
                code = code * 2 + int(cell.size and cell.mean() < 128)
            page = by_page.get(code)
            if page is None:
                print(f"scan page {k}: page number {code} not in {form} - skipped")
                continue
            for s, slot in enumerate(page["slots"], 1):
                a, b, c, d = (v * px for v in slot["box_mm"])
                crop = flat[int(b) + 3:int(d) - 3, int(a) + 3:int(c) - 3]
                if (crop < 128).sum() < 0.002 * crop.size:               # (left empty)
                    continue
                name = f"p{code:02d}_s{s}.png"
                cv2.imwrite(os.path.join(out, name), crop)
                lab.write(json.dumps({"image": name, "text": slot["text"], "kind": slot["kind"], "page": code,
                                      "slot": s}, ensure_ascii=False) + "\n")
                n += 1
            print(f"scan page {k}: form page {code}")
    print(f"{n} lines -> {out}/labels.jsonl")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default=os.path.join(HERE, "forms", "writing"))
    m.add_argument("--pages", type=int, default=12)
    m.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("cut")
    c.add_argument("form")
    c.add_argument("scan")
    c.add_argument("--out", required=True)
    c.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.out, a.pages, a.seed)
    else:
        cut(a.form, a.scan, a.out, a.dpi)


if __name__ == "__main__":
    main()
