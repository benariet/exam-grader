"""Handwritten Boolean expressions made up, to test the reading of the ones students write ('Vx = A ⊕ B', 'Y2 = AB +
C(A ⊕ B)'): each symbol a real person's handwriting - MathWriting's isolated symbols (google-research, CC BY-NC-SA
4.0; prepare_mathwriting.py keeps them), a different writer's each time, jittered - laid out as a hand lays an
expression out: letters side by side for 'and', '+', '⊕', brackets, a prime or a bar over a letter or a group for
'not', 'Vx' with its x low; the function each one is known.

    python tools/bool_synth.py --out <folder> --n 300 [--page booklet/moedC/p5.png --rows auto]
        -> <folder>/<i>.png (the expression alone, on white, as cut from a page) and <i>.json ({"name", "latex",
           "function"}); with --page, <i>_page.png too: the expression written on the booklet's page, in its box's row
"""
import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MW = os.environ.get("MATHWRITING", os.path.join(HERE, "data", "mathwriting"))
GLYPHS = {"A": "A", "B": "B", "C": "C", "V": "V", "x": "x", "+": "+", "^": "\\oplus", "(": "(", ")": ")",
          "'": "\\prime", ".": "\\cdot", "0": "0", "1": "1", "S": "S", "W": "W", "s": "s", "w": "w", "/": "/",
          "\\": "\\backslash"}
# each glyph's height (letter heights) and where its middle sits (above the baseline, letter heights)
SIZE = {"A": (1.0, 0.5), "B": (1.0, 0.5), "C": (1.0, 0.5), "V": (1.0, 0.5), "x": (0.5, -0.05), "+": (0.6, 0.45),
        "^": (0.7, 0.45), "(": (1.3, 0.45), ")": (1.3, 0.45), "'": (0.35, 1.0), ".": (0.12, 0.45), "0": (1.0, 0.5),
        "1": (1.0, 0.5), "S": (1.0, 0.5), "W": (1.0, 0.5), "s": (0.6, 0.3), "w": (0.6, 0.3), "/": (1.2, 0.5),
        "\\": (1.2, 0.5)}


def load_glyphs(root=MW, want=GLYPHS):
    """{glyph: [strokes, each an (n, 2) array]} from MathWriting's symbols (labels.tsv + symbols/*.inkml)"""
    by = {v: k for k, v in want.items()}
    out = {k: [] for k in want}
    for line in open(os.path.join(root, "labels.tsv"), encoding="utf-8"):
        split, f, lab = (line.rstrip("\n").split("\t") + ["", ""])[:3]
        if split != "symbols" or lab not in by:
            continue
        txt = open(os.path.join(root, "symbols", f), encoding="utf-8").read()
        strokes = []
        for tr in re.findall(r"<trace(?:\s[^>]*)?>(.*?)</trace>", txt, re.S):
            pts = [p.split()[:2] for p in tr.strip().split(",") if p.strip()]
            if pts:
                strokes.append(np.array(pts, dtype=float))
        if strokes:
            out[by[lab]].append(strokes)
    return out


# ---- expressions: ('v', name) | ('not', e, 'prime'|'bar') | ('and', [e]) | ('or', [e]) | ('xor', [e])

def to_sympy(e, env=None):
    import sympy as sp
    k = e[0]
    if k == "v":
        return (env or {}).get(e[1], sp.Symbol(e[1]))
    if k == "c":
        return sp.true if e[1] else sp.false
    if k == "not":
        return sp.Not(to_sympy(e[1], env))
    args = [to_sympy(x, env) for x in e[1]]
    return {"and": sp.And, "or": sp.Or, "xor": sp.Xor}[k](*args)


def to_latex(e):
    k = e[0]
    if k == "v":
        return "V_{x}" if e[1] == "Vx" else e[1]
    if k == "c":
        return str(e[1])
    if k == "not":
        inner = to_latex(e[1])
        if e[2] == "bar":
            return "\\overline{" + inner + "}"
        return (inner if e[1][0] == "v" else "(" + inner + ")") + "'"
    sep = {"and": " ", "or": " + ", "xor": " \\oplus "}[k]
    return sep.join(_paren(x, k, to_latex(x)) for x in e[1])


def _needs_paren(child, parent):
    rank = {"or": 1, "xor": 2, "and": 3}                   # (as grader.circuit_check.parse_bool reads them: & ^ |)
    return child[0] in rank and rank[child[0]] < rank[parent]


def _paren(child, parent, s):
    return "(" + s + ")" if _needs_paren(child, parent) else s


V = lambda n: ("v", n)                                     # noqa: E731
NOT = lambda e, s="prime": ("not", e, s)                   # noqa: E731
AND = lambda *a: ("and", list(a))                          # noqa: E731
OR = lambda *a: ("or", list(a))                            # noqa: E731
XOR = lambda *a: ("xor", list(a))                          # noqa: E731


def _nots(rnd):
    return rnd.choice(["prime", "bar"])


def forms(rnd):
    """[(name, expression)] - the Moed C 1(a) answers students may write: right ones in their many forms, and wrong
    ones"""
    A, B, C, X = V("A"), V("B"), V("C"), V("Vx")
    s = _nots(rnd)
    vx = [XOR(A, B), OR(AND(NOT(A, s), B), AND(A, NOT(B, s))), AND(OR(A, B), OR(NOT(A, s), NOT(B, s))),
          AND(A, B), OR(A, B), NOT(XOR(A, B), "bar"), OR(AND(NOT(A, s), NOT(B, s)), AND(A, B))]
    y1 = [XOR(A, B, C), XOR(X, C), XOR(XOR(A, B), C) if rnd.random() < 0.5 else XOR(A, XOR(B, C)),
          OR(AND(NOT(X, s), C), AND(X, NOT(C, s))),
          OR(AND(NOT(A, s), NOT(B, s), C), AND(NOT(A, s), B, NOT(C, s)), AND(A, NOT(B, s), NOT(C, s)), AND(A, B, C)),
          XOR(A, B), AND(NOT(X, "bar"), C), OR(A, B, C)]
    y2 = [OR(AND(A, B), AND(C, XOR(A, B))), OR(AND(A, B), AND(A, C), AND(B, C)), OR(AND(A, B), AND(X, C)),
          OR(AND(A, B), AND(C, X)), OR(AND(A, B), C), AND(A, B, C), AND(A, B), OR(AND(A, B), AND(NOT(X, s), C))]
    out = [("Vx", e) for e in vx] + [("Y1", e) for e in y1] + [("Y2", e) for e in y2]
    return out


# ---- layout and drawing

def _glyph_seq(e, rnd, dot_and):
    """the expression -> [(glyph, group depth)] with bars: [('bar', start, end)] (indices into the glyphs)"""
    seq, bars = [], []

    def emit(e):
        k = e[0]
        if k == "v":
            if e[1] == "Vx":
                seq.extend(["V", "x"])
            else:
                seq.append(e[1])
        elif k == "c":
            seq.append(str(e[1]))
        elif k == "not":
            if e[2] == "bar":
                i = len(seq)
                emit(e[1])
                bars.append((i, len(seq)))
            else:
                if e[1][0] == "v":
                    emit(e[1])
                else:
                    seq.append("(")
                    emit(e[1])
                    seq.append(")")
                seq.append("'")
        else:
            for j, x in enumerate(e[1]):
                if j:
                    if k == "or":
                        seq.append("+")
                    elif k == "xor":
                        seq.append("^")
                    elif dot_and:
                        seq.append(".")
                if _needs_paren(x, k):
                    seq.append("(")
                    emit(x)
                    seq.append(")")
                else:
                    emit(x)
    emit(e)
    return seq, bars


def draw(e, glyphs, rnd, h=None, ink=None, width=None):
    """the expression handwritten -> (grayscale image (uint8, white paper), the baseline's y)"""
    seq, bars = _glyph_seq(e, rnd, dot_and=rnd.random() < 0.15)
    return draw_seq(seq, bars, glyphs, rnd, h, ink, width)


def draw_seq(seq, bars, glyphs, rnd, h=None, ink=None, width=None):
    """glyphs side by side (bars: [(first, past the last)] - a bar over those) -> (grayscale image, the baseline's y)"""
    from PIL import Image, ImageDraw
    h = h or rnd.uniform(16, 26)                           # (letter height, pixels at 150 dpi)
    slant = rnd.uniform(-0.15, 0.2)
    gap = rnd.uniform(0.12, 0.35)
    S = 4                                                  # (drawn 4x, shrunk: smooth strokes)
    placed, x = [], 0.0
    for gch in seq:
        sample = rnd.choice(glyphs[gch])
        pts = np.concatenate(sample)
        lo, hi = pts.min(0), pts.max(0)
        gh, gw = max(hi[1] - lo[1], 1e-6), max(hi[0] - lo[0], 1e-6)
        rel, mid = SIZE[gch]
        th = h * rel * rnd.uniform(0.88, 1.12)
        k = th / gh
        if gch in ".":
            k = min(k, h * 0.15 / max(gw, 1e-6))
        w = gw * k
        ang = math.radians(rnd.uniform(-5, 5))
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        cy = -(mid * h) + rnd.uniform(-0.07, 0.07) * h     # (y down; the baseline at 0)
        g_gap = gap * h * rnd.uniform(0.6, 1.4)
        if gch in ("'", "x"):
            g_gap *= 0.25
        x0 = x + g_gap
        strokes = []
        for st in sample:
            p = (st - lo) * k - np.array([w / 2, th / 2])
            p = p @ rot.T
            p = p + np.array([x0 + w / 2, cy])
            p = p + rnd.gauss(0, 0.012) * h * np.cumsum(np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 1, p.shape), 0) / max(1, len(p)) ** 0.5
            strokes.append(p)
        placed.append((gch, strokes, x0, x0 + w))
        x = x0 + w
    for i, j in bars:                                      # (a bar over the group, a little past its ends)
        xa, xb = placed[i][2] - 0.1 * h, placed[j - 1][3] + 0.1 * h
        top = min(min(p[:, 1].min() for p in placed[t][1]) for t in range(i, j))
        y = top - rnd.uniform(0.15, 0.3) * h
        n = 12
        xs = np.linspace(xa, xb, n)
        ys = y + rnd.uniform(-0.08, 0.08) * h * np.linspace(-1, 1, n) + np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 0.02 * h, n)
        placed.append(("bar", [np.stack([xs, ys], 1)], xa, xb))
    allp = np.concatenate([p for _, ss, _, _ in placed for p in ss])
    allp = allp.copy()
    allp[:, 0] = allp[:, 0] - slant * allp[:, 1]
    pad = 0.4 * h
    x_lo, y_lo = allp.min(0) - pad
    x_hi, y_hi = allp.max(0) + pad
    W, H = int(math.ceil(x_hi - x_lo)), int(math.ceil(y_hi - y_lo))
    img = Image.new("L", (W * S, H * S), 255)
    d = ImageDraw.Draw(img)
    pen = (width or rnd.uniform(1.6, 3.0)) * S
    color = ink if ink is not None else rnd.randint(20, 110)
    for _, ss, _, _ in placed:
        for p in ss:
            q = p.copy()
            q[:, 0] = q[:, 0] - slant * q[:, 1]
            q = (q - np.array([x_lo, y_lo])) * S
            pts = [tuple(v) for v in q]
            if len(pts) == 1:
                pts = pts * 2
            d.line(pts, fill=color, width=max(1, int(round(pen))), joint="curve")
            for v in (pts[0], pts[-1]):
                r = pen / 2
                d.ellipse((v[0] - r, v[1] - r, v[0] + r, v[1] + r), fill=color)
    img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img), -y_lo


def cell_forms(rnd):
    """a truth table's cell as students fill it: a logic value and its strength - '1S', '0 W', '1W/S', '0(S)', 'w' small
    - -> (glyphs, value, strength or '')"""
    v = rnd.choice("01")
    st = rnd.choice(["S", "W", "S", "W", "W/S", ""])
    letter = (lambda c: c.lower() if rnd.random() < 0.25 else c)    # noqa: E731
    if st == "W/S":
        sep = rnd.choice(["/", "\\"])
        return [v, letter("W"), sep, letter("S")], v, "W/S"
    if not st:
        return [v], v, ""
    if rnd.random() < 0.15:
        return [v, "(", letter(st), ")"], v, st
    return [v, letter(st)], v, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mathwriting", default=MW)
    ap.add_argument("--page", default="", help="the booklet's page with the answer box: the expressions written in it too")
    ap.add_argument("--scale", type=float, default=1.0, help="the page's resolution / 150 dpi (2: a 300-dpi page)")
    ap.add_argument("--cells", action="store_true", help="truth-table cells (a value and its strength) instead")
    a = ap.parse_args()
    from PIL import Image
    rnd = random.Random(a.seed)
    glyphs = load_glyphs(a.mathwriting)
    missing = [k for k, v in glyphs.items() if not v]
    if missing:
        raise SystemExit(f"no MathWriting symbols for {missing} in {a.mathwriting}")
    os.makedirs(a.out, exist_ok=True)
    rows = None
    if a.page:
        from tools.expr_box import answer_rows, blank_answers
        t = np.asarray(Image.open(a.page).convert("L")) < 128
        rows = answer_rows(t)
        tpl = Image.fromarray(np.where(blank_answers(t, rows), 0, 255).astype(np.uint8)) if rows else None
        if not rows:
            raise SystemExit(f"no answer box found on {a.page}")
    for i in range(a.n):
        if a.cells:
            seq, v, st = cell_forms(rnd)
            img, _ = draw_seq(seq, [], glyphs, rnd, h=rnd.uniform(14, 24))
            Image.fromarray(img).save(os.path.join(a.out, f"{i}.png"))
            json.dump({"value": v, "strength": st, "glyphs": "".join(seq)}, open(os.path.join(a.out, f"{i}.json"), "w"))
            continue
        name, e = rnd.choice(forms(rnd))
        img, base = draw(e, glyphs, rnd)
        Image.fromarray(img).save(os.path.join(a.out, f"{i}.png"))
        meta = {"name": name, "latex": to_latex(e), "function": str(to_sympy(e))}
        if rows:
            page = _on_page(tpl, rows, name, img, base, rnd, a.scale)
            page.save(os.path.join(a.out, f"{i}_page.png"))
        json.dump(meta, open(os.path.join(a.out, f"{i}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{a.n} expressions -> {a.out}")


def _on_page(tpl, rows, name, img, base, rnd, scale=1.0):
    """the expression written on the booklet's page, in the box's row named name, after its label, as a scan: the
    print gray, paper not quite white, the page a little shifted"""
    from PIL import Image
    r = next(r for r in rows if r["name"] == name) if any(r.get("name") == name for r in rows) else \
        rows[["Vx", "Y1", "Y2"].index(name) if name in ("Vx", "Y1", "Y2") else 0]
    pg = np.asarray(tpl).astype(np.float32)
    pg = np.where(pg < 128, rnd.uniform(10, 60), rnd.uniform(235, 252)) + np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 3, pg.shape)
    x0 = r["x0"] + rnd.uniform(4, 25)
    yb = r["base"] + rnd.uniform(-3, 4)
    hh, ww = img.shape
    X, Y = int(x0), int(yb - base)
    sub = pg[Y:Y + hh, X:X + ww]
    hh2, ww2 = sub.shape
    pg[Y:Y + hh2, X:X + ww2] = np.minimum(sub, img[:hh2, :ww2])
    out = Image.fromarray(np.clip(pg, 0, 255).astype(np.uint8))
    dx, dy = rnd.uniform(-6, 6), rnd.uniform(-6, 6)
    out = out.rotate(rnd.uniform(-0.6, 0.6), resample=Image.BILINEAR, translate=(dx, dy), fillcolor=245)
    if scale != 1.0:
        out = out.resize((int(out.width * scale), int(out.height * scale)), Image.BICUBIC)
    return out


if __name__ == "__main__":
    main()
