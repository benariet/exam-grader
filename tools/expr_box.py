"""The Boolean expressions students write in a printed answer box - 'Vx = ... / Y1 = ... / Y2 = ...', the course's
Moed C 1(a) - read off the scans and checked against the solution's, on this computer (CPU: the booklet's template,
the formula reader recog/formula.py on ONNX Runtime).

The box is found on the booklet's template of the page (a ruled rectangle, a printed label 'name =' starting each of
its rows), laid on each scan (recog/printed.py's map), the print erased; each row's handwriting, right of its label,
cut out and read as LaTeX (the reader's likeliest readings), parsed (grader.circuit_check.parse_bool) and compared
with the solution's function for that row - a name the solution gives ('Vx' in 'Y1 = Vx ⊕ C') standing for the
solution's own function of it (the circuit's node).

    python tools/expr_box.py "<exams' PDFs>" --booklet booklet/moedC --page 5 \\
        --key "Vx = A ⊕ B; Y1 = A ⊕ B ⊕ C; Y2 = AB + AC + BC" --out <folder> [--grades "grades.xlsx" --question 1]
        -> how many rows right / wrong / unsure / empty (beside the points the lecturer took off the question);
           <out>/<n>.exprs.json per exam and <n>_expr_remarks.txt (Hebrew)
    python tools/expr_box.py --synth <tools/bool_synth.py's folder> --key ...   (the same reading of made-up ones:
        how often each verdict is right)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

N_BEST = 4
SURE_P = 0.5            # the weakest token of a reading less likely than this: not sure enough for a 'wrong'
MIN_INK = 30            # handwriting pixels in a row: fewer, an empty row


# ---- the box on the template

def answer_rows(template: np.ndarray, n_rows: int = 0):
    """the template's answer box (bool, ink True): a ruled rectangle with no ruled lines inside (not a table) whose
    rows each start with a printed label -> [{'box', 'label', 'x0', 'x1', 'y0', 'y1', 'base'}] top to bottom (x0: just
    past the label; y0-y1: the row's band, between its neighbours; base: the label's bottom), [] if none. n_rows: the
    box with that many rows, when the page has more than one"""
    from scipy import ndimage as ndi
    horiz = ndi.binary_opening(ndi.binary_closing(template, structure=np.ones((1, 5))),   # (gaps of a pixel or two
                               structure=np.ones((1, 150)))    # closed - not a word's letters into a line)
    vert = ndi.binary_opening(ndi.binary_closing(template & ~horiz, structure=np.ones((9, 1))),   # (a letter's stem
                              structure=np.ones((40, 1)))      # touching a thick ruled line is no ruled line)
    segs = lambda m: [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop) for sl in ndi.find_objects(ndi.label(m)[0])]  # noqa: E731
    hs = [s for s in segs(horiz) if s[2] - s[0] >= 150]
    vs = [s for s in segs(vert) if s[3] - s[1] >= 40]
    found = []
    for t in hs:                                       # (a top line, a bottom line as long, a side line at each end)
        for u in hs:
            y0, y1 = (t[1] + t[3]) // 2, (u[1] + u[3]) // 2
            if y1 - y0 < 40 or abs(t[0] - u[0]) > 10 or abs(t[2] - u[2]) > 10:
                continue
            x0, x1 = min(t[0], u[0]), max(t[2], u[2])
            side = lambda x: sum(max(0, min(v[3], y1) - max(v[1], y0)) for v in vs   # noqa: E731
                                 if abs((v[0] + v[2]) / 2 - x) <= 10) / (y1 - y0)
            if min(side(x0), side(x1)) < 0.5 or max(side(x0), side(x1)) < 0.8:
                continue                               # (a side may come out broken in a template: half of it)
            if any(y0 + 6 < (w[1] + w[3]) / 2 < y1 - 6 and min(w[2], x1) - max(w[0], x0) >= 0.5 * (x1 - x0) for w in hs) or \
                    any(x0 + 6 < (v[0] + v[2]) / 2 < x1 - 6 and min(v[3], y1) - max(v[1], y0) >= 0.5 * (y1 - y0) for v in vs):
                continue                               # (a table: ruled lines inside)
            if any(f[0]["box"] == [x0, y0, x1, y1] for f in found):
                continue
            rows = _label_rows(template & ~ndi.binary_dilation(horiz | vert, iterations=2), (x0, y0, x1, y1))
            if rows:
                found.append(rows)
    if n_rows:
        found = [r for r in found if len(r) == n_rows] or found
    return found[0] if found else []


def _label_rows(template, box):
    """the box's printed rows (template: its ruled lines taken out)"""
    from scipy import ndimage as ndi
    x0, y0, x1, y1 = box
    m = 3
    sub = template[y0 + m:y1 - m, x0 + m:x1 - m]
    lab, n = ndi.label(sub, structure=np.ones((3, 3)))
    parts = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        if int((lab[sl] == k + 1).sum()) >= 40:        # (the print's specks - a template's leftovers - left out)
            parts.append([sl[1].start + x0 + m, sl[0].start + y0 + m, sl[1].stop + x0 + m, sl[0].stop + y0 + m])
    parts.sort(key=lambda p: p[1])
    rows = []
    for p in parts:                                    # (one over the other's height: the same row)
        for r in rows:
            if min(r[3], p[3]) - max(r[1], p[1]) >= 0.4 * min(r[3] - r[1], p[3] - p[1]):
                r[0], r[1], r[2], r[3] = min(r[0], p[0]), min(r[1], p[1]), max(r[2], p[2]), max(r[3], p[3])
                r[4].append(p)
                break
        else:
            rows.append([p[0], p[1], p[2], p[3], [p]])
    out = []
    for r in sorted(rows, key=lambda r: r[1]):
        ps = sorted(r[4])
        if ps[0][0] - x0 > 0.3 * (x1 - x0):            # (a label starts the row)
            continue
        th = r[3] - r[1]
        end = ps[0][2]
        for p in ps[1:]:                               # (the label: the parts close together from the left)
            if p[0] - end > 0.7 * th:
                break
            end = max(end, p[2])
        out.append({"box": [x0, y0, x1, y1], "label": [ps[0][0], r[1], end, r[3]], "x0": end + 3, "x1": x1 - 4,
                    "base": r[3], "mid": (r[1] + r[3]) / 2})
    if len(out) < 1:
        return []
    for i, r in enumerate(out):
        r["y0"] = y0 + 4 if i == 0 else int((out[i - 1]["mid"] + r["mid"]) / 2)
        r["y1"] = y1 - 4 if i == len(out) - 1 else int((r["mid"] + out[i + 1]["mid"]) / 2)
    return out


def blank_answers(template: np.ndarray, rows) -> np.ndarray:
    """the template with the rows' answer areas blank: a template built from students' scans keeps specks of their
    writing there, which laid on a scan would erase the writing over them as print"""
    t = template.copy()
    for r in rows:
        t[r["y0"]:r["y1"], r["x0"]:r["x1"]] = False
    return t


# ---- a row's handwriting on a scan

def soft_hand(gray: np.ndarray, printed: np.ndarray, dark: int = 128, light: int = 175) -> np.ndarray:
    """the handwriting with its light edges: pixels lighter than the ink threshold (a pencil's, a thin pen's) kept
    where they join dark ones - the print left out"""
    from scipy import ndimage as ndi
    soft = (gray < light) & ~printed
    lab, n = ndi.label(soft, structure=np.ones((3, 3)))
    if not n:
        return soft
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[(gray < dark) & soft])] = True
    keep[0] = False
    return keep[lab]


def row_pieces(hand: np.ndarray, rows_scan):
    """the scan's handwriting (bool) -> [the row's handwriting mask (bool, the page's size, cropped to its box)
    or None] per row: the handwriting inside the box right of the labels, each piece to the row its middle is in (a
    flat stroke - a bar over a letter - to the row of the writing under it); pieces running out of the box kept whole,
    and the writing beside the box's right side that continues a row"""
    from scipy import ndimage as ndi
    bx0, by0, bx1, by1 = rows_scan[0]["box"]
    xs0 = min(r["x0"] for r in rows_scan)
    W = bx1 - bx0
    region = np.zeros_like(hand)
    ry0, ry1 = max(0, by0 - 2 * (by1 - by0) // 10), min(hand.shape[0], by1 + (by1 - by0) // 5)
    region[ry0:ry1, xs0:min(hand.shape[1], bx1 + int(0.6 * W))] = True
    mine = hand & region
    lab, n = ndi.label(ndi.binary_dilation(mine, iterations=1), structure=np.ones((3, 3)))   # (a stroke broken
    lab[~mine] = 0                                                                            #  a pixel apart: one)
    rh = float(np.median([r["y1"] - r["y0"] for r in rows_scan]))
    comps = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        size = int((lab[sl] == k + 1).sum())
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if size < 8:
            continue
        if (x1 - x0 >= 0.8 * W and y1 - y0 <= 5) or (y1 - y0 >= 0.8 * (by1 - by0) and x1 - x0 <= 5):
            continue                                   # (a ruled line's remnant)
        inside = x0 < bx1 - 3 and x1 > xs0 and y0 < by1 - 3 and y1 > by0 + 3
        comps.append({"k": k + 1, "sl": sl, "box": (x0, y0, x1, y1), "inside": inside,
                      "flat": (y1 - y0) <= 0.3 * rh and (x1 - x0) >= 2 * (y1 - y0)})

    def row_of(y):
        for i, r in enumerate(rows_scan):
            if r["y0"] <= y < r["y1"]:
                return i
        return None
    owner = {}
    for c in comps:                                    # (inside the box, not flat: by its middle)
        if c["inside"] and not c["flat"]:
            i = row_of((c["box"][1] + c["box"][3]) / 2)
            if i is not None:
                owner[c["k"]] = i
    for c in comps:                                    # (flat inside: the row of the writing under it)
        if c["inside"] and c["flat"] and c["k"] not in owner:
            x0, y0, x1, y1 = c["box"]
            below = [(o["box"][1] - y1, owner[o["k"]]) for o in comps if o["k"] in owner and
                     min(o["box"][2], x1) - max(o["box"][0], x0) > 0 and o["box"][1] >= y1 - 2]
            if below:
                owner[c["k"]] = min(below)[1]
            else:
                i = row_of((y0 + y1) / 2)
                if i is not None:
                    owner[c["k"]] = i
    changed = True
    while changed:                                     # (outside on the right: continuing a row's writing)
        changed = False
        for c in comps:
            if c["k"] in owner or c["inside"]:
                continue
            i = row_of((c["box"][1] + c["box"][3]) / 2)
            if i is None:
                continue
            right = max((o["box"][2] for o in comps if owner.get(o["k"]) == i), default=None)
            if right is not None and c["box"][0] - right <= 1.5 * rh:
                owner[c["k"]] = i
                changed = True
    out = []
    for i in range(len(rows_scan)):
        ks = [k for k, v in owner.items() if v == i]
        if not ks:
            out.append(None)
            continue
        m = np.isin(lab, ks)
        if int(m.sum()) < MIN_INK:
            out.append(None)
            continue
        out.append(m)
    return out


def crop_of(gray: np.ndarray, mask: np.ndarray, pad_rel=0.35, max_aspect=0.0, hi=None):
    """the row's handwriting alone (its gray values where the mask is, white elsewhere) with a margin -> PIL image;
    hi: the page at a higher resolution (cleaned as the reading one), the gray values taken from it"""
    from PIL import Image
    from scipy import ndimage as ndi
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if hi is not None:
        k = hi.shape[1] / gray.shape[1]
        sub = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
        Y0, X0 = int(y0 * k), int(x0 * k)
        Y1, X1 = min(hi.shape[0], int(np.ceil(y1 * k))), min(hi.shape[1], int(np.ceil(x1 * k)))
        mk = np.asarray(sub.resize((X1 - X0, Y1 - Y0), Image.NEAREST)) > 0
        mk = ndi.binary_dilation(mk, iterations=max(1, int(round(k))))
        piece = np.where(mk, hi[Y0:Y1, X0:X1], 255).astype(np.uint8)
    else:
        mk = ndi.binary_dilation(mask[y0:y1, x0:x1], iterations=1)
        piece = np.where(mk, gray[y0:y1, x0:x1], 255).astype(np.uint8)
    h, w = piece.shape
    pad = int(pad_rel * h) + 4
    H, W = h + 2 * pad, w + 2 * pad
    if max_aspect and W / H > max_aspect:
        H = int(W / max_aspect)
    canvas = np.full((H, W), 255, np.uint8)
    oy, ox = (H - h) // 2, pad
    canvas[oy:oy + h, ox:ox + w] = piece
    return Image.fromarray(canvas)


def box_view(img: np.ndarray, width: int, rows_scan):
    """the whole answer box as scanned - its print, its labels, the writing running out of it - for a reader that reads
    the rows together (tools/read_crops.py's 'box' prompt); img: the page (at any resolution; width: the reading
    resolution's page width, the rows' coordinates' frame)"""
    from PIL import Image
    k = img.shape[1] / width
    bx0, by0, bx1, by1 = rows_scan[0]["box"]
    W, H = bx1 - bx0, by1 - by0
    x0, y0 = max(0, int((bx0 - 10) * k)), max(0, int((by0 - H / 4) * k))
    x1, y1 = min(img.shape[1], int((bx1 + 0.6 * W) * k)), min(img.shape[0], int((by1 + H / 4) * k))
    return Image.fromarray(img[y0:y1, x0:x1])


def cleaned_hires(doc, pno: int, dpi: int) -> np.ndarray:
    """the scan's page rendered at dpi, its paper white and its ink darkened as recog.page.clean_scan does"""
    from recog.page import ink_gray, ink_threshold
    g = ink_gray(doc[pno - 1].render(scale=dpi / 72).to_pil().convert("RGB"))
    t = ink_threshold(g)
    return np.where(g >= t, 255, g * (110.0 / t)).astype(np.uint8)


# ---- the readings, parsed and judged

_W = r"\\math(?:rm|cal|bf|it|sf)\s*\{"                  # (the reader's '\mathrm{V}': a letter as it likes to write it)
GRAMMAR = (r"(?(DEFINE)(?P<vx>(?:[Vv]|" + _W + r"\s*[Vv]\s*\})\s*_\s*(?:\{\s*(?:[xX]|" + _W + r"\s*[xX]\s*\})\s*\}|[xX]))"
           r"(?P<neg>(?:\s*(?:'|\^\s*\{\s*(?:\\prime\s*)+\}|\^\s*\\prime))*)"
           r"(?P<atom>(?:[ABC]|(?&vx)|\(\s*(?&expr)\s*\)|\{\s*(?&expr)\s*\}|\\overline\s*\{\s*(?&expr)\s*\}|"
           + _W + r"\s*(?&expr)\s*\})(?&neg))"
           r"(?P<term>(?&atom)(?:\s*(?:\\cdot\s*)?(?&atom))*)"
           r"(?P<expr>(?&term)(?:\s*(?:\+|\\oplus)\s*(?&term))*))\s*(?&expr)\s*",
           "ABCVvxX_{}()'^+ \\oplusverlinecdtpmahbfs")
# (the reader made to read only a Boolean expression of the inputs and Vx: letters side by side, +, xor, a prime or a
#  bar, brackets - recog/formula.py's grammar; its characters: the tokens it may use)


def _ph(j):
    """the placeholder put for the solution's j-th name in a reading: a letter with a subscript no hand writes"""
    return f"Z_{{{j}}}"                                 # (one character: parse_bool's subscripts)


def normalize(tex: str, names=("Vx",), inputs="ABC") -> str:
    """the reader's LaTeX as grader.circuit_check.parse_bool reads it: primes ('A^{\\prime}' -> "A'"), '\\mathrm{..}'
    unwrapped, an input letter read as a subscript ('A_{B}': two letters side by side, as written) put back beside, a
    name of the solution's ('V_{x}', 'Vx') as a placeholder (_ph) to put the solution's function in for"""
    t = tex
    for _ in range(2):
        t = re.sub(r"\\(?:mathrm|mathit|mathbf|mathsf|mathcal|boldsymbol|text|operatorname)\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\^\s*\{?\s*((?:\\prime\s*)+)\}?", lambda m: "'" * m.group(1).count("\\prime"), t)
    t = re.sub(r"\^\s*\{\s*('+)\s*\}", r"\1", t)
    t = t.replace("\\prime", "'").replace("\u2032", "'").replace("`", "'")
    t = re.sub(r"\\(?:,|;|:|!|quad|qquad)|~", " ", t)
    for j, nm in enumerate(names):
        if len(nm) == 2:                               # 'Vx': V_{x}, V_x, V x, Vx, v_x
            pat = rf"(?:{nm[0]}|{nm[0].lower()})\s*(?:_\s*\{{\s*{nm[1]}\s*\}}|_\s*{nm[1]}|\s*{nm[1]})"
            t = re.sub(pat, " " + _ph(j) + " ", t, flags=re.I if nm[1].isalpha() else 0)
    ins = re.escape(inputs + inputs.lower())
    t = re.sub(rf"(?<=[A-Za-z}}'])\s*_\s*\{{\s*([{ins}](?:\s*[{ins}])*)\s*\}}", r" \1", t)
    t = re.sub(rf"(?<=[A-Za-z}}'])\s*_\s*([{ins}])", r" \1", t)
    return t


_BAR1 = r"\\overline\s*\{\s*\{?\s*([A-Za-z])\s*\}?\s*\}"
_BARN = re.compile(r"\\overline\s*\{\s*((?:\{?\s*[A-Za-z]\s*\}?\s*){2,})\}")


def bar_variants(tex: str):
    """the readings a hand's bars leave open: a bar over two letters side by side ('\\overline{AB}') or a bar over
    each ('\\overline{A}\\overline{B}') - bars a hair apart read as one, one bar read as two"""
    out = []
    split = _BARN.sub(lambda m: " ".join(f"\\overline{{{c}}}" for c in re.findall(r"[A-Za-z]", m.group(1))), tex)
    if split != tex:
        out.append(split)
    merged = re.sub(rf"(?:{_BAR1}\s*){{2,}}", lambda m: "\\overline{" + "".join(re.findall(_BAR1, m.group(0))) + "}", tex)
    if merged != tex:
        out.append(merged)
    return out


def parsed_functions(cands, env, inputs="ABC"):
    """the candidates that parse as a function of the inputs (the solution's names put in) -> [(candidate, SymPy)]; a
    chain ('A \\oplus B = A'B + AB'') its last side"""
    import sympy as sp
    from grader.circuit_check import parse_bool
    names = list(env)
    parsed = []
    for c in cands:
        t = normalize(c.text, names, inputs)
        parts = [p for p in t.split("=") if p.strip()]
        fs = []
        for p in parts:
            e = parse_bool(p)
            if e is None:
                fs = None
                break
            e = e.xreplace({x: sp.Symbol(str(x).upper()) for x in e.free_symbols})
            e = e.xreplace({sp.Symbol(_ph(j).replace("{", "").replace("}", "")): env[n] for j, n in enumerate(names)})
            if not {str(s) for s in e.free_symbols} <= set(inputs):
                fs = None
                break
            fs.append(e)
        if fs:
            parsed.append((c, fs[-1]))
    return parsed


def judge(cands, want, env, inputs="ABC", variants=True):
    """the reader's candidates (recog.formula.Candidate, likeliest first) -> (verdict, the reading used, the function
    read): 'right' (the likeliest reading that parses gives the function), 'wrong' (every reading that parses gives
    another, the likeliest sure), else 'unsure'"""
    import sympy as sp
    from sympy.logic.boolalg import Xor
    parsed = parsed_functions(cands, env, inputs)
    if not parsed:
        return "unsure", cands[0].text if cands else "", None
    same = lambda f: not sp.satisfiable(Xor(f, want))  # noqa: E731
    c0, f0 = parsed[0]
    if same(f0):
        return "right", c0.text, str(f0)
    if any(same(f) for _, f in parsed):
        return "unsure", c0.text, str(f0)
    if c0 is cands[0] and c0.min_token_p >= SURE_P:
        if _BARN.search(c0.text):                      # (a bar over letters side by side: how far it reaches, the
            return "unsure", c0.text, str(f0)          #  reader often misjudges - '\\overline{A}B' read '\\overline{AB}')
        alt = [type(c0)(t, c0.logp, c0.min_token_p) for t in bar_variants(c0.text)] if variants else []
        if any(judge([v], want, env, inputs, False)[0] == "right" for v in alt):
            return "unsure", c0.text, str(f0)          # (the function, its bars read the other way)
        return "wrong", c0.text, str(f0)
    return "unsure", c0.text, str(f0)


def read_row(fr, img, want, env, inputs="ABC", grammar=True):
    """a row's handwriting read twice - freely, and made to read a Boolean expression (GRAMMAR) - and judged ->
    {'verdict', 'read', 'function', 'free', 'forced'}: right when the free reading gives the function, or the forced
    one does and the free one is no sure other function; wrong when the free reading is surely another function and
    the forced one is not the function either; else unsure"""
    free = fr.recognize(img, n_best=N_BEST)
    vf, rf, ff = judge(free, want, env, inputs)
    out = {"free": free[0].text if free else "", "forced": ""}
    if vf == "right":
        return {**out, "verdict": "right", "read": rf, "function": ff}
    if not grammar:
        return {**out, "verdict": vf, "read": rf, "function": ff}
    longest = min(80, len(fr.tok.encode(free[0].text).ids) + 12) if free else 60   # (a reading forced off the
    forced = fr.recognize(img, n_best=3, grammar=GRAMMAR, max_new_tokens=longest)     # model's way can run on)
    vg, rg, fg = judge(forced, want, env, inputs)
    out["forced"] = forced[0].text if forced else ""
    if vg == "right":
        sure_other = vf == "wrong"
        return {**out, "verdict": "unsure" if sure_other else "right", "read": rg, "function": fg}
    if vf == "wrong" and vg in ("wrong", "unsure") and fg is not None:
        return {**out, "verdict": "wrong", "read": rf, "function": ff}
    return {**out, "verdict": "unsure", "read": rf or rg, "function": ff or fg}


def truth_vector(f, inputs="ABC"):
    """a function's values in a truth table's order: the inputs counting up in binary, the first the highest bit"""
    import sympy as sp
    syms = [sp.Symbol(x) for x in inputs]
    n = len(syms)
    return tuple(int(bool(f.subs({s: (r >> (n - 1 - i)) & 1 for i, s in enumerate(syms)}))) for r in range(2 ** n))


def second_opinion(res, extra, want, env, inputs="ABC", table=None):
    """res: read_row's verdict (Pix2Text's readings); extra: {reader: [Candidate]} - the GPU readers' (Uni-MuMER, the
    vision model; tools/read_crops.py); table: the student's own truth-table column for the row, when every cell of it
    was read (tools/truth_table.py) -> the verdict with them. The readers' likeliest readings that parse, grouped by the
    function they give: two or more giving the same one decide - right if it is the solution's; wrong if not, unless
    another reader gives the solution's, a bar over letters side by side is in their readings (how far it reaches is
    open), or the student's own table says otherwise (an expression against its own table: to a human). Without two
    agreeing, Pix2Text's own verdict, its 'wrong' unsure when another reader gives the solution's function"""
    import sympy as sp
    from sympy.logic.boolalg import Xor
    same = lambda f, g: not sp.satisfiable(Xor(f, g))  # noqa: E731
    tops = []                                          # [(reader, reading, function)]
    p2t = [type("C", (), {"text": t, "logp": 0.0, "min_token_p": 1.0})() for t in (res.get("free"), res.get("forced")) if t]
    got = parsed_functions(p2t, env, inputs)
    if got:
        tops.append(("pix2text", got[0][0].text, got[0][1]))
    views = {}                                         # (a reader at several scales: the function most views give)
    for name, cands in (extra or {}).items():
        got = parsed_functions(cands, env, inputs)
        if got:
            views.setdefault(name.split("@")[0], []).append((name, got[0][0].text, got[0][1]))
    for name, vs in views.items():
        best = max(vs, key=lambda v: (sum(1 for w in vs if same(w[2], v[2])), v[0] == name))
        tops.append((name, best[1], best[2]))
    out = dict(res)
    out["readers"] = {n: t for n, t, _ in tops}
    groups = []
    for n, t, f in tops:
        for g in groups:
            if same(g[0][2], f):
                g.append((n, t, f))
                break
        else:
            groups.append([(n, t, f)])
    family = lambda g: {n.replace("box", "") for n, _, _ in g}          # noqa: E731  (the vision model on the row and
    groups.sort(key=lambda g: (-len(family(g)), not same(g[0][2], want)))  # on the whole box: one reader)
    any_right = any(same(f, want) for _, _, f in tops)
    if groups and len(family(groups[0])) < 2 and table is not None:      # (the student's own table as the second:
        for n, t, f in tops:                                             #  a reading of Pix2Text's or Uni-MuMER's
            if n in ("pix2text", "mumer") and truth_vector(f, inputs) == tuple(table):   # giving its function -
                groups.insert(0, [(n, t, f), ("table", "", f)])          #  not the vision model's, which writes a
                break                                                    #  likely answer where it cannot read)
    if groups and len(family(groups[0])) >= 2:
        g = groups[0]
        f1 = g[0][2]
        out.update(read=g[0][1], function=str(f1), agreed=[n for n, _, _ in g])
        if table is not None:
            out["matches_table"] = truth_vector(f1, inputs) == tuple(table)
        if any_right and not same(f1, want) and g[-1][0] == "table":
            out["verdict"] = "unsure"
            return out
        if same(f1, want):
            out["verdict"] = "right"
        elif any_right or any(_BARN.search(t) for _, t, _ in g) or out.get("matches_table") is False:
            out["verdict"] = "unsure"
        else:
            out["verdict"] = "wrong"
        return out
    if res["verdict"] == "wrong" and any_right:
        out["verdict"] = "unsure"
    return out


def parse_key(key: str):
    """'Vx = A ⊕ B; Y1 = ...' -> [(name, SymPy function)], each earlier name put in for in the later ones"""
    import sympy as sp
    from grader.circuit_check import parse_bool
    out, env = [], {}
    for part in key.split(";"):
        name, expr = (s.strip() for s in part.split("=", 1))
        t = normalize(expr, list(env))
        e = parse_bool(t)
        if e is None:
            raise SystemExit(f"the key's '{part.strip()}' does not parse")
        e = e.xreplace({sp.Symbol(_ph(j).replace("{", "").replace("}", "")): env[n] for j, n in enumerate(env)})
        out.append((name, e))
        env[name] = e
    return out


def own_table(folder, exam, col, n_rows):
    """the student's own truth-table column (tools/truth_table.py's <exam>.table.json), when every cell of it was read
    surely (right or wrong against the key) -> (values), else None"""
    path = os.path.join(folder, f"{exam}.table.json")
    if col is None or not os.path.isfile(path):
        return None
    cells = {(c["row"], c["col"]): c for c in json.load(open(path, encoding="utf-8"))["cells"]}
    got = [cells.get((r + 1, col)) for r in range(n_rows)]
    if not all(c and c["verdict"] in ("right", "wrong") for c in got):
        return None
    return tuple(int(c["read"]) for c in got)


def _review_figure(img, rows_scan, masks, out, name, off):
    """one exam's box for review.html: the scan round it, each row's writing in its verdict's colour"""
    import base64
    import html
    import io
    from PIL import Image
    colors = {"right": (20, 160, 60), "wrong": (220, 30, 30), "unsure": (240, 140, 0)}
    arr = np.asarray(img).copy()
    for m, o in zip(masks, out):
        if m is not None and o["verdict"] in colors:
            arr[m] = colors[o["verdict"]]
    bx0, by0, bx1, by1 = rows_scan[0]["box"]
    W, H = bx1 - bx0, by1 - by0
    crop = Image.fromarray(arr).crop((max(0, bx0 - 20), max(0, by0 - H // 3), min(arr.shape[1], bx1 + int(0.6 * W)),
                                      min(arr.shape[0], by1 + H // 3)))
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    heb = {"right": "נכון", "wrong": "שגוי", "unsure": "לא בטוח", "empty": "ריק"}
    lines = [f"מבחן {name}" + (f" (המרצה הוריד בשאלה כולה {off:g})" if off is not None else "")]
    for o in out:
        lines.append(f"{o['name']}: {heb[o['verdict']]}" + (f" - נקרא: {o['read']}" if o["read"] else ""))
    return (f"<figure><img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
            f"<figcaption>{html.escape(chr(10).join(lines))}</figcaption></figure>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="")
    ap.add_argument("--booklet", default="")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--key", required=True)
    ap.add_argument("--inputs", default="ABC")
    ap.add_argument("--grades", default="")
    ap.add_argument("--question", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--synth", default="", help="tools/bool_synth.py's folder: its expressions read and judged instead")
    ap.add_argument("--max-aspect", type=float, default=0.0)
    ap.add_argument("--crops-only", default="", help="a folder: each row's handwriting cut out saved there as "
                    "<exam>_<row>.png (for tools/read_crops.py), nothing read")
    ap.add_argument("--table", default="", help="tools/truth_table.py's folder for the same exams: a row's function "
                    "against the student's own table column (--table-cols)")
    ap.add_argument("--table-cols", default="", help="each row's column in that table: 'Vx=2,Y1=4,Y2=5'")
    ap.add_argument("--reuse", default="", help="an earlier run's folder: Pix2Text's readings taken from it, not read "
                    "again (to judge them anew - with --second, --table)")
    ap.add_argument("--second", default="", help="tools/read_crops.py's readings of those crops: judged with them too")
    ap.add_argument("--max-seconds", type=int, default=0, help="stop starting new exams after this long (run again to "
                    "go on: an exam read is not read again)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-grammar", action="store_true", help="the free reading only")
    ap.add_argument("--review", action="store_true", help="<out>/review.html: each exam's box as scanned, each row's "
                    "writing coloured by its verdict, with what was read (to look over, on this computer)")
    ap.add_argument("--hires", type=int, default=300, help="the handwriting cut from the page rendered at this "
                    "resolution (the scans' own; 0: at the reading resolution, 150 dpi)")
    a = ap.parse_args()
    from recog.formula import FormulaRecognizer
    key = parse_key(a.key)
    names = [n for n, _ in key]
    env = dict(key)
    fr = FormulaRecognizer()
    if a.synth:
        return synth_eval(a, fr, key, env)
    from PIL import Image
    from recog.page import clean_scan
    from recog.printed import fit_map, printed_on
    from tools.truth_table import to_scan
    template = np.asarray(Image.open(os.path.join(a.booklet, f"p{a.page}.png")).convert("L")) < 128
    rows = answer_rows(template, len(key))
    if len(rows) != len(key):
        raise SystemExit(f"no answer box of {len(key)} rows on {a.booklet}/p{a.page}.png ({len(rows)} rows found)")
    blank = blank_answers(template, rows)
    second = json.load(open(a.second, encoding="utf-8")) if a.second else None
    tcols = {k.strip(): int(v) for k, v in (x.split("=") for x in a.table_cols.split(",") if "=" in x)}
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    pdfs = sorted((p for p in glob.glob(os.path.join(a.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    import pypdfium2 as pdfium
    table, review = [], []
    t_start = time.time()
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        if only and name not in only:
            continue
        done = os.path.join(a.out or ".", f"{name}.exprs.json")
        if os.path.isfile(done) and not a.crops_only:                       # (resumable: an exam read before is taken as it was)
            r0 = json.load(open(done, encoding="utf-8"))
            table.append(({k: r0[k] for k in ("right", "wrong", "unsure", "empty")},
                          marks.get(name, {}).get(a.question) if marks else None))
            continue
        if a.max_seconds and time.time() - t_start > a.max_seconds:
            print(f"stopped after {a.max_seconds} s - run again to go on", flush=True)
            break
        doc = pdfium.PdfDocument(pdf)
        if len(doc) < a.page:
            continue
        img = doc[a.page - 1].render(scale=150 / 72).to_pil().convert("RGB")
        page, _ = clean_scan(img)
        gray = np.asarray(page.convert("L"))
        ink = gray < 128
        hi = cleaned_hires(doc, a.page, a.hires) if a.hires > 150 else None
        M, b = fit_map(template, ink)
        hand = soft_hand(gray, printed_on(blank, ink))
        rs = []
        for r in rows:
            q = dict(r)
            q["box"] = list(to_scan(r["box"], M, b))
            q["x0"], q["y0"], q["x1"], q["y1"] = to_scan((r["x0"], r["y0"], r["x1"], r["y1"]), M, b)
            rs.append(q)
        masks = row_pieces(hand, rs)
        if a.crops_only:
            os.makedirs(a.crops_only, exist_ok=True)
            for (nm, _), mk in zip(key, masks):
                if mk is not None:
                    crop_of(gray, mk, max_aspect=a.max_aspect, hi=hi).save(os.path.join(a.crops_only, f"{name}_{nm}.png"))
            box_view(hi if hi is not None else gray, gray.shape[1], rs).save(os.path.join(a.crops_only, f"{name}_box.png"))
            continue
        res = {"right": 0, "wrong": 0, "unsure": 0, "empty": 0}
        out, remarks = [], []
        old = os.path.join(a.reuse, f"{name}.exprs.json") if a.reuse else ""
        reused = {x["name"]: x for x in json.load(open(old, encoding="utf-8"))["rows"]
                  if x["verdict"] != "empty"} if old and os.path.isfile(old) else {}
        for (nm, want), mk in zip(key, masks):
            if mk is None:
                v, used, f = "empty", "", None
            else:
                before = reused.get(nm)
                r = {k_: before[k_] for k_ in ("verdict", "read", "function", "free", "forced")} if before else \
                    read_row(fr, crop_of(gray, mk, max_aspect=a.max_aspect, hi=hi), want,
                             {n: env[n] for n in names if n != nm}, a.inputs, not a.no_grammar)
                tv = own_table(a.table, name, tcols.get(nm), 2 ** len(a.inputs)) if a.table and nm in tcols else None
                if second is not None:
                    ex = {rd: [type("C", (), {"text": t, "logp": lp, "min_token_p": 1.0})() for t, lp in cs]
                          for rd, cs in second.get(f"{name}_{nm}", {}).items()}
                    r = second_opinion(r, ex, want, {n: env[n] for n in names if n != nm}, a.inputs, tv)
                v, used, f = r["verdict"], r["read"], r["function"]
            res[v] += 1
            out.append({"name": nm, "verdict": v, "read": used, "function": f,
                        "free": r["free"] if mk is not None else "", "forced": r["forced"] if mk is not None else "",
                        "readers": r.get("readers", {}) if mk is not None else {},
                        "agreed": r.get("agreed", []) if mk is not None else [],
                        "matches_table": r.get("matches_table") if mk is not None else None})
            if v == "wrong":
                remarks.append(f"{nm}: נקרא {used} - שונה מהפתרון ({nm} = {want}).")
            if mk is not None and r.get("matches_table") is False and v == "right":
                remarks.append(f"{nm}: הביטוי נכון, אך עמודת {nm} בטבלת האמת של הסטודנט אינה תואמת אותו.")
        json.dump({"exam": name, "rows": out, **res}, open(done + ".tmp", "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        txt = [f"שאלה {a.question}(א) - הביטויים: {res['right']} נכונים, {res['wrong']} שגויים, {res['empty']} ריקים, "
               f"{res['unsure']} לא נקראו בוודאות (לבדיקה ידנית)."] + remarks
        open(os.path.join(a.out, f"{name}_expr_remarks.txt"), "w", encoding="utf-8").write("\n".join(txt) + "\n")
        off = marks.get(name, {}).get(a.question) if marks else None
        table.append((res, off))
        if a.review:
            open(os.path.join(a.out, f"{name}.review.part"), "w", encoding="utf-8").write(
                _review_figure(img, rs, masks, out, name, off))
        os.replace(done + ".tmp", done)
    if a.review:
        for pdf in pdfs:
            part = os.path.join(a.out, os.path.splitext(os.path.basename(pdf))[0] + ".review.part")
            if os.path.isfile(part):
                review.append(open(part, encoding="utf-8").read())
        page_html = ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Expressions review</title>",
                     "<style>body{font-family:Arial,sans-serif;margin:16px}figure{margin:12px 0}img{max-width:100%}"
                     "figcaption{font-size:14px;white-space:pre-wrap}</style>",
                     "<h1>הביטויים שבתיבה - מה נקרא</h1><p>ירוק: נכון לפי הפתרון. אדום: שגוי. כתום: לא נקרא בוודאות. "
                     "מתחת לכל תיבה: מה נקרא בכל שורה.</p>"] + review + ["</html>"]
        open(os.path.join(a.out, "review.html"), "w", encoding="utf-8").write("\n".join(page_html))
    tot = {k: sum(r[k] for r, _ in table) for k in ("right", "wrong", "unsure", "empty")}
    print(f"{len(table)} exams, {sum(tot.values())} rows: " + ", ".join(f"{k} {v}" for k, v in tot.items()))
    if marks:
        for what, sel in (("no row wrong", lambda r: r["wrong"] == 0), ("a row wrong", lambda r: r["wrong"] >= 1),
                          ("all rows right", lambda r: r["right"] == len(key))):
            offs = [o for r, o in table if sel(r) and o is not None]
            if offs:
                print(f"  {what}: {len(offs)} exams, mean points off question {a.question} {sum(offs) / len(offs):.1f}")


def synth_eval(a, fr, key, env):
    """tools/bool_synth.py's expressions read as the exams' rows are: how often each verdict is right"""
    import sympy as sp
    from PIL import Image
    from sympy.logic.boolalg import Xor
    files = sorted(glob.glob(os.path.join(a.synth, "*.json")), key=lambda f: int(re.sub(r"\D", "", os.path.basename(f)) or 0))
    files = files[: a.limit or None]
    want_of = dict(key)
    second = json.load(open(a.second, encoding="utf-8")) if a.second else None
    stats = {}
    for f in files:
        meta = json.load(open(f, encoding="utf-8"))
        nm = meta["name"]
        img = np.asarray(Image.open(f[:-5] + ".png").convert("L"))
        mask = img < 200
        truth = sp.sympify(meta["function"], locals={n: f for n, f in env.items()} | {s: sp.Symbol(s) for s in "ABC"},
                           convert_xor=False)
        is_right = not sp.satisfiable(Xor(truth, want_of[nm]))
        r = read_row(fr, crop_of(img, mask, max_aspect=a.max_aspect), want_of[nm], {n: env[n] for n in env if n != nm},
                     a.inputs, not a.no_grammar)
        if second is not None:
            ex = {rd: [type("C", (), {"text": t, "logp": lp, "min_token_p": 1.0})() for t, lp in cs]
                  for rd, cs in second.get(os.path.basename(f)[:-5], {}).items()}
            r = second_opinion(r, ex, want_of[nm], {n: env[n] for n in env if n != nm}, a.inputs)
        v = r["verdict"]
        s = stats.setdefault(("right answer" if is_right else "wrong answer"), {})
        s[v] = s.get(v, 0) + 1
        print(f"{os.path.basename(f)[:-5]}\t{nm}\t{meta['latex']}\t|\t{r['free']}\t|\t{r['forced']}\t{v}", flush=True)
    for k, s in stats.items():
        print(f"{k}s ({sum(s.values())}): " + ", ".join(f"{v} {n}" for v, n in sorted(s.items())))


if __name__ == "__main__":
    main()
