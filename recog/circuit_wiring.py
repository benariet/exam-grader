"""From a drawn circuit's pen strokes and the boxes of its components to a netlist: what is left of the
strokes once the components, labels and crossovers are cut out is wire; wire pieces that touch (or nearly
touch: a pen lifted for a moment) are one node; each component's ends are the wire pieces that reach its box.

    from recog.circuit_wiring import strokes, netlist_from_boxes
    mask = strokes(img)                                   # pen strokes, True = ink (classical; a learned one later)
    net = netlist_from_boxes(mask, comps, texts)          # grader.circuit.Netlist
        comps: [(cls, (x0, y0, x1, y1))] - cls as tools/circuit_data.py UNIFIED (resistor, voltage_dc, gnd, crossover ...)
        texts: [((x0, y0, x1, y1), text[, how sure its reading is])] - value labels / names (text may be None)

Two-terminal parts (resistor, capacitor, inductor, sources, diode) get their two ends from the two sides of
their box that the wires reach most (left/right or top/bottom); a ground joins its node to node 0; a
crossover keeps the two wires that pass through it apart (left-right and top-bottom pieces are paired).

Labels: a value ('5V', '1k') or a name ('R2', 'E1') goes to the nearest component; a lone letter nearer a wire
than any component names that node ('A'); a current's name ('I1') goes with the current arrow nearest it.
Current arrows are found among the stroke pieces no component end reaches: long and thin, beside a two-terminal
part and along it, with a head (ink on both sides of the line) at one end - the current runs through that part
towards the head. A current name with no arrow found keeps its part but no direction (Netlist.problems flags it).
"""
from __future__ import annotations

import re

import numpy as np
from scipy import ndimage as ndi

TWO_ENDS = {"resistor", "capacitor", "inductor", "voltage_dc", "voltage_ac", "current_dc", "current_ac", "diode", "zener",
            "dep_voltage", "dep_current", "switch", "meter", "lamp"}
ONE_END = {"gnd", "voltage_rail", "terminal"}
KIND = {"resistor": "resistor", "capacitor": "capacitor", "inductor": "inductor", "voltage_dc": "voltage_source",
        "voltage_ac": "voltage_source", "current_dc": "current_source", "current_ac": "current_source",
        "dep_voltage": "dependent_voltage_source", "dep_current": "dependent_current_source",
        "diode": "diode", "zener": "zener", "voltage_rail": "voltage_source", "switch": "switch", "meter": "meter", "lamp": "lamp",
        "bjt": "bjt", "fet": "fet", "opamp": "opamp", "thyristor": "thyristor", "transformer": "transformer",
        "and": "and", "or": "or", "not": "not", "nand": "nand", "nor": "nor", "xor": "xor", "ic": "ic",
        "other": "other"}
PREFIX = {"resistor": "R", "capacitor": "C", "inductor": "L", "voltage_source": "V", "current_source": "I", "zener": "DZ",
          "dependent_voltage_source": "E", "dependent_current_source": "G", "diode": "D", "switch": "S",
          "meter": "M", "lamp": "LP", "bjt": "Q", "fet": "Q", "opamp": "OA", "thyristor": "TH", "transformer": "T",
          "and": "U", "or": "U", "not": "U", "nand": "U", "nor": "U", "xor": "U", "ic": "U", "other": "X"}
GATES = {"and", "or", "not", "nand", "nor", "xor"}
POLAR = {"voltage_dc": "voltage_dc", "current_dc": "current_dc", "diode": "diode", "zener": "diode",   # polarity read,
         "dep_voltage": "voltage_dc", "dep_current": "current_dc"}                    # by which polarity model
_NAME = re.compile(r"^[A-Za-z]{1,2}_?\{?\d{0,3}\}?$")
_VALUE = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*[pnuµμmkKMG]?\s*(?:Ω|ohm|V|A|F|H|Hz)?\s*$")
_CURRENT = re.compile(r"^[Ii](?:_?\{?[0-9A-Za-z]{1,3}\}?)?$")
# what a dependent source says it is: a gain (a number or a name: 'g_m', 'beta') times the voltage or current it
# follows ('2 I_x', '0.5V1', 'g_m V_gs', 'βI_b')
_DEP = re.compile(r"^\s*(?P<gain>\d+(?:[.,]\d+)?|[A-Za-zβαµμ]{1,4}(?:_?\{?[0-9A-Za-z]{1,2}\}?)?)?\s*[*·]?\s*"
                  r"(?P<var>[IiVv]_?\{?[0-9A-Za-z]{1,3}\}?)\s*$")
_NAME_KINDS = {"R": ("resistor",), "C": ("capacitor",), "L": ("inductor",), "D": ("diode", "zener"),
               "V": ("voltage_source",), "E": ("voltage_source",), "U": ("voltage_source",), "B": ("voltage_source",),
               "I": ("current_source",), "J": ("current_source",)}
_UNIT_KINDS = {"Ω": ("resistor",), "ohm": ("resistor",), "V": ("voltage_source", "zener"), "A": ("current_source",),
               "F": ("capacitor",), "H": ("inductor",)}


def _label_kinds(t: str, is_value: bool):
    """the part types a label fits ('R2', '10kΩ': resistors) - () when it says nothing (a bare number)"""
    if is_value:
        m = re.search(r"(Ω|ohm|V|A|F|H)\s*$", t)
        return _UNIT_KINDS.get(m.group(1), ()) if m else ()
    return _NAME_KINDS.get(t[0].upper(), ())
_NODE = re.compile(r"^[A-HJ-Za-hj-z]$")


def _thin_ink(page: np.ndarray) -> np.ndarray:
    """ink in thin strokes - darker than the paper a few pixels away - not a shadow's or a table's broad dark"""
    return (page < 205) & (page < ndi.maximum_filter(page, size=15) - 40)


def _lines(hor: np.ndarray, page: np.ndarray | None = None):
    """the long thin lines of a page's line mask (rule_lines) -> (labels, count, [line]); each line a dict: k (its
    label), box, fit (y = a x + b), edge (it reaches the picture's side, within 3 % of its width), across (it spans
    most of the width), thin (a rule's thickness, not a dark area's: a table, a shadow), shade (its darkest pixel
    in each column, the paler quarter's - a rule joined to a wire drawn along part of it stays a rule), y (its
    height at the middle)"""
    H, W = hor.shape
    lab, n = ndi.label(hor, structure=np.ones((3, 3)))
    out = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        if o is None:
            continue
        sel = lab[o] == k
        ys, xs = np.nonzero(sel)
        ys, xs = ys + o[0].start, xs + o[1].start
        fit = np.polyfit(xs, ys, 1) if np.ptp(xs) > 10 else np.array([0.0, float(ys.mean())])
        span = o[1].stop - o[1].start
        shade = None
        if page is not None:
            shade = float(np.percentile(np.where(sel, page[o], 255.0).min(0)[sel.any(0)], 75))
        out.append(dict(k=k, box=o, fit=fit, shade=shade, thin=len(xs) <= 16 * span, across=span >= 0.45 * W,
                        edge=bool(o[1].start <= 0.03 * W or o[1].stop >= 0.97 * W), y=float(np.polyval(fit, W / 2))))
    return lab, n, out


def _ruled(ys, height: float):
    """heights of lines -> their spacing if they are a ruled page's (five or more, evenly spaced - a rule hidden or
    broken off leaves at most a double gap - together covering half the page's height), else None"""
    ys = np.sort(np.asarray(ys, dtype=float))
    if len(ys):
        ys = ys[np.r_[True, np.diff(ys) > 8]]                   # one rule broken in pieces: once
    if len(ys) < 5 or ys[-1] - ys[0] < 0.5 * height:
        return None
    gaps = np.diff(ys)
    base = float(np.median(gaps))
    k = np.maximum(1, np.round(gaps / base))
    if np.all(np.abs(gaps / k - base) <= 0.25 * base) and np.mean(k == 1) >= 0.5:
        return base
    return None


def rules_only(hor: np.ndarray, page: np.ndarray | None = None) -> np.ndarray:
    """of a page's long thin lines (rule_lines), those that are notebook rules, not a circuit's long wires. Where
    the page shows it (page: evened out), by shade: rules are paler than the pen - the thin lines reaching the
    picture's side (three or more) give the rules' shade, the thin strokes off the lines the pen's, and a thin line
    nearer the rules' shade is a rule if it reaches the side or lies along one that does (broken off where the
    drawing crosses it); a pale line alone, its ends gone faint, stays - it may be a pale pen drawn along a rule.
    Otherwise by place: the lines reaching the side are rules when they are a ruled page's - evenly spaced, covering
    it (_ruled); so then are the lines across the page on their spacing, and the pieces along a rule. A bus of long
    wires reaching the side covers only part of the page."""
    H, W = hor.shape
    lab, n, lines = _lines(hor, page)
    keep = np.zeros(n + 1, bool)
    side = [L for L in lines if L["edge"] and L["thin"]]
    at_side = {L["k"] for L in side}

    def along_side(L):                                          # a piece of a rule reaching the side, broken off
        xm = (L["box"][1].start + L["box"][1].stop) / 2         # where the drawing crosses it
        y = np.polyval(L["fit"], xm)
        return L["k"] in at_side or any(abs(np.polyval(R["fit"], xm) - y) <= 6 for R in side)
    if page is not None and len(side) >= 3:
        ink = _thin_ink(page) & ~ndi.binary_dilation(hor, iterations=3)
        if ink.sum() >= 200:
            pen = float(np.median(page[ink]))
            rule = float(np.median([L["shade"] for L in side]))
            if rule - pen >= 25:
                for L in lines:
                    keep[L["k"]] = L["thin"] and L["shade"] >= (rule + pen) / 2 and along_side(L)
                return keep[lab] & hor
    base = _ruled([L["y"] for L in side], H)
    if base is None:
        return keep[lab] & hor
    lattice = np.array([L["y"] for L in side])
    for L in lines:
        if L["thin"]:
            keep[L["k"]] = along_side(L) or (L["across"] and np.min(np.abs(lattice - L["y"])) <= 0.2 * base)
    return keep[lab] & hor


def pen_image(img) -> np.ndarray:
    """the page as the pen shows on it (_pen_image)"""
    return _pen_image(img)[0]


def _pen_image(img):
    """the page with its light evened out (paper ~255) and, where the pen's colour differs from the notebook rules',
    the rules' colour taken out: each pixel's darkness in the three colour channels, a vector, is measured along the
    pen's colour less its part in the rules' colour - the rules all but vanish, and the pen stays where it crosses
    them. Kept no darker than the darkest channel there, so paper grain is not raised to ink; where the pen is
    rule-coloured yet darker than a rule (a shadow greys it), off the rules, the darkest channel. The rules: the thin
    lines reaching the picture's side (a table's dark is not thin); the pen: thin strokes off the long lines. Grey
    pen on grey rules (or a scan): the darkest channel at each pixel (a light teal pen is pale in grey, not in red),
    the rules then told apart by shade and place (rules_only). -> (page, whether the rules' colour was taken out)"""
    from .page import _even_light, ink_gray, rule_lines
    gray = _even_light(ink_gray(img))
    if img.mode == "L":
        return gray, False
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    chans = np.stack([_even_light(rgb[..., c]) for c in range(3)], -1)
    low = np.minimum(gray, chans.min(-1))
    hor, _ = rule_lines(low, 0.15)
    ver = rule_lines(np.ascontiguousarray(low.T), 0.15)[0].T            # squared paper's columns, a margin
    if not hor.any() and not ver.any():
        return low, False
    rule_mask = np.zeros_like(hor)
    for mask, t in ((hor, False), (ver, True)):            # the thin lines reaching the picture's side (or top)
        if mask.any():
            m = np.ascontiguousarray(mask.T) if t else mask
            lab, n, lines = _lines(m)
            ruled = np.zeros(n + 1, bool)
            for L in lines:
                ruled[L["k"]] = L["thin"] and L["edge"]
            rule_mask |= ruled[lab].T if t else ruled[lab]
    rule_px = rule_mask & (low < 205)
    pen_px = _thin_ink(low) & ~ndi.binary_dilation(hor | ver, iterations=3)
    if rule_px.sum() < 500 or pen_px.sum() < 200:
        return low, False
    dark = 255.0 - chans
    core = pen_px & (low <= np.percentile(low[pen_px], 50))
    rule_c, pen_c = np.median(dark[rule_px], 0), np.median(dark[core], 0)
    if np.linalg.norm(rule_c) < 1 or np.linalg.norm(pen_c) < 1:
        return low, False
    rule_c /= np.linalg.norm(rule_c)
    away = pen_c - (pen_c @ rule_c) * rule_c
    if np.linalg.norm(away) < 0.12 * np.linalg.norm(pen_c):
        return low, False
    p = dark @ (away / np.linalg.norm(away))
    scale = (255 - np.percentile(low[pen_px], 20)) / max(1.0, float(np.percentile(p[pen_px], 80)))
    out = np.maximum(np.clip(255 - scale * p, 0, 255), low)
    kept = np.mean(out[pen_px] < 200) / max(1e-9, np.mean(low[pen_px] < 200))
    paler = np.median(out[pen_px]) - np.median(low[pen_px])
    if kept < 0.5 or paler > 28 or (paler > 20 and kept < 0.8):
        return low, False             # the pen paled: its colour misjudged, or too near the rules' (blue on blue)
    shadow = _thin_ink(low) & (low < np.median(low[rule_px]) - 25) & ~ndi.binary_dilation(rule_mask, iterations=3)
    yy, xx = np.ogrid[-5:6, -5:6]                          # a pen's width at most: a shadow's own dark rim is wider
    shadow &= ~ndi.binary_dilation(ndi.binary_opening(shadow, structure=xx * xx + yy * yy <= 25), iterations=2)
    out[shadow] = low[shadow]
    return out, True


def strokes(img, dark: float = 200.0, faint: float = 228.0) -> np.ndarray:
    """pen strokes of a photo or scan: light evened out, the rules' colour taken out where it differs from the pen's
    (_pen_image), notebook rules removed (rules_only: a long straight wire is no rule), then darker than `dark`
    (paper ~255) - and, once the rules' colour is out, the fainter stretches (darker than `faint`) joined to such
    ink: a pale pen pressed lightly. The columns of squared paper go the same way as rules. Classical - faint pencil
    on ruled paper needs the learned stroke model."""
    from .page import _remove_rules
    norm, clean = _pen_image(img)
    norm, _ = _remove_rules(norm, which=rules_only, max_slope=0.15)
    norm = _remove_rules(np.ascontiguousarray(norm.T), which=rules_only, max_slope=0.15)[0].T   # and squared paper's
    m = norm < dark
    if clean and m.any():                    # faint but thin: a lightly pressed pen, not a shadow's soft edge
        weak = (norm < faint) & (norm < ndi.maximum_filter(norm, size=15) - 15) | m
        lab, n = ndi.label(weak, structure=np.ones((3, 3)))
        m = np.isin(lab, np.unique(lab[m]))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m &= np.isin(lab, np.where(np.asarray(sizes) >= 15)[0] + 1)          # specks
    return m


def stroke_width(mask: np.ndarray) -> float:
    """typical pen width in pixels (twice the median distance from a stroke's middle to its edge)"""
    if not mask.any():
        return 3.0
    d = ndi.distance_transform_edt(mask)
    skel_like = d[(d >= ndi.maximum_filter(d, size=3)) & mask]
    return float(max(1.5, 2 * np.median(skel_like))) if len(skel_like) else 3.0


class _Union:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _clip(b, h, w, grow=0):
    x0, y0, x1, y1 = b
    return (max(0, int(x0 - grow)), max(0, int(y0 - grow)), min(w, int(np.ceil(x1 + grow))), min(h, int(np.ceil(y1 + grow))))


_POLARITY = []
_ROLES = []


def _roles_model():
    """recog/roles.py's models when trained (models/roles), else None - loaded once"""
    if not _ROLES:
        import os
        try:
            from .roles import DEFAULT_DIR, RolesModel
            _ROLES.append(RolesModel() if os.path.isfile(os.path.join(DEFAULT_DIR, "config.json")) else None)
        except ImportError:
            _ROLES.append(None)
    return _ROLES[0]


def _polarity_model():
    """recog/polarity.py's models when trained (models/polarity), else None - loaded once"""
    if not _POLARITY:
        import os
        try:
            from .polarity import DEFAULT_DIR, PolarityModel
            _POLARITY.append(PolarityModel() if os.path.isfile(os.path.join(DEFAULT_DIR, "config.json")) else None)
        except ImportError:                                # no torch here: the shape rule alone
            _POLARITY.append(None)
    return _POLARITY[0]


def _end_points(b, lab, e, h, w):
    """where each end's wire meets the component's box: (x, y) per end, None if not found"""
    x0, y0, x1, y1 = _clip(b, h, w)
    pts = []
    for v in e:
        sub = lab[max(0, y0 - 6):min(h, y1 + 6), max(0, x0 - 6):min(w, x1 + 6)]
        ys, xs = np.where(sub == v)
        if not len(xs):
            return None
        # the wire pixels closest to the box: its end, not the far part of the wire
        px, py = xs + max(0, x0 - 6), ys + max(0, y0 - 6)
        dx = np.maximum(0, np.maximum(x0 - px, px - x1))
        dy = np.maximum(0, np.maximum(y0 - py, py - y1))
        near = (dx + dy) <= (dx + dy).min() + 2
        pts.append((float(px[near].mean()), float(py[near].mean())))
    return pts


_ACROSS = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _switch_ends(sides, weight, b, lab, h, w):
    """a two-way switch: three wires reach the box strongly - [common end, throw 1, throw 2] (the common end alone
    on its side, or on the side across from none of the others; the throws in reading order, upper / left first),
    else None (a switch of two ends)"""
    top = max(weight.values()) if weight else 0
    if top <= 0:
        return None
    side_of = {}
    for sd, d in sides.items():
        for v, sc in d.items():
            if sc >= 0.3 * top and (v not in side_of or sc > sides[side_of[v]][v]):
                side_of[v] = sd
    if len(side_of) != 3:
        return None
    by_side = {}
    for v, sd in side_of.items():
        by_side.setdefault(sd, []).append(v)
    if len(by_side) == 2:
        common = next((vs[0] for vs in by_side.values() if len(vs) == 1), None)
    elif len(by_side) == 3:
        common = next(vs[0] for sd, vs in by_side.items() if _ACROSS[sd] not in by_side)
    else:
        return None
    throws = [v for v in side_of if v != common]
    pts = _end_points(b, lab, throws, h, w)
    if common is None or pts is None:
        return None
    (ax, ay), (bx, by) = pts
    upper_first = abs(ay - by) >= 0.25 * max(1.0, b[3] - b[1])
    if (ay > by) if upper_first else (ax > bx):
        throws = throws[::-1]
    return [common] + throws


def _transformer_sides(pts, b):
    """a transformer's four wire ends -> {'P+': k, 'P-': k, 'S+': k, 'S-': k}: the coils side by side (the primary the
    left one) or one above the other (the primary on top), a coil's dot taken at its top / left end (_coil_dots reads
    the dots when drawn); None when the ends do not fall into two clear pairs"""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    for coord, other, size in ((xs, ys, b[2] - b[0]), (ys, xs, b[3] - b[1])):
        order = sorted(range(4), key=lambda k: coord[k])
        if coord[order[2]] - coord[order[1]] >= 0.3 * max(1.0, size):
            first, second = order[:2], order[2:]
            p_plus, p_minus = sorted(first, key=lambda k: other[k])
            s_plus, s_minus = sorted(second, key=lambda k: other[k])
            return {"P+": p_plus, "P-": p_minus, "S+": s_plus, "S-": s_minus}
    return None



def _coil_dots(mask, b, pts, sides, sw, skip=()):
    """the dots of coupled coils: a small filled blob, alone (touching no stroke), clearly by one end of a coil, in or
    just around the transformer's box -> the sides with each coil's '+' at its dotted end, or None (not exactly one
    such dot by each coil). skip: boxes whose ink is not a dot (labels: a pin's '1')"""
    h, w = mask.shape
    x0, y0, x1, y1 = b
    g = 0.2 * max(x1 - x0, y1 - y0)
    X0, Y0, X1, Y1 = int(max(0, x0 - g)), int(max(0, y0 - g)), int(min(w, x1 + g)), int(min(h, y1 + g))
    win = mask[Y0:Y1, X0:X1]
    if not win.any():
        return None
    lab, _ = ndi.label(win, structure=np.ones((3, 3)))
    dt = ndi.distance_transform_edt(win)
    coil = {"P": (sides["P+"], sides["P-"]), "S": (sides["S+"], sides["S-"])}
    found = {"P": [], "S": []}
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not 1.5 * sw <= max(hh, ww) <= 5 * sw or max(hh, ww) > 2.2 * min(hh, ww):
            continue
        sub = lab[sl] == k
        if sub.sum() < 0.5 * hh * ww or dt[sl][sub].max() < 0.8 * sw:
            continue                                        # not filled: a ring, a letter's stroke
        ys, xs = np.nonzero(sub)
        cx, cy = float(xs.mean()) + sl[1].start + X0, float(ys.mean()) + sl[0].start + Y0
        if any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in skip):
            continue
        d = [float(np.hypot(cx - px, cy - py)) for px, py in pts]
        j = int(np.argmin(d))
        which = "P" if j in coil["P"] else "S"
        a, b_ = coil[which]
        other = b_ if j == a else a
        span = float(np.hypot(pts[a][0] - pts[b_][0], pts[a][1] - pts[b_][1]))
        if d[j] <= 0.75 * d[other] and d[j] <= 0.6 * max(span, 1.0):
            found[which].append(j)
    if len(found["P"]) != 1 or len(found["S"]) != 1:
        return None
    p, s_ = found["P"][0], found["S"][0]
    return {"P+": p, "P-": coil["P"][1] if p == coil["P"][0] else coil["P"][0],
            "S+": s_, "S-": coil["S"][1] if s_ == coil["S"][0] else coil["S"][0]}

def _switch_state(mask, b, pts):
    """the state a switch is drawn in, from its strokes: 'closed' when ink runs unbroken inside its box from one
    end to the other, else 'open'; a two-way switch (pts: common end, throw 1, throw 2): '1' / '2', the throw the
    common end's ink reaches. None when it cannot tell"""
    h, w = mask.shape
    x0, y0, x1, y1 = _clip(b, h, w, grow=3)
    lab, n = ndi.label(mask[y0:y1, x0:x1], structure=np.ones((3, 3)))
    if n == 0:
        return None

    def piece_at(p, r=4):
        px, py = int(round(p[0])) - x0, int(round(p[1])) - y0
        win = lab[max(0, py - r):max(0, py + r + 1), max(0, px - r):max(0, px + r + 1)]
        vals = win[win > 0]
        return int(np.bincount(vals).argmax()) if len(vals) else None
    ids = [piece_at(p) for p in pts]
    if None in ids:
        return None
    if len(ids) == 2:
        return "closed" if ids[0] == ids[1] else "open"
    hits = [str(k) for k in (1, 2) if ids[k] == ids[0]]
    return hits[0] if len(hits) == 1 else None


def _first_end(mask, b, lab, e, cls, polarity=None):
    """which end (0/1) has the class's first role ('+', arrow head, cathode): the polarity CNN when trained,
    else (DC sources only) the shape rule of _plus_side; None when unsure"""
    pm = polarity if polarity is not None else _polarity_model()
    cls = POLAR.get(cls, cls)                          # a diamond's '+ -' / arrow reads like a round source's
    if pm is not None and cls in pm.nets:
        pts = _end_points(b, lab, e, *mask.shape)
        if pts is None:
            return None
        p = pm.first_end(mask, pts[0], pts[1], cls)
        if p is None or abs(p - 0.5) < 0.15:
            return None
        return 0 if p > 0.5 else 1
    return _plus_side(mask, b, lab, e) if cls == "voltage_dc" else None


def _plus_side(mask, b, lab, e):
    """which end (0/1) of a DC source is '+': the half of the symbol, along the line between its two ends, whose
    inner ink reaches further both along and across that line - a '+' has a stroke along it that a '-' lacks;
    a battery's long plate reaches further across than its short one. None when it cannot tell."""
    h, w = mask.shape
    x0, y0, x1, y1 = _clip(b, h, w)
    pts = []
    for v in e:                                       # where each end's wire meets the box
        ys, xs = np.where(lab[max(0, y0 - 6):min(h, y1 + 6), max(0, x0 - 6):min(w, x1 + 6)] == v)
        if not len(xs):
            return None
        pts.append((xs.mean() + max(0, x0 - 6), ys.mean() + max(0, y0 - 6)))
    (ax, ay), (bx, by) = pts
    vx, vy = bx - ax, by - ay
    norm = (vx * vx + vy * vy) ** 0.5
    if norm < 1:
        return None
    vx, vy = vx / norm, vy / norm
    cw, ch = x1 - x0, y1 - y0
    ix0, iy0, ix1, iy1 = int(x0 + 0.22 * cw), int(y0 + 0.22 * ch), int(x1 - 0.22 * cw), int(y1 - 0.22 * ch)
    ys, xs = np.where(mask[iy0:iy1, ix0:ix1])
    if len(xs) < 10:
        return None
    xs, ys = xs + ix0, ys + iy0
    mx, my = (ax + bx) / 2, (ay + by) / 2
    along = (xs - mx) * vx + (ys - my) * vy          # < 0: the half of end 0
    across = -(xs - mx) * vy + (ys - my) * vx
    score = []
    for half in (along < 0, along >= 0):
        if half.sum() < 5:
            score.append(0.0)
            continue
        a_, c_ = along[half], across[half]
        score.append((np.percentile(a_, 95) - np.percentile(a_, 5)) + (np.percentile(c_, 95) - np.percentile(c_, 5)))
    if abs(score[0] - score[1]) < 0.08 * max(score):
        return None
    return 0 if score[0] > score[1] else 1


def _loose_ends(wires, lab, boxes, sw, keep):
    """wire ends that stop short of everything (a pen lifted for long, a stretch lost on a notebook rule):
    [(end (x, y), unit direction the wire heads to there, wire label)] - only ends of the wire labels in keep,
    not at a part's box (boxes: grown boxes)"""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return []
    sk = skeletonize(wires)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk.astype(np.uint8)
    h, w = sk.shape
    near = np.zeros_like(sk)
    for x0, y0, x1, y1 in boxes:
        near[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
    out = []
    for y, x in np.argwhere(sk & (nb == 1)):
        v = int(lab[y, x])
        if near[y, x] or v not in keep:
            continue
        path, prev, cur = [(y, x)], None, (y, x)
        while len(path) < 6 * sw:                          # walk back along the stroke
            cy, cx = cur
            nxt = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
                   and 0 <= cy + dy < h and 0 <= cx + dx < w and sk[cy + dy, cx + dx] and (cy + dy, cx + dx) != prev
                   and (cy + dy, cx + dx) not in path[-3:]]
            if len(nxt) != 1:
                break                                      # a junction or the other end
            prev, cur = cur, nxt[0]
            path.append(cur)
        if len(path) < 3 * sw:
            continue                                       # a stub, not a wire
        dy, dx = path[0][0] - path[-1][0], path[0][1] - path[-1][1]
        n = float(np.hypot(dx, dy))
        if n >= 1:
            out.append(((float(x), float(y)), (dx / n, dy / n), v))
    return out


_K8 = np.ones((3, 3), np.uint8)


def _crossings(sk: np.ndarray) -> np.ndarray:
    """each skeleton pixel's crossing number - the 0 -> 1 steps going round its 8 neighbours: 1 an end, 2 a line, 3+
    a junction (a staircase pixel with three neighbours is still a line)"""
    p = np.pad(sk.astype(np.int8), 1)
    h, w = sk.shape
    ring = [p[0:h, 1:w + 1], p[0:h, 2:w + 2], p[1:h + 1, 2:w + 2], p[2:h + 2, 2:w + 2],
            p[2:h + 2, 1:w + 1], p[2:h + 2, 0:w], p[1:h + 1, 0:w], p[0:h, 0:w]]       # N NE E SE S SW W NW
    cn = np.zeros(sk.shape, np.int8)
    for i in range(8):
        cn += ((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.int8)
    return np.where(sk, cn, 0)


def _prune(sk: np.ndarray, k: int) -> np.ndarray:
    """the skeleton without its spurs shorter than k pixels (a pen's hook, a thick stroke's corner)"""
    sk = sk.copy()
    for _ in range(k):
        ends = sk & (_crossings(sk) <= 1)
        if not ends.any():
            break
        sk &= ~ends
    return sk


def _skeleton_paths(sk: np.ndarray) -> list:
    """the skeleton cut at its junctions and ends: [[(y, x), ...]] each ordered along its stroke"""
    node = sk & (_crossings(sk) != 2)
    seen = np.zeros_like(sk)
    h, w = sk.shape
    out = []

    def nbrs(y, x):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]:
                    yield y + dy, x + dx
    for s in map(tuple, np.argwhere(node)):
        for n in nbrs(*s):
            if seen[n] or node[n]:
                continue
            path, prev, cur, inpath = [s, n], s, n, {s, n}
            seen[n] = True
            while True:
                nxt = [q for q in nbrs(*cur) if q not in inpath]
                if not nxt:
                    break
                q = next((q for q in nxt if node[q]), None)
                if q is not None and len(path) > 2:
                    path.append(q)
                    break
                nxt = [q for q in nxt if not node[q]] or nxt
                q = max(nxt, key=lambda q: (q[0] - prev[0]) ** 2 + (q[1] - prev[1]) ** 2)
                if seen[q]:
                    break
                seen[q] = True
                for r_ in nxt:                                    # a staircase's corner pixel: passed, not a branch
                    if r_ != q and not node[r_]:
                        seen[r_] = True
                path.append(q)
                inpath.add(q)
                prev, cur = cur, q
            out.append(path)
    return out


def _winding(path, sw: float, window: float = 10.0):
    """the most a stroke turns (degrees, every turn added whatever its sense) along any `window` pen widths of it,
    and where: a wire turns a corner or two, a zigzag's tips and a coil's humps turn over and over"""
    p = np.asarray(path, dtype=float)
    if len(p) < 6:
        return 0.0, (0.0, 0.0)
    sig = max(1.0, 0.5 * sw)
    ys, xs = ndi.gaussian_filter1d(p[:, 0], sig, mode="nearest"), ndi.gaussian_filter1d(p[:, 1], sig, mode="nearest")
    step = max(1, int(round(0.5 * sw)))
    th = np.unwrap(np.arctan2(ys[step:] - ys[:-step], xs[step:] - xs[:-step]))
    dth = np.abs(np.diff(th))
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ys), np.diff(xs))[:len(dth)])])
    cum = np.concatenate([[0.0], np.cumsum(dth)])
    best, j, at = 0.0, 0, 0
    for i in range(len(dth) + 1):
        while arc[i] - arc[j] > window * sw:
            j += 1
        if cum[i] - cum[j] > best:
            best, at = cum[i] - cum[j], (i + j) // 2
    at = min(at, len(xs) - 1)
    return float(np.degrees(best)), (float(xs[at]), float(ys[at]))


def _sharp_turns(path, arm: int, max_open: float = 85.0) -> int:
    """how many times a stroke turns back on itself: points where the angle between `arm` pixels back and `arm`
    ahead opens less than max_open degrees (a corner opens 90), at least arm apart"""
    p = np.asarray(path, dtype=float)
    n = len(p)
    if n < 2 * arm + 1:
        return 0
    a, b = p[:n - 2 * arm] - p[arm:n - arm], p[2 * arm:] - p[arm:n - arm]
    na, nb = np.hypot(a[:, 0], a[:, 1]), np.hypot(b[:, 0], b[:, 1])
    ok = (na >= 0.5 * arm) & (nb >= 0.5 * arm)                 # an arm crumpled up: no clean turn there
    cos = np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-9)
    ang = np.where(ok, np.degrees(np.arccos(np.clip(cos, -1, 1))), 180.0)
    picked = []
    for i in np.argsort(ang):
        if ang[i] >= max_open:
            break
        if all(abs(i - j) >= arm for j in picked):
            picked.append(int(i))
    return len(picked)


def _legs(sub: np.ndarray, obj: np.ndarray, sw: float) -> list:
    """the strokes that leave an object (a hole with its ring): how far each runs from it, in pen widths - pieces of
    ink starting within 4 pen widths of it"""
    d = ndi.distance_transform_edt(~obj)
    outside = sub & (d > 2.0 * sw)
    lab_o, n = ndi.label(outside, structure=np.ones((3, 3)))
    if not n:
        return []
    far = np.asarray(ndi.maximum(d, lab_o, range(1, n + 1)))
    starts = set(np.unique(lab_o[outside & (d <= 4.0 * sw)]).tolist()) - {0}
    return sorted(float(far[i - 1] / sw) for i in starts)


def _part_outline(legs: list) -> bool:
    """a loop with the wires of a part's outline: its two leads running on (12 pen widths and more; a ring's tail
    where the pen overlapped does not), a diode's bar or a lead ending at the next part beside them - not three
    or four wires running far (a junction drawn as a ring, a cell of squared paper)"""
    return sum(1 for f in legs if f >= 12) <= 3 and sum(1 for f in legs if f >= 25) <= 2


def _symbol_ink(wires: np.ndarray, lab: np.ndarray, keep, boxes, sw: float, size: float = 0.0, hidden=None,
                parts=()) -> list:
    """ink in the circuit's wires that looks like a part's symbol - a part the detector missed, whose strokes the
    wiring took for wire (its two nodes joined, or the part gone): [(what, (x, y))]. size: the parts' typical box
    (the median long side), or 0. Looked for: a closed loop of paper the size of a part's, wide inside, with two
    wires running far from it (a source's circle, a diode's triangle - a junction dot is smaller, a junction drawn as
    a ring or a cell of squared paper has more, a wire drawn twice leaves a sliver); a stroke
    that turns over and over (a zigzag, a coil's humps - not a small closed ring); two parallel strokes side by side
    ending facing nothing (a capacitor's plates; a pen lifted at a corner faces the other wire). On CGHD drafter 1
    (48 photos, the true boxes) quiet on all 48; with one part's box left out, most of those 192 parts are found."""
    out = []
    if not wires.any():
        return out
    h, w = wires.shape
    ys, xs = np.nonzero(wires)
    pad = int(6 * sw) + 2
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad), max(0, xs.min() - pad), min(w, xs.max() + pad)
    sub = wires[y0:y1, x0:x1]
    if wires[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.12:
        return [("dense", (0.0, 0.0))]                         # a circuit's wires are sparse lines (at most 6.5 % of
    ring = sub                                                 # its box on drafter 1; 10 - 44 % in a table / grid)
    if hidden is not None and hidden.any():                    # a label box over a source's circle hides an arc of
        hid = hidden[y0:y1, x0:x1]                             # it: the ink there that joins the wires is put back
        ring = ndi.binary_propagation(sub, structure=np.ones((3, 3)), mask=sub | hid)
    holes, nh = ndi.label(ndi.binary_fill_holes(ring) & ~ring)
    if nh:                                                     # holes close together (a source's signs split its
        areas = np.bincount(holes.ravel())[1:]                 # circle) are one object
        big = np.isin(holes, 1 + np.flatnonzero(areas >= 1.0 * sw ** 2))
        grown = ndi.binary_dilation(big, iterations=max(1, int(round(1.5 * sw)))) & (big | ring)
        objs, no = ndi.label(grown)
        for k, sl in enumerate(ndi.find_objects(objs)):
            obj = objs == k + 1
            inside = (holes > 0) & obj
            if inside.sum() < max(4 * sw ** 2, 0.01 * size ** 2):
                continue                                       # a junction dot's pores, a thick stroke's
            if ndi.distance_transform_edt(inside[sl]).max() < 0.8 * sw:
                continue                                       # a sliver between a wire and its retracing
            if _part_outline(_legs(ring, obj, sw)):
                out.append(("a closed loop", (x0 + (sl[1].start + sl[1].stop) / 2, y0 + (sl[0].start + sl[0].stop) / 2)))
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return out
    sk = _prune(skeletonize(sub), int(round(2 * sw)))
    arm = max(4, int(round(2.5 * sw)))
    for pth in _skeleton_paths(sk):
        if len(out) >= 25:
            break                                              # strokes this messy: enough said
        if len(pth) <= 10 or np.hypot(pth[0][0] - pth[-1][0], pth[0][1] - pth[-1][1]) < 2 * sw:
            continue                                           # a small closed ring (a junction drawn as a circle)
        deg, (wx, wy) = _winding(pth, sw)                      # a zigzag / coil drawn small for its pen
        if deg >= 220 or _sharp_turns(pth, arm) >= 3:          # a zigzag of long strokes for its pen (a fine pen)
            if deg < 220:
                wy, wx = pth[len(pth) // 2]
            px, py = x0 + wx, y0 + wy
            if any(np.hypot(max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3])) < 8 * sw for b in parts):
                continue                                       # a part's own zigzag sticking out of its box (a box
            out.append(("a stroke that turns over and over", (px, py)))     # set a little off: 4 - 7 pen widths)
    dist = ndi.distance_transform_edt(sub)
    pieces, _ = ndi.label(sub, structure=np.ones((3, 3)))
    r6 = int(max(6 * sw, 0.4 * size))                        # how far apart a capacitor's plates may be
    yy, xx = np.mgrid[-r6:r6 + 1, -r6:r6 + 1]
    rr = np.hypot(yy, xx)
    for (x, y), d, v in _loose_ends(wires, lab, boxes, sw, keep):
        if len(out) >= 25:
            break
        lx, ly = int(x) - x0, int(y) - y0
        by = min(max(int(round(ly - d[1] * 2 * sw)), 0), sub.shape[0] - 1)
        bx = min(max(int(round(lx - d[0] * 2 * sw)), 0), sub.shape[1] - 1)
        if 2 * dist[by, bx] < 0.5 * sw:
            continue                                           # a hairline (what is left of a notebook rule)
        own = pieces[ly, lx]
        cy, cx = ly + yy, lx + xx
        ok = (cy >= 0) & (cy < sub.shape[0]) & (cx >= 0) & (cx < sub.shape[1])
        cyc, cxc = np.clip(cy, 0, sub.shape[0] - 1), np.clip(cx, 0, sub.shape[1] - 1)
        cone = ok & (rr <= 5 * sw) & ((xx * d[0] + yy * d[1]) > 0.5 * np.maximum(rr, 1e-6))
        hit = pieces[cyc[cone], cxc[cone]]
        if ((hit > 0) & (hit != own)).any():
            continue                                           # faces another stroke: a pen lifted short of it
        across = parallel = False
        near = ok & (rr <= r6)
        for q in np.unique(pieces[cyc[near], cxc[near]]):
            if q == 0 or q == own:
                continue
            m = near & (pieces[cyc, cxc] == q)
            qy, qx = cy[m], cx[m]
            dq = np.hypot(qy - ly, qx - lx)
            j = int(np.argmin(dq))
            close = np.hypot(qy - qy[j], qx - qx[j]) <= 2 * sw
            if close.sum() < 3:
                continue
            ev, evec = np.linalg.eigh(np.cov(np.vstack([qx[close], qy[close]]).astype(float)))
            c = abs(evec[0, -1] * d[0] + evec[1, -1] * d[1])
            off = abs((qx[j] - lx) * d[0] + (qy[j] - ly) * d[1]) / max(float(dq[j]), 1e-6)
            if c < 0.7 and dq[j] <= 3 * sw:
                across = True                                  # another stroke runs across its way, right by it:
            elif c >= 0.7 and off < 0.5:                       # a corner drawn in two strokes
                parallel = True                                # a stroke beside it running the same way: a plate
        if parallel and not across:
            out.append(("two parallel strokes ending facing nothing", (float(x), float(y))))
    kept = []                                                  # one finding a spot (a capacitor's four plate ends)
    for what, (x, y) in out:
        if all(np.hypot(x - kx, y - ky) > 8 * sw for _, (kx, ky) in kept):
            kept.append((what, (x, y)))
    return kept


def _branches(sk: np.ndarray, cn: np.ndarray, start: set, limit: int):
    """from a junction's pixels, each branch walked up to `limit` pixels: [(pixels, how it ends: 'end' / 'node' /
    'far')]"""
    h, w = sk.shape
    out = []
    seen = set(start)
    firsts = []
    for (y, x) in start:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (dy or dx) and 0 <= q[0] < h and 0 <= q[1] < w and sk[q] and q not in seen:
                    seen.add(q)
                    firsts.append(q)
    for q in firsts:
        path, cur, stop = [q], q, "far"
        while len(path) < limit:
            if cn[cur] == 1:
                stop = "end"
                break
            if cn[cur] >= 3:
                stop = "node"
                break
            nxt = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy or dx) and 0 <= cur[0] + dy < h and 0 <= cur[1] + dx < w and sk[cur[0] + dy, cur[1] + dx]
                   and (cur[0] + dy, cur[1] + dx) not in seen]
            if not nxt:
                stop = "end"
                break
            cur = max(nxt, key=lambda r: abs(r[0] - path[-1][0]) + abs(r[1] - path[-1][1]) if len(path) > 1 else 0)
            for r in nxt:
                seen.add(r)
            path.append(cur)
        out.append((path, stop))
    return out


def _wire_arrows(wires: np.ndarray, sw: float) -> list:
    """arrowheads drawn on the wires themselves (a current's arrow on its wire): [((tip x, y), (ux, uy) the way it
    points)] - a chevron: two short strokes leaving a point of a wire backwards, one on either side of it"""
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return []
    if not wires.any():
        return []
    ys, xs = np.nonzero(wires)
    pad = 4
    y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    sub = wires[y0:ys.max() + pad + 1, x0:xs.max() + pad + 1]
    sk = _prune(skeletonize(sub), max(2, int(round(0.6 * sw))))   # without the hairs a stroke's round end leaves
    cn = _crossings(sk)
    near = ndi.binary_dilation(sk & (cn >= 3), iterations=max(1, int(round(0.75 * sw))))   # a head's two strokes
    jl, nj = ndi.label(near, structure=np.ones((3, 3)))                                    # may leave a wire apart
    out = []
    for k, sl in enumerate(ndi.find_objects(jl)):
        pix = {(int(a) + sl[0].start, int(b) + sl[1].start) for a, b in np.argwhere((jl[sl] == k + 1) & sk[sl])}
        if not pix:
            continue
        cy, cx = np.mean([p[0] for p in pix]), np.mean([p[1] for p in pix])
        brs = _branches(sk, cn, pix, int(8 * sw))
        spurs, longs = [], []
        for path, stop in brs:
            far = np.hypot(path[-1][0] - cy, path[-1][1] - cx)
            vec = np.array([path[-1][1] - cx, path[-1][0] - cy], float)
            if stop == "end" and 1.0 * sw <= far <= 5.0 * sw:
                spurs.append(vec / max(np.hypot(*vec), 1e-6))
            elif far >= 4.0 * sw:
                longs.append(vec / max(np.hypot(*vec), 1e-6))
        if len(spurs) != 2 or len(longs) != 2 or longs[0] @ longs[1] > -0.8:
            continue                                           # a chevron on a wire that runs on through it
        m = spurs[0] + spurs[1]
        if np.hypot(*m) < 0.3:
            continue
        u = -m / np.hypot(*m)                                  # the arrow points away from its two strokes
        axis = longs[0] if abs(longs[0] @ u) >= abs(longs[1] @ u) else longs[1]
        if abs(axis @ u) < 0.85:
            continue                                           # the head not along its wire
        side = [u[0] * sp_[1] - u[1] * sp_[0] for sp_ in spurs]
        ang = [np.degrees(np.arccos(np.clip(-(sp_ @ u), -1, 1))) for sp_ in spurs]
        if side[0] * side[1] >= 0 or not all(10 <= a <= 80 for a in ang):
            continue                                           # one stroke each side, slanting back
        out.append(((float(cx + x0), float(cy + y0)), (float(u[0]), float(u[1]))))
    return out


def _walk_wire(skp: np.ndarray, start, u, stop_at, limit: int):
    """from a point of a wire's (pruned) skeleton, along it the way u points: the first index stop_at gives (> 0: a
    part's box reached), or None at a junction / the stroke's end"""
    h, w = skp.shape
    cn = _crossings(skp)
    y, x = start
    d = np.array(u, float)
    seen = {(y, x)}
    for step in range(limit):
        if stop_at[y, x]:
            return int(stop_at[y, x])
        nxt = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and skp[y + dy, x + dx] and (y + dy, x + dx) not in seen]
        if step > 2 and (cn[y, x] >= 3 or any(cn[q] >= 3 for q in nxt)):
            return None                                        # the wire branches: not one part's current
        if not nxt:
            return None
        q = max(nxt, key=lambda r: (r[1] - x) * d[0] + (r[0] - y) * d[1])
        if (q[1] - x) * d[0] + (q[0] - y) * d[1] < -0.5 and step < 3:
            return None
        step_v = np.array([q[1] - x, q[0] - y], float)
        d = 0.7 * d + 0.3 * step_v / max(np.hypot(*step_v), 1e-6)
        d /= max(np.hypot(*d), 1e-6)
        seen.update(nxt)
        y, x = q
    return None


def _gate_output(mask, box, pts):
    """which end of a logic gate is its output: the one alone on its side, opposite the side its inputs come in at;
    of two ends on opposite sides (NOT), the one at the narrow end (the triangle's tip, the bubble) - None if unsure"""
    x0, y0, x1, y1 = box
    def side(p):
        d = {"left": abs(p[0] - x0), "right": abs(p[0] - x1), "top": abs(p[1] - y0), "bottom": abs(p[1] - y1)}
        return min(d, key=d.get)
    sides = [side(p) for p in pts]
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    count = {sd: sides.count(sd) for sd in set(sides)}
    if len(pts) >= 3:
        many = max(count, key=count.get)
        single = [k for k, sd in enumerate(sides) if sd == opposite[many] and count[sd] == 1]
        return single[0] if len(single) == 1 and count[many] >= 2 else None
    if len(pts) == 2 and sides[0] == opposite.get(sides[1]):
        h, w = mask.shape
        X0, Y0, X1, Y1 = (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))
        sub = mask[Y0:Y1, X0:X1]
        if sub.size == 0:
            return None
        horizontal = sides[0] in ("left", "right")
        n = sub.shape[1] if horizontal else sub.shape[0]
        k = max(2, n // 4)
        def spread(sl):                                    # how far across the gate the ink reaches in this slab
            part = sub[:, sl] if horizontal else sub[sl, :]
            idx = np.nonzero(part.any(axis=1 if horizontal else 0))[0]
            return idx.max() - idx.min() if len(idx) else 0
        first, last = spread(slice(0, k)), spread(slice(n - k, n))       # left / top slab, right / bottom slab
        if abs(first - last) < 0.15 * max(first, last, 1):
            return None
        narrow = "left" if horizontal and first < last else "right" if horizontal else "top" if first < last else "bottom"
        return 0 if sides[0] == narrow else 1
    return None


def _symbol_near(mask, box, gap):
    """pixels within gap + 1 of the symbol's own strokes in box (its big stroke pieces: a resistor's zigzag with
    its leads, both plates of a capacitor - not a small mark inside), as a whole-image mask; None if no ink"""
    x0, y0, x1, y1 = box
    inner = mask[y0:y1, x0:x1]
    il, ni = ndi.label(inner, structure=np.ones((3, 3)))
    if not ni:
        return None
    sz = np.bincount(il.ravel())[1:]
    main = np.zeros_like(mask)
    main[y0:y1, x0:x1] = np.isin(il, np.where(sz >= 0.3 * sz.max())[0] + 1)
    r = int(np.ceil(gap)) + 1
    m0, m1 = max(0, y0 - r), max(0, x0 - r)
    sub = ndi.binary_dilation(main[m0:y1 + r, m1:x1 + r], structure=np.ones((2 * r + 1, 2 * r + 1)))
    out = np.zeros_like(mask)
    out[m0:y1 + r, m1:x1 + r] = sub
    return out


def _arrow_on_axis(pts, u, sw, strict=True):
    """the piece's pixels pts seen along the unit direction u: a line with a head - ink spreading to both sides of
    the line - at one end -> (tip, tail) or None. strict: the piece must be a line of its own (its own main
    direction); else a stubbier piece (a big head on a short line) along a part's direction"""
    c = pts.mean(0)
    n = np.array([-u[1], u[0]])
    along = (pts - c) @ u
    across = (pts - c) @ n
    lo, hi = np.percentile(along, 1), np.percentile(along, 99)
    length = hi - lo
    if length < (5 if strict else 4) * sw or np.percentile(np.abs(across), 80) > (0.25 if strict else 0.45) * length:
        return None                                    # too short, or not a line
    third = length / 3

    def head(sel):
        """how far the ink of this end spreads to the less spread side of the line"""
        a = across[sel]
        if (a > 0).sum() < 3 or (a < 0).sum() < 3:
            return 0.0
        return float(min(np.percentile(a[a > 0], 90), np.percentile(-a[a < 0], 90)))
    h_lo, h_hi = head(along < lo + third), head(along > hi - third)
    big, small = max(h_lo, h_hi), min(h_lo, h_hi)
    if big < 1.2 * sw or big < 1.8 * max(small, 0.5 * sw):
        return None                                    # no head, or both ends alike
    tip_t, tail_t = (hi, lo) if h_hi > h_lo else (lo, hi)
    return tuple(c + tip_t * u), tuple(c + tail_t * u)


def _arrows(lab, loose, sw, axes=()):
    """current arrows among the stroke pieces `loose` (wire labels no component end reaches): long, thin, with a
    head at one end -> [(tip (x, y), tail (x, y))]. axes: [(unit direction, centre, reach)] of the two-terminal
    parts - a piece too stubby to be a line of its own is tried along a part it lies beside"""
    out = []
    objs = ndi.find_objects(lab)
    for v in loose:
        sl = objs[v - 1] if v - 1 < len(objs) else None
        if sl is None:
            continue
        ys, xs = np.nonzero(lab[sl] == v)
        if len(xs) < 6 * sw:
            continue
        pts = np.stack([xs + sl[1].start, ys + sl[0].start], 1).astype(float)
        c = pts.mean(0)
        _, vec = np.linalg.eigh(np.cov((pts - c).T))
        a = _arrow_on_axis(pts, vec[:, 1], sw)
        if a is None:
            for u, ctr, reach in sorted(axes, key=lambda x: np.hypot(*(c - np.asarray(x[1])))):
                if np.hypot(*(c - np.asarray(ctr))) <= reach:
                    a = _arrow_on_axis(pts, np.asarray(u, float), sw, strict=False)
                    if a is not None:
                        break
        if a is not None:
            out.append(a)
    return out


_SIGN = {"+": "+", "-": "-", "\u2212": "-", "\u2013": "-", "\u2014": "-"}
# a voltage's name as drawn beside its '+' / '-' marks: 'V_x', 'v_o', 'VR1', 'V_{out}', 'U_2', 'v(t)'
_VOLT = re.compile(r"^(?P<name>[VvUu](?:_?\{[0-9A-Za-z_]{1,6}\}|_?[0-9A-Za-z]{0,4}))(?:\(t\))?$")
_SIGNED = re.compile(r"^(?P<a>[+\-\u2212\u2013])\s*(?P<name>[VvUu](?:_?\{[0-9A-Za-z_]{1,6}\}|_?[0-9A-Za-z]{0,4}))\s*(?P<b>[+\-\u2212\u2013])$")


def _own_mark(c, placed) -> bool:
    """a '+' / '-' (or a name) at c that belongs to a part's own symbol: inside its box (a source's '+'), or by an
    op-amp's, a transistor's or an IC's box (its inputs' '+' / '-' drawn just outside the triangle)"""
    for comp, pb, _, cls in placed:
        if pb[0] <= c[0] <= pb[2] and pb[1] <= c[1] <= pb[3]:
            return True
        if cls in ("opamp", "bjt", "fet", "ic"):
            g = 0.3 * max(pb[2] - pb[0], pb[3] - pb[1])
            if pb[0] - g <= c[0] <= pb[2] + g and pb[1] - g <= c[1] <= pb[3] + g:
                return True
    return False


def _voltage_marks(texts, placed, terminals, lab, node, h, w):
    """voltages marked on the drawing: a '+' and a '-' beside a two-terminal part's two ends - or by two open
    terminals, or by the ends of two different parts (across a chain of them) - with a voltage's name near them
    ('V_x', 'v_o', 'V_{R1}'; '+ V_x -' read as one label too) -> [(name, '+' node, '-' node, the part across or None,
    the name's label, {the marks' labels})]. Marks inside a part's box are its own (a source's '+', an op-amp's
    inputs) and are left alone."""
    inside = lambda c: _own_mark(c, placed)                 # noqa: E731
    signs, names = [], []
    for k, (tb, t, _) in enumerate(texts):
        t = (t or "").strip()
        c = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        if not t or inside(c):
            continue
        if t in _SIGN:
            signs.append((k, _SIGN[t], c))
            continue
        m = _SIGNED.match(t)
        if m and _SIGN[m.group("a")] != _SIGN[m.group("b")]:      # '+ V_x -' in one label: the marks at its two ends
            names.append((k, m.group("name"), c, ((_SIGN[m.group("a")], (tb[0], c[1])), (_SIGN[m.group("b")], (tb[2], c[1])))))
            continue
        m = _VOLT.match(t)
        if m:
            names.append((k, m.group("name"), c, None))
    if not names or not signs and all(nm[3] is None for nm in names):
        return []
    parts, sizes = [], []                       # (id, ((x, y), node) per end, its box, its size)
    for comp, b, e, cls in placed:
        if cls in TWO_ENDS and len(e) == 2 and None not in e:
            pts = _end_points(b, lab, e, h, w)
            if pts is not None:
                size = max(b[2] - b[0], b[3] - b[1])
                parts.append((comp.id, [(pts[0], node(e[0])), (pts[1], node(e[1]))], b, size))
                sizes.append(size)
    if not parts:
        return []
    med = float(np.median(sizes))
    ends = [(pt, nd, pid, size) for pid, two, _, size in parts for pt, nd in two] + \
        [((x, y), nd, None, med) for nd, (x, y) in terminals]
    dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))      # noqa: E731

    def box_gap(c, b):
        return float(np.hypot(max(b[0] - c[0], 0, c[0] - b[2]), max(b[1] - c[1], 0, c[1] - b[3])))
    options = []                                # (cost, name index, plus node, minus node, part id, sign labels used)
    for ni, (k, name, c, own) in enumerate(names):
        if own is not None:
            pairs = [(own[0] if own[0][0] == "+" else own[1], own[1] if own[0][0] == "+" else own[0], set())]
        else:
            near = [sg for sg in signs if dist(sg[2], c) <= 1.5 * med]
            pairs = [(("+", p[2]), ("-", m[2]), {p[0], m[0]}) for p in near if p[1] == "+" for m in near if m[1] == "-"]
        for (_, pc), (_, mc), used in pairs:
            for pid, two, b, size in parts:     # both marks by one part's two ends: the voltage across it
                if box_gap(c, b) > 1.2 * size:
                    continue
                for i, j in ((0, 1), (1, 0)):
                    dp, dm = dist(pc, two[i][0]), dist(mc, two[j][0])
                    if dp < dist(pc, two[j][0]) and dm < dist(mc, two[i][0]) and max(dp, dm) <= 0.9 * size:
                        options.append(((dp + dm) / size, ni, two[i][1], two[j][1], pid, used))
            ep = min(ends, key=lambda e_: dist(pc, e_[0]))          # else each mark by its nearest end: a chain of
            em = min(ends, key=lambda e_: dist(mc, e_[0]))          # parts, two terminals
            if ep[1] != em[1] and dist(pc, ep[0]) <= 0.9 * ep[3] and dist(mc, em[0]) <= 0.9 * em[3]:
                same = ep[2] if ep[2] is not None and ep[2] == em[2] else None
                options.append((1.0 + (dist(pc, ep[0]) / ep[3] + dist(mc, em[0]) / em[3]), ni, ep[1], em[1], same, used))
    out, taken_n, taken_s = [], set(), set()
    for cost, ni, plus, minus, pid, used in sorted(options, key=lambda o: o[0]):
        if ni in taken_n or used & taken_s or plus == minus:
            continue
        taken_n.add(ni)
        taken_s |= used
        k, name, _, _ = names[ni]
        out.append((name, plus, minus, pid, k, used))
    return out



def _source_marks(texts, placed, skip, lab, node, h, w) -> dict:
    """a voltage source's '+' / '-' drawn beside it, outside its symbol (an AC one's too: its phasor's reference):
    {source id: (its '+' node, {the marks' labels})} - a mark counts when this source is the part nearest it and it
    sits clearly by one of its two ends (both marks: by different ends)"""
    marks = []
    for k, (tb, t, _) in enumerate(texts):
        t = (t or "").strip()
        if k in skip or t not in _SIGN:
            continue
        c = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        if _own_mark(c, placed):
            continue
        marks.append((k, _SIGN[t], c))
    if not marks:
        return {}

    def gap(c, b):
        return float(np.hypot(max(b[0] - c[0], 0, c[0] - b[2]), max(b[1] - c[1], 0, c[1] - b[3])))
    out = {}
    for comp, b, e, cls in placed:
        if comp.type not in ("voltage_source", "dependent_voltage_source") or len(e) != 2 or None in e:
            continue
        size = max(b[2] - b[0], b[3] - b[1])
        mine = [m for m in marks if gap(m[2], b) <= 0.6 * size and
                min(placed, key=lambda p: gap(m[2], p[1]))[0] is comp]
        pts = _end_points(b, lab, e, h, w) if mine else None
        if pts is None:
            continue
        votes = set()
        for k, sign, c in mine:
            d0, d1 = (float(np.hypot(c[0] - x, c[1] - y)) for x, y in pts)
            if min(d0, d1) > 0.7 * max(d0, d1):
                votes = None                                # a mark halfway between the ends: no reading
                break
            near = 0 if d0 < d1 else 1
            votes.add(near if sign == "+" else 1 - near)
        if votes and len(votes) == 1:
            out[comp.id] = (node(e[votes.pop()]), {m[0] for m in mine})
    return out



def _port_marks(texts, placed, terminals, ports, skip):
    """a port's '+' / '-' drawn by its two open terminals (no voltage's name by them) -> (the '+' terminal's node,
    {the marks' labels}) or None"""
    if len(ports) != 2 or not terminals:
        return None
    sizes = [max(pb[2] - pb[0], pb[3] - pb[1]) for _, pb, _, cls in placed if cls in TWO_ENDS]
    if not sizes:
        return None
    reach = 0.6 * float(np.median(sizes))
    votes, used = set(), set()
    for k, (tb, t, _) in enumerate(texts):
        t = (t or "").strip()
        if k in skip or t not in _SIGN:
            continue
        c = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        if _own_mark(c, placed):
            continue
        nd, (x, y) = min(terminals, key=lambda tm: np.hypot(tm[1][0] - c[0], tm[1][1] - c[1]))
        if nd not in ports or np.hypot(x - c[0], y - c[1]) > reach:
            continue
        votes.add(nd if _SIGN[t] == "+" else next(p for p in ports if p != nd))
        used.add(k)
    return (votes.pop(), used) if len(votes) == 1 else None


def _arrow_part(arrow, parts):
    """the two-terminal part an arrow runs beside and along: (index into parts, +1 if the arrow points from the
    part's end 0 to its end 1, else -1), or None. parts: [(end 0 (x, y), end 1 (x, y))]"""
    (tx, ty), (sx, sy) = arrow
    ax, ay = tx - sx, ty - sy
    la = (ax * ax + ay * ay) ** 0.5
    mx, my = (tx + sx) / 2, (ty + sy) / 2
    best = None
    for k, ((x0, y0), (x1, y1)) in enumerate(parts):
        px, py = x1 - x0, y1 - y0
        lp = (px * px + py * py) ** 0.5
        if lp < 1 or la < 1:
            continue
        cos = (ax * px + ay * py) / (la * lp)
        if abs(cos) < 0.85:                            # not along it (more than ~30 degrees off)
            continue
        t = ((mx - x0) * px + (my - y0) * py) / (lp * lp)          # where the arrow's middle falls along the part
        side = abs((mx - x0) * py - (my - y0) * px) / lp            # how far beside it
        if not -0.6 <= t <= 1.6 or side > 1.2 * lp + 0.5 * la:
            continue
        d = side + lp * max(0.0, -t, t - 1)
        if best is None or d < best[0]:
            best = (d, k, 1 if cos > 0 else -1)
    return None if best is None else best[1:]


def netlist_from_boxes(mask: np.ndarray, comps, texts=(), gap: float | None = None, repair: float = 8.0,
                       polarity=None, return_debug: bool = False, bridge: float = 60.0, guard: bool = True):
    """-> grader.circuit.Netlist (+ debug dict: wire labels, per-component end labels)
    polarity: a recog.polarity.PolarityModel, None = models/polarity if trained, False = leave polarity unread;
    guard: look for a part's symbol left in the wires (Netlist.notes, so its problems) - a part not detected"""
    from grader.circuit import Component, Netlist
    h, w = mask.shape
    sw = stroke_width(mask)
    gap = gap if gap is not None else max(2.0, 1.2 * sw)          # a pen lifted for this long still joins
    wires = mask.copy()
    cut = np.zeros_like(mask)
    for cls, b in comps:
        x0, y0, x1, y1 = _clip(b, h, w, grow=gap if cls == "crossover" else 0)
        cut[y0:y1, x0:x1] = True
    texts = [(t[0], t[1], t[2] if len(t) > 2 else 1.0) for t in texts]      # (box, text, how sure its reading is)
    part_cut = cut.copy()
    for b, _, _ in texts:
        x0, y0, x1, y1 = _clip(b, h, w)
        cut[y0:y1, x0:x1] = True
    wires &= ~cut
    under_labels = mask & cut & ~part_cut                          # ink the label boxes hide (a symbol's arc too)
    r = int(np.ceil(gap / 2))
    joined = ndi.binary_dilation(wires, structure=np.ones((2 * r + 1, 2 * r + 1))) & ~cut if r > 0 else wires
    lab, n = ndi.label(joined, structure=np.ones((3, 3)))
    lab[~wires] = 0                                                # labels of the real wire pixels
    uf = _Union()
    band = max(3, int(2 * gap))

    def labels_in(x0, y0, x1, y1):
        sub = lab[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        v, c = np.unique(sub[sub > 0], return_counts=True)
        return dict(zip(v.tolist(), c.tolist()))
    for cls, b in comps:                                           # crossovers: left<->right, top<->bottom
        if cls != "crossover":
            continue
        x0, y0, x1, y1 = _clip(b, h, w, grow=gap)
        for side_a, side_b in (((x0 - band, y0, x0, y1), (x1, y0, x1 + band, y1)),
                               ((x0, y0 - band, x1, y0), (x0, y1, x1, y1 + band))):
            la, lb = labels_in(*side_a), labels_in(*side_b)
            if la and lb:
                uf.join(max(la, key=la.get), max(lb, key=lb.get))
    ends = []
    objs = ndi.find_objects(lab)
    for k, (cls, b) in enumerate(comps):
        if cls in ("crossover", "text"):
            ends.append(None)
            continue
        x0, y0, x1, y1 = _clip(b, h, w)
        sides = {"left": labels_in(x0 - band, y0, x0, y1), "right": labels_in(x1, y0, x1 + band, y1),
                 "top": labels_in(x0, y0 - band, x1, y0), "bottom": labels_in(x0, y1, x1, y1 + band)}
        # a wire leads into the symbol: it leaves the box squarely (a current arrow or a label drawn beside the
        # part runs along it), it runs away from the box (a symbol's stroke sticking out of a tight box does
        # not), and, when the pen did not lift, it meets the symbol's own strokes at the box's edge
        near = _symbol_near(mask, (x0, y0, x1, y1), gap)
        deep = 4 * band
        strips = {"left": (lab[y0:y1, max(0, x0 - deep):x0], 1, lambda ys, xs, w_: w_ - xs),
                  "right": (lab[y0:y1, x1:x1 + deep], 1, lambda ys, xs, w_: xs + 1),
                  "top": (lab[max(0, y0 - deep):y0, x0:x1], 0, lambda ys, xs, w_: w_ - ys),
                  "bottom": (lab[y1:y1 + deep, x0:x1], 0, lambda ys, xs, w_: ys + 1)}
        for sd, d in sides.items():
            strip, along_x, outward = strips[sd]
            for v in list(d):
                o = objs[v - 1]
                reach = max(x0 - o[1].start, o[1].stop - x1, y0 - o[0].start, o[0].stop - y1, 0)
                touch = near is not None and bool((near[o] & (lab[o] == v)).any())
                ys, xs = np.nonzero(strip == v)
                out_ = outward(ys, xs, strip.shape[1] if along_x else strip.shape[0]).max() if len(xs) else 0
                along = np.ptp(xs if along_x == 0 else ys) if len(xs) else 0
                square = out_ / max(1.0, out_ + along)
                d[v] = d[v] * square ** 2 * (1.5 if touch else 1.0) * min(1.0, reach / (3.0 * band))
        best = {s: (max(d, key=d.get) if d else None) for s, d in sides.items()}
        weight = {s: (max(d.values()) if d else 0) for s, d in sides.items()}
        three = _switch_ends(sides, weight, b, lab, h, w) if cls == "switch" else None
        if three is not None:          # a two-way switch (common end, throw 1, throw 2) is drawn at one throw or the
            pts = _end_points(b, lab, three, h, w)                 # other: three wires without that are taken as two
            if pts is None or _switch_state(mask, b, pts) not in ("1", "2"):
                three = None
        if three is not None:
            ends.append(three)
        elif cls in TWO_ENDS:
            axis = ("left", "right") if weight["left"] + weight["right"] >= weight["top"] + weight["bottom"] else ("top", "bottom")
            pair = [best[axis[0]], best[axis[1]]]
            if None in pair or pair[0] == pair[1]:
                # a wire may come in at a corner or on the other axis: the two strongest pieces on different sides
                cands = sorted(((weight[sd], best[sd], sd) for sd in sides if best[sd] is not None), reverse=True)
                for i, (_, la, sa) in enumerate(cands):
                    other = next((lb for _, lb, sb in cands[i + 1:] if lb != la), None)
                    if other is not None:
                        pair = [la, other]
                        break
            ends.append(pair)
        elif cls in ONE_END:
            s = max(weight, key=weight.get)
            ends.append([best[s]] if weight[s] else [None])
        else:                                                      # transistors, op-amps, ICs: every wire reaching them
            allv = {}
            for d in sides.values():
                for v, c in d.items():
                    allv[v] = allv.get(v, 0) + c
            ends.append([v for v, c in allv.items() if c >= 2 * sw] or [None])
    touches = {}
    for e in ends:
        for v in (e or []):
            if v is not None:
                touches[uf.find(v)] = touches.get(uf.find(v), 0) + 1
    reach = repair * sw
    if reach > 0:
        for v, k in sorted(touches.items()):
            if k != 1 or uf.find(v) != v:
                continue                                           # only a node that one component end reaches
            members = [x for x in range(1, n + 1) if uf.find(x) == v]
            others = [x for x in range(1, n + 1) if uf.find(x) != v and uf.find(x) in touches]
            if not others:
                continue
            # only the window around the node's pieces matters (within reach of them)
            sl = [objs[x - 1] for x in members if objs[x - 1] is not None]
            r_ = int(np.ceil(reach)) + 1
            y0_, y1_ = max(0, min(o[0].start for o in sl) - r_), min(h, max(o[0].stop for o in sl) + r_)
            x0_, x1_ = max(0, min(o[1].start for o in sl) - r_), min(w, max(o[1].stop for o in sl) + r_)
            win = lab[y0_:y1_, x0_:x1_]
            dist = ndi.distance_transform_edt(~np.isin(win, members))
            near_ = [x for x in others if (win == x).any()]
            if not near_:
                continue
            d = np.atleast_1d(np.asarray(ndi.minimum(dist, win, near_), dtype=float))
            j = int(np.argmin(d))
            if d[j] <= reach:
                uf.join(v, near_[j])
    # two wire ends that stop short of everything and face each other along one line are one wire (a stretch
    # lost on a notebook rule, a long pen lift): joined across up to `bridge` pen widths when nothing lies between
    if bridge > 0:
        roots_touched = {uf.find(v) for v in touches}
        keep = {x for x in range(1, n + 1) if uf.find(x) in roots_touched}
        grown = [_clip(b, h, w, grow=band + gap) for cls, b in comps]
        tips = _loose_ends(wires, lab, grown, sw, keep)
        boxes_px = np.zeros_like(wires)
        for cls, b in comps:
            x0, y0, x1, y1 = _clip(b, h, w)
            boxes_px[y0:y1, x0:x1] = True
        pairs = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                (px, py), dp, vp = tips[i]
                (qx, qy), dq, vq = tips[j]
                vx, vy = qx - px, qy - py
                d = float(np.hypot(vx, vy))
                if d < 1 or d > bridge * sw:
                    continue
                if (dp[0] * vx + dp[1] * vy) / d < 0.9 or -(dq[0] * vx + dq[1] * vy) / d < 0.9:
                    continue                               # not facing each other (within ~25 degrees)
                pairs.append((d, i, j))
        for d, i, j in sorted(pairs):
            (px, py), _, vp = tips[i]
            (qx, qy), _, vq = tips[j]
            if uf.find(vp) == uf.find(vq):
                continue
            ts = np.linspace(0, 1, max(2, int(d)))[1:-1]
            xs, ys = (px + ts * (qx - px)).astype(int), (py + ts * (qy - py)).astype(int)
            between = {uf.find(int(v)) for v in lab[ys, xs] if v} - {uf.find(vp), uf.find(vq)}
            if boxes_px[ys, xs].any() or between & roots_touched:
                continue                                   # a part or another wire of the circuit in the way
            uf.join(vp, vq)
    ground = [uf.find(e[0]) for (cls, _), e in zip(comps, ends) if cls == "gnd" and e and e[0] is not None]
    root0 = ground[0] if ground else None
    for g in ground[1:]:
        uf.join(root0, g)
    root0 = uf.find(root0) if root0 is not None else None
    names, counter = {}, {}

    def node(v):
        if v is None:
            return None
        rv = uf.find(v)
        if root0 is not None and rv == root0:
            return "0"
        return names.setdefault(rv, f"n{len(names) + 1}")
    netlist = Netlist(ground="0" if root0 is not None else None)
    placed = []
    for (cls, b), e in zip(comps, ends):
        if e is None or cls == "gnd":
            continue
        if cls == "terminal":                                       # a port's open circle: its node is a terminal
            if e and e[0] is not None:
                netlist.terminals.append((node(e[0]), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
                if node(e[0]) not in netlist.ports:
                    netlist.ports.append(node(e[0]))
            continue
        kind = KIND.get(cls, "other")
        counter[kind] = counter.get(kind, 0) + 1
        comp = Component(f"{PREFIX[kind]}{counter[kind]}", kind, [node(v) for v in e if v is not None])
        comp.ac = cls in ("voltage_ac", "current_ac")
        if cls in ("bjt", "fet", "opamp") and polarity is not False and e and None not in e:
            rm = _roles_model()                                     # which leg is which (In+ / In- / Out, B C E ...)
            pts = _end_points(b, lab, e, h, w) if rm is not None else None
            if pts is not None:
                roles, kind = rm.assign(mask, b, pts, cls)
                if roles:
                    comp.roles = {r: node(v) for r, v in zip(roles, e) if r}
                    order = [comp.roles[r] for r in rm.roles[cls] if r in comp.roles]
                    comp.nodes = order + [node(v) for r, v in zip(roles, e) if not r]
                    if kind:
                        comp.type = kind                            # npn / pnp, nmos / pmos
        if cls in GATES and e and None not in e and len(e) >= 2:  # a gate's output and inputs
            pts = _end_points(b, lab, e, h, w)
            k = _gate_output(mask, b, pts) if pts is not None else None
            if k is not None:
                comp.roles = {"out": node(e[k])}
                for j, v in enumerate([v for i, v in enumerate(e) if i != k], 1):
                    comp.roles[f"in{j}"] = node(v)
                comp.nodes = [comp.roles["out"]] + [comp.roles[f"in{j}"] for j in range(1, len(e))]
        if cls == "transformer" and e and None not in e and len(e) == 4:
            pts = _end_points(b, lab, e, h, w)
            sides = _transformer_sides(pts, b) if pts is not None else None
            if sides is not None:                                   # primary / secondary, the dots taken at the top
                dotted = _coil_dots(mask, b, pts, sides, sw, [t[0] for t in texts])
                if dotted is not None:                              # unless drawn: each coil's '+' at its dot
                    sides = dotted
                    comp.state = "dots read"
                comp.roles = {r: node(e[k]) for r, k in sides.items()}
                comp.nodes = [comp.roles[r] for r in ("P+", "P-", "S+", "S-")]
        if cls == "switch" and e and None not in e and len(e) in (2, 3):
            pts = _end_points(b, lab, e, h, w)
            comp.state = _switch_state(mask, b, pts) if pts is not None else None   # as drawn: before t = 0
        if cls in POLAR and len(e) == 2 and None not in e and polarity is not False:
            side = _first_end(mask, b, lab, e, cls, polarity)
            if side is not None:
                comp.plus = node(e[side])          # '+' / the arrow's head / the cathode
        if cls == "voltage_rail":                                   # a '+5V' rail: a source from its node to ground
            comp.plus = comp.nodes[0] if comp.nodes else None
            comp.nodes = comp.nodes + ["0"]
            netlist.ground = "0"                                    # a rail implies the ground it is measured from
        netlist.components.append(comp)
        placed.append((comp, b, e, cls))
    in_circuit = {uf.find(v) for e in ends for v in (e or []) if v is not None}
    wire_px = np.isin(lab, [x for x in range(1, n + 1) if uf.find(x) in in_circuit]) if in_circuit else np.zeros_like(wires)
    if guard:                                                       # a part's symbol left in the wires: not detected?
        boxes_g = [_clip(b, h, w, grow=band + gap) for _, b in comps] + [_clip(t[0], h, w, grow=gap) for t in texts]
        keep_g = {x for x in range(1, n + 1) if uf.find(x) in in_circuit}
        sizes = [max(b[2] - b[0], b[3] - b[1]) for cls, b in comps if cls in TWO_ENDS]
        med = float(np.median(sizes)) if sizes else 0.0
        for what, (x, y) in _symbol_ink(wire_px, lab, keep_g, boxes_g, sw, med,
                                        hidden=under_labels, parts=[b for cls, b in comps if cls not in ("crossover",)]):
            if what == "dense":
                netlist.notes.append("the strokes round the circuit are too dense to tell its wires (a table, squared "
                                     "paper, a shadow taken for ink?)")
            else:
                netlist.notes.append(f"{what} in a wire at ({x:.0f}, {y:.0f}): a part's symbol not detected?")
    # voltages marked with '+' / '-' and a name: the name is the voltage's, not a part's (a source's own name stays its)
    from grader.circuit import Voltage
    used_texts = set()
    for name, plus, minus, pid, k_name, k_marks in _voltage_marks(texts, placed, netlist.terminals, lab, node, h, w):
        comp = next((c for c in netlist.components if c.id == pid), None)
        if comp is not None and comp.type in ("voltage_source", "dependent_voltage_source"):
            comp.plus = plus                                        # a source's marks: its polarity (an AC one's
            comp.conf = min([comp.conf] + [texts[k][2] for k in k_marks | {k_name}])   # reference), its name kept
            if not k_marks and comp.label is None:                  # '+ V_1 -' in one label by it
                comp.label = name
            used_texts |= k_marks
            continue
        netlist.voltages.append(Voltage(name, plus, minus, pid, min(texts[k][2] for k in k_marks | {k_name})))
        used_texts |= k_marks | {k_name}
    for cid, (plus, used) in _source_marks(texts, placed, used_texts, lab, node, h, w).items():
        comp = next(c for c in netlist.components if c.id == cid)
        comp.plus = plus                                            # '+' / '-' drawn by a source: its polarity
        comp.conf = min([comp.conf] + [texts[k][2] for k in used])
        used_texts |= used
    pm = _port_marks(texts, placed, netlist.terminals, netlist.ports, used_texts)
    if pm is not None:                                              # '+' / '-' by a port's terminals: which is '+'
        netlist.ports = [pm[0]] + [p for p in netlist.ports if p != pm[0]]
        netlist.ports_marked = True
        used_texts |= pm[1]
    current_texts = []
    for k_text, (tb, text, tconf) in enumerate(texts):              # a label belongs to the nearest component
        t = (text or "").strip()
        if not placed or not t or k_text in used_texts:
            continue
        cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2

        def dist(pb):
            bx = max(pb[0] - cx, 0, cx - pb[2])
            by = max(pb[1] - cy, 0, cy - pb[3])
            return (bx * bx + by * by) ** 0.5
        comp, pb, _, _ = min(placed, key=lambda p: dist(p[1]))
        if _CURRENT.match(t) and not (comp.type == "current_source" and comp.label is None and dist(pb) < 0.5 * max(pb[2] - pb[0], pb[3] - pb[1])):
            current_texts.append(((cx, cy), t, comp))
            continue
        if _NODE.match(t) and wire_px.any():
            r = int(3 * max(tb[2] - tb[0], tb[3] - tb[1])) + 4
            y0, x0 = max(0, int(cy) - r), max(0, int(cx) - r)
            ys, xs = np.nonzero(wire_px[y0:int(cy) + r, x0:int(cx) + r])
            if len(xs):
                d2 = (xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2
                j = int(np.argmin(d2))
                if d2[j] ** 0.5 < dist(pb):
                    nd = node(int(lab[ys[j] + y0, xs[j] + x0]))
                    if nd is not None:
                        netlist.node_labels[nd] = t
                        continue
        from grader.circuit import _AC, _BETA, _PARAM, complex_value, turns_ratio
        if complex_value(t) is not None:                            # 'j4Ω', '3+j4', '120∠0°V': a phasor-domain label
            unit = re.search(r"(Ω|\\Omega|ohm|V|A)\s*$", t)
            kinds = ("voltage_source", "current_source") if unit and unit.group(1) in ("V", "A") else \
                ("resistor", "inductor", "capacitor", "lamp") if unit else \
                ("resistor", "inductor", "capacitor", "lamp", "voltage_source", "current_source")
            cand = [p for p in placed if p[0].type in kinds and p[0].value is None]
            near = min(cand, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = t
                near[0].conf = min(near[0].conf, tconf)
                if near[0].type in ("voltage_source", "current_source"):
                    near[0].ac = True
            continue
        cm = re.fullmatch(r"\s*(?P<key>M|k|L_?\{?[12]\}?)\s*=\s*(?P<val>\d+(?:[.,]\d+)?\s*[munµ]?\s*H?)\s*", t)
        hv = re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*[munµ]?\s*H\s*", t)
        trs = [p for p in placed if p[0].type == "transformer" and turns_ratio(p[0].value) is None]
        near = min(trs, key=lambda p: dist(p[1]), default=None)
        if (cm or hv) and near is not None and dist(near[1]) <= 2 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
            free_l = [p for p in placed if p[0].type == "inductor" and p[0].value is None]
            if cm or not free_l or dist(near[1]) < min(dist(p[1]) for p in free_l):
                if cm:                                              # 'M = 2H', 'L_1 = 4H' by coupled coils
                    key = cm.group("key")
                    near[0].params[re.sub(r"[_{}]", "", key)] = cm.group("val")
                else:                                               # a bare '4H': the coil on its side
                    side = "L1" if cx < (near[1][0] + near[1][2]) / 2 else "L2"
                    near[0].params.setdefault(side, t)
                near[0].conf = min(near[0].conf, tconf)
                continue
        pm = _PARAM.match(t)
        if pm:                                                      # 'V_t = 1V', 'K = 0.5mA/V^2' by a MOSFET
            fs = [p for p in placed if p[0].type in ("fet", "nmos", "pmos")]
            near = min(fs, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].params[pm.group("key")] = pm.group("val")
                near[0].conf = min(near[0].conf, tconf)
            continue
        if _BETA.match(t):                                          # 'β=100' by a transistor: its current gain
            qs = [p for p in placed if p[0].type in ("bjt", "npn", "pnp") and p[0].value is None]
            near = min(qs, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = t
                near[0].conf = min(near[0].conf, tconf)
            continue
        if turns_ratio(t) is not None:                              # '1:2' by a transformer: its turns ratio
            trs = [p for p in placed if p[0].type == "transformer" and p[0].value is None]
            near = min(trs, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = t
                near[0].conf = min(near[0].conf, tconf)
            continue
        if _AC.match(t):                                            # '10cos(1000t)' by an AC source: its value
            acs = [p for p in placed if p[0].ac and p[0].value is None]
            near = min(acs, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = t
                near[0].conf = min(near[0].conf, tconf)
            continue
        if _DEP.match(t) and not _NAME.match(t):                    # '2 I_x' by a diamond: what it follows
            deps = [p for p in placed if p[0].type.startswith("dependent") and p[0].value is None]
            near = min(deps, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = t
                near[0].conf = min(near[0].conf, tconf)
            continue
        if re.fullmatch(r"\s*t\s*=\s*0\s*(?:\^?\s*[+-]?)?\s*", t):          # 't = 0' by a switch: it is the one that moves
            sws = [p for p in placed if p[0].type == "switch"]
            near = min(sws, key=lambda p: dist(p[1]), default=None)
            if near is not None and dist(near[1]) <= 3 * max(near[1][2] - near[1][0], near[1][3] - near[1][1]):
                near[0].value = "t=0"
            continue
        is_value = bool(_VALUE.match(t))
        if not is_value and not _NAME.match(t):
            continue
        # the nearest part the label can belong to: 'R2' / '10kΩ' to a resistor, 'E1' / '5V' to a voltage source ...
        kinds = _label_kinds(t, is_value)
        for comp, pb, _, _ in sorted(placed, key=lambda p: dist(p[1])):
            if dist(pb) > 3 * max(pb[2] - pb[0], pb[3] - pb[1]):
                break
            if kinds and comp.type not in kinds:
                continue
            if is_value and comp.value is None:
                comp.value = t
            elif not is_value and comp.label is None:
                comp.label = t
            else:
                continue
            comp.conf = min(comp.conf, tconf)
            break
    # a part named on the drawing is called by that name ('R2' is the resistor written R2); the others keep a
    # name of their kind that no written one took
    names_ = [re.sub(r"[_{}\s]", "", c.label) for c in netlist.components if c.label]
    taken = {x for x in names_ if names_.count(x) == 1}
    for c in netlist.components:
        if c.label and re.sub(r"[_{}\s]", "", c.label) in taken:
            c.id = re.sub(r"[_{}\s]", "", c.label)
    used_ids = set(taken)
    for c in netlist.components:
        if not (c.label and c.id in taken):
            base, k = re.match(r"[A-Za-z]+", c.id).group(0), 1
            while c.id in used_ids:
                k += 1
                c.id = f"{base}{k}"
            used_ids.add(c.id)
    # current arrows: beside a two-terminal part, pointing along it
    from grader.circuit import Current
    loose = [x for x in range(1, n + 1) if uf.find(x) not in in_circuit]
    two = [(comp, _end_points(b, lab, e, h, w), e) for comp, b, e, cls in placed
           if cls in TWO_ENDS and len(e) == 2 and None not in e]
    two = [(comp, pts, e) for comp, pts, e in two if pts is not None]
    axes = []
    for _, ((x0, y0), (x1, y1)), _ in two:
        ln = float(np.hypot(x1 - x0, y1 - y0))
        if ln >= 1:
            axes.append((((x1 - x0) / ln, (y1 - y0) / ln), ((x0 + x1) / 2, (y0 + y1) / 2), 1.5 * ln))
    arrows = _arrows(lab, loose, sw, axes)
    found = []                                                      # (arrow, comp, frm, to)
    for arrow in arrows:
        hit = _arrow_part(arrow, [pts for _, pts, _ in two])
        if hit is not None:
            comp, _, e = two[hit[0]]
            frm, to = (node(e[0]), node(e[1])) if hit[1] > 0 else (node(e[1]), node(e[0]))
            found.append((arrow, comp, frm, to))
    if current_texts and len(found) < len(current_texts) and wire_px.any():
        # arrows drawn on a wire: the current of the part that wire leads to (the way the head points; or the part
        # it comes from, when it branches ahead) - a part's two ends taken from where its wires meet its box
        wa = _wire_arrows(wire_px, sw)
        if wa:
            try:
                from skimage.morphology import skeletonize
                skp = _prune(skeletonize(wire_px), int(round(2 * sw)))
            except ImportError:
                skp = None
            stop_at = np.zeros(wire_px.shape, np.int32)
            for j, (comp, pts, e) in enumerate(two):
                b = next(pb for c_, pb, _, _ in placed if c_ is comp)
                x0, y0, x1, y1 = _clip(b, h, w, grow=band + gap)
                stop_at[y0:y1, x0:x1] = j + 1
            ys_k, xs_k = np.nonzero(skp) if skp is not None else (np.array([]), np.array([]))
            for (tx, ty), (ux, uy) in wa:
                if not len(xs_k):
                    break
                j0 = int(np.argmin((xs_k - tx) ** 2 + (ys_k - ty) ** 2))
                start = (int(ys_k[j0]), int(xs_k[j0]))
                wire_node = node(int(lab[int(round(ty)), int(round(tx))]) or None) if lab[int(round(ty)), int(round(tx))] else None
                if wire_node is None:
                    v = lab[start]
                    wire_node = node(int(v)) if v else None
                for sense in (1, -1):                       # ahead into a part, else back out of one
                    hit = _walk_wire(skp, start, (sense * ux, sense * uy), stop_at, 2 * (h + w))
                    if hit:
                        comp, _, e = two[hit - 1]
                        a, b_ = node(e[0]), node(e[1])
                        if wire_node not in (a, b_) or a == b_:
                            break
                        other = b_ if wire_node == a else a
                        frm, to = (wire_node, other) if sense == 1 else (other, wire_node)
                        tail = (tx - 3 * sw * ux, ty - 3 * sw * uy)
                        found.append(((tail, (tx, ty)), comp, frm, to))
                        break
    used = set()
    for (cx, cy), t, comp in sorted(current_texts, key=lambda x: x[1]):
        def gap_to(f):
            (tx, ty), (sx, sy) = f[0]
            return min(((cx - (tx + sx) / 2) ** 2 + (cy - (ty + sy) / 2) ** 2) ** 0.5,
                       ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5, ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5)
        free = [k for k in range(len(found)) if k not in used]
        k = min(free, key=lambda k: gap_to(found[k]), default=None)
        if k is not None:
            (tx, ty), (sx, sy) = found[k][0]
            if gap_to(found[k]) <= 1.5 * ((tx - sx) ** 2 + (ty - sy) ** 2) ** 0.5 + 40:
                used.add(k)
                netlist.currents.append(Current(t, found[k][1].id, found[k][2], found[k][3]))
                continue
        netlist.currents.append(Current(t, comp.id, None, None))
    if return_debug:
        return netlist, {"labels": lab, "stroke_width": sw, "gap": gap, "ends": ends, "arrows": arrows,
                         "arrow_parts": [(f[1].id, f[2], f[3]) for f in found]}
    return netlist
