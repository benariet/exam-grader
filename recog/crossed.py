"""Writing the student crossed out - not part of the answer, so not read into it. The ways it is done (all seen on
real exams):

  scribble : a patch inked over densely - a word, a short formula, a circled choice blotted out: ink covers far more of
             a letter-sized window than writing does, around many tiny holes
  strike   : a line along the middle of a word or a formula - straight, or wavy as a hand runs it: it crosses many
             short strokes (letters) - not long ones (a table's or a grid's lines) - and touches little else (a fraction
             bar has a gap to what is above and below it, an underline or an overline has writing on one side only)
  hatch    : long slanted lines over a block: an X or a zigzag (lines of both slants crossing each other), parallel
             slashes each through writing, or one long slash through a lot of writing

Lines are found as unbroken chains of ink that run on far in one direction without wandering (a dynamic program over
the columns of the page turned so that direction is level: each ink pixel's longest chain from the left, a step up or
down allowed at a small cost) - writing breaks between letters, a line drawn over it does not; then across each chain,
column by column, does the ink go on past it both above and below (a stroke crossing it), and how far (a letter: short;
a grid's or a wire's line: long).

    regions = crossed_regions(ink, lh)          # ink: the cleaned page < 128 (recog.page.clean_scan), lh: letter height
    covered(piece_box, regions, ink) -> {"kind", "share"} or None   (share: how much of the piece is under it)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

DENSE = 0.42            # ink share of a letter-sized window above which it is a scribble (writing stays under ~0.3)
HOLES_PER_LH2 = 1.5     # small enclosed holes per letter-size square in a dense patch (a scribble's loops)
SLANT = (20.0, 78.0)    # degrees from the horizontal: a hatch line (steeper is a wire, a stem; flatter a strike)
PASSES = (0, 22, -22, 45, -45, 67, -67)     # directions looked along (each covers about 11 degrees either way)
LOOPS = 8               # small holes in one stroke from which it is a looped scribble
HEADER = 0.11          # the top of an exam page: the printed header and logo - nothing crossed out there
STEP_COST = 0.25        # a chain's step up or down costs this much of its length (straight chains win)


def _runs(flags: np.ndarray, gap: int = 1) -> list:
    """[(start, stop)] of the True runs, breaks up to gap merged"""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b > gap + 1:
            out.append((a, b + 1))
            a = i
        b = i
    out.append((a, b + 1))
    return out


# ---------------------------------------------------------------- the page turned: a direction made level, and back
def _turn(ink: np.ndarray, deg: float):
    """the page turned so that a line at deg (degrees, y down: positive goes down to the right) is level ->
    (turned page, back) where back(ox, oy) -> (x, y) on the page"""
    if deg == 0:
        return ink, lambda ox, oy: (ox, oy)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    H, W = ink.shape
    Wo, Ho = int(W * abs(c) + H * abs(s)) + 2, int(W * abs(s) + H * abs(c)) + 2
    m = np.array([[c, s], [-s, c]])                     # (out row, col) -> (in row, col), about the centres
    cin, cout = np.array([(H - 1) / 2, (W - 1) / 2]), np.array([(Ho - 1) / 2, (Wo - 1) / 2])
    off = cin - m @ cout
    out = ndi.affine_transform(ink.astype(np.uint8), m, offset=off, output_shape=(Ho, Wo), order=0) > 0

    def back(ox, oy):
        r, q = m @ np.array([oy, ox], dtype=float) + off
        return float(q), float(r)
    return out, back


def _chains(M: np.ndarray, min_len: float, max_dev: float, max_gap: int = 2) -> list:
    """level chains of ink in M at least min_len long - breaks of up to max_gap pixels bridged (a thin pen's line
    breaks) - wandering no more than max_dev (or 3% of their length, a long line drawn by hand bows) from their
    straight fit (a wave within that is kept: a hand's strike) -> [(xs, ys)] (one y per column; a bridged break's
    columns included - not ink)"""
    H, W = M.shape
    score = np.zeros((H, W), np.float32)
    step = np.zeros((H, W), np.int8)
    prev = np.zeros(H, np.float32)
    gaps = np.zeros(H, np.int16)
    ninf = np.float32(-1e9)
    for x in range(W):
        up = np.r_[ninf, prev[:-1]] - STEP_COST           # from the row above (y - 1) in the column before
        dn = np.r_[prev[1:], ninf] - STEP_COST
        best = np.maximum(prev, np.maximum(up, dn))
        st = np.where(best == prev, 0, np.where(best == up, -1, 1)).astype(np.int8)
        g_from = np.where(best == prev, gaps, np.where(best == up, np.r_[0, gaps[:-1]], np.r_[gaps[1:], 0]))
        col = M[:, x]
        bridge = ~col & (best > 1) & (g_from < max_gap)
        cur = np.where(col, np.maximum(best, 0) + 1, np.where(bridge, best - 0.5, 0)).astype(np.float32)
        gaps = np.where(col, 0, np.where(bridge, g_from + 1, 0)).astype(np.int16)
        score[:, x] = cur
        step[:, x] = np.where((col | bridge) & (best > 0), st, 0)
        prev = cur
    ys, xs = np.nonzero(score >= min_len)
    if not len(ys):
        return []
    order = np.argsort(-score[ys, xs], kind="stable")
    used = np.zeros((H, W), bool)
    out = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if used[y, x]:
            continue
        py, px = [], []
        while x >= 0 and score[y, x] > 0:
            py.append(y)
            px.append(x)
            dy = int(step[y, x])
            if score[y, x] <= 1:
                break
            y, x = y + dy, x - 1
        px, py = np.array(px[::-1]), np.array(py[::-1])
        ink = M[py, px]
        if len(ink) and not ink[0]:                          # (a chain starts and ends on ink)
            k = int(np.argmax(ink))
            px, py = px[k:], py[k:]
        if len(px) < min_len or used[py, px].mean() > 0.5:
            continue                                        # (mostly a longer chain's: a branch off it)
        band = max(3, int(round(0.2 * min_len / 2.2)))
        for k in range(-band, band + 1):                    # (the line's own pixels: not another chain's)
            used[np.clip(py + k, 0, H - 1), px] = True
        fit = np.polyfit(px, py, 1)
        if np.abs(py - np.polyval(fit, px)).max() > max(max_dev, 0.03 * len(px)):
            continue                                        # (a curve, a word written joined up: not a line)
        out.append((px, py))
    return out


def _across(M: np.ndarray, px: np.ndarray, py: np.ndarray, lh: float) -> dict:
    """the strokes crossing a level chain in M. The chain's own edges, column by column (where its ink run is no
    thicker than usual: elsewhere filled in between), are taken out of the ink; what still touches it - connected,
    a diagonal step allowed - from above and from below at about the same place is a stroke crossing it: short (a
    letter, at most 2.5 letter heights followed) or long (a line of a grid, a wire, another long stroke); touching
    from one side only is a touch (an underline's letters, an overline's). A fraction bar's numerator and
    denominator do not touch it at all."""
    H, W = M.shape
    on = M[py, px]                                          # (a bridged break's columns: not the line's ink)
    px, py = px[on], py[on]
    n = len(px)
    r0s, r1s = np.empty(n, int), np.empty(n, int)
    for i, (x, y) in enumerate(zip(px, py)):
        r0 = r1 = int(y)
        while r0 > 0 and M[r0 - 1, x]:
            r0 -= 1
        while r1 < H - 1 and M[r1 + 1, x]:
            r1 += 1
        r0s[i], r1s[i] = r0, r1
    run = r1s - r0s + 1
    thick = max(1.0, float(np.median(run)))
    pure = run <= thick + 1
    res = {"length": n, "short": 0, "tall": 0, "long": 0, "touch": 0, "thick": thick}
    if pure.sum() < 3:
        return res
    idx = np.arange(n)
    top = np.round(np.interp(idx, idx[pure], r0s[pure])).astype(int)
    bot = np.round(np.interp(idx, idx[pure], r1s[pure])).astype(int)
    R = int(2.5 * lh)
    wx0, wx1 = max(0, int(px[0]) - 2), min(W, int(px[-1]) + 3)
    wy0, wy1 = max(0, int(top.min()) - R), min(H, int(bot.max()) + R + 1)
    sub = M[wy0:wy1, wx0:wx1].copy()
    for x, t, b in zip(px, top, bot):
        sub[max(0, t - wy0):max(0, b + 1 - wy0), x - wx0] = False
    lab, _ = ndi.label(sub, structure=np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    ups, dns = [], []
    for x, t, b in zip(px, top, bot):
        cx = x - wx0
        u = set(lab[t - 1 - wy0, max(0, cx - 1):cx + 2].tolist()) - {0} if t - 1 - wy0 >= 0 else set()
        d = set(lab[b + 1 - wy0, max(0, cx - 1):cx + 2].tolist()) - {0} if b + 1 - wy0 < sub.shape[0] else set()
        ups.append(u)
        dns.append(d)

    def contacts(sides):
        out = []
        for a, b in _runs(np.array([bool(v) for v in sides])):
            if b - a <= 1.5 * lh:                          # (wider: running along it - not a crossing)
                out.append((a, b, set().union(*sides[a:b])))
        return out
    cu, cd = contacts(ups), contacts(dns)
    g = max(3, int(0.3 * lh))
    paired = set()
    for a, b, iu in cu:
        best = None
        for k, (c, d, idn) in enumerate(cd):
            if k not in paired and c <= b + g and d >= a - g:
                best = k
                break
        if best is None:
            res["touch"] += 1
            continue
        paired.add(best)
        c, d, idn = cd[best]
        U = max(int(top[a:b].min()) - wy0 - objs[i - 1][0].start for i in iu)
        D = max(objs[i - 1][0].stop - 1 - (int(bot[c:d].max()) - wy0) for i in idn)
        if min(U, D) >= 1.2 * lh and max(U, D) >= 2.0 * lh:
            res["long"] += 1
        elif U + D <= 1.6 * lh:                             # a letter cut through its middle
            res["short"] += 1
        else:                                               # (taller: a fraction's numerator and denominator both
            res["tall"] += 1                                #  touching its bar - not a letter)
    res["touch"] += len(cd) - len(paired)
    return res


def lines(ink: np.ndarray, lh: float) -> list:
    """the long unbroken straight-ish strokes of the page, each with what crosses it -> [{"line": (x0, y0, x1, y1),
    "angle": degrees from the horizontal (0..90), "sign": +1 going down to the right, "length", "short", "long", ...}]"""
    out = []
    for deg in PASSES:
        M, back = _turn(ink, deg)
        min_len = 2.2 * lh if deg == 0 else 3.5 * lh
        for px, py in _chains(M, min_len, max_dev=0.5 * lh):
            cr = _across(M, px, py, lh)
            fit = np.polyfit(px, py, 1)
            x0, y0 = back(px[0], np.polyval(fit, px[0]))
            x1, y1 = back(px[-1], np.polyval(fit, px[-1]))
            a = math.degrees(math.atan2(y1 - y0, x1 - x0))
            a = (a + 90) % 180 - 90                          # (-90..90]
            if abs(a - deg) > 12 and not (deg == 0 and abs(a) <= 14):
                continue                                    # (another pass's direction: found there)
            ys = [back(x, y)[1] for x, y in zip(px[:: max(1, len(px) // 8)], py[:: max(1, len(px) // 8)])]
            out.append({"line": (x0, y0, x1, y1), "angle": abs(a), "sign": 1 if a > 0 else -1,
                        "box": (min(x0, x1), min(min(ys), y0, y1), max(x0, x1), max(max(ys), y0, y1)), **cr})
    return out


# ---------------------------------------------------------------- dense patches
def scribbles(ink: np.ndarray, lh: float) -> list:
    """dense patches: [{"kind": "scribble", "box", "density", "holes"}]"""
    win = max(5, int(round(lh)))
    D = ndi.uniform_filter(ink.astype(np.float32), win)
    dense = D > DENSE
    if not dense.any():
        return []
    lab, n = ndi.label(ndi.binary_dilation(dense, iterations=max(1, win // 3)))
    out = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        core = dense[o] & (lab[o] == k)
        if core.sum() < 0.35 * lh * lh:
            continue
        y0, y1 = max(0, o[0].start - win // 2), min(ink.shape[0], o[0].stop + win // 2)
        x0, x1 = max(0, o[1].start - win // 2), min(ink.shape[1], o[1].stop + win // 2)
        sub = ink[y0:y1, x0:x1]
        bg, nb = ndi.label(~sub)
        edge = set(np.unique(np.r_[bg[0], bg[-1], bg[:, 0], bg[:, -1]]).tolist())
        sizes = ndi.sum(np.ones_like(bg), bg, range(1, nb + 1)) if nb else []
        holes = sum(1 for i, sz in enumerate(sizes, 1) if i not in edge and 1 <= sz <= 0.08 * lh * lh)
        area = (y1 - y0) * (x1 - x0) / (lh * lh)
        density = float(D[o][core].mean())
        if holes / max(area, 1e-6) < HOLES_PER_LH2 and density < DENSE + 0.15:
            continue                                       # (a filled shape, a bold dot: dense without the loops)
        out.append({"kind": "scribble", "box": (x0, y0, x1, y1), "density": round(density, 2), "holes": holes})
    return out + loops(ink, lh)


def loops(ink: np.ndarray, lh: float) -> list:
    """a pen run in loops over a word (not densely - a looser hand's scribble): one connected stroke enclosing many
    small holes where it crosses itself - writing encloses few, and bigger ones (the inside of a letter) ->
    [{"kind": "scribble", "box", "loops"}]"""
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    comps = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if max(h, w) < 1.2 * lh:
            continue
        m = lab[o] == k
        hl, nh = ndi.label(ndi.binary_fill_holes(m) & ~m)
        small = 0
        if nh:
            sz = ndi.sum(np.ones_like(hl), hl, range(1, nh + 1))
            small = int(((sz >= 2) & (sz <= 0.12 * lh * lh)).sum())
        comps.append((o, small, small / max(1.0, w / lh)))
    if not comps:
        return []
    # against this hand's own writing: a thick pen's letters touch and enclose small holes too (8 to 19 in one word
    # on a page of exam 6, all of it plain writing) - a scribble has far more than the page's usual
    usual = float(np.percentile([r for _, _, r in comps], 90))
    out = []
    for o, small, rate in comps:
        if small >= LOOPS and rate >= max(2.0, 2.5 * usual):
            out.append({"kind": "scribble", "box": (o[1].start, o[0].start, o[1].stop, o[0].stop), "loops": small,
                        "usual": round(usual, 2)})
    return out


# ---------------------------------------------------------------- putting it together
def _intersect(a, b, tol: float = 0.0) -> bool:
    """two segments cross (or come within tol of each other)"""
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    p1, p2, p3, p4 = a[:2], a[2:], b[:2], b[2:]
    d1, d2, d3, d4 = ccw(p3, p4, p1), ccw(p3, p4, p2), ccw(p1, p2, p3), ccw(p1, p2, p4)
    if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
        return True
    if tol <= 0:
        return False

    def dist(p, s):
        x0, y0, x1, y1 = s
        L2 = (x1 - x0) ** 2 + (y1 - y0) ** 2 or 1e-9
        t = max(0.0, min(1.0, ((p[0] - x0) * (x1 - x0) + (p[1] - y0) * (y1 - y0)) / L2))
        return math.hypot(p[0] - x0 - t * (x1 - x0), p[1] - y0 - t * (y1 - y0))
    return min(dist(p1, b), dist(p2, b), dist(p3, a), dist(p4, a)) <= tol


def _overlap(a, b) -> float:
    """the share of the smaller box inside the other"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def crossed_regions(ink: np.ndarray, lh: float | None = None, debug: bool = False) -> list:
    """-> [{"kind": "scribble" | "strike" | "hatch", "box": (x0, y0, x1, y1), ...}] on the page; debug: every line
    looked at too (kind "line": not a crossing-out)"""
    if lh is None:
        from recog.page import letter_height
        lh = letter_height(ink)
    lh = max(8.0, float(lh))
    out = scribbles(ink, lh)
    found = lines(ink, lh)
    rest = []
    for f in found:
        per = f["short"] / (f["length"] / lh)
        if f["angle"] <= 14:                                # a strike: a flat line through letters
            if f["short"] >= 3 and per >= 0.5 and f["long"] <= max(1, f["short"] // 4) and f["tall"] <= f["short"] // 2:
                out.append({"kind": "strike", "box": tuple(int(round(v)) for v in f["box"]),
                            "line": tuple(int(round(v)) for v in f["line"]), "crossings": f["short"], "per_lh": round(per, 2)})
                continue
        elif SLANT[0] <= f["angle"] <= SLANT[1]:
            rest.append(f)
            continue
        if debug:
            out.append({"kind": "line", **{k: v for k, v in f.items() if k != "sign"}})
    strikes = [r for r in out if r["kind"] == "strike"]
    out = [r for r in out if r["kind"] != "strike"]
    for r in sorted(strikes, key=lambda r: -(r["line"][2] - r["line"][0]) ** 2 - (r["line"][3] - r["line"][1]) ** 2):
        if not any(k["kind"] == "strike" and _overlap(k["box"], r["box"]) > 0.6 for k in out):
            out.append(r)                                   # (one line found again a row or two off: once)
    used = set()
    tol = 0.3 * lh
    for i, f in enumerate(rest):                            # an X, a zigzag: slanted lines of both slants crossing
        if i in used:
            continue
        group = {i}
        changed = True
        while changed:
            changed = False
            for j, g in enumerate(rest):
                if j not in group and j not in used and any(
                        rest[h]["sign"] != g["sign"] and _intersect(rest[h]["line"], g["line"], tol) for h in group):
                    group.add(j)
                    changed = True
        if len(group) >= 2 and sum(rest[h]["short"] + rest[h]["tall"] + rest[h]["long"] for h in group) >= 3:
            used |= group                                   # (over something - not two lines meeting on empty paper)
            out.append({"kind": "hatch", "how": "crossing", "lines": len(group),
                        "box": tuple(int(round(v)) for v in _union([rest[h]["box"] for h in group])),
                        "crossings": sum(rest[h]["short"] + rest[h]["tall"] + rest[h]["long"] for h in group)})
    through = [i for i, f in enumerate(rest) if i not in used and f["short"] >= 3 and f["short"] / (f["length"] / lh) >= 0.25]
    for i in through:                                       # parallel slashes through writing; one long slash
        if i in used:
            continue
        f = rest[i]
        par = [j for j in through if j != i and j not in used and rest[j]["sign"] == f["sign"]
               and abs(rest[j]["angle"] - f["angle"]) <= 12 and _overlap(rest[j]["box"], f["box"]) > 0.3]
        if par:
            used |= {i, *par}
            out.append({"kind": "hatch", "how": "parallel", "lines": 1 + len(par),
                        "box": tuple(int(round(v)) for v in _union([f["box"]] + [rest[j]["box"] for j in par])),
                        "crossings": f["short"] + sum(rest[j]["short"] for j in par)})
        elif f["short"] >= 6 and f["short"] / (f["length"] / lh) >= 0.4:
            used.add(i)
            out.append({"kind": "hatch", "how": "slash", "lines": 1, "box": tuple(int(round(v)) for v in f["box"]),
                        "crossings": f["short"]})
    if debug:
        out += [{"kind": "line", **{k: v for k, v in f.items() if k != "sign"}} for i, f in enumerate(rest) if i not in used]
    top = HEADER * ink.shape[0]                             # (the printed header: the university's logo is dense)
    out = [r for r in out if r["box"][3] > top]
    return _merge_patches(out, 0.5 * lh)


def _merge_patches(regions: list, gap: float) -> list:
    """scribbles and hatches touching or within gap of each other -> one region (a zigzag found in parts, a scribble
    beside it): a hatch if any part was one"""
    patches = [dict(r) for r in regions if r["kind"] in ("scribble", "hatch")]
    others = [r for r in regions if r["kind"] not in ("scribble", "hatch")]
    merged = True
    while merged:
        merged = False
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                a, b = patches[i]["box"], patches[j]["box"]
                g = gap if "hatch" in (patches[i]["kind"], patches[j]["kind"]) else 0    # (scribbles: when they overlap)
                if a[0] - g <= b[2] and b[0] - g <= a[2] and a[1] - g <= b[3] and b[1] - g <= a[3]:
                    kind = "hatch" if "hatch" in (patches[i]["kind"], patches[j]["kind"]) else "scribble"
                    patches[i] = {"kind": kind, "box": _union([a, b]),
                                  "parts": patches[i].get("parts", 1) + patches[j].get("parts", 1)}
                    del patches[j]
                    merged = True
                    break
            if merged:
                break
    others = [r for r in others if not any(_overlap(r["box"], q["box"]) > 0.7 for q in patches)]   # (inside one: it)
    return patches + others


def covered(box, regions, ink: np.ndarray | None = None):
    """a reading piece (box) under crossing-out -> {"kind", "share"} (share: of the piece's width under a strike through
    its middle part; of its ink - of its area without the ink - inside a scribble or a hatch); None when none of it is"""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    best = None
    for r in regions:
        if r["kind"] == "line":
            continue
        if r["kind"] == "strike":
            lx0, ly0, lx1, ly1 = r["line"]
            if lx1 < lx0:
                lx0, ly0, lx1, ly1 = lx1, ly1, lx0, ly0
            a, b = max(x0, lx0), min(x1, lx1)
            if b <= a:
                continue
            xm = (a + b) / 2
            ly = ly0 + (ly1 - ly0) * ((xm - lx0) / max(1, lx1 - lx0))
            if not (y0 + 0.2 * h <= ly <= y1 - 0.2 * h):
                continue                                    # (along its top or bottom: an overline, an underline)
            share = (b - a) / w
        else:
            bx0, by0, bx1, by1 = (int(round(v)) for v in r["box"])
            ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if ink is not None:
                tot = ink[y0:y1, x0:x1].sum()
                share = float(ink[iy0:iy1, ix0:ix1].sum()) / tot if tot else 0.0
            else:
                share = (ix1 - ix0) * (iy1 - iy0) / (w * h)
        if share > 0.2 and (best is None or share > best["share"]):
            best = {"kind": r["kind"], "share": round(share, 2)}
    return best
