"""CMOS gates drawn by hand, with their netlists known - a test of the circuit reader on the course's kind of drawing
(8.11: no hand-drawn CMOS circuit is published with its netlist, and the reader had never been measured on one).

Each drawing is a static CMOS gate - an inverter, NAND / NOR of 2 or 3 inputs, AND-OR-invert / OR-AND-invert - laid
out as a textbook draws it: the supply arrow on top, the PMOS network under it, the output in the middle, the NMOS
network, grounds below; each transistor's gate a short wire with its input's letter by it ('A', 'B', 'C' - the same
letter is the same input, as a student labels the gates instead of wiring them). The symbols are real hand-drawn ones:
Digitize-HCD's MOSFETs (only those its legs model never learned from - every 10th block of 50), grounds and supply
arrows, cut from their crops with each leg's place known, turned / mirrored to stand as the layout needs (the
detector may have seen their circuits: Digitize-HCD's crops are cut from its drawings); the letters CGHD's test
drafters wrote (hands the label reader never learned from), placed clear of the wires; the wires drawn by a
wavering pen, the paper and ink varied. Each symbol's and letter's box is kept with the netlist (tools/synth_failures.py
sorts the reader's misses by them).

    python tools/cmos_synth.py --out ~/eg-data/cmos_synth --n 60 [--seed 0] [--words 0.5]
        (--words: that share of the supplies and grounds drawn as their names - 'VDD' / '+5V' over a bar or at the
        wire's end, 'GND' / 'VSS' / '0V' under - as CGHD's test drafters wrote them)
        -> <out>/<name>.png and <out>/<name>.json (the netlist: each transistor's kind and its Gate / Drain / Source
           nodes, the supply as a source to ground; tools/circuit_eval.py --synth <out> measures the reader on them)
"""
import argparse
import io
import json
import math
import os
import random
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

BASE = "Digitize-HCD Dataset/Component Port Location Data/"
CELL = 170                         # a transistor's place in the layout, pixels (on a page of about 1500 x 1150, as the
#                                    reader sees a photo: its long side 1600 - a pen 2.5-4.5 pixels wide)
CELLW = 1.7 * CELL                 # its width: room on its left for its gate's wire and letter
OFF = CELLW - 0.38 * CELL          # where in its place the transistor's spine (drain - source) runs

# the gates: the pull-down network as a series / parallel expression of the inputs (the pull-up network its dual)
GATES = {"inv": "A", "nand2": ("s", "A", "B"), "nor2": ("p", "A", "B"), "nand3": ("s", "A", "B", "C"),
         "nor3": ("p", "A", "B", "C"), "aoi21": ("p", ("s", "A", "B"), "C"), "oai21": ("s", ("p", "A", "B"), "C")}


def dual(e):
    if isinstance(e, str):
        return e
    return ("p" if e[0] == "s" else "s",) + tuple(dual(x) for x in e[1:])


def inputs_of(e):
    return [e] if isinstance(e, str) else [i for x in e[1:] for i in inputs_of(x)]


# -- the symbols -------------------------------------------------------------------------------------------------------
def _frame(rgb):
    """the green box Digitize-HCD drew round the part -> (x0, y0, x1, y1) or None"""
    green = (rgb[..., 1].astype(int) - np.maximum(rgb[..., 0], rgb[..., 2]).astype(int)) > 40
    ys, xs = np.nonzero(green)
    if len(xs) < 50:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def cut(z, fam, stem, k):
    """a crop's part inside its frame -> (ink as a bool array, {port: (x, y)} in its pixels) or None"""
    rgb = np.asarray(Image.open(io.BytesIO(z.read(f"{BASE}{fam}/Input Images/{stem}_{k}.jpg"))).convert("RGB"))
    fr = _frame(rgb)
    if fr is None:
        return None
    x0, y0, x1, y1 = fr
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    green = (rgb[..., 1].astype(int) - np.maximum(rgb[..., 0], rgb[..., 2]).astype(int)) > 40
    ink = (g < min(float(np.median(g)) - 60, 170)) & ~green
    m = 4
    ink = ink[y0 + m:y1 - m, x0 + m:x1 - m]
    if ink.sum() < 80:
        return None
    ports = {}
    for line in z.read(f"{BASE}{fam}/XY Coordinates/{stem}_{k}.txt").decode().splitlines():
        p = line.split()
        if len(p) == 3:
            ports[p[0]] = (float(p[1]) - x0 - m, float(p[2]) - y0 - m)
        elif len(p) == 2:
            ports["pin"] = (float(p[0]) - x0 - m, float(p[1]) - y0 - m)
    return ink, ports


def orient(ink, ports, want):
    """the part turned / mirrored so each port lies nearest where `want` puts it ({port: (fx, fy)} as fractions of
    the box) -> (ink, ports, error)"""
    best = None
    for mirror in (False, True):
        for k in range(4):
            a, p = ink, dict(ports)
            h, w = a.shape
            if mirror:
                a = a[:, ::-1]
                p = {n: (w - 1 - x, y) for n, (x, y) in p.items()}
            for _ in range(k):                         # a quarter turn counter-clockwise
                h, w = a.shape
                a = np.rot90(a)
                p = {n: (y, w - 1 - x) for n, (x, y) in p.items()}
            h, w = a.shape
            err = sum(math.hypot(p[n][0] / w - fx, p[n][1] / h - fy) for n, (fx, fy) in want.items() if n in p)
            if best is None or err < best[2]:
                best = (np.ascontiguousarray(a), p, err)
    return best


def _clean(ink, ports, share):
    """the symbol without specks and a label's remnants: the ink pieces smaller than `share` of the largest go,
    the box shrunk to what stays -> (ink, ports)"""
    from scipy import ndimage as ndi
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n < 2:
        return ink, ports
    sz = np.bincount(lab.ravel())[1:]
    keep = np.isin(lab, 1 + np.flatnonzero(sz >= share * sz.max()))
    ys, xs = np.nonzero(keep)
    y0, x0 = ys.min(), xs.min()
    return keep[y0:ys.max() + 1, x0:xs.max() + 1], {k: (x - x0, y - y0) for k, (x, y) in ports.items()}


def _upright(ink, ports):
    """a ground / a supply standing on its pin's line: its ink on both sides of the pin alike (an arrow turned
    sideways, a bar off to one side - drawn so in their circuits - is not how a CMOS gate's supply is drawn)"""
    h, w = ink.shape
    px = ports["pin"][0]
    ys, xs = np.nonzero(ink)
    left, right = int((xs < px - 0.08 * w).sum()), int((xs > px + 0.08 * w).sum())
    return min(left, right) >= 0.4 * max(left, right, 1) and abs(xs.mean() - px) < 0.12 * w


def symbols(zip_path, rnd, n_each=60, train=False):
    """the stamps: NMOS / PMOS standing (gate on the left; NMOS drain up, source down; PMOS source up, drain down),
    grounds (pin up), supply arrows (pin down) - [(ink, ports)] each"""
    z = zipfile.ZipFile(zip_path)
    out = {"nmos": [], "pmos": [], "gnd": [], "vdd": []}
    held = [k for k in range(2000) if ((k // 50) % 10 == 9) != train]   # (the legs model's held-out crops - for a
    #                                                        detector's training, the others)
    for kind, fam, stem, want in (("nmos", "MOSFET-N", "mosfet_n", {"Gate": (0, .5), "Drain": (.5, 0), "Source": (.5, 1)}),
                                  ("pmos", "MOSFET-P", "mosfet_p", {"Gate": (0, .5), "Source": (.5, 0), "Drain": (.5, 1)})):
        for k in rnd.sample(held, len(held)):
            try:
                got = cut(z, fam, stem, k)
            except KeyError:
                continue
            if got is None or not {"Gate", "Drain", "Source"} <= set(got[1]):
                continue
            ink, ports, err = orient(*got, want)
            if err < 0.9:
                out[kind].append(_clean(ink, ports, 0.03))
            if len(out[kind]) >= n_each:
                break
    for kind, fam, stem, want in (("gnd", "GND", "gnd", {"pin": (.5, 0)}), ("vdd", "V-DC (one-port)", "vdc_1p", {"pin": (.5, 1)})):
        pool = range(4000, 7300 if kind == "gnd" else 4500) if train else range(4000)   # (the test's never drawn)
        for k in rnd.sample(pool, min(600, len(pool))):
            try:
                got = cut(z, fam, stem, k)
            except KeyError:
                continue
            if got is None or "pin" not in got[1]:
                continue
            ink, ports, err = orient(*got, want)
            if err < 0.35:
                ink, ports = _clean(ink, ports, 0.15)
                if _upright(ink, ports):
                    out[kind].append((ink, ports))
            if len(out[kind]) >= n_each:
                break
    return out


def letters(cghd, rnd, want=("A", "B", "C"), n_each=40, drafters=(1, 5, 10, 15, 20, 25, 30)):
    """handwritten letters from CGHD's test drafters (their text boxes holding just that letter: hands the label
    reader never learned from) -> {letter: [ink]}"""
    import glob
    import xml.etree.ElementTree as ET
    from PIL import ImageOps
    from recog.gates import ink as ink_of
    out = {c: [] for c in want}
    xmls = [x for x in glob.glob(os.path.join(cghd, "drafter_*", "annotations", "*.xml"))
            if int(re.search(r"drafter_(-?\d+)", x).group(1)) in drafters]
    rnd.shuffle(xmls)
    for x in xmls:
        if all(len(v) >= n_each for v in out.values()):
            break
        objs = [(o.findtext("text") or "").strip() for o in ET.parse(x).getroot().findall("object")]
        if not any(t in want for t in objs):
            continue
        imgs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(x)), "images", os.path.basename(x)[:-4] + ".*"))
        if not imgs:
            continue
        img = None
        for o in ET.parse(x).getroot().findall("object"):
            t = (o.findtext("text") or "").strip()
            if o.findtext("name") != "text" or t not in want or len(out[t]) >= n_each:
                continue
            if img is None:
                img = ImageOps.exif_transpose(Image.open(imgs[0])).convert("L")
            b = o.find("bndbox")
            x0, y0, x1, y1 = (int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
            m = ink_of(img.crop((x0 - 4, y0 - 4, x1 + 4, y1 + 4)))
            if m.sum() < 30:
                continue
            # the photo's other strokes - a wire passing by or through the letter, a part's edge - run out of the
            # box: what reaches the crop's border goes; a letter that loses much with it was crossed (not kept)
            from scipy import ndimage as ndi
            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
            kept = m & ~np.isin(lab, list(edge))
            if kept.sum() < 0.8 * m.sum():
                continue
            ys, xs = np.nonzero(kept)
            kept = kept[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if kept.shape[1] > 1.5 * kept.shape[0] or kept.mean() > 0.45 or len(xs) < 30:
                continue                               # (two letters, or a blot)
            lab, n = ndi.label(kept, structure=np.ones((3, 3)))
            hh = kept.shape[0]
            if any(sl[0].stop <= 0.3 * hh and (sl[0].stop - sl[0].start) < 0.2 * hh
                   and (sl[1].stop - sl[1].start) > 0.4 * kept.shape[1] for sl in ndi.find_objects(lab)):
                continue                               # (a bar over it: 'not A' - another input)
            if marked_glyph(kept):
                continue                               # (a bar or a "'" by the circuit reader's own test: 8.11, five
                #                                        of the 70 test drawings' 'A's had the drafter's bar over them)
            out[t].append(kept)
    return out


def marked_glyph(kept) -> bool:
    """a letter's ink with a complement's mark beside it (recog.circuit_reader.complement_mark)"""
    from recog.circuit_reader import complement_mark
    h, w = kept.shape
    p = max(h, w)
    canvas = np.full((h + 2 * p, w + 2 * p), 255, np.uint8)
    canvas[p:p + h, p:p + w][kept] = 0
    return complement_mark(Image.fromarray(canvas).convert("RGB"), (p, p, p + w, p + h)) is not None


SUPPLY_WORDS, GROUND_WORDS = ("VDD", "VCC", "+5V", "V+", "+12V", "+9V"), ("GND", "VSS", "0V")


def words(cghd, rnd, want=SUPPLY_WORDS + GROUND_WORDS, n_each=30, drafters=(1, 5, 10, 15, 20, 25, 30)):
    """rails' names as CGHD's test drafters wrote them ('VDD', 'GND', '+5V' ...: their text boxes) -> {word: [ink]}"""
    import glob
    import xml.etree.ElementTree as ET
    from PIL import ImageOps
    from scipy import ndimage as ndi
    from recog.gates import ink as ink_of
    out = {w: [] for w in want}
    xmls = [x for x in glob.glob(os.path.join(cghd, "drafter_*", "annotations", "*.xml"))
            if int(re.search(r"drafter_(-?\d+)", x).group(1)) in drafters]
    rnd.shuffle(xmls)
    for x in xmls:
        objs = [o for o in ET.parse(x).getroot().findall("object")
                if o.findtext("name") == "text" and (o.findtext("text") or "").strip() in want]
        if not objs:
            continue
        imgs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(x)), "images", os.path.basename(x)[:-4] + ".*"))
        if not imgs:
            continue
        img = ImageOps.exif_transpose(Image.open(imgs[0])).convert("L")
        for o in objs:
            t = o.findtext("text").strip()
            if len(out[t]) >= n_each:
                continue
            b = o.find("bndbox")
            x0, y0, x1, y1 = (int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
            if y1 - y0 > x1 - x0:                      # (written turned: not as a rail's name is)
                continue
            m = ink_of(img.crop((x0 - 4, y0 - 4, x1 + 4, y1 + 4)))
            if m.sum() < 40:
                continue
            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
            kept = m & ~np.isin(lab, list(edge))
            if kept.sum() < 0.8 * m.sum():
                continue
            ys, xs = np.nonzero(kept)
            kept = kept[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if kept.mean() > 0.45:
                continue
            out[t].append(kept)
    return {w: v for w, v in out.items() if v}


# -- the drawing -------------------------------------------------------------------------------------------------------
class Page:
    def __init__(self, w, h, rnd):
        self.rnd = rnd
        self.w, self.h = w, h
        self.ink = Image.new("L", (w, h), 0)           # how much ink at each pixel (0..255)
        self.d = ImageDraw.Draw(self.ink)
        self.pen = rnd.uniform(2.5, 4.5)
        self.boxes = []                                # (what, box) of each symbol stamped - the drawing's truth boxes

    def line(self, pts):
        """a wavering pen stroke through the points"""
        rnd = self.rnd
        path = []
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 25))
            for t in range(n):
                f = t / n
                path.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
        path.append(pts[-1])
        wob = rnd.uniform(0.5, 2.0)
        drift = [0.0, 0.0]
        out = []
        for i, (x, y) in enumerate(path):
            if 0 < i < len(path) - 1:
                drift = [0.7 * drift[0] + rnd.gauss(0, wob), 0.7 * drift[1] + rnd.gauss(0, wob)]
                out.append((x + drift[0], y + drift[1]))
            else:
                out.append((x, y))
        self.d.line(out, fill=255, width=int(round(self.pen)), joint="curve")

    def dot(self, x, y):
        r = self.pen * self.rnd.uniform(1.2, 1.8)
        self.d.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def hop(self, x, y, horizontal_wire_y=None):
        """a wire crossing a vertical one at (x, y) with a little arc over it"""
        r = self.rnd.uniform(9, 14)
        self.d.arc((x - r, y - r, x + r, y + r), 180, 360, fill=255, width=int(round(self.pen)))

    def stamp(self, ink, ports, cx, cy, size, what=None, thicken=True):
        """a symbol scaled so its longer side is `size`, centred at (cx, cy) -> its ports in page pixels, its box"""
        from scipy import ndimage as ndi
        h, w = ink.shape
        s = size / max(h, w)
        small = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize((max(1, int(w * s)), max(1, int(h * s))),
                                                                                  Image.BILINEAR)) > 60
        if thicken and self.pen > 4.0:                 # (its strokes as thick as the pen's wires)
            small = ndi.binary_dilation(small)
        img = Image.fromarray(small.astype(np.uint8) * 255)
        x0, y0 = int(cx - img.width / 2), int(cy - img.height / 2)
        self.ink.paste(Image.fromarray(np.maximum(np.asarray(self.ink.crop((x0, y0, x0 + img.width, y0 + img.height))),
                                                  np.asarray(img))), (x0, y0))
        box = (x0, y0, x0 + img.width, y0 + img.height)
        if what:
            self.boxes.append((what, box))
        return {n: (x0 + x * s, y0 + y * s) for n, (x, y) in ports.items()}, box

    def crowded(self, box, pad=6):
        """how much ink already lies in the box (grown by pad): a fraction of its area"""
        x0, y0, x1, y1 = (int(v) for v in (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
        a = np.asarray(self.ink.crop((x0, y0, x1, y1))) > 0
        return float(a.mean()) if a.size else 0.0

    def render(self):
        rnd = self.rnd
        paper = np.full((self.h, self.w), rnd.uniform(215, 248), np.float32)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        paper += rnd.uniform(-15, 15) * (xx / self.w) + rnd.uniform(-15, 15) * (yy / self.h)   # uneven light
        paper += np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 3, paper.shape)
        a = np.asarray(self.ink.filter(ImageFilter.GaussianBlur(rnd.uniform(0.4, 1.0))), np.float32) / 255.0
        color = np.array(rnd.choice([(30, 30, 35), (25, 35, 110), (40, 40, 90), (60, 50, 50)]), np.float32)
        rgb = paper[..., None] * (1 - a[..., None]) + color * a[..., None]
        img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        if rnd.random() < 0.7:
            ang = rnd.uniform(-3, 3)
            img = img.rotate(ang, resample=Image.BILINEAR, fillcolor=tuple(int(v) for v in paper[0, 0].repeat(3)))
            t, (cx, cy) = math.radians(ang), (self.w / 2, self.h / 2)

            def turned(b):                             # (the boxes turned with the page: PIL turns counter-clockwise)
                pts = [(cx + (x - cx) * math.cos(t) + (y - cy) * math.sin(t), cy - (x - cx) * math.sin(t) + (y - cy) * math.cos(t))
                       for x in (b[0], b[2]) for y in (b[1], b[3])]
                return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
            self.boxes = [(what, turned(b)) for what, b in self.boxes]
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rnd.randint(70, 92))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def size_of(e):
    """a network's place in cells: (width, height)"""
    if isinstance(e, str):
        return 1, 1
    kids = [size_of(x) for x in e[1:]]
    if e[0] == "s":
        return max(k[0] for k in kids), sum(k[1] for k in kids)
    return sum(k[0] for k in kids), max(k[1] for k in kids)


class Builder:
    def __init__(self, page, syms, rnd):
        self.page, self.syms, self.rnd = page, syms, rnd
        self.comps, self.gate_pins = [], []            # gate_pins: (input, x, y) - where each input wire must reach
        self.nodes = 0
        self.y_out = None                              # the output's line

    def node(self):
        self.nodes += 1
        return f"N{self.nodes}"

    def place(self, e, kind, x0, y0, top, bottom, top_pt, bot_pt):
        """the network e in the cell box from (x0, y0), between nodes `top` / `bottom` at points top_pt / bot_pt
        (where the network's own top / bottom wires begin)"""
        p, rnd = self.page, self.rnd
        w, h = size_of(e)
        if isinstance(e, str):
            cx, cy = x0 + OFF, y0 + CELL / 2
            ink, ports = rnd.choice(self.syms[kind])
            pts, _ = p.stamp(ink, ports, cx + rnd.uniform(-6, 6), cy + rnd.uniform(-6, 6), CELL * rnd.uniform(0.62, 0.8),
                             what=f"fet Q{len(self.comps) + 1}")
            up, down = ("Drain", "Source") if kind == "nmos" else ("Source", "Drain")
            p.line([top_pt, (top_pt[0], pts[up][1] - 4), pts[up]] if abs(top_pt[0] - pts[up][0]) > 3 else [top_pt, pts[up]])
            p.line([pts[down], (bot_pt[0], pts[down][1] + 4), bot_pt] if abs(bot_pt[0] - pts[down][0]) > 3 else [pts[down], bot_pt])
            n = f"Q{len(self.comps) + 1}"
            roles = {up: top, down: bottom}
            self.comps.append({"id": n, "type": kind, "roles": roles, "gate_input": e})
            self.gate_pins.append((e, pts["Gate"], n))
            return
        kids = e[1:]
        if e[0] == "s":                                 # one above the other, a node between each two
            yy, prev, prev_pt = y0, top, top_pt
            for i, k in enumerate(kids):
                kw, kh = size_of(k)
                nxt = bottom if i == len(kids) - 1 else self.node()
                nxt_pt = bot_pt if i == len(kids) - 1 else (top_pt[0], yy + kh * CELL)
                self.place(k, kind, x0, yy, prev, nxt, prev_pt, nxt_pt if i < len(kids) - 1 else bot_pt)
                prev, prev_pt, yy = nxt, nxt_pt, yy + kh * CELL
            return
        xx = x0                                         # side by side, joined by a wire above and one below
        # (a bus on the output's node is drawn on the output's line, and its branches meet it with dots: the
        # pull-up network's branches from above and the pull-down's from below at one place would be a crossing
        # without them)
        y_t = top_pt[1] if abs(top_pt[1] - self.y_out) < 1 else y0 + 8
        y_b = bot_pt[1] if abs(bot_pt[1] - self.y_out) < 1 else y0 + h * CELL - 8
        tops, bots = [], []
        for k in kids:
            kw, kh = size_of(k)
            tops.append((xx + OFF, y_t))
            bots.append((xx + OFF, y_b))
            self.place(k, kind, xx, y0, top, bottom, tops[-1], bots[-1])
            xx += kw * CELLW
        p.line([(min(t[0] for t in tops + [top_pt]), y_t), (max(t[0] for t in tops + [top_pt]), y_t)])
        p.line([(min(b[0] for b in bots + [bot_pt]), y_b), (max(b[0] for b in bots + [bot_pt]), y_b)])
        if abs(top_pt[1] - y_t) > 1:
            p.line([top_pt, (top_pt[0], y_t)])
        if abs(bot_pt[1] - y_b) > 1:
            p.line([(bot_pt[0], y_b), bot_pt])
        for y_bus, pts in ((y_t, tops), (y_b, bots)):
            if abs(y_bus - self.y_out) < 1:
                for t in pts[1:]:
                    p.dot(*t)
        if self.rnd.random() < 0.6:
            for t in tops[1:-1] + [tops[0]]:
                p.dot(*t)


def draw(name, gate, syms, rnd, p_words=0.0):
    pdn = GATES[gate]
    pun = dual(pdn)
    ins = sorted(set(inputs_of(pdn)))
    pw, ph = size_of(pun)
    nw, nh = size_of(pdn)
    width = max(pw, nw)
    need_w, need_h = int(width * CELLW + 300), int(130 + (ph + nh) * CELL + 260)
    W, H = max(1450, need_w + 200), max(1100, need_h + 150)   # a page, the drawing somewhere on it
    left = 40 + rnd.uniform(0, W - need_w - 80)
    top = rnd.uniform(0, H - need_h - 40)
    page = Page(W, H, rnd)
    b = Builder(page, syms, rnd)
    xc = left + OFF                                     # the networks' spine
    # the supply on top, the PMOS network, the output, the NMOS network, the ground
    y_top = top + 130
    sup_style = rnd.choice(["bar", "end"]) if p_words and rnd.random() < p_words else "symbol"
    if sup_style == "symbol":
        vink, vports = rnd.choice(syms["vdd"])
        vpts, _ = page.stamp(vink, vports, xc, top + 60, rnd.uniform(70, 95), what="voltage_rail")
        page.line([vpts["pin"], (xc, y_top)])
    else:                                               # the supply named: 'VDD' over a bar, or at the wire's end
        _rail_word(page, rnd, syms, SUPPLY_WORDS, xc, top + 85, y_top, sup_style, above=True)
    y_out = y_top + ph * CELL
    b.y_out = y_out
    b.place(pun, "pmos", left, y_top, "VDD", "Y", (xc, y_top), (xc, y_out))
    y_bot = y_out + nh * CELL
    b.place(pdn, "nmos", left, y_out, "Y", "0", (xc, y_out), (xc, y_bot))
    gnd_style = rnd.choice(["bar", "end"]) if p_words and rnd.random() < p_words else "symbol"
    if gnd_style == "symbol":
        gink, gports = rnd.choice(syms["gnd"])
        gpts, gbox = page.stamp(gink, gports, xc, y_bot + 70, rnd.uniform(70, 95), what="gnd")
        page.line([(xc, y_bot), gpts["pin"]])
    else:
        _rail_word(page, rnd, syms, GROUND_WORDS, xc, y_bot + 55, y_bot, gnd_style, above=False)
    page.line([(xc, y_out), (left + width * CELLW + 220, y_out)])   # the output, to the right
    if rnd.random() < 0.6:
        page.dot(xc, y_out)
    # each gate: a short wire to the left, its input's letter at its end
    for inp, pt, _ in b.gate_pins:
        ink = rnd.choice(syms["letters"][inp])
        hgt = rnd.uniform(30, 46)
        wid = hgt * ink.shape[1] / ink.shape[0]
        tries = [(rnd.uniform(40, 70), rnd.uniform(-6, 6))] + [(st, dy) for st in (45, 60, 80, 100) for dy in (0, 14, -14, 26, -26)]
        for stub, dy in tries:                         # (as a student writes it: clear of the wires already drawn)
            cx, cy = pt[0] - stub - 12 - wid / 2, pt[1] + dy
            if page.crowded((cx - wid / 2, cy - hgt / 2, cx + wid / 2, cy + hgt / 2)) < 0.004 and \
                    page.crowded((pt[0] - stub, pt[1] - 3, pt[0] - 8, pt[1] + 3), pad=2) < 0.02:
                break
        page.line([pt, (pt[0] - stub, pt[1])])
        page.stamp(ink, {}, cx, cy, hgt * max(1.0, ink.shape[1] / ink.shape[0]), what=f"text {inp}", thicken=False)
    img = page.render()
    comps = []
    for c in b.comps:
        r = {"Gate": c["gate_input"], **c["roles"]}
        comps.append({"id": c["id"], "type": c["type"], "nodes": [r["Gate"], r["Drain"], r["Source"]],
                      "roles": {k: r[k] for k in ("Gate", "Drain", "Source")}})
    comps.append({"id": "VDD", "type": "voltage_source", "nodes": ["VDD", "0"], "plus": "VDD"})
    truth = {"note": f"a static CMOS {gate} drawn from Digitize-HCD's hand-drawn symbols by tools/cmos_synth.py",
             "gate": gate, "components": comps, "ground": "0", "node_labels": {i: i for i in ins},
             "page": [W, H], "boxes": [(what, [round(v, 1) for v in bx]) for what, bx in page.boxes]}
    return img, truth


def _rail_word(page, rnd, syms, choices, xc, y_end, y_from, style, above):
    """a rail drawn as its name: the wire from y_from to y_end, then a short bar across it ('bar') or nothing ('end'),
    the name beyond it ('VDD' above, 'GND' below)"""
    have = [w for w in choices if w in syms["words"]]
    word = rnd.choice(have)
    ink = rnd.choice(syms["words"][word])
    page.line([(xc, y_from), (xc, y_end)])
    if style == "bar":
        half = rnd.uniform(22, 40)
        page.line([(xc - half, y_end), (xc + half, y_end)])
    hgt = rnd.uniform(30, 44)
    wid = hgt * ink.shape[1] / ink.shape[0]
    cy = y_end - 10 - hgt / 2 if above else y_end + 10 + hgt / 2
    page.stamp(ink, {}, xc + rnd.uniform(-8, 8), cy, max(hgt, wid), what=f"word {word}", thicken=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(HERE, "data", "circuits_raw", "dhcd", "dhcd.zip"))
    ap.add_argument("--cghd", default=os.path.join(HERE, "data", "circuits_raw", "cghd"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--words", type=float, default=0.0, help="the share of supplies / grounds drawn as their names "
                    "('VDD' over a bar or at the wire's end, 'GND' under) instead of a symbol")
    ap.add_argument("--train", action="store_true", help="drawings for training the detector: symbols the test's are "
                    "never drawn from, letters of CGHD's training drafters; --coco writes their boxes as COCO")
    ap.add_argument("--coco", default="", help="with --train: <out>/<this>.json, the boxes in the detector's classes")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    syms = symbols(a.zip, rnd, n_each=200 if a.train else 60, train=a.train)
    hands = tuple(d for d in range(-1, 40) if d not in (1, 5, 10, 15, 20, 25, 30, 3, 13, 23)) if a.train else \
        (1, 5, 10, 15, 20, 25, 30)
    syms["letters"] = letters(a.cghd, rnd, n_each=150 if a.train else 40, drafters=hands)
    if a.words:
        syms["words"] = words(a.cghd, rnd, n_each=100 if a.train else 30, drafters=hands)
        print({k: len(v) for k, v in syms["words"].items()}, flush=True)
    print({k: len(v) for k, v in syms.items() if k != "letters"}, {k: len(v) for k, v in syms["letters"].items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    names = list(GATES)
    coco = None
    if a.coco:
        classes = json.load(open(os.path.join(HERE, "data", "circuits", "classes.json"), encoding="utf-8"))["classes"]
        cat = {c: i + 1 for i, c in enumerate(classes)}
        coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]}
    for k in range(a.n):
        gate = names[k % len(names)]
        name = f"cmos_{k:03d}_{gate}"
        img, truth = draw(name, gate, syms, random.Random(a.seed * 100003 + k), a.words)
        if coco is None:
            img.save(os.path.join(a.out, name + ".png"))
            json.dump(truth, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
            continue
        sc = 1600 / max(img.size)                      # (as the detector's images: the long side 1600, JPEG)
        img = img.resize((round(img.width * sc), round(img.height * sc)), Image.BILINEAR)
        fn = f"{name}_{a.seed}.jpg"
        img.save(os.path.join(a.out, fn), quality=92)
        iid = len(coco["images"]) + 1
        coco["images"].append({"id": iid, "file_name": fn, "width": img.width, "height": img.height, "source": "cmos_synth"})
        for what, b in truth["boxes"]:
            kind = what.split()[0]
            cls = {"fet": "fet", "voltage_rail": "voltage_rail", "gnd": "gnd", "text": "text", "word": "text"}[kind]
            x0, y0, x1, y1 = (v * sc for v in b)
            an = {"id": len(coco["annotations"]) + 1, "image_id": iid, "category_id": cat[cls],
                  "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                  "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0}
            if cls == "text":
                an["text"] = what.split(maxsplit=1)[1]
            coco["annotations"].append(an)
    if coco is not None:
        json.dump(coco, open(os.path.join(a.out, a.coco + ".json"), "w", encoding="utf-8"))
    print(f"{a.n} drawings -> {a.out}")


if __name__ == "__main__":
    main()
