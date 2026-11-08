"""Pass-transistor networks drawn by hand, with their netlists known - the circuit reader measured on the kind of
drawing the course's Moed C 2.1 asks for (S = A + B + C with the fewest transistors: four NMOS pass transistors, two
stages), and on its relatives: a 2:1 multiplexer, XOR / XNOR, OR, AND passing a ground.

Laid out as a student draws them: each transistor lying on its side, its channel along a wire, its gate a short stub
up (or down) with the input's letter at its end; an input's letter at the left end of its wire; the branches of a node
joined by a wire across their ends; the output's letter at the right. A complement written as students write it -
"B'" (a stroke at the letter's top right) or B with a bar over it (the same drawing uses one way). The symbols are
tools/cmos_synth.py's: Digitize-HCD's hand-drawn MOSFETs its legs model never learned from, CGHD's test drafters'
letters, the same pen and paper.

    python tools/ptl_synth.py --out ~/eg-data/ptl_synth --n 60 [--seed 0]
        -> <out>/<name>.png and <name>.json (each transistor's Gate / Drain / Source nodes, the letters at the nodes;
           tools/circuit_eval.py --synth <out> --inputs ABC measures the reader on them: the network exactly, the
           Boolean function it gives)
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from tools.cmos_synth import CELL, Page, letters, symbols  # noqa: E402

# each network: its output's letter and its branches - (the gate's letter, what is passed: an input's letter, "0" for
# ground, or a network of its own)
NETS = {
    "moedc": ("S", [("C'", ("X", [("B'", "A"), ("B", "B")])), ("C", "C")]),       # S = A + B + C (the solution's)
    "moedc_b": ("S", [("C", "C"), ("C'", ("X", [("B", "B"), ("B'", "A")]))]),
    "moedc_a": ("S", [("C'", ("X", [("A'", "B"), ("A", "A")])), ("C", "C")]),
    "mux2": ("Y", [("S'", "A"), ("S", "B")]),
    "xor2": ("Y", [("A'", "B"), ("A", "B'")]),
    "xnor2": ("Y", [("A'", "B'"), ("A", "B")]),
    "or2": ("Y", [("A'", "B"), ("A", "A")]),
    "and2": ("Y", [("B", "A"), ("B'", "0")]),
}
ROW = 1.2 * CELL                   # a branch's height
COLW = 2.1 * CELL                  # a stage's width: its transistors, the wire to the next stage


def rows(net):
    return sum(rows(src) if isinstance(src, tuple) else 1 for _, src in net[1])


def depth(net):
    return 1 + max([depth(src) for _, src in net[1] if isinstance(src, tuple)] or [0])


def marked(ink, style, rnd):
    """a letter with its complement's mark: "'" at its top right, or a bar over it -> (ink, the letter's own height)"""
    h, w = ink.shape
    t = max(2, int(round(rnd.uniform(0.08, 0.12) * h)))
    if style == "prime":
        W, H, ox, oy = int(w + 0.4 * h), h, 0, 0
        seg = [(w + rnd.uniform(0.18, 0.3) * h, rnd.uniform(-0.02, 0.06) * h),
               (w + rnd.uniform(0.06, 0.14) * h, rnd.uniform(0.28, 0.4) * h)]
    else:
        W, H, ox, oy = int(w * 1.2) + 2, int(h * 1.35), int(0.1 * w), int(0.35 * h)
        y = rnd.uniform(0.06, 0.12) * h
        seg = [(rnd.uniform(-0.02, 0.06) * w + ox * 0.5, y), (w * rnd.uniform(1.0, 1.12) + ox * 0.5, y + rnd.uniform(-0.04, 0.04) * h)]
    img = Image.new("L", (W, H), 0)
    img.paste(Image.fromarray(ink.astype(np.uint8) * 255), (ox, oy))
    ImageDraw.Draw(img).line(seg, fill=255, width=t)
    a = np.asarray(img) > 127
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1], h


def hat_symbol(rnd, kind="nmos"):
    """a MOSFET drawn as the course draws one, standing alone: the channel a raised step (a hat) in a wire, the gate a
    bar over it with its stub up - a P-channel one with a bubble on its gate - by a wavering pen -> (ink: bool array,
    {Gate, Drain, Source: (x, y)}); the drain on the left or the right at random (the symbol does not tell them)"""
    L = rnd.uniform(70, 130)
    cw, ch = L * rnd.uniform(0.4, 0.55), L * rnd.uniform(0.18, 0.3)
    gap, pw = L * rnd.uniform(0.08, 0.14), cw * rnd.uniform(1.0, 1.3)
    stub, t = L * rnd.uniform(0.1, 0.25), max(2, int(round(rnd.uniform(2.0, 4.5))))
    r = L * rnd.uniform(0.05, 0.09) if kind == "pmos" else 0.0
    m = 8 + t
    cx = m + L / 2
    yp = m + stub + 2 * r + (t if r else 0)
    yt, yb = yp + gap, yp + gap + ch
    xa, xb = cx - cw / 2, cx + cw / 2
    W, H = int(L + 2 * m), int(yb + m)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)

    wob = rnd.uniform(0.3, 1.0)

    def j(pts):                                      # (a hand's waver along each stroke)
        out, dx, dy = [], 0.0, 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / 8))
            for k in range(n):
                dx, dy = 0.6 * dx + rnd.gauss(0, wob), 0.6 * dy + rnd.gauss(0, wob)
                out.append((x0 + (x1 - x0) * k / n + dx, y0 + (y1 - y0) * k / n + dy))
        return out + [pts[-1]]
    d.line(j([(m, yb), (xa, yb), (xa, yt), (xb, yt), (xb, yb), (m + L, yb)]), fill=255, width=t, joint="curve")
    d.line(j([(cx - pw / 2, yp), (cx + pw / 2, yp)]), fill=255, width=t)
    if r:
        cyb = yp - t / 2 - r
        d.ellipse((cx - r, cyb - r, cx + r, cyb + r), outline=255, width=max(2, t - 1))
        d.line(j([(cx, m), (cx, cyb - r)]), fill=255, width=t)
    else:
        d.line(j([(cx, m), (cx, yp)]), fill=255, width=t)
    ends = [(m, yb), (m + L, yb)]
    rnd.shuffle(ends)
    return np.asarray(img) > 127, {"Gate": (cx, m), "Drain": ends[0], "Source": ends[1]}


class Drawing:
    def __init__(self, page, syms, rnd, mark, gate_up, hat=False):
        self.page, self.syms, self.rnd = page, syms, rnd
        self.mark, self.gate_up, self.hat = mark, gate_up, hat
        self.comps, self.n = [], 0
        self.labels = {}                                # node -> the letter written at it
        self.hgt = rnd.uniform(30, 44)                  # the student's letters' height

    def node(self):
        self.n += 1
        return f"N{self.n}"

    def letter(self, text, cx, cy, anchor="c"):
        """a letter written centred at (cx, cy) - 'r': its right edge at cx, 'b': its bottom at cy, 't': its top"""
        rnd = self.rnd
        base = text.rstrip("'")
        ink = rnd.choice(self.syms["letters"][base])
        body = ink.shape[0]
        if text.endswith("'"):
            ink, body = marked(ink, self.mark, rnd)
        k = self.hgt / body
        wid, hh = ink.shape[1] * k, ink.shape[0] * k
        if anchor == "r":
            cx -= wid / 2
        elif anchor == "b":
            cy -= hh / 2
        elif anchor == "t":
            cy += hh / 2
        self.page.stamp(ink, {}, cx, cy, max(wid, hh), what=f"text {text}", thicken=False)

    def hat_fet(self, cx, cy):
        """an NMOS as the course's solution draws one: the channel a raised step in the wire (a hat), the gate a bar
        over it - drawn by the pen -> its ports: Gate, Left, Right"""
        rnd, p = self.rnd, self.page
        cx, cy = cx + rnd.uniform(-5, 5), cy + rnd.uniform(-5, 5)
        L = CELL * rnd.uniform(0.75, 1.0)
        cw, ch = L * rnd.uniform(0.4, 0.55), L * rnd.uniform(0.18, 0.3)
        gap, pw = L * rnd.uniform(0.08, 0.14), cw * rnd.uniform(1.0, 1.3)
        sg = -1 if self.gate_up else 1                   # (towards the gate)
        yb = cy - sg * ch / 2
        yt, yp = yb + sg * ch, yb + sg * (ch + gap)
        x0, x1, xa, xb = cx - L / 2, cx + L / 2, cx - cw / 2, cx + cw / 2
        p.line([(x0, yb), (xa, yb), (xa, yt), (xb, yt), (xb, yb), (x1, yb)])
        p.line([(cx - pw / 2, yp), (cx + pw / 2, yp)])
        pad = 0.06 * L
        p.boxes.append((f"fet M{len(self.comps) + 1}", (x0, min(yb, yp) - pad, x1, max(yb, yp) + pad)))
        return (cx, yp), (x0, yb), (x1, yb)

    def fet(self, cx, cy):
        """an NMOS lying on its side, its gate up (or down) -> its ports: Gate, Left, Right"""
        rnd = self.rnd
        if self.hat:
            return self.hat_fet(cx, cy)
        ink, ports = rnd.choice(self.syms["nmos"])       # (standing: gate on the left, drain up, source down)
        h, w = ink.shape
        if self.gate_up:                                 # a quarter turn clockwise: the gate up, the drain right
            ink2, f = np.rot90(ink, -1), lambda x, y: (h - 1 - y, x)
        else:                                            # counter-clockwise: the gate down, the drain left
            ink2, f = np.rot90(ink, 1), lambda x, y: (y, w - 1 - x)
        p2 = {k: f(*v) for k, v in ports.items()}
        pts, _ = self.page.stamp(np.ascontiguousarray(ink2), p2, cx + rnd.uniform(-5, 5), cy + rnd.uniform(-5, 5),
                                 CELL * rnd.uniform(0.62, 0.8), what=f"fet M{len(self.comps) + 1}")
        lr = sorted(("Drain", "Source"), key=lambda k: pts[k][0])
        return pts["Gate"], pts[lr[0]], pts[lr[1]]

    def place(self, net, x_bus, y0):
        """the network with its branches' ends joined at x_bus, from y0 down -> (its node, the y its output leaves at)"""
        p, rnd = self.page, self.rnd
        name, branches = net
        out = name if name in ("X",) else self.node()
        y, ys = y0, []
        for gate, src in branches:
            if isinstance(src, tuple):
                xs_bus = x_bus - COLW
                src_node, yt = self.place(src, xs_bus, y)
                y += rows(src) * ROW
            else:
                src_node, yt = None, y + ROW / 2
                y += ROW
            g, left, right = self.fet(x_bus - rnd.uniform(0.42, 0.55) * COLW, yt)
            n = f"M{len(self.comps) + 1}"
            p.line([right, (x_bus, right[1]), (x_bus, yt)] if abs(right[1] - yt) > 3 else [right, (x_bus, yt)])
            if src_node is not None:                     # the stage before: its node's wire to this channel
                p.line([(xs_bus, yt), (left[0], yt), left] if abs(left[1] - yt) > 3 else [(xs_bus, yt), left])
                if rnd.random() < 0.5:
                    p.dot(xs_bus, yt)
            elif src == "0":                             # ground passed: the wire left, a ground under its end
                xe = left[0] - rnd.uniform(50, 80)
                gink, gports = rnd.choice(self.syms["gnd"])
                gy = yt + rnd.uniform(45, 60)
                gpts, _ = p.stamp(gink, gports, xe, gy, rnd.uniform(60, 85), what="gnd")
                p.line([left, (xe, left[1]), gpts["pin"]])
                src_node = "0"
            else:                                        # an input: its wire left, its letter at the end
                xe = left[0] - rnd.uniform(55, 110)
                p.line([left, (xe, left[1])])
                self.letter(src, xe - rnd.uniform(8, 16), left[1] + rnd.uniform(-5, 5), anchor="r")
                src_node = src
                self.labels[src] = src
            stub = rnd.uniform(35, 60)
            ge = (g[0], g[1] - stub) if self.gate_up else (g[0], g[1] + stub)
            p.line([g, ge])
            self.letter(gate, ge[0] + rnd.uniform(-6, 6), ge[1] - rnd.uniform(6, 12) if self.gate_up else ge[1] + rnd.uniform(6, 12),
                        anchor="b" if self.gate_up else "t")
            self.labels[gate] = gate
            self.comps.append({"id": n, "type": "nmos", "roles": {"Gate": gate, "Drain": src_node, "Source": out}})
            ys.append(yt)
        p.line([(x_bus, ys[0]), (x_bus, ys[-1])])
        yc = (ys[0] + ys[-1]) / 2
        if name == "X" and rnd.random() < 0.5:           # the inner node named, as the solution names it
            self.letter("X", x_bus + rnd.uniform(10, 20), ys[0] - rnd.uniform(10, 20), anchor="b")
            self.labels[out] = "X"
        return out, yc


def draw(kind, syms, rnd, p_hat=0.0):
    net = NETS[kind]
    W, H = 1450, 1100
    need_w, need_h = depth(net) * COLW + 260, rows(net) * ROW + 120
    page = Page(W, H, rnd)
    d = Drawing(page, syms, rnd, rnd.choice(["prime", "bar"]), rnd.random() < 0.75, hat=p_hat > 0 and rnd.random() < p_hat)
    x_bus = rnd.uniform(need_w - 120, W - 260)
    y0 = rnd.uniform(80, max(90, H - need_h))
    out, yc = d.place(net, x_bus, y0)
    xe = x_bus + rnd.uniform(110, 200)
    page.line([(x_bus, yc), (xe, yc)])
    if rnd.random() < 0.5:
        page.dot(x_bus, yc)
    if rnd.random() < 0.75:                              # the output named, as the question names it
        d.letter(net[0], xe + rnd.uniform(10, 18), yc + rnd.uniform(-5, 5), anchor="c")
        d.labels[out] = net[0]
    img = page.render()
    comps = []
    for c in d.comps:
        r = c["roles"]
        comps.append({"id": c["id"], "type": "nmos", "nodes": [r["Gate"], r["Drain"], r["Source"]], "roles": r})
    truth = {"note": f"a pass-transistor network ({kind}) drawn from Digitize-HCD's hand-drawn symbols by tools/ptl_synth.py",
             "gate": kind, "components": comps, "ground": "0" if any("0" in c["nodes"] for c in comps) else None,
             "node_labels": d.labels, "mark": d.mark, "hat": d.hat, "page": [W, H],
             "boxes": [(what, [round(v, 1) for v in bx]) for what, bx in page.boxes]}
    return img, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(HERE, "data", "circuits_raw", "dhcd", "dhcd.zip"))
    ap.add_argument("--cghd", default=os.path.join(HERE, "data", "circuits_raw", "cghd"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default="", help="only these networks, comma-separated ('moedc,moedc_b,moedc_a': the "
                    "exam question's)")
    ap.add_argument("--hat", type=float, default=0.0, help="the share of drawings whose transistors are drawn as the "
                    "course's solution draws them (a raised step, a bar over it) instead of Digitize-HCD's symbols")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    syms = symbols(a.zip, rnd, n_each=60)
    syms["letters"] = letters(a.cghd, rnd, want=("A", "B", "C", "S", "X", "Y"), n_each=40)
    print({k: len(v) for k, v in syms.items() if k != "letters"}, {k: len(v) for k, v in syms["letters"].items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    kinds = [k for k in a.kinds.split(",") if k] or list(NETS)
    for k in range(a.n):
        kind = kinds[k % len(kinds)]
        name = f"ptl_{k:03d}_{kind}"
        img, truth = draw(kind, syms, random.Random(a.seed * 100003 + k), a.hat)
        img.save(os.path.join(a.out, name + ".png"))
        json.dump(truth, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    print(f"{a.n} drawings -> {a.out}")


if __name__ == "__main__":
    main()
