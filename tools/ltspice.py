"""LTspice schematics (.asc) -> grader.circuit.Netlist, for measuring netlists read from drawings against the
circuits they were drawn from (CGHD drafter 1's twelve circuits come with their .asc files).

    from ltspice import asc_netlist
    net = asc_netlist("C4.asc")

Wires join at their ends and where an end lies on another wire; a symbol's pins join the wires there. Pin
positions of the common symbols (R0 orientation, LTspice's own symbol library); rotations R0/90/180/270 and
mirrors M0/90/180/270.
"""
import re

PINS = {   # symbol: [(x, y, pin name)]; the first pin is grader.circuit's 'plus': a voltage source's '+', the head of a
           # current source's arrow (LTspice's points from its top pin to its bottom one), a diode's cathode
    "res": [(16, 16, "a"), (16, 96, "b")], "cap": [(16, 0, "a"), (16, 64, "b")], "polcap": [(16, 0, "+"), (16, 64, "-")],
    "ind": [(16, 16, "a"), (16, 96, "b")], "voltage": [(0, 16, "+"), (0, 96, "-")], "current": [(0, 80, "to"), (0, 0, "from")],
    "diode": [(16, 64, "k"), (16, 0, "a")], "LED": [(16, 64, "k"), (16, 0, "a")], "zener": [(16, 64, "k"), (16, 0, "a")],
    "Misc\\\\battery": [(0, 16, "+"), (0, 96, "-")], "battery": [(0, 16, "+"), (0, 96, "-")],
    "npn": [(64, 0, "c"), (0, 48, "b"), (64, 96, "e")], "pnp": [(64, 0, "e"), (0, 48, "b"), (64, 96, "c")]}
TYPES = {"res": "resistor", "cap": "capacitor", "polcap": "capacitor", "ind": "inductor", "voltage": "voltage_source",
         "current": "current_source", "diode": "diode", "LED": "diode", "zener": "zener", "Misc\\\\battery": "voltage_source",
         "battery": "voltage_source", "npn": "other", "pnp": "other"}


def _rot(x, y, r):
    m = r.startswith("M")
    if m:
        x = -x
    a = int(r[1:]) % 360
    return {0: (x, y), 90: (-y, x), 180: (-x, -y), 270: (y, -x)}[a]


def asc_netlist(path):
    from grader.circuit import Component, Netlist
    wires, symbols, flags = [], [], []
    cur = None
    for line in open(path, encoding="latin-1"):
        p = line.split()
        if not p:
            continue
        if p[0] == "WIRE":
            wires.append(tuple(map(int, p[1:5])))
        elif p[0] == "FLAG":
            flags.append((int(p[1]), int(p[2]), p[3]))
        elif p[0] == "SYMBOL":
            cur = {"sym": p[1], "x": int(p[2]), "y": int(p[3]), "rot": p[4], "name": None, "value": None}
            symbols.append(cur)
        elif p[0] == "SYMATTR" and cur is not None:
            if p[1] == "InstName":
                cur["name"] = p[2]
            elif p[1] == "Value":
                cur["value"] = " ".join(p[2:])
            elif p[1] == "Value2":
                cur["value2"] = " ".join(p[2:])
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def on_wire(pt):
        for x0, y0, x1, y1 in wires:
            if (x0 == x1 == pt[0] and min(y0, y1) <= pt[1] <= max(y0, y1)) or \
               (y0 == y1 == pt[1] and min(x0, x1) <= pt[0] <= max(x0, x1)):
                yield (x0, y0)
    for x0, y0, x1, y1 in wires:
        join((x0, y0), (x1, y1))
    for x0, y0, x1, y1 in wires:                      # an end on another wire's middle
        for pt in ((x0, y0), (x1, y1)):
            for w in on_wire(pt):
                join(pt, w)
    comps, unknown = [], []
    for s in symbols:
        key = s["sym"]
        if key not in PINS:
            unknown.append(key)
            continue
        pts = []
        for px, py, pname in PINS[key]:
            dx, dy = _rot(px, py, s["rot"])
            pt = (s["x"] + dx, s["y"] + dy)
            find(pt)
            for w in on_wire(pt):
                join(pt, w)
            pts.append((pt, pname))
        comps.append((s, pts))
    ground = None
    for x, y, name in flags:
        if name == "0":
            ground = find((x, y)) if ground is None else (join((x, y), ground) or find(ground))
    names = {}

    def node(pt):
        r = find(pt)
        if ground is not None and r == find(ground):
            return "0"
        return names.setdefault(r, f"N{len(names) + 1}")
    net = Netlist(ground="0" if ground is not None else None)
    for s, pts in comps:
        t = TYPES[s["sym"]]
        c = Component(s["name"] or s["sym"], t, [node(pt) for pt, _ in pts], label=s["name"], value=s["value"])
        ac = (s["value"] or "").strip('"').upper().startswith(("SINE", "PULSE", "AC")) or "AC" in (s.get("value2") or "").upper()
        if ac:                                         # an AC source: no polarity to read off the drawing
            c.value = s.get("value2") or s["value"]
            c.ac = True
        elif t in ("voltage_source", "current_source", "diode", "zener"):
            c.plus = node(pts[0][0])
        net.components.append(c)
    net.unknown_symbols = unknown
    return net
