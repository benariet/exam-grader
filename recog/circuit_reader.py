"""A hand-drawn circuit, read end to end into a netlist (grader/circuit.py checks and solves it):

  1. the detector (models/circuit-detector, tools/train_circuit_detector.py) finds the components, value
     labels, grounds and crossovers;
  2. the stroke model (models/stroke-unet, tools/train_strokes.py; the classical threshold if it is missing)
     finds the pen strokes, faint pencil on ruled paper included;
  3. the label reader (models/label-reader, tools/train_labels.py; the formula reader if it is missing) reads
     each label ('5V', '1kΩ', 'R2');
  4. recog/circuit_wiring.py joins it all: wires -> nodes, each component's two ends, '+' / arrow / cathode
     (models/polarity).

    from recog.circuit_reader import CircuitReader
    rd = CircuitReader()
    net, found = rd.read(img)          # Netlist, [(cls, box, score)] in the image's own pixels
"""
from __future__ import annotations

import os
import re

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR_DIR = os.path.join(PROJECT, "models", "circuit-detector")
LONG_SIDE = 1600                     # the scale the detector and the stroke model were trained at
# what a label in a circuit drawing can be, as the formula reader writes it (tokens spaced): a name ('R_{1}', 'E1',
# 'A', 'I_{x}') or a value ('10 k \Omega', '5 V', '4.7 \mu F'); reading only these: 35 -> 45 % of CGHD's names /
# values exactly right on its test drafters
_NAME = r"[A-Za-z](\s*[A-Za-z])?(\s*_\s*\{\s*[0-9A-Za-z](\s*[0-9A-Za-z]){0,2}\s*\}|(\s*[0-9]){1,3})?"
_VALUE = (r"[0-9](\s*[0-9])*(\s*\.\s*[0-9](\s*[0-9])*)?(\s*(k|K|m|M|n|p|u|\\mu))?"
          r"(\s*(\\Omega|V|A|F|H|\\mathrm\s*\{\s*(V|A|F|H|\\Omega|k\s*\\Omega)\s*\}))?")
LABEL_GRAMMAR = (f"({_NAME}|{_VALUE})", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _{}.\\")


def label_text(tex: str) -> str:
    """the formula reader's LaTeX of a label -> plain text: 'R _ { 2 }' -> 'R2', '1 k \\Omega' -> '1kΩ'; other
    commands (an arrow drawn into the label's box: '\\rightarrow') are dropped"""
    t = re.sub(r"\\(?:mathrm|text|rm|mathit|boldsymbol|mathbf)\s*\{([^{}]*)\}", r"\1", tex)
    t = t.replace("\\Omega", "Ω").replace("\\mu", "µ").replace("\\omega", "Ω")
    t = re.sub(r"\\[A-Za-z]+", "", t)
    t = re.sub(r"[_^{}\\\s~,]", "", t)
    return t


class CircuitDetector:
    def __init__(self, model_dir: str = DETECTOR_DIR, device: str | None = None, threshold: float = 0.4,
                 spare_threshold: float = 0.15):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = RTDetrImageProcessor.from_pretrained(model_dir)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_dir).to(self.device).eval()
        self.names = self.model.config.id2label
        self.threshold = threshold
        self.spare_threshold = spare_threshold
        self.last_spares = []          # the last image's detections below the threshold (a part to fall back on)

    def __call__(self, img):
        """[(cls, (x0, y0, x1, y1), score)] in img's pixels; those below the threshold (down to spare_threshold) are
        kept in last_spares"""
        torch = self.torch
        img = img.convert("RGB")
        enc = self.proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            out = self.model(pixel_values=enc["pixel_values"].to(self.device))
        res = self.proc.post_process_object_detection(out, threshold=min(self.threshold, self.spare_threshold),
                                                      target_sizes=[(img.height, img.width)])[0]
        every = []
        for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            every.append((self.names[int(l)] if int(l) in self.names else self.names[str(int(l))], tuple(b), s))
        found = _dedupe([f for f in every if f[2] >= self.threshold])
        self.last_spares = [f for f in _dedupe(every) if f[2] < self.threshold]
        return found


def _dedupe(found, iou: float = 0.6, nested: float = 0.85):
    """one box per object: of boxes that overlap a lot (any class), or of one class one lying in the other (a
    transistor boxed whole and its symbol alone), the more confident stays"""
    def overlap(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
        return ix * iy / u if u > 0 else 0.0

    def inside(p, q):                                  # the share of the smaller box inside the other
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        return ix * iy / max(1e-6, min((p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1])))
    keep = []
    for f in sorted(found, key=lambda f: -f[2]):
        if all(overlap(f[1], k[1]) < iou and not (f[0] == k[0] != "text" and inside(f[1], k[1]) >= nested)
               for k in keep):
            keep.append(f)
    return keep


def _noted_spots(net) -> list:
    """where the wiring found a part's symbol left in the wires ('... at (x, y): ...' in its notes)"""
    return [(float(x), float(y)) for n in net.notes for x, y in re.findall(r"at \((-?\d+), (-?\d+)\)", n)]


def shift_notes(net, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0):
    """the spots in the wiring's notes ('... at (x, y): ...') moved to another frame: x * scale + dx"""
    net.notes = [re.sub(r"at \((-?\d+), (-?\d+)\)",
                        lambda m: f"at ({float(m.group(1)) * scale + dx:.0f}, {float(m.group(2)) * scale + dy:.0f})", n)
                 for n in net.notes]
    return net


def noted_spots(net) -> list:
    """where the notes point (in the netlist's frame)"""
    return _noted_spots(net)


def recover_missed(net, dbg, mask, comps, texts, spares, wire, most: int = 3):
    """a part's symbol left in the wires (the wiring's notes): the detector's best guess below its threshold that
    covers the spot is taken as that part, when the reading then has fewer problems -> (net, dbg, comps, taken)"""
    taken = []
    spares = [s for s in spares if s[0] != "text"]
    for _ in range(most):
        spots = _noted_spots(net)
        if not spots or not spares:
            break

        def covers(b):
            gx, gy = 0.2 * (b[2] - b[0]), 0.2 * (b[3] - b[1])
            return any(b[0] - gx <= x <= b[2] + gx and b[1] - gy <= y <= b[3] + gy for x, y in spots)
        cand = [s for s in spares if covers(s[1])]
        if not cand:
            break
        best = max(cand, key=lambda s: s[2])
        spares.remove(best)
        comps2 = comps + [(best[0], best[1])]
        net2, dbg2 = wire(mask, comps2, texts)
        if len(net2.problems()) < len(net.problems()):
            net, dbg, comps = net2, dbg2, comps2
            taken.append(best)
    return net, dbg, comps, taken


def recover_supply(net, dbg, mask, comps, texts, spares, wire, least: float = 0.15, most: int = 2):
    """a supply drawn at a wire's end (a CMOS gate's VDD arrow on top: Digitize-HCD's one-port supplies, which the
    detector scores 0.2 - 0.4 there, or takes for a ground) that the detector saw below its threshold where no part
    is boxed: taken when the wiring reads it as a supply joined to the circuit (its node another part's too) and
    the reading has no more problems -> (net, dbg, comps, taken)"""
    taken = []
    if net.components and all(c.type == "nmos" for c in net.components):
        return net, dbg, comps, taken                  # (a pass network - NMOS alone - has no supply: an output's
    #                                                    letter at its wire's end, 'Y', looks like an arrow to the detector)

    def inside(b, o):                                  # how much of b lies in o
        ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
        return ix / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    # (a 'ground' the detector half saw there too: the wiring takes one fed from below for a supply)
    cand = sorted((s for s in spares if (s[0] == "voltage_rail" and s[2] >= least or s[0] == "gnd" and s[2] >= 2 * least)
                   and all(inside(s[1], b) < 0.2 and inside(b, s[1]) < 0.5 for _, b in comps)), key=lambda s: -s[2])
    for best in cand[:most]:
        comps2 = comps + [(best[0], best[1])]
        net2, dbg2 = wire(mask, comps2, texts)
        before = sum(c.type == "voltage_source" for c in net.components)
        new = [c for c in net2.components if c.type == "voltage_source"]
        joined = len(new) > before and any(
            any(n in o.nodes for o in net2.components if o is not c) for c in new for n in c.nodes if n != "0")
        if joined and len(net2.problems()) <= len(net.problems()):
            net, dbg, comps = net2, dbg2, comps2
            taken.append(best)
    return net, dbg, comps, taken


def retry_darker(small, net, dbg, mask, comps, texts, wire, strokes_fn, darks=(170.0, 140.0)):
    """a reading with problems - the strokes may have taken in a table, a shadow, a stain round the drawing: the
    classical strokes again with a darker threshold, kept when the reading is sounder (reading_score; a pale wire
    lost to it usually leaves a loose end, so it is not kept then) - noted, since a wire wholly as pale as the grain
    would go unnoticed -> (net, dbg, mask)"""
    used = None
    for dark in darks:
        if reading_score(net)[0] == 0:
            break
        other = strokes_fn(small, dark=dark, faint=min(228.0, dark + 28.0))
        net2, dbg2 = wire(other, comps, texts)
        if reading_score(net2) < reading_score(net):
            net, dbg, mask, used = net2, dbg2, other, dark
    if used is not None:                      # a wire drawn as pale as the grain went with it, maybe: still a doubt
        net.notes.append(f"read again with darker strokes (threshold {used:.0f}): the strokes round the drawing were "
                         "noisy - a pale wire may be lost")
    return net, dbg, mask


# a value's unit settles a part the detector confuses (8.11, CGHD drafter 1: a capacitor's two plates taken for a
# battery '1µF' beside it, an upright coil taken for a resistor or a capacitor '10nH' beside it): unit -> (the part it
# is, the classes taken for it)
UNIT_PART = {"F": ("capacitor", {"voltage_dc", "inductor", "resistor"}),
             "H": ("inductor", {"resistor", "capacitor", "diode", "zener"}),
             "Ω": ("resistor", {"inductor", "capacitor"})}
SIBLINGS = ({"diode", "zener"},)          # one part boxed twice as both: the surer stays


def _gap(tb, pb) -> float:
    cx, cy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
    return float(np.hypot(max(pb[0] - cx, 0, cx - pb[2]), max(pb[1] - cy, 0, cy - pb[3])))


GATE_NAMES = {"AND": "and", "OR": "or", "NOT": "not", "NAND": "nand", "NOR": "nor", "XOR": "xor"}
# the classes a gate is taken for: another gate, a triangle (an op-amp's, a NOT's), a switch
GATE_CONFUSED = {"and", "or", "not", "nand", "nor", "xor", "opamp", "switch"}


def gate_name(t: str):
    """a label that names a gate ('OR', 'NOB' read for 'NOR', 'No7' for 'NOT', '00' for 'OR') -> its class, None if it
    names none or is as near two names"""
    from rapidfuzz.distance import Levenshtein
    u = re.sub(r"[^A-Z0-9]", "", (t or "").upper()).replace("0", "O").replace("7", "T").replace("8", "B")
    if not 2 <= len(u) <= 5:
        return None
    for v in (u, u.replace("B", "R")):                 # (an R read as a B: 'NOB')
        if v in GATE_NAMES:
            return GATE_NAMES[v]
    if len(u) < 3:                                     # ('OP', 'XO': too short to stand one letter off)
        return None
    d = sorted((Levenshtein.distance(u, n), n) for n in GATE_NAMES)
    return GATE_NAMES[d[0][1]] if d[0][0] <= 1 and d[1][0] > d[0][0] else None


def retype_by_names(comps, texts, reach: float = 1.0):
    """a gate's class set by the name written by it (drafter 10 writes 'OR', 'NOR', 'XOR' under each gate; the detector
    takes an OR for a NOR, an XOR for an op-amp, a NOT for a switch): the part nearest a label naming a gate, within
    `reach` times its size, of a class a gate is taken for -> (comps, [(old, new)])"""
    out, changed = list(comps), []
    for item in texts:
        b, t = item[0], item[1]
        g = gate_name(t)
        if g is None:
            continue
        parts = [(k, c, pb) for k, (c, pb) in enumerate(out) if c in GATE_CONFUSED]
        if not parts:
            continue
        k, c, pb = min(parts, key=lambda x: _gap(b, x[2]))
        if c != g and _gap(b, pb) <= reach * max(pb[2] - pb[0], pb[3] - pb[1]):
            out[k] = (g, pb)
            changed.append((c, g))
    return out, changed


def coils_to_transformer(comps):
    """two coils drawn side by side, facing each other (parallel, overlapping along their length, apart by less than
    their width - drafter 1's C7) are a transformer: one box round both (its four ends read by the wiring)"""
    out = list(comps)
    while True:
        coils = [k for k, (c, _) in enumerate(out) if c == "inductor"]
        pair = None
        for i in coils:
            for j in coils:
                if j <= i:
                    continue
                a, b = out[i][1], out[j][1]
                wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
                if ha >= 1.5 * wa and hb >= 1.5 * wb:            # upright coils, side by side
                    along = min(a[3], b[3]) - max(a[1], b[1])
                    gap = max(a[0], b[0]) - min(a[2], b[2])
                    if along >= 0.5 * min(ha, hb) and gap <= 1.0 * max(wa, wb):
                        pair = (i, j)
                elif wa >= 1.5 * ha and wb >= 1.5 * hb:          # lying coils, one above the other
                    along = min(a[2], b[2]) - max(a[0], b[0])
                    gap = max(a[1], b[1]) - min(a[3], b[3])
                    if along >= 0.5 * min(wa, wb) and gap <= 1.0 * max(ha, hb):
                        pair = (i, j)
                if pair:
                    break
            if pair:
                break
        if pair is None:
            return out
        i, j = pair
        a, b = out[i][1], out[j][1]
        out[i] = ("transformer", (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))
        del out[j]


def retype_by_shape(small, comps, classifier, skip=(), sure: float = 0.8, margin: float = 0.15):
    """a gate's class by the gate classifier's look at its box (recog/gates.py) where it is sure and differs from the
    detector's - not a box whose class its written name set (skip) -> (comps, [(old, new)])"""
    from .circuit_wiring import GATES
    kinds = set(getattr(classifier, "kinds", ())) or GATES
    idx = [k for k, (c, b) in enumerate(comps) if c in (GATES | {"opamp"}) & (kinds | GATES) and tuple(b) not in skip]
    if not idx or classifier is None:
        return comps, []
    crops = []
    for k in idx:
        x0, y0, x1, y1 = comps[k][1]
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        crops.append(small.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), int(x1 + mx), int(y1 + my))))
    out, changed = list(comps), []
    for k, (kind, p) in zip(idx, classifier.classify(crops)):
        if p >= sure and kind != out[k][0]:
            changed.append((out[k][0], kind))
            out[k] = (kind, out[k][1])
    return out, changed


def retype_by_units(comps, texts, reach: float = 1.5):
    """[(cls, box)] with a part's class set by its value's unit where the detector confused it: the part nearest a
    value ('1µF', '10nH', '250Ω'), within `reach` times its size, of a class taken for the unit's part - unless a part
    of the unit's kind is near enough to be the value's own (the label goes to it) -> (comps, [(old, new)])"""
    from .circuit_wiring import TWO_ENDS, _VALUE
    out, changed = list(comps), []
    for item in texts:
        b, t = item[0], (item[1] or "").strip()
        if not t or not _VALUE.match(t):
            continue
        m = re.search(r"(Ω|ohm|F|H)\s*$", t)
        if not m:
            continue
        target, taken_for = UNIT_PART["Ω" if m.group(1) == "ohm" else m.group(1)]
        parts = [(k, c, pb) for k, (c, pb) in enumerate(out) if c in TWO_ENDS]
        if not parts:
            continue
        k, c, pb = min(parts, key=lambda x: _gap(b, x[2]))
        size = max(pb[2] - pb[0], pb[3] - pb[1])
        own = [x for x in parts if x[1] == target and _gap(b, x[2]) <= 3 * max(x[2][2] - x[2][0], x[2][3] - x[2][1])]
        if c in taken_for and _gap(b, pb) <= reach * size and not own:
            out[k] = (target, pb)
            changed.append((c, target))
    return out, changed


def drop_siblings(found, share: float = 0.15):
    """[(cls, box, score)] with a part boxed twice as sibling classes (a diode and a Zener over each other - drafter 1:
    two diodes in parallel where one is drawn) once: of two such boxes overlapping by `share` of the smaller, the
    surer"""
    def over(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        small = min((p[2] - p[0]) * (p[3] - p[1]), (q[2] - q[0]) * (q[3] - q[1]))
        return ix * iy / small if small > 0 else 0.0
    keep = []
    for f in sorted(found, key=lambda f: -f[2]):
        if not any(f[0] != k[0] and any({f[0], k[0]} <= sib for sib in SIBLINGS) and over(f[1], k[1]) > share
                   for k in keep):
            keep.append(f)
    return [f for f in found if f in keep]


def _components_of(comps, ends) -> dict:
    """box index -> index of its part in the netlist (netlist_from_boxes places the boxes in order, leaving out
    labels, crossovers, grounds, terminals and a box no wire reaches)"""
    out, k = {}, 0
    for i, (cls, _) in enumerate(comps):
        if i < len(ends) and ends[i] is not None and cls not in ("gnd", "terminal", "text", "crossover"):
            out[i] = k
            k += 1
    return out


def drop_doubles(net, dbg, mask, comps, scores, wire):
    """one part boxed twice (drafter 1: an LED's body and its light arrows, both boxed as diodes): two boxes of one
    kind (or siblings: a diode and a Zener) that overlap and whose parts join the same two nodes - the less sure
    left out -> (net, dbg, comps, [(cls, box) left out])"""
    dropped = []
    while True:
        idx = _components_of(comps, dbg.get("ends") or [])
        if len(idx) != len(net.components):
            return net, dbg, comps, dropped
        pair = None
        for i in idx:
            for j in idx:
                if j <= i:
                    continue
                (ci, bi), (cj, bj) = comps[i], comps[j]
                same = ci == cj or any({ci, cj} <= sib for sib in SIBLINGS)
                a, b = net.components[idx[i]], net.components[idx[j]]
                touch = min(bi[2], bj[2]) > max(bi[0], bj[0]) and min(bi[3], bj[3]) > max(bi[1], bj[1])
                if same and touch and len(a.nodes) == len(b.nodes) == 2 and set(a.nodes) == set(b.nodes) and \
                        len(set(a.nodes)) == 2:
                    pair = (i, j)
                    break
            if pair:
                break
        if pair is None:
            return net, dbg, comps, dropped
        i, j = pair
        k = i if scores.get(tuple(comps[i][1]), 0.0) < scores.get(tuple(comps[j][1]), 0.0) else j
        dropped.append(comps[k])
        comps = comps[:k] + comps[k + 1:]
        net, dbg = wire(mask, comps, None)


def drop_one_ended(net, dbg, mask, comps, texts, wire, read_label):
    """a 'part' wired at one end or none - a label or a current's name boxed as a part (Tal's drawing: 'I1' by its
    arrow taken for a switch) - left out and read as a label, when the reading then has fewer problems
    (reading_score) -> (net, dbg, comps, texts, [(cls, box) left out])"""
    from .circuit_wiring import TWO_ENDS
    dropped = []
    while True:
        ends = dbg.get("ends") or []
        cand = [k for k, (cls, _) in enumerate(comps) if cls in TWO_ENDS and k < len(ends) and ends[k] is not None
                and sum(1 for v in ends[k] if v is not None) < 2]
        better = None
        for k in cand:
            b = comps[k][1]
            comps2 = comps[:k] + comps[k + 1:]
            texts2 = list(texts) + [(b,) + tuple(read_label(b))]
            net2, dbg2 = wire(mask, comps2, texts2)
            if reading_score(net2) < reading_score(better[0] if better else net):
                better = (net2, dbg2, comps2, texts2, comps[k])
        if better is None:
            return net, dbg, comps, texts, dropped
        net, dbg, comps, texts = better[:4]
        dropped.append(better[4])


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,3}$")


def unwired_as_labels(net, dbg, mask, comps, texts, wire, read_label, sure: float = 0.5):
    """a ground or a supply that no wire reaches, read as a name ('A', 'V1', 'Vout' - a CMOS gate's input letter 'A'
    boxed as a ground: its triangle) - a label, when the reading then is no less sound (reading_score)
    -> (net, dbg, comps, texts, [(cls, box) left out])"""
    ends = dbg.get("ends") or []
    cand = [k for k, (cls, _) in enumerate(comps) if cls in ("gnd", "voltage_rail") and k < len(ends)
            and not any(v is not None for v in (ends[k] or []))]
    new_texts, gone = [], []
    for k in cand:
        t, conf = read_label(comps[k][1])
        if t and _NAME.match(t.strip()) and conf >= sure:
            new_texts.append((comps[k][1], t.strip(), conf))
            gone.append(k)
    if not gone:
        return net, dbg, comps, texts, []
    comps2 = [c for k, c in enumerate(comps) if k not in gone]
    texts2 = list(texts) + new_texts
    net2, dbg2 = wire(mask, comps2, texts2)
    if reading_score(net2) > reading_score(net):
        return net, dbg, comps, texts, []
    return net2, dbg2, comps2, texts2, [comps[k] for k in gone]


_LETTER = re.compile(r"^\s*([A-Za-z])\s*[,.'`1lI|]?\s*$")


def complement_mark(small, box) -> str | None:
    """a letter written with its complement's mark - a bar over it, or "'" at its top right ('B\'') - as students
    write an input's complement by a pass transistor's gate: the letter's ink and the strokes around it (each a piece
    of its own, none running out of the look - a wire does): 'bar' / 'prime' / None. The label reader has no such
    mark among its characters ('B\'' reads 'B', 'B,')"""
    from scipy import ndimage as ndi
    from .gates import ink as ink_of
    x0, y0, x1, y1 = box
    bh = y1 - y0
    if bh < 8:
        return None
    cx0, cy0 = max(0, int(x0 - 0.3 * bh)), max(0, int(y0 - 0.6 * bh))
    cx1, cy1 = min(small.width, int(x1 + 0.7 * bh)), min(small.height, int(y1 + 0.2 * bh))
    m = ink_of(small.crop((cx0, cy0, cx1, cy1)))
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if not n:
        return None
    h, w = m.shape
    pieces = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        a, b, c, d = sl[1].start, sl[0].start, sl[1].stop, sl[0].stop      # (x0, y0, x1, y1) in the look
        size = int((lab[sl] == k).sum())
        if size < 4 or a == 0 or b == 0 or c >= w or d >= h:
            continue                                   # (a speck; a wire, running out of the look)
        pieces.append([a, b, c, d, size])
    if not pieces:
        return None
    bx = [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]      # the letter: the biggest piece in the box, and the pieces
    inside = [p for p in pieces if p[0] < bx[2] and p[2] > bx[0] and p[1] < bx[3] and p[3] > bx[1]]
    if not inside:
        return None
    body = list(max(inside, key=lambda p: p[4])[:4])   # over it (two strokes of one letter)
    rest = [p for p in pieces if p[:4] != body]
    grown = True
    while grown:
        grown = False
        for p in list(rest):
            ov_x = min(p[2], body[2]) - max(p[0], body[0])
            ov_y = min(p[3], body[3]) - max(p[1], body[1])
            if ov_x >= 0.5 * (p[2] - p[0]) and ov_y >= 0.5 * (p[3] - p[1]):
                body = [min(body[0], p[0]), min(body[1], p[1]), max(body[2], p[2]), max(body[3], p[3])]
                rest.remove(p)
                grown = True
    lw, lh = body[2] - body[0], body[3] - body[1]
    if lh < 6:
        return None
    for a, b, c, d, _ in rest:
        pw, ph = c - a, d - b
        over = min(c, body[2]) - max(a, body[0])
        if d <= body[1] + 0.15 * lh and d >= body[1] - 0.8 * lh and ph <= 0.3 * lh and pw >= 0.55 * lw and \
                pw >= 2.5 * ph and over >= 0.5 * lw:
            return "bar"
        if body[2] - 0.2 * lw <= a <= body[2] + 0.6 * lh and b <= body[1] + 0.2 * lh and d <= body[1] + 0.65 * lh and \
                0.12 * lh <= ph <= 0.6 * lh and pw <= 0.45 * lh and ph >= 0.6 * pw:
            return "prime"
    return None


def with_complements(small, boxes, reads):
    """the readings of single letters written with a complement's mark (complement_mark) as "B'" -> reads"""
    out = []
    for b, (t, conf) in zip(boxes, reads):
        m = _LETTER.match(t or "")
        out.append((m.group(1) + "'", conf) if m and complement_mark(small, b) else (t, conf))
    return out


def hinted_letters(texts, letters: str, logp, margin: float = 8.0, mark=None):
    """the letters the question names ('ABC': its inputs, 'S' its output): a short reading that is none of them nor
    '0' / '1' (a constant) - '3', '8', 'D' for a 'B' - read as the one of them the label reader finds nearly as
    likely (log P within margin); a complement's mark kept ("3'" -> "B'"). A reading with a complement's mark by it
    (mark(box) -> True: a bar over it, a "'") is a letter whatever it looks like - the likeliest of them, clearly.
    logp(box, [text, ...]) -> [log P]. -> (texts, [(index, before, after)])"""
    want = [c for c in dict.fromkeys(letters.upper()) if c.isalpha()]
    if not want:
        return texts, []
    texts, changed = list(texts), []
    for i, (b, t, *rest) in enumerate(texts):
        s = (t or "").strip()
        base, mark_ = (s[:-1], "'") if s.endswith("'") else (s, "")
        if not base or len(base) > 2 or base.upper() in want or base in ("0", "1") or \
                (len(base) == 2 and re.fullmatch(r"[A-Za-z]\d", base) and base[0].upper() not in want):
            continue
        cand = [x for c in want for x in (c, c.lower())]
        sc = logp(b, [base] + cand)
        per = sorted(((max(sc[1 + 2 * j], sc[2 + 2 * j]), c) for j, c in enumerate(want)), reverse=True)
        m_ = margin
        if not mark_ and len(base) == 1 and not base.isalpha() and mark is not None and mark(b):
            mark_, m_ = "'", float("inf")            # ('3' with a "'" by it: a complement - of a letter)
        if per[0][0] >= sc[0] - m_ and (len(per) == 1 or per[0][0] >= per[1][0] + 1.0):   # (one of them clearly)
            texts[i] = (b, per[0][1] + mark_, *rest)
            changed.append((i, t, per[0][1] + mark_))
    return texts, changed


def input_score(net, inputs: str) -> int:
    """how many of a transistor network's nodes get a Boolean function of the question's inputs alone"""
    want = {c.upper() for c in inputs if c.isalpha()}
    return sum(1 for f in net.switch_logic(inputs=inputs).values() if {str(x).upper() for x in f.free_symbols} <= want)


def gate_letters(net, texts, named_by, inputs: str, logp, rewire, margin: float = 8.0):
    """a transistor's gate is driven by one of the inputs the question names ('ABC') or its complement - a letter at
    a gate read as another ('S', the question's output, for a 'B' written like one) read as the input the label
    reader finds nearly as likely (log P within margin, one of them clearly), when more of the network's nodes then
    get functions of the inputs alone (input_score) and the reading is no less sound - one letter at a time, the
    likeliest first. named_by: {text index: the node name it gave} (the wiring's); rewire(texts) -> (net, dbg).
    -> (net, dbg or None, texts, [(index, before, after)])"""
    want = [c for c in dict.fromkeys(inputs.upper()) if c.isalpha()]
    fets = [c for c in net.components if c.type in ("nmos", "pmos") and c.roles.get("Gate")]
    if not want or not fets:
        return net, None, texts, []
    gates = {c.roles["Gate"] for c in fets}
    at_gate = {str(net.node_labels[n]).strip() for n in gates if n in net.node_labels}
    cand = []
    for i, name in named_by.items():
        s_ = (texts[i][1] or "").strip()
        base, mark_ = (s_[:-1], "'") if s_.endswith("'") else (s_, "")
        if name.strip() not in at_gate or len(base) != 1 or base.upper() in want or base in ("0", "1"):
            continue
        opts = [x for c in want for x in (c, c.lower())]
        sc = logp(texts[i][0], [base] + opts)
        per = sorted(((max(sc[1 + 2 * j], sc[2 + 2 * j]), c) for j, c in enumerate(want)), reverse=True)
        if per[0][0] >= sc[0] - margin and (len(per) == 1 or per[0][0] >= per[1][0] + 1.0):
            cand.append((per[0][0] - sc[0], i, per[0][1] + mark_))
    dbg, changed = None, []
    score = input_score(net, inputs)
    for _, i, new in sorted(cand, reverse=True):
        b, t, *rest = texts[i]
        texts2 = list(texts)
        texts2[i] = (b, new, *rest)
        net2, dbg2 = rewire(texts2)
        s2 = input_score(net2, inputs)
        if s2 > score and reading_score(net2) <= reading_score(net):
            net, dbg, texts, score = net2, dbg2, texts2, s2
            changed.append((i, t, new))
    return net, dbg, texts, changed


def consistent_letters(texts, comps, logp, margin: float = 6.0, pair_margin: float = 5.0, name_margin: float = 2.5):
    """the inputs' letters of a CMOS gate drawn with its transistors' gates labelled: each input drives a PMOS and an
    NMOS, so each letter is written an even number of times - a short reading that is no name ('8', 'D', '1B' for a
    'B') or a letter written once is read as the letter written an odd number of times that the label reader finds
    nearly as likely (log P within margin), one at a time, the likeliest first; two readings that are no names, as
    the letter not yet written that both are nearly as likely to be. Only with two MOSFETs or more.
    logp(box, [text, ...]) -> [log P]. -> (texts, [(index, before, after)])"""
    if sum(1 for c, _ in comps if c == "fet") < 2:
        return texts, []
    texts = list(texts)
    changed = []

    def name(t):
        t = (t or "").strip()
        if re.fullmatch(r"[A-Za-z]'", t):              # (a complement, "B'": a name of its own - kept)
            return t.upper()
        return t.upper() if len(t) == 1 and t.isalpha() and t.upper() != "I" else None

    for _ in range(len(texts)):
        counts = {}
        for _b, t, *_ in texts:
            if name(t):
                counts[name(t)] = counts.get(name(t), 0) + 1
        odd = {n for n, k in counts.items() if k % 2}
        best = None
        for i, (b, t, *rest) in enumerate(texts):
            s = (t or "").strip()
            free = len(s) == 1 and not name(s) or len(s) == 2 and not re.match(r"^[A-Za-z]\d$", s) and not name(s) or \
                name(s) in odd and "'" not in s
            # (a part's name - 'T1', 'R1' - only when the letter is nearly as likely: an 'A' read 'T1')
            m_ = margin if free else name_margin if re.match(r"^[A-Za-z]\d$", s) else None
            targets = sorted(n for n in odd - {name(s)} if "'" not in n)
            if not s or m_ is None or not targets:
                continue
            cand = [x for n in targets for x in (n, n.lower())]
            scores = logp(b, [s] + cand)
            own, k = scores[0], int(np.argmax(scores[1:]))
            gain = scores[1 + k] - own
            if gain >= -m_ and (best is None or gain > best[0]):
                best = (gain, i, cand[k].upper())
        if best is None:
            break
        _, i, new = best
        b, t, *rest = texts[i]
        texts[i] = (b, new, *rest)
        changed.append((i, t, new))
    # two readings that are no names ('3' and '8' for both 'B's): the one letter not yet written that both are
    # nearly as likely to be
    letters = list("ABCDEFGHJKLMNOPQRSTUVWXYZ")
    for _ in range(len(texts)):
        used = {name(t) for _b, t, *_ in texts if name(t)}
        free = [i for i, (_b, t, *_) in enumerate(texts) if (t or "").strip() and
                (len(t.strip()) == 1 and not name(t) or len(t.strip()) == 2 and not re.match(r"^[A-Za-z]\d$", t.strip())
                 and not name(t))]
        cand = [x for x in letters if x not in used]
        if len(free) < 2 or not cand:
            break
        gains = {}
        for i in free:
            b, t = texts[i][0], texts[i][1].strip()
            sc = logp(b, [t] + [y for x in cand for y in (x, x.lower())])
            gains[i] = {x: max(sc[1 + 2 * k], sc[2 + 2 * k]) - sc[0] for k, x in enumerate(cand)}
        best = max(((gains[i][x] + gains[j][x], i, j, x) for a_, i in enumerate(free) for j in free[a_ + 1:] for x in cand
                    if gains[i][x] >= -pair_margin and gains[j][x] >= -pair_margin), default=None)
        if best is None:
            break
        _, i, j, x = best
        for k in (i, j):
            b, t, *rest = texts[k]
            texts[k] = (b, x, *rest)
            changed.append((k, t, x))
    return texts, changed


def complementary_kinds(net):
    """a CMOS gate whose network is not complementary as read - some inputs leave its output joined to both rails or
    to neither - while one transistor's kind turned (an NMOS read for a PMOS: the arrow is a small mark) makes it so:
    that transistor taken as the other kind, and noted (a doubt - the student may have drawn it so) -> [ids turned]"""
    fets = [c for c in net.components if c.type in ("nmos", "pmos")]
    if len(fets) < 2:
        return []
    _, bad = net.switch_logic(report=True)
    if not bad:
        return []
    fixes = []
    for c in fets:
        c.type = "pmos" if c.type == "nmos" else "nmos"
        out, bad2 = net.switch_logic(report=True)
        c.type = "pmos" if c.type == "nmos" else "nmos"
        if out and not bad2:
            fixes.append(c)
    if len(fixes) != 1:                                # (none, or several alike: left as read)
        return []
    c = fixes[0]
    was = c.type
    c.type = "pmos" if was == "nmos" else "nmos"
    net.notes.append(f"{c.id}: read as {'an NMOS' if was == 'nmos' else 'a PMOS'}, but the gate's network is "
                     f"complementary only with {'a PMOS' if was == 'nmos' else 'an NMOS'} there - taken so")
    return [c.id]


def reading_score(net):
    """how sound a reading looks, lower better: its problems (loose wires, a part with one end, an unread
    polarity ...), then fewer two-terminal parts with two distinct ends"""
    two = sum(1 for c in net.components if c.type in ("resistor", "capacitor", "inductor", "voltage_source",
                                                      "current_source", "diode", "zener") and len(set(c.nodes)) == 2)
    return (len(net.problems()), -two)


class CircuitReader:
    def __init__(self, detector: CircuitDetector | None = None, strokes=None, formula=None, labels=None,
                 letters: str = "", inputs: str = ""):
        self.detector = detector or CircuitDetector()
        self.letters = letters                         # the letters the question names ('ABCS'): hinted_letters
        self.inputs = inputs                           # its inputs ('ABC'): a gate's letter one of them (gate_letters)
        if strokes is None:
            from .strokes import DEFAULT_DIR, StrokeModel
            strokes = StrokeModel() if os.path.isfile(os.path.join(DEFAULT_DIR, "model.safetensors")) else False
        self.strokes = strokes
        if labels is None:                             # the label reader when trained (tools/train_labels.py) ...
            from .labels import DEFAULT_DIR as LABELS_DIR, LabelReader
            labels = LabelReader() if os.path.isfile(os.path.join(LABELS_DIR, "model.safetensors")) else False
        self.labels = labels
        self.gates = None                              # the gate classifier (tools/train_gates.py), when trained
        from .gates import DEFAULT_DIR as GATES_DIR
        if os.path.isfile(os.path.join(GATES_DIR, "gates.safetensors")):
            from .gates import GateClassifier
            self.gates = GateClassifier()
        if formula is None and not labels:             # ... else the formula reader, held to the label grammar
            from .formula import FormulaRecognizer
            formula = FormulaRecognizer()
        self.formula = formula

    def read(self, img, return_debug: bool = False):
        """-> (Netlist, [(cls, box, score)] in img's pixels) (+ the wiring's debug dict, at LONG_SIDE: 'mask',
        'labels' (wire pieces), 'arrows', 'scale', 'texts')"""
        from .circuit_wiring import netlist_from_boxes, strokes as classical
        from .strokes import scale_to
        img = img.convert("RGB")
        s = LONG_SIDE / max(img.size)
        small = scale_to(img, LONG_SIDE)
        found = drop_siblings(self.detector(small))
        mask = self.strokes(small) if self.strokes else classical(small)
        comps = [(c, b) for c, b, _ in found if c != "text"]
        boxes = [b for c, b, _ in found if c == "text"]
        reads = self._read_labels(small, boxes, comps)
        texts = [(b, t or None, conf) for b, (t, conf) in zip(boxes, reads)]
        if self.letters and self.labels:               # a letter read as none the question names ('3' for its 'B')
            texts, _ = hinted_letters(texts, self.letters, lambda b, ts: self.labels.logp(self._label_crop(small, b), ts),
                                      mark=lambda b: complement_mark(small, b) is not None)
        comps, retyped = retype_by_units(comps, texts)
        before = list(comps)
        comps, renamed = retype_by_names(comps, texts)
        named = {tuple(b) for (c0, b), (c1, _) in zip(before, comps) if c0 != c1}
        comps, reshaped = retype_by_shape(small, comps, self.gates, skip=named)
        retyped = retyped + renamed + reshaped
        paired = coils_to_transformer(comps)
        if len(paired) != len(comps):                  # (two facing coils: the found boxes say one transformer)
            sc = {tuple(b): v for _, b, v in found}
            comps = paired
            found = [f for f in found if f[0] == "text"] + [(c, b, sc.get(tuple(b), 0.9)) for c, b in comps]
            retyped = []
        if retyped:                                    # (the boxes returned say the class the reading used)
            fixed = iter(comps)
            found = [(next(fixed)[0], b, sc) if c != "text" else (c, b, sc) for c, b, sc in found]
        net, dbg = netlist_from_boxes(mask, comps, texts, return_debug=True)
        if self.strokes:                               # the classical strokes too: each misses what the other keeps
            other = classical(small)                   # (a faint wire / a notebook rule) - the reading with fewer
            net2, dbg2 = netlist_from_boxes(other, comps, texts, return_debug=True)   # problems is kept
            if reading_score(net2) < reading_score(net):
                net, dbg, mask = net2, dbg2, other
        net, dbg, mask = retry_darker(small, net, dbg, mask, comps, texts,
                                      lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True), classical)
        spares = list(getattr(self.detector, "last_spares", []) or [])
        if net.notes and spares:                       # a part's symbol left in the wires: a part the detector
            net, dbg, comps, taken = recover_missed(   # saw below its threshold there?
                net, dbg, mask, comps, texts, spares,
                lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True))
            found = found + taken
        if spares:                                     # a supply arrow at a wire's end, seen below the threshold
            net, dbg, comps, taken = recover_supply(
                net, dbg, mask, comps, texts, spares, lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True))
            found = found + taken
        net, dbg, comps, doubles = drop_doubles(       # one part boxed twice: once
            net, dbg, mask, comps, {tuple(b): sc for _, b, sc in found},
            lambda m, c, _: netlist_from_boxes(m, c, texts, return_debug=True))
        for cls, b in doubles:
            found = [f for f in found if not (f[0] == cls and tuple(f[1]) == tuple(b))]
        if any(p.startswith(c.id + ":") and "ends" in p for c in net.components for p in net.problems()):
            net, dbg, comps, texts, dropped = drop_one_ended(   # a label boxed as a part: read as a label
                net, dbg, mask, comps, texts, lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True),
                lambda b: self._read_labels(small, [b], comps)[0])
            for cls, b in dropped:
                found = [f for f in found if not (f[0] == cls and tuple(f[1]) == tuple(b))]
        net, dbg, comps, texts, unwired = unwired_as_labels(   # a letter boxed as a ground: a label
            net, dbg, mask, comps, texts, lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True),
            lambda b: self._read_labels(small, [b], comps)[0])
        for cls, b in unwired:
            found = [f for f in found if not (f[0] == cls and tuple(f[1]) == tuple(b))] + [("text", b, 0.0)]
        if self.labels and {"nmos", "pmos"} <= {c.type for c in net.components}:   # a CMOS gate's input letters:
            # each written twice (a pass network's - an NMOS each - are not)
            texts2, changed = consistent_letters(texts, comps, lambda b, ts: self.labels.logp(self._label_crop(small, b), ts))
            if changed:
                net2, dbg2 = netlist_from_boxes(mask, comps, texts2, return_debug=True)
                if reading_score(net2) <= reading_score(net):
                    net, dbg, texts = net2, dbg2, texts2
        if self.inputs and self.labels:                # a transistor's gate: an input's letter by it
            net2, dbg2, texts, changed = gate_letters(
                net, texts, dbg.get("named_by") or {}, self.inputs,
                lambda b, ts: self.labels.logp(self._label_crop(small, b), ts),
                lambda t: netlist_from_boxes(mask, comps, t, return_debug=True))
            if changed:
                net, dbg = net2, dbg2
        complementary_kinds(net)                       # a CMOS gate's PMOS read as an NMOS (or the other way)
        back = [(c, tuple(v / s for v in b), sc) for c, b, sc in found]
        if not return_debug:
            shift_notes(net, 1.0 / s)                  # the notes' spots in img's pixels, as the boxes returned
        if return_debug:
            return net, back, {**dbg, "mask": mask, "scale": s, "texts": texts, "comps": comps}
        return net, back

    def _label_crop(self, small, box):
        x0, y0, x1, y1 = box
        m = 0.15 * (y1 - y0) if self.labels else 0.2 * (y1 - y0)
        return small.crop((max(0, int(x0 - m)), max(0, int(y0 - m)), int(x1 + m), int(y1 + m)))

    def _read_labels(self, small, boxes, comps):
        """each label box's reading -> [(text, how sure)], read knowing the kind of part beside it"""
        crops = [self._label_crop(small, b) for b in boxes]
        if not crops:
            return []
        if self.labels:
            from .labels import kind_of
            return with_complements(small, boxes, self.labels.read(crops, kinds=[kind_of(b, comps) for b in boxes]))
        reads = []
        for crop in crops:
            try:
                c = self.formula.recognize(crop, n_best=1, max_new_tokens=24, grammar=LABEL_GRAMMAR)[0]
                reads.append((label_text(c.text), float(np.exp(min(0.0, c.logp)))))
            except Exception:                          # noqa: BLE001
                reads.append(("", 0.0))
        return with_complements(small, boxes, reads)
