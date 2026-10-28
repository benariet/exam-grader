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


def _dedupe(found, iou: float = 0.6):
    """one box per object: of boxes that overlap a lot (any class), the more confident stays"""
    def overlap(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix * iy
        return ix * iy / u if u > 0 else 0.0
    keep = []
    for f in sorted(found, key=lambda f: -f[2]):
        if all(overlap(f[1], k[1]) < iou for k in keep):
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


def reading_score(net):
    """how sound a reading looks, lower better: its problems (loose wires, a part with one end, an unread
    polarity ...), then fewer two-terminal parts with two distinct ends"""
    two = sum(1 for c in net.components if c.type in ("resistor", "capacitor", "inductor", "voltage_source",
                                                      "current_source", "diode", "zener") and len(set(c.nodes)) == 2)
    return (len(net.problems()), -two)


class CircuitReader:
    def __init__(self, detector: CircuitDetector | None = None, strokes=None, formula=None, labels=None):
        self.detector = detector or CircuitDetector()
        if strokes is None:
            from .strokes import DEFAULT_DIR, StrokeModel
            strokes = StrokeModel() if os.path.isfile(os.path.join(DEFAULT_DIR, "model.safetensors")) else False
        self.strokes = strokes
        if labels is None:                             # the label reader when trained (tools/train_labels.py) ...
            from .labels import DEFAULT_DIR as LABELS_DIR, LabelReader
            labels = LabelReader() if os.path.isfile(os.path.join(LABELS_DIR, "model.safetensors")) else False
        self.labels = labels
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
        found = self.detector(small)
        mask = self.strokes(small) if self.strokes else classical(small)
        comps = [(c, b) for c, b, _ in found if c != "text"]
        boxes = [b for c, b, _ in found if c == "text"]
        crops = []
        for x0, y0, x1, y1 in boxes:
            m = 0.15 * (y1 - y0) if self.labels else 0.2 * (y1 - y0)
            crops.append(small.crop((max(0, int(x0 - m)), max(0, int(y0 - m)), int(x1 + m), int(y1 + m))))
        if self.labels:                                # each label read knowing the kind of part beside it
            from .labels import kind_of
            reads = self.labels.read(crops, kinds=[kind_of(b, comps) for b in boxes])
        else:
            reads = []
            for crop in crops:
                try:
                    c = self.formula.recognize(crop, n_best=1, max_new_tokens=24, grammar=LABEL_GRAMMAR)[0]
                    reads.append((label_text(c.text), float(np.exp(min(0.0, c.logp)))))
                except Exception:                      # noqa: BLE001
                    reads.append(("", 0.0))
        texts = [(b, t or None, conf) for b, (t, conf) in zip(boxes, reads)]
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
        back = [(c, tuple(v / s for v in b), sc) for c, b, sc in found]
        if not return_debug:
            shift_notes(net, 1.0 / s)                  # the notes' spots in img's pixels, as the boxes returned
        if return_debug:
            return net, back, {**dbg, "mask": mask, "scale": s, "texts": texts, "comps": comps}
        return net, back
