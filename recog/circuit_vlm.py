"""A hand-drawn circuit read by a local vision-language model (Qwen3-VL, recog/hebrew_vlm.py) into a netlist
(grader/circuit.py checks it and solves the circuit).

    from recog.circuit_vlm import CircuitReader
    rd = CircuitReader("models/qwen3-vl-8b")
    net, raw = rd.read("circuit.png")          # grader.circuit.Netlist, the model's answer
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PROMPT_NETLIST = """This is a hand-drawn electrical circuit (it may be drawn on lined paper - ignore the paper's lines). Read it as a netlist.

Rules:
- A node is a set of wire points joined by wires only, with no component in between. Name the nodes n0, n1, n2, ... A letter written at a node (like A) goes in node_labels.
- Every component joins exactly two nodes. Types: resistor (zigzag or box), voltage_source (circle with + and -, or battery), current_source (circle with an arrow), capacitor, inductor, diode.
- For each source, "plus" is the node on its + side.
- "label": the name written next to the component (R2, E1, ...), or null if nothing is written. "value": a number written next to it, or null.
- "ground": the node with the ground symbol, or null.
- "currents": every current arrow drawn, with its name (I1, ...): the component it runs through, the node it comes from and the node it points to.

Answer with JSON only, in this form:
{"components": [{"id": "R1", "type": "resistor", "label": null, "value": null, "nodes": ["n1", "n2"], "plus": null}],
 "ground": "n0", "node_labels": {"n2": "A"},
 "currents": [{"label": "I1", "through": "R1", "from": "n1", "to": "n2"}]}"""

PROMPT_TRANSISTORS = """This is a hand-drawn digital circuit made of MOSFET transistors (pass-transistor logic, transmission gates or static CMOS). A transistor is drawn as a raised step in a wire (its channel, with an end on each side) with a bar over or under it (its gate, with a short wire to a letter or to another node); a P-channel transistor has a small circle on its gate. Some drawings use the textbook symbol with an arrow instead.

Read it as a netlist:
- A node is a set of wire points joined by wires only, with no transistor in between. Name the nodes n0, n1, n2, ...
- A letter written at a node or at the free end of a wire names that node: put it in node_labels. A letter with a bar over it or a prime after it is a complement: write it with a prime, like "B'". A supply (VDD, an arrow or a T on top) is named "VDD", a ground "GND".
- For every transistor: "type" nmos or pmos, "gate" its gate's node, "ends" the two nodes its channel joins.

Answer with JSON only, in this form:
{"transistors": [{"id": "M1", "type": "nmos", "gate": "n3", "ends": ["n1", "n2"]}], "node_labels": {"n1": "A", "n3": "B'", "n2": "S"}}"""

PROMPT_DESCRIBE = ("Describe this hand-drawn electrical circuit: every component with the name written next to it, "
                   "what each end of it is connected to, the sources' polarity, the ground, and every current arrow "
                   "with its name and direction.")

PROMPT_KCL = "Write Kirchhoff's current law at node A of this circuit, using the current names drawn on it."


def crop_to_drawing(img: Image.Image, margin: float = 0.04) -> Image.Image:
    """the rows of the page that hold the drawing (found on the cleaned page: notebook rules removed), the
    full width kept - parts drawn lighter (a phone photo's lit edge) vanish from the cleaned page, not from this"""
    from .page import clean_scan
    clean, _ = clean_scan(img)
    g = np.asarray(clean.convert("L"))
    ys, xs = np.where(g < 128)
    if len(ys) < 50:
        return img
    lo, hi = np.percentile(ys, [0.2, 99.8]).astype(int)
    m = int(margin * img.height)
    return img.crop((0, max(0, lo - m), img.width, min(img.height, hi + m)))


class CircuitReader:
    def __init__(self, model_dir: str, load_in_4bit: bool | None = None, device: str | None = None, dtype=None):
        from . import gpu
        from .hebrew_vlm import HebrewRecognizer
        q4 = gpu.needs_4bit(gpu.params_b(model_dir)) if load_in_4bit is None else load_in_4bit
        self.vlm = HebrewRecognizer(model_dir, load_in_4bit=q4, device=device, dtype=dtype)
        self.quantized = q4

    def ask(self, img, prompt: str, max_new_tokens: int = 900) -> str:
        if isinstance(img, str):
            img = Image.open(img)
        return self.vlm.recognize(img.convert("RGB"), n_best=1, max_new_tokens=max_new_tokens, prompt=prompt)[0].text

    def read_transistors(self, img, max_new_tokens: int = 700):
        """a transistor network's drawing (already cut out) -> (Netlist of nmos / pmos with Gate / Drain / Source, the
        supply a source from 'VDD' to ground, 'GND' the ground; the model's answer)"""
        from grader.circuit import Component, Netlist, parse_json
        raw = self.ask(img, PROMPT_TRANSISTORS, max_new_tokens=max_new_tokens)
        try:
            d = parse_json(raw) or {}
        except Exception:                              # noqa: BLE001  (no JSON in the answer: nothing read)
            return Netlist(), raw
        labels = {str(k): str(v) for k, v in (d.get("node_labels") or {}).items() if v}
        rails = {n: str(t).strip().upper() for n, t in labels.items() if str(t).strip().upper() in ("VDD", "GND", "VSS", "0")}
        gnd = next((n for n, t in rails.items() if t != "VDD"), None)
        comps = []
        for k, t in enumerate(d.get("transistors") or []):
            g, ends = t.get("gate"), [str(e) for e in (t.get("ends") or []) if e is not None]
            kind = "pmos" if str(t.get("type", "")).lower().startswith("p") else "nmos"
            if g is None or len(ends) != 2:
                continue
            nodes = [str(g)] + ends
            nodes = ["0" if n == gnd else n for n in nodes]
            comps.append(Component(str(t.get("id") or f"M{k + 1}"), kind, nodes,
                                   roles={"Gate": nodes[0], "Drain": nodes[1], "Source": nodes[2]}))
        vdd = [n for n, t in rails.items() if t == "VDD"]
        if vdd and gnd is not None:
            comps.append(Component("VDD", "voltage_source", [vdd[0], "0"], plus=vdd[0]))
        labels = {("0" if n == gnd else n): t for n, t in labels.items() if n not in rails}
        return Netlist(comps, ground="0" if gnd is not None else None, node_labels=labels), raw

    def read(self, img, crop: bool = True):
        from grader.circuit import Netlist
        if isinstance(img, str):
            img = Image.open(img)
        img = crop_to_drawing(img) if crop else img
        raw = self.ask(img, PROMPT_NETLIST)
        return Netlist.from_json(raw), raw
