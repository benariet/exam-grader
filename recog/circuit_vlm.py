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

    def read(self, img, crop: bool = True):
        from grader.circuit import Netlist
        if isinstance(img, str):
            img = Image.open(img)
        img = crop_to_drawing(img) if crop else img
        raw = self.ask(img, PROMPT_NETLIST)
        return Netlist.from_json(raw), raw
