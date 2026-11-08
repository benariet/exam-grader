"""Which logic gate a box holds - a second look at the detector's gates (8.11: on drafter 10's diagrams it took an OR
for a NOR, an XOR for an op-amp; its training had 34 XORs and 51 NORs). A small CNN on the box's ink alone (the strokes
darker than the paper around them, squared, 64 x 64), trained on HDL (Umm Al-Qura University students' hand-drawn
gates, 150 of each of 8 kinds, CC BY-SA 4.0) and the detector's own training boxes (tools/train_gates.py).

    from recog.gates import GateClassifier
    gc = GateClassifier()
    gc.classify([crop, ...])     # -> [(kind, p), ...]: 'and', 'or', 'not', 'nand', 'nor', 'xor', 'xnor', 'buffer'
"""
from __future__ import annotations

import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "gates")
KINDS = ["and", "or", "not", "nand", "nor", "xor", "xnor", "buffer"]
SIZE = 64


def ink(img) -> np.ndarray:
    """a crop's strokes: darker than the paper around them (a shadow, a photo's uneven light kept out) -> bool array"""
    from scipy import ndimage as ndi
    g = np.asarray(img.convert("L"), dtype=np.float32)
    if g.size == 0:
        return np.zeros((1, 1), bool)
    k = max(9, int(0.08 * max(g.shape)) | 1)
    paper = ndi.uniform_filter(ndi.maximum_filter(g, size=k), size=2 * k + 1)
    return g < paper - max(25.0, 0.18 * float(np.median(paper)))


def square(mask: np.ndarray, size: int = SIZE) -> np.ndarray:
    """the ink's bounding box, padded to a square with a margin, scaled to size x size -> float32 in [0, 1]"""
    from PIL import Image
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((size, size), np.float32)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    s = int(max(h, w) * 1.1) + 2
    out = np.zeros((s, s), np.float32)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    if s > 4 * size:                                   # (thin strokes survive the scaling: thickened first)
        from scipy import ndimage as ndi
        out = ndi.maximum_filter(out, size=max(1, s // (2 * size)))
    return np.asarray(Image.fromarray((out * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def net(n_classes: int = len(KINDS)):
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 128),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_classes))


class GateClassifier:
    def __init__(self, model_dir: str = DEFAULT_DIR, device: str = "cpu"):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        self.model = net()
        self.model.load_state_dict(load_file(os.path.join(model_dir, "gates.safetensors")))
        self.model.eval().to(device)
        self.device = device

    def classify(self, crops) -> list:
        """[(kind, probability)] per crop"""
        if not crops:
            return []
        torch = self.torch
        x = torch.from_numpy(np.stack([square(ink(c)) for c in crops]))[:, None].to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model(x), -1).cpu().numpy()
        return [(KINDS[int(r.argmax())], float(r.max())) for r in p]
