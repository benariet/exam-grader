"""Which end of a polar part is which: a DC source's '+', a current source's arrow head, a diode's cathode - small
CNNs (tools/train_polarity.py) trained on Digitize-HCD's named port locations (Ahmed et al. 2025, CC BY 4.0).

The part's pen strokes are turned so that its two ends lie on a horizontal line, end A on the left, and a CNN
says how likely end A is the '+' (voltage source) / the end the arrow points to (current source) / the cathode
(diode).

    from recog.polarity import PolarityModel
    pm = PolarityModel()
    p = pm.first_end(mask, end_a, end_b, "voltage_dc")    # P(end_a is '+'); mask: bool strokes, ends: (x, y)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from torch import nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "polarity")
H, W = 48, 96
ROLES = {"voltage_dc": "Positive", "current_dc": "Flowing To", "diode": "Cathode"}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = [1, 16, 32, 64, 96]
        layers = []
        for a, b in zip(c, c[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(96 * (H // 16) * (W // 16), 1))

    def forward(self, x):
        return self.head(self.f(x))


def aligned(mask: np.ndarray, a, b, pad: float = 0.25) -> np.ndarray:
    """the strokes around the segment a-b, turned so that a is on the left and b on the right of a H x W
    picture (the segment spans its middle 1 - 2 pad of the width); float 0/1"""
    from PIL import Image
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay)
    if L < 4:
        return np.zeros((H, W), np.float32)
    ang = math.degrees(math.atan2(by - ay, bx - ax))
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    half_w = L / (2 * (1 - 2 * pad)) if pad < 0.5 else L
    half_h = half_w * H / W
    r = int(math.ceil(math.hypot(half_w, half_h))) + 2
    x0, y0 = int(cx) - r, int(cy) - r
    sub = np.zeros((2 * r, 2 * r), np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(mask.shape[0], y0 + 2 * r), min(mask.shape[1], x0 + 2 * r)
    if sy1 > sy0 and sx1 > sx0:
        sub[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1] * 255
    im = Image.fromarray(sub).rotate(ang, resample=Image.BILINEAR, center=(cx - x0, cy - y0))
    c = r
    im = im.crop((int(c - half_w), int(c - half_h), int(c + half_w), int(c + half_h))).resize((W, H), Image.BILINEAR)
    return (np.asarray(im, np.float32) / 255.0)


class PolarityModel:
    def __init__(self, model_dir: str = DEFAULT_DIR, device: str = "cpu"):
        from safetensors.torch import load_file
        self.nets = {}
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        for cls in cfg["classes"]:
            net = Net()
            net.load_state_dict(load_file(os.path.join(model_dir, f"{cls}.safetensors")))
            self.nets[cls] = net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def first_end(self, mask, a, b, cls: str) -> float | None:
        """P(end a is the class's first role); both orders are asked and averaged (b-first must be the opposite)"""
        net = self.nets.get(cls)
        if net is None:
            return None
        x = torch.from_numpy(np.stack([aligned(mask, a, b), aligned(mask, b, a)]))[:, None].to(self.device)
        p = torch.sigmoid(net(x)).flatten().tolist()
        return 0.5 * (p[0] + 1 - p[1])
