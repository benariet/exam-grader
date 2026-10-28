"""Which leg of a multi-terminal part is which - an op-amp's In+ / In- / Out, a BJT's base / collector / emitter, a
MOSFET's gate / drain / source - and the transistor's kind (NPN / PNP, N / P channel): small CNNs (tools/
train_roles.py) trained on Digitize-HCD's named port locations (Ahmed et al. 2025, CC BY 4.0).

The part's strokes are cut out with a margin and scaled into a SIZE x SIZE square; for each wire end found at the
part, a second channel marks that point, and the net says which role it is. The roles are then given to the ends
all at once, the most likely way (each role once; an end left over - an op-amp's supply pin - gets none).

    from recog.roles import RolesModel
    rm = RolesModel()
    roles, kind = rm.assign(mask, box, [(x, y), ...], "opamp")     # ['In+', 'Out', 'In-'], None
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import torch
from torch import nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "roles")
SIZE = 96
ROLES = {"bjt": ["Base", "Collector", "Emitter"], "fet": ["Gate", "Drain", "Source"], "opamp": ["In+", "In-", "Out"]}
KINDS = {"bjt": ["npn", "pnp"], "fet": ["nmos", "pmos"]}


class Net(nn.Module):
    def __init__(self, in_ch: int, n_out: int):
        super().__init__()
        c = [in_ch, 24, 48, 96, 128]
        layers = []
        for a, b in zip(c, c[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.f = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128 * 4, n_out))

    def forward(self, x):
        return self.head(self.f(x))


def square(ink: np.ndarray, pts, size: int = SIZE):
    """strokes (bool / 0-1) -> (size x size float picture, the points moved along): scaled to fit, centred"""
    from PIL import Image
    h, w = ink.shape
    s = size / max(h, w, 1)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = Image.fromarray((ink.astype(np.float32) * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    out = np.zeros((size, size), np.float32)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = np.asarray(im, np.float32) / 255.0
    return out, [(x * s + ox, y * s + oy) for x, y in pts]


def marker(pt, size: int = SIZE, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.exp(-((xx - pt[0]) ** 2 + (yy - pt[1]) ** 2) / (2 * sigma * sigma)).astype(np.float32)


class RolesModel:
    def __init__(self, model_dir: str = DEFAULT_DIR, device: str = "cpu"):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.roles, self.kinds, self.nets, self.kind_nets = cfg["roles"], cfg.get("kinds", {}), {}, {}
        for fam in cfg["families"]:
            n = Net(2, len(self.roles[fam]))
            n.load_state_dict(load_file(os.path.join(model_dir, f"{fam}_roles.safetensors")))
            self.nets[fam] = n.to(device).eval()
            kp = os.path.join(model_dir, f"{fam}_kind.safetensors")
            if fam in self.kinds and os.path.isfile(kp):
                k = Net(1, len(self.kinds[fam]))
                k.load_state_dict(load_file(kp))
                self.kind_nets[fam] = k.to(device).eval()
        self.device = device

    @torch.no_grad()
    def assign(self, mask, box, ends, fam: str, margin: float = 0.2):
        """mask: the page's strokes; box: the part (x0, y0, x1, y1); ends: [(x, y)] where its wires meet it
        -> ([role or None per end], kind or None); (None, None) when the family has no model"""
        net = self.nets.get(fam)
        if net is None or not ends:
            return None, None
        x0, y0, x1, y1 = box
        m = margin * max(x1 - x0, y1 - y0)
        X0, Y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
        X1, Y1 = int(min(mask.shape[1], x1 + m)), int(min(mask.shape[0], y1 + m))
        sq, pts = square(mask[Y0:Y1, X0:X1], [(x - X0, y - Y0) for x, y in ends])
        x = torch.from_numpy(np.stack([np.stack([sq, marker(p)]) for p in pts])).to(self.device)
        logp = torch.log_softmax(net(x), -1).cpu().numpy()                      # [ends, roles]
        names = self.roles[fam]
        best, best_lp = None, -1e9
        k = min(len(ends), len(names))
        for chosen in itertools.permutations(range(len(ends)), k):              # each role once, to the ends
            for perm in itertools.permutations(range(len(names)), k):
                lp = sum(logp[e, r] for e, r in zip(chosen, perm))
                if lp > best_lp:
                    best, best_lp = list(zip(chosen, perm)), lp
        roles = [None] * len(ends)
        for e, r in best or []:
            roles[e] = names[r]
        kind = None
        if fam in self.kind_nets:
            p = torch.softmax(self.kind_nets[fam](torch.from_numpy(sq)[None, None].to(self.device)), -1)[0].cpu().numpy()
            kind = self.kinds[fam][int(p.argmax())] if p.max() >= 0.7 else None
        return roles, kind
