"""Pen strokes of a drawing, found by a small U-Net trained on CGHD's hand-made stroke maps (tools/train_strokes.py):
photos of circuits on every kind of paper (ruled, squared, coloured), in every light, with pens and pencils - the
notebook's rules and the paper's grain are background, a faint pencil wire is not.

    from recog.strokes import StrokeModel
    sm = StrokeModel()                       # models/stroke-unet
    mask = sm(img)                           # bool array, True = pen stroke (image scaled to 1600 px on its long side)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "stroke-unet")
LONG_SIDE = 1600


class _Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4 levels, 16..128 channels (~1.9 M parameters); input RGB in 0..1, output one logit per pixel"""

    def __init__(self, ch=(16, 32, 64, 128), cin=3):
        super().__init__()
        self.down = nn.ModuleList()
        c = cin
        for k in ch:
            self.down.append(_Block(c, k))
            c = k
        self.mid = _Block(ch[-1], ch[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = ch[-1] * 2
        for k in reversed(ch):
            self.up.append(nn.ConvTranspose2d(c, k, 2, stride=2))
            self.dec.append(_Block(2 * k, k))
            c = k
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for blk in self.down:
            x = blk(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, s], 1))
        return self.out(x)


def scale_to(img, long_side: int = LONG_SIDE):
    from PIL import Image
    s = long_side / max(img.size)
    return img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR) if abs(s - 1) > 1e-3 else img


class StrokeModel:
    def __init__(self, model_dir: str = DEFAULT_DIR, device: str | None = None):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.net = UNet(tuple(cfg.get("channels", (16, 32, 64, 128))))
        self.net.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device).eval()
        self.threshold = float(cfg.get("threshold", 0.5))
        self.long_side = int(cfg.get("long_side", LONG_SIDE))

    @torch.no_grad()
    def probabilities(self, img, tile: int = 512, overlap: int = 64) -> np.ndarray:
        """stroke probability per pixel of the image as scaled to long_side (tiles with overlap)"""
        img = scale_to(img.convert("RGB"), self.long_side)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        _, H, W = x.shape
        ph, pw = (-H) % 16, (-W) % 16
        x = nn.functional.pad(x, (0, pw, 0, ph), value=1.0)
        H2, W2 = x.shape[1:]
        out = torch.zeros(H2, W2)
        cnt = torch.zeros(H2, W2)
        step = tile - overlap
        for y in range(0, max(1, H2 - overlap), step):
            for xx in range(0, max(1, W2 - overlap), step):
                y0, x0 = min(y, max(0, H2 - tile)), min(xx, max(0, W2 - tile))
                y1, x1 = min(H2, y0 + tile), min(W2, x0 + tile)
                p = torch.sigmoid(self.net(x[None, :, y0:y1, x0:x1].to(self.device)))[0, 0].float().cpu()
                out[y0:y1, x0:x1] += p
                cnt[y0:y1, x0:x1] += 1
        return (out / cnt.clamp(min=1))[:H, :W].numpy()

    def __call__(self, img) -> np.ndarray:
        return self.probabilities(img) >= self.threshold
