"""Layout for mixed Hebrew + math answers.

page -> lines (row gaps) -> chunks (column gaps) -> Hebrew / math (small CNN on skeletonised
strokes) -> adjacent chunks of the same kind merged into segments.

Every crop is normalised the same way before classification: binarise, scale to a fixed
height, skeletonise, redraw at one stroke width. That removes pen / scanner style, so the
classifier has to use the *shapes* (Hebrew letters vs. Latin letters, digits, '=', fraction bars).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_DIR = os.path.join(PROJECT, "models", "script-classifier")
NORM_H = 40
CANVAS_W = 200


# ------------------------------------------------------------------ normalisation
def ink_mask(img: Image.Image, thresh: int = 128) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8) < thresh


def normalize_crop(img: Image.Image, height: int = NORM_H, max_w: int = 320, tight_vertical: bool = False) -> np.ndarray:
    """-> float32 [height, W] with ink = 1, uniform 3-px strokes.

    By default the crop keeps its full height - pass a chunk cut from the *line band*, so a lone
    '=' or '.' keeps its true size and position relative to the line instead of being blown up
    to the full height (which made small formula pieces unrecognisable)."""
    from skimage.morphology import dilation, disk, skeletonize

    ink = ink_mask(img)
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return np.zeros((height, 8), np.float32)
    ink = ink[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] if tight_vertical else ink[:, xs.min(): xs.max() + 1]
    h, w = ink.shape
    inner = height - 8
    nw = max(4, min(max_w - 8, int(round(w * inner / h))))
    small = Image.fromarray((ink * 255).astype(np.uint8)).resize((nw, inner), Image.BILINEAR)
    b = np.asarray(small) > 64
    sk = dilation(skeletonize(b), disk(1))
    out = np.zeros((height, nw + 8), np.float32)
    out[4:4 + inner, 4:4 + nw] = sk
    return out


def to_canvas(x: np.ndarray, width: int = CANVAS_W) -> np.ndarray:
    """Place a normalised crop on a fixed NORM_H x width canvas (scaled down if wider, centred).
    A fixed size matters: with per-batch padding the classifier learned 'how much padding'
    (it depends on the other crops in the batch) instead of what is written."""
    h, w = x.shape
    if w > width:
        nh = max(4, int(round(h * width / w)))
        x = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((width, nh), Image.BILINEAR), np.float32) / 255.0
        h, w = x.shape
    out = np.zeros((NORM_H, width), np.float32)
    y0, x0 = (NORM_H - h) // 2, (width - w) // 2
    out[y0:y0 + h, x0:x0 + w] = x
    return out


# ------------------------------------------------------------------ segmentation
def pad_crop(page: Image.Image, box, frac: float = 0.25, min_pad: int = 8) -> Image.Image:
    """Crop with a white margin - recognizers misread symbols that touch the image border
    (a tight crop turned 'V = I.R' into '\\sqrt{-I.R}')."""
    x0, y0, x1, y1 = box
    p = max(min_pad, int(frac * (y1 - y0)))
    crop = page.crop((x0, y0, x1, y1)).convert("L")
    out = Image.new("L", (crop.width + 2 * p, crop.height + 2 * p), 255)
    out.paste(crop, (p, p))
    return out


def _runs(mask_1d: np.ndarray, gap: int):
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            runs.append((int(s), int(p) + 1)); s = i
        p = i
    runs.append((int(s), int(p) + 1))
    return runs


def segment_lines(page: Image.Image, min_gap: int = 6, min_height: int = 12):
    ink = ink_mask(page)
    rows = ink.sum(1) > max(1, ink.shape[1] // 500)
    return [(a, b) for a, b in _runs(rows, min_gap) if b - a >= min_height]


def segment_chunks(line: Image.Image, gap_factor: float = 0.30):
    """Chunks separated by column gaps wider than gap_factor * ink height."""
    ink = ink_mask(line)
    ys = np.where(ink.any(1))[0]
    h = (ys.max() - ys.min() + 1) if len(ys) else line.height
    band = ink[max(0, int(ys.min() + 0.1 * h)) if len(ys) else 0: int(ys.max() - 0.1 * h) + 1 if len(ys) else None]
    cols = band.sum(0) > 0
    return _runs(cols, max(2, int(gap_factor * h)))


def split_words(img: Image.Image, n_words: int):
    """Word boxes for a line whose transcription has n_words words, or None if the column gaps
    can't produce exactly that many chunks (then the word<->image mapping would be uncertain).
    Returned right-to-left (first Hebrew word first)."""
    ink = ink_mask(img)
    h = ink.shape[0]
    cols = ink[int(h * 0.15): int(h * 0.85)].sum(0) > 0
    matches = [g for g in range(2, h * 2) if len(_runs(cols, g)) == n_words]
    if not matches:
        return None
    return _runs(cols, matches[len(matches) // 2])[::-1]


# ------------------------------------------------------------------ classifier
class ScriptClassifier:
    """p(math) for a crop. ONNX model trained by train_segment_classifier.py."""

    def __init__(self, model_dir: str = CLS_DIR):
        import onnxruntime as ort
        from . import session_options
        so = session_options()
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.meta = json.load(open(os.path.join(model_dir, "meta.json"), encoding="utf-8"))

    def p_math(self, crop: Image.Image) -> float:
        tight = self.meta.get("crop_mode", "tight") == "tight"        # older models: per-chunk height
        x = to_canvas(normalize_crop(crop, tight_vertical=tight))[None, None]
        logits = self.sess.run(None, {"image": x})[0][0]
        z = float(logits[1] - logits[0])
        a, b = self.meta.get("platt_a", 1.0), self.meta.get("platt_b", 0.0)   # calibration (if available)
        return float(1 / (1 + np.exp(-(a * z + b))))


@dataclass
class Segment:
    kind: str                    # "hebrew" | "math"
    box: tuple                   # (x0, y0, x1, y1) on the page
    p_kind: float                # classifier probability of `kind` (lowest over merged chunks)
    nbest: list                  # recognizer candidates (text, logp)


def smooth_line(p_math: list[float], p_stay: float = 0.7) -> list[float]:
    """Forward-backward over the chunks of one line: Hebrew words and formulas come in runs,
    so an uncertain chunk between two Hebrew words is probably Hebrew. A confident chunk keeps
    its label (a single 'I' between Hebrew words stays math if the classifier is sure)."""
    n = len(p_math)
    if n <= 1:
        return list(p_math)
    T = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    E = np.array([[1 - p, p] for p in np.clip(p_math, 1e-4, 1 - 1e-4)])     # columns: hebrew, math
    a = np.zeros((n, 2)); b = np.ones((n, 2))
    a[0] = 0.5 * E[0]; a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = (a[t - 1] @ T) * E[t]; a[t] /= a[t].sum()
    for t in range(n - 2, -1, -1):
        b[t] = T @ (E[t + 1] * b[t + 1]); b[t] /= b[t].sum()
    post = a * b
    return list(post[:, 1] / post.sum(1))


def layout(page: Image.Image, clf: ScriptClassifier, smooth: bool = True):
    """-> list of lines; each line = list of (kind, box, p_kind) in *reading order* (right to left)."""
    lines = []
    for y0, y1 in segment_lines(page):
        line = page.crop((0, y0, page.width, y1))
        boxes = segment_chunks(line)
        pm = [clf.p_math(line.crop((x0, 0, x1, line.height))) for x0, x1 in boxes]
        if smooth:
            pm = smooth_line(pm)
        chunks = [[x0, x1, "math" if p >= 0.5 else "hebrew", max(p, 1 - p)] for (x0, x1), p in zip(boxes, pm)]
        # merge neighbours of the same kind (chunks are left-to-right here)
        merged = []
        for c in chunks:
            if merged and merged[-1][2] == c[2]:
                merged[-1][1] = c[1]; merged[-1][3] = min(merged[-1][3], c[3])
            else:
                merged.append(list(c))
        lines.append([(k, (x0, y0, x1, y1), p) for x0, x1, k, p in reversed(merged)])   # RTL reading order
    return lines
