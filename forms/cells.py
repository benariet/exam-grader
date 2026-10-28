"""Read one handwritten character in one answer cell: a digit, '-', '.', nothing, or something
unreadable (a crossed-out cell, two characters squeezed into one).

    python -m forms.cells train --sources mnist_train,emnist_train,hwd,cvl_fit --cache <dir> --out <dir>

Training data: every cell is synthesized - real handwritten digits (MNIST; EMNIST, NIST's 3,600 writers;
HWD-V1; CVL's European-style writers) and MathWriting's isolated digits placed in a cell-sized crop at a
random size and position, a minus, a decimal point, an empty cell with specks, or a scribble; parts of the
cell's border, blur, noise and uneven paper. --cache keeps the synthesized cells, training is checkpointed
and continues where it stopped, --augment adds a random turn / shear / scale per cell. Testing uses digit
collections the readers never saw (USPS, Semeion, screen-drawn, CVL's other writers).

The classifier sees the cell interior at a fixed scale (not normalized to the character's size), so
a decimal point stays small and a minus stays flat. Its probabilities are calibrated (temperature)
on the validation cells; the grader uses them as reading probabilities. The readers in use are
models/cell-reader-v2 (ensemble.json: their probabilities are averaged); models/cell-reader is the first,
MNIST-only reader.
"""
from __future__ import annotations

import io
import itertools
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "cell-reader-v2")      # the readers in use (ensemble.json); the
#                                                                first, MNIST-only reader is models/cell-reader
TRAIN_OUT = os.path.join(HERE, "models", "cell-reader-trained")
def _first_dir(*paths):
    return next((p for p in paths if os.path.isdir(p)), paths[-1])


# digit collections: download_digits.py puts them in data/digits (the development machine keeps them elsewhere)
DIGITS = os.environ.get("DIGITS_DIR") or _first_dir("/home/claude/datasets/digits", os.path.join(HERE, "data", "digits"))
MNIST = os.environ.get("MNIST_DIR") or _first_dir("/home/claude/datasets/mnist/mnist", os.path.join(DIGITS, "mnist", "mnist"))
MW = os.environ.get("MATHWRITING_DIR", "/home/claude/datasets/mw_full")   # optional (prepare_mathwriting.py)
# digit sources: name -> (what it is). TRAIN on NIST-derived sets (+ HWD-V1); TEST students are written
# with sets the reader never saw, collected elsewhere (USPS envelopes, Semeion's Italian writers,
# digits drawn on screens)
SOURCES = {"mnist_train": "MNIST training digits (NIST)", "mnist_test": "MNIST test digits (NIST)",
           "emnist_train": "EMNIST digits, 240k (NIST SD19)", "emnist_test": "EMNIST digits test, 40k",
           "hwd": "HWD-V1, 137k (Kaggle, via leobottaro/handwritten-digits-pack)",
           "semeion": "Semeion, 1593 digits by ~80 Italian writers (UCI), 16x16",
           "usps_test": "USPS test, 2007 digits from envelopes, 16x16",
           "screen_test": "nguyenminh4099/handwritten-digits test, 10770 digits drawn on screens",
           "cvl_train": "CVL single digits, 7000 by ~60 Austrian writers (ICDAR 2013), scanned",
           "cvl_test": "CVL single digits validation, 7000 by OTHER writers",
           "cvl_fit": "CVL training digits of 57 of its 67 writers", "cvl_hold": "CVL training digits of the other 10 writers"}

CLASSES = list("0123456789") + ["-", ".", " ", "?"]          # ' ' empty, '?' unreadable
EMPTY, JUNK = CLASSES.index(" "), CLASSES.index("?")
IN_H, IN_W = 40, 32


# ------------------------------------------------------------------ the classifier's input
def clean_edges(ink: np.ndarray, band: int = 7, frac: float = 0.8) -> np.ndarray:
    """erase straight lines along the crop's edges - the printed cell border (or a block's border line)
    that a crop a few pixels off takes in, and that would read as a minus. A line counts when it runs
    across >= frac of the crop within `band` px of an edge; a written minus is shorter and central."""
    ink = ink.copy()
    H, W = ink.shape
    on = ink > 0.35
    for y in list(range(min(band, H))) + list(range(max(0, H - band), H)):
        if on[y].mean() >= frac:
            ink[y] = 0
    for x in list(range(min(band, W))) + list(range(max(0, W - band), W)):
        if on[:, x].mean() >= frac:
            ink[:, x] = 0
    return ink


def cell_input(gray: np.ndarray) -> np.ndarray:
    """cell interior (uint8 grayscale, ink dark) -> IN_H x IN_W float32 in [0, 1], ink = 1. The paper
    level is the crop's bright percentile; a crop without ink stays near 0 (no stretching of noise);
    border lines along the edges are erased (clean_edges)."""
    g = gray.astype(np.float32)
    bg = np.percentile(g, 90)
    ink = clean_edges(np.clip((bg - g) / max(60.0, bg - np.percentile(g, 0.5)), 0, 1))
    im = Image.fromarray((ink * 255).astype(np.uint8)).resize((IN_W, IN_H), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


# ------------------------------------------------------------------ synthesis
def _load_mnist(split):
    import pandas as pd
    df = pd.read_parquet(os.path.join(MNIST, f"{split}-00000-of-00001.parquet"))
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"]))) for b in df["image"]])
    return imgs, df["label"].to_numpy()


def load_source(name: str):
    """-> (images: uint8 arrays, ink bright on black; labels: int array)"""
    import pandas as pd

    def decode(col, invert=False, alpha=False):
        out = []
        for b in col:
            im = Image.open(io.BytesIO(b["bytes"]))
            a = np.asarray(im)
            if alpha and a.ndim == 3:
                a = a[..., 3]
            elif a.ndim == 3:
                a = np.asarray(im.convert("L"))
            out.append(255 - a if invert else a)
        return out
    if name in ("mnist_train", "mnist_test"):
        return _load_mnist(name.split("_")[1])
    if name in ("emnist_train", "emnist_test"):
        df = pd.read_parquet(os.path.join(DIGITS, "ernestchu__emnist-digits", "data", f"{name.split('_')[1]}-00000-of-00001.parquet"))
        return np.stack(decode(df["image"])), df["label"].to_numpy()
    if name == "hwd":
        df = pd.read_parquet(os.path.join(DIGITS, "leobottaro__handwritten-digits-pack", "train.parquet")).iloc[280000:416920]
        return np.stack(decode(df["image"], invert=True)), df["label"].to_numpy()
    if name == "usps_test":
        df = pd.read_parquet(os.path.join(DIGITS, "flwrlabs__usps", "data", "test-00000-of-00001.parquet"))
        return np.stack(decode(df["image"])), df["label"].to_numpy()
    if name == "screen_test":
        df = pd.read_parquet(os.path.join(DIGITS, "nguyenminh4099__handwritten-digits", "data", "test-00000-of-00001.parquet"))
        return np.stack(decode(df["img"], alpha=True)), df["label"].to_numpy()
    if name in ("cvl_train", "cvl_test", "cvl_fit", "cvl_hold") or name.startswith("cvl_first"):
        d = os.path.join(DIGITS, "cvl", "valid" if name == "cvl_test" else "train")
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if name in ("cvl_fit", "cvl_hold"):            # the training writers split by writer: 10 held out
            writers = sorted({f.split("-")[1] for f in files})
            hold = set(writers[-10:])
            files = [f for f in files if (f.split("-")[1] in hold) == (name == "cvl_hold")]
        elif name.startswith("cvl_first"):              # cvl_firstN: the first N training writers only - as if
            n = int(name[len("cvl_first"):])            # N people had filled copy sheets
            writers = sorted({f.split("-")[1] for f in files})[:n]
            files = [f for f in files if f.split("-")[1] in set(writers)]
        imgs, labs = [], []
        for f in files:
            if True:
                a = 255 - np.asarray(Image.open(os.path.join(d, f)).convert("L"))
                a = np.where(a > 40, a, 0).astype(np.uint8)          # paper texture off
                imgs.append(a)
                labs.append(int(f.split("-")[0]))
        return imgs, np.array(labs)
    if name == "semeion":
        rows = np.loadtxt(os.path.join(DIGITS, "semeion.data"))
        imgs = (rows[:, :256].reshape(-1, 16, 16) * 255).astype(np.uint8)
        return imgs, rows[:, 256:].argmax(1)
    raise ValueError(name)


def source_writers(name: str):
    """the writer of every image of a collection that records it (CVL: the file name), aligned with
    load_source(name); None for collections without writer IDs"""
    if name not in ("cvl_train", "cvl_test", "cvl_fit", "cvl_hold"):
        return None
    d = os.path.join(DIGITS, "cvl", "valid" if name == "cvl_test" else "train")
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if name in ("cvl_fit", "cvl_hold"):
        writers = sorted({f.split("-")[1] for f in files})
        hold = set(writers[-10:])
        files = [f for f in files if (f.split("-")[1] in hold) == (name == "cvl_hold")]
    return np.array([f.split("-")[1] for f in files])


def _load_mw_digits():
    """isolated handwritten digits / '-' / '.' of MathWriting (symbols split + one-character expressions)"""
    sys.path.insert(0, HERE)
    from recog.ink import read_traces
    out = []
    want = set("0123456789-.")
    if not os.path.isfile(os.path.join(MW, "labels.tsv")):
        print(f"(MathWriting not found at {MW}: training without its digits)", flush=True)
        return out
    for line in open(os.path.join(MW, "labels.tsv"), encoding="utf-8"):
        split, name, lab = line.rstrip("\n").split("\t")[:3]
        lab = lab.replace(" ", "")
        if lab in want and split in ("symbols", "train"):
            path = os.path.join(MW, "symbols" if split == "symbols" else "expr", name)
            if os.path.isfile(path):
                out.append((read_traces(path), lab))
    return out


def _ink_glyph(strokes, height, pen, rng):
    """render online strokes to a white-on-black glyph image of the given height"""
    allp = np.concatenate(strokes)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    s = (height - pen - 2) / max(1e-6, y1 - y0)
    if (x1 - x0) * s > 1.4 * height:                 # a flat stroke ('-'): scale by width instead
        s = (1.4 * height) / max(1e-6, x1 - x0)
    W, H = int((x1 - x0) * s) + pen + 3, int((y1 - y0) * s) + pen + 3
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for st in strokes:
        pts = [((x - x0) * s + pen / 2 + 1, (y - y0) * s + pen / 2 + 1) for x, y in st]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse((x - pen / 2, y - pen / 2, x + pen / 2, y + pen / 2), fill=255)
        else:
            d.line(pts, fill=255, width=pen, joint="curve")
    return im


def _mnist_glyph(img28, height, rng):
    ys, xs = np.nonzero(img28 > 30)
    if not len(ys):
        ys, xs = np.nonzero(img28 > 0)
    g = Image.fromarray(img28[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    w = max(2, int(round(g.width * height / g.height)))
    g = g.resize((w, height), Image.BILINEAR)
    r = rng.random()
    if r < 0.2:
        g = g.filter(ImageFilter.MaxFilter(3))              # a thicker pen
    elif r < 0.3 and height > 26:
        g = g.filter(ImageFilter.MinFilter(3))              # a thinner one
    return g


def _paste(canvas: np.ndarray, glyph: Image.Image, cx, cy, ink_level):
    """darken the canvas by the glyph (0-255 coverage) centred at (cx, cy)"""
    g = np.asarray(glyph, dtype=np.float32) / 255.0
    h, w = g.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = canvas.shape
    a0, b0 = max(0, -y0), max(0, -x0)
    a1, b1 = min(h, H - y0), min(w, W - x0)
    if a1 <= a0 or b1 <= b0:
        return
    region = canvas[y0 + a0:y0 + a1, x0 + b0:x0 + b1]
    cov = g[a0:a1, b0:b1]
    canvas[y0 + a0:y0 + a1, x0 + b0:x0 + b1] = region * (1 - cov) + ink_level * cov


def synth_cell(kind: str, rng, glyph_src=None, size=None) -> np.ndarray:
    """one cell interior as uint8 grayscale; kind: '0'-'9', '-', '.', ' ', '?' """
    h, w = size or (int(rng.integers(50, 62)), int(rng.integers(40, 52)))
    paper = float(rng.uniform(190, 252))
    c = np.full((h, w), paper, np.float32)
    ink = float(rng.uniform(10, 110))
    if kind.isdigit():
        gh = int(rng.uniform(0.42, 0.95) * h)
        if glyph_src is not None and isinstance(glyph_src, list):             # online strokes
            g = _ink_glyph(glyph_src, gh, int(rng.integers(2, 5)), rng)
        else:
            g = _mnist_glyph(glyph_src, gh, rng)
        if rng.random() < 0.4:
            g = g.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR, expand=True)
        _paste(c, g, rng.uniform(0.32, 0.68) * w, rng.uniform(0.32, 0.68) * h, ink)
    elif kind == "-":
        L = rng.uniform(0.25, 0.7) * w
        t = int(rng.integers(2, 6))
        g = Image.new("L", (int(L) + 8, 16), 0)
        d = ImageDraw.Draw(g)
        bend = rng.uniform(-2, 2)
        d.line([(4, 8), (4 + L / 2, 8 + bend), (4 + L, 8 + rng.uniform(-2, 2))], fill=255, width=t, joint="curve")
        g = g.rotate(float(rng.uniform(-10, 10)), resample=Image.BILINEAR, expand=True)
        _paste(c, g, rng.uniform(0.3, 0.7) * w, rng.uniform(0.38, 0.66) * h, ink)
    elif kind == ".":
        r = rng.uniform(1.6, 3.8)
        g = Image.new("L", (16, 16), 0)
        ImageDraw.Draw(g).ellipse((8 - r, 8 - r * rng.uniform(0.8, 1.2), 8 + r, 8 + r), fill=255)
        _paste(c, g, rng.uniform(0.2, 0.8) * w, rng.uniform(0.45, 0.96) * h, ink)    # a decimal point sits low
    elif kind == "?":
        r = rng.random()
        d_img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(d_img)
        if r < 0.5 and glyph_src is not None:                       # a digit crossed out
            g = _mnist_glyph(glyph_src, int(rng.uniform(0.5, 0.85) * h), rng)
            _paste(c, g, rng.uniform(0.35, 0.65) * w, rng.uniform(0.38, 0.62) * h, ink)
            t = int(rng.integers(2, 5))
            if rng.random() < 0.5:                                  # an X
                d.line([(rng.uniform(0, .25) * w, rng.uniform(0, .25) * h), (rng.uniform(.75, 1) * w, rng.uniform(.75, 1) * h)], fill=255, width=t)
                d.line([(rng.uniform(.75, 1) * w, rng.uniform(0, .25) * h), (rng.uniform(0, .25) * w, rng.uniform(.75, 1) * h)], fill=255, width=t)
            else:                                                   # scribbled over
                pts = [(rng.uniform(.1, .9) * w, rng.uniform(.15, .85) * h) for _ in range(int(rng.integers(5, 12)))]
                d.line(pts, fill=255, width=t, joint="curve")
        else:                                                       # a blot / scribble
            pts = [(rng.uniform(.1, .9) * w, rng.uniform(.1, .9) * h) for _ in range(int(rng.integers(6, 16)))]
            d.line(pts, fill=255, width=int(rng.integers(2, 6)), joint="curve")
        _paste(c, d_img, w / 2, h / 2, ink)
    # ' ': nothing written
    if rng.random() < 0.35:                                         # specks / dust
        for _ in range(int(rng.integers(1, 4))):
            r = rng.uniform(0.4, 1.1)
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            yy, xx = np.ogrid[:h, :w]
            m = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            c[m] = np.minimum(c[m], rng.uniform(120, 200))
    if rng.random() < 0.4:                                          # the cell's border creeping in
        side = int(rng.integers(0, 4))
        t = int(rng.integers(1, 4))
        o = int(rng.integers(0, 5))                                 # up to a few px inside the crop
        lvl = rng.uniform(80, 170)
        if side == 0:
            c[o:o + t, :] = np.minimum(c[o:o + t, :], lvl)
        elif side == 1:
            c[h - o - t:h - o, :] = np.minimum(c[h - o - t:h - o, :], lvl)
        elif side == 2:
            c[:, o:o + t] = np.minimum(c[:, o:o + t], lvl)
        else:
            c[:, w - o - t:w - o] = np.minimum(c[:, w - o - t:w - o], lvl)
    if rng.random() < 0.6:
        c = np.asarray(Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.1)))), dtype=np.float32)
    c += rng.normal(0, rng.uniform(1, 9), c.shape)
    c += np.linspace(0, rng.uniform(-25, 25), w)[None, :]           # uneven light
    return np.clip(c, 0, 255).astype(np.uint8)


def _u8(a):
    return (a * 255).round().astype(np.uint8)


def make_set(digits, labels, mw, n_other, seed, mw_repeat=0):
    """-> (uint8 cells N x IN_H x IN_W, labels); kept as uint8 from the start (a float copy of half a
    million cells does not fit in 8 GB next to the source images)"""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for img, lab in zip(digits, labels):
        X.append(_u8(cell_input(synth_cell(str(lab), rng, img))))
        Y.append(int(lab))
    for _ in range(mw_repeat):
        for strokes, lab in mw:
            k = lab if lab in "-." else lab
            if lab == ".":
                continue                       # MathWriting dots are drawn at the symbol's own scale
            X.append(_u8(cell_input(synth_cell(k, rng, strokes) if k.isdigit() else synth_cell(k, rng))))
            Y.append(CLASSES.index(k))
    for kind, n in (("-", n_other), (".", n_other), (" ", int(n_other * 1.3)), ("?", int(n_other * 0.8))):
        for _ in range(n):
            src = digits[int(rng.integers(len(digits)))]
            X.append(_u8(cell_input(synth_cell(kind, rng, src))))
            Y.append(CLASSES.index(kind))
    return np.stack(X), np.array(Y, np.int64)


# ------------------------------------------------------------------ model
def make_model(arch: str = "light"):
    import torch.nn as nn

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
    c1, c2, c3, c4 = (16, 32, 64, 96) if arch == "light" else (24, 48, 96, 128)
    return nn.Sequential(*block(1, c1), *block(c1, c2), nn.MaxPool2d(2),       # 20 x 16
                         *block(c2, c3), nn.MaxPool2d(2),                      # 10 x 8
                         *block(c3, c4), nn.MaxPool2d(2),                      # 5 x 4
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(c4 * 5 * 4, 128), nn.ReLU(),
                         nn.Linear(128, len(CLASSES)))


def build_sets(sources, val, seed, n_val=None, real=None, real_repeat=20, cache=None, real_people=None):
    """synthesized training and validation cells (uint8), plus collected real cells (forms/collect.py)
    when `real` is given (only real_people's, if listed), each repeated real_repeat times.
    cache: a folder - the synthesized cells are saved there once (Xtr.npy ...) and loaded from it next
    time (synthesizing half a million cells takes ~10 min); real cells are never cached"""
    names = ("Xtr", "Ytr", "Xva", "Yva")
    if cache and all(os.path.isfile(os.path.join(cache, n + ".npy")) for n in names):
        Xtr, Ytr, Xva, Yva = [np.load(os.path.join(cache, n + ".npy")) for n in names]
        print(f"cells loaded from {cache}", flush=True)
    else:
        xs, ys = zip(*[load_source(s) for s in sources])
        tr_x, tr_y = list(itertools.chain(*xs)), np.concatenate(ys)
        te_x, te_y = load_source(val)
        if n_val:
            k = np.random.default_rng(seed).permutation(len(te_x))[:n_val]
            te_x, te_y = te_x[k], te_y[k]
        mw = _load_mw_digits()
        print(f"train digits {len(tr_x)} from {sources}, val {len(te_x)} from {val}; MathWriting digits {len(mw)}", flush=True)
        n_other = max(7000, len(tr_x) // 9)
        Xtr, Ytr = make_set(tr_x, tr_y, mw, n_other, seed=seed, mw_repeat=20)
        Xva, Yva = make_set(te_x, te_y, [], max(1500, len(te_x) // 7), seed=seed + 100)
        if cache:
            os.makedirs(cache, exist_ok=True)
            for n, a in zip(names, (Xtr, Ytr, Xva, Yva)):
                np.save(os.path.join(cache, n + ".npy"), a)
            print(f"cells saved to {cache}", flush=True)
    if real:
        from forms.collect import load_real
        rc, rl = load_real(real, people=real_people)
        if rc:
            Xr = (np.stack([cell_input(c) for c in rc]) * 255).round().astype(np.uint8)
            Xtr = np.concatenate([Xtr] + [Xr] * real_repeat)
            Ytr = np.concatenate([Ytr] + [np.array(rl, np.int64)] * real_repeat)
            print(f"real cells: {len(rc)} x {real_repeat}", flush=True)
    return Xtr, Ytr, Xva, Yva


def _calibrate_and_save(model, xv, yv, out, info):
    """temperature on the validation cells, then model.pt / model.onnx / meta.json in out"""
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xv[i:i + 1024]) for i in range(0, len(xv), 1024)])
    best = min((float(F.cross_entropy(logits / T, yv)), T) for T in np.arange(0.5, 3.01, 0.05))
    T = float(best[1])
    p = torch.softmax(logits / T, 1)
    conf, pred = p.max(1)
    report = {"val_acc": float((pred == yv).float().mean()), "temperature": T,
              "per_class_acc": {CLASSES[k]: float((pred[yv == k] == k).float().mean()) for k in range(len(CLASSES))}}
    for th in (0.9, 0.95, 0.99):
        sel = conf >= th
        report[f"conf>={th}"] = {"share": float(sel.float().mean()), "error": float((pred[sel] != yv[sel]).float().mean())}
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    dummy = torch.zeros(1, 1, IN_H, IN_W)
    torch.onnx.export(model, dummy, os.path.join(out, "model.onnx"), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"classes": CLASSES, "in_h": IN_H, "in_w": IN_W, "temperature": T, **info, "report": report}, fh, indent=1)
    return report


def _random_affine(xb, gen_np, max_rot=10.0, max_shear=0.25, scale=(0.9, 1.1)):
    """every cell of the batch turned, sheared and scaled a little (a different slant and size of the same
    stroke) - geometric augmentation on top of the synthesized variety"""
    import torch
    import torch.nn.functional as F
    n = xb.shape[0]
    rot = np.deg2rad(gen_np.uniform(-max_rot, max_rot, n))
    sh = gen_np.uniform(-max_shear, max_shear, n)
    sc = gen_np.uniform(scale[0], scale[1], n)
    a = np.zeros((n, 2, 3), np.float32)
    a[:, 0, 0] = np.cos(rot) / sc
    a[:, 0, 1] = (-np.sin(rot) + sh) / sc
    a[:, 1, 0] = np.sin(rot) / sc
    a[:, 1, 1] = np.cos(rot) / sc
    grid = F.affine_grid(torch.from_numpy(a), list(xb.shape), align_corners=False)
    # the classifier's input is ink-bright on a dark (0) background: pad with background
    return F.grid_sample(xb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def train(epochs=8, out=TRAIN_OUT, sources=("mnist_train",), val="mnist_test", seed=1, arch="light", n_val=None,
          real=None, real_repeat=20, init=None, lr=3e-3, cache=None, ckpt_every=300, augment=False, real_people=None):
    """real: a folder of collected cells (forms/collect.py) mixed in, each repeated real_repeat times;
    init: a model folder to start from (fine-tuning: a few epochs, a lower lr);
    cache: folder for the synthesized cells (see build_sets);
    training state is checkpointed to out/ckpt.pt every ckpt_every steps and a run that finds one
    continues from it (long runs survive being stopped); the model is exported after every epoch"""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Xva, Yva = build_sets(sources, val, seed, n_val, real, real_repeat, cache, real_people)
    print("train", Xtr.shape, "val", Xva.shape, flush=True)
    info = {"arch": arch, "seed": seed, "train_sources": list(sources), "val_source": val, "epochs": epochs,
            "init": init, "lr": lr, "augment": augment}
    model = make_model(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    per_epoch = math.ceil(len(Xtr) / 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * per_epoch)
    xt, yt = torch.from_numpy(Xtr)[:, None], torch.from_numpy(Ytr)          # uint8: converted per batch
    xv, yv = torch.from_numpy(Xva.astype(np.float32) / 255.0)[:, None], torch.from_numpy(Yva)
    ckpt_path = os.path.join(out, "ckpt.pt")
    start_ep, start_step, perm = 0, 0, None
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)        # our own file
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, start_step, perm = ck["epoch"], ck["step"], ck["perm"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["np_rng"])
        print(f"continuing from {ckpt_path}: epoch {start_ep + 1}, step {start_step}", flush=True)
    elif init:
        model.load_state_dict(torch.load(os.path.join(init, "model.pt")))
        print("starting from", init, flush=True)
    os.makedirs(out, exist_ok=True)
    for ep in range(start_ep, epochs):
        model.train()
        if perm is None or ep != start_ep:
            perm = torch.randperm(len(xt))
        first = start_step if ep == start_ep else 0
        tot, n_seen = 0.0, 0
        for s in range(first, per_epoch):
            idx = perm[s * 256:(s + 1) * 256]
            xb = xt[idx].float() / 255.0
            dx, dy = np.random.randint(-2, 3), np.random.randint(-2, 3)     # small random shifts
            xb = torch.roll(xb, shifts=(dy, dx), dims=(2, 3))
            if augment:
                xb = _random_affine(xb, np.random)
            loss = F.cross_entropy(model(xb), yt[idx], label_smoothing=0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            n_seen += len(idx)
            if s % 250 == 0:
                print(f"  epoch {ep + 1} step {s}/{per_epoch} loss {float(loss.detach()):.3f}", flush=True)
            if (s + 1) % ckpt_every == 0 and s + 1 < per_epoch:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "epoch": ep, "step": s + 1, "perm": perm, "torch_rng": torch.get_rng_state(),
                            "np_rng": np.random.get_state()}, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
        rep = _calibrate_and_save(model, xv, yv, out, {**info, "epochs_done": ep + 1})
        print(f"epoch {ep + 1}: loss {tot / max(1, n_seen):.4f}  val acc {rep['val_acc']:.4f}  T {rep['temperature']:.2f}",
              flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "step": 0, "perm": None, "torch_rng": torch.get_rng_state(),
                    "np_rng": np.random.get_state()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    print(json.dumps(json.load(open(os.path.join(out, "meta.json")))["report"], indent=1), flush=True)
    print("TRAINING DONE", flush=True)


class CellReader:
    """probabilities over CLASSES for a batch of cell interiors"""

    def __init__(self, model_dir=MODEL_DIR):
        """model_dir: one model, or a folder with ensemble.json listing member folders (their
        probabilities are averaged), or a list of model folders"""
        import onnxruntime as ort
        dirs = model_dir if isinstance(model_dir, (list, tuple)) else [model_dir]
        if len(dirs) == 1 and os.path.isfile(os.path.join(dirs[0], "ensemble.json")):
            members = json.load(open(os.path.join(dirs[0], "ensemble.json")))["members"]
            dirs = [os.path.join(dirs[0], m) for m in members]
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("EXAM_GRADER_THREADS", "0")) or 0
        self.members = []
        for d in dirs:
            meta = json.load(open(os.path.join(d, "meta.json")))
            self.classes = meta["classes"]
            self.members.append((ort.InferenceSession(os.path.join(d, "model.onnx"), so, providers=["CPUExecutionProvider"]),
                                 meta["temperature"]))

    @staticmethod
    def _zoom(x: np.ndarray, s: float) -> np.ndarray:
        """the batch's cells scaled by s around their centre (background 0 where nothing is left)"""
        import cv2
        n, _, h, w = x.shape
        m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), 0.0, s)
        return np.stack([cv2.warpAffine(c[0], m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)[None] for c in x])

    def probs(self, crops: list, tta: bool = True) -> np.ndarray:
        """tta: the average over the crop and four 2-px shifts of it - steadier, and better calibrated
        on real pages than one look (EXAM_GRADER_CELL_ZOOM=1 adds the cell zoomed to 90 % and 110 %)"""
        if not crops:
            return np.zeros((0, len(self.classes)), np.float32)
        x = np.stack([cell_input(c) for c in crops])[:, None].astype(np.float32)
        shifts = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)] if tta else [(0, 0)]
        views = [np.roll(x, (dy, dx), axis=(2, 3)) for dy, dx in shifts]
        if tta and os.environ.get("EXAM_GRADER_CELL_ZOOM", "") not in ("", "0"):
            views += [self._zoom(x, 0.9), self._zoom(x, 1.1)]
        acc = 0
        for sess, T in self.members:
            for xs in views:
                z = sess.run(None, {"x": xs})[0] / T
                z -= z.max(1, keepdims=True)
                e = np.exp(z)
                acc = acc + e / e.sum(1, keepdims=True)
        return acc / (len(views) * len(self.members))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default=TRAIN_OUT)
    ap.add_argument("--sources", default="mnist_train")
    ap.add_argument("--val", default="mnist_test")
    ap.add_argument("--n-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arch", default="light", choices=["light", "plus"])
    ap.add_argument("--real", default=None, help="collected cells (forms/collect.py read) to mix in")
    ap.add_argument("--real-repeat", type=int, default=20)
    ap.add_argument("--init", default=None, help="start from this model folder (fine-tuning)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cache", default=None, help="folder to save / load the synthesized cells")
    ap.add_argument("--augment", action="store_true", help="random turn / shear / scale of every training cell")
    a = ap.parse_args()
    train(a.epochs, a.out, tuple(a.sources.split(",")), a.val, a.seed, a.arch, a.n_val or None,
          a.real, a.real_repeat, a.init, a.lr, a.cache, augment=a.augment)
