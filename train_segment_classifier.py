"""Train the Hebrew-vs-math chunk classifier used for mixed answers (recog/segment.py).

Train on exactly what the classifier sees at run time: mixed lines are composed from
  * Hebrew word spans cut from the Hebrew training lines (data/train/*.parquet), and
  * MathWriting handwritten expressions (prepare_mathwriting.py), at line scale,
then split with the same segment_lines / segment_chunks as at run time. Every chunk - a word,
a whole formula, or a formula fragment such as '=' or 'Y^2' - is labelled by its overlap with the
true parts and cut from the line band, so small pieces keep their size relative to the line.
Each chunk is skeletonised and redrawn at one stroke width on a fixed 40x200 canvas, so the
classifier can't separate the classes by scanner vs. pen style or by batch padding.

Validation / calibration: lines composed from test-split writers (data/hebrew_lines, minus the
writers used by eval_mixed.py) and MathWriting's valid/test splits. The model ends with Platt
calibration, so its probabilities can drive the human-review routing.

    python train_segment_classifier.py --mathwriting data/mathwriting
Output: models/script-classifier/{model.onnx, meta.json}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recog.ink import render  # noqa: E402
from recog.segment import (CANVAS_W, NORM_H, ink_mask, normalize_crop, segment_chunks,  # noqa: E402
                           segment_lines, split_words, to_canvas)

DEMO_EXCLUDE = {"V=I\\cdot R", "V=5V", "I=\\frac{1}{R}V"}     # kept out of training: used in demo_mixed.py
_TALL = re.compile(r"\\frac|\^\{|_\{|\\sum|\\int|\\sqrt")


# ------------------------------------------------------------------ sources
def eval_writers():
    """Writers used by eval_mixed.py - kept out of validation and calibration."""
    p = os.path.join(HERE, "data", "mixed_eval", "truth.json")
    return {t["writer"] for t in json.load(open(p, encoding="utf-8"))} if os.path.isfile(p) else set()


def hebrew_source_parquets(parquets, per_file, exclude_writers=frozenset(), shares=None):
    """[(line image, words, word boxes RTL)]; `shares` = {file-name substring: fraction} (default equal)."""
    out = []
    for p in parquets:
        n = 0
        quota = per_file
        if shares:
            frac = next((v for k, v in shares.items() if k in os.path.basename(p)), 1.0 / len(parquets))
            quota = int(per_file * len(parquets) * frac)
        for _, r in pd.read_parquet(p).sample(frac=1, random_state=0).iterrows():
            if "writer" in r and r["writer"] in exclude_writers:
                continue
            words = r["text"].split()
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")
            boxes = split_words(img, len(words))
            if boxes:
                out.append((img, words, boxes)); n += 1
            if n >= quota:
                break
    return out


def hebrew_source_dir(lines_dir, exclude_writers):
    out = []
    for it in json.load(open(os.path.join(lines_dir, "labels.json"), encoding="utf-8")):
        if it.get("writer") in exclude_writers:
            continue
        img = Image.open(os.path.join(lines_dir, it["file"])).convert("L")
        words = it["text"].split()
        boxes = split_words(img, len(words))
        if boxes:
            out.append((img, words, boxes))
    return out


def math_items(mw_dir, splits, n, rng, max_len=30):
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(mw_dir, "labels.tsv"), encoding="utf-8")]
    items = [(os.path.join(mw_dir, "expr", f), lab) for s, f, lab in rows
             if s in splits and lab not in DEMO_EXCLUDE and len(lab) <= max_len
             and os.path.isfile(os.path.join(mw_dir, "expr", f))]
    rng.shuffle(items)
    return items[:n]


# ------------------------------------------------------------------ composition
def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    s = rng.uniform(-0.25, 0.25)
    img = img.transform((w + int(abs(s) * h), h), Image.AFFINE, (1, s, -s * h if s > 0 else 0, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.4:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BILINEAR, expand=True, fillcolor=255)
    sx = rng.uniform(0.85, 1.2)
    return img.resize((max(4, int(img.width * sx)), img.height), Image.BILINEAR)


def _tight_rows(img):
    a = ink_mask(img)
    ys = np.where(a.any(1))[0]
    return img.crop((0, max(0, ys.min() - 3), img.width, min(img.height, ys.max() + 4))) if len(ys) else img


def compose_line(src, math, rng):
    """-> (line image, [(kind, x0, x1)])  parts in reading order (right to left)."""
    img, words, boxes = src
    k = rng.randint(1, min(3, len(words)))
    s0 = rng.randint(0, len(words) - k)
    span = boxes[s0:s0 + k]
    x0, x1 = min(b[0] for b in span), max(b[1] for b in span)
    heb = _tight_rows(img.crop((max(0, x0 - 2), 0, min(img.width, x1 + 2), img.height)))
    parts = []
    hs = 1.0
    r = rng.random()
    if math is None or r < 0.12:                                   # Hebrew only
        parts = [("hebrew", heb)]
    else:
        path, lab = math
        # wide size range on purpose: with band-relative crops a narrow range taught the model
        # "small relative to the line = Hebrew" (it then called a real 'V=5V' Hebrew at 99 %)
        scale = rng.uniform(0.9, 2.0) if _TALL.search(lab) else rng.uniform(0.5, 1.35)
        form = render(path, height=max(14, int(heb.height * scale)), pen=rng.choice([2, 3, 3, 4]))
        hs = rng.uniform(0.8, 1.25)                                 # vary the Hebrew size too
        heb = heb.resize((max(4, int(heb.width * hs)), max(8, int(heb.height * hs))), Image.BILINEAR)
        if r < 0.20:                                               # formula only
            parts = [("math", form)]
        elif r < 0.45 and k >= 2:                                  # formula in the middle
            cut = rng.randint(1, k - 1)
            bx = int((span[cut - 1][0] - x0 + 2) * hs)                # word boundary in the resized span
            bx = min(max(bx, 6), heb.width - 6)
            parts = [("hebrew", heb.crop((bx, 0, heb.width, heb.height))), ("math", form),
                     ("hebrew", heb.crop((0, 0, max(1, bx - 4), heb.height)))]
        elif r < 0.72:
            parts = [("hebrew", heb), ("math", form)]               # formula at the end (left)
        else:
            parts = [("math", form), ("hebrew", heb)]               # formula first (right)
    H = max(p[1].height for p in parts) + 12
    gaps = [rng.randint(22, 46) for _ in parts]
    W = sum(p[1].width for p in parts) + sum(gaps) + 24
    line = Image.new("L", (W, H), 255)
    x, spans = W - 12, []
    for (kind, im), g in zip(parts, gaps):
        x -= im.width
        line.paste(im, (x, (H - im.height) // 2 + rng.randint(-2, 2)))
        spans.append((kind, x, x + im.width))
        x -= g
    return line, spans


def line_chunks(sources, maths, n_lines, rng, aug):
    """Compose n_lines mixed lines and return (chunk crop from the line band, label) pairs."""
    out = []
    for _ in range(n_lines):
        src = rng.choice(sources)
        math = rng.choice(maths) if maths else None
        line, spans = compose_line(src, math, rng)
        if aug:
            # shear changes x positions; keep it mild and recompute spans by the same transform
            s = rng.uniform(-0.15, 0.15)
            w, h = line.size
            dx = -s * h if s > 0 else 0
            line = line.transform((w + int(abs(s) * h), h), Image.AFFINE, (1, s, dx, 0, 1, 0),
                                  resample=Image.BILINEAR, fillcolor=255)
            spans = [(k, a - dx - s * h / 2, b - dx - s * h / 2) for k, a, b in spans]
        bands = segment_lines(line)
        if not bands:
            continue
        y0, y1 = bands[0][0], bands[-1][1]
        band = line.crop((0, y0, line.width, y1))
        for cx0, cx1 in segment_chunks(band):
            cover = {"hebrew": 0.0, "math": 0.0}
            for k, a, b in spans:
                cover[k] += max(0, min(cx1, b) - max(cx0, a))
            tot = cx1 - cx0
            main = max(cover, key=cover.get)
            other = "math" if main == "hebrew" else "hebrew"
            if cover[main] < 0.6 * tot or cover[other] > 0.15 * tot:       # ambiguous / merged chunk
                continue
            out.append((band.crop((cx0, 0, cx1, band.height)), 1 if main == "math" else 0))
    return out


def to_x(crop):
    return to_canvas(normalize_crop(crop))                         # band mode (keeps relative size)


# ------------------------------------------------------------------ model
class ScriptCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(1, 16), nn.MaxPool2d(2), blk(16, 32), nn.MaxPool2d(2),
                               blk(32, 64), nn.MaxPool2d(2), blk(64, 96), blk(96, 96))
        self.head = nn.Linear(192, 2)

    def forward(self, x):
        f = self.f(x)
        return self.head(torch.cat([f.amax(dim=(2, 3)), f.mean(dim=(2, 3))], dim=1))


def batches(samples, bs, shuffle):
    idx = list(range(len(samples)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [samples[j] for j in idx[i:i + bs]]
        yield torch.from_numpy(np.stack([x for x, _ in chunk])[:, None]), torch.tensor([y for _, y in chunk])


def evaluate(model, data):
    model.eval()
    conf = np.zeros((2, 2), int)
    with torch.no_grad():
        for X, y in batches(data, 256, False):
            for a, b in zip(y.tolist(), model(X).argmax(1).tolist()):
                conf[a, b] += 1
    model.train()
    return conf.trace() / max(1, conf.sum()), conf


def platt(z, y):
    a, b = 1.0, 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        w = p * (1 - p)
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mathwriting", default=os.path.join(HERE, "data", "mathwriting"))
    ap.add_argument("--train-dir", default=os.path.join(HERE, "data", "train"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "script-classifier"))
    ap.add_argument("--lines", type=int, default=3500, help="composed training lines per epoch")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val-parquet", default=os.path.join(HERE, "data", "val", "matan_test.parquet"),
                    help="Hebrew lines of test-split writers for validation (writers of data/mixed_eval are skipped)")
    args = ap.parse_args()
    rng = random.Random(0); random.seed(0); torch.manual_seed(0)
    t0 = time.time()

    parquets = sorted(os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith(".parquet"))
    heb_tr = hebrew_source_parquets(parquets, 2500, shares={"matan": 0.7, "human": 0.3})
    if os.path.isfile(args.val_parquet):          # many writers -> a stable validation / calibration set
        heb_va = hebrew_source_parquets([args.val_parquet], 600, eval_writers())
    else:
        heb_va = hebrew_source_dir(os.path.join(HERE, "data", "hebrew_lines"), eval_writers())
    m_tr = math_items(args.mathwriting, {"train"}, 30000, rng)
    m_va = math_items(args.mathwriting, {"valid", "test"}, 3000, rng)
    vrng = random.Random(123)
    val = [(to_x(c), y) for c, y in line_chunks(heb_va, m_va, 1500, vrng, aug=False)]
    print(f"sources: hebrew lines train {len(heb_tr)} / val {len(heb_va)}   formulas train {len(m_tr)} / val {len(m_va)}"
          f"   val chunks {len(val)} (math {sum(y for _, y in val)})  ({time.time() - t0:.0f}s)", flush=True)

    model = ScriptCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        train = [(to_x(c), y) for c, y in line_chunks(heb_tr, m_tr, args.lines, rng, aug=True)]   # fresh lines each epoch
        for X, y in batches(train, 64, True):
            loss = nn.functional.cross_entropy(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        acc, conf = evaluate(model, val)
        model.eval()
        with torch.no_grad():
            z = np.concatenate([(lambda l: (l[:, 1] - l[:, 0]).numpy())(model(X)) for X, _ in batches(val, 256, False)])
        model.train()
        yv = np.array([y for _, y in val], np.float64)
        a, b = platt(z.astype(np.float64), yv)
        pc = 1 / (1 + np.exp(-(a * z + b)))
        cal_acc = float(((pc >= 0.5) == (yv == 1)).mean())
        order = np.argsort(z); ranks = np.empty(len(z)); ranks[order] = np.arange(1, len(z) + 1)
        npos, nneg = yv.sum(), len(yv) - yv.sum()
        auc = float((ranks[yv == 1].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))
        print(f"epoch {ep}  train chunks {len(train)}  val acc {acc:.4f} (raw)  calibrated {cal_acc:.4f}  AUC {auc:.4f}  "
              f"hebrew->math {conf[0, 1]}/{conf[0].sum()}  math->hebrew {conf[1, 0]}/{conf[1].sum()}  ({time.time() - t0:.0f}s)", flush=True)
        os.makedirs(args.out, exist_ok=True)
        model.eval()
        torch.onnx.export(model, (torch.zeros(1, 1, NORM_H, CANVAS_W),), os.path.join(args.out, f"model_ep{ep}.onnx"),
                          input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}},
                          opset_version=17, dynamo=False)
        model.train()
        if cal_acc > best:
            best = cal_acc
            import shutil
            shutil.copy(os.path.join(args.out, f"model_ep{ep}.onnx"), os.path.join(args.out, "model.onnx"))
            json.dump({"classes": ["hebrew", "math"], "norm_height": NORM_H, "canvas_width": CANVAS_W, "crop_mode": "band",
                       "val_acc_raw": round(float(acc), 4), "val_acc_calibrated": round(cal_acc, 4), "val_auc": round(auc, 4),
                       "val_confusion_raw": conf.tolist(), "epoch": ep, "platt_a": a, "platt_b": b,
                       "calibration": "Platt on val chunks (test-split writers; writers of data/mixed_eval excluded)",
                       "excluded_from_training": sorted(DEMO_EXCLUDE)},
                      open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"best calibrated val acc {best:.4f} -> {args.out}", flush=True)




def calibrate_existing(model_dir, mw_dir, val_parquet, n_lines=1500):
    """Platt-calibrate an already trained classifier on run-time-like chunks (composed val lines),
    using the model's own crop mode. Writes platt_a / platt_b into its meta.json."""
    from recog.segment import ScriptClassifier
    rng = random.Random(123)
    if os.path.isfile(val_parquet):
        src = hebrew_source_parquets([val_parquet], 600, eval_writers())
    else:
        src = hebrew_source_dir(os.path.join(HERE, "data", "hebrew_lines"), eval_writers())
    maths = math_items(mw_dir, {"valid", "test"}, 3000, rng)
    clf = ScriptClassifier(model_dir)
    clf.meta.pop("platt_a", None); clf.meta.pop("platt_b", None)          # raw logits
    zs, ys = [], []
    for crop, y in line_chunks(src, maths, n_lines, rng, aug=False):
        p = min(max(clf.p_math(crop), 1e-7), 1 - 1e-7)
        zs.append(np.log(p / (1 - p))); ys.append(y)
    z, y = np.array(zs), np.array(ys, np.float64)
    a, b = platt(z, y)
    pc = 1 / (1 + np.exp(-(a * z + b)))
    acc_raw = float(((z >= 0) == (y == 1)).mean()); acc_cal = float(((pc >= 0.5) == (y == 1)).mean())
    meta_p = os.path.join(model_dir, "meta.json")
    meta = json.load(open(meta_p, encoding="utf-8"))
    meta.update({"platt_a": a, "platt_b": b, "calib_chunks": int(len(y)), "calib_acc_raw": round(acc_raw, 4),
                 "calib_acc_calibrated": round(acc_cal, 4), "calibration": "Platt on run-time-like chunks; writers of data/mixed_eval excluded"})
    json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{model_dir}: {len(y)} chunks  raw acc {acc_raw:.4f}  calibrated {acc_cal:.4f}  (a={a:.3f}, b={b:.3f})")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:            # python train_segment_classifier.py --calibrate MODEL_DIR [MW_DIR] [VAL_PARQUET]
        i = sys.argv.index("--calibrate")
        rest = sys.argv[i + 1:] + [os.path.join(HERE, "data", "mathwriting"), os.path.join(HERE, "data", "val", "matan_test.parquet")][len(sys.argv[i + 2:]):]
        calibrate_existing(rest[0], rest[1], rest[2])
    else:
        main()
