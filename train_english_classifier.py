"""Train the Hebrew-vs-English chunk classifier (second opinion next to the Hebrew/math classifier).

The Hebrew/math classifier never saw English handwriting; cursive English words look more like
Hebrew to it than like formulas, so English words were read as (garbled) Hebrew. This classifier is
trained the same way as train_segment_classifier.py - chunks of composed lines, cut from the line
band, skeletonised on a fixed canvas - but on Hebrew word spans vs English word spans:

  Hebrew : data/train/*.parquet (training writers); validation: test-split writers (minus eval_mixed's)
  English: IAM handwritten lines (Teklia/IAM-line: train / validation; the test split is kept for
           the evaluations - prepare_data.py --english downloads it to data/english_lines)

    python train_english_classifier.py --iam data/english_lines
Output: models/english-classifier/{model.onnx, meta.json}  (classes: hebrew, english; Platt-calibrated)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recog.segment import CANVAS_W, NORM_H, segment_chunks, segment_lines, split_words  # noqa: E402
from train_segment_classifier import (ScriptCNN, _tight_rows, batches, eval_writers, evaluate,  # noqa: E402
                                      hebrew_source_parquets, platt, to_x)


def english_sources(parquet, n, seed=0):
    """[(line image, n words, word boxes)] from an IAM-style parquet (image, text)."""
    out = []
    df = pd.read_parquet(parquet).sample(frac=1, random_state=seed)
    for _, r in df.iterrows():
        words = [w for w in r["text"].split() if any(ch.isalnum() for ch in w)]   # punctuation sticks to a word
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")
        boxes = split_words(img, len(words)) if words else None
        if boxes:
            out.append((img, len(words), sorted(boxes)))          # left to right
        if len(out) >= n:
            break
    return out


def span(src, rng, kmax=3):
    img, n, boxes = src
    k = rng.randint(1, min(kmax, n))
    s0 = rng.randint(0, n - k)
    x0, x1 = boxes[s0][0], boxes[s0 + k - 1][1]
    return _tight_rows(img.crop((max(0, x0 - 2), 0, min(img.width, x1 + 2), img.height)))


def compose(heb_src, en_src, rng):
    """-> (line image, [(kind, x0, x1)]): Hebrew and English spans in a random order."""
    himg, hwords, hboxes = heb_src
    heb = span((himg, len(hwords), sorted(hboxes)), rng)
    en = span(en_src, rng)
    en = en.resize((max(4, int(en.width * heb.height * rng.uniform(0.8, 1.25) / en.height)),
                    max(8, int(heb.height * rng.uniform(0.8, 1.25)))), Image.BILINEAR)
    r = rng.random()
    if r < 0.15:
        parts = [("hebrew", heb)]
    elif r < 0.30:
        parts = [("english", en)]
    elif r < 0.65:
        parts = [("hebrew", heb), ("english", en)]
    else:
        parts = [("english", en), ("hebrew", heb)]
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


def chunks(heb, en, n_lines, rng):
    out = []
    for _ in range(n_lines):
        line, spans = compose(rng.choice(heb), rng.choice(en), rng)
        bands = segment_lines(line)
        if not bands:
            continue
        band = line.crop((0, bands[0][0], line.width, bands[-1][1]))
        for cx0, cx1 in segment_chunks(band):
            cover = {"hebrew": 0.0, "english": 0.0}
            for k, a, b in spans:
                cover[k] += max(0, min(cx1, b) - max(cx0, a))
            main = max(cover, key=cover.get)
            other = "english" if main == "hebrew" else "hebrew"
            if cover[main] < 0.6 * (cx1 - cx0) or cover[other] > 0.15 * (cx1 - cx0):
                continue
            out.append((band.crop((cx0, 0, cx1, band.height)), 1 if main == "english" else 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iam", default=os.path.join(HERE, "data", "english_lines"), help="folder with train/validation parquet")
    ap.add_argument("--train-dir", default=os.path.join(HERE, "data", "train"))
    ap.add_argument("--val-parquet", default=os.path.join(HERE, "data", "val", "matan_test.parquet"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "english-classifier"))
    ap.add_argument("--lines", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    rng = random.Random(0); random.seed(0); torch.manual_seed(0)
    t0 = time.time()
    parquets = sorted(os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith(".parquet"))
    heb_tr = hebrew_source_parquets(parquets, 2500, shares={"matan": 0.7, "human": 0.3})
    heb_va = hebrew_source_parquets([args.val_parquet], 600, eval_writers())
    en_tr = english_sources(os.path.join(args.iam, "iam_train.parquet"), 4000)
    en_va = english_sources(os.path.join(args.iam, "iam_validation.parquet"), 600, seed=1)
    val = [(to_x(c), y) for c, y in chunks(heb_va, en_va, 1500, random.Random(123))]
    print(f"sources: hebrew {len(heb_tr)}/{len(heb_va)}  english {len(en_tr)}/{len(en_va)}  val chunks {len(val)} "
          f"(english {sum(y for _, y in val)})  ({time.time() - t0:.0f}s)", flush=True)
    model = ScriptCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1.0
    os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        train = [(to_x(c), y) for c, y in chunks(heb_tr, en_tr, args.lines, rng)]
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
        print(f"epoch {ep}  train chunks {len(train)}  val acc {acc:.4f}  calibrated {cal_acc:.4f}  "
              f"hebrew->english {conf[0, 1]}/{conf[0].sum()}  english->hebrew {conf[1, 0]}/{conf[1].sum()}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        torch.onnx.export(model, (torch.zeros(1, 1, NORM_H, CANVAS_W),), os.path.join(args.out, f"model_ep{ep}.onnx"),
                          input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}},
                          opset_version=17, dynamo=False)
        model.train()
        if cal_acc > best:
            best = cal_acc
            shutil.copy(os.path.join(args.out, f"model_ep{ep}.onnx"), os.path.join(args.out, "model.onnx"))
            json.dump({"classes": ["hebrew", "english"], "norm_height": NORM_H, "canvas_width": CANVAS_W,
                       "crop_mode": "band", "val_acc_raw": round(float(acc), 4), "val_acc_calibrated": round(cal_acc, 4),
                       "val_confusion_raw": conf.tolist(), "epoch": ep, "platt_a": a, "platt_b": b,
                       "calibration": "Platt on val chunks: Hebrew test-split writers (minus eval_mixed's), IAM validation"},
                      open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"best calibrated val acc {best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
