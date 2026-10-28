"""Train the compact Hebrew handwriting line recognizer (CNN + BiLSTM + CTC) and export it to ONNX.

Data (public, downloaded by prepare_data.py into data/train/):
  * cyttic/trocr-hebrew-matan  train split - 4,740 lines, 437 writers (test split = 109 other writers)
  * cyttic/trocr-hebrew-human  train split - 4,775 lines (scanned forms, coloured ink)
Validation: 5% of the matan *writers*, held out (writer-independent).

    python train_hebrew.py                      # GPU if available, otherwise CPU
    python train_hebrew.py --minutes 30         # stop after 30 minutes (keeps the best checkpoint)

Output: models/hebrew-crnn/{model.onnx, model.pt, meta.json}
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recog.hebrew_text import CHARSET, clean, decode, encode, to_visual  # noqa: E402

H = 48          # input height
MAXW = 1600     # cap on input width


# ------------------------------------------------------------------ image preprocessing
def load_gray(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("L")
    return ImageOps.autocontrast(img, cutoff=1)


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    if rng.random() < 0.7:                                    # shear / slant
        s = rng.uniform(-0.35, 0.35)
        img = img.transform((w + int(abs(s) * h), h), Image.AFFINE,
                            (1, s, -s * h if s > 0 else 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, expand=False, fillcolor=255)
    r = rng.random()
    if r < 0.2:
        img = img.filter(ImageFilter.MinFilter(3))           # thicker strokes
    elif r < 0.35:
        img = img.filter(ImageFilter.MaxFilter(3))           # thinner strokes
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return img


def to_tensor(img: Image.Image, stretch: float = 1.0) -> torch.Tensor:
    w, h = img.size
    nw = max(16, min(MAXW, int(round(w * H / h * stretch))))
    img = img.resize((nw, H), Image.BILINEAR)
    x = 1.0 - np.asarray(img, dtype=np.float32) / 255.0        # ink -> 1, paper -> 0
    return torch.from_numpy(x)[None]


class Lines(Dataset):
    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        data, target = self.items[i]
        img = load_gray(data)
        stretch = 1.0
        if self.train:
            rng = random.Random()
            img = augment(img, rng)
            stretch = rng.uniform(0.8, 1.2)
        return to_tensor(img, stretch), torch.tensor(encode(target), dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    wmax = max(x.shape[-1] for x in xs)
    wmax = (wmax + 3) // 4 * 4
    X = torch.zeros(len(xs), 1, H, wmax)
    for i, x in enumerate(xs):
        X[i, :, :, : x.shape[-1]] = x
    in_lens = torch.tensor([x.shape[-1] // 4 for x in xs], dtype=torch.long)
    tgt_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return X, torch.cat(ys), in_lens, tgt_lens


# ------------------------------------------------------------------ model
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            block(1, 32), nn.MaxPool2d(2, 2),                              # 24 x W/2
            block(32, 64), nn.MaxPool2d(2, 2),                             # 12 x W/4
            block(64, 128), block(128, 128), nn.MaxPool2d((2, 1)),         # 6
            block(128, 192), block(192, 192), nn.MaxPool2d((2, 1)),        # 3
            nn.Dropout2d(0.1),
            nn.Conv2d(192, 256, (3, 1), bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),   # 1
        )
        self.rnn = nn.LSTM(256, 192, num_layers=2, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(384, n_classes)

    def forward(self, x):                      # x: [B, 1, H, W]
        f = self.cnn(x).squeeze(2)             # [B, C, T]
        f = f.permute(2, 0, 1)                 # [T, B, C]
        f, _ = self.rnn(f)
        return self.fc(f).log_softmax(-1)      # [T, B, K]


# ------------------------------------------------------------------ helpers
def greedy(logp: torch.Tensor) -> list[str]:
    best = logp.argmax(-1).T.cpu().numpy()     # [B, T]
    out = []
    for row in best:
        ids, prev = [], 0
        for k in row:
            if k != prev and k != 0:
                ids.append(k)
            prev = k
        out.append(decode(ids))
    return out


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def load_items(paths, val_frac=0.05, seed=0):
    train, val = [], []
    for p in paths:
        df = pd.read_parquet(p)
        val_writers = set()
        if "writer" in df.columns:
            ws = sorted(df["writer"].unique())
            val_writers = set(random.Random(seed).sample(ws, max(1, int(len(ws) * val_frac))))
        for _, r in df.iterrows():
            t = clean(r["text"])
            if not t:
                continue
            item = (r["image"]["bytes"], to_visual(t))
            (val if r.get("writer") in val_writers else train).append(item)
    return train, val


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = chars = 0
    for X, Y, in_lens, tgt_lens in loader:
        hyp = greedy(model(X.to(device)))
        refs, k = [], 0
        for n in tgt_lens.tolist():
            refs.append(decode(Y[k:k + n].tolist()))
            k += n
        for r, h in zip(refs, hyp):
            errs += lev(r, h)
            chars += len(r)
    model.train()
    return errs / max(1, chars)


def export(model, out_dir, info):
    os.makedirs(out_dir, exist_ok=True)
    model = model.cpu().eval()
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    dummy = torch.zeros(1, 1, H, 400)
    torch.onnx.export(model, (dummy,), os.path.join(out_dir, "model.onnx"),
                      input_names=["image"], output_names=["logp"],
                      dynamic_axes={"image": {0: "batch", 3: "width"}, "logp": {0: "time", 1: "batch"}},
                      opset_version=17, dynamo=False)
    json.dump({"height": H, "max_width": MAXW, "charset": CHARSET, "blank": 0, "time_stride": 4,
               "order": "visual (use recog.hebrew_text.to_logical)", **info},
              open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data", "train"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "hebrew-crnn"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--minutes", type=float, default=0, help="time budget (0 = no limit)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training lines")
    ap.add_argument("--init", default=None, help="start from these weights (model.pt) - continue training / fine-tune on new data")
    ap.add_argument("--extra", action="append", default=[], help="labels.json from label_tool.py (your own scans); repeatable")
    ap.add_argument("--extra-weight", type=int, default=3, help="how many times each of your own lines is repeated per epoch")
    ap.add_argument("--pct-start", type=float, default=0.1, help="warm-up fraction of the one-cycle schedule")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    files = sorted(os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
    train, val = load_items(files)
    for lab in args.extra:                                   # your own corrected lines (label_tool.py)
        base = os.path.dirname(os.path.abspath(lab))
        own = []
        for it in json.load(open(lab, encoding="utf-8")):
            t = clean(it["text"])
            if t:
                own.append((open(os.path.join(base, it["file"]), "rb").read(), to_visual(t)))
        train += own * args.extra_weight
        print(f"+ {len(own)} own lines from {lab} (x{args.extra_weight})", flush=True)
    if args.limit:
        random.Random(0).shuffle(train)
        train, val = train[: args.limit], val[: max(16, args.limit // 10)]
    print(f"device={device}  train lines={len(train)}  val lines={len(val)} (held-out writers)  files={[os.path.basename(f) for f in files]}", flush=True)

    workers = args.workers if args.workers >= 0 else min(4, max(0, (os.cpu_count() or 2) // 2))
    tl = DataLoader(Lines(train, True), batch_size=args.batch, shuffle=True, num_workers=workers,
                    collate_fn=collate, drop_last=True, persistent_workers=workers > 0)
    vl = DataLoader(Lines(val, False), batch_size=32, shuffle=False, num_workers=workers, collate_fn=collate)

    model = CRNN(len(CHARSET) + 1).to(device)
    start_cer = math.inf
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        start_cer = evaluate(model, vl, device)
        print(f"initialised from {args.init}  (val CER before training: {start_cer:.3f}; "
              f"a checkpoint is written only if it beats that)", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=args.pct_start)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    t0, best, step = time.time(), start_cer, 0
    for ep in range(1, args.epochs + 1):
        tot = n = 0
        for X, Y, in_lens, tgt_lens in tl:
            logp = model(X.to(device))
            loss = ctc(logp, Y.to(device), in_lens.to(device), tgt_lens.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            tot += loss.item(); n += 1
        cer = evaluate(model, vl, device)
        mins = (time.time() - t0) / 60
        flag = ""
        if cer < best:
            best = cer
            export(model, args.out, {"val_cer": round(cer, 4), "epoch": ep, "train_lines": len(train)})
            model.to(device)
            flag = "  *saved*"
        print(f"epoch {ep:3d}  loss {tot / max(1, n):.3f}  val CER {cer:.3f}  ({mins:.1f} min){flag}", flush=True)
        if args.minutes and mins > args.minutes:
            print("time budget reached", flush=True)
            break
    print(f"best val CER {best:.3f}  ->  {args.out}", flush=True)


if __name__ == "__main__":
    main()
