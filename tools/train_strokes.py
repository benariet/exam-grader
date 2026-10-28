"""Train the stroke model (recog/strokes.py): a small U-Net that marks pen strokes in photos of hand-drawn circuits,
on CGHD's 356 hand-made stroke maps (Thoma, Bayer et al., CC BY-SA 3.0; the same drafter split as the detector:
test drafters 1 5 10 15 20 25 30, validation 3 13 23 - their hands never seen in training).

    python tools/train_strokes.py --minutes 45          # GPU; saves models/stroke-unet (safetensors)

Measures stroke-pixel F1 on the test drafters (the classical threshold of recog/circuit_wiring.strokes beside it)
-> results/stroke_eval.json.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--cghd", default=os.path.join(HERE, "data", "circuits_raw", "cghd"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "stroke-unet"))
ap.add_argument("--minutes", type=float, default=45)
ap.add_argument("--crop", type=int, default=384)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default=None)
ap.add_argument("--val-every", type=int, default=1000, help="measure on the validation drafters every this many steps")
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter, ImageOps  # noqa: E402

from recog.strokes import LONG_SIDE, UNet, scale_to  # noqa: E402

TEST, VAL = {1, 5, 10, 15, 20, 25, 30}, {3, 13, 23}
DEV = a.device or ("cuda" if torch.cuda.is_available() else "cpu")


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def pairs():
    """(drafter, image path, stroke map path) for every image that has a stroke map"""
    out = []
    for seg in sorted(glob.glob(os.path.join(a.cghd, "drafter_*", "segmentation", "*"))):
        d = int(re.search(r"drafter_(-?\d+)", seg).group(1))
        name = os.path.splitext(os.path.basename(seg))[0]
        imgs = glob.glob(os.path.join(a.cghd, f"drafter_{d}", "images", name + ".*"))
        if imgs:
            out.append((d, imgs[0], seg))
    return out[: a.limit or None]


def load_pair(img_path, seg_path):
    """the photo and its stroke map at the same size and orientation (maps follow the stored or the EXIF-turned
    picture), scaled to LONG_SIDE; the map as 0/1"""
    raw = Image.open(img_path)
    seg = Image.open(seg_path).convert("L")
    turned = ImageOps.exif_transpose(raw)
    img = turned if seg.size == turned.size and seg.size != raw.size else raw
    img = img.convert("RGB")
    if seg.size != img.size:
        seg = seg.resize(img.size, Image.NEAREST)
    img = scale_to(img, LONG_SIDE)
    seg = seg.resize(img.size, Image.BILINEAR)
    return img, (np.asarray(seg) < 128).astype(np.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, n):
        self.items, self.n = items, n
        self.cache = {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rnd = random.Random()
        k = rnd.randrange(len(self.items))
        if k not in self.cache:                          # decoding a phone photo is slow: once per worker
            d, ip, sp = self.items[k]
            img, m = load_pair(ip, sp)
            self.cache[k] = (np.asarray(img, dtype=np.uint8), m > 0.5)
        arr, mb = self.cache[k]
        img, m = Image.fromarray(arr), mb.astype(np.float32)
        s = rnd.uniform(0.6, 1.4)                                     # pen width / photo size vary
        if abs(s - 1) > 0.05:
            img = img.resize((max(a.crop, int(img.width * s)), max(a.crop, int(img.height * s))), Image.BILINEAR)
            m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
        W, H = img.size
        ys, xs = np.where(m > 0.5)
        if len(xs) and rnd.random() < 0.8:                            # most crops on the drawing
            k = rnd.randrange(len(xs))
            cx, cy = xs[k], ys[k]
        else:
            cx, cy = rnd.randrange(W), rnd.randrange(H)
        x0 = int(np.clip(cx - a.crop // 2, 0, max(0, W - a.crop)))
        y0 = int(np.clip(cy - a.crop // 2, 0, max(0, H - a.crop)))
        img = img.crop((x0, y0, x0 + a.crop, y0 + a.crop))
        m = m[y0:y0 + a.crop, x0:x0 + a.crop]
        if m.shape != (a.crop, a.crop):
            m = np.pad(m, ((0, a.crop - m.shape[0]), (0, a.crop - m.shape[1])))
        if rnd.random() < 0.3:
            img = ImageOps.grayscale(img).convert("RGB")
        if rnd.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.2)))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = x * rnd.uniform(0.7, 1.15) + rnd.uniform(-0.12, 0.12)    # light
        x = x * np.array([rnd.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)   # colour cast
        x = np.clip(x + np.random.normal(0, rnd.uniform(0, 0.03), x.shape).astype(np.float32), 0, 1)
        k = rnd.randrange(4)
        x, m = np.rot90(x, k).copy(), np.rot90(m, k).copy()
        if rnd.random() < 0.5:
            x, m = x[:, ::-1].copy(), m[:, ::-1].copy()
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(m))[None]


def f1(pred, true):
    tp = float((pred & true).sum())
    p, t = float(pred.sum()), float(true.sum())
    return 2 * tp / max(1.0, p + t), tp / max(1.0, p), tp / max(1.0, t)


def evaluate(model_dir, items):
    from recog.circuit_wiring import strokes
    from recog.strokes import StrokeModel
    sm = StrokeModel(model_dir, device=DEV)
    rows = []
    for d, ip, sp in items:
        img, m = load_pair(ip, sp)
        true = m > 0.5
        pred = sm(img)
        classical = strokes(img)
        rows.append((f1(pred, true), f1(classical, true)))
    mean = lambda k, j: float(np.mean([r[k][j] for r in rows])) if rows else 0.0  # noqa: E731
    return {"images": len(rows), "F1_learned": round(mean(0, 0), 4), "precision_learned": round(mean(0, 1), 4),
            "recall_learned": round(mean(0, 2), 4), "F1_classical": round(mean(1, 0), 4),
            "precision_classical": round(mean(1, 1), 4), "recall_classical": round(mean(1, 2), 4)}


def main():
    from huggingface_hub import snapshot_download
    if not glob.glob(os.path.join(a.cghd, "drafter_*", "segmentation", "*")):
        say("downloading CGHD's stroke maps (~0.1 GB) ...")
        snapshot_download("lowercaseonly/cghd", repo_type="dataset", local_dir=a.cghd, allow_patterns=["*/segmentation/*"])
    items = pairs()
    train = [x for x in items if x[0] not in TEST | VAL]
    val = [x for x in items if x[0] in VAL]
    test = [x for x in items if x[0] in TEST]
    say(f"{len(train)} train / {len(val)} val / {len(test)} test stroke maps, device {DEV}")
    net = UNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    kw = dict(num_workers=a.workers, multiprocessing_context="spawn", persistent_workers=True) if a.workers else {}
    dl = torch.utils.data.DataLoader(Crops(train, 100000), batch_size=a.batch, **kw)
    total = int(a.minutes * 60)
    scaler = torch.amp.GradScaler(enabled=DEV.startswith("cuda"))
    pos_w = torch.tensor(4.0, device=DEV)
    t0, step, run = time.time(), 0, 0.0
    os.makedirs(a.out, exist_ok=True)
    best = -1.0
    net.train()
    for x, m in dl:
        frac = min(1.0, (time.time() - t0) / total)
        for g in opt.param_groups:
            g["lr"] = a.lr * (min(1.0, (step + 1) / 200)) * 0.5 * (1 + math.cos(math.pi * frac))
        x, m = x.to(DEV), m.to(DEV)
        with torch.autocast(DEV.split(":")[0], dtype=torch.float16, enabled=DEV.startswith("cuda")):
            logit = net(x)
        logit = logit.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, m, pos_weight=pos_w)
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * m).sum() + 1) / (p.sum() + m.sum() + 1)
        loss = bce + dice
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1
        run += loss.detach().item()
        if step % 200 == 0 or frac >= 1.0:
            say(f"  step {step}  loss {run / 200:.3f}  ({(time.time() - t0) / 60:.0f} min)")
            run = 0.0
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
            json.dump({"channels": [16, 32, 64, 128], "long_side": LONG_SIDE, "threshold": 0.5}, open(os.path.join(a.out, "config.json"), "w"))
            if val and (step % a.val_every == 0 or frac >= 1.0):
                r = evaluate(a.out, val)
                say(f"  validation stroke F1 {r['F1_learned']:.3f} (classical {r['F1_classical']:.3f})")
                if r["F1_learned"] > best:
                    best = r["F1_learned"]
                    import shutil
                    shutil.copy(os.path.join(a.out, "model.safetensors"), os.path.join(a.out, "best.safetensors"))
                net.train()
        if frac >= 1.0:
            break
    if os.path.isfile(os.path.join(a.out, "best.safetensors")):
        os.replace(os.path.join(a.out, "best.safetensors"), os.path.join(a.out, "model.safetensors"))
    rep = {"test": evaluate(a.out, test), "validation_best_F1": round(best, 4), "steps": step}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, "results", "stroke_eval.json"), "w"), indent=1)
    say(f"TEST (drafters never seen): stroke F1 learned {rep['test']['F1_learned']:.3f}  classical {rep['test']['F1_classical']:.3f}  {rep['test']}")


if __name__ == "__main__":
    main()
