"""Train the gate classifier (recog/gates.py): which logic gate a box holds.

    python tools/train_gates.py --hdl ~/eg-data/hdl/ImagesDataset

Data: HDL (hand-drawn gates of Umm Al-Qura University's students, 150 of each of AND OR NOT NAND NOR XOR XNOR BUFFER,
CC BY-SA 4.0 - every 7th of each kind held out) and the gates boxed in the detector's training images
(data/circuits/train.json: CGHD's training drafters). Each image turned by quarter turns, mirrored, scaled, its
strokes thickened or thinned, its box's margin varied. Measured on HDL's held-out gates and on the gates boxed in the
detector's test images (CGHD's test drafters - never seen) -> models/gates/gates.safetensors, results/gates_eval.json
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--hdl", default=os.path.expanduser("~/eg-data/hdl/ImagesDataset"))
ap.add_argument("--data", default=os.path.join(HERE, "data", "circuits"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "gates"))
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--device", default=None)
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from recog.gates import KINDS, SIZE, ink, net, square  # noqa: E402

DEV = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
MARGIN = 0.35                       # a CGHD box is kept with this much around it; a training sample takes part of it


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def hdl_items():
    tr, te = [], []
    for kind in KINDS:
        folder = os.path.join(a.hdl, kind.upper())
        if not os.path.isdir(folder):                      # (op-amps: CGHD's boxes only)
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for i, f in enumerate(files):
            (te if i % 7 == 0 else tr).append(("hdl", os.path.join(folder, f), None, KINDS.index(kind)))
    return tr, te


def cghd_items(split):
    d = json.load(open(os.path.join(a.data, f"{split}.json"), encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    files = {im["id"]: im["file_name"] for im in d["images"]}
    out = []
    for an in d["annotations"]:
        name = cats[an["category_id"]]
        if name in KINDS:
            out.append(("cghd", os.path.join(a.data, files[an["image_id"]]), tuple(an["bbox"]), KINDS.index(name)))
    return out


def load(item, rnd=None):
    """an item's image as the classifier sees it (rnd: turned, mirrored, scaled, thickened - for training)"""
    src, path, bbox, y = item
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if bbox is not None:                                   # a box: some margin around it (training: a random part)
        x, yy, w, h = bbox
        m = (rnd.uniform(0.0, MARGIN) if rnd else 0.15)
        img = img.crop((int(x - m * w), int(yy - m * h), int(x + w + m * w), int(yy + h + m * h)))
    if max(img.size) > 400:
        img.thumbnail((400, 400))
    if rnd:
        k = rnd.randrange(4)
        if k:
            img = img.rotate(90 * k, expand=True, fillcolor=255)
        if rnd.random() < 0.5:
            img = ImageOps.mirror(img)
        s = rnd.uniform(0.6, 1.4)
        img = img.resize((max(8, int(img.width * s)), max(8, int(img.height * s))))
        if rnd.random() < 0.5:
            img = img.rotate(rnd.uniform(-8, 8), expand=True, fillcolor=255)
    mask = ink(img)
    if rnd and rnd.random() < 0.4:
        from scipy import ndimage as ndi
        if rnd.random() < 0.7:
            mask = ndi.binary_dilation(mask)
        else:
            er = ndi.binary_erosion(mask)
            mask = er if er.sum() > 0.3 * mask.sum() else mask
    return square(mask, SIZE), y


class Data(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rnd = random.Random(random.getrandbits(32)) if self.train else None
        x, y = load(self.items[i], rnd)
        return torch.from_numpy(x)[None], y


def accuracy(model, items):
    dl = torch.utils.data.DataLoader(Data(items, False), batch_size=128, num_workers=a.workers)
    right = np.zeros(len(KINDS)), np.zeros(len(KINDS))
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(DEV)).argmax(-1).cpu().numpy()
            for t, q in zip(y.numpy(), p):
                right[1][t] += 1
                right[0][t] += q == t
    model.train()
    per = {KINDS[k]: f"{int(right[0][k])}/{int(right[1][k])}" for k in range(len(KINDS)) if right[1][k]}
    return round(float(right[0].sum() / max(1, right[1].sum())), 4), per


def main():
    from safetensors.torch import save_file
    torch.multiprocessing.set_start_method("spawn", force=True)
    hdl_tr, hdl_te = hdl_items()
    cg_tr, cg_te = cghd_items("train"), cghd_items("test")
    train = hdl_tr + cg_tr
    say(f"train: {len(hdl_tr)} HDL + {len(cg_tr)} CGHD boxes; test: {len(hdl_te)} HDL, {len(cg_te)} CGHD test-drafter boxes")
    counts = np.bincount([it[3] for it in train], minlength=len(KINDS))
    weights = [1.0 / counts[it[3]] for it in train]          # (every kind as often: XOR / NOR are few in CGHD)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    dl = torch.utils.data.DataLoader(Data(train, True), batch_size=64, sampler=sampler, num_workers=a.workers,
                                     persistent_workers=a.workers > 0)
    model = net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * len(dl))
    for ep in range(a.epochs):
        tot = 0.0
        for x, y in dl:
            loss = torch.nn.functional.cross_entropy(model(x.to(DEV)), y.to(DEV), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            say(f"epoch {ep + 1}: loss {tot / len(dl):.3f}, HDL held out {accuracy(model, hdl_te)[0]}")
    hdl_acc, hdl_per = accuracy(model, hdl_te)
    cg_acc, cg_per = accuracy(model, cg_te)
    rep = {"hdl_held_out": hdl_acc, "hdl_by_kind": hdl_per, "cghd_test_boxes": cg_acc, "cghd_by_kind": cg_per,
           "train": {"hdl": len(hdl_tr), "cghd": len(cg_tr)}}
    say(rep)
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, os.path.join(a.out, "gates.safetensors"))
    json.dump({"kinds": KINDS, "size": SIZE}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, "results", "gates_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
