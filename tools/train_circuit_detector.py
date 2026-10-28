"""Train the circuit detector: RT-DETRv2-R18 (Apache-2.0; COCO-pretrained weights PekingU/rtdetr_v2_r18vd, safetensors)
fine-tuned on data/circuits (tools/circuit_data.py: CGHD + Digitize-HCD) to find the components, value labels,
grounds and wire crossovers of hand-drawn circuits.

    python download_models.py rtdetr-v2-r18
    python tools/train_circuit_detector.py --minutes 240          # trains on the GPU, keeps the best on validation
    python tools/train_circuit_detector.py --eval-only            # measures models/circuit-detector again

The best model on the validation drafters goes to models/circuit-detector; at the end it is measured on the test
drafters, whose handwriting was never seen in training: AP50 per class (area under precision-recall, a box right
when it overlaps the true one by half), recall and precision at confidence 0.5 -> results/circuit_detector_eval.json.
"""
import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "data", "circuits"))
ap.add_argument("--base", default=os.path.join(HERE, "models", "rtdetr-v2-r18"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "circuit-detector"))
ap.add_argument("--size", type=int, default=1024, help="the network's input (square), pixels")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--minutes", type=float, default=240, help="stop after this long (the best model so far is kept)")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="at most this many images per split (a quick try)")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--device", default=None)
a = ap.parse_args()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import tv_tensors  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection  # noqa: E402

DEV = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
LOG = open(os.path.join(HERE, "results", "circuit_detector_train.log"), "a", encoding="utf-8") if os.path.isdir(os.path.join(HERE, "results")) else None


def say(*xs):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in xs)
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


class Circuits(torch.utils.data.Dataset):
    def __init__(self, split, train):
        d = json.load(open(os.path.join(a.data, f"{split}.json"), encoding="utf-8"))
        self.cats = d["categories"]
        self.images = d["images"][: a.limit or None]
        keep = {im["id"] for im in self.images}
        self.anns = {}
        for an in d["annotations"]:
            if an["image_id"] in keep:
                self.anns.setdefault(an["image_id"], []).append(an)
        self.train = train
        self.aug = T.Compose([
            T.RandomPhotometricDistort(p=0.5),
            T.RandomApply([T.RandomAffine(degrees=4, translate=(0.03, 0.03), scale=(0.8, 1.1), shear=3)], p=0.7),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ClampBoundingBoxes(), T.SanitizeBoundingBoxes(min_size=2)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(os.path.join(a.data, im["file_name"])).convert("RGB")
        anns = self.anns.get(im["id"], [])
        boxes = [[x, y, x + w, y + h] for x, y, w, h in (an["bbox"] for an in anns)]
        labels = [an["category_id"] - 1 for an in anns]
        if self.train and boxes:
            bb = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img.height, img.width))
            img, out = self.aug(img, {"boxes": bb, "labels": torch.tensor(labels)})
            boxes, labels = out["boxes"].tolist(), out["labels"].tolist()
        target = {"image_id": im["id"], "annotations": [
            {"bbox": [x0, y0, x1 - x0, y1 - y0], "category_id": c, "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}
            for (x0, y0, x1, y1), c in zip(boxes, labels)]}
        return img, target, (img.height, img.width)


_PROC = None
CLASSES = []


def proc():
    """the image processor (also in the data-loading worker processes, which do not run main())"""
    global _PROC
    if _PROC is None:
        _PROC = RTDetrImageProcessor.from_pretrained(a.out if a.eval_only else a.base, size={"height": a.size, "width": a.size})
    return _PROC


def collate(batch):
    imgs, targets, sizes = zip(*batch)
    enc = proc()(images=list(imgs), annotations=list(targets), return_tensors="pt")
    return enc["pixel_values"], enc["labels"], list(targets), list(sizes)


def loader(ds, shuffle, workers):
    kw = dict(num_workers=workers, collate_fn=collate)
    if workers:                                  # the same on Linux as on Windows (spawn), workers kept alive
        kw.update(multiprocessing_context="spawn", persistent_workers=True)
    return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=shuffle, drop_last=shuffle, **kw)


def ap50(dets, gts, n_classes, iou=0.5):
    """dets: [(img, cls, score, box)], gts: {img: [(cls, box)]} -> per class AP (all-point), recall/precision @0.5 conf"""
    def overlap(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / u if u > 0 else 0.0
    out = {}
    for c in range(n_classes):
        g = {im: [b for k, b in lst if k == c] for im, lst in gts.items()}
        n_gt = sum(len(v) for v in g.values())
        if not n_gt:
            continue
        ds = sorted((d for d in dets if d[1] == c), key=lambda d: -d[2])
        used = {im: [False] * len(v) for im, v in g.items()}
        tp = []
        for im, _, score, box in ds:
            best, bj = 0.0, -1
            for j, gb in enumerate(g.get(im, [])):
                o = overlap(box, gb)
                if o > best and not used[im][j]:
                    best, bj = o, j
            ok = best >= iou
            if ok:
                used[im][bj] = True
            tp.append((score, ok))
        prec, rec, hits, area, last_r = [], [], 0, 0.0, 0.0
        for k, (_, ok) in enumerate(tp, 1):
            hits += ok
            prec.append(hits / k)
            rec.append(hits / n_gt)
        for k in range(len(prec) - 2, -1, -1):              # precision envelope
            prec[k] = max(prec[k], prec[k + 1])
        for p, r in zip(prec, rec):
            area += p * (r - last_r)
            last_r = r
        sure = [ok for s, ok in tp if s >= 0.5]
        out[CLASSES[c]] = {"AP50": round(area, 4), "true": n_gt, "recall@0.5": round(sum(sure) / n_gt, 4),
                           "precision@0.5": round(sum(sure) / max(1, len(sure)), 4)}
    return out


@torch.no_grad()
def evaluate(model, ds):
    model.eval()
    dl = loader(ds, False, min(2, a.workers))
    dets, gts = [], {}
    for pix, _, targets, sizes in dl:
        with torch.autocast(DEV.split(":")[0], dtype=torch.float16, enabled=DEV.startswith("cuda")):
            out = model(pixel_values=pix.to(DEV))
        res = proc().post_process_object_detection(out, threshold=0.05, target_sizes=sizes)
        for t, r in zip(targets, res):
            gts[t["image_id"]] = [(an["category_id"], [an["bbox"][0], an["bbox"][1], an["bbox"][0] + an["bbox"][2],
                                                        an["bbox"][1] + an["bbox"][3]]) for an in t["annotations"]]
            for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
                dets.append((t["image_id"], l, s, b))
    per = ap50(dets, gts, len(CLASSES))
    m = sum(v["AP50"] for v in per.values()) / max(1, len(per))
    model.train()
    return m, per


def main():
    global CLASSES
    train_ds = Circuits("train", True)
    CLASSES = [c["name"] for c in train_ds.cats]
    id2label = dict(enumerate(CLASSES))
    src = a.out if a.eval_only else a.base
    model = RTDetrV2ForObjectDetection.from_pretrained(src, num_labels=len(CLASSES), id2label=id2label,
                                                       label2id={v: k for k, v in id2label.items()},
                                                       ignore_mismatched_sizes=True).to(DEV)
    val_ds, test_ds = Circuits("val", False), Circuits("test", False)
    say(f"device {DEV}, {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test images, {len(CLASSES)} classes, input {a.size}px")
    best = -1.0
    if not a.eval_only:
        backbone = [p for n, p in model.named_parameters() if "backbone" in n]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n]
        opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr / 10}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
        dl = loader(train_ds, True, a.workers)
        steps = a.epochs * len(dl)
        warm = min(500, steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
        scaler = torch.amp.GradScaler(enabled=DEV.startswith("cuda"))
        t0, step = time.time(), 0
        model.train()
        for ep in range(a.epochs):
            run = 0.0
            for k, (pix, labels, _, _) in enumerate(dl, 1):
                labels = [{kk: vv.to(DEV) for kk, vv in lab.items()} for lab in labels]
                with torch.autocast(DEV.split(":")[0], dtype=torch.float16, enabled=DEV.startswith("cuda")):
                    loss = model(pixel_values=pix.to(DEV), labels=labels).loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                run += loss.detach().item()
                if k % 100 == 0:
                    say(f"  epoch {ep + 1} step {k}/{len(dl)} loss {run / 100:.3f}")
                    run = 0.0
                if (time.time() - t0) / 60 > a.minutes:
                    break
            m, per = evaluate(model, val_ds)
            say(f"epoch {ep + 1}: validation AP50 {m:.3f} ({(time.time() - t0) / 60:.0f} min)")
            if m > best:
                best = m
                model.save_pretrained(a.out)
                proc().save_pretrained(a.out)
                say(f"  saved (best so far) -> {os.path.relpath(a.out, HERE)}")
            if (time.time() - t0) / 60 > a.minutes:
                say("time limit reached")
                break
        model = RTDetrV2ForObjectDetection.from_pretrained(a.out).to(DEV)
    m, per = evaluate(model, test_ds)
    sources = sorted({im.get("source", "?") for im in test_ds.images})
    by_src = {}
    for s in sources:                                       # each collection on its own
        sub = Circuits("test", False)
        sub.images = [im for im in sub.images if im.get("source") == s]
        by_src[s] = evaluate(model, sub)[0]
    rep = {"test_AP50_mean": round(m, 4), "test_AP50_by_source": {k: round(v, 4) for k, v in by_src.items()},
           "best_val_AP50": round(best, 4), "per_class": per, "input": a.size,
           "images": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}}
    json.dump(rep, open(os.path.join(HERE, "results", "circuit_detector_eval.json"), "w", encoding="utf-8"), indent=1)
    say(f"TEST (writers never seen): AP50 {m:.3f}  by collection {rep['test_AP50_by_source']}")
    for c, v in sorted(per.items(), key=lambda kv: -kv[1]["true"]):
        say(f"  {c:14s} AP50 {v['AP50']:.3f}  recall {v['recall@0.5']:.3f}  precision {v['precision@0.5']:.3f}  ({v['true']} true)")


if __name__ == "__main__":
    main()
