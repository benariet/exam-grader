"""Teach Mishkefet-v1 (recog/mishkefet.py) a hand: lines of handwriting with their text known - the writing forms
(tools/writing_forms.py cut -> labels.jsonl) - its CTC model fine-tuned on them, the best checkpoint by the held-out
lines' character error rate kept:

    python tools/finetune_mishkefet.py data/handwriting/tal/labels.jsonl [more.jsonl ...] --out models/mishkefet-tal.pt
        [--epochs 40] [--lr 5e-5] [--val 0.15] [--kinds hebrew] [--val-labels held_out.jsonl]
        [--mix other_lines.jsonl --mix-share 0.3]

The held-out lines (a fixed share, by the line's text and image name) are read before training and after each epoch;
the checkpoint (the original's format: model, config, charset) is saved only when they are read better. The
recognizer reads it as the original (recog/mishkefet.recognizer(checkpoint=...)). --val-labels: these lines held out
instead (another writer's page, lines transcribed by hand), all the others learned. --mix: lines of other hands
added to each epoch (a share of the epoch's lines, drawn anew each epoch) - so a model taught one writer keeps the
others it read (one writer's thousands of lines alone would pull it to that hand)."""
import argparse
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def lines(paths: list, kinds: set) -> list:
    """[(image path, text)] of the lines of these kinds"""
    out = []
    for p in paths:
        base = os.path.dirname(os.path.abspath(p))
        for row in open(p, encoding="utf-8"):
            r = json.loads(row)
            if r.get("kind", "hebrew") in kinds and r.get("text", "").strip():
                out.append((os.path.join(base, r["image"]), r["text"]))
    return out


def held_out(item, share: float) -> bool:
    h = int(hashlib.md5((item[0] + item[1]).encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < share


def cer(pred: list, gold: list) -> float:
    from rapidfuzz.distance import Levenshtein
    errs = sum(Levenshtein.distance(p, g) for p, g in zip(pred, gold))
    return errs / max(1, sum(len(g) for g in gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--kinds", default="hebrew")
    ap.add_argument("--aug", type=float, default=0.6, help="augmentation strength (hebocr's Augment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-labels", nargs="*", default=[])
    ap.add_argument("--mix", nargs="*", default=[])
    ap.add_argument("--mix-share", type=float, default=0.3)
    ap.add_argument("--keep", choices=["best", "last"], default="best",
                    help="last: the checkpoint after the last epoch (when the held-out lines are the test, not a guide)")
    args = ap.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from recog import mishkefet
    mishkefet._import()
    from hebocr.charset import BLANK
    from hebocr.data.transforms import Augment

    random.seed(args.seed)
    data = lines(args.labels, set(args.kinds.split(",")))
    if args.val_labels:
        val = lines(args.val_labels, set(args.kinds.split(",")))
        held = {p for p, _ in val}
        train = [d for d in data if d[0] not in held]
    else:
        val = [d for d in data if held_out(d, args.val)]
        train = [d for d in data if not held_out(d, args.val)]
    mix = lines(args.mix, set(args.kinds.split(","))) if args.mix else []
    print(f"{len(train)} lines to learn, {len(val)} held out, {len(mix)} of other hands to mix in", flush=True)
    data = data + [d for d in val if d not in data] + mix
    if not train or not val:
        sys.exit("too few lines")
    rec = mishkefet.recognizer(page=False, beam_width=0)          # (greedy: the held-out reading is the model's own)
    base = mishkefet.load_checkpoint(os.path.join(mishkefet.WEIGHTS, "mishkefet-v1.pt"))
    model, charset, dev = rec.model, rec.charset, rec.device
    miss = sorted({c for _, t in data for c in t if charset.coverage(c) < 1})
    if miss:
        print(f"characters the model does not have (left out of the targets): {''.join(miss)}", flush=True)
    imgs = {p: Image.open(p).convert("L") for p, _ in data}

    def evaluate():
        model.eval()
        pred = rec.read([imgs[p] for p, _ in val])
        return cer(pred, [t for _, t in val]), pred

    best, _ = evaluate()
    print(f"held-out CER before: {best:.3f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    aug = Augment(np.random.default_rng(args.seed), strength=args.aug)
    for ep in range(1, args.epochs + 1):
        model.train()
        epoch = train + (random.sample(mix, min(len(mix), int(len(train) * args.mix_share / (1 - args.mix_share))))
                         if mix else [])
        random.shuffle(epoch)
        t0, total, n = time.time(), 0.0, 0
        for k in range(0, len(epoch), args.batch):
            chunk = epoch[k:k + args.batch]
            arrays = [torch.from_numpy(aug(imgs[p])) for p, _ in chunk]   # (Augment: the preprocessed array)
            widths = torch.tensor([a.shape[-1] for a in arrays])
            x = torch.zeros(len(arrays), 1, arrays[0].shape[-2], int(widths.max()))
            for j, a in enumerate(arrays):
                x[j, :, :, :a.shape[-1]] = a
            targets = [torch.tensor(charset.encode(t), dtype=torch.long) for _, t in chunk]
            tl = torch.tensor([len(t) for t in targets])
            lp = model(x.to(dev))
            il = model.output_lengths(widths)
            keep = (il >= tl) & (tl > 0)
            if not keep.any():
                continue
            loss = F.ctc_loss(lp[keep].permute(1, 0, 2).float(), torch.cat([t for t, k_ in zip(targets, keep) if k_]),
                              il[keep].to(dev), tl[keep].to(dev), blank=BLANK, reduction="mean", zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total, n = total + float(loss), n + 1
        score, pred = evaluate()
        mark = ""
        if score < best or (args.keep == "last" and ep == args.epochs):
            best, mark = min(best, score), "  <- saved"
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({"model": model.state_dict(), "config": base.get("config", {}), "charset": charset.chars,
                        "epoch": ep, "step": None, "finetuned_on": args.labels, "held_out_cer": best}, args.out)
        print(f"epoch {ep}: loss {total / max(1, n):.3f}, held-out CER {score:.3f} ({time.time() - t0:.0f} s){mark}",
              flush=True)
    print(f"best held-out CER {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
