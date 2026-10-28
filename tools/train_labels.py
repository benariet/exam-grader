"""Train the circuit-label reader (recog/labels.py) on the labels CGHD and Digitize-HCD transcribe - the 'text' boxes
of data/circuits (tools/circuit_data.py), the same split by drafter as the detector (test drafters never seen).

    python tools/train_labels.py --minutes 15          # GPU; saves models/label-reader (safetensors)

Measures exact readings on the test split - all labels, and the names / values a circuits exam has ('R1', 'I_2',
'10kΩ', '5V') - greedy and with the beam search weighed by a character n-gram model of the training labels (saved
beside the model as lm.json, which recog.labels.LabelReader then uses), beside the formula reader with the label
grammar (what recog/circuit_reader.py used before) -> results/label_eval.json.
"""
import argparse
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
ap.add_argument("--data", default=os.path.join(HERE, "data", "circuits"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "label-reader"))
ap.add_argument("--minutes", type=float, default=15)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default=None)
ap.add_argument("--formula-test", type=int, default=300, help="test labels also read by the formula reader (0: skip)")
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from recog.labels import CHARS, CRNN, H, KIND_CODE, CharLM, KindLM, batch, beam_decode, decode, prep  # noqa: E402

DEV = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
EXAM = re.compile(r"^(?:[A-Za-z]{1,2}_?\d{0,3}|\d+(?:[.,]\d+)?\s*[pnuµmkKM]?\s*(?:Ω|V|A|F|H)?)$")


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def clean(t: str):
    """a transcription -> the reader's alphabet, or None (unknown signs, too long)"""
    t = (t or "").replace("\n", " ").replace("μ", "µ").replace("Ω", "Ω").replace("–", "-").strip()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) > 24 or any(c not in CHARS for c in t):
        return None
    return t


def crops(split, limit=0):
    """[(label image: the box plus a margin, as grey, text, source)] of one split, cut once"""
    d = json.load(open(os.path.join(a.data, f"{split}.json"), encoding="utf-8"))
    by_img = {}
    for an in d["annotations"]:
        t = clean(an.get("text"))
        if t:
            by_img.setdefault(an["image_id"], []).append((an["bbox"], t))
    out = []
    for im in d["images"]:
        if im["id"] not in by_img:
            continue
        img = Image.open(os.path.join(a.data, im["file_name"])).convert("L")
        for (x, y, w, h), t in by_img[im["id"]]:
            m = 0.3 * h                                  # room for the margin the training draws from
            box = (max(0, int(x - m)), max(0, int(y - m)), min(img.width, int(x + w + m)), min(img.height, int(y + h + m)))
            out.append((img.crop(box), (x - box[0], y - box[1], x - box[0] + w, y - box[1] + h), t, im.get("source", "?")))
        if limit and len(out) >= limit:
            break
    return out


def kinds(split, limit=0):
    """the kind of part each label of crops(split) stands by (recog.labels.kind_of), in the same order"""
    from recog.labels import kind_of
    d = json.load(open(os.path.join(a.data, f"{split}.json"), encoding="utf-8"))
    names = {c["id"]: c["name"] for c in d.get("categories", [])}
    by_img, parts = {}, {}
    for an in d["annotations"]:
        x, y, w, h = an["bbox"]
        if names.get(an.get("category_id")) == "text":
            if clean(an.get("text")):
                by_img.setdefault(an["image_id"], []).append((x, y, x + w, y + h))
        else:
            parts.setdefault(an["image_id"], []).append((names.get(an.get("category_id"), "other"), (x, y, x + w, y + h)))
    out = []
    for im in d["images"]:
        for box in by_img.get(im["id"], []):
            out.append(kind_of(box, parts.get(im["id"], [])))
        if limit and len(out) >= limit:
            break
    return out


def view(item, rnd=None):
    """the label as the reader sees it: its box plus a margin (random in training), turned / blurred / thickened"""
    img, (x0, y0, x1, y1), _, _ = item
    hgt = y1 - y0
    m = 0.15 * hgt if rnd is None else rnd.uniform(0.0, 0.3) * hgt
    jit = (lambda: 0.0) if rnd is None else (lambda: rnd.uniform(-0.08, 0.08) * hgt)
    c = img.crop((int(x0 - m + jit()), int(y0 - m + jit()), int(x1 + m + jit()), int(y1 + m + jit())))
    if rnd is not None:
        if rnd.random() < 0.5:
            c = c.rotate(rnd.uniform(-4, 4), resample=Image.BILINEAR, expand=True, fillcolor=255)
        u = rnd.random()
        if u < 0.2:
            c = c.filter(ImageFilter.MinFilter(3))          # a thicker pen
        elif u < 0.35:
            c = c.filter(ImageFilter.MaxFilter(3))          # a thinner one
        if rnd.random() < 0.3:
            c = c.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.0)))
        if rnd.random() < 0.3:                              # squeezed / stretched
            c = c.resize((max(4, int(c.width * rnd.uniform(0.75, 1.3))), c.height))
    x = prep(c)
    if rnd is not None and rnd.random() < 0.3:
        x = np.clip(x + np.random.default_rng(rnd.randrange(1 << 30)).normal(0, 0.06, x.shape), 0, 1).astype(np.float32)
    return x


def targets(texts):
    idx = [CHARS.index(c) + 1 for t in texts for c in t]
    return torch.tensor(idx, dtype=torch.long), torch.tensor([len(t) for t in texts], dtype=torch.long)


def read_all(net, items, bs=128, lm=None, lms=None):
    """lm: one character model for all; lms: one for each item (the kind of part it stands by)"""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            arrs = [view(it) for it in items[i:i + bs]]
            logits = net(batch(arrs).to(DEV)).float().cpu()
            widths = [-(-x.shape[1] // 4) for x in arrs]
            if lm is None and lms is None:
                out += decode(logits, CHARS, widths)
            else:
                lp = logits.log_softmax(-1).numpy()
                out += [beam_decode(lp[b, :n], CHARS, lms[i + b] if lms is not None else lm, **LM_CFG)
                        for b, n in enumerate(widths)]
    net.train()
    return out


LM_CFG = {"lam": 0.2, "bonus": 1.0}                 # tuned on the validation drafters (3 13 23) with a 40-minute model


def norm(t):
    return (t or "").replace(" ", "").replace(",", ".").replace("Ω", "").lower()


def score(items, reads):
    ok = [norm(r[0]) == norm(it[2]) for it, r in zip(items, reads)]
    exam = [k for k, it in enumerate(items) if EXAM.match(it[2])]
    by = {}
    for k, it in enumerate(items):
        by.setdefault(it[3], []).append(ok[k])
    return {"all": round(float(np.mean(ok)), 4) if ok else None, "n": len(ok),
            "names_values": round(float(np.mean([ok[k] for k in exam])), 4) if exam else None, "n_names_values": len(exam),
            "by_source": {s: round(float(np.mean(v)), 4) for s, v in by.items()}}


def main():
    from safetensors.torch import save_file
    train, val, test = crops("train"), crops("val"), crops("test")
    say(f"labels: {len(train)} train / {len(val)} val / {len(test)} test, device {DEV}")
    net = CRNN().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    rnd = random.Random(0)
    total, t0, step, best = a.minutes * 60, time.time(), 0, -1.0
    os.makedirs(a.out, exist_ok=True)
    net.train()
    while True:
        frac = min(1.0, (time.time() - t0) / total)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 300) * 0.5 * (1 + math.cos(math.pi * frac))
        items = [train[rnd.randrange(len(train))] for _ in range(a.batch)]
        arrs = [view(it, rnd) for it in items]
        x = batch(arrs).to(DEV)
        logp = net(x).log_softmax(-1).permute(1, 0, 2)               # [T, B, C]
        tg, tl = targets([it[2] for it in items])
        il = torch.tensor([min(logp.shape[0], -(-x_.shape[1] // 4)) for x_ in arrs], dtype=torch.long)
        loss = ctc(logp.float().cpu(), tg, il, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 200 == 0 or frac >= 1.0:
            say(f"  step {step} loss {loss.item():.3f} ({(time.time() - t0) / 60:.0f} min)")
        if step % 1000 == 0 or frac >= 1.0:
            r = score(val, read_all(net, val))
            say(f"  validation: {r}")
            if r["all"] is not None and r["all"] > best:
                best = r["all"]
                save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
                json.dump({"chars": CHARS, "height": H}, open(os.path.join(a.out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False)
        if frac >= 1.0:
            break
    from safetensors.torch import load_file
    net.load_state_dict(load_file(os.path.join(a.out, "model.safetensors")))
    texts = [it[2] for it in train]                      # what labels look like, for the beam search
    ks = kinds("train")                                  # and what they look like by each kind of part
    kind_texts = [KIND_CODE.get(k, KIND_CODE["?"]) + t for t, k in zip(texts, ks)] if len(ks) == len(texts) else []
    json.dump({"n": 4, **LM_CFG, "texts": texts, "kind_n": 5, "kind_texts": kind_texts},
              open(os.path.join(a.out, "lm.json"), "w", encoding="utf-8"), ensure_ascii=False)
    lm = CharLM(texts, 4)
    greedy = read_all(net, test)
    reads = read_all(net, test, lm=lm)
    if kind_texts:                                       # read knowing the part each label stands by
        klm, tks = CharLM(kind_texts, 5), kinds("test")
        if len(tks) == len(test):
            say(f"  test, each label read with its part's kind: {score(test, read_all(net, test, lms=[KindLM(klm, k) for k in tks]))}")
    rep = {"test": score(test, reads), "test_greedy": score(test, greedy), "validation_best": best, "steps": step,
           "examples": [(it[2], r[0]) for it, r in list(zip(test, reads))[:40]]}
    if a.formula_test:                                   # the formula reader with the label grammar, beside it
        from recog.circuit_reader import LABEL_GRAMMAR, label_text
        from recog.formula import FormulaRecognizer
        fr = FormulaRecognizer()
        sub = test[:a.formula_test]
        fr_reads = []
        for it in sub:
            img, (x0, y0, x1, y1), _, _ = it
            m = 0.2 * (y1 - y0)
            c = img.crop((int(x0 - m), int(y0 - m), int(x1 + m), int(y1 + m))).convert("RGB")
            try:
                fr_reads.append((label_text(fr.recognize(c, n_best=1, max_new_tokens=24, grammar=LABEL_GRAMMAR)[0].text), 1.0))
            except Exception:                            # noqa: BLE001
                fr_reads.append(("", 0.0))
        rep["formula_reader_same_labels"] = score(sub, fr_reads)
        rep["label_reader_same_labels"] = score(sub, reads[:len(sub)])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, "results", "label_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say(f"TEST (drafters never seen): {rep['test']} (greedy decoding: {rep['test_greedy']['all']})")
    if a.formula_test:
        say(f"  on {len(sub)} of them: label reader {rep['label_reader_same_labels']['all']}, "
            f"formula reader {rep['formula_reader_same_labels']['all']}")


if __name__ == "__main__":
    main()
