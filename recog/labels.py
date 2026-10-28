"""The labels written in circuit drawings - names ('R1', 'E_2', 'I3', 'A') and values ('10kΩ', '5V', '4.7µF') - read
by a small CRNN (convolutions + a bidirectional LSTM, CTC) trained on the labels CGHD and Digitize-HCD transcribe
(tools/train_labels.py). The formula reader, made for whole formulas, gets 35-45 % of them exactly right. With the
training labels beside the model (lm.json), the reading is a CTC beam search weighed by a character n-gram model of
what labels look like ('10kΩ' rather than '1okΩ'): on drafters never seen, 62 % exactly right against greedy
decoding's 58 % (the same network); each label read knowing the kind of part it stands by (lm.json's kind_texts:
'10µF' by a capacitor, 'R1' by a resistor) another +0.9 points.

    from recog.labels import LabelReader
    lr = LabelReader()                       # models/label-reader
    lr.read([crop, ...])                     # [(text, how sure), ...]; crop: PIL image of one label
    lr.read(crops, kinds=[kind_of(box, parts), ...])     # each read as by the part nearest it
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from torch import nn

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "label-reader")
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-/%Ωµ_()=~ "
H = 32                                   # the crop's height going in
MAX_W = 320


class CRNN(nn.Module):
    def __init__(self, n_out: int = len(CHARS) + 1):
        super().__init__()

        def block(a, b, pool):
            return [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)] + \
                   ([nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*block(1, 32, (2, 2)), *block(32, 64, (2, 2)), *block(64, 128, None),
                                 *block(128, 128, (2, 1)), *block(128, 192, None), *block(192, 192, (2, 1)),
                                 *block(192, 256, (2, 1)))           # height 32 -> 1, width / 4
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(256, n_out)

    def forward(self, x):                                            # x: [B, 1, H, W] -> [B, W/4, classes]
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        return self.out(self.rnn(f)[0])


def prep(img, h: int = H, max_w: int = MAX_W) -> np.ndarray:
    """a label's crop -> [h, w] float, ink 1 on paper 0 (light evened out by the crop's own paper level)"""
    from PIL import Image
    g = np.asarray(img.convert("L"), dtype=np.float32)
    paper = np.percentile(g, 90) if g.size else 255.0
    ink = np.clip((paper - g) / max(30.0, paper - np.percentile(g, 2)), 0, 1)
    im = Image.fromarray((ink * 255).astype(np.uint8))
    w = max(8, min(max_w, round(im.width * h / max(1, im.height))))
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def batch(arrs):
    """[h, w] arrays -> [B, 1, h, W] tensor, padded on the right to a common width (a multiple of 4)"""
    w = max(a.shape[1] for a in arrs)
    w += (-w) % 4
    x = np.zeros((len(arrs), 1, arrs[0].shape[0], w), np.float32)
    for k, a in enumerate(arrs):
        x[k, 0, :, :a.shape[1]] = a
    return torch.from_numpy(x)


def decode(logits: torch.Tensor, chars: str = CHARS, widths=None):
    """CTC greedy decoding -> [(text, how sure: the probability of that best path)]; widths: each item's own
    number of frames (the rest is padding)"""
    lp = logits.detach().float().log_softmax(-1)
    best, idx = lp.max(-1)
    out = []
    for b in range(idx.shape[0]):
        n = idx.shape[1] if widths is None else int(widths[b])
        text, prev = [], 0
        for t in range(n):
            k = int(idx[b, t])
            if k != prev and k != 0:
                text.append(chars[k - 1])
            prev = k
        out.append(("".join(text).strip(), float(np.exp(float(best[b, :n].sum())))))
    return out


class CharLM:
    """what labels look like: a character n-gram model (Witten-Bell interpolation) of label texts; '^' pads the
    start, '$' is the end"""

    def __init__(self, texts, n: int = 4):
        from collections import defaultdict
        self.n = n
        counts = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
        vocab = {"$"}
        for t in texts:
            s = "^" * (n - 1) + t + "$"
            vocab |= set(t)
            for i in range(n - 1, len(s)):
                for k in range(n):                                   # k characters of context
                    counts[k][s[i - k:i]][s[i]] += 1
        self.V = len(vocab) + 1
        self.counts = [{c: dict(d) for c, d in ck.items()} for ck in counts]
        self.tot = [{c: sum(d.values()) for c, d in ck.items()} for ck in self.counts]
        self.cache = {}

    def logp(self, ctx: str, ch: str) -> float:
        ctx = ("^" * (self.n - 1) + ctx)[-(self.n - 1):] if self.n > 1 else ""
        key = (ctx, ch)
        if key not in self.cache:
            p = 1.0 / self.V
            for k in range(self.n):
                c = ctx[len(ctx) - k:] if k else ""
                tot = self.tot[k].get(c, 0)
                if tot:
                    seen = self.counts[k][c]
                    lam = tot / (tot + len(seen))
                    p = lam * seen.get(ch, 0) / tot + (1 - lam) * p
            self.cache[key] = math.log(p)
        return self.cache[key]


def _lse(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(lp: np.ndarray, chars: str = CHARS, lm: CharLM | None = None, lam: float = 0.2, bonus: float = 1.0,
                beam: int = 8, topk: int = 6):
    """CTC prefix beam search over one item's log-probabilities [frames, classes] (blank first), each character
    weighed by lam * log P_lm (and the end) plus a bonus -> (text, how sure: the probability of that text, all its
    alignments summed)"""
    beams = {"": (0.0, -math.inf)}                            # prefix -> (log p ending in a blank, in a character)
    for t in range(lp.shape[0]):
        row = lp[t]
        nxt = {}
        for pre, (pb, pn) in beams.items():
            tot = _lse(pb, pn)
            e = nxt.setdefault(pre, [-math.inf, -math.inf])
            e[0] = _lse(e[0], tot + row[0])
            for k in np.argsort(row)[::-1][:topk]:
                if k == 0:
                    continue
                ch = chars[k - 1]
                w = (lam * lm.logp(pre, ch) if lm else 0.0) + bonus
                ext = nxt.setdefault(pre + ch, [-math.inf, -math.inf])
                if pre and pre[-1] == ch:                     # the same character again: only after a blank
                    e[1] = _lse(e[1], pn + row[k])
                    ext[1] = _lse(ext[1], pb + row[k] + w)
                else:
                    ext[1] = _lse(ext[1], tot + row[k] + w)
        beams = dict(sorted(((p, tuple(v)) for p, v in nxt.items()), key=lambda x: -_lse(*x[1]))[:beam])

    def final(item):
        p, (pb, pn) = item
        return _lse(pb, pn) + (lam * lm.logp(p, "$") if lm else 0.0)
    best, (pb, pn) = max(beams.items(), key=final)
    weights = sum(((lam * lm.logp(best[:i], ch)) if lm else 0.0) + bonus for i, ch in enumerate(best))
    return best.strip(), float(math.exp(min(0.0, _lse(pb, pn) - weights)))


# the kind of part a label stands by, as a character the n-gram model's context can start with: 'R1' / '10kΩ' by a
# resistor, '4.7µF' by a capacitor, '5V' / 'E1' by a voltage source (control characters, not in CHARS)
PART_KIND = {"resistor": "R", "capacitor": "C", "inductor": "L", "voltage_dc": "V", "voltage_rail": "V",
             "voltage_ac": "V", "dep_voltage": "V", "current_dc": "I", "current_ac": "I", "dep_current": "I",
             "diode": "D", "zener": "D", "thyristor": "D", "bjt": "Q", "fet": "Q", "opamp": "O", "gnd": "G",
             "terminal": "T", "switch": "S", "transformer": "L"}
KIND_CODE = {k: chr(1 + i) for i, k in enumerate("CDGILOQRSTVX?")}       # X: any other part, ?: none near


def kind_of(text_box, parts, far: float = 2.5) -> str:
    """the kind of the part nearest a label ('?' when none is within `far` label heights); parts: [(cls, box)]"""
    x0, y0, x1, y1 = text_box
    cx, cy, hgt = (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, 1.0)
    best, bd = "?", far * hgt
    for cls, (px0, py0, px1, py1) in parts:
        if cls in ("text", "crossover"):
            continue
        d = math.hypot(max(px0 - cx, 0, cx - px1), max(py0 - cy, 0, cy - py1))
        if d < bd:
            best, bd = PART_KIND.get(cls, "X"), d
    return best


class KindLM:
    """the label model asked as the label stands by a part of one kind: its context starts with the kind's code"""
    def __init__(self, base: "CharLM", kind: str):
        self.base, self.code = base, KIND_CODE.get(kind, KIND_CODE["?"])

    def logp(self, ctx: str, ch: str) -> float:
        return self.base.logp(self.code + ctx, ch)


class LabelReader:
    def __init__(self, model_dir: str = DEFAULT_DIR, device: str | None = None):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
        self.chars = cfg.get("chars", CHARS)
        self.net = CRNN(len(self.chars) + 1)
        self.net.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")))
        self.device = device or "cpu"
        self.net.to(self.device).eval()
        self.lm, self.lm_cfg, self.kind_lm = None, {}, None
        lm_path = os.path.join(model_dir, "lm.json")
        if os.path.isfile(lm_path):                           # the training labels: what labels look like
            d = json.load(open(lm_path, encoding="utf-8"))
            self.lm = CharLM(d["texts"], d.get("n", 4))
            self.lm_cfg = {k: d[k] for k in ("lam", "bonus") if k in d}
            if d.get("kind_texts"):                           # the same labels, each after its part's kind
                self.kind_lm = CharLM(d["kind_texts"], d.get("kind_n", 5))

    @torch.no_grad()
    def read(self, crops, kinds=None) -> list[tuple[str, float]]:
        """kinds: the kind of the part each label stands by (kind_of), when known - read with the model of labels
        by such parts"""
        if not crops:
            return []
        arrs = [prep(c) for c in crops]
        x = batch(arrs).to(self.device)
        logits = self.net(x).float().cpu()
        widths = [-(-a.shape[1] // 4) for a in arrs]
        if self.lm is None:
            return decode(logits, self.chars, widths)
        lp = logits.log_softmax(-1).numpy()
        if kinds is not None and self.kind_lm is not None:
            return [beam_decode(lp[b, :n], self.chars, KindLM(self.kind_lm, k), **self.lm_cfg)
                    for b, (n, k) in enumerate(zip(widths, kinds))]
        return [beam_decode(lp[b, :n], self.chars, self.lm, **self.lm_cfg) for b, n in enumerate(widths)]
