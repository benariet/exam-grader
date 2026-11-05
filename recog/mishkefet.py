"""Hebrew handwriting with Mishkefet-v1 (4.11: no vision-language model read a line of Hebrew handwriting - Qwen3.6-27B
wrote twenty lines about computers over Tal's page; this one read most of its words).

Mishkefet-v1 (github.com/itayinbarr/heb-ocr - code MIT, weights CC-BY-NC-SA-4.0: research, not commercial use) is a
30M-parameter CNN + ViT line recognizer with a CTC head, second on the ivrit.ai Hebrew handwriting board (median CER
0.175 a line, after Gemini Flash), with a page mode that finds the lines (deskew, ruled lines out). Its code was read
before it was run here; it is loaded safely: the checkpoint with torch.load(weights_only=True) - tensors and plain
values only - and its language models (pickles of plain dicts) through an unpickler that refuses any class or function.

The package is not installed: HEBOCR_SRC (default ~/heb-ocr-src) holds its clone, and its weights in weights/
(mishkefet-v1.pt, hebrew_char6.pkl, hebrew_words.pkl - `hf download itayinbar/Mishkefet-v1 ... --local-dir weights`).
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict

SRC = os.path.expanduser(os.environ.get("HEBOCR_SRC", "~/heb-ocr-src"))
WEIGHTS = os.path.join(SRC, "weights")


class _Plain(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused: {module}.{name}")


def _plain(path: str):
    with open(path, "rb") as fh:
        return _Plain(fh).load()


def _import():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import hebocr.recognize as R
    return R


def load_checkpoint(path: str) -> dict:
    """a Mishkefet checkpoint: tensors and plain values only (torch.load weights_only)"""
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def char_lm(path: str | None = None):
    _import()
    from hebocr.lm import CharNGramLM
    d = _plain(path or os.path.join(WEIGHTS, "hebrew_char6.pkl"))
    lm = CharNGramLM(order=d["order"], backoff=d["backoff"])
    lm.total = d["total"]
    lm.counts = [defaultdict(int, c) for c in d["counts"]]
    return lm


def word_lm(path: str | None = None):
    _import()
    from hebocr.wordlm import WordUnigramLM
    d = _plain(path or os.path.join(WEIGHTS, "hebrew_words.pkl"))
    lm = WordUnigramLM(oov_logprob=d["oov_logprob"], prefix_discount=d["prefix_discount"])
    lm.counts = Counter(d["counts"])
    lm.total = d["total"]
    return lm


def recognizer(checkpoint: str | None = None, page: bool = True, device: str | None = None, **kw):
    """the line reader (beam search with the character LM; page=True adds the word LM, as its page mode)"""
    import torch
    R = _import()
    real = torch.load
    torch.load = lambda f, map_location=None, weights_only=True, **k: real(f, map_location=map_location,
                                                                          weights_only=True)
    try:
        args = dict(lm=char_lm(), lm_weight=0.4, beam_width=12, device=device)
        if page:
            args.update(word_lm=word_lm(), word_weight=0.2)
        args.update(kw)
        return R.Recognizer(checkpoint or os.path.join(WEIGHTS, "mishkefet-v1.pt"), **args)
    finally:
        torch.load = real


def no_vertical_rules(img):
    """a notebook's margin rules - long, light vertical lines - whitened (they were read as a final nun, 'ן', at
    every line's start). Ink is what is darker than the paper around it (5.11: a fixed level, 235, took all of a
    yellowed or photographed page - grey paper - for ink, and whitened the whole page: no line found); a mask over
    more than 2% of the page is not rules, and nothing is whitened"""
    import cv2
    import numpy as np
    from PIL import Image
    g = np.asarray(img.convert("L"), dtype=np.float32)
    small = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) if min(g.shape) > 400 else g
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(small.shape) / 30.0), (g.shape[1], g.shape[0]))
    ink = (g < 0.94 * bg).astype(np.uint8)            # (white paper, 250: below 235 - the level it had)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, g.shape[0] // 12)))
    vert = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k), np.ones((3, 5), np.uint8))
    if vert.mean() > 0.02:
        return img
    a = np.asarray(img.convert("RGB")).copy()
    a[vert > 0] = 255
    return Image.fromarray(a)


SMALL_SIDE, ENLARGED = 1400, 1800     # (a 370x550 scan: 2 lines found of 16; enlarged to 1800, all 16)
TURN_BELOW = 0.90                     # a page read less surely than this is tried turned (90, 270, 180 degrees)


def page_quality(recs: list) -> float:
    """how surely a page was read: the lines' confidence weighted by their length (lines of 3 letters or more),
    times the share of a line of text the lines' median length is (8 letters and up: all of it); 0 when none. A
    page turned sideways reads as scraps of 2-4 letters, each read surely (5.11: Rachel's 'Gan Naul' photographed
    sideways - 150 scraps at 0.96; turned, 15 lines of the poem at 0.94)"""
    import statistics
    rs = [(r["confidence"], len(r["text"].replace(" ", ""))) for r in recs]
    rs = [(c, n) for c, n in rs if n >= 3]
    if not rs:
        return 0.0
    sure = sum(c * n for c, n in rs) / sum(n for _, n in rs)
    return sure * min(1.0, statistics.median(n for _, n in rs) / 8)


def no_printed_rules(img):
    """a printed grid or ruled lines - light, thin, long - whitened, and with them anything as light (5.11: Hannah
    Senesh's diaries, on squared paper: its squares were read as letters, a page gave 2 lines of 22 - with the grid
    out, 21). The rules are the thin long straight runs on the paper; a page with some is cleaned of everything
    lighter than halfway between the rules and the darker part of the writing - and only where the writing is
    clearly darker than the rules (pencil on a grid is left alone) -> (the page, what was measured)"""
    import cv2
    import numpy as np
    from PIL import Image
    g = np.asarray(img.convert("L"), dtype=np.float32)
    small = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) if min(g.shape) > 400 else g
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(small.shape) / 30.0), (g.shape[1], g.shape[0]))
    flat = g / np.maximum(bg, 1.0)
    raw = cv2.GaussianBlur(g, (0, 0), 6)
    paper = raw > 0.55 * np.percentile(raw, 95)                    # (not the dark table around a photo)
    paper = cv2.erode(paper.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    mark = ((flat < 0.93) & paper).astype(np.uint8)

    def thin(m, k):                                                # (a rule is a thin line: bands, edges go)
        thick = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        return m & ~cv2.dilate(thick, np.ones((5, 5), np.uint8))
    h = cv2.morphologyEx(mark, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, g.shape[1] // 15), 1)))
    v = cv2.morphologyEx(mark, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, g.shape[0] // 15))))
    rules = (thin(h, (1, 9)) | thin(v, (9, 1))) > 0
    info = {"rules": round(float(rules.mean()), 4)}
    if rules.mean() < 0.004:
        return img, info
    ink = (mark > 0) & ~(cv2.dilate(rules.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    r_med = float(np.median(flat[rules]))
    i_dark = float(np.percentile(flat[ink], 30)) if ink.any() else 1.0
    info.update(rule=round(r_med, 2), ink=round(i_dark, 2))
    if r_med < i_dark + 0.15:
        return img, info
    t = (r_med + i_dark) / 2
    info["t"] = round(t, 2)
    a = np.asarray(img.convert("RGB")).copy()
    a[flat > t] = 255
    return Image.fromarray(a), info


def _read_lines(img, rec, clean: bool = False, max_side: int = 2000) -> list:
    _import()
    from hebocr.page.segment import segment_page
    if clean:
        img = no_printed_rules(img)[0]
    crops, boxes, angle = segment_page(no_vertical_rules(img), max_side=max_side)
    if not crops:
        return []
    texts = rec.read_tta(crops) if rec.tta and rec.tta > 1 else rec.read(crops)
    _, confs = rec.read(crops, return_confidence=True)
    return [{"text": t, "confidence": c, "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h}, "skew": angle, "crop": im}
            for t, c, b, im in zip(texts, confs, boxes, crops)]


def read_page(img, rec=None, turn: bool = True, clean: bool = False) -> list:
    """a page of handwriting -> [{"text", "confidence", "box": {x, y, w, h}, "crop": the line's image, "turned":
    degrees}] in reading order (its segmenter's: deskewed, the ruled lines out - the boxes are in the deskewed page,
    the crops are cut). A small scan is enlarged first (the segmenter's thresholds follow the line pitch, but
    stop at a few pixels). A page read unsurely is read again with the segmenter working smaller (1000 and 1400
    pixels a side, not 2000 - 5.11: a stained diary page, its faint squares taken for lines: 6 lines found at
    2000, 22 at 1000), and with `turn` turned by 90, 270 and 180 degrees - a photo taken sideways; the surest
    reading is kept (the boxes are then in the turned page). With `clean`, a printed grid or ruled lines are taken
    out first (no_printed_rules - squared paper)"""
    rec = rec or recognizer()
    from PIL import Image
    s = ENLARGED / max(img.size) if max(img.size) < SMALL_SIDE else 1.0
    if s != 1.0:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    best = _read_lines(img, rec, clean)
    q = page_quality(best)
    for side in (1000, 1400):
        if q < TURN_BELOW:
            alt = _read_lines(img, rec, clean, max_side=side)
            qa = page_quality(alt)
            if qa > q + 0.02:
                best, q = alt, qa
    if turn and q < TURN_BELOW:
        for deg in (90, 270, 180):
            alt = _read_lines(img.rotate(deg, expand=True, fillcolor=(255, 255, 255)), rec, clean)
            qa = page_quality(alt)
            if qa > q + 0.02:
                best, q = [dict(r, turned=deg) for r in alt], qa
    for r in best:                                  # (the boxes back in the page as given)
        r["box"] = {k: round(v / s) for k, v in r["box"].items()}
    return [dict({"turned": 0}, **r) for r in best]


def nbest_decode(logprobs, lengths, charset, n: int = 12, beam_width: int = 24, top_k: int = 8, lm=None,
                 lm_weight: float = 0.4, length_bonus: float = 0.6, word_lm=None, word_weight: float = 0.2) -> list:
    """Mishkefet's CTC prefix beam search (hebocr.decode.beam_decode, ported - MIT) keeping the n best readings of
    each line with their scores, for a language model that sees whole sentences to choose among -> [[(text, score)]]"""
    import math
    neg = float("-inf")

    def lse(a, b):
        if a == neg:
            return b
        if b == neg:
            return a
        hi, lo = (a, b) if a > b else (b, a)
        return hi + math.log1p(math.exp(lo - hi))

    out = []
    for row, valid in zip(logprobs.cpu(), lengths.cpu()):
        row = row[: int(valid)]
        beams = {"": [0.0, neg]}
        cand = row.topk(min(top_k, row.shape[-1]), dim=-1).indices
        for t in range(row.shape[0]):
            frame, nxt = row[t], {}
            for prefix, (pb, pnb) in beams.items():
                ptot = lse(pb, pnb)
                for idx in cand[t].tolist():
                    p = float(frame[idx])
                    if idx == 0:                                   # (the blank)
                        e = nxt.setdefault(prefix, [neg, neg])
                        e[0] = lse(e[0], ptot + p)
                        continue
                    ch = charset.chars[idx]
                    if prefix and ch == prefix[-1]:
                        e = nxt.setdefault(prefix, [neg, neg])
                        e[1] = lse(e[1], pnb + p)
                        frm = pb
                    else:
                        frm = ptot
                    if frm == neg:
                        continue
                    bonus = (lm_weight * lm.logprob(prefix, ch) + length_bonus) if lm is not None else 0.0
                    if word_lm is not None and word_weight and ch == " ":
                        closed = prefix.rsplit(" ", 1)[-1]
                        if closed:
                            bonus += word_weight * word_lm.bonus(closed)
                    e = nxt.setdefault(prefix + ch, [neg, neg])
                    e[1] = lse(e[1], frm + p + bonus)
            beams = dict(sorted(nxt.items(), key=lambda kv: lse(*kv[1]), reverse=True)[:beam_width]) or {"": [0.0, neg]}

        def final(kv):
            s = lse(*kv[1])
            tail = kv[0].rsplit(" ", 1)[-1]
            return s + (word_weight * word_lm.bonus(tail) if word_lm is not None and word_weight and tail else 0.0)
        out.append(sorted(((k, final((k, v))) for k, v in beams.items()), key=lambda x: -x[1])[:n])
    return out


def read_nbest(rec, images: list, n: int = 12, beam_width: int = 24) -> list:
    """each line image's n best readings [(text, score)] - the recognizer's model and language models"""
    import torch
    _import()
    from hebocr.data.transforms import preprocess
    out = []
    with torch.no_grad():
        for im in images:
            a = torch.from_numpy(preprocess(im))[None]
            lp = rec.model(a.to(rec.device)).float()
            out += nbest_decode(lp, rec.model.output_lengths(torch.tensor([a.shape[-1]])), rec.charset, n=n,
                                beam_width=beam_width, lm=rec.lm, lm_weight=rec.lm_weight,
                                length_bonus=rec.length_bonus, word_lm=rec.word_lm, word_weight=rec.word_weight)
    return out


class HebrewLM:
    """a Hebrew language model (DictaLM-3.0-1.7B, models/dictalm-3.0-1.7b-base) scoring a line after the lines before
    it: the readings a hand allows, chosen by what Hebrew says (rescore)"""

    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()

    def logprob(self, text: str, context: str = "") -> float:
        """log P(text | context), summed over text's tokens"""
        import torch
        ctx = self.tok(context, add_special_tokens=False)["input_ids"] if context else []
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        x = torch.tensor([bos + ctx + ids], device=self.dev)
        with torch.no_grad():
            lp = torch.log_softmax(self.model(x).logits[0, :-1].float(), -1)
        start = max(1, len(bos) + len(ctx))                 # (with nothing before it the first token is not scored)
        tgt = x[0, start:]
        return float(lp[start - 1:].gather(1, tgt[:, None]).sum()) if len(tgt) else 0.0


def rescore(nbest: list, hlm, weight: float = 0.3, context: list | None = None) -> list:
    """per line, the reading with the best hand + weight * Hebrew score, each line read after the ones chosen before
    it (two of them as context; `context`: lines before these). Tal's page, its Hebrew lines: 24% of the characters
    wrong -> 21% (weight 0.3); the right reading is often not among the 16 - the hand is what is not known"""
    chosen = list(context or [])
    first = len(chosen)
    for cands in nbest:
        ctx = " ".join(chosen[-2:])
        best = max(cands, key=lambda c: c[1] + weight * hlm.logprob(" " + c[0].strip() if ctx else c[0].strip(), ctx))
        chosen.append(best[0].strip())
    return chosen[first:]


def ctc_logprob(rec, image, text: str) -> float:
    """log P(text | the line's image) under Mishkefet's model (CTC, all alignments summed): how well the handwriting
    supports a reading - a language model's correction checked against what is written"""
    import torch
    import torch.nn.functional as F
    _import()
    from hebocr.data.transforms import preprocess
    a = torch.from_numpy(preprocess(image))[None]
    tgt = torch.tensor(rec.charset.encode(text), dtype=torch.long)
    with torch.no_grad():
        lp = rec.model(a.to(rec.device)).float()
        n = rec.model.output_lengths(torch.tensor([a.shape[-1]]))
        if len(tgt) == 0 or int(n[0]) < len(tgt):
            return float("-inf")
        return -float(F.ctc_loss(lp.permute(1, 0, 2), tgt[None].to(rec.device), n.to(rec.device),
                                 torch.tensor([len(tgt)], device=rec.device), blank=0, reduction="sum",
                                 zero_infinity=False))


def verified_words(rec, image, base: str, proposed: str, slack: float = 2.0) -> tuple:
    """the proposed line's words taken where the image supports them: word by word (aligned by position where the
    word counts agree, else the whole line), a changed word kept only if the line with it scores within `slack` of
    the line without it -> (the line, the words changed)"""
    b, p = base.split(), proposed.split()
    if not p:
        return base, []
    if len(b) != len(p):
        return (proposed, ["*"]) if ctc_logprob(rec, image, proposed) >= ctc_logprob(rec, image, base) - slack \
            else (base, [])
    cur, changed = list(b), []
    for k, (x, y) in enumerate(zip(b, p)):
        if x == y:
            continue
        trial = cur[:k] + [y] + cur[k + 1:]
        if ctc_logprob(rec, image, " ".join(trial)) >= ctc_logprob(rec, image, " ".join(cur)) - slack:
            cur, changed = trial, changed + [y]
    return " ".join(cur), changed
