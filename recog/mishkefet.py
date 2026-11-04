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
    every line's start)"""
    import cv2
    import numpy as np
    from PIL import Image
    g = np.asarray(img.convert("L"))
    ink = (g < 235).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, g.shape[0] // 12)))
    vert = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k), np.ones((3, 5), np.uint8))
    a = np.asarray(img.convert("RGB")).copy()
    a[vert > 0] = 255
    return Image.fromarray(a)


def read_page(img, rec=None) -> list:
    """a page of handwriting -> [{"text", "confidence", "box": {x, y, w, h}, "crop": the line's image}] in reading
    order (its segmenter's: deskewed, the ruled lines out - the boxes are in the deskewed page, the crops are cut)"""
    rec = rec or recognizer()
    _import()
    from hebocr.page.segment import segment_page
    crops, boxes, angle = segment_page(no_vertical_rules(img))
    if not crops:
        return []
    texts = rec.read_tta(crops) if rec.tta and rec.tta > 1 else rec.read(crops)
    _, confs = rec.read(crops, return_confidence=True)
    return [{"text": t, "confidence": c, "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h}, "skew": angle, "crop": im}
            for t, c, b, im in zip(texts, confs, boxes, crops)]
