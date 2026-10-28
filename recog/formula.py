"""Handwritten-formula recognizer: Pix2Text MFR (TrOCR-style) running on ONNX Runtime only.

No torch / transformers / optimum needed at runtime: onnxruntime + tokenizers + numpy + pillow.
Model files are read from a local folder (default: <project>/models/pix2text-mfr), so it works offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from . import session_options
from .beam import beam_search

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "pix2text-mfr")


@dataclass
class Candidate:
    text: str
    logp: float              # total log-probability of the sequence
    min_token_p: float       # weakest token — a useful per-reading confidence signal


def _to_rgb_on_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


class FormulaRecognizer:
    def __init__(self, model_dir: str = DEFAULT_DIR, providers: list[str] | None = None):
        if not os.path.isfile(os.path.join(model_dir, "encoder_model.onnx")):
            raise FileNotFoundError(f"model not found in {model_dir} — run download_models.py")
        providers = providers or ["CPUExecutionProvider"]
        so = session_options()
        self.enc = ort.InferenceSession(os.path.join(model_dir, "encoder_model.onnx"), so, providers=providers)
        self.dec = ort.InferenceSession(os.path.join(model_dir, "decoder_model.onnx"), so, providers=providers)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        pre = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.size = (pre["size"]["width"], pre["size"]["height"])
        self.mean = np.array(pre["image_mean"], dtype=np.float32)[:, None, None]
        self.std = np.array(pre["image_std"], dtype=np.float32)[:, None, None]
        self.resample = pre.get("resample", 3)
        gen_p = os.path.join(model_dir, "generation_config.json")
        gen = json.load(open(gen_p)) if os.path.isfile(gen_p) else {}
        # the start token sits in config.json, generation_config.json or the decoder config, by export
        self.start_id = next(v for v in (cfg.get("decoder_start_token_id"), gen.get("decoder_start_token_id"),
                                         cfg.get("decoder", {}).get("decoder_start_token_id")) if v is not None)
        self.eos_id = cfg.get("eos_token_id") if cfg.get("eos_token_id") is not None else gen["eos_token_id"]
        self.special = {t.id for t in []}  # filled lazily by decode(skip_special_tokens=True)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = _to_rgb_on_white(img).resize(self.size, resample=self.resample)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    def grammar(self, pattern: str, chars: str):
        """-> allowed(tokens) for beam_search: only readings that fully match the regular expression
        `pattern` (the `regex` module's partial matching keeps every prefix that can still become one),
        built from tokens made of `chars` only; the end token only once the reading is complete"""
        import regex
        if not hasattr(self, "_surf"):
            # a token's text: sentencepiece marks a leading space with '▁', byte-level BPE with 'Ġ'
            self._surf = {i: s.replace("▁", " ").replace("Ġ", " ") for s, i in self.tok.get_vocab().items()}
        cset = set(chars)
        cand = np.array(sorted(i for i, s in self._surf.items() if s and set(s) <= cset and i != self.eos_id), dtype=np.int64)
        rx = regex.compile(pattern)
        cache: dict[str, np.ndarray] = {}

        def allowed(tokens: list[int]) -> np.ndarray:
            text = "".join(self._surf.get(t, "") for t in tokens if t != self.eos_id)
            if text not in cache:
                ok = [i for i in cand if rx.fullmatch((text + self._surf[i]).lstrip(), partial=True)]
                if rx.fullmatch(text.strip()):
                    ok.append(self.eos_id)
                cache[text] = np.array(ok, dtype=np.int64)
            return cache[text]
        return allowed

    def _grammar_cached(self, grammar):
        if not hasattr(self, "_grammars"):
            self._grammars = {}
        if grammar not in self._grammars:
            self._grammars[grammar] = self.grammar(*grammar)
        return self._grammars[grammar]

    def recognize(self, img: Image.Image | str, n_best: int = 3, max_new_tokens: int = 160,
                  grammar: tuple[str, str] | None = None) -> list[Candidate]:
        """grammar: (regular expression, characters) - read only what fits it (see grammar())"""
        if isinstance(img, str):
            img = Image.open(img)
        enc = self.enc.run(None, {"pixel_values": self.preprocess(img)})[0]        # [1, S, H]

        def step(ids: np.ndarray) -> np.ndarray:
            e = np.repeat(enc, ids.shape[0], axis=0)
            logits = self.dec.run(None, {"input_ids": ids, "encoder_hidden_states": e})[0]
            return logits[:, -1, :]

        hyps = beam_search(step, start_id=self.start_id, eos_id=self.eos_id,
                           num_beams=n_best, max_new_tokens=max_new_tokens,
                           allowed=self._grammar_cached(grammar) if grammar else None)
        out = []
        for h in hyps:
            toks = [t for t in h.tokens[1:] if t != self.eos_id]
            text = self.tok.decode(toks, skip_special_tokens=True)
            out.append(Candidate(text, h.logp, float(np.exp(min(h.token_logps))) if h.token_logps else 0.0))
        return out
