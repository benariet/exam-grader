"""Handwritten formulas read by Uni-MuMER (phxember/Uni-MuMER-Qwen3-VL-2B: Qwen3-VL-2B fine-tuned on
CROHME / HME100K / MathWriting, Apache-2.0; CROHME 2014 / 16 / 19: 83 / 79 / 79 % exact) - a second, much
bigger formula reader for a machine with a GPU (2B parameters; ~10 s per formula on 2 CPU cores).
On an RTX 2080 Ti (Turing: no fast bfloat16) it runs in float32 (~8 GB) - see recog/gpu.py.

The model was saved by transformers 5; this loader runs on transformers 4.57 too: it reads the model
with Qwen3VLForConditionalGeneration, the images with the plain Qwen2VLImageProcessor (no torchvision
needed) and builds the chat prompt itself (models/uni-mumer-v4 holds the settings in the older layout,
next to links to the weights in models/uni-mumer-qwen3vl-2b).

    from recog.mumer import MumerReader
    r = MumerReader()
    r.recognize("formula.png")         # -> [Candidate(text, logp, min_token_p)]
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(PROJECT, "models", "uni-mumer-v4")
SYSTEM = "You are a helpful assistant."
PROMPT = ("I have an image of a handwritten mathematical expression. Please write out the expression of the formula "
          "in the image using LaTeX format.")                       # the paper's prompt (arXiv 2505.23566, A.1)


SOURCE_DIR = os.path.join(PROJECT, "models", "uni-mumer-qwen3vl-2b")      # as downloaded (download_models.py)


def ensure_v4_layout(src: str = SOURCE_DIR, dst: str = DEFAULT_DIR) -> str:
    """the downloaded model's settings rewritten for transformers 4.57 (the weights linked, not copied):
    tokenizer without the list-style extra tokens, rope settings under rope_scaling, the image processor's
    settings in preprocessor_config.json (the plain, torchvision-free processor)"""
    import json
    import shutil
    if os.path.isfile(os.path.join(dst, "preprocessor_config.json")):
        return dst
    os.makedirs(dst, exist_ok=True)
    for f in ("model.safetensors", "tokenizer.json", "chat_template.jinja", "generation_config.json"):
        target, link = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.exists(target) or os.path.exists(link):
            continue
        for make in (lambda: os.symlink(os.path.abspath(target), link), lambda: os.link(target, link),
                     lambda: shutil.copy(target, link)):
            try:
                make()
                break
            except OSError:
                continue
    t = json.load(open(os.path.join(src, "tokenizer_config.json"), encoding="utf-8"))
    for k in ("extra_special_tokens", "backend", "is_local"):
        t.pop(k, None)
    json.dump(t, open(os.path.join(dst, "tokenizer_config.json"), "w", encoding="utf-8"), indent=1)

    def fix(d):
        if isinstance(d, dict):
            if "rope_parameters" in d and "rope_scaling" not in d:
                d["rope_scaling"] = d.pop("rope_parameters")
                if isinstance(d["rope_scaling"], dict) and "rope_theta" in d["rope_scaling"]:
                    d["rope_theta"] = d["rope_scaling"]["rope_theta"]
            for v in d.values():
                fix(v)
    c = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    fix(c)
    json.dump(c, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=1)
    p = json.load(open(os.path.join(src, "processor_config.json"), encoding="utf-8"))
    ip = dict(p.get("image_processor", {}), image_processor_type="Qwen2VLImageProcessor")
    json.dump(ip, open(os.path.join(dst, "preprocessor_config.json"), "w", encoding="utf-8"), indent=1)
    return dst


@dataclass
class Candidate:
    text: str
    logp: float
    min_token_p: float


class MumerReader:
    """dtype None: chosen for the machine (recog/gpu.py) - float32 on an RTX 20xx when it fits (~8 GB),
    bfloat16 on newer GPUs and on CPUs that compute it; a float16 copy that overflows is reloaded in float32"""

    def __init__(self, model_dir: str | None = None, device: str | None = None, threads: int | None = None,
                 dtype=None):
        import torch
        from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLForConditionalGeneration

        from . import gpu
        model_dir = model_dir or ensure_v4_layout()
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.gpu, self.model_dir = torch, gpu, model_dir
        self.n_b = gpu.params_b(model_dir) or 2.13
        self.device, self.dtype = gpu.pick(self.n_b, device) if dtype is None else (gpu.device(device), dtype)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.imp = Qwen2VLImageProcessor.from_pretrained(model_dir)
        self._cls = Qwen3VLForConditionalGeneration
        self.parked = False
        self._load()

    def _load(self):
        self.model = self._cls.from_pretrained(self.model_dir, dtype=self.dtype).to(self.device).eval()

    def _reload_wider(self):
        del self.model
        self.gpu.release()
        self.dtype = self.gpu.fallback_dtype(self.n_b)
        print(f"[mumer] fp16 overflowed (NaN) - reloaded in {self.gpu.SHORT.get(self.dtype)}", flush=True)
        self._load()

    def park(self):
        """free the model's memory for another reader (recog/gpu.py park); the next read brings it back"""
        self.gpu.park(self)

    def recognize(self, img, n_best: int = 1, max_new_tokens: int = 200) -> list[Candidate]:
        torch = self.torch
        if self.parked:
            self.gpu.unpark(self)
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        if min(img.size) < 32:                                  # the vision tower needs a few patches
            s = 32 / min(img.size)
            img = img.resize((max(32, int(img.width * s)), max(32, int(img.height * s))), Image.BICUBIC)
        vis = self.imp(images=[img], return_tensors="pt")
        n_img = int(vis["image_grid_thw"].prod()) // (self.imp.merge_size ** 2)
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = text.replace("<|image_pad|>", "<|image_pad|>" * n_img, 1)
        enc = self.tok(text, return_tensors="pt")
        for attempt in range(2):
            inputs = {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device),
                      "pixel_values": vis["pixel_values"].to(self.device, self.model.dtype),
                      "image_grid_thw": vis["image_grid_thw"].to(self.device)}
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=n_best,
                                          num_return_sequences=n_best, output_scores=True, return_dict_in_generate=True)
            if attempt or self.dtype != torch.float16 or not self.gpu.has_nan(out.scores):
                break
            self._reload_wider()
        n_in = inputs["input_ids"].shape[1]
        trans = self.model.compute_transition_scores(out.sequences, out.scores, getattr(out, "beam_indices", None),
                                                     normalize_logits=n_best == 1)
        cands = []
        for seq, ts in zip(out.sequences[:, n_in:], trans):
            ts = ts[torch.isfinite(ts)].float()
            t = self.tok.decode(seq, skip_special_tokens=True).strip()
            t = t.strip("$ ").replace("\\[", "").replace("\\]", "").strip()
            cands.append(Candidate(t, float(ts.sum()) if len(ts) else 0.0,
                                   float(math.exp(ts.min())) if len(ts) else 0.0))
        return cands
