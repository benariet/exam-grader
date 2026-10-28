"""Hebrew handwriting via a local vision-language model (Qwen3-VL family) - EXPERIMENTAL.

Zero-shot Qwen3-VL-2B could not read Hebrew handwriting (CER ~0.93 on our test lines);
kept for trying bigger models or a Hebrew-HTR LoRA on a GPU. The default recognizer is
recog/hebrew.py (trained CRNN).

Returns an n-best list of transcriptions with their total log-probabilities (beam search),
so the grading layer can measure how much the reading uncertainty matters.

Model weights are read from <project>/models/<name> (downloaded once by download_models.py),
so nothing goes over the network at runtime.
"""
from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass

import torch
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT, "models", "qwen3-vl-2b")

from .prompts import PROMPT, PROMPT_ENGLISH, PROMPT_MATH  # noqa: E402,F401  (re-exported)


@dataclass
class Candidate:
    text: str
    logp: float              # total log-probability of the sequence
    min_token_p: float       # probability of the weakest token


def normalize_hebrew(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")      # niqqud
    s = s.replace("‏", "").replace("‎", "").replace("‫", "").replace("‬", "")
    return " ".join(s.split())


def pick_device() -> tuple[str, torch.dtype]:
    from . import gpu
    return gpu.pick()


class HebrewRecognizer:
    """dtype None: chosen for the machine (recog/gpu.py). load_in_4bit (bitsandbytes, a CUDA GPU): weights
    in 4 bits - an 8B model in ~7 GB; the rest and the arithmetic in float16 on an RTX 20xx (bfloat16 on
    newer GPUs). A float16 model that overflows (NaN) is reloaded wider (float32, or bfloat16 if that does
    not fit)."""

    def __init__(self, model_dir: str = DEFAULT_MODEL, adapter_dir: str | None = None,
                 device: str | None = None, load_in_4bit: bool = False, dtype=None):
        from transformers import AutoProcessor

        from . import gpu
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model not found in {model_dir} — run download_models.py")
        self.gpu, self.model_dir, self.adapter_dir, self.load_in_4bit = gpu, model_dir, adapter_dir, load_in_4bit
        self.n_b = gpu.params_b(model_dir)
        self.device, self.dtype = gpu.pick(None if load_in_4bit else self.n_b, device, fast=load_in_4bit)
        if dtype is not None:
            self.dtype = dtype
        self.parked = False
        self._load()
        self.proc = AutoProcessor.from_pretrained(model_dir)

    def _load(self):
        from transformers import AutoModelForImageTextToText
        kwargs = {"dtype": self.dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            # what does not fit the GPU's free memory (a card that also drives the screen may have only half of
            # it free) stays on the CPU in float32 - usually just the last layers and the output layer
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype, bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True)
            kwargs["device_map"] = "auto"
            if self.device.startswith("cuda"):
                budget = max(1.0, self.gpu.gpu_free_gb() - self.gpu.MARGIN_GB - 0.5)
                kwargs["max_memory"] = {0: f"{budget:.1f}GiB", "cpu": "96GiB"}
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_dir, **kwargs)
        if not self.load_in_4bit:
            self.model.to(self.device)
        if self.adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)
        self.model.eval()

    def _reload_wider(self):
        del self.model
        self.gpu.release()
        if self.load_in_4bit:
            self.dtype = torch.float32
            try:
                self._load()
            except torch.cuda.OutOfMemoryError:
                self.gpu.release()
                self.dtype = torch.bfloat16
                self._load()
        else:
            self.dtype = self.gpu.fallback_dtype(self.n_b)
            self._load()
        print(f"[vlm] fp16 overflowed (NaN) - reloaded in {self.gpu.SHORT.get(self.dtype)}", flush=True)

    def park(self):
        """free the model's memory for another reader (recog/gpu.py park: a 4-bit model is dropped and
        loaded again on the next read)"""
        self.gpu.park(self, quantized=self.load_in_4bit)

    def _unpark(self):
        if self.parked:
            self.gpu.unpark(self)

    @torch.inference_mode()
    def recognize(self, img: Image.Image | str, n_best: int = 3, max_new_tokens: int = 96,
                  prompt: str = PROMPT, system: str | None = None) -> list[Candidate]:
        self._unpark()
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        if system:
            msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        for attempt in range(2):
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(self.model.device)
            if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      num_beams=n_best, num_return_sequences=n_best, length_penalty=1.0,
                                      output_scores=True, return_dict_in_generate=True)
            if attempt or self.dtype != torch.float16 or not self.gpu.has_nan(out.scores):
                break
            self._reload_wider()
        prompt_len = inputs["input_ids"].shape[1]
        gen = out.sequences[:, prompt_len:]
        trans = self.model.compute_transition_scores(
            out.sequences, out.scores, getattr(out, "beam_indices", None), normalize_logits=n_best == 1)
        pad = self.proc.tokenizer.pad_token_id
        cands = []
        for seq, ts in zip(gen, trans):
            valid = seq != pad if pad is not None else torch.ones_like(seq, dtype=torch.bool)
            tok_lp = ts[: len(seq)][valid[: ts.shape[0]]].float()
            tok_lp = tok_lp[torch.isfinite(tok_lp)]
            text = normalize_hebrew(self.proc.tokenizer.decode(seq, skip_special_tokens=True))
            cands.append(Candidate(text, float(tok_lp.sum()) if len(tok_lp) else 0.0,
                                   float(tok_lp.min().exp()) if len(tok_lp) else 0.0))
        return cands
