"""A vision-language model served by vLLM on the same GPU server (4.11: Qwen3.6-27B through the transformers loader
ran at under 10% of two GPUs - 8-bit weights, one token at a time - about 20 s a math piece; vLLM serves FP8 weights
on Ampere cards through its Marlin kernels, spread over the job's GPUs, many requests at once).

The tools take it in place of a local model: `--model http://127.0.0.1:8000/v1` (tools/server_try_reader.sbatch starts
`vllm serve ... --served-model-name reader` in the job and points the tools at it). Only the job's own machine is
spoken to - the scans never leave the server.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
import urllib.request

from .hebrew_vlm import Candidate, no_thinking, normalize_hebrew


def is_url(model: str) -> bool:
    return str(model).startswith(("http://", "https://"))


class RemoteReader:
    """the reader at `url` (an OpenAI-compatible /v1 of vLLM), served as `name`; greedy - one reading a request"""

    def __init__(self, url: str, name: str = "reader", timeout: float = 300.0):
        self.url, self.name, self.timeout = url.rstrip("/"), name, timeout
        self.device, self.dtype = url, "served"

    def _ask(self, img, prompt: str, max_new_tokens: int, repetition_penalty: float = 1.0) -> tuple[str, list]:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        body = {"model": self.name, "temperature": 0.0, "max_tokens": max_new_tokens, "logprobs": True,
                "repetition_penalty": repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": False},      # (Qwen3.5 / 3.6 think first otherwise)
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                              + base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": prompt}]}]}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5)
        choice = out["choices"][0]
        lps = [t.get("logprob", 0.0) for t in ((choice.get("logprobs") or {}).get("content") or [])]
        return no_thinking(choice["message"]["content"] or ""), lps

    def recognize(self, img, n_best: int = 1, max_new_tokens: int = 96, prompt: str = "", **_) -> list[Candidate]:
        """as recog/hebrew_vlm.HebrewRecognizer.recognize - one candidate (greedy), its log-probability summed"""
        text, lps = self._ask(img, prompt, max_new_tokens)
        lps = [x for x in lps if math.isfinite(x)]
        return [Candidate(normalize_hebrew(text), float(sum(lps)), float(math.exp(min(lps))) if lps else 0.0)]

    def read(self, img, max_new_tokens: int = 1500, prompt: str = "", repetition_penalty: float = 1.1) -> str:
        """as tools/read_blocks.BlockReader.read before its clean-up: the answer's lines kept"""
        return self._ask(img, prompt, max_new_tokens, repetition_penalty)[0]
