"""Read a Hebrew line in context: the handwriting model says what the letters look like, a Hebrew
language model says what a person would write - the reading that is best for both wins.

    P(text | image)  ~  P(image | text) * P(text) ** alpha

* P(image | text): the CRNN's CTC probability of a whole candidate text - exact for ANY text, not only
  for what its beam search found (`ctc_logp`).
* P(text): Dicta-LM 3.0 1.7B (dicta-il/DictaLM-3.0-1.7B-Base, Apache-2.0, a Hebrew-first LLM; one
  forward pass gives the probability of a whole sentence).
* candidates: the beam search's n-best, and for every word of the best reading the words of the
  vocabulary that are one or two letters away from it ('אור' -> 'אותך', 'אזר' -> 'איך', 'באן' -> 'בגן');
  a word the handwriting model finds much less likely than the one it read (more than PRUNE nats) is
  not tried. The line is improved one word at a time, as long as the total score rises.

    from recog.hebrew_context import HebrewContext
    ctx = HebrewContext()                      # loads the language model (3.4 GB; a GPU if there is one)
    best = ctx.read(img)                       # -> Candidate (text, logp, ...), like HebrewRecognizer
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

from .hebrew import LM_DIR, PROJECT, Candidate, HebrewRecognizer, WordLM
from .hebrew_text import CHAR2ID, clean, to_visual

LM_MODEL_DIR = os.path.join(PROJECT, "models", "dictalm-3.0-1.7b-base")
ALPHA = 0.5            # weight of the language model (tuned on held-out training writers - see tools/hebrew_context_eval.py)
WORD_PENALTY = 0.0     # per word (the language model alone would prefer fewer, more common words)
PRUNE = 12.0           # a replacement word may cost the handwriting model at most this many nats
PER_WORD = 8           # ... and of those, the handwriting model's likeliest PER_WORD are tried
MAX_EDITS = 2          # vocabulary words this many letters away are tried (one for words of 1-3 letters)
VOCAB_TOP = 150_000    # the most frequent words of the word list (+ the course words) are the vocabulary
_HEB_WORD = re.compile(r"^[א-ת'\"]+$")


def ctc_logp(lp: np.ndarray, labels) -> float:
    """log P(labels | image) under CTC: lp [T, K] log-probabilities (blank = 0), labels: ids in the
    model's (visual) order. Forward algorithm over the blank-interleaved label sequence."""
    labels = list(labels)
    T = lp.shape[0]
    if not labels:
        return float(lp[:, 0].sum())
    ext = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    ext[1::2] = labels
    S = len(ext)
    if T < len(labels):
        return -math.inf
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
    alpha = np.full(S, -np.inf)
    alpha[0] = lp[0, 0]
    alpha[1] = lp[0, ext[1]]
    neg = np.array([-np.inf])
    for t in range(1, T):
        a2 = np.concatenate([neg, alpha[:-1]])
        a3 = np.concatenate([neg, neg, alpha[:-2]])
        a3[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, a2), a3) + lp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def text_ids(text: str):
    """logical text -> the model's label ids (visual order); None if a character is outside its charset"""
    vis = to_visual(clean(text))
    ids = [CHAR2ID[c] for c in vis if c in CHAR2ID]
    return ids if ids else None


@dataclass
class Reading:
    text: str
    acoustic: float        # log P(image | text), mean over the views read
    lm: float              # log P(text) under the language model
    score: float           # acoustic + alpha * lm - penalty * words


class HebrewLM:
    """sentence log-probabilities from a causal Hebrew LLM (Dicta-LM 3.0), cached. On a GPU without fast
    bfloat16 (RTX 20xx) it runs in float16 for speed; if that overflows (NaN), it is reloaded in float32."""

    def __init__(self, model_dir: str = LM_MODEL_DIR, device: str | None = None, threads: int | None = None,
                 dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from . import gpu
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.gpu, self.model_dir = torch, gpu, model_dir
        self.n_b = gpu.params_b(model_dir) or 1.7
        self.device, self.dtype = gpu.pick(self.n_b, device, fast=True) if dtype is None else (gpu.device(device), dtype)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self._cls = AutoModelForCausalLM
        self.parked = False
        self._load()
        self.start = self.tok.bos_token or self.tok.eos_token or ""
        self.cache = {}

    def _load(self):
        self.model = self._cls.from_pretrained(self.model_dir, dtype=self.dtype).to(self.device).eval()

    def park(self):
        """free the model's memory for another reader (recog/gpu.py park); the next call brings it back"""
        self.gpu.park(self)

    def _scores(self, chunk):
        torch = self.torch
        enc = self.tok([self.start + s for s in chunk], return_tensors="pt", padding=True).to(self.device)
        ids, att = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
        return (lp * att[:, 1:].float()).sum(1).tolist()

    def logp(self, sentences, batch: int = 16):
        todo = [s for s in dict.fromkeys(sentences) if s not in self.cache]
        if todo and self.parked:
            self.gpu.unpark(self)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            tot = self._scores(chunk)
            if self.dtype == self.torch.float16 and not all(math.isfinite(v) for v in tot):
                del self.model
                self.gpu.release()
                self.dtype = self.gpu.fallback_dtype(self.n_b)
                print(f"[hebrew LM] fp16 overflowed (NaN) - reloaded in {self.gpu.SHORT.get(self.dtype)}", flush=True)
                self._load()
                tot = self._scores(chunk)
            for s, v in zip(chunk, tot):
                self.cache[s] = v
        return [self.cache[s] for s in sentences]


class HebrewContext:
    def __init__(self, rec: HebrewRecognizer | None = None, lm: HebrewLM | None = None, alpha: float = ALPHA,
                 word_penalty: float = WORD_PENALTY, prune: float = PRUNE, max_edits: int = MAX_EDITS,
                 vocab_top: int = VOCAB_TOP, lm_dir: str = LM_DIR, per_word: int = PER_WORD):
        self.rec = rec or HebrewRecognizer()
        self.lm = lm or HebrewLM()
        self.alpha, self.word_penalty, self.prune, self.max_edits = alpha, word_penalty, prune, max_edits
        self.per_word = per_word
        words = WordLM(lm_dir)
        ranked = sorted(words.table.items(), key=lambda kv: -kv[1])[:vocab_top]
        self.vocab = sorted({w for w, _ in ranked} | set(words.course_words))
        self.by_len = {}
        for w in self.vocab:
            self.by_len.setdefault(len(w), []).append(w)

    # -- candidates ----------------------------------------------------------------------------
    PREFIXES = "והבכלמש"

    def _near(self, word: str, k: int, limit: int):
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
        pool = [w for n in range(max(1, len(word) - k), len(word) + k + 1) for w in self.by_len.get(n, [])]
        return [w for w, d, _ in process.extract(word, pool, scorer=Levenshtein.distance, score_cutoff=k, limit=limit)]

    def neighbours(self, word: str, limit: int = 40):
        """vocabulary words within MAX_EDITS letters of the word (one for words of 1-2 letters), also behind
        the prefix letters it starts with ('מהממהרם' -> 'מה' + 'ממהרים': Hebrew glues ו ה ב כ ל מ ש to words,
        and the word list does not hold every glued form)"""
        k = self.max_edits if len(word) > 2 else 1
        out = dict.fromkeys(self._near(word, k, limit))
        for i in range(1, min(3, len(word) - 2) + 1):
            if word[i - 1] not in self.PREFIXES:
                break
            for w in self._near(word[i:], k if len(word) - i > 2 else 1, limit // 2):
                out.setdefault(word[:i] + w)
        out.pop(word, None)
        return list(out)

    def acoustic(self, views, text: str) -> float:
        return self.acoustic_many(views, [text])[0]

    def acoustic_many(self, views, texts):
        """mean over the views of log P(image | text), for many texts at once (torch's CTC loss, batched;
        `ctc_logp` is the same in plain numpy)"""
        import torch
        out = [-math.inf] * len(texts)
        ok = [(i, text_ids(t)) for i, t in enumerate(texts)]
        ok = [(i, ids) for i, ids in ok if ids]
        if not ok:
            return out
        tg = torch.tensor([x for _, ids in ok for x in ids], dtype=torch.long)
        tl = torch.tensor([len(ids) for _, ids in ok], dtype=torch.long)
        tot = torch.zeros(len(ok), dtype=torch.float64)
        for lp in views:
            x = torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32))[:, None, :].expand(-1, len(ok), -1)
            il = torch.full((len(ok),), lp.shape[0], dtype=torch.long)
            nll = torch.nn.functional.ctc_loss(x, tg, il, tl, blank=0, reduction="none", zero_infinity=False)
            tot += -nll.double()
        for (i, _), v in zip(ok, (tot / len(views)).tolist()):
            out[i] = v if math.isfinite(v) else -math.inf
        return out

    def views(self, img, tta: bool = True):
        from PIL import Image
        scales = self.rec.TTA_SCALES if tta else (1.0,)
        out = []
        for s in scales:
            v = img if s == 1.0 else img.resize((max(16, int(img.width * s)), img.height), Image.BILINEAR)
            out.append(self.rec.log_probs(v))
        return out

    def score(self, reading_text, ac, lmv, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return ac + alpha * lmv - self.word_penalty * len(reading_text.split())

    def start(self, img, tta: bool = True, n_best: int = 8):
        """the views' log-probabilities and the beam search's readings (the search's starting points)"""
        views = self.views(img, tta)
        start = [c.text for c in self.rec.recognize(img, n_best=n_best, tta=tta)]
        return views, [t for t in dict.fromkeys(start) if t.strip()] or [""]

    def read(self, img, tta: bool = True, n_best: int = 8, passes: int = 2, trace: bool = False,
             alpha: float | None = None, prepared=None, acs: dict | None = None):
        """-> (best Reading, [the starting n-best Readings]) - improved word by word. prepared: start()'s
        result, acs: a {text: acoustic score} cache for that line (both: to try several alphas cheaply)"""
        views, start = prepared or self.start(img, tta, n_best)
        acs = {} if acs is None else acs
        new = [t for t in start if t not in acs]
        acs.update(zip(new, self.acoustic_many(views, new)))
        lms = dict(zip(start, self.lm.logp(start)))
        pool = [Reading(t, acs[t], lms[t], self.score(t, acs[t], lms[t], alpha)) for t in start]
        best = max(pool, key=lambda r: r.score)
        for _ in range(passes):
            improved = False
            words = best.text.split()
            for i, w in enumerate(words):
                core = w.strip(".,:;?!\"'()")
                if not core or not _HEB_WORD.match(core):
                    continue
                alts = set(self.neighbours(core))
                for r in pool:                                   # the same place in other readings
                    ws = r.text.split()
                    if len(ws) == len(words):
                        alts.add(ws[i].strip(".,:;?!\"'()"))
                alts.discard(core)
                texts = [" ".join(words[:i] + [w.replace(core, a)] + words[i + 1:]) for a in alts]
                new = [t for t in texts if t not in acs]
                acs.update(zip(new, self.acoustic_many(views, new)))
                cands = sorted((t for t in texts if acs[t] >= best.acoustic - self.prune), key=lambda t: -acs[t])
                cands = cands[:self.per_word]                    # the handwriting model's likeliest few
                if not cands:
                    continue
                for t, lmv in zip(cands, self.lm.logp(cands)):
                    s = self.score(t, acs[t], lmv, alpha)
                    if s > best.score + 1e-9:
                        if trace:
                            print(f"   {best.text}  ->  {t}   (ac {acs[t] - best.acoustic:+.1f}, lm {lmv - best.lm:+.1f})")
                        best, improved = Reading(t, acs[t], lmv, s), True
                words = best.text.split()
            if not improved:
                break
        return best, pool

    def recognize(self, img, n_best: int = 3, tta: bool = True) -> list[Candidate]:
        """drop-in for HebrewRecognizer.recognize: the context reading first, then the beam's others"""
        best, pool = self.read(img, tta=tta)
        out = [Candidate(best.text, best.score, 1.0, best.acoustic)]
        for r in sorted(pool, key=lambda r: -r.score):
            if r.text != best.text and len(out) < n_best:
                out.append(Candidate(r.text, r.score, 1.0, r.acoustic))
        return out
