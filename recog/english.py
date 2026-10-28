"""English handwriting recognizer: TrOCR-small-handwritten (Microsoft, trained on IAM handwritten
lines) running on ONNX Runtime with the same beam search as the formula reader, so each reading
comes with its total log-probability. Model files: models/trocr-small-handwritten (ONNX export by
Xenova of microsoft/trocr-small-handwritten), fetched by download_models.py.

english_score() tells how much a reading looks like English words - used to decide whether a
Latin-script chunk is English text or a formula.
"""
from __future__ import annotations

import os
import re

from .formula import PROJECT, Candidate, FormulaRecognizer  # noqa: F401  (Candidate re-exported)

DEFAULT_DIR = os.path.join(PROJECT, "models", "trocr-small-handwritten")
BASE_DIR = os.path.join(PROJECT, "models", "trocr-base-handwritten")
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def default_dir() -> str:
    """the English reader to use: TrOCR-small, unless EXAM_GRADER_ENGLISH=base (TrOCR-base, fetched by
    `python download_models.py trocr-base-handwritten`: CER 4.5 % vs 6.9 % on 150 IAM test lines, 63 % vs
    58 % of 400 real handwritten multiple-choice answers read right, ~5x slower) or a model folder"""
    v = os.environ.get("EXAM_GRADER_ENGLISH", "").strip()
    if v == "base" and os.path.isfile(os.path.join(BASE_DIR, "encoder_model.onnx")):
        return BASE_DIR
    if v and os.path.isfile(os.path.join(v, "encoder_model.onnx")):
        return v
    return DEFAULT_DIR


class EnglishRecognizer(FormulaRecognizer):
    def __init__(self, model_dir: str | None = None, providers: list[str] | None = None):
        super().__init__(model_dir or default_dir(), providers)

    def recognize(self, img, n_best: int = 3, max_new_tokens: int = 64, grammar=None) -> list[Candidate]:
        out = super().recognize(img, n_best=n_best, max_new_tokens=max_new_tokens, grammar=grammar)
        for c in out:
            c.text = " ".join(c.text.split())
        return out


_known = None


def _english_words(min_zipf: float = 2.0):
    """set of English words at least as frequent as min_zipf (wordfreq; the long tail of its list is
    junk such as 'vout'), or an empty set when wordfreq is not installed"""
    global _known
    if _known is None:
        try:
            import wordfreq
            words = wordfreq.top_n_list("en", 200000)          # most frequent first
            lo, hi = 0, len(words)
            while lo < hi:                                      # first word below min_zipf
                mid = (lo + hi) // 2
                if wordfreq.zipf_frequency(words[mid], "en") >= min_zipf:
                    lo = mid + 1
                else:
                    hi = mid
            _known = set(words[:lo])
        except ImportError:
            _known = set()
    return _known


def english_score(text: str) -> float:
    """fraction of the letters that belong to known English words (0..1); 0 when there are none"""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    known = _english_words()
    if not known:                                 # no word list: count plain words of >= 2 letters
        return sum(len(w) for w in words if len(w) >= 2) / max(1, sum(len(w) for w in words))
    return sum(len(w) for w in words if w.lower() in known) / sum(len(w) for w in words)


MATH_WORDS = {"log", "ln", "sin", "cos", "tan", "cot", "sec", "csc", "exp", "lim", "max", "min", "det", "sinh",
              "cosh", "tanh", "arg", "mod", "gcd", "lcm", "sgn", "rect", "sinc"}


SHORT_WORDS = {"a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it", "me", "my",
               "no", "of", "on", "or", "so", "to", "up", "us", "we"}


def looks_english(text: str, allow_short: bool = False) -> bool:
    """a reading made of English words - not a formula ('V = IR', 'x2 + 1', 'sin x', 'RC' are not).
    allow_short: also accept a lone short word ('is', 'to') - only when another signal says English."""
    t = text.strip()
    if not t or "=" in t:
        return False
    letters = sum(ch.isalpha() for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    if letters < (1 if allow_short else 3) or digits > 0.3 * letters:
        return False
    words = _WORD.findall(t)
    long_words = [w for w in words if len(w) >= 3 and w.lower() not in MATH_WORDS]
    if not long_words and not (allow_short and words and all(w.lower() in SHORT_WORDS for w in words)):
        return False
    return english_score(t) >= 0.75
