"""Hebrew handwriting line recognizer: compact CRNN-CTC model on ONNX Runtime (CPU is enough).

recognize() returns an n-best list from CTC prefix beam search. Each candidate's `logp` is the
total probability of that text under the model (summed over all CTC alignments), which is what
the grading-confidence layer needs. Train / re-train the model with train_hebrew.py.

Word language model (shallow fusion): the CRNN alone confuses look-alike letters (כ/ב/פ, ן/ו) -
it read 'כאשר' as 'באשר' and 'ולכן' as 'ולכו' even though the right words were in its n-best list.
Each time a word is completed during the beam search, lm_weight * log P(word) + word_bonus is added
(unigram, models/hebrew-lm/words.tsv: OpenSubtitles word frequencies - deliberately not the handwriting
datasets' transcriptions, whose texts many writers copied, which would make test words "known").
Unknown words are penalised, never forbidden.

Course vocabulary: the general word list lacks technical terms (טרנזיסטור, עכבה, קיבול ...), which
would then be pushed towards common look-alike words. Every Hebrew word in
models/hebrew-lm/course_words.txt (any course text: a word list, lecture notes, old exams) counts as a
frequent word, and is also found behind up to 3 prefix letters (ו ה ב כ ל מ ש), paying prefix_logp
per letter - 'בטרנזיסטור' -> 'טרנזיסטור'. Prefixes are not stripped from general words: that did not
help on held-out writers and produced readings such as 'כאה' (= כ + 'אה').
"""
from __future__ import annotations

import io
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from . import session_options
from .hebrew_text import CHAR2ID, ID2CHAR, to_logical

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# first existing one wins: the locally trained model, then the shipped ones (best first). hebrew-crnn-cpu2 =
# hebrew-crnn-cpu trained 3 more epochs (26.10): CER with the word LM, five widths, 300 test lines 8.55 -> 8.17 %
# (chosen on held-out training writers: 8.04 -> 7.61 %)
CANDIDATE_DIRS = [os.path.join(PROJECT, "models", d) for d in ("hebrew-crnn", "hebrew-crnn-cpu2", "hebrew-crnn-cpu",
                                                               "hebrew-crnn-cpu5ep")]
DEFAULT_DIR = next((d for d in CANDIDATE_DIRS if os.path.isfile(os.path.join(d, "model.onnx"))), CANDIDATE_DIRS[0])
for _d in filter(None, [os.environ.get("EXAM_GRADER_HEBREW_MODEL", "")]):   # e.g. =models/hebrew-crnn-cpu (the previous one)
    for _p in (_d, os.path.join(PROJECT, _d)):
        if os.path.isfile(os.path.join(_p, "model.onnx")):
            DEFAULT_DIR = _p
            break
LM_DIR = os.path.join(PROJECT, "models", "hebrew-lm")
NEG_INF = -np.inf
SPACE_ID = CHAR2ID[" "]
_HEB = re.compile(r"[א-ת]+")
_WORD_SPLIT = re.compile(r"[\s/\\,.;:!?()\[\]{}<>=+*]+")       # keeps ' and " (ג'אול, ע"פ) and hyphens


@dataclass
class Candidate:
    text: str                # logical (normal reading) order
    logp: float              # log P(text | image) [+ LM terms when a word LM is used]
    min_char_p: float        # weakest character on the best path (greedy)
    acoustic_logp: float | None = None   # image-only log-probability (no LM)


class WordLM:
    """Unigram log P(word) for Hebrew words (letters only); unknown words get oov_logp."""

    PREFIXES = "והבכלמש"

    def __init__(self, lm_dir: str = LM_DIR, course_file: str | None = None):
        self.meta = json.load(open(os.path.join(lm_dir, "meta.json"), encoding="utf-8"))
        self.oov = self.meta["oov_logp"]
        self.prefix_logp = self.meta.get("prefix_logp")          # None: no prefix back-off
        self.table = {}
        with open(os.path.join(lm_dir, "words.tsv"), encoding="utf-8") as f:
            for line in f:
                w, lp = line.rstrip("\n").split("\t")
                self.table[w] = float(lp)
        ranked = sorted(self.table.values(), reverse=True)
        # course words count as frequent words (default: as frequent as the 1000th most common word)
        self.course_logp = self.meta.get("course_logp", ranked[min(999, len(ranked) - 1)])
        self.course_words = set()
        course_file = course_file or os.path.join(lm_dir, "course_words.txt")
        if os.path.isfile(course_file):
            with open(course_file, encoding="utf-8") as f:
                for line in f:
                    if line.lstrip().startswith("#"):
                        continue
                    for tok in _WORD_SPLIT.split(line):          # 'אופ-אמפ' -> אופאמפ: the reader keeps a
                        self.course_words.add(self.key(tok))     # hyphenated word as one, letters only
            self.course_words = {w for w in self.course_words if len(w) >= 2}
            for w in self.course_words:
                self.table[w] = max(self.table.get(w, NEG_INF), self.course_logp)
        self._cache = {}

    @classmethod
    def from_table(cls, table: dict, oov_logp: float, prefix_logp: float | None = None):
        """an LM from an in-memory {word: log P} table (tests, experiments)"""
        lm = cls.__new__(cls)
        lm.meta, lm.oov, lm.prefix_logp = {}, oov_logp, prefix_logp
        lm.table, lm.course_words, lm.course_logp, lm._cache = dict(table), set(), None, {}
        return lm

    @staticmethod
    def key(word: str) -> str:
        word = unicodedata.normalize("NFC", word)
        return "".join(_HEB.findall("".join(ch for ch in word if unicodedata.category(ch) != "Mn")))

    def _lookup(self, k: str) -> float:
        best = self.table.get(k, self.oov)                       # a rare form in the list ('בטרנזיסטור')
        if self.prefix_logp is not None and self.course_words:  # still gets its course-word stem's score
            for i in range(1, min(3, len(k) - 3) + 1):           # a course word of >= 3 letters behind
                if k[i - 1] not in self.PREFIXES:                # 1-3 prefix letters
                    break
                if k[i:] in self.course_words:
                    best = max(best, self.table[k[i:]] + i * self.prefix_logp)
        return best

    def logp(self, word: str):
        k = self.key(word)
        if not k:
            return None                                          # no Hebrew letters -> no LM term
        if k not in self._cache:
            self._cache[k] = self._lookup(k)
        return self._cache[k]


def _lse(*xs):
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + np.log(sum(np.exp(x - m) for x in xs))


def ctc_prefix_beam_search(logp: np.ndarray, beam: int = 12, n_best: int = 3, prune: float = -8.0,
                           lm: WordLM | None = None, lm_weight: float = 0.0, word_bonus: float = 0.0):
    """logp: [T, K] log-probabilities, blank = 0. Prefixes are in VISUAL order (CTC reads left to
    right), so a finished word is reversed back to logical order before the LM lookup.
    Returns [(ids, total score, acoustic score)] best first."""
    use_lm = lm is not None and lm_weight != 0.0
    cache = {}

    def word_term(ids):                               # ids of one finished word, visual order
        if not ids:
            return 0.0
        if ids not in cache:
            lp = lm.logp(to_logical("".join(ID2CHAR.get(i, "") for i in ids)))
            cache[ids] = 0.0 if lp is None else lm_weight * lp + word_bonus
        return cache[ids]

    def last_word(prefix):
        try:
            k = len(prefix) - 1 - prefix[::-1].index(SPACE_ID)
        except ValueError:
            k = -1
        return prefix[k + 1:]

    beams = {(): (0.0, NEG_INF)}                      # prefix -> (log P ending in blank, ... in non-blank)
    lm_score = {(): 0.0}                              # prefix -> LM terms of its finished words
    for t in range(logp.shape[0]):
        row = logp[t]
        cands = np.where(row > prune)[0]
        if len(cands) == 0:
            cands = [int(row.argmax())]
        nxt = defaultdict(lambda: (NEG_INF, NEG_INF))
        for prefix, (pb, pnb) in beams.items():
            for c in cands:
                p = row[c]
                if c == 0:
                    b, nb = nxt[prefix]
                    nxt[prefix] = (_lse(b, pb + p, pnb + p), nb)
                    continue
                newp = prefix + (int(c),)
                if use_lm and newp not in lm_score:
                    finished = c == SPACE_ID and prefix and prefix[-1] != SPACE_ID
                    lm_score[newp] = lm_score[prefix] + (word_term(last_word(prefix)) if finished else 0.0)
                b, nb = nxt[newp]
                if prefix and prefix[-1] == c:
                    nxt[newp] = (b, _lse(nb, pb + p))          # repeat needs a blank in between
                    ob, onb = nxt[prefix]
                    nxt[prefix] = (ob, _lse(onb, pnb + p))     # collapsed repeat
                else:
                    nxt[newp] = (b, _lse(nb, pb + p, pnb + p))
        if use_lm:
            beams = dict(sorted(nxt.items(), key=lambda kv: -(_lse(*kv[1]) + lm_score.get(kv[0], 0.0)))[:beam])
        else:
            beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    res = []
    for pre, v in beams.items():
        ac = _lse(*v)
        total = ac
        if use_lm:
            total += lm_score.get(pre, 0.0) + (word_term(last_word(pre)) if pre and pre[-1] != SPACE_ID else 0.0)
        res.append((pre, total, ac))
    res.sort(key=lambda x: -x[1])
    return res[:n_best]


class HebrewRecognizer:
    def __init__(self, model_dir: str = DEFAULT_DIR, lm_dir: str | None = LM_DIR,
                 lm_weight: float | None = None, word_bonus: float | None = None):
        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"no model in {model_dir} - run train_hebrew.py (or setup.cmd)")
        self.model_dir = model_dir
        self.meta = json.load(open(os.path.join(model_dir, "meta.json"), encoding="utf-8"))
        self.sess = ort.InferenceSession(onnx_path, session_options(), providers=["CPUExecutionProvider"])
        self.H = self.meta["height"]
        self.maxw = self.meta.get("max_width", 1600)
        self.device = "cpu (onnxruntime)"
        has_lm = lm_dir and all(os.path.isfile(os.path.join(lm_dir, f)) for f in ("words.tsv", "meta.json"))
        self.lm = WordLM(lm_dir) if has_lm else None
        lm_meta = self.lm.meta if self.lm else {}
        self.lm_weight = lm_weight if lm_weight is not None else lm_meta.get("lm_weight", 0.0)
        self.word_bonus = word_bonus if word_bonus is not None else lm_meta.get("word_bonus", 0.0)

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
        w, h = img.size
        nw = max(16, min(self.maxw, int(round(w * self.H / h))))
        nw = (nw + 3) // 4 * 4
        img = img.resize((nw, self.H), Image.BILINEAR)
        x = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
        return x[None, None]

    def log_probs(self, img: Image.Image | str | bytes) -> np.ndarray:
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        elif isinstance(img, str):
            img = Image.open(img)
        return self.sess.run(None, {"image": self.preprocess(img)})[0][:, 0, :]    # [T, K]

    TTA_SCALES = (1.0, 0.8, 0.9, 1.1, 1.2)     # chosen on held-out TRAINING writers (micro CER, 300 lines): one look
    #                                            9.3 %, (0.85, 1.15) 8.3 %, (0.9, 1.1) 8.5 %, these five 8.0 %

    def recognize(self, img, n_best: int = 3, beam: int = 12, use_lm: bool = True, tta: bool = False) -> list[Candidate]:
        """tta: also read the line squeezed and stretched (80-120 % width, TTA_SCALES) and pool the n-best
        lists - a reading found in several views gets their evidence summed (~5x the time). For whole
        Hebrew lines only (300 test lines: CER 9.1 -> 8.6 %): on the short Hebrew pieces of mixed
        Hebrew / formula lines it changed the confidences the layout decisions use, and made them worse
        (mixed_eval3: layout right 82.1 -> 79.8 %), so the mixed reader never asks for it."""
        if tta:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                img = Image.open(img)
            pools, info = {}, {}
            for s in self.TTA_SCALES:
                v = img if s == 1.0 else img.resize((max(16, int(img.width * s)), img.height), Image.BILINEAR)
                for c in self.recognize(v, n_best=max(n_best, 5), beam=beam, use_lm=use_lm, tta=False):
                    pools.setdefault(c.text, []).append(c.logp)
                    if c.text not in info or s == 1.0:
                        info[c.text] = c
            scored = []
            for t, xs in pools.items():
                m = max(xs)
                pooled = m + float(np.log(sum(np.exp(x - m) for x in xs))) + float(np.log(len(xs)))
                c = info[t]
                scored.append(Candidate(t, pooled, c.min_char_p, c.acoustic_logp))
            scored.sort(key=lambda c: -c.logp)
            return scored[:n_best]
        lp = self.log_probs(img)
        best_path = lp.argmax(-1)
        chars = [lp[t, k] for t, k in enumerate(best_path) if k != 0]
        min_p = float(np.exp(min(chars))) if chars else 1.0
        lm = self.lm if use_lm else None
        out = []
        for ids, score, ac in ctc_prefix_beam_search(lp, beam=beam, n_best=n_best, lm=lm,
                                                     lm_weight=self.lm_weight, word_bonus=self.word_bonus):
            visual = "".join(ID2CHAR.get(i, "") for i in ids)
            out.append(Candidate(" ".join(to_logical(visual).split()), float(score), min_p, float(ac)))
        return out
