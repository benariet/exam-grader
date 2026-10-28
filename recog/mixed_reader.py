"""Read a mixed Hebrew + math answer: layout -> both recognizers -> cross-check.

The Hebrew/math decision is the weakest link: a classifier can be *confidently* wrong (one
version called a clean 'V=5V' Hebrew at 99 % and the answer was auto-graded wrong). So every
segment is also read by the *other* recognizer, and two independent signals must agree:

  * classified Hebrew, but the formula reader returns a plain, confident formula
    (only ordinary symbols, contains '=' or a digit)                 -> conflict
  * classified math, but the Hebrew reader returns >= 3 Hebrew letters at high confidence -> conflict

A conflict caps the segment's layout confidence at CONFLICT_CAP, which routes the answer to a
human (grader.mixed.grade_mixed reviews anything below min_layout_conf).

English: a Latin-script chunk is English text rather than a formula when the English reader
(recog/english.py) returns English words with confidence and the formula reader does not return a
plain formula. The Hebrew/math classifier never saw English handwriting and calls most cursive
English Hebrew, so chunks it calls Hebrew get a second classifier, Hebrew vs English
(models/english-classifier, train_english_classifier.py): English when it says so and the English
reader agrees; when only one of the two says English, the answer goes to review.

English only (MixedReader(language="english"), an exam written in English): no Hebrew at all - the
Hebrew reader is not run, and what the Hebrew/math classifier calls Hebrew (it has seen only Hebrew
and math) is text, i.e. English; the English reader and the formula reader cross-check each other
the same way (a disagreement -> review).

Split formulas: a student who leaves wide gaps ('V  =  5V') gets the formula cut into chunks, and a
lone 'V' or '5V' between Hebrew words may be classified Hebrew. An operator cannot stand alone, so a
math chunk that reads as a bare operator is joined with its neighbours (unless they are clearly
Hebrew) and re-read; if the joined reading is a plain formula it replaces the pieces (-> review).
"""
from __future__ import annotations

import math
import os
import re
import time

import numpy as np
from PIL import Image

from .english import DEFAULT_DIR as EN_DIR, EnglishRecognizer, english_score, looks_english
from .formula import FormulaRecognizer
from .hebrew import HebrewRecognizer
from .segment import CLS_DIR, PROJECT, ScriptClassifier, Segment, layout, pad_crop

CONFLICT_CAP = 0.5
EN_CLS_DIR = os.path.join(PROJECT, "models", "english-classifier")
# English decision - tuned on development sets only (data/mixed_dev_en, data/mixed_eval_en, data/mixed_eval;
# tune_english_rules.py) for safety: a wrong layout that is not sent to review costs as much as 5 reviews
EN_MIN_CONF = 0.90          # English reading: per-character probability to count as English
EN_SURE_CONF = 0.90         # ... and to trust it without review
EN_DOUBT_CONF = 0.80        # ... English words read at this level in a Hebrew / math chunk -> review
EN_AGREE = 0.30             # Hebrew/English classifier: enough for English when the English reader agrees
EN_ALONE = 0.90             # ... English on its own (then -> review)
EN_CONTEXT = 0.70           # ... English next to an English chunk (then -> review)
EN_DOUBT = 0.50             # ... the reader does not agree: keep Hebrew, but -> review
EN_BOUNDARY = 0.30          # ... a Hebrew / math chunk right next to an English word -> review
COLUMN_GAP = 4.0            # a gap wider than this x the letter height separates two formulas (columns)
# whole runs (the chunks between column gaps) read at once - the readers do much better with context
RUN_F_CONF = 0.80           # formula reader on the run: per-token probability for "this run is one formula"
RUN_E_CONF = 0.90           # English reader on the run: per-character probability for "one English phrase"
RUN_MIN_PEN = 0.10          # ... and no chunk of it clearly Hebrew to the Hebrew/English classifier (below this:
#                             77 % of Hebrew chunks, 14 % of formula chunks on the development sets)
RUN_HEB_CONF = 0.50         # ... and (formula) no chunk the Hebrew reader reads as a word this confidently
#                             (79 % of Hebrew chunks, 11 % of formula chunks)
#                             a run read differently as a whole than piece by piece always goes to review
TINY = 0.6                  # chunks narrower than this x the ink height (dots, commas) take no English decision
#                             of their own: inside an English phrase they join it, at its edge -> review
# ordinary exam-math tokens. Deliberately NOT included: what the formula reader tends to emit on Hebrew
# handwriting (\\cap, \\mathbb, \\mathbf, \\underset, \\varrho, \\it, \\sim, \\geq, \\sigma, \\rho, \\gamma ...)
_PLAIN = re.compile(r"^(?:[A-Za-z0-9=+\-.,()/^_{}|' ]|\\(?:cdot|times|frac|sqrt|pi|omega|Omega|mu|tau|alpha|beta|"
                    r"theta|phi|Delta|delta|lambda|epsilon|approx|left|right|mathrm|text|int|sum|prod|infty|lim|log|ln|"
                    r"sin|cos|tan|exp|hat|bar|vec|dot|prime|partial|nabla|to|cdots|ldots|,|;|!| ))+$")
_HEB = re.compile(r"[א-ת]")


_OPERATOR_ONLY = re.compile(r"^\s*(?:=|\+|-|\\cdot|\\times|/|\\approx|<|>|\\leq?|\\geq?|\\neq)\s*$")
NEIGHBOUR_MIN_P_MATH = 0.15         # a neighbour below this (raw classifier) is clearly Hebrew: never joined


def _acoustic(c) -> float:
    """image-only log-probability of a candidate (Hebrew candidates also carry LM terms in .logp)"""
    a = getattr(c, "acoustic_logp", None)
    return c.logp if a is None else a


def looks_looped(tex: str, repeats: int = 5) -> bool:
    """the formula decoder sometimes loops on writing it cannot read ('\\log _ { 4 } \\log _ { 4 } ...',
    or a long '\\put ( 1 , 0 ) { \\line ( 1 , 0 ) { 1 0 } }' again and again): a short unit repeated
    `repeats` times, a unit of 4+ tokens 4 times"""
    toks = tex.split()
    for n in range(1, 41):
        need = repeats if n < 4 else min(repeats, 4)
        for i in range(0, len(toks) - n * need + 1):
            unit = toks[i:i + n]
            if all(t.isdigit() or t in ".," for t in unit):     # 1 0 0 0 0 0 is a number, not a loop
                continue
            if all(toks[i + k * n: i + (k + 1) * n] == unit for k in range(1, need)):
                return True
    return False


def math_plausible(tex: str) -> bool:
    t = tex.strip()
    return bool(t) and bool(_PLAIN.match(t)) and ("=" in t or any(ch.isdigit() for ch in t))


class EnsembleClassifier:
    """Average of calibrated logits of several ScriptClassifiers (each with its own crop mode)."""

    def __init__(self, dirs):
        self.models = [ScriptClassifier(d) for d in dirs]

    def p_math(self, crop: Image.Image) -> float:
        z = []
        for m in self.models:
            p = min(max(m.p_math(crop), 1e-6), 1 - 1e-6)
            z.append(math.log(p / (1 - p)))
        return float(1 / (1 + math.exp(-np.mean(z))))


def default_classifier():
    """models/script-classifier (+ models/script-classifier-tight as a second opinion, if present)."""
    dirs = [d for d in (CLS_DIR, os.path.join(PROJECT, "models", "script-classifier-tight"))
            if os.path.isfile(os.path.join(d, "model.onnx"))]
    return EnsembleClassifier(dirs) if len(dirs) > 1 else ScriptClassifier(dirs[0])


def _hebrew_word(c, min_conf: float | None = None) -> bool:
    """chunk [.., heb_strong (9), Hebrew reading (10), its confidence (11)]: the Hebrew reader reads a
    word here - surely (heb_strong), or at least min_conf per letter. One letter does not count: it
    reads a lone digit or '+' as a Hebrew letter quite confidently."""
    if len(_HEB.findall(c[10] if len(c) > 10 else "")) < 2:
        return False
    return bool(c[9]) if min_conf is None else (len(c) > 11 and c[11] >= min_conf)


def _letter_height(ink: np.ndarray) -> float:
    """median height of the strokes in a line (descenders and tall marks do not count much)"""
    from scipy import ndimage as ndi
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if not n:
        return float(ink.shape[0])
    hs = np.array([o[0].stop - o[0].start for o in ndi.find_objects(lab)])
    areas = np.asarray(ndi.sum(ink, lab, range(1, n + 1)))
    ok = areas >= 20
    return float(np.median(hs[ok])) if ok.any() else float(ink.shape[0])


class MixedReader:
    def __init__(self, clf=None, heb=None, frm=None, eng=None, join_split_formulas: bool = True,
                 english: bool = True, whole_runs: bool = True, language: str = "hebrew", hebrew_tta: bool | None = None):
        """language: "hebrew" - Hebrew, math and (english=True) English; "english" - English and math only.
        hebrew_tta: re-read the text of every Hebrew piece at five widths after the layout is decided - on
        (EXAM_GRADER_HEBREW_TTA=0 turns it off): whole Hebrew lines CER 9.1 -> 8.6 %, the Hebrew pieces of
        mixed_eval3 30.0 -> 29.8 % with the same layout and the same answers flagged"""
        if hebrew_tta is None:
            hebrew_tta = os.environ.get("EXAM_GRADER_HEBREW_TTA", "1") not in ("", "0")
        self.hebrew_tta = hebrew_tta
        # Hebrew text read in context (recog/hebrew_context.py: + a Hebrew language model) - opt-in, it needs
        # models/dictalm-3.0-1.7b-base and is slow without a GPU (~5 s per line on 2 CPU cores)
        self.hebrew_context = os.environ.get("EXAM_GRADER_HEBREW_CONTEXT", "0") not in ("", "0")
        self._ctx = None
        # a second formula reader for every math piece (recog/mumer.py: Uni-MuMER 2B) - opt-in, it needs
        # models/uni-mumer-qwen3vl-2b and a GPU for speed; its reading is kept as seg.math2 and weighed with
        # the first reader's by the math validation (grader/selfcheck.py)
        self.formula2 = os.environ.get("EXAM_GRADER_FORMULA2", "").strip().lower() == "mumer"
        self._mumer = None
        # the text of every English piece re-read by a vision-language model (recog/hebrew_vlm.py with the
        # English prompt; Qwen3-VL-8B reads IAM lines at ~2 % CER, TrOCR-small at ~7 %) - opt-in:
        # EXAM_GRADER_ENGLISH_VLM=<model folder> (1: models/qwen3-vl-8b; 4-bit when it does not fit the GPU in
        # 16 bits). The layout is still decided with TrOCR's reading, only the piece's text is replaced.
        v = os.environ.get("EXAM_GRADER_ENGLISH_VLM", "").strip()
        self.english_vlm = (os.path.join(PROJECT, "models", "qwen3-vl-8b") if v == "1" else v) or None
        self._en_vlm = None
        # these three heavy readers can wait until a whole document's layout is read (defer_heavy - set by
        # read_page.read_pages): finish() then runs each over all its pieces, one model on the GPU at a time
        # (an 11 GB card holds any one of them, not all three)
        self.defer_heavy = False
        self._pending = []
        if language not in ("hebrew", "english"):
            raise ValueError(f"language: 'hebrew' or 'english', not {language!r}")
        self.english_only = language == "english"
        if self.english_only:
            english = True
        self.clf = clf or default_classifier()
        self.heb = None if self.english_only else (heb or HebrewRecognizer())
        self.frm = frm or FormulaRecognizer()
        if eng is None and english and os.path.isfile(os.path.join(EN_DIR, "encoder_model.onnx")):
            eng = EnglishRecognizer()
        self.eng = eng if english else None        # None: Hebrew / math only
        if self.english_only and self.eng is None:
            raise RuntimeError("English only, but the English reader is not installed (models/trocr-small-handwritten)")
        self.en_clf = (ScriptClassifier(EN_CLS_DIR) if self.eng is not None and not self.english_only
                       and os.path.isfile(os.path.join(EN_CLS_DIR, "model.onnx")) else None)
        self.join_split_formulas = join_split_formulas
        self.whole_runs = whole_runs

    # -- chunk-level second opinion (before merging) -------------------------------------
    def _opinions(self, crop, english: bool = True):
        f = self.frm.recognize(crop, n_best=1)[0]
        f_conf = math.exp(f.logp / max(1, len(f.text.split())))
        if getattr(self, "heb", None) is not None:
            h = self.heb.recognize(crop, n_best=1)[0]
            h_text, h_conf = h.text, math.exp(_acoustic(h) / max(1, len(h.text)))
        else:                                      # English only: no Hebrew opinion
            h_text, h_conf = "", 0.0
        toks = f.text.split()
        math_strong = (math_plausible(f.text) and ("=" in f.text or len(toks) >= 2)
                       and f_conf >= 0.9 and f.min_token_p >= 0.5)
        heb_strong = len(_HEB.findall(h_text)) >= 1 and h_conf >= 0.85
        eng, eng_sure, eng_short, eng_doubt, e_text, e_conf = False, False, False, False, "", 0.0
        if self.eng is not None and english:
            e = self.eng.recognize(crop, n_best=1)[0]
            e_conf = math.exp(e.logp / max(1, len(e.text)))
            e_text = e.text
            ok = e.min_token_p >= 0.3 and not math_strong
            words, short = looks_english(e.text), looks_english(e.text, allow_short=True)   # 'is', 'to'
            eng = ok and e_conf >= EN_MIN_CONF and words
            eng_short = ok and e_conf >= EN_MIN_CONF and short                    # needs another signal
            eng_doubt = ok and e_conf >= EN_DOUBT_CONF and (words or short)       # enough to ask a human
            eng_sure = eng and e_conf >= EN_SURE_CONF and english_score(e.text) >= 0.9 and not math_plausible(f.text)
        return {"math": math_strong and not heb_strong, "heb": heb_strong and not math_strong, "eng": eng,
                "eng_sure": eng_sure, "eng_short": eng_short, "eng_doubt": eng_doubt,
                "f": f.text, "h": h_text, "h_conf": h_conf, "e": e_text, "e_conf": e_conf if e_text else 0.0}

    def layout_checked(self, page, bands=None):
        """Like segment.layout(), plus a second opinion per chunk: a clear disagreement corrects an
        uncertain chunk and always caps the confidence (-> human review). bands: [(y0, y1)] line
        bands found elsewhere (recog.page.find_lines); default: rows with ink separated by gaps."""
        from .segment import segment_chunks, segment_lines, smooth_line
        lines = []
        for y0, y1 in (bands or segment_lines(page)):
            line = page.crop((0, y0, page.width, y1))
            boxes = segment_chunks(line)
            raw = [self.clf.p_math(line.crop((x0, 0, x1, line.height))) for x0, x1 in boxes]
            pm = smooth_line(raw)
            ink = np.asarray(line.convert("L")) < 128
            rows = np.where(ink.any(1))[0]
            ink_h = (rows.max() - rows.min() + 1) if len(rows) else line.height
            chunks = []
            for (x0, x1), p, p_raw in zip(boxes, pm, raw):
                kind, pk, note = ("math" if p >= 0.5 else "hebrew"), max(p, 1 - p), ""
                tiny = (x1 - x0) < TINY * ink_h
                if getattr(self, "english_only", False):
                    chunks.append(self._english_only_chunk(line, x0, x1, p, p_raw, tiny))
                    continue
                p_en = (self.en_clf.p_math(line.crop((x0, 0, x1, line.height)))
                        if self.en_clf is not None and not tiny else 0.0)
                # the English reader reads every chunk that is not a dot or a comma: even where the
                # classifier sees no English, confident English words from it send the answer to review
                need_en = self.eng is not None and not tiny
                op = self._opinions(pad_crop(line, (x0, 0, x1, line.height)), english=need_en)
                if kind == "hebrew" and op["math"]:
                    note = f"formula reader: {op['f']}"
                    if p > 0.15:
                        kind = "math"
                elif kind == "math" and op["heb"]:
                    note = f"Hebrew reader: {op['h']}"
                    if p < 0.85:
                        kind = "hebrew"
                if note:
                    pk = min(pk, CONFLICT_CAP)
                if kind == "math" and not note and not tiny:
                    if op["eng"]:                                           # Latin script: words, not a formula
                        kind = "english"
                        if not op["eng_sure"]:
                            pk, note = min(pk, CONFLICT_CAP), f"English or a formula? English reader: {op['e']}"
                    elif op["eng_doubt"]:
                        pk, note = min(pk, CONFLICT_CAP), f"formula or English? English reader: {op['e']}"
                elif kind == "hebrew" and not op["heb"] and not note and not tiny and self.en_clf is not None:
                    if p_en >= EN_AGREE and (op["eng"] or op["eng_short"]):
                        kind, pk = "english", 1 - (1 - p_en) * (1 - EN_MIN_CONF)    # classifier + reader agree
                    elif p_en >= EN_ALONE:
                        kind, pk, note = "english", CONFLICT_CAP, f"English? English reader: {op['e']}"
                    elif p_en >= EN_DOUBT or op["eng_doubt"]:
                        pk, note = min(pk, CONFLICT_CAP), f"Hebrew or English? English reader: {op['e']}"
                chunks.append([x0, x1, kind, pk, note, p_raw, op["f"], p_en, tiny, op["heb"], op["h"], op.get("h_conf", 0.0)])
            self._english_neighbours(chunks, boundary=not getattr(self, "english_only", False))
            if self.join_split_formulas:
                chunks = self._join_split_formulas(line, chunks)
            lh = _letter_height(ink)
            if getattr(self, "whole_runs", False):
                self._whole_runs(line, chunks, lh)
            merged = []
            for c in chunks:              # neighbours of one kind form one segment (two formulas a column apart: two)
                if merged and merged[-1][2] == c[2] and (c[2] != "math" or c[0] - merged[-1][1] <= COLUMN_GAP * lh):
                    merged[-1][1] = c[1]; merged[-1][3] = min(merged[-1][3], c[3])
                    merged[-1][4] = "; ".join(x for x in (merged[-1][4], c[4]) if x)
                else:
                    merged.append(list(c))
            order = merged if getattr(self, "english_only", False) else reversed(merged)   # reading order
            lines.append([(k, (a, y0, b, y1), pk, note) for a, b, k, pk, note, *_ in order])
        return lines

    def _whole_runs(self, line, chunks, lh):
        """A run of chunks between column gaps is also read as a whole, by the formula reader and by
        the English reader: an arithmetic line written with gaps between the characters, or an
        English sentence, is read far better at once than piece by piece. The run becomes one
        formula when the formula reader returns a plain formula with '=' confidently (a narrow first
        chunk - an item number like '2)' - may be left out), or one English phrase when the English
        reader returns English words confidently and no chunk of the run is clearly Hebrew."""
        runs, cur = [], [0]
        for i in range(1, len(chunks)):
            if chunks[i][0] - chunks[i - 1][1] > COLUMN_GAP * lh:
                runs.append(cur); cur = [i]
            else:
                cur.append(i)
        runs.append(cur)
        for run in runs:
            cs = [chunks[i] for i in run]
            real = [c for c in cs if not c[8]]
            if not real:
                continue
            if max(c[5] for c in cs) >= 0.2:                                   # some chunk may be math
                cands = [cs] + ([cs[1:]] if len(cs) > 1 and cs[0][1] - cs[0][0] < 1.2 * lh else [])
                best = None
                for cand in cands:
                    if not all(c[7] >= RUN_MIN_PEN and not _hebrew_word(c, RUN_HEB_CONF) for c in cand if not c[8]):
                        continue                                 # a piece looks clearly Hebrew: not one formula
                    f = self.frm.recognize(pad_crop(line, (cand[0][0], 0, cand[-1][1], line.height)), n_best=1)[0]
                    conf = math.exp(f.logp / max(1, len(f.text.split())))
                    if (math_plausible(f.text) and "=" in f.text and conf >= RUN_F_CONF and f.min_token_p >= 0.3
                            and not looks_looped(f.text) and (best is None or conf > best[0])):
                        best = (conf, cand, f.text)
                if best:
                    conf, cand, text = best
                    if any(c[2] != "math" for c in cand):        # the pieces disagreed: the whole decides,
                        for c in cand:                           # and a human checks (a Hebrew word read
                            c[2], c[3] = "math", CONFLICT_CAP    # into a formula must never pass unseen)
                            c[4] = "; ".join(x for x in (f"read as one formula: {text}", c[4]) if x)
                    continue
            if self.eng is not None and (self.en_clf is not None or getattr(self, "english_only", False)) and all(c[7] >= RUN_MIN_PEN and not _hebrew_word(c) for c in real):
                e = self.eng.recognize(pad_crop(line, (cs[0][0], 0, cs[-1][1], line.height)), n_best=1)[0]
                conf = math.exp(e.logp / max(1, len(e.text)))
                if looks_english(e.text) and conf >= RUN_E_CONF and e.min_token_p >= 0.3 \
                        and any(c[2] != "english" for c in cs):
                    for c in cs:
                        c[2], c[3] = "english", CONFLICT_CAP
                        c[4] = "; ".join(x for x in (f"read as one English phrase: {e.text}", c[4]) if x)

    def _english_only_chunk(self, line, x0, x1, p, p_raw, tiny):
        """English-only exam: a chunk is a formula or English text. The script classifier's 'Hebrew'
        means 'text' here; the formula reader and the English reader cross-check it like in the
        Hebrew mode (a disagreement caps the confidence -> review)."""
        kind, pk, note = ("math" if p >= 0.5 else "english"), max(p, 1 - p), ""
        if not tiny:
            op = self._opinions(pad_crop(line, (x0, 0, x1, line.height)), english=True)
            if kind == "english" and op["math"]:
                note = f"formula reader: {op['f']}"
                if p > 0.15:
                    kind = "math"
            elif kind == "math" and op["eng"]:
                kind = "english"
                if not op["eng_sure"]:
                    note = f"English or a formula? English reader: {op['e']}"
            elif kind == "math" and op["eng_doubt"]:
                note = f"formula or English? English reader: {op['e']}"
            elif kind == "english" and op["e"].strip() and not re.search(r"[A-Za-z]", op["e"]) \
                    and re.search(r"\d", op["e"]):
                kind = "math"                          # digits only ('1 5'): a number, which the formula reader reads
                if not math_plausible(op["f"]):
                    note = f"a number? English reader: {op['e']}, formula reader: {op['f']}"
            elif kind == "english" and not (op["eng"] or op["eng_short"]):
                note = f"text or a formula? English reader: {op['e']} (formula reader: {op['f']})"
            if note:
                pk = min(pk, CONFLICT_CAP)
            f_text = op["f"]
        else:
            f_text = ""
        # p_en 1.0 / no Hebrew word: the whole-run checks then only ask the readers themselves
        return [x0, x1, kind, pk, note, p_raw, f_text, 1.0, tiny, False, "", 0.0]

    @staticmethod
    def _english_neighbours(chunks, boundary: bool = True):
        """Line-level English rules on chunks [x0, x1, kind, p_kind, note, p_raw, ftxt, p_en, tiny,
        heb_strong] (left to right): context, dots/commas, boundaries. tune_english_rules.py mirrors this."""
        def note(c, text):
            c[3] = min(c[3], CONFLICT_CAP)
            c[4] = "; ".join(x for x in (text, c[4]) if x)

        def english_at(j):
            return 0 <= j < len(chunks) and chunks[j][2] == "english" and not chunks[j][8]
        for i, c in enumerate(chunks):        # an unsure chunk next to an English word is English too
            if c[2] == "hebrew" and not c[8] and not c[9] and c[7] >= EN_CONTEXT and (english_at(i - 1) or english_at(i + 1)):
                c[2] = "english"
                note(c, "English next to English")
        for i, c in enumerate(chunks):        # a dot / comma / single letter next to English words
            if not c[8] or c[2] == "english":
                continue
            left = next((j for j in range(i - 1, -1, -1) if not chunks[j][8]), None)
            right = next((j for j in range(i + 1, len(chunks)) if not chunks[j][8]), None)
            le, ri = left is not None and chunks[left][2] == "english", right is not None and chunks[right][2] == "english"
            if le and ri:                     # inside an English phrase: part of it
                c[2], c[3] = "english", min(chunks[left][3], chunks[right][3], c[3] if c[4] else 1.0)
            elif le or ri:                    # at its edge: could belong to either side
                note(c, "a mark next to an English word")
        for i, c in enumerate(chunks):        # a chunk next to an English word that might be English too
            if boundary and c[2] != "english" and not c[8] and c[7] >= EN_BOUNDARY and (english_at(i - 1) or english_at(i + 1)):
                note(c, "next to an English word - English too?")

    def _join_split_formulas(self, line, chunks):
        """chunks: [x0, x1, kind, p_kind, note, p_math_raw, formula_text] left to right."""
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(chunks):
                if c[2] != "math" or not _OPERATOR_ONLY.match(c[6]):
                    continue
                lo = i - 1 if i > 0 and chunks[i - 1][5] >= NEIGHBOUR_MIN_P_MATH else i
                hi = i + 1 if i + 1 < len(chunks) and chunks[i + 1][5] >= NEIGHBOUR_MIN_P_MATH else i
                if lo == hi:
                    continue
                x0, x1 = chunks[lo][0], chunks[hi][1]
                f = self.frm.recognize(pad_crop(line, (x0, 0, x1, line.height)), n_best=1)[0]
                if not (math_plausible(f.text) and len(f.text.split()) > len(c[6].split())):
                    continue
                note = f"formula pieces joined: {f.text}"
                joined = [x0, x1, "math", CONFLICT_CAP, note, max(ch[5] for ch in chunks[lo:hi + 1]), f.text,
                          1.0 if getattr(self, "english_only", False) else 0.0, False, False, "", 0.0]
                chunks = chunks[:lo] + [joined] + chunks[hi + 1:]
                changed = True
                break
        return chunks

    def read(self, page: Image.Image, n_best: int = 3, cross_check: bool = True, bands=None):
        """-> (lines, segments): lines = [[Segment, ...] in reading order], segments flattened.
        bands: optional [(y0, y1)] line bands (e.g. the whole image when it is one line)."""
        page = page.convert("L")
        lines, segments = [], []
        checked = self.layout_checked(page, bands) if cross_check else [[(k, b, p, "") for k, b, p in ln] for ln in layout(page, self.clf)]
        for line in checked:
            segs = []
            for kind, box, p_kind, chunk_note in line:
                crop = pad_crop(page, box)
                if kind == "english":
                    cands = self.eng.recognize(crop, n_best=n_best)
                    seg = Segment(kind, box, p_kind, [(c.text, c.logp) for c in cands])
                    seg.note = chunk_note
                    if self.english_vlm:
                        self._heavy_job("english", seg, crop, n_best)
                    segs.append(seg); segments.append(seg)
                    continue
                heb = (self.heb.recognize(crop, n_best=n_best) if getattr(self, "heb", None) is not None
                       and (kind == "hebrew" or cross_check) else None)
                frm = self.frm.recognize(crop, n_best=n_best) if (kind == "math" or cross_check) else None
                note = chunk_note
                if cross_check and not note and heb is not None:
                    f0 = frm[0]
                    f_conf = math.exp(f0.logp / max(1, len(f0.text.split())))
                    h0 = heb[0]
                    h_conf = math.exp(_acoustic(h0) / max(1, len(h0.text)))
                    if kind == "hebrew" and math_plausible(f0.text) and f_conf >= 0.9 and f0.min_token_p >= 0.5:
                        note = f"formula reader is confident: {f0.text}"
                    elif kind == "math" and len(_HEB.findall(h0.text)) >= 3 and h_conf >= 0.9:
                        note = f"Hebrew reader is confident: {h0.text}"
                    if note:
                        p_kind = min(p_kind, CONFLICT_CAP)
                cands = heb if kind == "hebrew" else frm
                in_context = kind == "hebrew" and self.hebrew_context and self.heb is not None
                if kind == "hebrew" and self.hebrew_tta and not (in_context and not self.defer_heavy):
                    # the layout is decided (above, with one look); only the text of a Hebrew piece is re-read at
                    # five widths (recog/hebrew.py, tta) - the decisions keep the confidences they were tuned on
                    cands = self.heb.recognize(crop, n_best=n_best, tta=True)
                if kind == "math" and looks_looped(cands[0].text):
                    note = "; ".join(x for x in (note, "formula reader looped - unreadable") if x)
                    p_kind = min(p_kind, CONFLICT_CAP)
                seg = Segment(kind, box, p_kind, [(c.text, c.logp) for c in cands])
                seg.note = note
                if in_context:                           # the text in context: handwriting x Hebrew language model
                    self._heavy_job("hebrew", seg, crop, n_best)
                if kind == "math" and self.formula2:
                    self._heavy_job("math2", seg, crop, 1)
                if kind != "math" and frm is not None:   # what the formula reader made of it (grader/selfcheck.py
                    seg.formula_nbest = [(c.text, c.logp) for c in frm]     # looks there for a missing value)
                segs.append(seg); segments.append(seg)
            lines.append(segs)
        return lines, segments

    # -- heavy readers (Hebrew in context, Uni-MuMER, an English VLM) ----------------------
    HEAVY = ("hebrew", "math2", "english")

    def _heavy_job(self, name, seg, crop, n_best):
        if self.defer_heavy:
            self._pending.append((name, seg, crop, n_best))
        else:
            self._heavy_apply(name, seg, self._heavy_read(name, crop, n_best))

    @staticmethod
    def _heavy_apply(name, seg, out):
        if name == "math2":
            seg.math2 = out                      # weighed with the first reading by grader/selfcheck.py
        else:
            seg.nbest = out

    def _heavy_reader(self, name, park_others: bool = False):
        """the heavy reader `name`, loaded on first use; park_others: the other heavy readers leave the GPU"""
        if park_others:
            for other, r in zip(self.HEAVY, (self._ctx.lm if self._ctx is not None else None, self._mumer, self._en_vlm)):
                if other != name and r is not None and hasattr(r, "park"):
                    r.park()
        if name == "hebrew":
            if self._ctx is None:
                from .hebrew_context import HebrewContext
                self._ctx = HebrewContext(rec=self.heb)
            return self._ctx
        if name == "math2":
            if self._mumer is None:
                from .mumer import MumerReader
                self._mumer = MumerReader()
            return self._mumer
        if self._en_vlm is None:
            from . import gpu
            from .hebrew_vlm import HebrewRecognizer as VisionLanguageReader
            self._en_vlm = VisionLanguageReader(self.english_vlm, load_in_4bit=gpu.needs_4bit(gpu.params_b(self.english_vlm)))
        return self._en_vlm

    def _heavy_read(self, name, crop, n_best, park_others: bool = False):
        r = self._heavy_reader(name, park_others)
        if name == "hebrew":
            cands = r.recognize(crop, n_best=n_best, tta=self.hebrew_tta)
        elif name == "math2":
            cands = r.recognize(crop, n_best=n_best)
        else:
            from .prompts import PROMPT_ENGLISH
            cands = r.recognize(crop, n_best=n_best, prompt=PROMPT_ENGLISH)
        return [(c.text, c.logp) for c in cands]

    HEAVY_LABEL = {"hebrew": "Hebrew in context (Dicta-LM)", "math2": "formulas, second reader (Uni-MuMER)",
                   "english": "English (VLM)"}

    def finish(self, progress=None):
        """the deferred heavy reads (defer_heavy): one reader at a time over all its pieces, the others off
        the GPU. Until this runs, the pieces hold the ordinary readings. progress: a function given a line of
        text as it goes (the reader's model loading, every tenth piece with the time left)"""
        pending, self._pending = self._pending, []
        say = progress or (lambda msg: None)
        for name in self.HEAVY:
            jobs = [j for j in pending if j[0] == name]
            if not jobs:
                continue
            label = self.HEAVY_LABEL.get(name, name)
            say(f"{label}: {len(jobs)} pieces - loading its model")
            t0, t1 = time.time(), None
            try:
                for k, (_, seg, crop, n_best) in enumerate(jobs, 1):
                    self._heavy_apply(name, seg, self._heavy_read(name, crop, n_best, park_others=True))
                    if k == 1:
                        t1 = time.time()             # (the first piece carries the model's loading)
                        say(f"{label}: model ready in {t1 - t0:.0f} s, piece 1/{len(jobs)}")
                    elif k % 10 == 0 or k == len(jobs):
                        each = (time.time() - t1) / (k - 1)
                        say(f"{label}: {k}/{len(jobs)} pieces ({each:.1f} s each" +
                            (f", about {each * (len(jobs) - k) / 60:.1f} min left)" if k < len(jobs) else ")"))
            except Exception as e:               # noqa: BLE001  (a missing / broken model: keep the ordinary readings)
                print(f"[mixed reader] the {name} reader failed ({type(e).__name__}: {e}) - its pieces keep "
                      f"the ordinary readings", flush=True)
                say(f"{label}: FAILED ({type(e).__name__}: {e}) - its pieces keep the ordinary readings")
