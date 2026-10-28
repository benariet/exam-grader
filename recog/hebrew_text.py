"""Text utilities for the Hebrew line recognizer: character set, label cleaning and
logical <-> visual order.

The recognizer reads the image left-to-right (as CTC does), so its targets are in *visual*
order: Hebrew letters reversed, while digit / Latin runs keep their left-to-right order and
brackets are mirrored. `to_visual` is an involution (applying it twice returns the input),
so the same function converts the model output back to normal (logical) order.
"""
from __future__ import annotations

import re
import unicodedata

HEBREW = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
DIGITS = "0123456789"
PUNCT = " .,-?:\"()!;/'%"
CHARSET = PUNCT[0] + HEBREW + DIGITS + PUNCT[1:]          # index 0 is reserved for the CTC blank
CHAR2ID = {c: i + 1 for i, c in enumerate(CHARSET)}
ID2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

_MIRROR = str.maketrans("()[]{}<>", ")(][}{><")
_LTR_RUN = re.compile(r"[0-9A-Za-z]+(?:[.,:/%\-][0-9A-Za-z]+)*")
_SUBST = {"׳": "'", "״": '"', "’": "'", "‘": "'", "“": '"', "”": '"',
          "־": "-", "–": "-", "—": "-"}


def clean(text: str) -> str:
    """NFC, drop niqqud/cantillation, normalise quotes/dashes, keep only CHARSET, squeeze spaces."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(_SUBST.get(ch, ch) for ch in text if unicodedata.category(ch) != "Mn")
    text = "".join(ch for ch in text if ch in CHAR2ID)
    return " ".join(text.split())


def to_visual(text: str) -> str:
    """logical -> visual (and, being an involution, visual -> logical)."""
    t = text[::-1].translate(_MIRROR)
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], t)


to_logical = to_visual


def encode(visual: str) -> list[int]:
    return [CHAR2ID[c] for c in visual if c in CHAR2ID]


def decode(ids) -> str:
    return "".join(ID2CHAR.get(int(i), "") for i in ids)
