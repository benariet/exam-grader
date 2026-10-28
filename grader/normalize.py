"""Text normalization for OCR / VLM output of handwritten numbers and formulas.

The goal is to be forgiving about the *surface form* (decimal comma, unicode
symbols, spacing, stray LaTeX wrappers) without changing the *meaning*.
"""
from __future__ import annotations

import re

# Unicode look-alikes that OCR/VLMs emit inconsistently.
_CHAR_MAP = {
    "−": "-", "–": "-", "—": "-",          # minus / dashes
    "×": "*", "·": "*", "⋅": "*",         # multiplication
    "÷": "/",
    "Ω": "ohm", "Ω": "ohm", "ohms": "ohm",  # ohm sign (U+2126) and Greek omega (U+03A9)
    "μ": "u", "µ": "u",                   # Greek mu / micro sign -> 'u' (pint accepts u as micro)
    "°": " deg",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
}

_LATEX_UNIT_WRAPPERS = re.compile(r"\\(?:text|mathrm|textrm|mbox)\{([^}]*)\}")


# a box, an underline, a colour around an answer ('\boxed{V = 10 V}', '\underline{10 V}', '\textcolor{red}{...}'): the
# content alone. A label over a sign ('\overset{KVL}{=}', '\stackrel{def}{=}', '\xrightarrow{KVL}'): the bare sign. A
# definition's sign (':=', '\triangleq', '\coloneqq', '\equiv' - but not a congruence's, 'mod n'): an equals sign.
_DECOR = re.compile(r"\\(?:boxed|fbox|framebox|bbox|underline|uline|uuline|uwave|mathbf|boldsymbol|bm|textcolor\s*\{[^{}]*\}|"
                    r"colorbox\s*\{[^{}]*\}|fcolorbox\s*\{[^{}]*\}\s*\{[^{}]*\})\s*(?:\[[^\[\]]*\]\s*)*\{")
_STYLE = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|bf|it|boldmath|large|Large|LARGE|small|normalsize|"
                    r"color\s*\{[^{}]*\})(?![A-Za-z])")
_OVER_SIGN = re.compile(r"\\(?:overset|underset|stackrel)\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*"
                        r"\{\s*(=|\\[A-Za-z]*arrow|\\implies|\\approx)\s*\}")
_X_ARROW = re.compile(r"\\x(right|left|Right|Left|leftright|Leftright)arrow\s*(?:\[[^\[\]]*\]\s*)?\{(?:[^{}]|\{[^{}]*\})*\}")
_DEF_SIGN = re.compile(r":\s*=(?!=)|\\coloneqq(?![A-Za-z])|\\triangleq(?![A-Za-z])|\\doteq(?![A-Za-z])|"
                       r"\\equiv(?![A-Za-z])|\\eqdef(?![A-Za-z])")


_ARROWS = {"⟹": "Longrightarrow", "⇒": "Rightarrow", "⟶": "longrightarrow", "→": "rightarrow", "⇔": "Leftrightarrow",
           "⟺": "Leftrightarrow", "↔": "leftrightarrow", "∴": "therefore"}


def undecorate(s: str) -> str:
    """'\\boxed{V = 10 V}' -> ' V = 10 V ', 'I \\overset{KVL}{=} 2' -> 'I = 2', 'R := 5' -> 'R = 5' (see _DECOR); an x
    between a number and a power of ten is a times sign ('2.55 x 10^{-3}', as typed or read)"""
    t = _STYLE.sub(" ", s)
    # a statement in bold text ('\\textbf{I = 3 mA}': it was read as text, letters of a formula) - a word in bold stays text
    t = re.sub(r"\\textbf\s*\{([^{}]*=[^{}]*)\}", r" \1 ", t)
    for ch, cmd in _ARROWS.items():                # '2x = 6 ⇒ x = 3' as typed: an arrow between statements ('6⇒x = 3'
        t = t.replace(ch, f" \\{cmd} ")           # was one false step)
    t = t.replace("∞", " \\infty ")                # ('v(∞) = 10', '\lim_{t → ∞}' as typed: no verdict)
    t = re.sub(r"(?<=\d)\s*[xX×]\s*(?=1\s*0\s*\^)", r" \\times ", t)
    # 'P = 2 x 5 = 10': a lone x with spaces between two numbers is a times sign too (a letter x is written beside its
    # number: '3x = 12'); not after a unit's letters ('5 m x 2 m'? no number before)
    t = re.sub(r"(?<=\d)\s+x\s+(?=\d)", r" \\times ", t)
    t = re.sub(r"(\d\s*(?:(?:[kmunpµMG]|\\mu\s*)?(?:V|A|W|\\Omega|Ω|F|H|Hz|s)|[kMGnpµ]|\\mu)|\[\s*[^\[\]]{1,8}\]|"
               r"\\mathrm\s*\{[^{}]{1,8}\})\s+x\s+(?=\d)", lambda m: m.group(1) + " \\times ", t)
    #                                                ('4.7M x 100n', '100 [\mu A] x 12 M\Omega', '10 \mathrm{\mu A} x 5 \Omega' too)
    #                                                ('2 mA x 5 k\Omega': after a number's unit too)
    t = _OVER_SIGN.sub(lambda m: m.group(1), t)
    t = _X_ARROW.sub(lambda m: "\\" + m.group(1) + "arrow ", t)
    if not re.search(r"\\[pb]?mod(?![A-Za-z])|(?<![A-Za-z])mod(?![A-Za-z])", t):
        t = _DEF_SIGN.sub("=", t)
    while True:
        m = _DECOR.search(t)
        if not m:
            return t
        depth, i = 1, m.end()
        while i < len(t) and depth:
            depth += {"{": 1, "}": -1}.get(t[i], 0)
            i += 1
        if depth:                                  # never closed: only the command dropped
            t = t[:m.start()] + " " + t[m.end():]
        else:
            t = t[:m.start()] + " " + t[m.end():i - 1] + " " + t[i:]


def strip_math_delims(s: str) -> str:
    s = s.strip()
    for a, b in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            return s[len(a):-len(b)].strip()
    return s


def normalize_chars(s: str) -> str:
    for k, v in _CHAR_MAP.items():
        s = s.replace(k, v)
    return s


def normalize_decimal_comma(s: str) -> str:
    """'2,5' -> '2.5' but leave '1,000' style thousands separators alone
    (rare in Israeli exams; a comma between digit groups of exactly 3 is kept)."""
    return re.sub(r"(?<=\d),(?=\d(?!\d{2}(?:\D|$)))", ".", s)


def normalize_numeric_text(s: str) -> str:
    """Normalize a free-form numeric answer such as '4,7 kΩ' or '$3.3\\,\\text{mA}$'."""
    s = strip_math_delims(undecorate(s)).strip()
    s = _LATEX_UNIT_WRAPPERS.sub(r"\1", s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ").replace("~", " ")
    s = s.replace("\\Omega", "ohm").replace("\\mu", "u").replace("\\cdot", "*").replace("\\times", "*")
    s = normalize_chars(s)
    s = normalize_decimal_comma(s)
    # "10^{-3}" / "10^-3" -> "1e-3"  (common on exams)
    s = re.sub(r"10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?", r"1e\1", s)
    # "4.7*1e3" -> "4.7e3"
    s = re.sub(r"(\d)\s*\*\s*1e([+-]?\d+)", r"\1e\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_formula_text(s: str) -> str:
    """Light normalization for a formula; Math-Verify does the heavy lifting."""
    s = s.strip()
    s = normalize_chars(s)
    s = normalize_decimal_comma(s)
    return s
