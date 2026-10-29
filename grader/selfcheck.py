"""Math validation - the tool checks a student's mathematics itself, no answer key needed: it finds
each problem's formula and data in what the student wrote, checks every '=' of his computation, solves
the formula for the quantity he answered, and compares.

    'a^2+b^2=c^2 | a=3, b=4'  then  'c = 9'
        -> with a=3, b=4 the formula gives c = 5: WRONG
    'c^2 = 9+16 = 24'
        -> the step 9+16 = 24 is false (25): the slip is there
    '4^2+2^2 = \\sqrt{20}'
        -> false as written (20, not 4.47) - but 20 is c^2, so c = \\sqrt{20} is right: RIGHT VALUE, WRONG EQUALITY

    from grader.selfcheck import check_pages
    for c in check_pages(read_page.read_pages("scan.pdf")):  print(c.text)

How it reads the page. The math pieces are taken in reading order and cut into statements at '|',
',', ';', '=>' (an item number like '2)' in front is dropped; '\\approx' counts as '='):
  * a statement whose sides hold two or more letters is a FORMULA and opens a problem
    ('a^2+b^2=c^2', 'A = \\pi r^2', 'V = I R');
  * 'x = 5', 'I = 2 A' (a unit after the number is set aside), 'x = \\sqrt{20}' is a VALUE;
  * a chain 'c^2 = 9+16 = 25' is checked step by step, and a side with one unknown letter next to a
    number gives that letter's value (c^2 = 25 -> c = 5);
  * statements with numbers only are checked as written; a letter that is not in the problem's
    formula ('y^2+2^2=...' under 'a^2+b^2=c^2') is reported - most likely a misread digit.
The last value written for a letter of the formula is the student's answer; the values written before
it are the data. The formula is solved for that letter and compared: equal up to 1 % (79 for 78.54), or
a decimal number up to half a unit in its last decimal (78.5 for 25 pi = 78.54).

The readings come from a recognizer, so every problem is checked over the readings' n-best: the
verdict's probability is the probability of the readings that lead to it. A verdict below SURE, or
resting on a piece the layout was unsure about, is a suggestion for the human who checks - it changes
no grade by itself.
"""
from __future__ import annotations

import functools
import itertools
import cmath
import math
import re
import sys
from dataclasses import dataclass, field

import sympy as sp

SURE = 0.8                     # verdict probability (readings x layout) to act on without a human
REL_TOL = 0.01
ALT_CONF = 0.5                 # a value found in a piece the layout read as text: at most this sure
NBEST = 3                      # readings per piece taken into account
MAX_COMBOS = 64

_SEP = re.compile(r"\\(?:vert|mid|quad|qquad|Rightarrow|Longrightarrow|implies|therefore|Leftrightarrow|iff|"
                  r"rightarrow|longrightarrow|to)(?![A-Za-z])|\\(?:text|textrm|textbf|textit|emph|mbox)\s*\{(?P<w>[^{}]*)\}|"
                  r"(?<=\s)(?:or|and|so|thus|hence|therefore|then|where|gives|if|for)(?=\s)|[\u0590-\u05FF](?:[\u0590-\u05FF\s\"']|-(?!\s*\d))*|"
                  r"(?<![\\|])\|(?!\|)|(?<!\\);|"
                  r"(?<!\\),(?!\d{3}\b)|=>|->|"          # '\,' and '\;' are LaTeX spaces, not separators
                  # an inequality between statements ('f_s \ge 2 f_{max} = 2 \cdot 4k = 8 kHz': the chain after it - the
                  # whole was dropped); not inside brackets, where it is a condition ('P(X \le 2)')
                  r"\\(?:geq?|leq?|gg|ll|geqslant|leqslant|gtrsim|lesssim|neq?)(?![A-Za-z])|[≥≤≠≫≪]|(?<![<>\\])[<>](?![<>=])")
_INEQ = re.compile(r"\\(?:geq?|leq?|gg|ll|geqslant|leqslant|gtrsim|lesssim|neq?)(?![A-Za-z])|[≥≤≠≫≪<>]")
_ARROW = re.compile(r"\\(?:rightarrow|longrightarrow|to)(?![A-Za-z])")   # '3x = 12 \rightarrow x = 4' - but not in
#                                                     'V(t \to \infty)' or a limit's '_{x \to 0}': inside brackets
_ITEM = re.compile(r"^\s*(?:\(?\s*\d{1,2}\s*\)(?!\s*[_^])|\d{1,2}\s*\.\s+(?=\\?[A-Za-z]))\s*")    # ('2) ...', '1. I = ...' -
#                                                     not '(17)_8 = 15', a number in base 8, nor '1.5 = ...', a decimal)
_SUB_MARK = r"(?:[a-h]|i{1,3}|iv|vi{0,3}|[א-ח])"       # a sub-question's mark: a-h, i-viii, א-ח
_AFTER_MARK = r"(?=\s*(?:[A-Za-z]|\\(?!cdot|times|div|pm|mp|cdots|ldots)[A-Za-z]))"      # (a statement after it)
_SUB_ITEM = re.compile(r"^\s*(?:\(\s*(?:" + _SUB_MARK + r"|[A-H])\s*\)" + _AFTER_MARK + r"|" + _SUB_MARK + r"\s*[)']" +
                       _AFTER_MARK + r"|(?:[a-h]|[א-ח])\s*\.\s+" + _AFTER_MARK + r")\s*")
# ('b) P = V I = 25 mW', '(a) R = 1 k\Omega', 'c. ...', 'א) ...': the part of the exercise it answers - the line was
# lost ('b) P' no side), or read as a formula ('(a) R = 1 k\Omega': a times R) - its values none of the problem's)
_MARK_TEXT = re.compile(r"^\s*\\(?:text|textbf|mathrm|mathbf|textit)\s*\{\s*((?:(?:Question|Problem|Exercise|Part|Q|Ex)\s*\.?\s*)?"
                        r"\(?\s*(?:\d{1,2}\s*[a-h]?|" + _SUB_MARK + r"|[A-H])\s*[).':]?)\s*\}")    # ('\text{b)} ...': the mark
#                                                  as text)
_Q_LABEL = re.compile(r"^\s*(?:(?:Question|Problem|Exercise|Task)\s*\.?\s*(?P<n1>\d{1,2})(?P<l1>\s*[a-h])?\s*[).:]?|"
                      r"(?:Q|Ex|Prob)\s*\.?\s*(?P<n2>\d{1,2})(?P<l2>\s*[a-h])?\s*[).:]|"
                      r"\(?\s*(?P<n3>\d{1,2})\s*\)?\s*:|"
                      r"\(?\s*(?P<n4>\d{1,2})\s*\.?\s*(?:\(\s*(?P<l4>[a-h])\s*\)|(?P<l5>[a-h])\s*[).:])|"
                      r"\(?\s*(?P<n6>\d{1,2})\s*\.\s*(?P<s6>\d{1,2})\s*\))\s*" + _AFTER_MARK)
# ('Q1: ...', 'Question 2 ...', '1a) ...', '1(b) ...', '3.2) ...', '1: ...' after 'שאלה': a question's number - read as a
# formula ('Q1: R = 5 k\Omega': Q1 times R), a computation ('1a) R': 1 times a) or lost)
_PART_LABEL = re.compile(r"^\s*(?:(?:Part|Section)\s*\(?\s*(?:[a-hA-H]|\d{1,2}|i{1,3}|iv|vi{0,3})\s*\)?\s*[.:]?|"
                         r"\(?\s*" + _SUB_MARK + r"\s*\)?\s*:(?=\s))\s*" + _AFTER_MARK)       # ('Part a: ...', '(b): ...')


def _label(p: str) -> tuple:
    """a question's or a sub-question's mark at the start of a part -> (the part without it, a new exercise?, marked?): a
    question's number is one ('Q2: ...', '2a) ...'), its later parts not ('2b) ...', 'b) ...', 'Part c: ...')"""
    m = _Q_LABEL.match(p)
    if m:
        letter = next((m.group(g).strip() for g in ("l1", "l2", "l4", "l5") if m.group(g)), None)
        later = (letter is not None and letter != "a") or (m.group("s6") is not None and m.group("s6") != "1")
        return p[m.end():], not later, True
    for rx in (_PART_LABEL, _SUB_ITEM):
        m = rx.match(p)
        if m:
            return p[m.end():], False, True
    return p, False, False
_UNIT = re.compile(r"^(?P<num>.*?(?:\d[\d\s.]*|10\s*\^\s*\{\s*[+\-−]?\s*\d+\s*\}\s*))\s*(?P<rm>\\(?:mathrm|text|rm)\s*\{\s*)?(?P<unit>(?:[kmMunpµf]|\\mu\s*)?"
                   r"(?:V\s*A\s*R|V\s*A|var|W\s*h|A\s*h(?![A-Za-z])|Wb|eV|V|A|W|\\Omega|Ω|ohm|F|H|Hz|s|S|J|N|m|cm|mm|km|kg|g|C|K|Pa|T|deg|rad|°|d\s*B(?:\s*[mW]|i|c)?|"
                   r"bits?|bps|baud)|"
                   r"(?:[kmunpµ]|\\mu\s*)S|"
                   r"(?:[kMunpµ]|\\mu)(?![A-Za-z]))\s*(?(rm)\})\s*$")
# (the number may end in a power of ten, '2.553 \cdot 10^{-2} A' - its unit was left a letter of the problem)
# (a '}' after the unit only closes its own \mathrm{: '\frac{10}{2 k\Omega}' is no number with a unit - it was read
# as (\frac{10}{2) k\Omega, and '= 5 mA' after it judged a false wrong step)
# (a bare prefix, as EE students write values: 'R_1 = 1k', 'C = 10\mu' - kept as written, like '1 k\Omega')
_C_UNIT = re.compile(r"^(?P<num>.*?(?:\d|\\circ\s*\}?|°|\)|(?<=\d)j))\s*(?:\\left\s*)?(?:\[\s*)?(?P<rm>\\(?:mathrm|text|rm)\s*\{\s*)?"
                     r"(?P<unit>(?:[kmMunpµ]|\\mu\s*)?(?:V\s*A\s*R|V\s*A|var|V|A|W|\\Omega|Ω|ohm|S))\s*(?(rm)\})\s*(?:\\right\s*)?(?:\]\s*)?$")
_U_ATOM = (r"(?:\\mu\s*|[pnuµmkMGc])?\s*(?:Hz|Pa|rad|sec|min|ohm|\\Omega|Ω|Vs|V|A|W|F|H|s|J|N|m|g|C|K|S|h|L)"
           r"(?:\s*\^\s*(?:\{\s*-?\s*\d\s*\}|-?\d))?")
_U_PROD = rf"{_U_ATOM}(?:(?:\s*\\(?:cdot|times)\s*|\s+){_U_ATOM})*"
_COMPOUND = re.compile(rf"^(?P<num>.*?(?:\d[\d\s.]*|10\s*\^\s*\{{\s*[+\-−]?\s*\d+\s*\}}\s*))\s*(?P<unit>\\frac\s*\{{\s*(?:{_U_PROD}|1)\s*\}}\s*\{{\s*{_U_PROD}\s*\}}|"
                       rf"{_U_PROD}\s*/\s*{_U_PROD}|(?:[cmkn]?m|µm|\\mu\s*m)\s*\^\s*(?:\{{\s*[-−]?\s*[1-3]\s*\}}|[-−]?[1-3])|"
                       rf"(?:[mnµ]?s|\\mu\s*s)\s*\^\s*(?:\{{\s*[-−]\s*[12]\s*\}}|[-−][12])|"
                       rf"(?:\\Omega|Ω)\s*(?:\\cdot\s*)?(?:c|m)?m(?![A-Za-z]))\s*$")
# ('v = 10 \frac{m}{s}', 'a = 9.8 m/s^2', '\rho = 2000 \frac{kg}{m^3}', 'A = 0.5 m^2' - a product alone ('0.5 C V^2')
# is a formula, not a unit; a density, a resistivity as semiconductor physics writes them: 'n = 2.25 \cdot 10^4 cm^{-3}',
# '\frac{1}{cm^3}', '\rho = 0.463 \Omega \cdot cm' - they were letters of the problem, no verdict)


_TIME = re.compile(r"^(?P<num>[\d\s.]*\d)\s*(?P<rm>\\(?:mathrm|text)\s*\{\s*)?(?P<unit>min|h|hr)\s*(?(rm)\})\s*$")
_TIMED = re.compile(r"\d\s*(?:\\(?:mathrm|text)\s*\{\s*)?(?:s|ms|sec|min|h|hr)\s*\}?\s*$")


_SCI_E = re.compile(r"(?<=\d)\s*[eE]\s*[+-]?\s*\d")


def _has_letters(num: str) -> bool:
    """a number's text has a letter in it: a unit's number ('\\frac{1}{2}', '5 \\cdot 10^{3}', a calculator's '2.553e-2')
    has none"""
    return bool(re.search(r"[A-Za-z]", _SCI_E.sub("0", re.sub(r"\\(?:frac|sqrt|pi|cdot|times)", "", num))))


def _letter_used(part: str, unit: str) -> bool:
    """a one-letter unit that is a letter of the problem on the same line ('F = \\mu N = 0.3 N = 6 N': N the normal force,
    not newtons)"""
    u = re.sub(r"[\\\s{}]", "", unit)
    if len(u) != 1:
        return False
    rest = re.sub(r"(?:\d|\})\s*(?:\\mu\s*|[pnuµmkMG])?\s*" + re.escape(u) + r"(?![A-Za-z])", "", part)   # ('1 \mu F',
    #                                                  '10^{-6} F': a unit after a number, its prefix between)
    return bool(re.search(r"(?<![\\A-Za-z])" + re.escape(u) + r"(?![A-Za-z_])", rest))


def _whole_si(t: str, time_ok: bool = False):
    """a side that is a number with its unit and nothing else ('72 \\frac{km}{h}', '20 m/s', '5 mA') -> (that number
    in SI units as text, the unit is a compound one); None: not such a side (a lone 'm' - metres or milli? - neither)"""
    t = _join_digits(t)
    m, compound = _UNIT.match(t), False
    if not m:
        m, compound = _COMPOUND.match(_unwrap_rm(t)), True
    if not m and time_ok:                            # 't = 2 min = 120 s': minutes, hours beside a time in seconds
        m, compound = _TIME.match(t), True
    if not m or _has_letters(m.group("num")) or m.group("unit").strip() == "m" \
            or m.group("num").count("{") != m.group("num").count("}"):
        return None
    f = _si(m.group("unit"))
    if f is None or (_bare_prefix(m.group("unit")) and not _plain_num(m.group("num"))) or \
            re.sub(r"[\s{}]", "", m.group("unit")) == "T":
        return None                                  # ('12 / 4.7k': a bare prefix is the last number's, not the side's;
    #                                                  a bare T may be a period's letter)
    num = m.group("num").strip()
    return (num if f == 1 else f"({num}) \\cdot {_factor_tex(f)}"), compound


def _den_si(t: str, time_ok: bool = False):
    """'12 / 4 k\\Omega', '3 \\cdot 10^8 / 100 MHz': the unit after a divisor's number is that number's (I = V/R = 12 /
    4 kΩ = 3 mA) - the side in SI units so ('12 / (4 \\cdot 10^{3})'); None: no such side (one number, a product - '2
    \\cdot 4 kHz' reads the same either way -, a fraction - '\\frac{12}{4} mA' is 3 mA -, a sum)"""
    if _whole_si(t, time_ok) is None:
        return None
    t = _join_digits(t)
    m = _UNIT.match(t)
    if not m:
        return None
    num, f = m.group("num"), _si(m.group("unit"))
    if f in (None, 1.0):
        return None
    depth, cut = 0, None
    for i, ch in enumerate(num):
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif depth == 0 and ch in "+-" and num[:i].strip() and not re.search(r"(?:\^|\\cdot|\\times|[*/(])\s*$", num[:i]) \
                and not re.search(r"[eE]\s*$", num[:i]):
            return None                              # (a sum: '2 + 6 / 3 k\Omega' - its unit whose?)
        elif depth == 0 and ch == "/":
            cut = i
    if cut is None or not num[:cut].strip() or not _plain_num(num[cut + 1:]):
        return None
    return f"{num[:cut].strip()} / ({num[cut + 1:].strip()} \\cdot {_factor_tex(f)})"


# the unit a quantity's usual name is measured in ('I_1' A, 'V_{th}' V, '\\tau' s, 'f_0' Hz) - names of one meaning only (not
# E: an EMF, a field or an energy; P: a power or a pressure; L: an inductance or a length; C, Q, T, B)
_NAME_UNIT = {"V": "V", "v": "V", "I": "A", "i": "A", "R": "Ω", "r": "Ω", "Z": "Ω", "X": "Ω", "G": "S", "f": "Hz", "t": "s",
              "tau": "s", "lambda": "m", "omega": "Hz"}


def _name_unit(name: str):
    """the unit the quantity a name usually stands for is measured in ('I_1' -> 'A'), or None"""
    m = re.match(r"\s*\\?([A-Za-z]+?)(?=_|\{|\s*$|\^|')", name or "")
    return _NAME_UNIT.get(m.group(1)) if m else None


def _name_unit_differs(name: str, unit: str) -> bool:
    """is a unit of another kind than the quantity a name usually stands for ('I' in kΩ, 'f' in ms)? False when either
    is unknown"""
    want = _name_unit(name)
    got = _dimension(unit)
    if want is None or got is None:
        return False
    return sp.simplify(_dimension(want) / got) != 1


def _pick_den(texts, alts, name="", units=None):
    """the sides of a chain in SI units (texts), with the divisor's reading of each side that has one (alts: _den_si or
    None): that reading where the side next to it holds so, or neither holds the whole side's; when neither holds, the
    quantity's name decides ('I' in kΩ: the divisor's; 'V' in mV: the whole side's - the message then reads '5 mV is not
    6 mV'), and by default the divisor's (as '= 12 / 4 k\\Omega = 3 mA' is written; '= 10 / 2 mV = 5 mV' the whole side's).
    units: each side's unit as written (for the name)"""
    out = list(texts)
    for k, d in enumerate(alts):
        if d is None:
            continue
        vd, vw = (_num(e) if e is not None else None for e in (_parse(d), _parse(out[k])))
        near = [_num(e) if e is not None else None for e in (_parse(out[j]) for j in (k + 1, k - 1) if 0 <= j < len(out))]
        near = [v for v in near if v is not None]
        if vd is None:
            continue
        if near and vw is not None and _close(near[0], vw, "") and not _close(near[0], vd, ""):
            continue                                 # the whole side's reading holds
        if near and not _close(near[0], vd, "") and units and units[k] and _name_unit(name) and \
                not _name_unit_differs(name, units[k]):
            continue                                 # neither holds; a V in mV: the whole side's
        out[k] = d
    return out


def _bare_prefix(unit: str) -> bool:
    return re.sub(r"[\\\s{}]", "", unit or "") in ("k", "M", "G", "u", "µ", "μ", "mu", "n", "p")


def _plain_num(num: str) -> bool:
    """a number as written, nothing else ('4.7', '2.55 \\cdot 10^{-3}' - not '12 / 4.7')"""
    return bool(_NUM_TEXT.fullmatch(re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", (num or "").strip())))


def _factor_tex(f: float) -> str:
    """a unit's factor to SI as it reads best: '10^{-3}', '\\frac{5}{18}' (km/h), else its digits"""
    k = round(math.log10(f))
    if math.isclose(f, 10.0 ** k, rel_tol=1e-9):
        return f"10^{{{k}}}"
    short = f"{f:.6g}"
    if "e" not in short and math.isclose(float(short), f, rel_tol=1e-9):   # ('3.6' for mAh in coulombs, not '18/5')
        return short
    q = sp.nsimplify(f, rational=True, tolerance=1e-9 * f)
    if q.q <= 3600 and math.isclose(float(q), f, rel_tol=1e-9):
        return str(q.p) if q.q == 1 else f"\\frac{{{q.p}}}{{{q.q}}}"
    mant, ex = f"{f:.9e}".split("e")                 # ('1.602176634 \cdot 10^{-19}': an electron-volt - not '1.602176634e-19',
    return f"({float(mant):.10g} \\cdot 10^{{{int(ex)}}})"   # read as 1.602 e - 19)


def _unwrap_rm(t: str) -> str:
    """'\\mathrm{\\frac{m}{s}}' -> '\\frac{m}{s}': the upright-text wrappers of units taken off"""
    while True:
        m = re.search(r"\\(?:mathrm|text|rm)\s*\{", t)
        if not m:
            return t
        depth, i = 1, m.end()
        while i < len(t) and depth:
            depth += {"{": 1, "}": -1}.get(t[i], 0)
            i += 1
        t = t[:m.start()] + t[m.end():i - 1] + t[i:]


def _words(content: str, before: str) -> bool:
    """'\text{ ולכן }', '\text{ so }', '\text{or}': words between two statements (not a unit, '\text{mA}', nor part of a
    name, 'V_\text{out}')"""
    c = content.strip()
    if before.rstrip().endswith(("_", "^")):
        return False
    if re.search(r"[\u0590-\u05FF]", c):                  # (Hebrew, a single letter too: '\text{ ו- }', 'and')
        return True
    if len(re.findall(r"[A-Za-z]", c)) < 2:
        return False
    return _si(c) is None and not re.fullmatch(r"(?:\\mu\s*|[pnuµmkMG])?(?:Hz|Pa|rad|sec|min|ohm|\\Omega|Ω|V|A|W|F|H|s|J|N|m|g|C|K|S|h|L|dB)", c)


def _bare_braces(tex: str) -> str:
    """braces no command takes, as formula readers write them ('I = {\\frac{12}{4k}} = 3 mA', 'I = {12}'): brackets - the side
    was not read at all (the line lost); '{a \\over b}' a fraction, '{\\rm k}\\Omega' upright (a unit). Not a command's
    argument ('\\frac{a}{b}', '\\sqrt{x}', 'x^{2}', 'R_{eq}'), nor '\\{'"""
    if "{" not in tex or "\\begin" in tex:          # (an array's cells: '\\begin{array}{cc} {1} & {2} ...' - its own)
        return tex
    t = re.sub(r"\{\s*\\(?:rm|mathrm|textrm|text)\s+([^{}]*?)\s*\}", r"\\mathrm{\1}", tex)      # ('{\\rm k}')
    for _ in range(4):                               # ('{12 \\over 4}': 12/4)
        t2 = re.sub(r"\{([^{}]*?)\\over(?![A-Za-z])([^{}]*)\}", r"\\frac{\1}{\2}", t)
        if t2 == t:
            break
        t = t2
    out, i = [], 0
    while True:
        j = t.find("{", i)
        if j < 0:
            out.append(t[i:])
            return "".join(out)
        k = j - 1
        while k >= 0 and t[k] == " ":
            k -= 1
        prev = t[k] if k >= 0 else ""
        if prev and (prev.isalpha() or prev in "^_}]\\{"):
            out.append(t[i:j + 1])
            i = j + 1
            continue
        depth, e = 0, -1
        for q in range(j, len(t)):
            depth += {"{": 1, "}": -1}.get(t[q], 0)
            if depth == 0:
                e = q
                break
        if e < 0:
            out.append(t[i:])
            return "".join(out)
        if re.search(r"=|&|\\\\", t[j:e]):                 # (a row the reader wraps in braces, '{ 2 x + y = 5 }': _unbrace's)
            out.append(t[i:j + 1])
            i = j + 1
            continue
        out.append(t[i:j] + "(")
        t = t[:e] + ")" + t[e + 1:]
        i = j + 1


def _unbrace(t: str) -> str:
    """'{ 2 x + y = 5 }' -> '2 x + y = 5': a row (or a cell) the formula reader wraps in braces"""
    t = t.strip()
    while t.startswith("{") and t.endswith("}"):
        depth = 0
        for i, ch in enumerate(t):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth == 0 and i < len(t) - 1:
                return t                           # '{a} + {b}': not one pair around it all
        t = t[1:-1].strip()
    return t


def _top_level_split(tex: str):
    """cut at separators outside braces"""
    tex = re.sub(r"\(\s*[\u0590-\u05FF][^()]*\)", " ; ", tex)         # '(מתח)': a word in brackets, no math
    tex = re.sub(r"(?:^|(?<=[\s;,:]))[\u05d0-\u05d7]\s*[).':](?=\s*(?:[A-Za-z]|\\[A-Za-z]))", " ; ", tex)   # 'א) I = ...',
    #                                                  'ב. ...': a sub-question's mark (its ')' was left as the statement's)
    out, depth, paren, start, i = [], 0, 0, 0, 0
    while i < len(tex):
        ch = tex[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch in "([":
            paren += 1
        elif ch in ")]":
            paren -= 1
        elif depth == 0:
            m = _SEP.match(tex, i)
            if m and m.group("w") is not None and not _words(m.group("w"), tex[:i]):
                m = None                             # '\text{mA}', 'V_\text{out}': a unit, a name - no words between
            if m and not (paren > 0 and (_ARROW.match(m.group()) or _INEQ.match(m.group()))):
                out.append(tex[start:i])
                start = i = m.end()
                continue
        i += 1
    out.append(tex[start:])
    return out


def _split_eq(part: str) -> list:
    """a statement's sides: cut at each '=' outside braces ('\\sum_{k=1}^{4} k = 10' has two sides) and brackets ('P(X = 2)
    = \\binom{5}{2} 0.3^2 0.7^3': its first side - it was cut into 'P(X' and '2)', and '2) = ...' a false wrong step);
    brackets that do not close: at every '=' outside braces, as before"""
    for brackets in (True, False):
        out, depth, paren, start = [], 0, 0, 0
        for i, ch in enumerate(part):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch in "([" and brackets:
                paren += 1
            elif ch in ")]" and brackets:
                paren -= 1
            elif ch == "=" and depth == 0 and paren == 0:
                out.append(part[start:i])
                start = i + 1
        if paren == 0 or not brackets:
            return out + [part[start:]]
    return [part]


def _join_digits(s: str) -> str:
    """the formula reader spaces its tokens: '7 8 . 5 4' -> '78.54', '\\sqrt { 2 0 }' -> '\\sqrt {20}'"""
    return re.sub(r"(?<=[\d.])(?<![\^_]\d)\s+(?=[\d.])", "", s)   # ('\sin^2 30^\circ', '\int_0^1 4t^2': a power, a
    #                                                                     limit stays apart


_PRIME = re.compile(r"(?<![\\A-Za-z])([A-Za-z])(?:_\{([^{}]*)\}|_([A-Za-z0-9]))?\s*('+|\^\s*\{\s*(?:\\prime\s*)+\}|\^\s*\\prime(?![A-Za-z]))")


def _primes(side: str) -> str:
    """y', f''(x), V_A': a derivative's (or a superposition share's) name kept apart - the parser drops the prime,
    so y' was read as y, and V_A = V_A' + V_A'' as V_A = 2 V_A -> y_{\\prime}, f_{\\prime\\prime}(x), V_{A\\prime}"""
    def rep(m):
        n = m.group(4).count("'") + m.group(4).count("prime")
        sub = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        return m.group(1) + "_{" + sub + "\\prime" * n + "}"
    return _PRIME.sub(rep, side)


_D_OPAQUE = {}                 # name -> (expression, variable, order): a derivative kept unworked (dy/dx)


def _opaque_derivatives(e, partial: bool = False):
    """d/dx of a lone letter or of an expression with other letters (dy/dx, d(xy)/dx): the parser takes those letters
    for constants (dy/dx = 0, a false 'wrong step' under '\\frac{dy}{dx} = 3x^2 = 12') - kept as a name of their own"""
    subs = {}
    for d in e.atoms(sp.Derivative):
        variables = {v for v, _ in d.variable_count}
        if any(isinstance(f, sp.core.function.AppliedUndef) for f in d.expr.atoms(sp.Function)):
            continue                               # d/dx f(x): worked out once f is known (_functions)
        if d.expr.free_symbols - variables and not (partial and not d.expr.is_Symbol):   # a partial derivative: the
            v = d.variable_count[0][0]                                                   # other letters are constants
            n = sum(c for _, c in d.variable_count)
            o = "" if n == 1 else f"^{{{n}}}"
            name = (f"\\frac{{d{o}{d.expr}}}{{d{v}{o}}}" if d.expr.is_Symbol else f"\\frac{{d{o}}}{{d{v}{o}}}({d.expr})")
            _D_OPAQUE[name] = (d.expr, v, n if len(d.variable_count) == 1 else 0)
            subs[d] = sp.Symbol(name)
    return e.xreplace(subs) if subs else e


_BR_OPEN = re.compile(r"\\left\s*\[|\\[Bb]igg?l?\s*\[|\\left\s*\.|(?<!\\)\[")
_BR_LIMIT = r"(\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|[^\s{}\\^_])"
_BR_LIMITS = re.compile(rf"\s*(?:_\s*{_BR_LIMIT}\s*\^\s*{_BR_LIMIT}|\^\s*{_BR_LIMIT}\s*_\s*{_BR_LIMIT})")


def _eval_brackets(t: str) -> str:
    """'\\left[ \\frac{x^3}{3} \\right]_0^2' (or '\\left. ... \\right|_0^2', '[...]_0^2') -> F(2) - F(0), as the value of an
    integral is written out; left as it is unless the bracket holds one letter and its limits are numbers"""
    pos = 0
    for _ in range(4):
        m = _BR_OPEN.search(t, pos)
        if not m:
            return t
        bar = m.group().endswith(".")
        depth, i, close = 0, m.end(), None
        while i < len(t):
            if t.startswith("\\left", i):
                depth += 1
                i += 5
                continue
            if bar and t.startswith("\\right", i) and depth == 0:
                k = re.match(r"\\right\s*\|", t[i:])
                if k:
                    close = (i, i + k.end())
                break
            if t.startswith("\\right", i):
                depth -= 1
            if not bar and depth <= 0 and t[i] == "]":
                if depth == 0 or t[max(0, i - 7):i].rstrip().endswith("\\right"):
                    start = i
                    k = re.search(r"\\(?:right|[Bb]igg?r?)\s*$", t[:i])
                    close = ((k.start() if k else start), i + 1)
                    break
            if not bar and t[i] == "[" and not t.startswith("\\left", max(0, i - 6)):
                depth += 1
            if not bar and t[i] == "]" and depth > 0:
                depth -= 1
            i += 1
        if close is None:
            pos = m.end()
            continue
        lim = _BR_LIMITS.match(t, close[1])
        if not lim:
            pos = m.end()
            continue
        lo, hi = (lim.group(1), lim.group(2)) if lim.group(1) is not None else (lim.group(4), lim.group(3))
        inner = _parse(t[m.end():close[0]])
        a, b = (_parse(x[1:-1] if x.startswith("{") else x) for x in (lo, hi))
        if inner is None or a is None or b is None or len(inner.free_symbols) != 1 or a.free_symbols or b.free_symbols:
            pos = m.end()
            continue
        (v,) = inner.free_symbols
        try:
            val = complex(sp.N(inner.subs(v, b) - inner.subs(v, a)))
        except (TypeError, ValueError):
            pos = m.end()
            continue
        if abs(val.imag) > 1e-12 or not math.isfinite(val.real):
            pos = m.end()
            continue
        t = t[:m.start()] + f"\\left({val.real:.12g}\\right)" + t[lim.end():]
        pos = m.start() + 1
    return t


_LAPLACE = re.compile(r"^\s*(?:\\(?:mathcal|mathscr|mathfrak|mathbb|mathrm)\s*\{\s*L\s*\}|\\mathcal\s*L|(?P<plain>L))\s*"
                      r"(?P<inv>\^\s*\{?\s*-\s*1\s*\}?)?\s*(?:\\left\s*)?(?P<open>\\\{|\\lbrace|\{|\[|\()(?P<inner>.*?)"
                      r"(?:\\right\s*)?(?:\\\}|\\rbrace|\}|\]|\))\s*$")


def _signals(e):
    """u(t) the unit step, \\delta(t) the impulse - as a Laplace transform's table writes them"""
    from sympy.core.function import AppliedUndef
    rep = {}
    for f in e.atoms(AppliedUndef):
        name = str(f.func).strip("\\{} ")
        if len(f.args) == 1 and name in ("u", "H", "Heaviside"):
            rep[f] = sp.Heaviside(f.args[0])
        elif len(f.args) == 1 and name in ("delta", "δ"):
            rep[f] = sp.DiracDelta(f.args[0])
    return e.xreplace(rep) if rep else e


@functools.lru_cache(maxsize=1024)
def _laplace_side(text: str):
    """'\\mathcal{L}\\{e^{-2t}\\}' -> 1/(s + 2); '\\mathcal{L}^{-1}\\left\\{\\frac{1}{s(s+1)}\\right\\}' -> 1 - e^{-t} (times the step);
    None: not a transform, or one sympy cannot do"""
    m = _LAPLACE.match(text)
    if not m or (m.group("plain") and m.group("open") not in ("\\{", "\\lbrace")):
        return None
    inner = _parse(m.group("inner"))
    if inner is None:
        return None
    inner = _signals(_calls(inner, {"u", "delta"}))     # ('u(t-2)': the parser's product of u and t - 2 - a step)
    s_sym, t_sym = _parse("s"), _parse("t")
    try:
        if m.group("inv"):
            if s_sym not in inner.free_symbols:
                return None
            ps = sp.Symbol("s")
            out = sp.inverse_laplace_transform(inner.xreplace({s_sym: ps}), ps, t_sym)
        else:
            var = t_sym if t_sym in inner.free_symbols else (next(iter(inner.free_symbols)) if len(inner.free_symbols) == 1 else None)
            if var is None:
                return None
            out = sp.laplace_transform(inner, var, s_sym, noconds=True)
    except Exception:                              # noqa: BLE001
        return None
    if out is None or out.has(sp.LaplaceTransform, sp.InverseLaplaceTransform, sp.Integral):
        return None
    return out


def _parse(side: str):
    return _parse_full(side)[0]


# a complex number's real / imaginary part, its angle or conjugate ('\\Re\\{Z\\}', 'Re(Z)', '\\angle Z' - not an angle in
# terms of pi, '\\angle \\frac{\\pi}{4}' -, '\\arg Z', 'I^*',
# '\\overline{I}'): the parser takes them for products of letters (R e Z) - no expression at all, rather than a false one
_COMPLEX_PARTS = re.compile(r"\\(?:Re|Im|arg)(?![A-Za-z])|\\(?:mathrm|text|textrm|operatorname|rm)\s*\{\s*(?:Re|Im|arg)\s*\}|"
                            r"(?<![A-Za-z\\])(?:Re|Im)\s*(?:\(|\\\{|\\left|\[)|\\angle\s*(?!\\pi(?![A-Za-z])|\\frac\s*\{[^{}]*\\pi)\\?[A-Za-z]|\^\s*\{?\s*(?:\*|\\ast)|"
                            r"\\(?:overline|bar)\s*\{?\s*\\?[A-Za-z]")


# a side whose value is not what the parser makes of it: an ellipsis ('1 + 2 + ... + 10' read as 13), a modulo ('5 \\bmod 3'
# read as 3) - no expression rather than a false one
_NO_VALUE = re.compile(r"\.\.\.|…|\\[cl]?dots(?![A-Za-z])|\\[bp]?mod(?![A-Za-z])|(?<![A-Za-z\\])mod(?![A-Za-z])|"
                       # a probability, an expectation, a variance: 'P(X \le 2)' (read as P of a relation), 'E[X^2]'
                       # (as e times X^2), 'Var(X)' - not values the parser can work out
                       r"\([^()]*(?:<|>|=|\\le(?:q)?(?![A-Za-z])|\\ge(?:q)?(?![A-Za-z])|≤|≥|\\lt(?![A-Za-z])|\\gt(?![A-Za-z]))[^()]*\)|"
                       r"(?<![A-Za-z\\])(?:E|\\mathbb\s*\{\s*E\s*\}|\\mathrm\s*\{\s*E\s*\})\s*\[|(?<![A-Za-z\\])(?:Var|Cov|var|cov)\s*\(")


_ARG_TIME = re.compile(r"(?P<f>(?<![A-Za-z\\])[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?)\s*(?P<o>\(|\\left\s*\()\s*(?P<n>\d+(?:\.\d+)?)\s*"
                       r"(?:\\,|\\ |~)?\s*(?:\\(?:mathrm|text|rm)\s*\{\s*)?(?P<p>m|\\mu\s*|µ|u|n|p)?\s*(?P<u>sec|s)\s*\}?\s*"
                       r"(?P<c>\)|\\right\s*\))")
_PREFIX_S = {"m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12}


def _arg_seconds(m):
    """'v_C(1 ms)' -> 'v_C(0.001)'; a bare 's' only after a time function's lowercase name (H(2s) is H at 2s)"""
    pre = re.sub(r"[\\\s]|mu", lambda x: "µ" if x.group(0) == "mu" else "", m.group("p") or "")
    if not pre and not m.group("f")[0].islower():
        return m.group(0)
    v = float(m.group("n")) * _PREFIX_S.get(pre, 1.0)
    return f"{m.group('f')}({format(v, '.15f').rstrip('0').rstrip('.') or '0'})"      # (no '2e-06': read as 2e - 6)


_BASE_NUM = re.compile(r"(?<![A-Za-z0-9_.\\])(?:(?:\\left\s*)?\(\s*(?P<p>[0-9A-Fa-f]+(?:\s*\.\s*[0-9A-Fa-f]+)?)\s*(?:\\right\s*)?\)|"
                       r"(?P<d>\d+(?:\.\d+)?))\s*_\s*(?:\{\s*(?:\\left\s*)?\(?\s*(?P<b1>\d\s*\d?)\s*(?:\\right\s*)?\)?\s*\}|"
                       r"(?:\\left\s*)?\(\s*(?P<b2>\d\s*\d?)\s*(?:\\right\s*)?\)|(?P<b3>\d))")


_LETTER_BASE = {"H": 16, "h": 16, "hex": 16, "B": 2, "b": 2, "bin": 2, "O": 8, "o": 8, "oct": 8, "D": 10, "d": 10, "dec": 10}
_BASE_MORE = re.compile(
    r"(?<![A-Za-z0-9_.\\])(?:0[xX](?P<x>[0-9A-Fa-f]+)|0[bB](?P<bn>[01]+)|"
    r"(?:\\left\s*)?\(\s*(?P<pl>[0-9A-Fa-f]+(?:\s*\.\s*[0-9A-Fa-f]+)?)\s*(?:\\right\s*)?\)\s*_\s*\{?\s*(?P<plb>hex|bin|oct|dec|[HBOD])\s*\}?|"
    r"(?P<hd>[0-9A-Fa-f]*[A-Fa-f][0-9A-Fa-f]*)\s*_\s*(?:\{\s*16\s*\}|\{\s*\(\s*16\s*\)\s*\})|"
    r"(?P<ld>[0-9A-Fa-f]*\d[0-9A-Fa-f]*)\s*_\s*\{?\s*(?P<lb>hex|bin|oct|dec|[HBOD])\s*\}?)(?![0-9A-Za-z])")


def _more_bases(side: str) -> str:
    """'0x1F', '0b1011', '1F_{16}' and 'FF_{16}' (hex digits unbracketed), '(1F)_H', '1011_B', '17_O': a number in its base, as
    '(1F)_{16}' (they were letters - no verdict - or a product)"""
    def rep(m):
        if m.group("x"):
            return f"({m.group('x')})_{{16}}"
        if m.group("bn"):
            return f"({m.group('bn')})_{{2}}"
        if m.group("pl"):
            return f"({m.group('pl')})_{{{_LETTER_BASE[m.group('plb')]}}}"
        if m.group("hd"):
            return f"({m.group('hd')})_{{16}}" if len(m.group("hd")) >= 2 else m.group(0)    # ('F_{16}': a name)
        if len(m.group("ld")) >= 2:                  # ('C_B', 'V_D': names - a digit and two characters at least)
            return f"({m.group('ld')})_{{{_LETTER_BASE[m.group('lb')]}}}"
        return m.group(0)
    return _BASE_MORE.sub(rep, side)


def _integer_side(side: str):
    """'17 \\bmod 5', '\\gcd(12, 18)', '\\operatorname{lcm}(4, 6)', '1 + 2 + \\ldots + 10' (an arithmetic series) -> its value; None
    for any other side (the parser read 'mod' as letters, '\\ldots' as nothing)"""
    m = re.fullmatch(r"\s*(\d+)\s*(?:\\bmod|\\mod|mod|\\%)\s*(\d+)\s*", side)
    if m and int(m.group(2)) > 0:
        return sp.Integer(int(m.group(1)) % int(m.group(2)))
    m = re.fullmatch(r"\s*\\?(?:operatorname\s*\{\s*)?(max|min)\s*\}?\s*(?:\\left\s*)?\(\s*(-?\d+(?:\.\d+)?)\s*[,\ue002]\s*(-?\d+(?:\.\d+)?)\s*"
                     r"(?:\\right\s*)?\)\s*", side)
    if m:
        a, b = float(m.group(2)), float(m.group(3))
        return sp.nsimplify(max(a, b) if m.group(1) == "max" else min(a, b))
    m = re.fullmatch(r"\s*\\?(?:operatorname\s*\{\s*)?(gcd|lcm)\s*\}?\s*(?:\\left\s*)?\(\s*(\d+)\s*[,\ue002]\s*(\d+)\s*(?:\\right\s*)?\)\s*", side)
    if m:
        a, b = int(m.group(2)), int(m.group(3))
        return sp.Integer(math.gcd(a, b) if m.group(1) == "gcd" else (a * b // math.gcd(a, b) if a and b else 0))
    m = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(?:\\[lc]dots|\.\.\.|…)\s*\+\s*(\d+)\s*", side)
    if m:
        a, b, z = (int(m.group(k)) for k in (1, 2, 3))
        step = b - a
        if step > 0 and z >= b and (z - a) % step == 0:
            n = (z - a) // step + 1
            return sp.Integer(n * (a + z)) / 2
    return None


def _normal_tails(side: str) -> str:
    """'\\Phi(1)', 'Q(\\sqrt{2 \\cdot 4})', 'Q(2)': the standard normal distribution's CDF / tail at a number, put in as that
    number (the parser read Phi times 1); left as written when the argument is no number"""
    out, i = [], 0
    for m in re.finditer(r"(\\Phi|(?<![A-Za-z\\])Q)\s*(\\left\s*)?\(", side):
        if m.start() < i:
            continue
        depth, j = 1, m.end()
        while j < len(side) and depth:
            depth += {"(": 1, ")": -1}.get(side[j], 0)
            j += 1
        if depth:
            return side
        inner = re.sub(r"\\right\s*$", "", side[m.end():j - 1])
        x = _num(_parse(inner)) if _parse(inner) is not None else None
        if not isinstance(x, (int, float)):
            continue
        v = 0.5 * (1 + math.erf(x / math.sqrt(2))) if m.group(1) == "\\Phi" else 0.5 * math.erfc(x / math.sqrt(2))
        out.append(side[i:m.start()] + f"({v:.12g})")
        i = j
    return "".join(out) + side[i:]


def _from_bases(side: str):
    """numbers written in a base -> in decimal ('(101.1)_2' -> 5.5, '(FF)_{16}' -> 255); None when a digit is not one
    of its base's ('(129)_2': a misreading)"""
    bad = []

    def rep(m):
        digits = re.sub(r"\s+", "", m.group("p") or m.group("d"))
        base = int(re.sub(r"\s+", "", m.group("b1") or m.group("b2") or m.group("b3")))
        if not 2 <= base <= 36:
            bad.append(m.group(0))
            return m.group(0)
        whole, _, frac = digits.partition(".")
        try:
            v = int(whole, base) + sum(int(c, base) * base ** -(k + 1) for k, c in enumerate(frac))
        except ValueError:
            bad.append(m.group(0))
            return m.group(0)
        return f"({v:.12g})"
    out = _BASE_NUM.sub(rep, side)
    return None if bad else out


_AT_JW = re.compile(r"(?<![A-Za-z\\])([A-Za-z])(_\{[^{}]*\}|_[A-Za-z0-9])?\s*(?:\\left\s*)?\(\s*j\s*(?:\\omega|w|ω|\d+(?:\.\d+)?)\s*"
                    r"(?:\\right\s*)?\)")


# a matrix written out ('\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}', bmatrix, Bmatrix, a plain matrix), and a transpose
# after it ('^T', '^t' as Israeli courses write it, '^{\top}'); a vmatrix is a determinant - a number
_MATRIX = re.compile(r"\\begin\s*\{\s*(?P<env>[pbB]?matrix|smallmatrix)\s*\}(?P<body>.*?)\\end\s*\{\s*(?P=env)\s*\}"
                     r"(?P<t>\s*\^\s*(?:\{\s*(?:T|t|\\top|\\intercal|\\mathsf\s*\{\s*T\s*\}|\\mathrm\s*\{\s*T\s*\})\s*\}|"
                     r"[Tt](?![A-Za-z])|\\top(?![A-Za-z])|\\intercal(?![A-Za-z])))?", re.S)
_DET_BEFORE = re.compile(r"(?:\\det|\\operatorname\s*\{\s*det\s*\}|\\mathrm\s*\{\s*det\s*\}|(?<![A-Za-z\\])det)\s*"
                         r"(?:\\left\s*)?\(?\s*$")


def _matrix_tex(side: str):
    """a side with matrices written out -> (its text for the parser: every matrix a pmatrix, a transpose done by hand;
    [the rows of cell texts of each matrix - None for one with rows of different lengths, a misreading]). A matrix in
    a determinant ('\\det \\begin{pmatrix} ... \\end{pmatrix}') is a number: left as it is, not in the list"""
    found = []

    def rep(m):
        if _DET_BEFORE.search(m.string[:m.start()]):
            return m.group(0)
        rows = [[c.strip() for c in r.split("&")] for r in re.split(r"\\\\", m.group("body")) if r.strip()]
        if not rows or len({len(r) for r in rows}) != 1 or any(not c for r in rows for c in r):
            found.append(None)
            return m.group(0)
        if m.group("t"):
            rows = [list(c) for c in zip(*rows)]
        found.append(rows)
        return " \\begin{pmatrix} " + " \\\\ ".join(" & ".join("{" + c + "}" for c in r) for r in rows) + " \\end{pmatrix} "
    #                                              (each entry in braces: the parser read 'c & d \end' as a differential)
    return _MATRIX.sub(rep, side), found


def _matrix_side(side: str):
    """a side of a statement -> ('scalar', None): no matrix in it (but in a determinant); ('matrix', M): its value
    worked out - an explicit sympy matrix, its letters real, j the imaginary unit; ('bad', None): matrices the parser
    cannot work out ('M + 2', a ragged one, a product of the wrong shapes)"""
    from .symbolic import _ece, _wrap, parse
    side = _join_digits(side.strip())
    t, found = _matrix_tex(side)
    if not found:
        return "scalar", None
    if None in found or sum(len(r) * len(r[0]) for r in found) > 81:
        return "bad", None
    try:
        e = next((x for x in parse(_wrap(t)) if isinstance(x, (sp.Basic, sp.MatrixBase))), None)
        if e is None:
            return "bad", None
        e = _ece(sp.ImmutableMatrix(e) if isinstance(e, sp.MatrixBase) else e)
        m = e.doit()
        if isinstance(m, sp.MatrixExpr) and not isinstance(m, sp.MatrixBase):
            m = m.as_explicit()
        if not isinstance(m, sp.MatrixBase) or max(m.shape) > 9:
            return "bad", None
        m = sp.ImmutableMatrix(m)
        plain = re.sub(r"\\[A-Za-z]+", " ", side)
        lone_j = re.search(r"(?<![A-Za-z])j(?![A-Za-z])", plain)
        m = m.xreplace({x: (sp.I if x.name == "j" and lone_j else sp.Symbol(x.name, real=True))
                        for x in m.free_symbols if isinstance(x, sp.Symbol)})
    except Exception:                              # noqa: BLE001
        return "bad", None
    return "matrix", m


def _fmt_mat(m) -> str:
    """a matrix for a message: '[-2, 1; 1.5, -0.5]' (rows by ';', as in MATLAB)"""
    return "[" + "; ".join(", ".join(_fmt_num(x) for x in row) for row in m.tolist()) + "]"


def _matrix_close(student, value, text: str):
    """two matrices in numbers equal, entry by entry - each to the digits it is written with when the side is a matrix
    written out (text); None: an entry with no value"""
    t, found = _matrix_tex(_join_digits((text or "").strip()))
    whole = re.fullmatch(r"\s*\\begin\{pmatrix\}.*\\end\{pmatrix\}\s*", t, re.S) and len(found) == 1 and found[0]
    cells = [c for r in found[0] for c in r] if whole else []
    for k, (a, b) in enumerate(zip(student, value)):
        ca, cb = _cnum(a), _cnum(b)
        if ca is None or cb is None:
            return None
        ct = cells[k] if len(cells) == len(student) else ""
        if abs(ca.imag) < 1e-12 and abs(cb.imag) < 1e-12:
            if not _close(ca.real, cb.real, ct):
                return False
        elif abs(ca - cb) > _ctol(ct, cb):
            return False
    return True


def _linear_in_letters(e) -> bool:
    syms = sorted(e.free_symbols, key=str)
    if not syms:
        return False
    try:
        return sp.Poly(sp.together(e).as_numer_denom()[0], *syms).total_degree() == 1
    except Exception:                              # noqa: BLE001
        return False


def _matrix_statements(part, sides_txt, conf, line, piece):
    """a statement about matrices written out -> its statements; None: no matrix in it (a determinant is a number);
    []: nothing to check (a matrix the parser cannot work out, a lone one). A chain of matrices in numbers ('M^{-1} =
    \\frac{1}{-2} [...] = [...]'): each step compared entry by entry, the entries to the digits written; a step in
    letters that holds: right; 'A x = b' in letters with A and b in numbers: its rows - a system of equations, for the
    values written after it. Names before the matrices ('A^{-1} = ...', '\\vec{V} = ...') are left out"""
    got = [_matrix_side(t) for t in sides_txt]
    if all(k == "scalar" for k, _ in got):
        return None
    first = next(i for i, (k, _) in enumerate(got) if k != "scalar")
    if any(k != "matrix" for k, _ in got[first:]):
        return []
    mats, texts = [m for _, m in got[first:]], sides_txt[first:]
    if len(mats) < 2 or len({m.shape for m in mats}) != 1:
        return []
    steps, rows = [], []
    for k in range(len(mats) - 1):
        a, b, txt = mats[k], mats[k + 1], f"{_pretty(texts[k])} = {_pretty(texts[k + 1])}"
        if not a.free_symbols and not b.free_symbols:
            ok = _matrix_close(b, a, texts[k + 1])
            if ok is not None:
                steps.append((txt, ok, f"'{txt}'" + ("" if ok else f": {_fmt_mat(a)} is not {_fmt_mat(b)}")))
            continue
        same = [_same_expr(x, y) for x, y in zip(a, b)]
        if all(v is True for v in same):
            steps.append((txt, True, f"'{txt}'"))
            continue
        eqs = [(x, y) for x, y, v in zip(a, b, same) if v is not True]
        if all(_linear_in_letters(x - y) for x, y in eqs) and not (a.free_symbols & b.free_symbols):
            rows += eqs                              # ('A x = b'; not a step in letters that does not hold: '[a, b; c,
            #                                          d]^{-1} = \frac{1}{ad - bc} [a, -b; -c, d]' - no equations to solve)
    out = []
    if steps:
        out.append(Statement("matrix", part, list(mats), conf, line=line, piece=piece, side_texts=list(texts),
                             calc={"steps": steps}))
    for x, y in rows:
        out += statements(f"{sp.latex(x)} = {sp.latex(y)}", conf, line, piece)
    return out


def _matrix_checks(no, s):
    return [Check(no, "right" if ok else "wrong step", msg, s.conf, s.conf >= SURE, [s.text], line=s.line)
            for _, ok, msg in s.calc.get("steps", [])]


# the phase of a response at a frequency ('\angle H(j1000)', '\arg H(j\omega)', '\angle\left(H(j2)\right)')
_ANGLE_AT = re.compile(r"\s*(?:\\angle|\\arg|\\measuredangle|\\sphericalangle|\\phase)\s*(?:\\left\s*)?\(?\s*"
                       r"([A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?)\s*(?:\\left\s*)?\(\s*j\s*(?:\\omega|w|ω|\d+(?:\.\d+)?)\s*"
                       r"(?:\\right\s*)?\)\s*(?:\\right\s*)?\)?\s*")


# a part of a complex letter, a whole side: its angle ('\angle Z', '\arg(Z)'), real or imaginary part ('\Re\{Z\}', 'Re(I)',
# '\mathrm{Im}\{V\}')
_PART_NAME = r"(?P<n>[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?)"
_PART_OF = re.compile(r"\s*(?:(?P<a>\\angle|\\arg|\\measuredangle|\\phase)|(?P<re>\\Re|\\mathrm\s*\{\s*Re\s*\}|\\operatorname\s*\{\s*Re\s*\}|"
                      r"(?<![A-Za-z\\])Re)|(?P<im>\\Im|\\mathrm\s*\{\s*Im\s*\}|\\operatorname\s*\{\s*Im\s*\}|(?<![A-Za-z\\])Im))\s*"
                      r"(?:(?:\\left\s*)?(?:\\\{|\{|\()\s*" + _PART_NAME + r"\s*(?:\\right\s*)?(?:\\\}|\}|\))|" +
                      _PART_NAME.replace("?P<n>", "?P<n2>") + r")\s*")


_SUB_NAME = re.compile(r"(?<![A-Za-z\\])[A-Za-z]\s*_\s*(?:\{\s*[A-Za-z]|[A-Za-z])")   # a letter with a subscript (a name, never a number)


@functools.lru_cache(maxsize=8192)
def _parse_full(side: str):
    """a side -> (its expression worked out, as written: sums, integrals, limits not worked out); (None, None): no
    expression"""
    from .symbolic import _ece, _first_sympy, _wrap, parse
    side = _join_digits(side.strip())
    side = re.sub(r"(\d\.\d+)\s*(?:\.\.\.|…|\\[cl]?dots(?![A-Za-z]))\s*$", r"\1", side)   # '0.333...': the decimals go on
    ang = _ANGLE_AT.fullmatch(side) if side else None
    if ang:                                        # '\angle H(j1000)': the phase of a response - a name here, its value
        x = sp.Symbol("angle_" + re.sub(r"\W", "", ang.group(1)), real=True)     # worked out by _at_jw
        return x, x
    part = _PART_OF.fullmatch(side) if side else None
    if part:                                       # '\angle Z', '\Re\{Z\}', 'Im(I)': a part of a complex letter (the
        name = _parse_full(part.group("n") or part.group("n2"))[0]     # parser read R times e times Z) - worked out with
        #                                                  its value (_exact)
        if isinstance(name, sp.Symbol):
            kind = "Arg" if part.group("a") else ("Re" if part.group("re") else "Im")
            x = sp.Function(kind)(name)
            return x, x
    whole = _integer_side(side) if side else None
    if whole is not None:                          # '17 \bmod 5', '\gcd(12, 18)', '1 + 2 + \ldots + 10': worked out (no value before)
        return whole, whole
    if side and re.search(r"\\Phi\s*(?:\\left\s*)?\(|(?<![A-Za-z\\])Q\s*(?:\\left\s*)?\(", side):
        side = _normal_tails(side)                 # 'Q(2) = 0.0228', '\Phi(1) = 0.8413': the normal distribution's
    na = re.fullmatch(r"\s*(?:\\angle|\\arg|\\(?:operatorname|mathrm|text)\s*\{\s*arg\s*\}|arg)\s*(?:\\left\s*)?\(\s*(.+?)\s*(?:\\right\s*)?\)\s*",
                      side) if side else None
    if na:                                         # '\angle(1 + j)', '\arg(3 + j4)': a complex number's angle (no verdict: the
        inner = _parse_full(na.group(1))[0]        # parser read angle times (1 + j))
        if isinstance(inner, sp.Basic) and not inner.free_symbols and _cnum(inner) is not None and abs(_cnum(inner)) > 0:
            x = sp.arg(inner)
            return x, x
    if not side or _COMPLEX_PARTS.search(side) or _NO_VALUE.search(side):
        return None, None
    if _MATRIX.search(side) and _matrix_tex(side)[1]:     # a matrix written out (not in a determinant): no number - the
        return None, None                                  # parser took 'M + 2' for 2, 'M^{\top}' for 4 (_matrix_side)
    if _ARG_TIME.search(side):                     # 'v_C(1 ms)': a time function at a time with its unit, in seconds
        side = _ARG_TIME.sub(_arg_seconds, side)   # (it was read as v_C times 1 m s)
    if _BASE_MORE.search(side):
        side = _more_bases(side)
    if _BASE_NUM.search(side):                     # '(1011)_2', '1011_{(2)}', '(FF)_{16}': a number in its base (it was
        side = _from_bases(side)                   # read as 1011, or a falling factorial: a false 'wrong step')
        if side is None:
            return None, None
    # 'H(j\\omega)', 'Z(j1000)': a response at a frequency, by its letter (the parser read H times j times omega - and
    # '|H| = 0.707' was then worked out for that product)
    side = _AT_JW.sub(lambda m: m.group(1) + (m.group(2) or ""), side)
    # '\frac{d^2}{dx^2} x^3', '\frac{d^2 y}{dx^2}': the second derivative - the parser takes d/dx twice, not d^2
    side = re.sub(r"\\frac\s*\{\s*d\s*\^\s*\{?\s*([23])\s*\}?\s*([A-Za-z]?)\s*\}\s*\{\s*d\s*([A-Za-z])\s*\^\s*\{?\s*\1\s*\}?\s*\}",
                  lambda m: ("\\frac{d}{d" + m.group(3) + "}") * (int(m.group(1)) - 1)
                  + "\\frac{d" + m.group(2) + "}{d" + m.group(3) + "}", side)
    try:
        if "[" in side or "\\left." in side:
            side = _eval_brackets(side)
        side = _close_denominators(side)           # ('R_2/(R_1 + R_2) V_{in}': V_{in} times the ratio)
        e = _ece(_first_sympy(parse(_wrap(_primes(side)))))
    except Exception:                              # noqa: BLE001 - any parser failure: not a statement
        return None, None
    if isinstance(e, sp.Basic) and not e.free_symbols and _SUB_NAME.search(side):
        from sympy.core.function import AppliedUndef
        if not e.atoms(AppliedUndef):              # a name's letters lost on the way ('0.69R_{eq}C_{in\_ inv}' read as 0.69 - a
            return None, None                      # piece of the side for all of it: a false wrong step): no expression
    if isinstance(e, sp.Basic):                    # every letter real, as the parser makes them - but for the t of
        e = e.xreplace({x: sp.Symbol(x.name, real=True) for x in e.free_symbols   # 'i(t)', which it makes complex
                        if isinstance(x, sp.Symbol) and x.is_real is not True and x.name != "j"})
    raw = e
    if isinstance(e, sp.Expr) and e.has(sp.Derivative):
        e = raw = _opaque_derivatives(e, partial="\\partial" in side)
    if isinstance(e, sp.Expr) and e.has(sp.Derivative, sp.Integral, sp.Limit, sp.Sum, sp.Product):
        try:                                       # a sum, a limit, an integral worked out: its variable is no letter
            done = e.doit()                        # of the problem ('\sum_{k=1}^{4} k = 10')
            if done.has(sp.Piecewise) and e.has(sp.Integral):
                # '\int_0^\infty e^{-at} dt': a convergence condition on a - the problem's letters taken positive, as
                # the student takes them ('= \frac{1}{a}' got no verdict)
                pos = {x: sp.Symbol(x.name, positive=True) for x in e.free_symbols if isinstance(x, sp.Symbol)}
                again = e.xreplace(pos).doit()
                if not again.has(sp.Piecewise, sp.Integral):
                    done = again.xreplace({v: k for k, v in pos.items()})
            e = done
        except Exception:                          # noqa: BLE001
            pass
    js = [x for x in e.free_symbols if isinstance(x, sp.Symbol) and x.name == "j"] if isinstance(e, sp.Expr) else []
    if js:                                         # j is the imaginary unit ('3 + j4'), as engineers write it - a capital
        plain = re.sub(r"\\[A-Za-z]+", " ", side)   # J alone is a letter (a current density, a moment of inertia)
        if re.search(r"(?<![A-Za-z])j(?![A-Za-z])", plain) or not re.search(r"(?<![A-Za-z_])J(?![A-Za-z])", plain):
            e = e.xreplace({x: sp.I for x in js})
        else:
            e = e.xreplace({x: sp.Symbol("J", real=True) for x in js})
    return (e, raw) if isinstance(e, sp.Expr) else (None, None)


def _decimals(text: str) -> int:
    m = re.search(r"\d\s*\.\s*((?:\d\s*)+)$", text.strip())
    return len(re.sub(r"\s", "", m.group(1))) if m else 0


# a number as written: its digits, its decimals, a power of ten after it ('2.6 \\cdot 10^{-2}', a calculator's '2.6E-2')
_NUM_TEXT = re.compile(r"\(?\s*[+-]?\s*(\d+)(?:\.(\d+))?\s*\)?\s*"
                       r"(?:(?:\\cdot|\\times|\*|×|·)\s*10\s*\^\s*\{?\s*([+\-−]?\s*\d+)\s*\}?|[eE]\s*([+-]?\s*\d+))?")


def _half_unit(text) -> float:
    """half a unit in the last digit written, as a slack: '1.4' -> 0.051, '2.6 \\cdot 10^{-2}' -> 0.00051, '14' -> 0.51
    (an integer from two digits on - '5' for 4.6 is wrong); 0 for none. A text that is no number alone: the decimals at
    its end, as before"""
    t = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", (text or "").strip())
    pct = re.fullmatch(r"(.*?)\s*\\cdot\s*0\.010", t)          # '83\%': half a unit of 83, in hundredths
    if pct:
        return _half_unit(pct.group(1)) * 0.01
    m = _NUM_TEXT.fullmatch(t)
    if not m:
        d = _decimals(t)
        return 0.51 * 10 ** -d if d else 0.0
    whole, dec, ex = m.group(1), m.group(2), m.group(3) or m.group(4)
    e = int(re.sub(r"\s", "", ex).replace("−", "-")) if ex else 0
    if dec:
        return 0.51 * 10 ** (e - len(dec))
    return 0.51 * 10 ** e if len(whole.lstrip("0")) >= 2 else 0.0


def _close(student: float, value: float, text: str = "", scale: float = 1.0) -> bool:
    """equal up to 1 %, or up to half a unit in the last digit written ('\\sqrt{2} = 1.4', '\\frac{10}{3} = 3.3',
    '2.6 \\cdot 10^{-2}' for 0.02553) - an integer gets such slack only from two digits on: 5 for 4.6 is wrong, 14 for
    14.29 is a rounding, 79 for 78.54 is within 1 %. scale: what the text's number was multiplied by (a unit's prefix:
    '26' of '26 mA' compared in SI)"""
    if isinstance(student, complex) or isinstance(value, complex):
        return abs(student - value) <= _ctol(text, value)
    slack = min(_half_unit(text) * abs(scale), 0.15 * abs(value))      # (at most 15 %: '0.0001' is no rounding of
    # (0.00005); a floor only beside a zero (rounding noise of a sum that is 0): with one for all values, every value
    # under 1e-9 was equal to every other - '10 pF + 20 pF = 50 pF', '1.6 \cdot 10^{-19} \cdot 5 = 8 \cdot 10^{-18}' right
    tol = max(REL_TOL * abs(value), slack, 1e-9 if student == 0 or value == 0 else 0.0)
    return abs(student - value) <= tol * (1 + 1e-9)       # ('1.01' for 1: 1 % by the float's last bit - it was wrong)


_DEG_NUM = re.compile(r"\s*[+-]?\s*\d+(?:\.\d+)?\s*(?:\^\s*\{?\s*\\circ\s*\}?|°)\s*")     # '45^\circ', '-30°': an angle alone
_TRIG = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
_ATRIG = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc)
# the angles of simple ratios, in degrees (53.13 = arctan 4/3, 36.87, 26.57 = arctan 1/2, 30 = arcsin 1/2)
_RATIO_ANGLES = sorted({math.degrees(math.atan2(p, q)) for p in range(1, 13) for q in range(1, 13)} |
                       {math.degrees(math.asin(p / q)) for q in range(2, 13) for p in range(1, q)})


def _imag_arg(f):
    """an exp of j times a number without pi - an angle, in degrees as written ('e^{j45}')? -: that number, else None"""
    a = f.args[0] if f.args else None
    if a is None or not a.is_number or a.has(sp.pi):
        return None
    try:
        z = complex(a)
    except (TypeError, ValueError):
        return None
    return z.imag if abs(z.real) < 1e-12 and abs(z.imag) > 1e-12 else None


def _in_degrees(e):
    """e with its angles in degrees: sin / cos / tan of x (x without pi) -> of x pi/180, e^{jx} -> e^{jx pi/180};
    arcsin / arccos / arctan -> times 180/pi. None: nothing to change (or it cannot be rebuilt)"""
    if not isinstance(e, sp.Basic) or not (e.has(*_TRIG, *_ATRIG) or any(_imag_arg(f) is not None for f in e.atoms(sp.exp))):
        return None

    def walk(x):
        if not x.args:
            return x
        args = [walk(a) for a in x.args]
        if isinstance(x, _TRIG) and not x.args[0].has(sp.pi):
            return x.func(args[0] * sp.pi / 180)
        if isinstance(x, _ATRIG):
            return x.func(*args) * 180 / sp.pi
        if isinstance(x, sp.exp) and _imag_arg(x) is not None:
            return sp.exp(args[0] * sp.pi / 180)
        return x.func(*args)
    try:
        return walk(e)
    except Exception:                              # noqa: BLE001 - a form that does not rebuild from its arguments
        return None


def _angle_args(e) -> list:
    """the angles _in_degrees reads in degrees: the numbers in sin / cos / tan and in e^{jx}"""
    if not isinstance(e, sp.Basic):
        return []
    return ([_num(f.args[0]) for f in e.atoms(*_TRIG) if not f.args[0].has(sp.pi)] +
            [v for v in (_imag_arg(f) for f in e.atoms(sp.exp)) if v is not None])


def _angle_radians(e):
    """a phasor's angle read in radians ('5 \\angle 0.927', taken as degrees): cos / sin of c pi/180 -> of c, for |c| up to
    2 pi -> (expression, sure: no angle a whole number) or None"""
    rep, vals = {}, []
    for f in e.atoms(sp.sin, sp.cos) if isinstance(e, sp.Basic) else ():
        c = f.args[0] * 180 / sp.pi
        if c.is_number and not c.has(sp.pi):
            v = float(c)
            if 1e-9 < abs(v) <= 2 * math.pi + 1e-9:
                rep[f], _ = f.func(sp.Float(v)), vals.append(v)
    return (e.xreplace(rep), all(abs(v - round(v)) > 1e-9 for v in vals)) if rep else None


def _degree_like(x) -> bool:
    """an angle as written in degrees without the mark: a multiple of 15 (or 37, 53), or the angle of a simple ratio
    to the decimals written (53.13, 36.87, 116.57); never one up to 2 pi - a radian argument worked out from omega t
    is rarely either ('5 \\cos(10) = 4.92' with 10 = 1000 \\cdot 0.01: a calculator in degrees, a mistake)"""
    if x is None:
        return False
    a = abs(float(x))
    if a <= 2 * math.pi or a > 360.5:
        return False
    if abs(a / 15 - round(a / 15)) < 1e-9 or a in (37.0, 53.0):
        return True
    d = next((k for k in range(1, 5) if abs(a * 10 ** k - round(a * 10 ** k)) < 1e-6 * 10 ** k), None)
    if d is None:
        return False
    a = a % 180
    a = min(a, 180 - a)
    return any(abs(a - r) <= 0.51 * 10 ** -d for r in _RATIO_ANGLES)


def _angle_readings(e, text="") -> list:
    """other readings of a side's angles: [(expression, sure)] - in degrees, the mark left out ('\\cos(60)', 'e^{j45}');
    a phasor's angle in radians ('5 \\angle 0.927')"""
    out = []
    d = _in_degrees(e)
    if d is not None:
        out.append((d, all(_degree_like(v) for v in _angle_args(e))))
    if "angle" in str(text) and not re.search(r"\\circ|°", str(text)):
        r = _angle_radians(e)
        if r is not None:
            out.append(r)
    return out


def _angles_step(a, b, ta="", tb="", value=None, close=None):
    """a step false with its angles as read ('\\cos(60) = 0.5', '\\arctan(\\frac{4}{3}) = 53.13', '1 + j = 1.414 e^{j45}'):
    does it hold with them in degrees (or a phasor's in radians)? True (each angle as students write degrees, or an
    arctan's value in degrees), None (it holds, but an angle may be radians a calculator in degrees took wrong), False"""
    value = value or _num
    close = close or (lambda x, y: _close(y, x) or _close(x, y))
    ra, rb = [(a, True)] + _angle_readings(a, ta), [(b, True)] + _angle_readings(b, tb)
    got = False
    for k1, (x, sx) in enumerate(ra):
        for k2, (y, sy) in enumerate(rb):
            if k1 == 0 and k2 == 0:
                continue
            vx, vy = value(x), value(y)
            if vx is not None and vy is not None and close(vx, vy):
                if sx and sy:
                    return True
                got = None
    return got


def _is_db(unit) -> bool:
    return bool(unit) and re.sub(r"\s+", "", unit) == "dB"


def _db_match(student_db, ratio, text: str = "") -> int:
    """'= 40 dB' for a ratio of 100: 20 (20 log10, an amplitude's), 10 (10 log10, a power's) or 0 - within half a unit
    in the last decimal written ('40 dB' for 39.6, a rounding), at least 0.05 dB"""
    if not isinstance(ratio, (int, float)) or not isinstance(student_db, (int, float)) or ratio == 0 or not math.isfinite(ratio):
        return 0
    d = _decimals(text)
    tol = max(0.51 * 10 ** -d, 0.05)
    for k in (20, 10):
        if abs(k * math.log10(abs(ratio)) - student_db) <= tol:
            return k
    return 0


_BIN_TOKEN = re.compile(r"(?<![\d.])[01]+(?![\d.])")
_MIXED = re.compile(r"(?<![\d.^_{])(\d+)\s*(?=\\[dt]?frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\})")   # '2 \frac{1}{2}': 2.5, or 1?


def _binary_step(ta: str, tb: str):
    """a step of binary arithmetic or Boolean algebra written without the base, as a digital-systems exam has it ('1 + 1 =
    10', '101 + 011 = 1000', '1111 = 15', '1 + 1 = 1'), false in decimal: 'binary' or 'boolean' when it holds read so,
    else None. Binary: every number of 0s and 1s, one of two digits at least (a lone one beside a decimal number: a
    conversion); Boolean: single 0s and 1s with + and \\cdot only - a sum or product of them is 1 when its value is positive"""
    nums = re.findall(r"\d+(?:\.\d+)?", f"{ta} {tb}")
    if not nums or any("." in n for n in nums):
        return None
    bin_a = all(set(n) <= {"0", "1"} for n in re.findall(r"\d+", ta))
    bin_b = all(set(n) <= {"0", "1"} for n in re.findall(r"\d+", tb))
    b2 = lambda t: _BIN_TOKEN.sub(lambda m: str(int(m.group(0), 2)), t)      # noqa: E731
    cands = []
    if bin_a and bin_b and any(len(n) >= 2 for n in nums):
        cands.append((b2(ta), b2(tb)))
    if bin_a and not bin_b and re.fullmatch(r"\s*[01]{2,}\s*", ta):
        cands.append((b2(ta), tb))                   # '1111 = 15'
    if bin_b and not bin_a and re.fullmatch(r"\s*[01]{2,}\s*", tb):
        cands.append((ta, b2(tb)))
    for x, y in cands:
        vx, vy = _num(_parse(x)), _num(_parse(y))
        if vx is not None and vy is not None and abs(vx - vy) < 1e-9:
            return "binary"
    for bits, other in ((ta, tb), (tb, ta)):         # '-5 = 1011': two's complement, in as many bits as written (also
        m = re.fullmatch(r"\s*(?:\\left\s*)?\(?\s*(1[01]+)\s*(?:\\right\s*)?\)?\s*(?:_\s*\{?\s*\(?\s*2\s*\)?\s*\}?)?\s*", bits)   # '(11111011)_2')
        vo = _num(_parse(other)) if m and not re.fullmatch(r"\s*[01]+\s*", other) else None
        if vo is not None and vo < 0 and abs(int(m.group(1), 2) - 2 ** len(m.group(1)) - vo) < 1e-9:
            return "binary"
    if bin_a and bin_b and all(len(n) == 1 for n in nums) and not re.search(r"-|/|\^|\\frac|\\overline|\\bar|'|[A-Za-z]",
                                                                           re.sub(r"\\cdot|\\times", "", ta + tb)):
        va, vb = _num(_parse(ta)), _num(_parse(tb))
        if va is not None and vb in (0, 1) and va >= 0 and (va > 0) == (vb > 0):
            return "boolean"
    return None


def _binary_hint(ta: str, tb: str):
    """a binary step false either way ('101 + 011 = 1001'): the left side's value in binary ('1000'), else None"""
    nums = re.findall(r"\d+(?:\.\d+)?", f"{ta} {tb}")
    if not nums or any("." in n or set(n) - {"0", "1"} for n in nums) or not any(len(n) >= 2 for n in nums):
        return None
    v = _num(_parse(_BIN_TOKEN.sub(lambda m: str(int(m.group(0), 2)), ta)))
    return bin(int(round(v)))[2:] if v is not None and v >= 0 and abs(v - round(v)) < 1e-9 else None


# a step right, not for sure: its note after this mark (in the step's first value); a step wrong, not for sure: its
# message after the other mark
_UNSURE = "\ue030"
_MAYBE = "\ue031"
_SUM_READ = "\ue041"                    # (a slash's reading with brackets the student left out: 'V/R_1+R_2' as V/(R_1+R_2))
_DEG_NOTE = "with its angle in degrees (or radians, and a calculator in degrees?)"
_K_NOTE = "with its numbers in units they were not written in (kΩ, so mA?) - 1000 apart"
_CM_NOTE = "with its numbers in centimetres, as semiconductor physics keeps them (cm, cm^{-3}) - the result in another unit"
_BOOL_NOTE = "in Boolean algebra (1 + 1 = 1) - a digital-systems exam?"
_DB10_NOTE = "in dB as a power ratio, 10 log10 (an amplitude's is 20 log10)"


def _step_check(no, txt, ok, va, vb, conf, texts):
    """a chain's step -> its Check: 'wrong step' with its two values; a step right only with an angle in degrees that
    may have been radians, or with dB as a power's: right, not sure"""
    unsure = bool(ok) and isinstance(va, str) and va.startswith(_UNSURE)
    if not ok and isinstance(va, str) and va.startswith(_MAYBE):        # ('1000 apart: units not written?')
        return Check(no, "wrong step", f"'{txt}': {va[len(_MAYBE):]}", conf, False, texts)
    return Check(no, "right" if ok else "wrong step", f"'{txt}'" + ("" if ok else f": {_fmt(va)} is not {_fmt(vb)}") +
                 (f" - {va[len(_UNSURE):]}" if unsure else ""), conf, conf >= SURE and not unsure, texts)


_POLAR_TEXT = re.compile(r"(\d+(?:\s*\.\s*\d+)?)\s*\\angle\s*\{?\s*[+-]?\s*(\d+(?:\s*\.\s*\d+)?)")
_RECT_TEXT = re.compile(r"(\d+(?:\s*\.\s*\d+)?)\s*[+-]\s*(?:j\s*(\d+(?:\s*\.\s*\d+)?)|(\d+(?:\s*\.\s*\d+)?)\s*j)")


def _ctol(text: str, value) -> float:
    """how far a complex number as written may be from the value: 1 %, or half a unit in the last decimal of each of
    its numbers - '0.20\\angle 41.82^\\circ' for 0.196\\angle 41.82° (the size to 2 decimals), '1.4 - j1.9' for 1.44 - j1.92"""
    v = abs(complex(value))
    tol = max(REL_TOL * v, 1e-9)
    dec = lambda t: len(re.sub(r"\s", "", t.split(".")[1])) if "." in t else 0          # noqa: E731
    p10 = re.search(r"\\cdot\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?\s*\)?\s*$", text or "")
    k = 10.0 ** int(p10.group(1)) if p10 else 1.0  # ('((-0.74 - j2.21) \cdot 10^{-3})', put in SI units: its decimals in mA -
    #                                                the half unit of its last decimal in amperes, not 0.005 A)
    m = _POLAR_TEXT.search(text or "")
    if m:
        dm, da = dec(m.group(1)), dec(m.group(2))
        return max(tol, (0.51 * 10 ** -dm * k if dm else 0.0) + v * (0.51 * 10 ** -da) * math.pi / 180)
    m = _RECT_TEXT.search(text or "")
    if m:
        dr, di = dec(m.group(1)), dec(m.group(2) or m.group(3))
        return max(tol, math.hypot(0.51 * 10 ** -dr * k if dr else 0.0, 0.51 * 10 ** -di * k if di else 0.0))
    d = _decimals(text or "")
    return max(tol, 0.51 * 10 ** -d if d else 0.0)


def _cnum(e):
    """a side's value as a complex number (None: letters left, or no number)"""
    try:
        v = complex(sp.N(_exact(e)))
    except Exception:                              # noqa: BLE001 - (a matrix power: sympy recursed without end)
        return None
    return v if math.isfinite(v.real) and math.isfinite(v.imag) else None


def _fmt_c(v: complex) -> str:
    if abs(v.imag) < 1e-9:
        return _fmt(v.real)
    if abs(v.real) < 1e-9 * max(1.0, abs(v.imag)):            # '-j2', not '0 - j2'
        return f"{'-' if v.imag < 0 else ''}j{_fmt(abs(v.imag))}"
    return f"{_fmt(v.real)} {'+' if v.imag >= 0 else '-'} j{_fmt(abs(v.imag))}"


def _exact(e):
    """the decimals of an expression as exact fractions ('0.5 + 0.3 - 0.8' is 0, not 5.6e-17): a value near zero is then
    no rounding noise of binary floats - and needs no floor of tolerance that made every value under 1e-9 equal; the
    parts of a complex number (Re, Im, Arg of _PART_OF) worked out"""
    if isinstance(e, sp.Basic) and e.has(sp.Float):
        e = e.xreplace({f: sp.Rational(str(f)) for f in e.atoms(sp.Float)})
    if isinstance(e, sp.Basic):
        from sympy.core.function import AppliedUndef
        rep = {}
        for f in e.atoms(AppliedUndef):
            if str(f.func) in ("Re", "Im", "Arg") and len(f.args) == 1 and f.args[0].is_number:
                rep[f] = {"Re": sp.re, "Im": sp.im, "Arg": sp.arg}[str(f.func)](f.args[0])
        if rep:
            e = e.xreplace(rep)
    return e


def _slash_reading(text):
    """'1/2\\pi f C' -> '\\frac{1}{2\\pi f C}': a slash before a product - its denominator the whole product, as students
    write it (the parser takes (1/2) \\pi f C, by the rules); None when there is no such slash (a single factor after it:
    the same either way)"""
    t = (text or "").strip()
    m = re.fullmatch(r"(?P<pre>.*?)(?P<num>\d+(?:\.\d+)?|[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?|\([^()]*\))\s*/\s*(?P<den>.+)", t, re.S)
    if not m or m.group("pre").count("{") != m.group("pre").count("}"):
        return None
    den, depth = m.group("den"), 0
    for i, ch in enumerate(den):                     # a product only: no + or - outside brackets
        depth += {"{": 1, "(": 1, "[": 1, "}": -1, ")": -1, "]": -1}.get(ch, 0)
        if depth < 0 or (depth == 0 and ch in "+-=" and i > 0 and den[i - 1] not in "^_{"):
            return None
    if depth or re.fullmatch(r"\s*(?:\d+(?:\.\d+)?|[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?|\\[A-Za-z]+)\s*", den):
        return None
    return f"{m.group('pre')}\\frac{{{m.group('num')}}}{{{den}}}"


_SL_FUNCS = ("log", "ln", "lg", "sin", "cos", "tan", "cot", "arctan", "arcsin", "arccos", "sinh", "cosh", "tanh", "exp")
_SL_OPS = ("cdot", "times", "div", "pm", "mp", "le", "ge", "leq", "geq", "approx", "neq", "to", "quad", "qquad", "text",
           "mathrm", "rm", "textrm", "operatorname", "right", "left", "end", "begin")
# the factor just before a slash: a number, a letter with its subscript, a bracketed group
_SL_NUM = re.compile(r"(?:(?<![A-Za-z0-9.\\])\d+(?:\.\d+)?|(?<![A-Za-z\\])[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?|\([^()]*\))\s*$")


def _sl_group_end(t: str, i: int) -> int:
    """t[i] an opening bracket: the index past its closing one; -1 when unclosed"""
    depth = 0
    for j in range(i, len(t)):
        if t[j] in "({[":
            depth += 1
        elif t[j] in ")}]":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _sl_factor_end(t: str, i: int, depth: int = 0) -> int:
    """the end of the factor at t[i] (spaces before it skipped): a number, a letter with its subscript, a command with its
    braced arguments ('\\sqrt{LC}', '\\pi', '\\omega_0'), a function with its argument ('\\sin(x)'), a bracketed group -
    a power after any of them; -1 when no factor starts there"""
    while i < len(t) and t[i] == " ":
        i += 1
    if i >= len(t) or depth > 12:                    # (functions of functions without end: no factor - it recursed past the
        return -1                                    # stack's depth: '1/\sin \sin \sin ... x')
    c = t[i]
    if c.isdigit():
        j = i + re.match(r"\d+(?:\.\d+)?", t[i:]).end()
    elif c.isalpha():
        j = i + re.match(r"[A-Za-z](?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]))?", t[i:]).end()
    elif c == "\\":
        m = re.match(r"\\([A-Za-z]+)", t[i:])
        if not m or m.group(1) in _SL_OPS:
            return -1
        j = i + m.end()
        m2 = re.match(r"\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", t[j:])
        if m2:
            j += m2.end()
        while True:                                  # its braced arguments ('\frac{1}{2}', '\sqrt[3]{x}')
            k = j
            while k < len(t) and t[k] == " ":
                k += 1
            if k < len(t) and (t[k] == "{" or (t[k] == "[" and m.group(1) == "sqrt")):
                e = _sl_group_end(t, k)
                if e < 0:
                    return -1
                j = e
            else:
                break
        if m.group(1) in _SL_FUNCS:                  # a function: its argument with it ('\sin(x)', '\ln 2')
            e = _sl_factor_end(t, j, depth + 1)
            if e < 0:
                return -1
            j = e
    elif c in "([":
        j = _sl_group_end(t, i)
        if j < 0:
            return -1
    else:
        return -1
    m = re.match(r"\s*\^\s*", t[j:])                 # a power
    if m:
        k = j + m.end()
        if k < len(t) and t[k] == "{":
            e = _sl_group_end(t, k)
            if e < 0:
                return -1
            j = e
        else:
            m3 = re.match(r"[A-Za-z0-9]|\\[A-Za-z]+", t[k:])
            if not m3:
                return -1
            j = k + m3.end()
    m = re.match(r"\s*(?:'|!)", t[j:])               # (a prime, a factorial)
    if m:
        j += m.end()
    return j


def _slash_alternatives(text) -> list:
    """a side's other readings at its slashes: [(text, how it was read)] - 'E = 1/2 C V^2': a half times C V^2 (the parser
    takes 1/(2 C V^2), as it must '1/2\\pi f C'), '(R)/(L)s': R/L times s, 'R_2/(R_1 + R_2) V_{in}': V_{in} times the
    ratio, '20 \\log(V_o)/(V_i)': the log of the ratio; each slash by itself, then all of them one way; [] when no slash
    leaves a choice (a slash in braces, as in 'e^{-t/RC}', is the exponent's: t over RC)"""
    t = re.sub(r"\\left\s*([(\[])", r"\1", text or "")
    t = re.sub(r"\\right\s*([)\]])", r"\1", t)
    depth, slashes, top = 0, [], set()
    for p, ch in enumerate(t):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "/" and depth == 0:
            slashes.append(p)
    pdepth = 0
    for p, ch in enumerate(t):                       # (the slashes outside any bracket: a sum may be meant above or
        if ch in "({[":                              # below them)
            pdepth += 1
        elif ch in ")}]":
            pdepth -= 1
        elif ch == "/" and pdepth == 0:
            top.add(p)

    def inner(x):
        return x[1:-1] if x.startswith("(") and _sl_group_end(x, 0) == len(x) else x

    def has_sum(x):                                  # a + or - between terms, outside brackets
        d = 0
        for i, ch in enumerate(x):
            if ch in "({[":
                d += 1
            elif ch in ")}]":
                d -= 1
            elif ch in "+-" and d == 0:
                before = x[:i].rstrip()
                if before and before[-1] not in "^_=(*/+-" and not re.search(r"\\(?:cdot|times)$", before):
                    return True
        return False
    narrow, wide, sums = [], [], []                  # [(start, end, the text there, how it reads)]
    for p in slashes[:4]:
        m = _SL_NUM.search(t[:p])
        ends, j = [], p + 1
        while len(ends) < 12:
            e = _sl_factor_end(t, j)
            if e < 0:
                break
            ends.append(e)
            j = e
        if p in top and ends:                        # 'V/R_1+R_2': V over the sum; 'V_{CC} - V_{BE}/R_B': the difference
            rest, before = t[p + 1:].strip(), t[:p].strip()          # over R_B - as written without its brackets
            f1 = t[p + 1:ends[0]].strip()
            if m and has_sum(rest) and not re.search(r"/|\\frac", rest):
                sums.append((m.start(), len(t), f"\\frac{{{inner(m.group(0).strip())}}}{{{rest}}}", _SUM_READ + f"{m.group(0).strip()}/({rest})"))
            if has_sum(before) and not has_sum(t[ends[0]:]) and not re.search(r"/|\\frac", before):
                sums.append((0, ends[0], f"\\frac{{{before}}}{{{inner(f1)}}}", _SUM_READ + f"({before})/{f1}"))
        if not m:
            continue
        num, start = m.group(0).strip(), m.start()
        if not ends:
            continue
        f1 = t[p + 1:ends[0]].strip()
        if len(ends) >= 2:                           # a product after the slash: its first factor alone below, or all
            rest = t[ends[0]:ends[-1]].rstrip()
            narrow.append((start, ends[0], f"\\frac{{{inner(num)}}}{{{inner(f1)}}}", f"({num}/{f1}){rest}"))
            whole = t[p + 1:ends[-1]].strip()
            wide.append((start, ends[-1], f"\\frac{{{inner(num)}}}{{{whole}}}", f"{num}/({whole})"))
        elif num.startswith("("):
            fm = re.search(r"\\(?:" + "|".join(_SL_FUNCS) + r")(?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]))?(?:\s*\^\s*(?:\{[^{}]*\}|-?\d))?\s*$",
                           t[:start])                # ('\log_{10}', '\tan^{-1}')
            if fm:                                   # '\log(V_o)/(V_i)': the log of the ratio
                narrow.append((start, ends[0], f"\\left(\\frac{{{inner(num)}}}{{{inner(f1)}}}\\right)",
                               f"{fm.group(0).strip()}({num}/{f1})"))

    def apply(parts):
        s, labels = t, []
        for start, end, rep, label in sorted(parts, key=lambda x: -x[0]):
            s = s[:start] + rep + s[end:]
            labels.append(label)
        return s, ", ".join(reversed(labels))
    out = [apply([x]) for x in narrow + wide]
    if len(narrow) >= 2:
        out.append(apply(narrow))
    if len(wide) >= 2:
        out.append(apply(wide))
    out += [apply([x]) for x in sums]
    return list(dict.fromkeys(out))[:8]


def _close_denominators(side: str) -> str:
    """'R_2/(R_1 + R_2) V_{in}', '(R)/(L) s', '1/(2\\pi) \\sqrt{k/m}': a denominator its bracket closes - what follows it
    multiplies the fraction (the parser put it under the slash too, as it must in '1/2\\pi f C': a false sure wrong) - not
    a bracket after it ('1/(s+1)(s+2)' is 1 over the product, as written in a transfer function), nor a function's
    argument ('\\log(V_o)/(V_i)': the log over V_i, or of the ratio? _slash_alternatives); in braces as it was"""
    if "/" not in side or "(" not in side:
        return side
    t = re.sub(r"\\left\s*([(\[])", r"\1", side)
    t = re.sub(r"\\right\s*([)\]])", r"\1", t)
    for _ in range(6):
        depth, hit = 0, None
        for p, ch in enumerate(t):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "/" and depth == 0:
                m = _SL_NUM.search(t[:p])
                q = p + 1
                while q < len(t) and t[q] == " ":
                    q += 1
                if not m or q >= len(t) or t[q] != "(":
                    continue
                if m.group(0).startswith("(") and re.search(r"(?:\\[A-Za-z]+(?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]))?|[\^_])\s*$", t[:m.start()]):
                    continue                         # (a function's argument, an exponent's)
                e1 = _sl_factor_end(t, q)
                g1 = _sl_group_end(t, q)
                k = e1
                while 0 <= k < len(t) and t[k] == " ":
                    k += 1
                if e1 < 0 or g1 < 0 or k >= len(t) or t[k] in "([" or _sl_factor_end(t, e1) < 0:
                    continue
                num = m.group(0).strip()
                if num.startswith("(") and _sl_group_end(num, 0) == len(num):
                    num = num[1:-1]
                den = t[q + 1:g1 - 1] if g1 == e1 else t[q:e1]      # (with a power: '(R_1 + R_2)^2' whole)
                hit = (m.start(), e1, f"\\frac{{{num}}}{{{den}}}")
                break
        if hit is None:
            break
        t = t[:hit[0]] + hit[2] + t[hit[1]:]
    return t


def _exp_slack(e, *datas) -> float:
    """a relative slack for a value worked out through an exponential of a large argument ('10^{-14} e^{0.7/0.025}',
    'e^{-1.12/0.0518}'): the argument kept to 3 digits on the way (21.6 for 21.62) moves the value by up to e^{0.05} - 1,
    5 % - beyond a plain rounding's 1 % (a wrong factor or sign is still far off); 0 when there is none. datas: the
    data to put in (before: sympy works an exponential of a decimal out at once, and it is no exponential then)"""
    if not isinstance(e, sp.Basic) or not e.has(sp.exp):
        return 0.0
    out = 0.0
    for f in e.atoms(sp.exp):
        sizes = []
        for d in datas or ({},):
            try:
                sizes.append(abs(complex(sp.N(_exact(f.args[0].subs(d) if d else f.args[0])))))
            except Exception:                      # noqa: BLE001 - (letters in it)
                pass
        a = max((x for x in sizes if math.isfinite(x)), default=0.0)
        if a >= 10:
            out += math.expm1(0.5 * 10 ** (math.floor(math.log10(a)) - 2))
    return min(out, 0.1)


def _round_slack(text, data=None, prec=None) -> float:
    """a relative slack for a side worked out of rounded decimals: how much its value moves when each decimal written in
    it moves by half a unit in its last digit, summed ('(13.2 mA)^2 \\cdot 1.13 k\\Omega' is 0.1969 - '= 199 mW', worked
    out with I = 13.27 mA unrounded, is 1.07 % off: within the 1.2 % its inputs' rounding allows, no wrong step) - of
    decimals of three digits and more (a value worked out and rounded; '3.3', '2.2', '4.7' are components' values,
    exact); at most 2 %; 0 when none"""
    if not isinstance(text, str):
        return 0.0
    nums = [m for m in re.finditer(r"(?<![\d.])\d+\.(\d+)(?![\d.])", text)
            if len(m.group(0).replace(".", "").lstrip("0")) >= 3]
    keys = {str(k).lower(): k for k in list(data or {}) + list(prec or {}) if isinstance(k, sp.Symbol)}

    def parse(t):                                    # (the data's letters case-folded as the statements' are: R_{eq}, r_{eq})
        e = _parse(t)
        return e.xreplace({x: keys[str(x).lower()] for x in e.free_symbols if str(x).lower() in keys}) if e is not None else None
    base = parse(text)
    lets = [x for x in (base.free_symbols if base is not None else ()) if x in (prec or {}) and x in (data or {})]
    if not (nums or lets) or len(nums) > 6 or base is None:
        return 0.0
    v0 = _num(base.subs(data or {}))
    if not v0:
        return 0.0
    total = 0.0
    for m in nums:
        t2 = text[:m.start()] + repr(float(m.group(0)) + 0.5 * 10 ** -len(m.group(1))) + text[m.end():]
        e2 = parse(t2)
        v2 = _num(e2.subs(data or {})) if e2 is not None else None
        if v2 is None:
            return 0.0
        total += abs(v2 - v0) / abs(v0)
    for x in lets:                                   # (a value written rounded above, used by its name: 'P = I^2 R_{eq} =
        d2 = dict(data)                              # 305 mW' under 'I = 15.7 mA', 'R_{eq} = 1.25 k\Omega')
        try:
            d2[x] = d2[x] * (1 + prec[x])
        except TypeError:
            return 0.0
        v2 = _num(base.subs(d2))
        if v2 is None:
            return 0.0
        total += abs(v2 - v0) / abs(v0)
    return min(total, 0.02)


def _rel_half_unit(text, value=None) -> float:
    """half a unit in the last digit of a value written rounded, relative to it ('15.7 mA', '(15.7) \\cdot 10^{-3}':
    0.0032) - for a decimal of three digits and more (components' values, '3.3', '2.2', are exact); 0 otherwise"""
    t = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", str(text or ""))
    nums = re.findall(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", t)
    if len(nums) != 1:                               # (one number: a value, not a computation)
        return 0.0
    whole, dec = nums[0]
    if len((whole + dec).lstrip("0")) < 3 or float(f"{whole}.{dec}") == 0:
        return 0.0
    return 0.5 * 10 ** -len(dec) / float(f"{whole}.{dec}")


def _num(e) -> float | None:
    try:
        v = complex(sp.N(_exact(e)))
    except Exception:                              # noqa: BLE001 - (a matrix power: sympy recursed without end)
        return None
    if not (math.isfinite(v.real) and math.isfinite(v.imag)):
        return None
    return v.real if abs(v.imag) < 1e-9 else None


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_UNSUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_UNSUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _pretty(tex: str) -> str:
    """for messages: 'a ^ { 2 } + b ^ { 2 }' -> 'a²+b²', '\\sqrt { 2 0 }' -> '√20', '7 8 . 5 4' -> '78.54'"""
    t = re.sub(r"\^\s*\{\s*(-?\s*\d+)\s*\}|\^\s*(-?\s*\d+)", lambda m: re.sub(r"\s", "", m.group(1) or m.group(2)).translate(_SUP),
               str(tex).replace("**", "^"))                  # '2^10' -> '2¹⁰' (it showed '2¹0'), before the spaces go
    t = re.sub(r"\\(" + "|".join(sorted(_GREEK, key=len, reverse=True)) + r")(?![A-Za-z])", lambda m: _GREEK[m.group(1)], t)   # 'ω_n', 'ζ'
    t = re.sub(r"\\(infty|to|rightarrow|pm|int|sum|lim|log|ln|sin|cos|tan|exp|arcsin|arccos|arctan|sinh|cosh|tanh|cot|sec|csc|arg|gcd|lcm|bmod|mod|max|min)(?![A-Za-z])",
               lambda m: {"infty": "∞", "to": "→", "rightarrow": "→", "pm": "±", "int": "∫", "sum": "Σ"}.get(m.group(1), m.group(1)), t)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"\\begin\{([pbBvV]?matrix|smallmatrix)\}(.*?)\\end\{\1\}", lambda m: ("det[" if m.group(1) == "vmatrix" else
               "·[" if re.search(r"[\d})]$", m.string[:m.start()]) else "[")    # '[1, 2; 3, 4]', '1/-2·[4, -2; -3, 1]'
               + "; ".join(", ".join(r.split("&")) for r in re.split(r"\\\\", m.group(2)) if r) + "]", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√" + (m.group(1) if re.fullmatch(r"[\w.]+|\\[A-Za-z]+", m.group(1))
                                                     else f"({m.group(1)})"), t)   # '√(316²-50²)', '√20'
    t = t.replace("sqrt(", "√(").replace("\\det", "det")
    t = re.sub(r"\\[dt]?binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1,\2)", t)            # '\binom{5}{2}' -> 'C(5,2)'
    t = re.sub(r"\\[lc]dots(?![A-Za-z])", "…", t).replace("\ue002", ",")
    t = t.replace("\\pi", "π").replace("\\cdot", "·").replace("\\times", "×").replace("\\approx", "≈")
    t = re.sub(r"·0\.010(?!\d)", "%", t)                  # a percentage, as statements() read it
    t = re.sub(r"\\[lr]vert", "|", t)
    t = re.sub(r"\\(?:left|right)(?=\\[{}]|[()\[\]|.])", "", t).replace("\\{", "\ue010").replace("\\}", "\ue011")
    t = re.sub(r"\\(?:mathcal|mathscr)\{?L\}?", "L", t)
    t = re.sub(r"\\(?:mathrm|text|rm|operatorname)\{([A-Za-z]+)\}", r"\1", t)      # 'pf', 'BW' (one name: _ee_words)
    t = re.sub(r"\^\{([+-]\d+(?:\.\d+)?)\}", r"^\1", re.sub(r"_\{(\w+(?:,\w+)*)\}", r"_\1", t))     # 10^{-12}, V_{in}, C_{g,min}
    t = re.sub(r"\^\{([^{}]+)\}", lambda m: "^" + (m.group(1) if len(m.group(1)) == 1 else f"({m.group(1)})"), t)  # 'e^(-2s)'
    atom = re.compile(r"[+-]?\d+(?:\.\d+)?|[A-Za-zα-ωΑ-Ωπ](?:_\w+)?|d[A-Za-z]|\\[A-Za-z]+|\([^()]*\)")

    def frac(m):                                   # (a number after it: '2/3·12', not '2/312')
        return "/".join(x if atom.fullmatch(x) else f"({x})" for x in (m.group(1), m.group(2))) + \
            ("·" if m.string[m.end():m.end() + 1].isdigit() else "")
    t = re.sub(r"(?<=\d)(?=\\[dt]?frac)", "\u2009", t)     # '2\frac{1}{2}' -> '2 1/2', not '21/2'
    while True:                                    # '\frac{10}{3}' -> '10/3', '\frac{1}{2π}' -> '1/(2π)' (it was '\frac102π')
        t2 = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", frac, t)
        t2 = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√" + (m.group(1) if re.fullmatch(r"[\w.]+|\\[A-Za-z]+", m.group(1))
                                                          else f"({m.group(1)})"), t2)    # ('\sqrt{\frac{2}{50}}' -> '√(2/50)')
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\\(?:vert|mid)", "|", t).replace("{", "").replace("}", "")
    t = re.sub(r"(d/d[A-Za-z])(?=[A-Za-z(\\])", r"\1 ", t).replace("\\angle", "∠").replace("^\\circ", "°")
    if "Q" in t or "q" in t:                       # a value at an instant as written ('v_C(0⁺)': statements() made it one name)
        t = _INSTANT_SHOWN.sub(lambda m: ("_" + m.group(1) if m.group(1) else "") + {"p": "(0⁺)", "m": "(0⁻)", "i": "(∞)", "z": "(0)"}[m.group(2).lower()], t)
    return t.replace("\ue010", "{").replace("\ue011", "}")


def _fmt(v) -> str:
    if isinstance(v, str):                         # a complex step's value, already written out
        return v.lstrip(_UNSURE + _MAYBE)
    if isinstance(v, complex):
        return _fmt_c(v)
    if abs(v) < 1e-100:                            # (an exact zero's evaluation noise: '1.8e-189')
        return "0"
    return f"{v:.4g}" if abs(v - round(v)) > 1e-9 * max(1.0, abs(v)) or abs(v) < 0.5 or abs(v) >= 1e7 else str(int(round(v)))
    #                                                  ('3e-11', not '0'; '2e+21', not 2000000000000000000000)


def _letters(e) -> set:
    return set(e.free_symbols)


def _as_data(v):
    """a value to put into a formula: an integer exactly (so roots stay exact), a complex number as sympy's"""
    if isinstance(v, complex):
        return sp.Float(v.real) + sp.I * sp.Float(v.imag) if abs(v.imag) > 1e-12 else _as_data(v.real)
    return sp.nsimplify(v, rational=True) if float(v).is_integer() else v


def _fmt_num(v) -> str:
    """a value of the data (sympy's or Python's, real or complex) for a message"""
    c = complex(v)
    return _fmt_c(c) if abs(c.imag) > 1e-12 else _fmt(c.real)


def _value(e):
    """a side's value: a real number, else a complex one (an impedance, a phasor), else None"""
    v = _num(e)
    return v if v is not None else _cnum(e)


def _positive(sols):
    """an equation's positive roots (complex ones are no such roots)"""
    return [x for x in sols if not isinstance(x, complex) and x > 0]


@dataclass
class Statement:
    kind: str                  # 'formula' | 'value' | 'chain' | 'numeric'
    text: str
    sides: list
    conf: float
    symbol: object = None      # value: the letter
    number: float | None = None
    value_text: str = ""       # value: the number as written (its decimals set the tolerance)
    unit: str = ""
    line: int = -1             # the page line it was read on
    piece: int = -1            # index of the page piece it came from
    side_texts: list = field(default_factory=list)     # the sides as written (for messages: '4²+2²', not '2²+4²')
    calc: dict = field(default_factory=dict)           # identity: {'indefinite': [variables], 'may_be_wrong': bool}


@dataclass
class Check:
    problem: int
    verdict: str               # 'right' | 'wrong' | 'wrong step' | 'right value, wrong equality' | 'cannot check' | 'misread?'
    text: str                  # one line for a human
    confidence: float          # probability of this verdict over the readings, times the layout confidence
    sure: bool                 # confident enough to act on (else: a suggestion for the human who checks)
    pieces: list = field(default_factory=list)
    expected: list = field(default_factory=list)   # values the formula gives for the answered letter
    answer: str = ""           # the student's answer as read
    steps: list = field(default_factory=list)      # [(step text, True/False)]
    line: int = -1             # the page line it is about (the last of a problem's), for marking the page


def _decimal_commas(tex: str) -> str:
    """'7,5' (also spaced, '7 , 5') a decimal comma - not two statements; '1 , 000' thousands, joined ('1000'); a pair
    in parentheses ('(3, 4)': a point) left as it is"""
    t = re.sub(r"\(\s*(-?\s*\d[\d\s.]*)\s*,\s*(-?\s*\d[\d\s.]*)\)", lambda m: f"({m.group(1)}\ue000{m.group(2)})", tex)
    t = re.sub(r"_\s*\{\s*(\d)\s*,\s*(\d)\s*\}", "_{\\1\ue000\\2}", t)       # 'x_{1,2}': two roots' indices
    t = re.sub(r"(?<=\d)\s*,\s*(?=\d(?:\s*\d)?(?!\s*\d))", ".", t)
    # thousands: the digits joined ('10,000' was kept - the side with it then unread: '\frac{12}{10,000} = 1.2 mA' had
    # no verdict, a value alone was read)
    t = re.sub(r"(?<=\d)\s*,\s*(\d)\s*(\d)\s*(\d)(?!\s*\d)", r"\1\2\3", t)
    return t.replace("\ue000", ",")


_REPEATING = re.compile(r"(?<![\d.])(\d+)\s*\.\s*(\d*)(?:\s|~|\\[,;:! ])*\\(?:overline|bar)\s*\{\s*\{?\s*(\d+)\s*\}?\s*\}")
# a unit in square brackets, as Israeli exams write it: '3.3 [V]', '2 \\left[ mA \\right]', '[k\\Omega]', '[m/s]'
_BRACKETED = re.compile(r"(?:\\left\s*)?\[\s*(?:\\(?:mathrm|text|rm)\s*\{\s*)?(?P<u>(?:\\mu\s*|[pnuµmkKM])?\s*"
                        r"(?:V\s*A\s*R|V\s*A|var|V|A|W|F|Hz|H|S|J|C|N|Pa|kg|g|sec|s|\\Omega|Ω|ohm|rad\s*/\s*s|d\s*B|m\s*/\s*s|m|°|deg))"
                        r"\s*\}?\s*(?:\\right\s*)?\]")


def repeating(m) -> str:
    """'3.\\overline{3}' -> '3.333333333333': a repeating decimal, to twelve places"""
    whole, fixed, rep = m.group(1), m.group(2), m.group(3)
    return f"{whole}." + (fixed + rep * 12)[:12]


_INLINE_UNIT = re.compile(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*(?P<rm>\\mathrm\s*\{\s*)?(\\mu\s*|[fpnuµmkMG])?\s*"
                          r"(VAR|VAr|VA|var|V|A|W|\\Omega|Ω|F|H|Hz|J)(?![A-Za-z_'])(?(rm)\s*\})"
                          r"(?:\s*\^\s*(?:\{\s*(?P<pw>[23])\s*\}|(?P<pw2>[23])(?![\d.])))?")
# (a unit squared is the unit's power: '\frac{144 V^2}{4 \Omega} = 36 W', '4 A^2 \cdot 2 \Omega = 8 W' - the power was left on
# the number: 144² over 4, a false sure wrong step)
# (not a letter with a subscript or a prime: '\frac{2 V_m}{\pi}' is 2 times V_m, not 2 volts and a stray '_m'; volt-amperes
# too: '\frac{800 W}{1 kVA}', '1 kVA \cdot 0.6 = 600 VAR' - they were letters of the problem, no verdict)


# a unit after a bracket of numbers ('12 / (2 + 2) k\Omega', '10 / (3 + j4) k\Omega', '2 mA \cdot (1 + 2) k\Omega'): the
# bracket's - it was letters of the problem (no verdict)
_PAREN_UNIT = re.compile(r"\)\s*(?P<rm>\\mathrm\s*\{\s*)?(?P<p>\\mu\s*|[pnuµmkMG])?\s*(?P<u>VAR|VA|var|V|A|W|\\Omega|Ω|F|H|Hz|J)(?![A-Za-z_'])"
                         r"(?(rm)\s*\})")


# a unit after a phasor ('5 \\angle 53.13^\\circ k\\Omega', '2 \\angle 30^\\circ mA'): the phasor's - it was letters (no verdict)
_POLAR_UNIT = re.compile(r"(?<![\w.\\])(?P<ph>\d+(?:\.\d+)?\s*\\angle\s*\{?\s*[+-]?\s*\d+(?:\.\d+)?\s*(?:\^\s*\{?\s*\\circ\s*\}?|°)?\s*\}?)"
                         r"\s*(?P<rm>\\mathrm\s*\{\s*)?(?P<p>\\mu\s*|[pnuµmkMG])?\s*(?P<u>V|A|W|\\Omega|Ω|F|H|VA|VAR|var)(?![A-Za-z_'])"
                         r"(?(rm)\s*\})")


# a complex value's unit, the value unbracketed ('Z = 3 + j4 \\Omega', '\\frac{12 + j5 V}{4 + j4 k\\Omega}', '= 3 \\Omega - j4
# \\Omega', '\\frac{10 V}{-j4 \\Omega}'): the whole value's - it was the j-part's alone, a letter of the problem (no verdict)
_CPLX_U = r"(?:\\mu\s*|[pnuµmkMG])?\s*(?:VAR|VA|var|V|A|W|\\Omega|Ω|F|H|Hz|J)(?![A-Za-z_'])"
_CPLX_UNIT = r"(?:\\mathrm\s*\{\s*" + _CPLX_U + r"\s*\}|" + _CPLX_U + r")"
_RECT_UNIT = re.compile(r"(?P<l>(?:^|[={(,;])\s*)(?P<z>[-+]?\s*\d+(?:\.\d+)?\s*[-+]\s*(?:j\s*(?:\\cdot\s*)?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:\\cdot\s*)?j))\s*"
                        r"(?P<u>" + _CPLX_UNIT + r")(?=\s*(?:$|[=},;)]|\\right(?![A-Za-z])|\\q?quad|\\\\))")
# (a side of its own, or a fraction's part, or a bracket's: '2 + 3 + j4 \\Omega' stays - whose unit is it?)
_IMAG_UNIT = re.compile(r"(?P<s>[-+]\s*|(?<![\w\\.]))j\s*(?:\\cdot\s*)?(?P<b>\d+(?:\.\d+)?)\s*(?P<u>" + _CPLX_UNIT + r")"
                        r"(?=\s*(?:$|[-+=},;)]|\\right(?![A-Za-z])|\\q?quad|\\\\|\\cdot(?![A-Za-z])|\\times(?![A-Za-z])))")
_CPLX_END = re.compile(r"(?:[={(,;]|\\left\s*\(|(?:\d|\))\s*" + _CPLX_UNIT + r"(?:\s*(?:\\cdot|\\times|/)(?![A-Za-z]))?)\s*$")
# (after a start, a bracket, or a term with its unit - a sum's or a product's: '3 \Omega - j4 \Omega', '2 A \cdot j4 \Omega')


def _complex_units(text: str) -> str:
    """'3 + j4 \\Omega' -> '(3 + j4) \\Omega', '3 \\Omega - j4 \\Omega' -> '3 \\Omega - (j4) \\Omega': a complex value's unit is
    the whole value's - as a unit after a bracket (_paren_units); a j-term's own only after a start or a term with its
    unit ('2 + 3 - j4 k\\Omega' stays: kilo of the j-part alone, or of all three?)"""
    def last(m, before):                           # a last side: its unit was read before ('= 1.2 - j1.6 mA', '= -j4
        return before.rstrip().endswith("=") and \
            re.match(r"\s*(?:$|[,;]|\\q?quad|\\\\)", m.string[m.end():])     # k\Omega' - in the student's units)

    text = _RECT_UNIT.sub(lambda m: m.group(0) if last(m, m.group("l")) else
                          f"{m.group('l')}({re.sub(r'\s+', ' ', m.group('z').strip())}) {m.group('u')}", text)

    def rep(m):
        before = text[:m.start()].rstrip()
        if before and not _CPLX_END.search(before) or last(m, before):
            return m.group(0)
        return f"{m.group('s')}(j{m.group('b')}) {m.group('u')}"
    return _IMAG_UNIT.sub(rep, text)


def _polar_units(text: str) -> str:
    """'10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega' -> '10 \\angle 0^\\circ / (5 \\angle 53.13^\\circ \\cdot 10^{3})'"""
    exp = {"f": -15, "p": -12, "n": -9, "u": -6, "µ": -6, "\\mu": -6, "m": -3, "k": 3, "M": 6, "G": 9}

    def rep(m):
        pre = re.sub(r"\s+", "", m.group("p") or "")
        return f"({m.group('ph').strip()} \\cdot 10^{{{exp[pre]}}})" if pre else m.group("ph").strip()
    return _POLAR_UNIT.sub(rep, text)


def _paren_units(text: str) -> str:
    """'12 / (2 + 2) k\\Omega' -> '12 / ((2 + 2) \\cdot 10^{3})': a unit after a bracket of numbers (j too) is the bracket's"""
    exp = {"f": -15, "p": -12, "n": -9, "u": -6, "µ": -6, "μ": -6, "\\mu": -6, "m": -3, "k": 3, "M": 6, "G": 9}
    while True:
        for m in _PAREN_UNIT.finditer(text):
            depth, k = 0, m.start()
            while k >= 0:
                depth += {")": 1, "(": -1}.get(text[k], 0)
                if depth == 0:
                    break
                k -= 1
            inner = text[k + 1:m.start()] if k >= 0 else ""
            if k < 0 or re.search(r"[A-Za-ik-z]", re.sub(r"\\(?:cdot|times|frac|sqrt|left|right)", "", inner)):
                continue                             # (letters in it: '(V_1 - V_2) A' - a product)
            lft = re.search(r"\\left\s*$", text[:k])          # ('\left( 3 + j4 \right) k\Omega', as the reader writes it)
            k = lft.start() if lft else k
            pre = re.sub(r"\s+", "", m.group("p") or "")
            group = text[k:m.start() + 1]
            text = text[:k] + (f"({group} \\cdot 10^{{{exp[pre]}}})" if pre else group) + " " + text[m.end():]
            break
        else:
            return text


def _si_inline(text: str, bare: bool = True, milli: bool = False, nano: bool = False) -> str:
    """'2 mA \\cdot 5 k\\Omega' -> '(2 \\cdot 10^{-3}) \\cdot (5 \\cdot 10^{3})': every number with its unit, in SI units
    (bare: a bare k, M, G, \\mu too - '\\frac{5}{1k}'; milli: a bare m, n, p - see _bare_m_ok; nano: a bare n, p alone)"""
    exp = {"f": -15, "p": -12, "n": -9, "u": -6, "µ": -6, "μ": -6, "\\mu": -6, "m": -3, "k": 3, "M": 6, "G": 9}

    def rep(m):
        pre = re.sub(r"\s+", "", m.group(3) or "")
        if pre and re.search(r"\d\s*[-+]\s*j\s*(?:\\cdot\s*)?$", m.string[:m.start()]):
            return m.group(0)                        # ('2 \cdot 6 + j 8 k\Omega': kilo of the j-part alone, or of the
        #                                              whole value? - left unread, no verdict: it was j8000 - a false wrong)
        pw = int(m.group("pw") or m.group("pw2") or 1)
        return f"({m.group(1)} \\cdot 10^{{{exp[pre] * pw}}})" if pre else m.group(1)
    text = _paren_units(_INLINE_UNIT.sub(rep, _polar_units(text)))
    text = _INLINE_MORE.sub(_more, text)
    if milli or nano:
        text = _BARE_M.sub(lambda m: f"({m.group(1)} \\cdot 10^{{{exp[m.group(2)]}}})" if milli or m.group(2) != "m" else m.group(0), text)
    return _BARE_INLINE.sub(lambda m: f"({m.group(1)} \\cdot 10^{{{exp[m.group(2)]}}})", text) if bare else text


# more units a substitution carries - only those no letter of a formula is spelt like ('2 kg \cdot 3 \frac{m}{s^2}', '20 kPa',
# '5 cm', '2 ms', '50 \mu C'): a lone s may be Laplace's, g the gravity, N a force, C a capacitance, m a mass
_INLINE_MORE = re.compile(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*(?P<rm>\\mathrm\s*\{\s*)?(?P<u>"
                          r"\\frac\s*\{\s*[A-Za-z\\]+\s*\}\s*\{\s*[A-Za-z]+\s*(?:\^\s*\{?\s*\d\s*\}?)?\s*\}|"
                          r"(?:k?m|cm|rad)\s*/\s*(?:s|h|min)(?:\s*\^\s*\{?\s*2\s*\}?)?|"
                          r"kg|mg|kN|mN|MPa|kPa|Pa|cm|mm|km|ms|\\mu\s*s|µs|ns|ps|fs|\\mu\s*C|µC|nC|mC|pC|mS|\\mu\s*S|µS|uS|kS)(?![A-Za-z_']|\s*/\s*[A-Za-z\\])"
                          r"(?(rm)\s*\})")
# (not a unit over a unit, 'ms/V'; a division by a number is no such: 'e^{-1 ms/1 ms}')


# a number with a unit no letter of a formula is spelt like: ohms, or a unit with its prefix ('2 k\\Omega', '5 mA', '10 \\mu F')
_UNAMBIGUOUS = re.compile(r"(?<![A-Za-z_\\\d.])\d+(?:\.\d+)?\s*(?:(?:\\mu\s*|[pnuµmkMG])?\s*(?:\\Omega|Ω)|(?:\\mu\s*|[pnuµmkMG])\s*(?:V|A|W|F|H|Hz))"
                          r"(?![A-Za-z_'])")


def _no_exps(text: str) -> str:
    """a side's text without its exponentials' exponents ('2(1 - e^{-2 ms / 1 ms})' -> '2(1 - e)'): units there divide
    out"""
    out, i = [], 0
    while i < len(text):
        m = re.compile(r"e\s*\^\s*|\\exp\s*(?:\\left\s*)?").match(text, i)
        if m and (i == 0 or not text[i - 1].isalpha() or text[i] == "\\"):
            j = m.end()
            if j < len(text) and text[j] in "{(":
                close, depth = {"{": "}", "(": ")"}[text[j]], 0
                for k in range(j, len(text)):
                    depth += {"{": 1, "(": 1, "}": -1, ")": -1}.get(text[k], 0)
                    if depth == 0 and text[k] in "})":
                        break
                out.append("e")
                i = k + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _more(m):
    f = _si(m.group("u"))
    if f is None:
        return m.group(0)
    return m.group(1) if f == 1 else f"({m.group(1)} \\cdot {_factor_tex(f)})"


def _bare_side(part: str, side: str) -> bool:
    """a value with a bare prefix in a substitution, as EE students write ('\\frac{5}{1k}', '2k \\| 2k', '10k + 5k') - not
    when the letter is the problem's own (a spring's k: 'F = kx = 0.1k')"""
    side = _join_digits(side)
    if _BARE_M.search(side) and _bare_m_ok(part):
        return True
    if not _BARE_INLINE.search(side):
        return False
    rest = re.sub(r"(\d)\s*(?:[kMG]|\\mu|µ|μ)(?![A-Za-z])", r"\1", _join_digits(part))
    rest = re.sub(r"\\(?:mathrm|text|rm)\s*\{[^{}]*\}", " ", rest)          # (a unit's '\mathrm{k\Omega}': no letter k)
    if re.search(r"(?<![\\A-Za-z\d.])(?:[kMG]|µ|μ)(?![A-Za-z])|\\mu(?![A-Za-z])", rest):
        return False                                 # (a friction's or a mobility's \mu: 'F = \mu N = 0.3 \cdot 10')
    return bool(re.search(r"\\frac|\\\||\\parallel|/", side)) or len(_BARE_INLINE.findall(side)) >= 2 \
        or _prefixed_result(part) or bool(_INLINE_UNIT.search(side))   # ('12 / 2k'; '2 \cdot 4k = 8 kHz': the result's prefix
    #                                                  says it; '(12 \mu A)(2.2M)': units written beside it)


def _last_unit(part: str) -> str:
    m = _UNIT.match(_join_digits(_split_eq(part)[-1]))
    return re.sub(r"\s+", "", m.group("unit")) if m and not _has_letters(m.group("num")) else ""


def _prefixed_result(part: str) -> bool:
    """the statement's result in a unit with a prefix ('= 8 kHz', '= 2 mA'): its values in prefixes too"""
    u = _last_unit(part)
    return bool(re.match(r"(?:[kMGunpµm]|\\mu)[A-Za-z\\Ωμ]", u)) and _si(u) not in (None, 1.0)


# a bare m, n or p after a number: milli, nano, pico, as EE students write values ('-80m \cdot 2k', '\frac{100}{80 m} =
# 1.25 k\Omega', '10k \cdot 100n')
_BARE_M = re.compile(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*([mnpf])(?!\s*[A-Za-z_'])(?!(?<=m)\s*/\s*\\?[A-Za-z])(?!\s*\(\s*[A-Za-z])")
# (not 'm(t)'; an m before '/' and a letter may be metres per - 'm/s' - an n or p not: '\sqrt{250n/100p}'; before a number
# milli: '\tau = 10m/10 = 1 ms'; a bare f femto, as a CMOS course writes a capacitance: '0.69 \cdot 2f \cdot 5k = 6.9 ps')
_ELECTRIC = re.compile(r"(?:[kMGunpµm]|\\mu)?(?:\\Omega|Ω|ohm|VAR|VA|var|V|A|W|F|H|S|Hz|s|C|J)")   # (a charge, an energy too)


def _bare_m_ok(part: str) -> bool:
    """a bare m in a substitution is milli when another value has a bare prefix ('(2m)^2 \\cdot 1k') or the result is in an
    electrical unit ('\\frac{100}{80 m} = 1.25 k\\Omega') - and no letter m in the statement (a mass: 'F = m a'). Never
    the last side ('\\lambda = 3 m': metres, a unit)"""
    p = _join_digits(part)
    rest = re.sub(r"\\[A-Za-z]+|_\s*(?:\{[^{}]*\}|[A-Za-z0-9])", " ", _BARE_M.sub(" ", p))     # (g_m: a name, not m)
    if re.search(r"(?<![A-Za-z])[mnp](?![A-Za-z])", rest) or (
            re.search(r"(?<![A-Za-z_\\\d.])\d+(?:\.\d+)?\s*f(?![A-Za-z])", p) and re.search(r"(?<![A-Za-z\\])f(?![A-Za-z])", rest)):
        return False                                 # (a bare f with a frequency f in the statement: 'f = 2f')
    return bool(_BARE_INLINE.search(p)) or bool(_ELECTRIC.fullmatch(_last_unit(part)))


_BARE_INLINE = re.compile(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*([kMG]|\\mu|µ|μ|u(?!\s*\())(?!\s*[A-Za-z_'])")   # '\frac{5}{1k}': 1
#                                                     kilo-ohm, '100 \mu': 100 micro-farads, as EE students write a value
#                                                     (not before a name: '\sqrt{2 k I_D}' - k a letter, a MOSFET's); a
#                                                     typed u ('1k*1u') - not a step's 'u(t)'



_PM = re.compile(r"\\(?:pm|mp)(?![A-Za-z])|±|∓")
_PM_NAME = re.compile(r"^\s*([A-Za-z](?:\\[A-Za-z]+)?|\\[A-Za-z]+)\s*_\s*\{\s*1\s*,\s*2\s*\}")


def _pm_variants(part: str):
    """'x_{1,2} = \\frac{5 \\pm 1}{2}' -> x_1 with +, x_2 with -; 'x = \\pm 3' -> x = 3 and x = -3; [(part, sign)].
    Any other '\\pm' (a tolerance, '100 \\pm 5') is left alone"""
    if not _PM.search(part):
        return [(part, 0)]
    sides = _split_eq(part)
    named = _PM_NAME.match(sides[0])
    lone = len(sides) == 2 and re.fullmatch(r"\s*(?:\\pm|±)\s*\d+(?:\.\d+)?\s*", sides[1])
    if not named and not lone and len(sides) == 2 and _PM.search(sides[1]) and \
            re.fullmatch(r"\s*(?:[A-Za-z]|\\[A-Za-z]+)\s*", sides[0]):
        # 's = -1 \pm j3' after 's^2 + 2s + 10 = 0': the two roots, named apart as 's_{1,2} = ...' would be (no verdict before)
        part = sides[0].rstrip() + "_{1,2} =" + sides[1]
        sides = _split_eq(part)
        named = _PM_NAME.match(sides[0])
    if not named and not lone:
        return [(part, 0)]
    if lone and not re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+", "", sides[0])):
        # '\sqrt{16} = \pm 4': the roots of a number written loosely - the root itself is checked, not a false '-4'
        return [(_PM.sub(lambda m: "+" if m.group() in ("\\pm", "±") else "-", part), 0)]
    out = []
    for sign, k in ((1, 1), (-1, 2)):
        t = _PM.sub(lambda m: "+" if (sign > 0) == (m.group() in ("\\pm", "±")) else "-", part)
        if named:
            t = _PM_NAME.sub(lambda m: f"{m.group(1)}_{{{k}}}", t, count=1)
        out.append((t, sign))
    return out


# a sequence by its index ('x[n]', 'h[n - 1]', 'y[2]', '\\delta[n]'): as a function's value - 'h[n] = 0.5^n u[n]', then 'h[2] =
# 0.25'; not a unit in brackets ('5 [V]'), nor a bracket after anything else
_SEQ = re.compile(r"(?<![A-Za-z\\])([A-Za-z]|\\delta|\\Delta)(\s*_\s*\{[^{}]*\}|_[A-Za-z0-9])?\s*(?:\\left\s*)?\[\s*"
                  r"([nk](?:\s*[-+]\s*\d+)?|[-+]?\s*\d+)\s*(?:\\right\s*)?\]")


# a condition in brackets at the end of a statement ('i(t) = 2e^{-t} (t > 0)', '(0 < t < 1)'): the function's domain - the
# side was unreadable with it (a relation in brackets), and the function lost; not after a name ('P(X > 2)': a probability)
_DOMAIN = re.compile(r"([^A-Za-z\s\\])\s*(?:\\left\s*)?\(\s*(?:[-+]?\d+(?:\.\d+)?\s*(?:<|\\leq?|≤|\\lt)\s*)?[A-Za-z]\s*"
                     r"(?:>|<|\\geq?|\\leq?|≥|≤|\\gt|\\lt)\s*[-+]?\s*\d+(?:\.\d+)?\s*(?:\\right\s*)?\)\s*$")


_R_UNIT = r"(?:[kMGmunpµ]|\\mu\s*)?(?:Hz|V|A|W|\\Omega|Ω|F|H|s|m)(?![A-Za-z])"
_RANGE = re.compile(r"(?P<pre>(?:^|=|:)\s*)(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>" + _R_UNIT + r")\s*(?:-|–|—|\\sim(?![A-Za-z])|∼)\s*"
                    r"(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>" + _R_UNIT + r")(?=\s*(?:$|,|;|\\q?quad|\\text|\\\\))")
# (a side of its own, last: '= 20 Hz - 20 kHz' - not '1 mA - 2 mA + 3 mA = 2 mA')


def _range(m):
    """'20 Hz - 20 kHz', '5 V \\sim 12 V': two values of one kind, the smaller first - a range, not a difference (it was a
    value: 'f = 20 Hz - 20 kHz' made f = -19980 Hz, every formula after it a false wrong) -> an ellipsis, a side no
    value is read from; a difference stays ('12 V - 0.7 V')"""
    fa, fb = _si(m.group("ua")), _si(m.group("ub"))
    da, db = _dimension(m.group("ua")), _dimension(m.group("ub"))
    if None in (fa, fb, da, db) or sp.simplify(da / db) != 1 or not float(m.group("a")) * fa < float(m.group("b")) * fb:
        return m.group(0)
    return m.group("pre") + " \\ldots "


# EE quantities named by several letters ('pf = 0.8', '\theta = \arccos(pf)', 'BW = \frac{f_0}{Q}', 'SNR = ...'): one name each
# - they were products of letters (p times f): 'no value for f, p'; not a picofarad's '100 pf'
_EE_WORD = re.compile(r"(?<![A-Za-z\\])(pf|PF|BW|SNR|CMRR|THD)(?![A-Za-z(])")


# a name with its subscript written in line ('Req', 'Rth', 'Vth', 'Vrms', 'Vcc', 'fmax', 'gm', 'Av'): the subscript's - R
# times e times q before ('Req = R_1 + R_2', then 'Req = 5k': no verdict); a known subscript only ('RC' stays a product)
_INLINE_SUB = re.compile(r"(?<![A-Za-z\\_])(?:([RVIZPCLXGYQESTAFfr])(eq|th|in|out|rms|avg|max|min|pp|dc|ac|tot|load|ref|cc|ee|bb|ce|be|"
                         r"cb|gs|ds|ov|sat|pk)|(g)(m)|(A)([vi])|(h)(fe|FE|ie|oe|re))(?![A-Za-z_])")


def _ee_words(tex: str) -> str:
    tex = _EE_WORD.sub(lambda m: m.group(0) if re.search(r"[\d.]\s*$", tex[:m.start()]) else "\\mathrm{" + m.group(1) + "}", tex)

    def sub(m):
        if re.search(r"(?:\\(?:text|textrm|mathrm|rm|mbox|operatorname)\s*|[_^]\s*)\{\s*$", tex[:m.start()]):
            return m.group(0)                        # (words in \text{...}, a subscript's own letters: 'V_{Rth}')
        return next(a for a in m.groups()[0::2] if a) + "_{" + next(b for b in m.groups()[1::2] if b) + "}"
    return _INLINE_SUB.sub(sub, tex)


def _boolean_statements(tex: str, conf: float, line: int, piece: int, need_ops: bool = True):
    """a line in Boolean algebra -> its statements (kind 'boolean': the sides' logic expressions in calc['bool'], a name on
    the left - 'F = ...', 'F(A,B,C) = ...' - in calc['name']); None when some side is no such expression (a mean '\\bar{x}
    = 5.2', a statement of another kind), or no side has an operator ('\\bar{v} = 0')"""
    from . import boolean as bl
    out, ops = [], 0
    masked = re.sub(r"\([^()]*\)", lambda m: m.group(0).replace(",", "\ue001"), tex)    # ('F(A,B,C) = ...': one name)
    named = False
    for part in _top_level_split(masked):
        part = part.replace("\ue001", ",").strip()
        if "=" not in part:
            continue
        sides = [x.strip() for x in _split_eq(part)]
        if len(sides) < 2 or any(not x for x in sides):
            return None
        # a logic value with its strength, as a digital-circuits course writes it ('V_x = 1S', 'Y = 0W': strong, weak): the
        # value (it was 1 and S - a letter: 'not the same for S=0', a false sure wrong)
        parsed = [bl.parse(re.sub(r"(?<![A-Za-z0-9.])([01])\s*(?:S|W)(?![A-Za-z])", r"\1", t)) for t in sides]
        if any(p is None for p in parsed[1:]):
            return None
        rest = set().union(*(bl.variables(p[0]) for p in parsed[1:]))
        nm = bl.name_side(sides[0])
        name = None
        if nm and "(" in sides[0]:                   # 'F(A,B,C)', 'Y(P)': a function's name (not Y times P)
            name, parsed[0] = nm, None
        elif nm and (parsed[0] is None or (parsed[0][0][0] == "var" and (parsed[0][0][1] not in rest or len(sides) > 2))):
            # ('Y = (X + Y)' = X'Y'': a name still - a letter alone, then a simplification)
            name = parsed[0][0][1] if parsed[0] is not None else nm      # ('F = A\bar{B}': a name, not a letter of it)
            parsed[0] = None
        elif parsed[0] is None:
            return None
        ops += sum(p[1] for p in parsed if p is not None)
        named = named or nm is not None              # (a letter alone on the left: a name, or a letter defined)
        out.append(Statement("boolean", part, [], conf, line=line, piece=piece, side_texts=sides,
                             calc={"bool": [p[0] if p is not None else None for p in parsed], "name": name}))
    if not out:
        return None
    if not need_ops:                                 # (a page's Boolean reading of a plain line: kept on a Boolean page only)
        return out
    double = re.search(r"\\(?:overline|bar)\s*\{\s*\\(?:overline|bar)|''|"       # (or complements of single letters
                       r"\\(?:overline|bar)\s*\{?\s*[A-Za-z]\s*\}?\s*\\(?:overline|bar)\s*\{?\s*[A-Za-z]\s*\}?", tex)   # side by side)
    signs = re.search(r"\+|\\oplus|\\odot|⊕|⊙|\\vee|\\wedge|\\lor|\\land|∨|∧|\\cdot|·|\*|(?<![\d.])[01](?![\d.])", tex)
    if double or bl.primed(tex) or (named and (ops or signs)) or (ops and signs):
        return out                                   # (not letters side by side only, nameless: '\\overline{AB} = \\overline{CD}',
    return None                                      # two segments; nor '\\bar{v} = 0', a mean)


def _boolean_page(stmts: list) -> list:
    """a page with Boolean algebra on it: each statement that reads as Boolean algebra too ('F = A', 'F = AB + C') read so"""
    out, done = [], set()
    for s in stmts:
        if s.kind != "boolean" and s.text and "=" in s.text and _WEAK_BOOL.fullmatch(s.text):
            key = (s.line, s.piece, s.text)
            got = _boolean_statements(s.text, s.conf, s.line, s.piece, need_ops=False)
            if got:
                if key not in done:                  # ('F = AB + AC = A(B + C)': an identity and a chain - one line)
                    out += got
                    done.add(key)
                continue
        out.append(s)
    return out


def _boolean_names(stmts: list) -> list:
    """a letter alone on the left of Boolean lines again and again ('Y = \\overline{XY}', then 'Y = \\bar{X}\\bar{Y}'): the
    function's name, though a letter of it too (it was taken for an equation in Y - a false wrong)"""
    from dataclasses import replace
    from . import boolean as bl
    firsts = [s.side_texts[0].strip() for s in stmts if s.kind == "boolean" and s.calc.get("name") is None and s.side_texts]
    again = {t for t in firsts if firsts.count(t) >= 2 and re.fullmatch(r"[A-Za-z](?:_\{?\w+\}?)?", t)}
    again |= {t for s in stmts if s.kind == "boolean" and s.calc.get("name") is not None and s.side_texts
              for t in [s.side_texts[0].strip()] if t in firsts}
    if not again:
        return stmts
    out = []
    for s in stmts:
        if s.kind == "boolean" and s.calc.get("name") is None and s.side_texts and s.side_texts[0].strip() in again:
            first = s.calc["bool"][0]
            nm = first[1] if first is not None and first[0] == "var" else None
            if nm is not None:
                s = replace(s, calc={**s.calc, "name": nm, "bool": [None] + s.calc["bool"][1:]})
        out.append(s)
    return out


def _boolean_checks(no, s, names: dict) -> list:
    """a line in Boolean algebra -> its steps' checks: each side against the next by their truth tables ('A + AB = A' right,
    '\\overline{A + B} = \\bar{A} + \\bar{B}' wrong - a row where they differ shown); 'F = ...' again below ('F = AB + A\\bar{B}',
    then 'F = A'): against what F was. names: {name: (its last expression, its text)} in the problem"""
    from . import boolean as bl
    exprs, texts, name = s.calc["bool"], s.side_texts, s.calc.get("name")
    pairs = []
    if name is not None and name in names and len(exprs) > 1:
        pairs.append((names[name][1], names[name][0], texts[1], exprs[1], True))
    for k in range(1 if name is not None else 0, len(exprs) - 1):
        pairs.append((texts[k], exprs[k], texts[k + 1], exprs[k + 1], False))
    out = []

    def atom(e):                                     # a letter or its complement, a value
        return e is not None and (e[0] in ("var", "const") or (e[0] == "not" and e[1][0] == "var"))
    # letters the problem defined as expressions ('X = A + B' - a node of the circuit, then 'S = X + C = A + B + C'): put
    # in when two sides differ without them (X was a letter of its own - a false sure wrong step, 'not the same for X=1')
    defs = {n: e for n, (e, _t) in names.items() if e is not None and not atom(e) and n != name and n not in bl.variables(e)}
    # the inputs' values written ('A = 1, B = 0', then 'Y = \overline{AB} = \overline{1 \cdot 0} = 1'): a side in values only
    # is the other at them (it was against every row - 'not the same for A=1, B=1', a false sure wrong step)
    states = {n: e[1] for n, (e, _t) in names.items() if e is not None and e[0] == "const" and n != name}
    for ta, ea, tb, eb, again in pairs:
        if atom(ea) and atom(eb) and (again or not (ea[0] == eb[0] == "const")):
            continue                                 # (a state, not an identity: 'V_x = 0, \bar{V_x} = 1', 'A = B = V_{DD}' -
            #                                          the values a circuit's nodes take: they were false sure wrong steps; a
            #                                          node's value again, another - 'V_x = 1S', then 'V_x = 0S': another time)
        got = bl.equivalent(ea, eb)
        if got is None:
            continue
        same, row = got
        notes, sure, at = [], s.conf >= SURE, None
        sa, sb = ea, eb
        used = sorted((bl.variables(ea) | bl.variables(eb)) & set(defs))
        if not same and used:
            sa, sb = bl.substitute(ea, defs), bl.substitute(eb, defs)
            got = bl.equivalent(sa, sb)
            if got is None:
                continue
            same, row = got
            notes.append("with " + ", ".join(f"{u} = {bl.pretty(names[u][1])}" for u in used))
        if not same:
            # (some of them put in, the rest kept - a multiplexer at its select's value: 'S = 1', then 'Y = \bar{S}D_0 +
            # SD_1 = \bar{1}D_0 + 1 \cdot D_1 = D_1')
            va, vb = bl.variables(sa), bl.variables(sb)
            gone = (va - vb) if vb <= va else (vb - va) if va <= vb else set()
            if gone and gone <= set(states):
                at = {n: states[n] for n in sorted(gone)}
                put = {n: ("const", v) for n, v in at.items()}
                got = bl.equivalent(bl.substitute(sa, put), bl.substitute(sb, put))
                if got is None:
                    continue
                same, row = got
                notes.append("at " + bl.row_text(at))
                sure = sure and same                 # (a wrong there not sure: the values of another part, perhaps)
            elif again and eb[0] == "const" and ea[0] != "const":
                sure = False                         # ('F = A + B', then 'F = 1': its value at some inputs, perhaps)
                notes.append(f"{name}'s value at some inputs?")
        msg = f"'{bl.pretty(ta)} = {bl.pretty(tb)}'" + (f" ({'; '.join(notes)})" if notes else "")
        if not same:
            msg += f": not the same for {bl.row_text(row)}" if row else ": not the same there"
        out.append(Check(no, "right" if same else "wrong step", msg, s.conf, sure, [s.text], line=s.line))
    if name is not None and len(exprs) > 1:
        names[name] = (exprs[-1], texts[-1])
    return out


_WEAK_BOOL = re.compile(r"(?:\s|[A-Za-z01_{}()\[\]'+*·=,]|\\cdot(?![A-Za-z])|\\left(?![A-Za-z])|\\right(?![A-Za-z]))+")


def _root_lists(tex: str) -> str:
    """'x_{1,2} = -1, -2', 's_{1,2,3} = ...' (alone, or after ' | ', '\\Rightarrow', ';'): each root its value - 'x_1 = -1, x_2 =
    -2' (the second was lost, the first taken for x)"""
    def one(seg):
        m = re.fullmatch(r"(\s*)(\\?[A-Za-z]+)\s*_\s*\{\s*(\d(?:\s*,\s*\d){1,3})\s*\}\s*=\s*(.+?)(\s*)", seg)
        if not m or _PM.search(m.group(4)):
            return seg
        idx = [x.strip() for x in m.group(3).split(",")]
        vals = [x.strip() for x in _top_level_split(m.group(4)) if x.strip()]
        if len(vals) != len(idx) or any("=" in v for v in vals):
            return seg
        return m.group(1) + ", ".join(f"{m.group(2)}_{{{i}}} = {v}" for i, v in zip(idx, vals)) + m.group(5)
    parts = re.split(r"(\s\|\s|\\Rightarrow|\\implies|;)", tex)
    return "".join(p if k % 2 else one(p) for k, p in enumerate(parts))


_AT_W = re.compile(r"(?<![A-Za-z\\])[A-Za-z](?:_\{?\w+\}?)?\s*(?:\\left\s*)?\(\s*j\s*(\d+(?:\.\d+)?)\s*(?:\\right\s*)?\)")


_INSTANT = re.compile(r"(?<![A-Za-z\\])(\\[A-Za-z]+|[A-Za-z])(?:\s*_\s*(?:\{\s*([A-Za-z0-9 ]*?)\s*\}|([A-Za-z0-9])))?\s*"
                      r"(?:\\left\s*)?\(\s*(?:0\s*\^\s*\{?\s*([-+])\s*\}?|(\\infty))\s*(?:\\right\s*)?\)")
_SQRT_VOLT = re.compile(r"(\d)\s*(?:\\[,; ]\s*)*(?:\\mathrm\s*\{\s*V\s*\}|V)\s*\^\s*(?:\{\s*(?:1\s*/\s*2|0\.5|\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*\})\s*\}|0\.5)"
                        r"(?![\d.])|(\d)\s*(?:\\[,; ]\s*)*\\sqrt\s*\{\s*(?:\\mathrm\s*\{\s*)?V\s*\}?\s*\}")
_INSTANT_SHOWN = re.compile(r"_(\w*?)[Qq][Qq]([pmizPMIZ])(?![A-Za-z0-9])")


def _instant_name(m) -> str:
    """'v_C(0^+)' -> 'v_{CQQp}', 'v(0^-)' -> 'v_{QQm}', 'v(\\infty)' -> 'v_{QQi}': one name (_pretty shows it as written)"""
    base = m.group(1)
    if base.startswith("\\") and base[1:] not in _GREEK:
        return m.group(0)                            # (a function's: '\\sin(0^+)')
    sub = (m.group(2) or m.group(3) or "").replace(" ", "")
    return f"{base}_{{{sub}QQ{'i' if m.group(5) else 'p' if m.group(4) == '+' else 'm'}}}"


def _line_breaks(tex: str) -> list:
    """a line break the reader writes between two lines of work ('V = 12 V \\\\ R = 4 k\\Omega', a crop of two lines): the
    lines apart - it was a product of their sides ('V = 12 V \\\\ R = 4 k\\Omega', then 'I = V/R = 3 mA': a false sure
    wrong). Not inside an environment ('\\begin{aligned}', a matrix: rows) or a brace; a line that begins with '=' goes on
    from the one before ('I = \\frac{12}{4k} \\\\ = 3 mA' - _continued)"""
    out, cur, env, depth, i = [], [], 0, 0, 0
    while i < len(tex):
        if tex.startswith("\\begin", i):
            env += 1
        elif tex.startswith("\\end", i):
            env = max(0, env - 1)
        c = tex[i]
        if c == "{" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1
        elif c == "}" and (i == 0 or tex[i - 1] != "\\"):
            depth = max(0, depth - 1)
        if env == 0 and depth == 0 and tex.startswith("\\\\", i):
            j = i + 2
            m = re.match(r"\s*\[[^\]]*\]", tex[j:])         # ('\\\\[2pt]': a spacing)
            j += m.end() if m else 0
            out.append("".join(cur))                 # (a line that begins with '=': _continued joins it on)
            cur = []
            i = j
            continue
        cur.append(c)
        i += 1
    out.append("".join(cur))
    return [p for p in out if p.strip()]


_BIDI = re.compile("[\\u200b-\\u200f\\u202a-\\u202e\\u2066-\\u2069\\ufeff\\u061c\\u00ad]")   # direction marks, zero-width spaces


def statements(tex: str, conf: float = 1.0, line: int = -1, piece: int = -1):
    """a line as read -> its statements; a response at a frequency written ('|H(j10)| = 0.707'): that frequency in
    calc['at_w'] (the formula's omega there)"""
    if tex and "\\\\" in tex:
        parts = _line_breaks(tex)
        if len(parts) > 1:                           # (each line read as a line of its own - see _line_breaks)
            return [st for p in parts for st in statements(p, conf, line, piece)]
    if tex and _BIDI.search(tex):
        # the direction marks Hebrew text carries, typed or pasted ('I = 3 mA' and a right-to-left mark): invisible - read
        # as I = 10, a false sure wrong, or a step broken ('12/4k = 3 mA' with a left-to-right mark after the k)
        tex = _BIDI.sub("", tex)
    if tex and "_" in tex and re.search(r"_\s*\{?\s*\\(?:textbf|textit|mathbf|mathit|mathsf|mathtt|textsf|texttt|boldsymbol|bm)\s*\{", tex):
        # a subscript's word in bold or italic ('V_{\textbf{out}}'): the name as plain ('V_{out}' - it was another letter)
        tex = re.sub(r"_\s*\{\s*\\(?:textbf|textit|mathbf|mathit|mathsf|mathtt|textsf|texttt|boldsymbol|bm)\s*\{([^{}]*)\}\s*\}|"
                     r"_\s*\\(?:textbf|textit|mathbf|mathit|mathsf|mathtt|textsf|texttt|boldsymbol|bm)\s*\{([^{}]*)\}",
                     lambda m: "_{" + (m.group(1) if m.group(1) is not None else m.group(2)).strip() + "}", tex)
    if tex and "{" in tex:                           # ('I = {\frac{12}{4k}} = 3 mA': the side was not read - the line lost)
        tex = _bare_braces(tex)
    if tex and ("\\left" in tex or "\\right" in tex) and "\\begin" not in tex:
        # "f' \left( x \right) = 2x": sizes only - the name before it was lost (not around an array, '\left( \begin{array}':
        # a matrix, as _statements reads it)
        tex = re.sub(r"\\right\s*\)", ")", re.sub(r"\\left\s*\(", "(", tex))
    if tex and "V" in tex and _SQRT_VOLT.search(tex):
        # the body effect's coefficient in square-root volts ('\gamma = 0.4 V^{1/2}', '0.4 \sqrt{V}'): the number (the
        # unit's power was the number's - 0.4 read as √0.4 = 0.632, a false sure wrong threshold; '\sqrt{V}' no value)
        tex = _SQRT_VOLT.sub(lambda m: m.group(1) or m.group(2), tex)
    if tex and "_" in tex and re.search(r"_\s*\{[^{}]*\\(?:min|max)(?![A-Za-z])|_\s*\\(?:min|max)(?![A-Za-z])", tex):
        # a subscript with \min, \max in it ('V_{OH,\max}', 'C_{g,\min}', as a formula reader writes them): its letters - the
        # statement was lost
        for _ in range(3):
            tex = re.sub(r"(_\s*\{[^{}]*?)\\(min|max)(?![A-Za-z])", r"\1\2", tex)
        tex = re.sub(r"_\s*\\(min|max)(?![A-Za-z])", r"_{\1}", tex)
    if tex and "\\_" in tex:
        # an escaped underscore inside a subscript ('C_{in\_ inv}', as a document converter writes it): a comma - the name's
        # letters were lost ('0.69R_{eq}C_{in\_ inv}' read as 0.69)
        for _ in range(3):
            tex = re.sub(r"(_\s*\{[^{}]*?)\\_\s*", r"\1,", tex)
    if tex and "(" in tex and _INSTANT.search(tex):
        # a value at an instant ('v_C(0^+) = 5 V', 'i_L(0^-)', 'V_{out}(\infty)', 'Y_{2}( 0^{-} )' - a transient's, a CMOS
        # node's): one name - it was lost ('v_C(0^-) = 5 V' no statement), and inside a step v_C times 0 ('\frac{10 -
        # V_C(0^{+})}{2 k\Omega} = 3 mA' and 'Y_2(0^-) = 1.8 - 0.649' false sure wrong steps); shown so again (_pretty)
        tex = _INSTANT.sub(_instant_name, tex)
    if tex and re.search(r"\d\s+[munpkMG]\s+(?:s|A|V|W|F|H|Hz)(?![A-Za-z])", tex):
        # a prefix and its unit spaced apart, as the formula reader spaces tokens ('1 m s', '2 m A'): one unit (letters before)
        tex = re.sub(r"(\d)\s+([munpkMG])\s+(s|A|V|W|F|H|Hz)(?![A-Za-z])", r"\1 \2\3", tex)
    if tex and "(" in tex and "j" in tex:            # 'H(j10^3)', 'H(j1k)': at 1000 (as 'H(j1000)')
        tex = re.sub(r"(\(\s*j\s*)(?:(\d+(?:\.\d+)?)\s*(?:\\cdot|\\times)\s*)?10\s*\^\s*\{?\s*(\d+)\s*\}?(?=\s*(?:\\right\s*)?\))",
                     lambda m: f"{m.group(1)}{float(m.group(2) or 1) * 10 ** int(m.group(3)):g}", tex)
        tex = re.sub(r"(\(\s*j\s*)(\d+(?:\.\d+)?)\s*k(?=\s*(?:\\right\s*)?\))", lambda m: f"{m.group(1)}{float(m.group(2)) * 1000:g}", tex)
    if tex and re.search(r"_\s*\{\s*\d\s*,", tex):
        tex = _root_lists(tex)
    if tex and re.search(r"(?:gcd|lcm|max|min)\s*\}?\s*(?:\\left\s*)?\(\s*-?\d", tex):     # ('\gcd(12, 18)', '\max(3, 5)': one call, not two statements)
        tex = re.sub(r"((?:gcd|lcm|max|min)\s*\}?\s*(?:\\left\s*)?\(\s*-?\d+(?:\.\d+)?\s*),(\s*-?\d+(?:\.\d+)?\s*(?:\\right\s*)?\))",
                     "\\1\ue002\\2", tex)
    if tex and re.search(r"\d\s*(?:k|M|m|µ|\\mu\s*)?(?:N|Pa)\s*$", tex):    # '\frac{k q_1 q_2}{(0.3 m)^2} = 0.2 N': metres squared
        tex = re.sub(r"\(\s*(\d+(?:\.\d+)?)\s*m\s*\)\s*(\^\s*\{?\s*[23]\s*\}?)", r"(\1)\2", tex)     # (m was a letter)
    if tex and "_" in tex:                           # 'x_₁': the subscript after an underscore ('_{1}')
        tex = re.sub(r"_\s*([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: "_{" + m.group(1).translate(_UNSUB) + "}", tex)
    if tex and "^" in tex and re.search(r"\(\s*(?:\\Omega|Ω)\s*(?:\\cdot\s*|·\s*)?c?m\s*\)\s*\^\s*\{?\s*-\s*1\s*\}?", tex):
        # a conductivity's unit ('0.216 (\Omega \cdot cm)^{-1}'): as S/cm (it was letters - no verdict)
        tex = re.sub(r"\(\s*(?:\\Omega|Ω)\s*(?:\\cdot\s*|·\s*)?(c?m)\s*\)\s*\^\s*\{?\s*-\s*1\s*\}?", r"S/\1", tex)
    if tex and "\\angle" in tex:                     # '2 mA \angle 30^\circ': the unit after the angle, as a phasor reads ('2
        tex = re.sub(r"(\d+(?:\.\d+)?)\s*((?:[kmMunpµ]|\\mu\s*)?(?:VA|V|A|W|\\Omega|Ω))\s*(\\angle\s*-?\s*\d+(?:\.\d+)?\s*"   # \angle
                     r"(?:\^\s*\{?\s*\\circ\s*\}?|°)?)", r"\1 \3 \2", tex)    # 30^\circ mA'; it was 2 times angle times 30)
    if tex and "=" in tex and re.search(r"[\d}]\s*(?:[kmMunpµ]|\\mu\s*)?(?:F|H|V|A|S|C|W)\s*/\s*m(?:\s*\^\s*\{?\s*2\s*\}?)?(?![A-Za-z])", tex):
        # a field's units in SI ('8.85 \cdot 10^{-12} F/m', '2 kV/m', '5 W/m^2'): their number (F and m were letters)
        pre = {"k": 1e3, "M": 1e6, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12, "": 1.0}
        tex = re.sub(r"(\d+(?:\.\d+)?|\})\s*((?:[kmMunpµ]|\\mu\s*)?)(?:F|H|V|A|S|C|W)\s*/\s*m(?:\s*\^\s*\{?\s*2\s*\}?)?(?![A-Za-z])",
                     lambda m: m.group(1) if not m.group(2).strip() else
                     f"{m.group(1)} \\cdot {_factor_tex(pre[m.group(2).replace(chr(92) + 'mu', 'µ').strip()])}", tex)   # ('10^{-12} F/m')
    if tex and "/" in tex and re.search(r"\d\s*(?:[kmMunpµ]|\\mu\s*)?A\s*/\s*V(?:\s*\^\s*\{?\s*2\s*\}?)?(?![A-Za-z])", tex):
        # a transconductance's unit, a MOSFET's k ('g_m = 2 mA/V', 'k_n = 2 mA/V^2'): as '\frac{mA}{V}' (A and V were letters
        # of a formula - no value: 'no value for k_n')
        tex = re.sub(r"(\d)\s*((?:[kmMunpµ]|\\mu\s*)?A)\s*/\s*(V(?:\s*\^\s*\{?\s*2\s*\}?)?)(?![A-Za-z])", r"\1 \\frac{\2}{\3}", tex)
    if tex and re.search(r"\d\s*(?:k|M)?bps\s*$", tex) and "=" in tex:     # '8 kHz \cdot 8 bits = 64 kbps': bits a count
        head, last = tex.rsplit("=", 1)
        tex = re.sub(r"(\d+(?:\.\d+)?)\s*bits?(?![A-Za-z])", r"\1", head) + "=" + last
    mv = re.search(r"\d\s*(?:k?m|cm)\s*/\s*(?:s|h)\s*(?:\^\s*\{?\s*2\s*\}?)?\s*$", tex or "")
    if mv and tex.count("=") >= 2:                   # '\frac{100 m}{10 s} = 10 m/s': lengths and times before a speed
        head, last = tex.rsplit("=", 1)
        f_lt = {"km": 1e3, "cm": 1e-2, "mm": 1e-3, "m": 1.0, "ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0}
        head = re.sub(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*(km|cm|mm|ms|min|m|s|h)(?![A-Za-z/])",
                      lambda m: m.group(1) if f_lt[m.group(2)] == 1.0 else f"({m.group(1)} \\cdot {_factor_tex(f_lt[m.group(2)])})", head)
        tex = head + "=" + last
    ml = re.search(r"\d\s*(km|cm|mm|m)\s*(?:\^\s*\{?\s*([23])\s*\}?)?\s*$", tex or "")
    if ml and tex.count("=") >= 2:                   # '5 m - 2 m = 3 m', '2m \cdot 3m = 6 m^2', '2^3 m = 8 m': lengths before a length
        head, last = tex.rsplit("=", 1)              # (a lone m was a letter there - metres or milli? - no verdict)
        f_len = {"km": 1e3, "cm": 1e-2, "mm": 1e-3, "m": 1.0}

        def _len(m):
            f = f_len[m.group(2)] ** int(m.group(3) or 1)
            return m.group(1) if f == 1.0 else f"({m.group(1)} \\cdot {_factor_tex(f)})"
        head = re.sub(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*(km|cm|mm|m)(?:\s*\^\s*\{?\s*([23])\s*\}?)?(?![A-Za-z/])", _len, head)
        # (not 'm/s': a speed's unit - '\frac{3 \cdot 10^8 m/s}{2.4 GHz} = 12.5 cm')
        tex = head + "=" + last
    if tex and re.search(r"\d\s*(?:[kMm]\s*)?(?:W|A|VA)\s*h\s*$", tex) and "=" in tex:
        # '100 W \cdot 2 h = 200 Wh': the hours before the last side as hours (h was a letter: no verdict)
        head, last = tex.rsplit("=", 1)
        tex = re.sub(r"(?<![A-Za-z_\\])(\d+(?:\.\d+)?)\s*h(?![A-Za-z])", r"(\1 \\cdot 3600)", head) + "=" + last
    out = _statements(tex, conf, line, piece)
    if tex and re.search(r"\\(?:sin|cos|tan|cot|sec|csc)\s*(?:\\left\s*)?\(\s*-?\s*\d+(?:\.\d+)?\s*rad", tex):
        for x in out:                                # ('\cos(60 rad) = 0.5': radians said - not degrees with the mark left out)
            x.calc = {**x.calc, "rad_arg": True}
    ws = {float(m.group(1)) for m in _AT_W.finditer(tex or "")}
    if len(ws) == 1 and out:
        for x in out:
            x.calc = {**x.calc, "at_w": next(iter(ws))}
    return out


def _statements(tex: str, conf: float = 1.0, line: int = -1, piece: int = -1):
    out = []
    from .circuit_check import _respace                    # the formula reader's habits: '\mathrm { m A }', 'V _ { o u t }'
    from .normalize import undecorate                      # '\boxed{V = 10 V}', '{\color{red} I = 2 A}': the content
    from . import boolean as _bl
    tex = _unbrace(undecorate(tex))
    if "\\" in tex and re.search(r"(?<!\\)\\(?=\s|$)", tex):
        # a control space ('Y_1 = 0.5V_{DD}\ ', '\mathbf{\ }' as a document's converter writes it): a space - V_{DD} before
        # it was dropped, 'Y_1 = 0.5' a value (a false one); not a line break, '\\'
        tex = re.sub(r"(?<!\\)\\(?=\s|$)", " ", tex).strip()
    if "=" in tex and (_bl.MARK.search(tex) or _bl.primed(tex)):    # 'F = A\bar{B} + \bar{A}B = A \oplus B': Boolean algebra (the line was
        got = _boolean_statements(tex, conf, line, piece)  # dropped - \bar no letter to the parser: no verdict)
        if got is not None:
            return got
    tex = re.sub(r"\\(mathrm|text|textrm|rm)\s*\{\s*([^{}^]*?)\s*\^\s*(\{[^{}]*\}|[^{}\s])\s*\}",    # a power inside upright
                 lambda m: "\\" + m.group(1) + "{" + m.group(2) + "}^" + m.group(3), tex)     # text: the text's ('\mathrm{cm^{-3}}')
    # superscripts and subscripts as typed ('8×10⁻¹⁹ J', 'log₂(8)'): in LaTeX before the units are looked for (the unit
    # of '8×10⁻¹⁹ J' was not found - J a letter: no verdict)
    tex = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^{" + m.group(0).translate(_UNSUP) + "}", tex)
    tex = re.sub(r"\\(?:no)?limits(?![A-Za-z])", "", tex)   # '\lim\limits_{x \to 0}': where the limits go, nothing else
    # a dot for times before a power of ten ('5.10^{9}', '2.10^3' - 5·10⁹, as many write it, not 5.1 to the 9th)
    tex = re.sub(r"(?<![\d.])([1-9]\d{0,2})\.(10\s*\^)", "\\1 \\\\cdot \\2", tex)
    tex = re.sub(r"\\left(?=\s*\\left)|\\right(?=\s*\\right(?![A-Za-z]))", "", tex)   # '\left \left(': a stray \left (a
    #                                                  bracket of the student's own then gets the reader's - one is enough)
    tex = re.sub(r"(?<=\d)(\s*)μ(?=\s*(?:F|H|A|V|W|s|S|m|Ω|\\Omega|C|J|T|Wb)(?![A-Za-z]))", "\\1µ", tex)   # a Greek mu as
    #                                                  micro before a unit ('10 μF' - the micro sign was known, not it)
    # units as students spell them after a number: '4.7 K\Omega' (a capital K - kelvin-ohms are no unit), '2 kohm',
    # '0.02 sec', '50 hz', '40 db' - they were letters of the problem: no verdict
    tex = re.sub(r"(?<=\d)(\s*)K(?=\s*(?:\\Omega|Ω)(?![A-Za-z]))", "\\1k", tex)
    # a resistor's code ('4k7', '2k2', '1M5', '4R7' - 4.7 kΩ ...) as a value of its own: the digits around its letter
    tex = re.sub(r"(=\s*)(\d+)([kKMR])(\d+)(?=\s*(?:\\Omega|Ω)?\s*(?:$|,|;|\\q?quad))",
                 lambda m: f"{m.group(1)}{m.group(2)}.{m.group(4)}" + ("" if m.group(3) == "R" else " " + m.group(3).replace("K", "k")), tex)
    # a value's kind after its unit ('10 V DC', '10 V_{rms}', '10 V (rms)', '10 V_{pp}', '5 A peak'): the value as written
    # (it was letters of the problem: no verdict)
    tex = re.sub(r"(\d\s*(?:[kmunpµMG]|\\mu\s*)?(?:V|A|W|Ω|\\Omega|H\s*z))\s*(?:_\s*\{?\s*(?:r\s*m\s*s|R\s*M\s*S|p\s*-?\s*p|D\s*C|A\s*C|d\s*c|"
                 r"a\s*c|p\s*e\s*a\s*k|m\s*a\s*x|e\s*f\s*f)\s*\}?|\s+(?:\(\s*)?(?:rms|RMS|DC|AC|peak|pp)(?:\s*\))?)(?=\s*(?:$|,|;|\\q?quad))",
                 "\\1", tex)
    # (after a number's unit only - 'V_p = \sqrt{2} V_{rms}' keeps its name)
    tex = re.sub(r"(?<=\d)(\s*)([kKM]?)[Oo]hms?(?![A-Za-z])", lambda m: f"{m.group(1)}{m.group(2).replace('K', 'k')}\\Omega ", tex)
    # units spelled out ('20 Watt', '5 volts', '2 Amps', '50 Hertz' - letters of the problem: no verdict)
    for word, unit in (("[Ww]atts?", "W"), ("[Vv]olts?", "V"), ("[Aa]mp(?:ere)?s?", "A"), ("[Hh]ertz", "Hz"), ("[Ff]arads?", "F"),
                       ("[Hh]enr(?:y|ies)", "H"), ("[Jj]oules?", "J"), ("[Cc]oulombs?", "C"), ("[Ss]iemens", "S")):
        tex = re.sub(r"(?<=\d)(\s*)([kKmMµun]?)" + word + r"(?![A-Za-z])", lambda m, u=unit: f"{m.group(1)}{m.group(2).replace('K', 'k')}{u} ", tex)
    # a degree mark written as a small o ('53.13^o', '30^{o}', '\cos 60^o': the line was unread - o a letter in a power)
    tex = re.sub(r"(?<=\d)\s*\^\s*(?:\{\s*o\s*\}|o)(?![A-Za-z])", "^\\\\circ", tex)
    # an angle's unit inside a trig function ('\cos(60 rad)', '\sin(30 deg)': rad and deg were letters)
    tex = re.sub(r"(\\(?:sin|cos|tan|cot|sec|csc)\s*(?:\\left\s*)?\(\s*-?\s*\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)(?=\s*(?:\\right\s*)?\))",
                 r"\1^\\circ", tex)
    tex = re.sub(r"(\\(?:sin|cos|tan|cot|sec|csc)\s*(?:\\left\s*)?\(\s*-?\s*\d+(?:\.\d+)?)\s*rad(?:ians?)?(?=\s*(?:\\right\s*)?\))",
                 r"\1", tex)
    # 'sec' read letter by letter, its e as a c ('51.86 p s c c', '3 s e c' - a reader that spaces its letters)
    tex = re.sub(r"(?<=\d)(\s*)([mµunpP]?)\s*s\s*[ec]\s*c(?![A-Za-z])", "\\1\\2sec", tex)
    # a capital P for pico after a number ('51.86 Psec', '1 P F' - a handwritten p read as P; peta-seconds are no answer)
    tex = re.sub(r"(?<=\d)(\s*)P\s*(?=(?:sec|s|F|H)(?![A-Za-z]))", "\\1p", tex)
    tex = re.sub(r"(?<=\d)(\s*)([mµunp]|\\mu\s*)?sec(?![A-Za-z])", lambda m: m.group(1) + (m.group(2) or "").strip() + ("" if not m.group(2) or not m.group(2).startswith("\\") else " ") + "s", tex)
    tex = re.sub(r"(?<=\d)(\s*)([kMG]?)hz(?![A-Za-z])", "\\1\\2Hz", tex)
    if not re.search(r"\\[oi]+nt|∫", tex):                # (not an integral's 'db')
        tex = re.sub(r"(?<=\d)(\s*)db(?![A-Za-z])", "\\1dB", tex)
    tex = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "_{" + m.group(0).translate(_UNSUB) + "}", tex)
    tex = _SEQ.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}({m.group(3)})", tex)     # 'h[n]', 'x[2]': a sequence's value
    # an array between bars / brackets is a determinant / a matrix, as the formula reader writes them
    for left, right, env in ((r"\|", r"\|", "vmatrix"), (r"\(", r"\)", "pmatrix"), (r"\[", r"\]", "bmatrix")):
        tex = re.sub(r"\\left\s*" + left + r"\s*\\begin\s*\{\s*array\s*\}\s*\{[^{}]*\}(.*?)\\end\s*\{\s*array\s*\}\s*\\right\s*" + right,
                     lambda m: "\\begin{" + env + "}" + m.group(1) + "\\end{" + env + "}", tex, flags=re.S)
    def rows(m):                                   # a system in a brace: its rows, one statement each
        body = m.group("b")
        if "=" not in body:
            return m.group(0)
        return " " + " ; ".join(" ".join(_unbrace(c) for c in r.split("&")) for r in re.split(r"\\\\", body)) + " "
    tex = re.sub(r"(?:\\left\s*\\\{\s*)?\\begin\s*\{\s*(?P<e>cases|array|aligned|align\*?|gathered|split|matrix|smallmatrix)\s*\}(?:\s*\{[lcr|\s]*\})?"
                 r"(?P<b>.*?)\\end\s*\{\s*(?P=e)\s*\}(?:\s*\\right\s*\.)?", rows, tex, flags=re.S)
    mats = re.compile(r"\\begin\s*\{\s*([pbvBV]?matrix)\s*\}.*?\\end\s*\{\s*\1\s*\}", re.S)
    tex = mats.sub(lambda m: m.group(0).replace("\\\\", "\ue020"), tex)      # a matrix's rows kept through the
    tex = _respace(tex).replace("\ue020", "\\\\")                         # formula reader's evening-out
    # a power factor's character ('= 0.6 \text{ lagging}'): a word, not letters of the last side (no verdict before)
    tex = re.sub(r"\(?\s*(?<![A-Za-z])(?:lagging|leading|inductive|capacitive|lag|lead)(?![A-Za-z])\s*\)?(?=\s*(?:$|[,;|]|\\quad))",
                 " ", tex, flags=re.IGNORECASE)
    tex = _RANGE.sub(_range, tex)                           # '20 Hz - 20 kHz', '5 V \sim 12 V': a range, no value
    # a tolerance ('R = 100 \Omega \pm 5\%', '5 V \pm 2\%', '10 V \pm 0.1 V' - after a unit): the value is the nominal one (it
    # was two values, 100.05 and 99.95, named apart as roots - R itself none)
    tex = re.sub(r"(?:\\pm|±)\s*\d+(?:\.\d+)?\s*\\?%", "", tex)
    tex = re.sub(r"(\d\s*(?:[kmunpµMG]|\\mu\s*)?(?:V|A|W|\\Omega|Ω|F|H|Hz|s))\s*(?:\\pm|±)\s*\d+(?:\.\d+)?\s*(?:[kmunpµMG]|\\mu\s*)?"
                 r"(?:V|A|W|\\Omega|Ω|F|H|Hz|s)?(?=\s*(?:$|,|;|\\q?quad))", lambda m: m.group(1), tex)
    #                                                  (after a number's unit only: 'x = b \pm 2' are two roots)
    tex = _REPEATING.sub(repeating, re.sub(r"\\approx|\\simeq|\\cong|\\thickapprox|\\approxeq|≈|≃|≅|(?:\\sim|\\thicksim|∼)(?=\s*[-+]?\s*\d)",
                                           "=", tex))     # ('\sim' before a number only: 'X \sim N(0, 1)' is a distribution)
    tex = _complex_units(tex)                              # '3 + j4 \Omega': the unit the whole value's
    tex = _ee_words(tex)                                   # 'pf', 'BW': one name each
    # '\frac{5}{10} \cdot 100 = 50\%': the percent sign names what the 100 made - 50, not 0.5 (a false wrong step)
    tex = re.sub(r"((?:\\cdot|\\times|\*|×|·)\s*100\s*=\s*\d+(?:\s*\.\s*\d+)?)\s*\\?%", r"\1", tex)
    tex = re.sub(r"(?<=[\d}\s])\\?%", r" \\cdot 0.010 ", tex)      # '80\%': 0.8 ('\cdot 100\%' in a formula: 1) - as
    #                                                  0.010, which messages show as '%' (a '\cdot 0.01' written stays)
    tex = re.sub(r"\\left\s*\|", r"\\lvert ", re.sub(r"\\right\s*\|", r" \\rvert", tex))    # |Z|, |3 + j4|: a size,
    tex = re.sub(r"(?<![\\|])\|([^|=]{1,40}?)\|(?!\|)", r"\\lvert \1 \\rvert", tex)    # not two statements' separator
    raw_parts = _top_level_split(_decimal_commas(tex))
    raw_parts = [_MARK_TEXT.sub(r"\1 ", p) for p in raw_parts]
    numbered = {k for k, p in enumerate(raw_parts) if _ITEM.match(p)}          # '2) ...': a new exercise
    labels = [_label(p) for p in raw_parts]                  # 'Q2: ...', '2a) ...' too; 'b) ...': a part of this one
    numbered |= {k for k, (_, new, _) in enumerate(labels) if new}
    marked = {k for k, (_, new, lab) in enumerate(labels) if lab and not new}
    raw_parts = [t for t, _, _ in labels]
    raw_parts = [_LABEL.sub("", p) for p in raw_parts]        # 'KCL: 3 + 2.5 = 1.2 + I': the label no letters of it
    # the punctuation words cut off before a part left ('נתון: V = 12 V' -> ': V = 12 V': unread), and a sentence's end
    # ('I = 3 mA.', 'I = 3 mA :תשובה': read as I = 10 - a false sure wrong; not an ellipsis, '0.333...'); words before it
    # ('given V = 12 V' - '\text{given }' - 'so I = 3 mA': read as letters of a formula)
    raw_parts = [_LEAD_WORDS.sub("", re.sub(r"(?<!\.)\s*[:.]\s*$", "", re.sub(r"^\s*[:.,]+\s*", "", p))) for p in raw_parts]
    raw_parts = [_DOMAIN.sub(r"\1", p) for p in raw_parts]     # 'i(t) = 2e^{-t} (t > 0)': the condition no part of the side
    # an equation's label after a unit ('= 2 A (1)', '= 2 A \quad (3)'): no factor of the side - it was unread (not '= 2 (3)',
    # a product)
    raw_parts = [re.sub(r"(?<=[A-Za-zΩ°}])\s+(?:\\q?quad\s*)?(?:\\left\s*)?\(\s*\d{1,2}\s*(?:\\right\s*)?\)\s*$", "", p) if "=" in p else p
                 for p in raw_parts]
    parts = [(_BRACKETED.sub(lambda m: " " + re.sub(r"\s+", "", m.group("u")), _ITEM.sub("", p)).strip(), k in numbered, k in marked)
             for k, p in enumerate(raw_parts)]
    for part, pm, item, sub in ((v, pm, item, sub) for p, item, sub in parts for v, pm in _pm_variants(p)):
        if part.count("=") < 1:
            continue
        unit_note, si_unit, si_eq, cm_unit = None, "", False, ""
        milli_alt = {}                               # {side: its text with a bare m as milli} where it was not taken so
        sides_txt = [s.strip() for s in _split_eq(part)]
        if len(sides_txt) >= 2 and not sides_txt[0] and all(sides_txt[1:]):
            # '= \frac{10}{5}', '= 2 A' on lines of their own: the chain above goes on (check_statements joins them)
            out.append(Statement("cont", part, [], conf, line=line, piece=piece, side_texts=sides_txt[1:]))
            continue
        if any(not s for s in sides_txt):
            continue
        if _MATRIX.search(part):                     # matrices written out: compared entry by entry, or a system's rows
            got = _matrix_statements(part, sides_txt, conf, line, piece)
            if got is not None:
                for g in got[:1] if item else []:
                    g.calc = {**g.calc, "item": True}
                out += got
                continue
        unit = ""
        time_ok = sum(1 for t in sides_txt if _TIMED.search(t)) >= 2
        whole = [_whole_si(t, time_ok) for t in sides_txt]
        whole = [None if w is not None and _letter_used(part, _UNIT.match(_join_digits(t)).group("unit") if _UNIT.match(_join_digits(t)) else "")
                 else w for w, t in zip(whole, sides_txt)]
        if all(w is not None for w in whole):        # '72 \frac{km}{h} = 20 \frac{m}{s}', '1 k\Omega = 1000 \Omega':
            sides_txt = _pick_den([w for w, _ in whole], [_den_si(t, time_ok) for t in sides_txt], sides_txt[0],
                                  [(_UNIT.match(_join_digits(t)) or {"unit": ""})["unit"] if _UNIT.match(_join_digits(t)) else "" for t in sides_txt])
            #                                          a value in two units,
            #                                          in SI units both ('12 / 4 k\Omega = 3 mA': the unit the divisor's)
        elif len(sides_txt) >= 3 and any(_INLINE_UNIT.search(_join_digits(t)) or _PAREN_UNIT.search(_join_digits(t)) or
                                         _POLAR_UNIT.search(_join_digits(t)) or _bare_side(part, t) or
                                         _INLINE_MORE.search(_join_digits(t)) or w is not None
                                         for t, w in zip(sides_txt[1:-1], whole[1:-1])):
            # a substitution written with its units ('= 2 mA \cdot 5 k\Omega = 10 V'): every side after the first in
            # SI units, so its steps compare (the units' letters would otherwise be read as letters of the problem)
            unit_note = _units_disagree(part, sides_txt[1:])
            si_unit = _last_unit(part) if whole[-1] is not None else ""     # (put in SI units below: for _chain's 1000s)
            if whole[-1] is not None and not si_unit and whole[-1][1]:   # (a compound one, 'cm^2/s': for _chain's
                mc = _COMPOUND.match(_unwrap_rm(_join_digits(sides_txt[-1])))       # centimetres - _cm_factor)
                cm_unit = re.sub(r"\s+", " ", mc.group("unit")).strip() if mc else ""
            milli, conv = _bare_m_ok(part), []
            for k, (t, w) in enumerate(zip(sides_txt[1:], whole[1:])):    # ('= 1 k\Omega \cdot 1 \mu F = 1 ms': the last too)
                bare = _bare_side(part, t) or k == len(sides_txt) - 2
                conv.append((w[0], w[0]) if w is not None else (_si_inline(_join_digits(t), bare, milli and k < len(sides_txt) - 2),
                                                                _si_inline(_join_digits(t), bare, nano=milli and k < len(sides_txt) - 2)))
                # (a bare n, p is nano, pico either way - '10k \cdot 100n': a wrong step is caught too; a bare m may be metres:
                # milli only where the step after it holds so)
            new = [a for a, _ in conv]
            for k in range(len(conv) - 2, -1, -1):  # a bare m taken for milli only where the step after it holds so (it
                if conv[k][0] != conv[k][1]:         # may be metres: '\rho \frac{100 m}{1 mm^2} = 1.7 \Omega')
                    va, vn = (_num(e) if e is not None else None for e in (_parse(new[k]), _parse(new[k + 1])))
                    lone = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*m\s*", new[k + 1]) if k + 1 == len(new) - 1 else None
                    vn = float(lone.group(1)) * 1e-3 if lone else vn      # ('= 1m' last: milli of the unit unsaid, below)
                    if va is None or vn is None or not _close(vn, va, new[k + 1]):
                        milli_alt[k + 1] = new[k]            # (kept: a wrong step so is a wrong step, not sure - '2m \cdot
                        new[k] = conv[k][1]                  # 4k = 9 V' had no verdict at all)
            # ('= 12 / 4 k\Omega = 3 mA', '= 3 \cdot 10^8 / 100 MHz = 3 m': the unit after a divisor's number is its)
            new = _pick_den(new, [_den_si(t, time_ok) if w is not None else None for t, w in zip(sides_txt[1:], whole[1:])],
                            sides_txt[0], [_UNIT.match(_join_digits(t)).group("unit") if _UNIT.match(_join_digits(t)) else ""
                                           for t in sides_txt[1:]])
            sides_txt = sides_txt[:1] + new
            lone = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*m\s*", _join_digits(sides_txt[-1]))
            prev = _parse(sides_txt[-2]) if lone else None
            if lone and prev is not None and not prev.free_symbols and _num(prev) is not None and \
                    _close(float(lone.group(1)) * 1e-3, _num(prev)):  # '= 2.2 k\Omega \cdot 10 \mu F = 22 m': milli
                sides_txt[-1] = f"({lone.group(1)}) \\cdot 10^{{-3}}"     # of the unit left unsaid, in SI as the rest
        um = _UNIT.match(_join_digits(sides_txt[-1])) if len(sides_txt) == 2 else None
        if len(sides_txt) == 2 and (um is None or _has_letters(um.group("num"))) and \
                (_INLINE_UNIT.search(_join_digits(sides_txt[-1])) or _PAREN_UNIT.search(_join_digits(sides_txt[-1]))):
            # 'I = \frac{12}{4 k\Omega}', 'V = 2 mA \cdot 5 k\Omega': a value worked out of numbers with their units - in SI
            # units (it was a formula in the units' letters: the formulas after it 'cannot check'); numbers only
            si_txt = _si_inline(_join_digits(sides_txt[-1]), bare=False)
            if not re.search(r"[A-Za-z]", re.sub(r"\\(?:frac|cdot|times|sqrt|left|right|pi)(?![A-Za-z])", "", si_txt)):
                sides_txt[-1] = si_txt
        um2 = [_UNIT.match(_join_digits(t)) for t in sides_txt] if len(sides_txt) == 2 else []
        if len(sides_txt) == 2 and _UNAMBIGUOUS.search(_join_digits(part)) and \
                not any(m_ is not None and not _has_letters(m_.group("num")) and
                        re.fullmatch(r"\s*\\?[A-Za-z]+(?:\s*_\s*(?:\{[^{}]*\}|[A-Za-z0-9]))?\s*", sides_txt[1 - k_]) for k_, m_ in enumerate(um2)) and \
                any(re.search(r"[A-Za-z]", re.sub(r"\\[A-Za-z]+|" + _UNAMBIGUOUS.pattern, "", _join_digits(t))) for t in sides_txt):
            #                                          (not a value - 'I_1 = 4.36 mA' keeps its number and unit)
            # an equation with numbers and their units ('12 V = 2 k\Omega \cdot I_1 + 1 k\Omega (I_1 - I_2)', a mesh's): every
            # number with its unit in SI units - when one unit at least is one no letter is spelt like (kΩ, mA - not '3 V I',
            # 3 times V times I); it had the units' letters as unknowns: 'cannot check'
            sides_txt, si_eq = [_si_inline(_join_digits(t), bare=False) for t in sides_txt], True
        if len(sides_txt) >= 2 and not _UNIT.match(_join_digits(sides_txt[-1])):
            sp_u = re.fullmatch(r"(.*?\d)\s+((?:\\?[A-Za-z]+\s*){2,4})", _join_digits(sides_txt[-1]))
            if sp_u:                                 # '= 2 m A', '= 1 k H z', '= 10 m s': a unit the reader spaced (it was
                joined = re.sub(r"\s+", "", sp_u.group(2))   # letters of the problem: no verdict)
                mj = _UNIT.match(f"{sp_u.group(1)} {joined}")
                if mj and re.sub(r"\s+", "", mj.group("unit")) == joined:
                    sides_txt[-1] = f"{sp_u.group(1)} {joined}"
        if len(sides_txt) >= 2:                      # 'I = 2 A': the unit is not a letter of the problem
            m = _UNIT.match(_join_digits(sides_txt[-1])) or _COMPOUND.match(_unwrap_rm(_join_digits(sides_txt[-1])))
            if m and m.group("num").count("{") != m.group("num").count("}"):
                m = None                             # ('\frac{10}{2 k\Omega}': a unit inside a fraction)
            if m and _bare_prefix(m.group("unit")) and not _plain_num(m.group("num")):
                m = None                             # ('= 12 / 4.7k': the prefix is the last number's)
            if m and re.sub(r"[\s{}]", "", m.group("unit")) == "T" and not re.match(r"\s*B(?![A-Za-z])", part):
                m = None                             # (tesla only for a B, or with a prefix: 't_1 = 2T' is two periods)
            den = _den_si(sides_txt[-1]) if m and len(sides_txt) == 2 else None
            if den is not None and _name_unit_differs(sides_txt[0], m.group("unit")):
                # 'I = 12 / 4 k\Omega', 'f = 1 / 2 ms': a current is no kΩ - the unit is the divisor's; the value in SI units
                # (it was I = 3 kΩ: every formula after it a false wrong)
                sides_txt[-1], m = den, None
            if m and not _has_letters(m.group("num")) and re.sub(r"[\s{}\\]", "", m.group("unit")) in ("°", "deg"):
                # '\phi = 36.87°', '36.87 deg': in radians, as '36.87^\circ' is - it was 36.87 radians ('1000 \sin\phi = 600'
                # under it a false wrong step)
                sides_txt[-1], m = m.group("num").strip() + "^\\circ", None
            if m and not _has_letters(m.group("num")):
                sides_txt[-1], unit = m.group("num").strip(), m.group("unit")
                f = _si(unit)
                prev = _parse(sides_txt[-2]) if len(sides_txt) >= 3 and f not in (None, 1.0) and unit.strip() != "m" else None
                last = _parse(sides_txt[-1]) if prev is not None and not prev.free_symbols else None
                if last is not None and not last.free_symbols and _num(prev) is not None and _num(last) is not None and \
                        _close(_num(last) * f, _num(prev)) and not _close(_num(last), _num(prev)):
                    # 'T = 1 / f = 1 / 50 = 20 ms': the last value in the unit written, the side before it in SI
                    sides_txt[-1] = f"({sides_txt[-1]}) \\cdot {_factor_tex(f)}"
            else:                                    # a phasor, an impedance with its unit ('3 + j4 \\Omega', '2\\angle -53^\\circ A')
                m = _C_UNIT.match(_join_digits(sides_txt[-1]))
                if m and m.group("num").count("{") == m.group("num").count("}") and re.search(r"(?<![A-Za-z])j(?![A-Za-z])|\\angle", m.group("num")) and not re.search(
                        r"[A-Za-z]", re.sub(r"\\(?:frac|sqrt|pi|cdot|times|angle|circ|left|right)|(?<![A-Za-z])j(?![A-Za-z])", "",
                                            m.group("num"))):
                    sides_txt[-1], unit = m.group("num").strip(), re.sub(r"\s+", "", m.group("unit"))
        lap = [_laplace_side(t) if "L" in t else None for t in sides_txt]
        parsed = [(x, x) if x is not None else _parse_full(t) for x, t in zip(lap, sides_txt)]
        sides = [p for p, _ in parsed]
        if any(s is None for s in sides):
            # a side the parser cannot read ('P(X > 2) = 1 - P(X \le 2) = 1 - 0.7 = 0.3'): the runs of readable sides
            # between them are still steps the student wrote - each checked on its own (all was dropped before)
            run = []
            for k, x in enumerate(sides + [None]):
                if x is not None and not _letters(x):  # (a side in letters ends a run too: 'E[X] = np = 10 \cdot 0.3 =
                    run.append(k)                      # 3' keeps '10 \cdot 0.3 = 3')
                    continue
                if len(run) >= 2 and len(run) < len(sides):
                    sub_txt = [sides_txt[j] for j in run]
                    st = _classify(" = ".join(sub_txt), [sides[j] for j in run], sub_txt,
                                   unit if run[-1] == len(sides) - 1 else "", conf, dict(line=line, piece=piece, side_texts=sub_txt))
                    if st is not None and st.kind in ("numeric", "chain") and not any(_letters(sides[j]) for j in run):
                        out.append(st)               # (numbers only: a run in letters may lack its context)
                run = []
            continue
        if len(sides) == 2:                          # 'j = 3': a letter (a counter), not the imaginary unit ('j^2 = -1'
            for k in (0, 1):                         # and 'j = \sqrt{-1}' stay about the unit)
                if sides_txt[k].strip() == "j" and _num(sides[1 - k]) is not None:
                    sides[k] = sp.Symbol("j", real=True)
        kw = dict(line=line, piece=piece, side_texts=sides_txt)
        if any(x is not None for x in lap):          # a Laplace transform worked out: compared as an expression
            sides = [_signals(x) for x in sides]
            got = _with_identity(part, sides, [r for _, r in parsed], sides_txt, unit, conf, kw,
                                 {"indefinite": [], "may_be_wrong": True})
            for g in got[:1] if item else []:
                g.calc = {**g.calc, "item": True}
            out += got
            continue
        if _CALC.search(part):                       # a derivative, an integral, a sum worked out in letters
            got = _with_identity(part, sides, [r for _, r in parsed], sides_txt, unit, conf, kw, _calculus(parsed))
            for g in got[:1] if item else []:
                g.calc = {**g.calc, "item": True}
            out += got
            continue
        st = _classify(part, sides, sides_txt, unit, conf, kw)
        if st is not None and si_eq:
            st.calc = {**st.calc, "si": True}      # its numbers in SI units (the answers after it compared so)
        if st is not None and item:
            st.calc = {**st.calc, "item": True}
        if st is not None and sub:                 # ('b) ...': a sub-question's first line)
            st.calc = {**st.calc, "sub": True}
        if st is not None and milli_alt and st.kind in ("chain", "numeric") and len(st.sides) == len(sides_txt):
            st.calc = {**st.calc, "milli_alt": dict(milli_alt)}
        if st is not None and si_unit:
            st.calc = {**st.calc, "si_unit": si_unit}
        if st is not None and cm_unit:
            st.calc = {**st.calc, "cm_unit": cm_unit}
        if st is not None and st.kind in ("formula", "chain") and len(sides) == 2 and not any(x.is_Symbol for x in sides) \
                and not any(_unworked(x) for x in sides) and (st.kind == "formula" or not all(_letters(x) for x in sides)):
            # an identity in letters ('(a+b)^2 = a^2 + 2ab + b^2', partial fractions): right when it holds; a partial
            # fraction expansion that does not hold is wrong - any other may be an equation to solve ('x^2 + 1 = 2x')
            same, pf = _same_expr(sides[0], sides[1]), _partial_fractions(sides)
            if same or (same is False and pf):
                out.append(Statement("identity", part, sides, conf, calc={"indefinite": [], "may_be_wrong": pf,
                                                                            "pairs": [(0, 1)]}, **kw))
                continue
        if st is not None and st.kind in ("formula", "chain") and len(sides) >= 3:
            # 'y = (x+1)^2 = x^2 + 2x + 1', 'X(s) = \frac{1}{s(s+1)} = \frac{1}{s} - \frac{1}{s+1}': the steps between sides
            # in the same letters - right when they hold; a partial fraction expansion that does not hold is wrong - any
            # other may be an equation to solve ('V_A = 10 - 2I = 4I + 2')
            first = 1 if _is_name(sides[0]) else 0
            pairs, pf_any = [], False
            for i in range(first, len(sides) - 1):
                a, b = sides[i], sides[i + 1]
                la, lb = _letters(a), _letters(b)
                if la and lb and la != lb and (la < lb or lb < la) and first == 1 and not _unworked(a) and not _unworked(b) and \
                        _expanded_wrong(a, b):       # ('(x - y)(x + y) = x^2': an expansion with a term dropped)
                    pairs.append((i, i + 1))
                    pf_any = True
                    continue
                if not la or la != lb or _unworked(a) or _unworked(b) or a.is_Symbol or b.is_Symbol:
                    continue
                same, pf = _same_expr(a, b), _partial_fractions([a, b])
                if same or (same is False and pf):
                    pairs.append((i, i + 1))
                    pf_any = pf_any or (same is False and pf)
                elif same is False and first == 1 and _expanded_wrong(a, b) and not (
                        _WEAK_BOOL.fullmatch(part) and _boolean_statements(part, conf, line, piece, need_ops=False)):
                    # (not a line Boolean algebra may be: 'Y = (A + B)(A + C) = A + BC' holds there)
                    # 'y = (x+1)^2 = x^2 + x + 1': an expansion (one variable, the same degree and leading term), not an
                    # equation to solve - wrong (it was no verdict)
                    pairs.append((i, i + 1))
                    pf_any = True
            if pairs:
                out.append(Statement("identity", part, sides, conf, calc={"indefinite": [], "may_be_wrong": pf_any,
                                                                            "pairs": pairs}, **kw))
        if st is not None:
            if pm:
                st.calc = {**st.calc, "pm": pm}      # one of a '\pm' pair ('x_{1,2} = ...')
            out.append(st)
        if unit_note and st is not None:             # '10 V \cdot 2 mA = 20 mA': a step's two sides in units of
            out.append(Statement("unit", part, [], conf, line=line, piece=piece, side_texts=[],   # different kinds
                                 calc={"msg": unit_note}))
    return out


_LABEL = re.compile(r"^\s*(?:[A-Z]{2,5}\d?|[Aa]ns(?:wer)?|[Ss]ol(?:ution)?|[Gg]iven|[Dd]ata|[Ss]olve|[Ff]ind|[Cc]alc(?:ulate)?|[Cc]ompute|"
                    r"[Cc]heck|[Pp]roof|[Nn]ote|[Rr]esults?|[Ff]inal|[Tt]herefore|[Hh]ence|[Tt]hus|[Ss]o|[Ww]here)\s*:\s*")
_LEAD_WORDS = re.compile(r"^\s*(?:given|let|with|since|because|so|hence|thus|therefore|then|where|and|or|i\.?e\.?|we\s+get|get|gives|"
                         r"answer|ans|result|final(?:\s+answer)?)\s*[:,]?\s+(?=\\?[A-Za-z])", re.IGNORECASE)
_DIM_UNIT = re.compile(r"(?<![A-Za-z_\\\d.])(\d+(?:\.\d+)?)\s*(?P<rm>\\mathrm\s*\{\s*)?(?P<u>(?:\\mu\s*|[pnuµmkMG])?\s*"
                       r"(?:V\s*A\s*R|V\s*A|var|V|A|W|\\Omega|Ω|Hz|J)|(?:\\mu\s*|[pnuµmk])\s*(?:F|H|S|s))(?![A-Za-z_'])(?(rm)\s*\})")


def _side_dimension(text: str):
    """the dimension of a side made of numbers with their units ('\\frac{10 V}{2 mA}' -> V/A, '10 V \\cdot 2 mA' -> W):
    None when it has a letter, a number without a unit beside one with, or a function"""
    t = _join_digits(text)
    dims = []

    def rep(m):
        dims.append(_dimension(re.sub(r"\s+", "", m.group("u")).replace("VAR", "W").replace("var", "W").replace("VA", "W")))
        return f"({m.group(1)} \\cdot u_{{{len(dims) - 1}}})"
    t = _DIM_UNIT.sub(rep, t)
    if not dims or any(d is None for d in dims):
        return None
    rest = re.sub(r"\(\d+(?:\.\d+)?\s*\\cdot\s*u_\{\d+\}\)", " ", t)          # a number with no unit beside them
    rest = re.sub(r"\^\s*(?:\{[^{}]*\}|-?\d)|\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}|\\frac\s*\{\s*1\s*\}|2\s*\\pi|\\sqrt\s*\{\s*2\s*\}", " ", rest)
    if re.search(r"\d", rest):                     # ('\\frac{10}{2 k\\Omega}': 10 is in volts unsaid) - its unit unknown
        return None
    e = _parse(t)
    marks = {sp.Symbol(f"u_{k}", real=True): d for k, d in enumerate(dims)}
    if e is None or e.free_symbols - set(marks) or e.atoms(sp.Function):
        return None
    e = sp.factor(sp.expand(e.xreplace(marks)))
    dim = sp.Mul(*[f for f in sp.Mul.make_args(e) if f.free_symbols])      # the numbers (2, pi, sqrt(10)) dropped
    if dim.has(sp.Add) or not dim.free_symbols:
        return None
    return dim


def _units_disagree(part: str, sides: list):
    """consecutive sides written with units of different kinds ('= 10 V \\cdot 2 mA = 20 mA': W against A) -> a
    note, or None"""
    dims = [(_side_dimension(t), t) for t in sides]
    for (a, ta), (b, tb) in zip(dims, dims[1:]):
        if a is None or b is None or sp.simplify(a / b) == 1:
            continue                                 # (the units read as the substitution's SI conversion reads them)
        return (f"'{_pretty(ta)} = {_pretty(tb)}': the left side comes out in {_dim_text(a)}, the right side is in "
                f"{_dim_text(b)} - units of different kinds")
    return None


_CALC = re.compile(r"\\frac\s*\{\s*d|\\d?frac\s*\{\s*\\partial|\\i+nt(?![A-Za-z])|\\lim(?![A-Za-z])|\\sum(?![A-Za-z])|"
                   r"\\prod(?![A-Za-z])")


def _classify(part, sides, sides_txt, unit, conf, kw):
    """a statement's sides -> its kind: numbers only ('numeric', 'chain'), a letter's value ('value'), a computation
    with one letter to find ('chain'), else a relation between letters ('formula'); None: a value it cannot use"""
    letters = set().union(*(_letters(s) for s in sides))
    if not letters:
        return Statement("numeric" if len(sides) == 2 else "chain", part, sides, conf, **kw)
    if len(sides) == 2 and any(s.is_Symbol for s in sides) and any(not _letters(s) for s in sides):
        sym = sides[0] if sides[0].is_Symbol else sides[1]
        val = sides[1] if sides[0].is_Symbol else sides[0]
        num = _num(val)
        if num is None:
            num = _cnum(val)                       # a complex value: an impedance, a phasor ('Z = 3 + j4', '2\angle -53^\circ')
            if num is None:
                return None
        return Statement("value", part, sides, conf, symbol=sym, number=num, unit=unit,
                         value_text=sides_txt[1] if sides[0].is_Symbol else sides_txt[0], **kw)
    if len(sides) > 2 or (len(letters) == 1 and any(not _letters(s) for s in sides)):
        return Statement("chain", part, sides, conf, unit=unit, **kw)       # (unit: its last side's, as written)
    return Statement("formula", part, sides, conf, **kw)


def _is_name(e) -> bool:
    """a quantity's name on the left: a letter, or a function of letters ('f(x)', 'X(s)')"""
    from sympy.core.function import AppliedUndef
    return isinstance(e, sp.Symbol) or isinstance(e, AppliedUndef)


def _unworked(e) -> bool:
    """letters the checks cannot work with: a function nobody defined, a derivative left as a name (dy/dx), an
    integral or a limit sympy could not work out"""
    from sympy.core.function import AppliedUndef
    return bool(e.atoms(AppliedUndef)) or e.has(sp.Derivative, sp.Integral, sp.Limit, sp.Sum, sp.Product) or \
        any(str(x) in _D_OPAQUE for x in e.free_symbols)


def _calculus(parsed) -> dict:
    """what a part's calculus is: the variables of its indefinite integrals; whether it can be wrong as an identity -
    a derivative or an indefinite integral can; with a limit, a sum or a bound in letters it may be an equation
    ('\\int_0^a x\\,dx = 2a', a = 4)"""
    raws = [r for _, r in parsed if r is not None]
    indefinite = sorted({i.limits[0][0] for r in raws for i in r.atoms(sp.Integral) if len(i.limits[0]) == 1}, key=str)
    derivative = any(r.has(sp.Derivative) for r in raws)
    bounded = any(r.has(sp.Limit, sp.Sum, sp.Product) or any(len(i.limits[0]) == 3 for i in r.atoms(sp.Integral)) for r in raws)
    return {"indefinite": indefinite, "may_be_wrong": (derivative or bool(indefinite)) and not bounded}


def _identity_pairs(orig, sides):
    """consecutive sides to compare as expressions: both in letters (or one in a Fourier series' whole-number index
    only: '\\int_0^{2\\pi} \\sin^2(nx) dx = \\pi'), neither left unworked, and not a definition's letter
    ('v = \\frac{d}{dt}(5t^2) = 10t' defines v; '= 10t' is the derivative)"""
    def comparable(a, b):
        la, lb = _letters(a), _letters(b)
        if la and lb:
            return True
        return bool(la or lb) and (la | lb) <= set(_integer_letters(a, b))
    return [(i, i + 1) for i in range(len(sides) - 1)
            if comparable(sides[i], sides[i + 1]) and not _unworked(sides[i]) and not _unworked(sides[i + 1])
            and not (i == 0 and orig[0] is not None and orig[0].is_Symbol)]


_SAFE_CONSTANTS = {"a", "b", "c", "k", "n", "m", "r", "l", "s", "I", "alpha", "beta", "gamma", "zeta", "sigma", "tau",
                   "omega"}


def _assumed_identity(part, sides, orig, side_texts, conf, kw):
    """d/dt of an expression with other letters ('\\frac{d}{dt}(A \\cos(\\omega t)) = -A \\omega \\sin(\\omega t)'): kept
    unworked, as a letter may be a function of t (v, i, \\Phi). Worked out with the other letters taken for constants,
    a side after it that agrees is right - sure when they are constants by the usual names (A, \\omega, \\tau, k);
    one that does not agree is a wrong step only then, and when the expression is in the variable ('A\\cos(\\omega t)',
    not '\\frac{1}{2} J \\omega^2', where \\omega may be what changes) - else it says nothing (the student may have taken
    a letter for a function)"""
    out = []
    for i in range(len(sides) - 1):
        a, b = sides[i], sides[i + 1]
        if i == 0 and orig[0] is not None and orig[0].is_Symbol and str(orig[0]) not in _D_OPAQUE:
            continue                               # 'v = ...': a definition's letter
        names = [x for x in a.free_symbols if str(x) in _D_OPAQUE and not _D_OPAQUE[str(x)][0].is_Symbol]
        if not names or _unworked(b) or not _letters(b):
            continue
        sub, others = {}, set()
        for x in names:
            expr, v, n = _D_OPAQUE[str(x)]
            if not n:
                break
            try:
                sub[x] = sp.diff(expr, v, n)
            except Exception:                      # noqa: BLE001
                break
            others |= expr.free_symbols - {v}
        else:
            a2 = a.xreplace(sub)
            if _unworked(a2):
                continue
            same = _same_expr(a2, b)
            safe = all(str(x).split("_")[0] in _SAFE_CONSTANTS or str(x).split("_")[0].lower() in _SAFE_CONSTANTS
                       for x in others)                                  # (A as a: the letters keep their case)
            in_var = all(_D_OPAQUE[str(x)][1] in _D_OPAQUE[str(x)][0].free_symbols for x in names)
            if not same and not (same is False and safe and in_var):
                continue
            sides2 = list(sides)
            sides2[i] = a2
            out.append(Statement("identity", part, sides2, conf, calc={"indefinite": [], "may_be_wrong": safe and in_var,
                                                                     "pairs": [(i, i + 1)], "unsure": not safe}, **kw))
    return out


def _with_identity(part, sides, orig, side_texts, unit, conf, kw, calc):
    """a part whose sides in letters are compared as expressions (calculus, a function's derivative) -> its statements:
    the 'identity', and what is left to check the usual way - a letter it defines ('v = ... = 10t'), an equation after
    it ("f'(x) = 3x^2 - 3 = 0"), or the whole when it may be an equation"""
    pairs = _identity_pairs(orig, sides)
    if not pairs:
        st = _classify(part, sides, side_texts, unit, conf, kw)
        return _assumed_identity(part, sides, orig, side_texts, conf, kw) + ([st] if st is not None else [])
    out = [Statement("identity", part, sides, conf, calc={**calc, "pairs": pairs}, **kw)]
    k = max(j for _, j in pairs)
    st = None
    if orig[0] is not None and orig[0].is_Symbol:
        st = _classify(part, sides, side_texts, unit, conf, kw)
    elif len(sides) - k >= 2:
        st = _classify(part, sides[k:], side_texts[k:], unit, conf, {**kw, "side_texts": side_texts[k:]})
    elif not calc["may_be_wrong"]:
        st = _classify(part, sides, side_texts, unit, conf, kw)
    if st is not None:
        out.append(st)
    return out


def _partial_fractions(sides) -> bool:
    """'\\frac{s+3}{(s+1)(s+2)} = \\frac{2}{s+1} - \\frac{1}{s+2}': one fraction in one letter, against a sum of fractions
    in the same letter - an expansion, not an equation"""
    a, b = sides
    if len(a.free_symbols) != 1 or a.free_symbols != b.free_symbols:
        return False
    (v,) = a.free_symbols
    num, den = sp.fraction(sp.together(a))
    try:
        if sp.degree(den, v) < 2:
            return False
    except Exception:                              # noqa: BLE001
        return False
    terms = sp.Add.make_args(b)
    return len(terms) >= 2 and sum(1 for t in terms if sp.fraction(t)[1].has(v)) >= 2


def _integer_letters(*es) -> list:
    """letters used as whole numbers - a Fourier series' index: n in (-1)^n, \\sin(n\\pi), \\cos(2\\pi k), e^{-jk\\pi}.
    Their identities hold at whole numbers only ('\\cos(n\\pi) = (-1)^n'): compared there as well"""
    out = set()
    for e in es:
        for x in e.free_symbols:
            if str(x) not in ("n", "k", "m") or x in out:
                continue
            if any(p.base == -1 and p.exp.has(x) for p in e.atoms(sp.Pow)):
                out.add(x)
                continue
            for f in e.atoms(sp.sin, sp.cos, sp.tan, sp.exp):
                arg = sp.expand(f.args[0])
                for term in sp.Add.make_args(arg):
                    if not term.has(x):
                        continue
                    for c in (sp.pi * x, sp.I * sp.pi * x):
                        q = sp.simplify(term / c)
                        if q.is_number and q.is_rational and q != 0:
                            out.add(x)
    return sorted(out, key=str)


def _same_at_integers(a, b, d, ints, rng):
    """d = 0 at points where the index letters are whole numbers (1..8) and the rest random - True / False / None"""
    syms = sorted(d.free_symbols | a.free_symbols | b.free_symbols, key=str)
    same = differ = 0
    for _ in range(14):
        pt = {x: (rng.randint(1, 8) if x in ints else rng.uniform(0.6, 2.4)) for x in syms}
        try:
            vd, va, vb = (complex(sp.N(x.subs(pt))) for x in (d, a, b))
        except (TypeError, ValueError, ZeroDivisionError):
            continue                               # a pole at this whole number (1 / (n^2 - 1) at n = 1): another
        if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in (vd, va, vb)):
            continue
        if abs(vd) <= 1e-7 * max(1.0, abs(va), abs(vb)):
            same += 1
        else:
            differ += 1
        if same + differ >= 6:
            break
    if same + differ < 4:
        return None
    return True if differ == 0 else (False if same == 0 else None)


@functools.lru_cache(maxsize=4096)
def _same_expr(a, b, indefinite=()):
    """two expressions in letters equal? (under an indefinite integral: up to a constant) - True / False / None (cannot
    tell). Decided at random points, where sympy's simplify may not see an identity (sin^2 + cos^2)"""
    import random
    d = a - b
    try:
        if sp.simplify(d) == 0:
            return True
    except Exception:                              # noqa: BLE001
        pass
    if indefinite:
        d = sum((sp.diff(d, v) for v in indefinite), sp.Integer(0))    # the same derivative: a constant apart
    syms = sorted(d.free_symbols | a.free_symbols | b.free_symbols, key=str)
    rng, same, differ = random.Random(11), 0, 0
    real = None
    for _ in range(6):
        pt = {x: rng.uniform(0.6, 2.4) for x in syms}
        try:
            vd, va, vb = (complex(sp.N(x.subs(pt))) for x in (d, a, b))
        except (TypeError, ValueError, ZeroDivisionError):
            break
        if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in (vd, va, vb)):
            break
        if abs(vd) <= 1e-7 * max(1.0, abs(va), abs(vb)):
            same += 1
        else:
            differ += 1
    else:
        real = True if differ == 0 else (False if same == 0 else None)
    if real is True:
        return True
    ints = _integer_letters(a, b)
    if not ints:
        return real
    at = _same_at_integers(a, b, d, ints, rng)     # an index n: the identity may hold at whole n only
    if at is True:
        return True
    return False if (real is False and at is False) else None


def _expanded_wrong(a, b) -> bool:
    """two sides, one a product or power of sums and the other a sum ('(x+1)^2' and 'x^2 + x + 1', '(a+b)^2' and 'a^2 +
    b^2', '2(x + 3)' and '2x + 3'), polynomials of the same degree whose top terms have one in common: an expansion or a
    factoring written, not an equation to solve"""
    lets = sorted(_letters(a) | _letters(b), key=str)
    if not 1 <= len(lets) <= 4 or any(isinstance(t, sp.Pow) and (not t.exp.is_Integer or abs(int(t.exp)) > 6)
                                      for e in (a, b) for t in sp.preorder_traversal(e) if isinstance(t, sp.Pow)):
        return False                                 # (no expansion of '(x+1)^{99}' - nor a root's)

    def factored(e):
        return any(isinstance(t, sp.Pow) and isinstance(t.base, sp.Add) for t in sp.preorder_traversal(e)) or \
            (isinstance(e, sp.Mul) and sum(isinstance(f, sp.Add) for f in e.args) >= 1 and len(e.args) >= 2)
    try:
        if not ((factored(a) and not factored(b) and b.free_symbols) or (factored(b) and not factored(a) and a.free_symbols)):
            return False                             # (the other side a sum or one term: '(x - y)(x + y) = x^2')
        pa, pb = sp.Poly(sp.expand(a), *lets), sp.Poly(sp.expand(b), *lets)
        da, db = pa.total_degree(), pb.total_degree()
        if da != db or da < 1 or pa == pb:         # ('2(x + 3) = 2x + 3' too: linear sides with one leading term are no
            return False                             # equation - none solves it)
        top_a = {m: c for m, c in pa.terms() if sum(m) == da}
        top_b = {m: c for m, c in pb.terms() if sum(m) == db}
        return any(m in top_b and top_b[m] == c for m, c in top_a.items())
    except Exception:                              # noqa: BLE001 - not polynomials
        return False


def _identity_checks(no, s, data=None):
    """an identity's pairs of sides -> checks. data: [{letter: value}] known so far (as written, in SI units) - a pair
    whose two sides differ in letters that have values ('I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}' with R = 2,
    L = 1 written above) is compared with them put in (it was a false wrong step)"""
    out = []
    shown = s.side_texts if len(s.side_texts) == len(s.sides) else [str(x) for x in s.sides]
    for i, j in s.calc.get("pairs", []):
        a, b, ta, tb = s.sides[i], s.sides[j], shown[i], shown[j]
        ind = tuple(s.calc.get("indefinite", ()))
        same = _same_expr(a, b, ind)
        extra = a.free_symbols ^ b.free_symbols
        if ind:                                    # an indefinite integral's constant is no letter of the problem
            extra = {x for x in extra if str(x) not in ("c", "k", "c_1", "c_0", "k_1", "a_0")}
        shown_a = _pretty(str(a))
        if not same and extra and data and any(extra <= set(d) for d in data):
            for d in data:                         # the values written above put in, as written and in SI units
                if not extra <= set(d):
                    continue
                vals = {k: (_as_data(v) if isinstance(v, (int, float, complex)) else v) for k, v in d.items()
                        if k in a.free_symbols | b.free_symbols}
                got = _same_expr(a.xreplace(vals), b.xreplace(vals), ind)
                if got:
                    same, extra = True, set()
                    break
                if got is False:
                    same, extra = False, set()
                    shown_a = f"{_pretty(str(a.xreplace(vals)))} with " + ", ".join(f"{_shown(k)}={_fmt_num(v)}" for k, v in vals.items())
        if same is False and extra and not extra <= {sp.Symbol(n, real=True) for n in ("x", "y", "z")}:
            # the two sides differ in letters with no values here: values put in for them - the problem's own,
            # printed on the exam ('= \\frac{10}{s(s+2)}' for R = 2, L = 1) - no verdict (a variable dropped, as y
            # from '\\frac{\\partial}{\\partial x}(x^2 y) = 2x', is still a wrong step)
            same = None
        if same:
            out.append(Check(no, "right", f"'{_pretty(ta)} = {_pretty(tb)}'" + (" - with the other letters taken for "
                             "constants" if s.calc.get("unsure") else ""), s.conf,
                             s.conf >= SURE and not s.calc.get("unsure"), [s.text]))
        elif same is False and s.calc.get("may_be_wrong") and all(
                " " not in str(x) and "\\" not in str(x) for x in a.free_symbols | b.free_symbols if str(x) not in _D_OPAQUE):
            # (a letter whose name came out with a space or a command in it - 'omega_{d t}' - was misread by the parser)
            msg = f"the left side is {shown_a}"
            if _expanded_wrong(a, b):                # ('(x+1)^2 = x^2 + x + 1': the expansion shown)
                fac_a = any(isinstance(t, sp.Pow) and isinstance(t.base, sp.Add) for t in sp.preorder_traversal(a)) or \
                    (isinstance(a, sp.Mul) and any(isinstance(f, sp.Add) for f in a.args))
                msg = f"expanded, {_pretty(sp.latex(sp.expand(a if fac_a else b)))}"
            out.append(Check(no, "wrong step", f"'{_pretty(ta)} = {_pretty(tb)}': {msg}", s.conf, s.conf >= SURE, [s.text]))
    return out


def _milli_data(data_si: dict, units: dict):
    """the data in SI units with each value written with a lone m in milli ('L = 10m': 10 mH, as EE students write it -
    metres in data_si); None when no value has one. units: {letter: its unit as written}"""
    ms = {k for k in data_si if (units.get(k) or "").strip() == "m"}
    return {k: (v * sp.Rational(1, 1000) if k in ms else v) for k, v in data_si.items()} if ms else None


def _prefixed(side, data, data_si) -> bool:
    """does a side take a letter whose value was written with a prefix ('R = 5 k\\Omega': 5 as written, 5000 in SI)?"""
    if not data_si or not isinstance(side, sp.Basic):
        return False
    for x in side.free_symbols:
        if x in data and x in data_si:
            try:
                if abs(complex(data[x]) - complex(data_si[x])) > 1e-12 * max(1.0, abs(complex(data_si[x]))):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _unit_text(u: str) -> str:
    """a unit for a message ('k\\Omega' -> 'kΩ', '\\mu A' -> 'µA')"""
    u = re.sub(r"\\(?:mathrm|text|textrm|rm)\s*\{([^{}]*)\}", r"\1", u or "")
    return re.sub(r"\\mu\s*", "µ", re.sub(r"\\Omega", "Ω", u)).replace("\\", "").replace("{", "").replace("}", "").strip()


def _in_prefix(v_si: float, unit: str):
    """a value in SI units for a message, in the unit written with the prefix that suits it ('ps', 5.224e-8 -> '52.24 ns');
    None when the unit has no plain base ('kΩ' -> Ω, 'µA' -> A, 'ps' -> s)"""
    m = re.fullmatch(r"(f|p|n|µ|u|m|k|M|G)?(Ω|V|A|W|F|H|Hz|s|J|S|VA|var|VAR)", unit or "")
    if not m or not v_si:
        return None
    for e, p in ((9, "G"), (6, "M"), (3, "k"), (0, ""), (-3, "m"), (-6, "µ"), (-9, "n"), (-12, "p"), (-15, "f")):
        if abs(v_si) >= 10.0 ** e * 0.9995:
            return f"{_fmt(v_si / 10.0 ** e)} {p}{m.group(2)}"
    return f"{_fmt(v_si / 1e-15)} f{m.group(2)}"


def _last_factor(s):
    """a chain's last value's unit as a factor to SI ('= 25.53 mA': 0.001); None when its last side was put in SI units
    already ('R = 220 + 3180 = 3.4 k\\Omega': '(3.4) \\cdot 10^{3}', its unit kept - 3400 taken for 3.4 kΩ again: 3.4 MΩ,
    and 'I = \\frac{V}{R} = 2.94 mA' below it 1000 apart), or no unit"""
    if not s.unit or s.unit.strip() == "m":
        return None
    if len(s.side_texts) == len(s.sides) and s.side_texts and isinstance(s.side_texts[-1], str) and \
            re.fullmatch(r"\(.*\)\s*\\cdot\s*10\s*\^\s*\{?\s*[-+]?\d+\s*\}?", s.side_texts[-1].strip()):
        return None
    return _si(s.unit)


def _chain(s: Statement, data: dict, steps: list, derived: list, data_si: dict | None = None, data_m: dict | None = None,
           prec: dict | None = None, db: set | None = None):
    """check a chain's steps with the data known so far; a side with one unknown letter next to a number
    gives that letter (c^2 = 25 -> c = 5, the positive root first). data_si: the same data in SI units - a step
    holds when it holds with either ('R = 5 k\\Omega', then 'V / R = 10 / 5000' as well as '10 / 5'). data_m: the SI data
    with a lone m read as milli (_milli_data) - a step that holds only so is right too"""
    subs = [x.subs(data) for x in s.sides]
    vals = [_num(x) if not _letters(x) else None for x in subs]
    vals_si = vals
    f_last = _last_factor(s)                                                  # '= 25.53 mA': the last value in SI too
    if (data_si and data_si != data) or f_last not in (None, 1.0):
        vals_si = [_num(x) if not _letters(x) else None for x in (y.subs(data_si or data) for y in s.sides)]
        if f_last not in (None, 1.0) and vals_si[-1] is not None:
            vals_si[-1] = vals_si[-1] * f_last
    vals_m = None
    if data_m:
        vals_m = [_num(x) if not _letters(x) else None for x in (y.subs(data_m) for y in s.sides)]
        if f_last not in (None, 1.0) and vals_m[-1] is not None:
            vals_m[-1] = vals_m[-1] * f_last
    shown = s.side_texts if len(s.side_texts) == len(s.sides) else s.sides
    cvals = [_cnum(x) if not _letters(x) else None for x in subs]
    # the same with the data in SI units ('L = 10 mH': j\\omega L = j10), and with complex data by their sizes
    # ('I = 2\\angle 30^\\circ', then 'P = I^2 R = 20': I for |I|, as power formulas are written)
    others = []
    if data_si and data_si != data:
        others.append([_cnum(x) if not _letters(x) else None for x in (y.subs(data_si) for y in s.sides)])
    for d in ([data] + ([data_si] if data_si and data_si != data else [])):
        if any(isinstance(v, complex) or (isinstance(v, sp.Basic) and v.has(sp.I)) for v in d.values()):
            sizes = {k: abs(complex(v)) for k, v in d.items()}
            others.append([_cnum(x) if not _letters(x) else None for x in (y.subs(sizes) for y in s.sides)])

    if f_last not in (None, 1.0):                  # '= 2 \angle -53.13^\circ mA': the last value in SI units too - the
        #                                            sides before it in its unit, so its decimals give the tolerance
        others += [[x / f_last if x is not None else None for x in o[:-1]] + o[-1:] for o in [cvals] + others if o[-1] is not None]

    def holds(ca, cb, b):
        return abs(ca - cb) <= max(_ctol(str(b), ca), REL_TOL * abs(cb))
    si_c = [_cnum(x) if not _letters(x) else None for x in (y.subs(data_si) for y in s.sides)] if data_si and data_si != data else cvals
    for k, (a, b, ca, cb) in enumerate(zip(shown, shown[1:], cvals, cvals[1:])):   # complex numbers ('\\frac{10}{3+j4} = 1.2-j1.6')
        if ca is not None and cb is not None and (abs(ca.imag) > 1e-9 or abs(cb.imag) > 1e-9):
            ok = holds(ca, cb, b) or any(o[k] is not None and o[k + 1] is not None and holds(o[k], o[k + 1], b) for o in others)
            fa = _fmt_c(ca)
            if ok and k == len(shown) - 2 and f_last is not None and si_c[k] is not None and _prefixed(s.sides[k], data, data_si) \
                    and not holds(si_c[k] / f_last, cb, b):
                # ('Z = 3 + j4 k\Omega', then 'I = \frac{10}{Z} = 1.2 - j1.6 A': as written only - 1.2 - j1.6 mA in SI)
                fa = _UNSURE + f"right with the numbers as written only: in SI units {_fmt_c(si_c[k])}"
            if not ok:                               # '1 + j = 1.414 e^{j45}', '5 \angle 0.927': the angle's unit
                deg = _angles_step(subs[k], subs[k + 1], a, b, _cnum, lambda x, y: holds(x, y, b))
                if deg is not False:
                    ok, fa = True, (fa if deg else _UNSURE + _DEG_NOTE)
            if not ok and k == len(shown) - 2 and f_last not in (None, 1.0) and abs(ca / f_last - cb) < abs(ca - cb):
                fa = f"{_fmt_c(ca / f_last)} {_unit_text(s.unit)}"     # (in the unit written: '1.2 - j1.6 mA is not 2 - j1')
            steps.append((f"{_pretty(a)} = {_pretty(b)}", ok, fa, _fmt_c(cb)))
    db_last = False                                # the last step a ratio in dB ('A_v = 100 = 40 dB')
    orig_sides = [x.strip() for x in _split_eq(s.text)]
    for k, (a, b, va, vb, sa, sb) in enumerate(zip(shown, shown[1:], vals, vals[1:], vals_si, vals_si[1:])):
        if va is not None and vb is not None:
            vb0, sb0 = vb, sb                        # (numbers still, after a message took their place below)
            # the digits written: a rounding ('\frac{10}{3} = 3.3', '= 26 mA' for 25.53 mA) is no wrong step - a side's
            # own number only, not an expression's last ('= 2 \cdot 1.5')
            ta_, tb_ = ((t if isinstance(t, str) and _NUM_TEXT.fullmatch(re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", t.strip())) else "")
                        for t in (a, b))
            f_b = f_last if (k == len(shown) - 2 and f_last not in (None, 1.0)) else 1.0
            exs = _exp_slack(s.sides[k], data, data_si or data)   # (an exponential's argument rounded on the way)
            ok_w = _close(vb, va, tb_) or _close(va, vb, ta_) or abs(vb - va) <= exs * abs(va)
            ok_si = sa is not None and sb is not None and (_close(sb, sa, tb_, f_b) or _close(sa, sb, ta_) or
                                                           abs(sb - sa) <= exs * abs(sa))
            ok_m = vals_m is not None and vals_m[k] is not None and vals_m[k + 1] is not None and \
                (_close(vals_m[k + 1], vals_m[k], tb_, f_b) or _close(vals_m[k], vals_m[k + 1], ta_))   # ('L = 10m')
            if not (ok_w or ok_si or ok_m) and isinstance(a, str):
                rs = _round_slack(a, data_si or data, prec)   # the decimals written in it rounded ('(13.2 mA)^2 \cdot 1.13
                if rs:                                   # k\Omega = 199 mW': 196.9 with them as written)
                    hb = _half_unit(tb_)
                    ok_w = abs(vb - va) <= rs * abs(va) + hb
                    ok_si = sa is not None and sb is not None and abs(sb - sa) <= rs * abs(sa) + hb * abs(f_b)
            ok = ok_w or ok_si or ok_m
            if ok and len(orig_sides) == len(shown) and any(_BASE_NUM.search(t) or _BASE_MORE.search(t) for t in orig_sides[k:k + 2]) \
                    and all(isinstance(x, (int, float)) and abs(x - round(x)) < 1e-9 for x in (va, vb)):
                ok = abs(va - vb) < 0.5                  # '(E8D)_{16} = 3726': a number in its base is exact - 3725 (it was
                #                                          within 1 %: right)
            used_db = [x for x in (s.sides[k].free_symbols if db and isinstance(s.sides[k], sp.Basic) else ())
                       if x in db and _num(data.get(x, sp.nan)) is not None and
                       not _additive(sp.Eq(sp.Symbol("_y"), s.sides[k]), x, sp.Symbol("_y"))]
            if used_db:                                  # 'A_v = 40 dB', then 'V_o = A_v V_i = 100 \cdot 0.1': a gain in dB
                ratio = {x: 10 ** (_num(data[x]) / 20) for x in used_db}                 # as its ratio
                v_r = _num(s.sides[k].subs({**data, **ratio}))
                v_rs = _num(s.sides[k].subs({**(data_si or data), **ratio}))
                r_ok = (v_r is not None and _close(vb, v_r, tb_)) or (v_rs is not None and sb is not None and _close(sb, v_rs, tb_, f_b))
                if r_ok and not ok:                      # (it was 40 times V_i: a false wrong step)
                    ok = True
                elif ok and not r_ok:                    # ('= 40 \cdot 0.1': multiplied as if a ratio - not sure, for a human)
                    va = _UNSURE + ", ".join(map(_shown, used_db)) + " in dB, multiplied as a ratio? its ratio is 10^(x/20)"
            if ok_w and not ok_si and not ok_m and sa is not None and sb is not None and k == len(shown) - 2 and \
                    f_last is not None and _prefixed(s.sides[k], data, data_si):
                # 'R = 5 k\Omega', then 'I = \frac{10}{R} = 2 A': right only with the numbers as written, past a prefix
                # written above - with its unit written the result is 2 mA: the prefix left out? - for a human
                va = _UNSURE + f"right with the numbers as written only: in SI units {_fmt(sa)}, not {_fmt(sb)}"
            f_cm, in_cm = None, False                # the last value read in centimetres (from its unit as written,
            if not ok and k == len(shown) - 2 and f_last is not None:     # or from SI units - its unit put in them above)
                f_cm = _cm_factor(s.unit)
                f_cm = f_cm if f_cm is not None and abs(f_cm - 1.0) > 1e-9 and abs(f_cm - f_last) > 1e-9 * abs(f_last) else None
            elif not ok and k == len(shown) - 2 and (s.calc.get("si_unit") or s.calc.get("cm_unit")):
                u_ = s.calc.get("si_unit") or s.calc.get("cm_unit")
                fu, fs = _cm_factor(u_), _si(u_)
                f_cm = fu / fs if fu is not None and fs and abs(fu - fs) > 1e-9 * abs(fs) else None
                in_cm = fu is not None and abs(fu - 1.0) < 1e-9          # (written in centimetres: 'cm^2/s')
            if f_cm is not None and (_close(vb * f_cm, va, tb_, f_cm) or (sa is not None and _close(vb * f_cm, sa, tb_, f_cm))):
                # 'L_n = \sqrt{D_n \tau_n} = \sqrt{34.97 \cdot 10^{-6}} = 59.1 \mu m': worked out in centimetres, as
                # semiconductor physics keeps lengths (D in cm^2/s), the result in micrometres - it was a sure wrong
                # step (59.1 µm is 0.00591 cm); right, not sure (a slip of a power of 100 looks the same) - sure when the
                # result is written in centimetres itself ('= 1000 \cdot 26 mV = 26 cm^2/s': the mV put the rest in SI)
                ok, va = True, (va if in_cm else _UNSURE + _CM_NOTE)
            db_wrong = None
            if not ok and k == len(vals) - 2 and _is_db(s.unit) and not re.search(r"log", str(a)):   # (a side in dB already:
                #                                                  '20\log(100) = 60 dB' - 40 is not 60, no ratio)
                how = _db_match(vb, va, str(shown[-1]))
                if how:
                    ok, db_last, b = True, True, f"{b} dB"
                    va = va if how == 20 else _UNSURE + _DB10_NOTE
                elif va != 0:                        # '100 = 30 dB': 100 is 40 dB
                    db_wrong = (f"{b} dB", f"{_fmt(va)} ({_fmt(20 * math.log10(abs(va)))} dB)", f"{_fmt(vb)} dB")
            if not db_last and len(orig_sides) == len(shown) and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                # 'A_v = 40 dB = 100', 'G = 100 = 40 dB = ...': a gain in dB beside its ratio, either way round (a middle
                # side's dB was dropped: '40 = 100' a false wrong step) - judged in dB only ('26 dB = 25.9' is no rounding)
                db_a, db_b = (bool(re.search(r"\d\s*(?:\\mathrm\s*\{\s*)?d\s*B\s*\}?\s*$", t)) for t in (orig_sides[k], orig_sides[k + 1]))
                plain = (tb_ if db_a else ta_)           # (the other side a number alone: '20\log(100) = 40 dB', '15 + 20 =
                if db_a != db_b and plain:               # 35 dB' are dB worked out)
                    how = _db_match(va, vb, orig_sides[k]) if db_a else _db_match(vb, va, orig_sides[k + 1])
                    ok = bool(how)
                    if how:
                        va = va if how == 20 else _UNSURE + _DB10_NOTE
                        a, b = (f"{a} dB", b) if db_a else (a, f"{b} dB")     # (the step shown with its dB)
            f_k = f_b if f_b != 1.0 else ((_si(s.calc.get("si_unit", "")) or 1.0) if k == len(shown) - 2 else 1.0)
            src_a = orig_sides[k] if len(orig_sides) == len(shown) else a          # (as written: units put in SI above)
            plain_a = _no_exps(src_a) if isinstance(src_a, str) else src_a     # (units in an exponent cancel: '2(1 - e^{-2
            #                                                                   ms / 1 ms})' is bare numbers still)
            bare_a = isinstance(src_a, str) and not _INLINE_UNIT.search(_join_digits(plain_a)) and \
                not _INLINE_MORE.search(_join_digits(plain_a)) and _whole_si(src_a) is None and not _letters(s.sides[k])
            # (numbers written bare - not letters, whose values come in their own units: 'I_L = \frac{V_{th}}{R_{th} + R_L} =
            # 1.9 mA' with R_th worked out in ohms and R_L given in kΩ was right 'in units not written', 1000 apart)
            if not ok and f_k != 1.0 and bare_a and any(_close(vb, va * 1000.0 ** p, tb_) for p in (1, 2, -1, -2)):
                # '\frac{3 - 0.7}{7.5 + 101 \cdot 1} = 21.2 \mu A': bare numbers worked out in the units an EE student keeps
                # (kΩ, so mA), the result written in another prefix - right, not sure (a slip of 1000 looks the same);
                # ('3^2 \cdot 4k = 36 mW': 3 for 3 mA - its side put in SI units and its unit taken off: si_unit)
                ok, va = True, _UNSURE + _K_NOTE
            src_b = orig_sides[k + 1] if len(orig_sides) == len(shown) else b
            bare_b = isinstance(src_b, str) and not _INLINE_UNIT.search(_join_digits(src_b)) and \
                not _INLINE_MORE.search(_join_digits(src_b)) and _whole_si(src_b) is None and not _letters(s.sides[k + 1])
            maybe_k = not ok and (bare_a or bare_b) and va != 0 and any(_close(vb, va * 1000.0 ** p, tb_)
                                                                      for p in (0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2))
            # (or the side after it bare: '= 12 / 4 k\Omega = 3' - 3 for 3 mA, its unit left out)
            # ('\frac{1}{2\pi\sqrt{10 \cdot 1}} = 50.3 Hz': a power of 1000 apart - bare numbers in units they were not
            # written in (C in µF), or a slip of 1000: wrong, not sure - for a human; the message below)
            if not ok and isinstance(a, str) and isinstance(b, str) and re.search(r"\\cdot\s*0\.010\s*$", b) and \
                    re.search(r"(?<![\d.])100(?![\d.])", a) and _close(vb * 100, va, re.sub(r"\s*\\cdot\s*0\.010\s*$", "", b)):
                ok = True                            # 'PO = 100 e^{...} = 35\%': the 100 made it percent - 35, not 0.35
            if not ok and isinstance(a, str) and isinstance(b, str) and (_MIXED.search(a) or _MIXED.search(b)):
                # '2 \frac{1}{2}': the parser reads a mixed number (2.5); an engineer means a product (1) - either
                ra = _num(_parse(_MIXED.sub(r"\1 \\cdot ", a)).subs(data)) if _MIXED.search(a) and _parse(_MIXED.sub(r"\1 \\cdot ", a)) is not None else va
                rb = _num(_parse(_MIXED.sub(r"\1 \\cdot ", b)).subs(data)) if _MIXED.search(b) and _parse(_MIXED.sub(r"\1 \\cdot ", b)) is not None else vb
                ok = ra is not None and rb is not None and (_close(rb, ra, tb_) or _close(ra, rb, ta_))
            if not ok and isinstance(a, str) and isinstance(b, str):
                how = _binary_step(a, b)             # '1 + 1 = 10', '1111 = 15': binary, the base left unsaid
                if how:
                    ok, va = True, (va if how == "binary" else _UNSURE + _BOOL_NOTE)
            deg_msg = None
            if not ok and isinstance(src_a, str) and isinstance(src_b, str):
                # '180 - 135 = 45^\circ', '\frac{180}{3} = 60^\circ': a computation in degrees in its numbers, the result
                # marked - in degrees both (the mark made it radians: a false wrong step); not a trig function's value,
                # nor pi's ('\arctan(1) = 45^\circ', '\frac{\pi}{4} = 45^\circ': radians against radians)
                for (pa, ea, ka), (pb, eb, kb) in (((src_a, s.sides[k], va), (src_b, s.sides[k + 1], vb)),
                                                   ((src_b, s.sides[k + 1], vb), (src_a, s.sides[k], va))):
                    if _DEG_NUM.fullmatch(_join_digits(pb)) and not re.search(r"\\circ|°", pa) and not _letters(ea) and \
                            isinstance(ea, sp.Basic) and not ea.has(*_TRIG, *_ATRIG, sp.pi) and \
                            not any(str(f.func) == "Arg" for f in ea.atoms(sp.Function)):
                        if _close(math.degrees(kb), ka, re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", "", pb)):
                            ok = True
                        elif pb is src_b:                  # (wrong: the message in degrees - '45° is not 50°')
                            deg_msg = (f"{_fmt(ka)}°", f"{_fmt(math.degrees(kb))}°")
                        break
            in_rad = (k == len(shown) - 2 and re.sub(r"[\s{}\\]", "", s.unit or "") == "rad") or bool(s.calc.get("rad_arg"))
            # ('= 0.785 rad', '\cos(60 rad)': no degrees)
            if not ok and not in_rad:                # '\cos(60) = 0.5': the angle in degrees, the mark left out
                deg = _angles_step(subs[k], subs[k + 1], shown[k], shown[k + 1])
                if deg is False and f_b != 1.0 and isinstance(subs[k + 1], sp.Basic):
                    # ('2000 \cos(45) = 1.414 kW', '10 \cdot 0.002 \cdot \cos(25.84) = 18 mW': in degrees, the result in
                    # SI units - it was a sure wrong step, 1.414 against 1051 W)
                    deg = _angles_step(subs[k], subs[k + 1] * f_b, shown[k], "")
                if deg is not False:
                    ok, va = True, (va if deg else _UNSURE + _DEG_NOTE)
                elif _in_degrees(subs[k]) is not None and all(_degree_like(v) for v in _angle_args(subs[k])):
                    vd = _num(_in_degrees(subs[k]))  # wrong either way: its value in degrees too ('\cos(60) = 0.6')
                    deg_msg = (f"{_fmt(va)} ({_fmt(vd)} with the angle in degrees)"       # (not when it is the same:
                               if vd is not None and abs(vd - va) > 1e-9 * max(1.0, abs(va)) else None)   # '\cos(60°)')
                if not ok and not isinstance(deg_msg, tuple) and isinstance(src_b, str) and _DEG_NUM.fullmatch(_join_digits(src_b)) \
                        and isinstance(va, (int, float)) and isinstance(vb, (int, float)) and isinstance(s.sides[k], sp.Basic) \
                        and s.sides[k].has(*_ATRIG):
                    deg_msg = (f"{_fmt(math.degrees(va))}°", f"{_fmt(math.degrees(vb))}°")    # ('\arctan(4/3) = 50°':
                    #                                              53.13° is not 50° - not 0.9273 against 0.8727, radians)
            if not ok and sa is not None and sb is not None and abs(sb - vb) < 1e-12 * max(1.0, abs(vb)) and not _close(sa, va):
                va = f"{_fmt(va)} ({_fmt(sa)} with the data in SI units)"     # 'v_C(1 ms) = 3.68': 6.321 in SI units
            elif not ok and sa is not None and b is shown[-1] and f_last not in (None, 1.0) and \
                    not (isinstance(va, (int, float)) and abs(va - vb) <= abs(sa / f_last - vb)):   # (the reading nearer
                #                                                  the student's: '4 is not 5' in mS too)
                va = f"{_fmt(sa / f_last)} {_unit_text(s.unit)}"               # '= 30 mA' for 25.53 mA
            hint = _binary_hint(a, b) if not ok and isinstance(a, str) and isinstance(b, str) else None
            if db_wrong and not ok:
                b, va, vb = db_wrong
            elif hint and not ok and not isinstance(va, str):
                va = f"{_fmt(va)} (in binary {hint})"
            elif isinstance(deg_msg, tuple) and not ok:
                va, vb = deg_msg
            elif deg_msg and not ok and not isinstance(va, str):
                va = deg_msg
            if not ok and any(isinstance(x, str) and "/" in x for x in (src_a, a)):
                # '1/2\pi \cdot 50 \cdot 10^{-4} = 31.8': 1 over the product; '1/2 C V^2 = 125 \mu J': a half times C V^2;
                # '10 \log_{10} (2 W)/(2 mW) = 30 dB': the log of the ratio (with its units put in SI - as a's)
                cands = []                   # (a reading, how - none: 1 over the whole product, as before: sure)
                for txt in dict.fromkeys(x for x in (src_a, a) if isinstance(x, str) and "/" in x):
                    sl = _slash_reading(txt)
                    cands += ([(sl, None)] if sl else []) + _slash_alternatives(txt)
                keys = {str(x).lower(): x for x in list(data) + list(data_si or {}) if isinstance(x, sp.Symbol)}
                nearest = None               # (wrong either way: the other reading's value too, the nearer one)
                for sl, how in dict.fromkeys(cands):
                    e_sl = _parse(sl)
                    if e_sl is None:
                        continue             # (the data's letters case-folded as the page's are: 'c', 'v')
                    e_sl = e_sl.xreplace({x: keys[str(x).lower()] for x in e_sl.free_symbols if str(x).lower() in keys})
                    v_sl = _num(e_sl.subs(data))
                    v_sl_si = _num(e_sl.subs(data_si)) if data_si else None
                    if (isinstance(v_sl, (int, float)) and (_close(vb0, v_sl, tb_) or (sb0 is not None and _close(sb0, v_sl, tb_, f_b)))) or \
                            (isinstance(v_sl_si, (int, float)) and sb0 is not None and _close(sb0, v_sl_si, tb_, f_b)):
                        ok = True                    # ('\arctan(40)/(30) = 53.13^\circ': it crashed - the degrees' message was in vb)
                        if how is not None and how.startswith(_SUM_READ):     # ('9/1+2 = 3': as written 11 - right only with
                            ok = False                                         # the brackets left out: wrong, not sure)
                            va = _MAYBE + f"{_fmt(va)} is not {_fmt(vb)} - right read as {_pretty(how[1:])}: brackets left out?"
                        elif how is not None:        # ('1/2 m v^2 = 9 J': a half times m v^2 - or was it 1/(2 m v^2)? right,
                            va = _UNSURE + f"read as {_pretty(how)}"           # not sure: '\alpha = 1/2RC' worked out as
                        break                        # (1/2) R C is a slip the arithmetic cannot tell)
                    if how is not None and isinstance(v_sl, (int, float)) and v_sl and vb0:
                        d = abs(math.log(abs(v_sl / vb0))) if v_sl / vb0 > 0 else math.inf
                        if nearest is None or d < nearest[0]:
                            nearest = (d, v_sl, how)
                if not ok and nearest is not None and isinstance(va, (int, float)) and va and vb0 and \
                        (va / vb0 <= 0 or nearest[0] < abs(math.log(abs(va / vb0)))):
                    va = f"{_fmt(va)} (read as {_pretty(nearest[2].lstrip(_SUM_READ))}: {_fmt(nearest[1])})"   # ('1/2CV² = 250': 0.002 is not 250
                    #                                                  (read as (1/2)CV²: 125))
            if isinstance(s.sides[k], sp.Basic) and any(str(f.func) == "Arg" for f in s.sides[k].atoms(sp.Function)) and \
                    isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                marked = bool(re.search(r"\\circ|°", str(b)))      # '\angle Z = 53.13': its angle in degrees, the mark left out
                dva = math.degrees(va)
                if not ok and not marked and abs((vb - dva + 180) % 360 - 180) <= max(_half_unit(tb_), 0.01 * abs(dva), 0.005):
                    ok = True
                if not ok:                           # (and the message in degrees: '53.13° is not 60°')
                    va, vb = f"{_fmt(dva)}°", (f"{_fmt(math.degrees(vb))}°" if marked else vb)
            if not ok and not maybe_k and not db_wrong and k == len(shown) - 2 and f_last not in (None, 1.0) and sa is not None \
                    and isinstance(va, (int, float)) and isinstance(vb, (int, float)) and _letters(s.sides[k]) and \
                    not _close(sa / f_last, va, ""):
                # the result in a prefixed unit ('= 52 ps'), the side before it in letters whose values carry prefixes of their
                # own (pF, µA): that side in the result's unit, as it is in SI - '52240 ps is not 52 ps' (it was '0.05224 is not
                # 52': the numbers as written, in no unit)
                u_ = _unit_text(s.unit)
                va, vb = _in_prefix(sa, u_) or f"{_fmt(sa / f_last)} {u_}", f"{_fmt(vb)} {u_}"
            if not ok and k == 0 and not maybe_k and not db_wrong and isinstance(va, (int, float)) and \
                    isinstance(s.sides[0], sp.Basic) and (s.sides[0].is_Symbol or (isinstance(s.sides[0], sp.Abs) and
                                                                                s.sides[0].args[0].is_Symbol)):
                # the chain's own name with a value from above ('|V_{Tp}| = ... = 0.537 V' at t = 0, then '|V_{Tp}| = 0.4 +
                # 0.4(\sqrt{0.6 + 0.9} - \sqrt{0.6}) = 0.580 V' at t_pHL): a new value of it, in another part or at another
                # time perhaps - wrong, not sure (it was a sure wrong step; the chain's own steps are judged as they are)
                va = _MAYBE + f"{_fmt(va)} is not {_fmt(vb)} - {_pretty(a)} was {_fmt(va)} above: a new value of it here?"
            if maybe_k and not ok and not db_wrong:
                va = _MAYBE + f"{_fmt(va)} is not {_fmt(vb)} - a power of 1000 apart: numbers in units they were not written in?"
            steps.append((f"{_pretty(a)} = {_pretty(b)}", ok, va, vb))
    alt = s.calc.get("milli_alt") or {}
    for k, (a, b) in enumerate(zip(shown, shown[1:])):
        if (k in alt or k + 1 in alt) and (vals[k] is None or vals[k + 1] is None):
            # ('V = I R = 2m \cdot 4k = 9 V': the m metres or milli? - as milli 8 is not 9; with m a letter the step had no
            # verdict at all: wrong, not sure)
            ea = _parse(alt[k]) if k in alt else None
            eb = _parse(alt[k + 1]) if k + 1 in alt else None
            va2 = _num(ea.subs(data)) if ea is not None else vals[k]
            vb2 = _num(eb.subs(data)) if eb is not None else vals[k + 1]
            metres = {x: 1 for x in set().union(*(getattr(y, "free_symbols", set()) for y in s.sides[k:k + 2])) if str(x) == "m"}
            va3, vb3 = (_num(y.subs({**data, **metres})) if isinstance(y, sp.Basic) else None for y in s.sides[k:k + 2])
            sl_how = None                            # ('10/2 \sqrt{1\mu/1m} = 0.158': a half of 10 times the root, the m
            if k in alt and "/" in alt[k] and isinstance(vb2, (int, float)) and isinstance(va2, (int, float)) and \
                    not _close(vb2, va2, str(b)):    # milli - right so, not sure)
                for t2, how in _slash_alternatives(alt[k]):
                    e2 = _parse(t2)
                    v2 = _num(e2.subs(data)) if e2 is not None else None
                    if isinstance(v2, (int, float)) and _close(vb2, v2, str(b)) and not how.startswith(_SUM_READ):
                        sl_how = how
                        break
            if isinstance(va3, (int, float)) and isinstance(vb3, (int, float)) and _close(vb3, va3, str(b)):
                # ('\rho \frac{100 m}{1 mm^2} = 1.7 \Omega': metres - right so; not sure)
                steps.append((f"{_pretty(a)} = {_pretty(b)}", True, _UNSURE + "with the m read as metres", vb3))
            elif sl_how is not None:
                steps.append((f"{_pretty(a)} = {_pretty(b)}", True, _UNSURE + f"with the bare m read as milli, read as {_pretty(sl_how)}", vb2))
            elif isinstance(va2, (int, float)) and isinstance(vb2, (int, float)) and not _close(vb2, va2, str(b)):
                both = f" ({_fmt(va3)} with it as metres)" if isinstance(va3, (int, float)) and isinstance(vb3, (int, float)) else ""
                steps.append((f"{_pretty(a)} = {_pretty(b)}", False,
                              _MAYBE + f"{_fmt(va2)} is not {_fmt(vb2)}, with the bare m read as milli{both}", vb2))
    # sides still in letters once the data are in, the same letters both ('10 + (0 - 10)e^{-t/\tau} = 10(1 - e^{-1000t})'
    # with tau = 1 ms): compared at points, within 1 % - with the data as written or in SI units; right only (a
    # value the student rounded, a convention unknown here: no wrong step)
    subs_si = [y.subs(data_si) for y in s.sides] if data_si and data_si != data else subs
    for k, (a, b) in enumerate(zip(shown, shown[1:])):
        for sa_, sb_ in ((subs[k], subs[k + 1]), (subs_si[k], subs_si[k + 1])):
            la, lb = _letters(sa_), _letters(sb_)
            if not la or la != lb or (k == 0 and s.sides[0].is_Symbol) or _unworked(sa_) or _unworked(sb_) \
                    or (s.sides[k] == sa_ and s.sides[k + 1] == sb_):
                continue                           # (no data in it: an identity's business, not this)
            if _close_at_points(sa_, sb_):
                steps.append((f"{_pretty(a)} = {_pretty(b)}", True, "", ""))
                break
    last = next((v for v in reversed(vals[:-1] if db_last else vals) if v is not None), None)   # what the student
    #                                                  concluded (a ratio, not its dB)
    last_j = next((j for j in range(len(vals) - (2 if db_last else 1), -1, -1) if vals[j] is not None), None)
    last_t = s.side_texts[last_j] if last_j is not None and len(s.side_texts) == len(s.sides) and isinstance(s.side_texts[last_j], str) \
        and _NUM_TEXT.fullmatch(re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", s.side_texts[last_j].strip())) else None
    # (its digits as written: '= 10 mW' is (10)·10⁻³ - as '0.01' one digit, 8.8 mW passed for it, within half a unit)
    for i, x in enumerate(subs):
        lets = _letters(x)
        if len(lets) != 1:
            continue
        num = last if i < len(vals) - 1 or last is None else next((v for v in vals if v is not None), None)
        if num is None:
            continue
        (sym,) = lets
        if x == sp.Abs(sym):                       # '|Z| = 5': a size, not Z's value (Z may be 3 + j4)
            derived.append((x, num, _fmt(num), s))
            continue
        if x.has(sp.Abs) and sym not in x.subs(sp.Abs(sym), sp.Dummy("size")).free_symbols:
            # ('2|\Phi_F| = 0.6': the letter in its size only - that size, 0.3, for a formula in |\Phi_F| - it was 'no value
            # for Phi_F', the body effect's formula unchecked)
            d = sp.Dummy("size", positive=True)
            try:
                sizes = [_num(r) for r in sp.solve(sp.Eq(x.subs(sp.Abs(sym), d), num), d)]
            except Exception:                      # noqa: BLE001
                sizes = []
            sizes = [v for v in sizes if isinstance(v, (int, float)) and v >= 0]
            if len(sizes) == 1:
                derived.append((sp.Abs(sym), sizes[0], _fmt(sizes[0]), s))
            continue
        if x.has(sp.Abs):                          # '|x - 1| = 2': two values - none taken
            continue
        try:
            sols = [_num(r) for r in sp.solve(sp.Eq(x, num), sym)]
        except Exception:                          # noqa: BLE001
            continue
        sols = sorted((v for v in sols if v is not None), key=lambda v: (v < 0, abs(v)))
        if sols:
            derived.append((sym, sols[0], last_t if (x == sym and last_t is not None and num is last) else _fmt(sols[0]), s))


def _scales(e, x) -> set:
    """the scales of x an expression changes over: 1/|c| for each exp / sin / cos of c x + ... in it (e^{-1000t}: 1 ms)"""
    out = set()
    for f in e.atoms(sp.exp, sp.sin, sp.cos):
        try:
            c = complex(sp.N(sp.diff(f.args[0], x)))
        except (TypeError, ValueError):
            continue
        if math.isfinite(abs(c)) and abs(c) > 1e-12 and not sp.diff(f.args[0], x, 2).free_symbols - {x} and sp.diff(f.args[0], x, 2) == 0:
            out.add(1 / abs(c))
    return out


def _close_at_points(a, b) -> bool:
    """two expressions in the same letter(s) within 1 % of each other at points - on the scales they change over (a
    decay e^{-1000t} looked at around t = 1 ms, not at t = 1 s, where every decay is 0 and e^{-100t} passed for it)"""
    import random
    syms = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(5)
    ranges = {x: sorted(_scales(a, x) | _scales(b, x)) or [1.0] for x in syms}
    points = []
    for x in syms:
        for sc in ranges[x]:
            for f in (0.15, 0.4, 0.9, 1.7, 3.0):
                pt = {y: rng.uniform(0.6, 2.4) * (ranges[y][0] if y != x else 1.0) for y in syms}
                pt[x] = f * sc
                points.append(pt)
    for pt in points[:40]:
        try:
            va, vb = complex(sp.N(a.subs(pt))), complex(sp.N(b.subs(pt)))
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in (va, vb)):
            return False
        if abs(va - vb) > max(REL_TOL * max(abs(va), abs(vb)), 1e-9):
            return False
    return bool(points)


def _solve_for(eq, data, sym, complex_ok=False):
    """the values of sym that the equation gives with the data: real ones first, then complex ones (a phasor, an
    impedance, the complex roots of a characteristic equation) - sought when the equation or its data hold j, or the
    student's answer does (complex_ok); the letters are made real otherwise"""
    plain = sp.Symbol(sym.name)                    # (a real letter equal to a complex number is False at once)
    ec = eq.xreplace({sym: plain}).subs(data)
    try:                                           # the complex roots of a polynomial of degree 5 or more: slow, and
        deg = sp.Poly(ec.lhs - ec.rhs, plain).degree() if isinstance(ec, sp.Equality) else 0    # never an exam's
    except Exception:                              # noqa: BLE001  (not a polynomial in it)
        deg = 0
    if complex_ok or (isinstance(ec, sp.Basic) and ec.has(sp.I)):
        tries = [(ec, plain)] if deg <= 4 else [(eq.subs(data), sym)]
    else:                                          # real roots first; complex ones when there are none ('x^2 + 4 = 0')
        tries = [(eq.subs(data), sym)] + ([(ec, plain)] if deg <= 4 else [])
    for e, target in tries:
        try:
            sols = [_value(x) for x in sp.solve(e, target)]
        except Exception:                          # noqa: BLE001
            continue
        sols = [x for x in sols if x is not None]
        if sols:
            return [x for x in sols if not isinstance(x, complex)] + [x for x in sols if isinstance(x, complex)]
    return []


_CMD = re.compile(r"\\[A-Za-z]+")


def _digit_elsewhere(s: Statement, letter, others):
    """a statement with a lone foreign letter ('y^2+2^2 = sqrt(20)' under a^2+b^2=c^2): does another reading of
    the same piece (the other formula reader's) have a digit in that place ('4^2+2^2=sqrt(20)')? -> the statement
    with that digit (its confidence capped: a repair is a suggestion), and the digit; else (None, None)"""
    name = str(letter)
    if len(name) != 1 or not others:
        return None, None
    text = re.sub(r"\s+", "", s.text)
    parts, pos = [], 0
    for m in _CMD.finditer(text + "\\end"):           # letters inside commands (\sqrt, \pi) are not the letter
        parts.append(("plain", text[pos:m.start()]))
        parts.append(("cmd", m.group()))
        pos = m.end()
    pattern, seen = "", False
    for kind, chunk in parts[:-1]:
        if kind == "cmd":
            pattern += re.escape(chunk)
            continue
        for ch in chunk:
            if ch == name:
                pattern += r"\1" if seen else r"(?<![\d.])(\d)"
                seen = True
            else:
                pattern += re.escape(ch)
    if not seen:
        return None, None
    for other in others:
        m = re.search(pattern + r"(?![\d.])", re.sub(r"\s+", "", other))
        if m:
            digit = m.group(1)
            fixed = statements(re.sub(r"(?<!\\)(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", digit, s.text),
                               conf=min(s.conf, ALT_CONF), line=s.line, piece=s.piece)
            if len(fixed) == 1 and fixed[0].kind in ("chain", "numeric") and not set().union(*(_letters(x) for x in fixed[0].sides)):
                return fixed[0], digit
    return None, None


_PREFIX_ONLY = {"k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "\\mu": 1e-6, "n": 1e-9, "p": 1e-12}


def _additive(eq, k, sym) -> bool:
    """does the letter asked for move one for one with k ('G = G_1 + G_2': dB added) - not times another letter ('V_o =
    A_v V_i')?"""
    f = eq.lhs - eq.rhs
    try:
        d = sp.simplify(-sp.diff(f, k) / sp.diff(f, sym))
        return bool(d.is_number) and abs(abs(complex(d)) - 1) < 1e-9
    except Exception:                              # noqa: BLE001
        return False


def _lecturer_pool(values) -> dict:
    """the problem's data as the lecturer gives it once ({'R_1': '2k', 'E': '10 V', 'C': '1 \\mu F'}) -> {letter: value
    statement} in SI units, for the derivation checks: a letter the student does not write a value for takes it from
    here (the values printed on the exam). Values that are no number (a waveform) are left out"""
    out = {}
    for k, v in (values or {}).items():
        try:
            got = [x for x in statements(f"{k} = {v}") if x.kind == "value" and x.symbol is not None]
        except Exception:                          # noqa: BLE001
            continue
        if not got:
            continue
        st = got[0]
        u = re.sub(r"[\s{}]", "", st.unit or "")
        f = _si(u) if u else 1.0
        if f is None:
            f = _PREFIX_ONLY.get(u.replace("\\mu", "µ"), None)
        if f is None:
            continue
        from dataclasses import replace
        deg = re.search(r"\\circ|°", st.value_text or "")      # (an angle in degrees: in radians already - marked so)
        out[st.symbol] = replace(st, number=st.number * f, unit="", value_text=repr(st.number * f) + ("^\\circ" if deg else ""),
                                 conf=1.0)
    return out


def _multiplicities(eq, sym, sols):
    """how many times each of sols is a root of the equation (a polynomial in sym: 'r^2 + 2r + 1 = 0' has -1 twice);
    None: not a polynomial with numbers for its coefficients - no count"""
    plain = sp.Symbol(sym.name)
    try:
        num = sp.together((eq.lhs - eq.rhs).xreplace({sym: plain})).as_numer_denom()[0]
        poly = sp.Poly(num, plain)
        if poly.free_symbols - {plain} or poly.degree() < 1:
            return None
        rts = sp.roots(poly)
        if sum(rts.values()) != poly.degree():
            return None
    except Exception:                              # noqa: BLE001
        return None
    out = []
    for x in sols:
        m = sum(k for r, k in rts.items() if (v := _value(r)) is not None and abs(complex(v) - complex(x)) <= 1e-9 * max(1.0, abs(complex(x))))
        if m < 1:
            return None
        out.append(m)
    return out


def _unmatched(values, slots):
    """values written (value, its text) against the roots, each root as many times as it is one (slots): a matching,
    the earlier values first -> the indices of the values left without a root of their own"""
    owner = {}

    def take(i, seen):
        for j, x in enumerate(slots):
            if j in seen or not _close(values[i][0], x, values[i][1]):
                continue
            seen.add(j)
            if j not in owner or take(owner[j], seen):
                owner[j] = i
                return True
        return False
    return [i for i in range(len(values)) if not take(i, set())]


def _named_roots(no, formula, eq, only, roots, readings, cplx, conf, data_texts):
    """the roots of an equation named apart ('r_1 = -1', 'r_2 = -2') against its roots, with the data of each reading
    (as written, in SI units) - the reading that holds best. A root written twice that is a single root ('r_1 = -1',
    'r_2 = -1' for r^2 + 3r + 2 = 0: -2 missing) is wrong the second time -> [Check], or None: no roots"""
    tried = []
    for data, note in readings:
        sols = _solve_for(eq, data, only, cplx)
        if not sols:
            continue
        mult = _multiplicities(eq.subs(data), only, sols)
        slots = [x for x, m in zip(sols, mult) for _ in range(m)] if mult else [x for x in sols for _ in roots]
        left = set(_unmatched([(a[1], a[2]) for a in roots], slots))
        oks = [any(_close(a[1], x, a[2]) for x in sols) and k not in left for k, a in enumerate(roots)]
        tried.append((sols, note, oks, data))
    if not tried:
        return None
    sols, note, oks, data = max(tried, key=lambda t: sum(t[2]))      # (the first of equals: as written)
    si = [t for t in tried if t[1]]
    if any(oks) and not note and si and not all(any(_close(x, y) for y in si[0][0]) for x in sols):
        # right only with the numbers as written, past the prefixes written ('L = 1 mH', 'C = 0.5 mF' -> 's_1 = -1 + j'):
        # the prefixes left out on the way? - for a human
        conf = min(conf, ALT_CONF)
        note = f" (as written; in SI units {' or '.join(f'{_shown(only)} = {_fmt(x)}' for x in si[0][0])})"
    with_vals = ", ".join(f"{_shown(k)}={_fmt_num(v)}" for k, v in data.items())
    want = " or ".join(f"{_shown(only)} = {_fmt(x)}" for x in sols)
    head = f"{_pretty(formula.text)}" + (f" with {with_vals}{note}" if with_vals else "") + f": {want}"
    out = []
    for k, ((sym, value, vtext, src), ok) in enumerate(zip(roots, oks)):
        why = ""
        if not ok and any(_close(value, x, vtext) for x in sols):
            other = [x for x in sols if not any(_close(r[1], x, r[2]) for r in roots)]
            why = f" ({_shown(only)} = {_pretty(vtext)} a second time" + (f"; {' or '.join(_fmt(x) for x in other)} missing" if other else "") + ")"
        written = _pretty(src.text) if src is not None and src.kind == "value" else f"{sym} = {vtext}"
        out.append(Check(no, "right" if ok else "wrong", f"{head} - written {written}: {'right' if ok else 'wrong'}{why}",
                         conf, conf >= SURE, [formula.text] + data_texts + ([src.text] if src is not None else []), expected=sols,
                         answer=written))
    return out


def _check_problem(no, formula, stmts, alts=None, others=None, given=None, far=None, lect=None):
    """one problem: its formula and the statements under it (best readings) -> [Check]. others: {piece: [texts
    of its other readings]} - where a letter foreign to the formula is a digit in another reading; given: {letter:
    value statement} the data written just before the formula ('V = 10 V', 'I = 2 mA', then 'R = V / I'); far: the
    values of the problems above (the student's own earlier results) - a verdict resting on them is a suggestion"""
    from dataclasses import replace
    repairs = []
    checks = _check_problem_core(no, formula, stmts, alts, others, repairs, given or {}, far or {}, lect or {})
    # the formula read another way at a slash ('E = 1/2 C V^2', then 'E = 125 \mu J': a half times C V^2 - the parser takes
    # 1/(2 C V^2), as it must '1/2\pi f C': a false sure wrong; 'V_o = R_2/(R_1 + R_2) V_{in}', '20 \log(V_o)/(V_i)',
    # 's^2 + (R)/(L)s + (1)/(LC) = 0'): the reading that makes its answers right, when one does - else the one its units
    # agree with
    slash_note, slash_sure = "", False
    if "/" in formula.text and any(c.verdict in ("wrong", "wrong step") for c in checks):
        for f2, label in _slash_formulas(formula):
            rep2 = []
            more = _check_problem_core(no, f2, stmts, alts, others, rep2, given or {}, far or {}, lect or {})
            better, slash_sure = _slash_better(checks, more)
            if better:
                formula, checks, repairs, slash_note = f2, more, rep2, label
                break
    # an equation's roots written one after another ('x = 4, x = 2' under 'x^2 - 5x + 6 = 0', 's = -1', 's = -3' under
    # 's^2 + 3s + 2 = 0'): each judged - the last one only was ('x = 4' never: a wrong root missed)
    # - and an answer written again at once ('P = 13 mW', then 'P = 10 mW': the first crossed out?): the last is the
    # answer; the one before it judged too, not sure (for a human)
    vals = [s for s in stmts if s.kind == "value" and s.symbol is not None and s.symbol in _letters_of(formula)]
    where = {id(x): k for k, x in enumerate(stmts)}
    for sym in dict.fromkeys(v.symbol for v in vals):
        group = [v for v in vals if v.symbol == sym]
        if len(group) < 2 or len(group) > 4 or all(v.calc.get("pm") for v in group):
            continue
        roots = all(_roots_of(formula, group[0], v) for v in group[1:])
        again = not roots and all(where[id(b)] == where[id(a)] + 1 for a, b in zip(group, group[1:]))
        if not roots and not again:
            continue
        for v in group[:-1]:
            sub = [x for x in stmts if not (x.kind == "value" and x.symbol == sym and x is not v)]
            more = _check_problem_core(no, formula, sub, alts, others, [], given or {}, far or {}, lect or {})
            for c in more:
                if c.answer and c.verdict in ("right", "wrong") and not any(c.text == d.text for d in checks):
                    if again:
                        c.sure, c.text = False, c.text + " (answered again below - crossed out?)"
                    checks.append(c)
    # a sequence's term ('a_n = 2^n', then 'a_3 = 9'; an index n, k or i): its value there (it was no letter of the formula -
    # no verdict)
    seq = re.fullmatch(r"([A-Za-z])_\{?([nki])\}?", str(formula.sides[0])) if formula.kind == "formula" and len(formula.sides) == 2 and \
        isinstance(formula.sides[0], sp.Symbol) else None
    if seq:
        var = next((x for x in formula.sides[1].free_symbols if str(x) == seq.group(2)), None)
        for x in stmts:
            m = re.fullmatch(re.escape(seq.group(1)) + r"_\{?(\d+)\}?", str(x.symbol)) if x.kind == "value" and x.symbol is not None else None
            if var is None or m is None or not isinstance(x.number, (int, float)) or (formula.sides[1].free_symbols - {var}):
                continue
            want = _num(formula.sides[1].subs(var, int(m.group(1))))
            if want is None:
                continue
            ok = _close(x.number, want, x.value_text)
            checks.append(Check(no, "right" if ok else "wrong", f"{_pretty(formula.text)} at {seq.group(2)} = {m.group(1)}: "
                                f"{_shown(x.symbol)} = {_fmt(want)}" + (f" - written {_pretty(x.text)}: right" if ok else
                                                                       f", but written {_pretty(x.text)}: wrong"),
                                min(formula.conf, x.conf), min(formula.conf, x.conf) >= SURE, [formula.text, x.text],
                                expected=[want], answer=_pretty(x.text), line=x.line))
    # the angle of the formula's value ('\angle H(j10) = -45^\circ' under 'H(j\omega) = \frac{10}{10 + j\omega}'): judged as the
    # formula's angle there (it was no letter of the formula - no verdict)
    if formula.kind == "formula" and len(formula.sides) == 2 and isinstance(formula.sides[0], sp.Symbol):
        name = f"angle_{formula.sides[0]}"
        for x in [y for y in stmts if y.kind == "value" and str(y.symbol) == name]:
            asym = sp.Symbol(name, real=True)
            af = Statement("formula", f"\\angle {formula.side_texts[0] if formula.side_texts else formula.sides[0]} = arg("
                                      f"{formula.side_texts[-1] if formula.side_texts else formula.sides[-1]})",
                           [asym, sp.arg(formula.sides[-1])], formula.conf, line=formula.line, piece=formula.piece,
                           side_texts=[f"\\angle {formula.side_texts[0] if formula.side_texts else ''}",
                                       f"\\arg({formula.side_texts[-1] if formula.side_texts else ''})"])
            sub = [y for y in stmts if y.kind == "value" and y is not x and not str(y.symbol).startswith("angle_")] + \
                [replace(x, symbol=asym)]
            checks += [c for c in _check_problem_core(no, af, sub, alts, others, [], given or {}, far or {}, lect or {})
                       if c.answer and c.verdict in ("right", "wrong")]
    # a response at several frequencies ('|H(j100)| = 0.707', '|H(j1000)| = 0.995'): each judged at its own (the last one
    # only was)
    at = [x for x in stmts if x.calc.get("at_w") is not None and x.kind in ("value", "chain", "numeric")]
    if len({x.calc["at_w"] for x in at}) >= 2:
        for x in at[:-1]:
            sub = [y for y in stmts if y is x or y not in at]
            more = _check_problem_core(no, formula, sub, alts, others, [], given or {}, far or {}, lect or {})
            checks += [c for c in more if c.answer and c.verdict in ("right", "wrong") and not any(c.text == d.text for d in checks)]
    if repairs:
        for c in checks:
            c.text += " (" + "; ".join(repairs) + ")"
    if slash_note:                                   # (right so, not sure - unless its units agree only so: '\alpha = 1/2RC',
        for c in checks:                             # '\alpha = 5 \cdot 10^{-4}' is (1/2) R C - a slip the arithmetic cannot tell)
            _slash_mark(c, slash_note, slash_sure)
    return checks


def _slash_mark(c, label: str, sure: bool = False):
    """a check judged with a slash read another way (_slash_alternatives): the reading said; right so - not sure (unless
    sure: its units agree only so), or wrong not sure when the reading takes brackets the student left out ('V/R_1+R_2'
    as V/(R_1+R_2): as written it does not hold)"""
    sums = label.startswith(_SUM_READ)
    if c.verdict == "right" and sums and c.text.rfind(": right") >= 0:
        k = c.text.rfind(": right")
        c.text = c.text[:k] + ": as written it does not hold - right" + c.text[k + len(": right"):]
    c.text += f" (read as {_pretty(label.lstrip(_SUM_READ))}" + ("; brackets left out?)" if sums else ")")
    if c.verdict == "right" and sums:
        c.verdict, c.sure, c.confidence = "wrong", False, min(c.confidence, ALT_CONF)
    elif c.verdict == "right" and not sure:
        c.sure, c.confidence = False, min(c.confidence, ALT_CONF)


def _slash_formulas(formula) -> list:
    """a formula of letters read another way at its slashes (_slash_alternatives): [(its statement so, how)]"""
    texts = formula.side_texts
    if formula.kind != "formula" or not texts or len(texts) != len(formula.sides):
        return []
    out = []
    for k, t in enumerate(texts):
        if not isinstance(t, str) or "/" not in t:
            continue
        for alt, label in _slash_alternatives(t):
            e = _parse(alt)
            if e is None or not isinstance(formula.sides[k], sp.Basic):
                continue
            try:
                if sp.simplify(e - formula.sides[k]) == 0:
                    continue                         # (the parser's own reading)
            except Exception:                      # noqa: BLE001
                continue
            new = " = ".join(alt if i == k else x for i, x in enumerate(texts))
            got = [x for x in statements(new, formula.conf, formula.line, formula.piece) if x.kind == "formula"]
            if len(got) != 1 or len(got[0].sides) != len(formula.sides):
                continue
            names = {str(x).lower(): x for x in _letters_of(formula)}     # (the page's letters case-folded: 'e', 'c')
            g = got[0]
            rep = {x: names[str(x).lower()] for x in _letters_of(g) if str(x).lower() in names}
            g.sides = [x.xreplace(rep) if isinstance(x, sp.Basic) else x for x in g.sides]
            if _letters_of(g) == _letters_of(formula):
                out.append((g, label))
    return out[:4]


def _slash_better(base: list, more: list) -> tuple:
    """the checks of a formula read another way (_slash_formulas) against its first reading's -> (better: fewer wrongs, an
    answer wrong before right now and none right before wrong now - or wrong both ways, its units agreeing only so; its
    units agree only so)"""
    bad = ("wrong", "wrong step")
    right_before = {c.answer for c in base if c.verdict == "right" and c.answer}
    wrong_before = {c.answer for c in base if c.verdict == "wrong" and c.answer}
    units = any("its unit:" in c.text for c in base if c.verdict == "wrong") and not any("its unit:" in c.text for c in more)
    if any(c.verdict in bad and c.answer in right_before for c in more):
        return False, False
    n_base, n_more = sum(c.verdict in bad for c in base), sum(c.verdict in bad for c in more)
    if n_more < n_base and any(c.verdict == "right" and c.answer in wrong_before for c in more) and \
            not any(c.verdict == "wrong" and c.answer for c in more):      # (every answer so: one root of two is no reading)
        return True, units
    return n_more == n_base and units, units


_EXPRESSION = re.compile(r"\\(?:frac|cdot|times|div|sqrt)|[+*/^(]|\d\s*-")


def _redone(spans: list, steps: list, stmts: list):
    """a computation worked out again just below itself and right the second time ('I = \\frac{V}{R} = 6.5 mA', then 'I =
    \\frac{V}{R} = 5 mA'): the first one's wrong steps not sure - crossed out? (the reader does not see strokes). spans:
    [(chain, its first step's index, past its last)] in the order written"""
    where = {id(x): k for k, x in enumerate(stmts)}
    for (a, sa, ea), (b, sb, eb) in zip(spans, spans[1:]):
        if not (len(a.sides) > 2 and len(b.sides) > 2 and isinstance(a.sides[0], sp.Symbol) and a.sides[0] == b.sides[0] and
                a.sides[1] == b.sides[1]) or id(a) not in where or id(b) not in where:
            continue
        between = stmts[where[id(a)] + 1: where[id(b)]]
        if any(x.kind != "value" for x in between) or eb == sb or not all(ok for _, ok, _, _ in steps[sb:eb]):
            continue
        for k in range(sa, ea):
            t, ok, va, vb = steps[k]
            if not ok and not (isinstance(va, str) and va.startswith(_MAYBE)):
                steps[k] = (t, ok, _MAYBE + f"{va if isinstance(va, str) else _fmt(va)} is not "
                                          f"{vb if isinstance(vb, str) else _fmt(vb)} - worked out again below, right (crossed out?)", vb)


def _rounding_moves(eq, sym, data: dict, texts: dict, fac=None, f_ans=None) -> float:
    """how far the formula's value for sym moves, summed, when each value used moves by half a unit in its last digit
    as written (texts: {letter: its value as written}; a lone digit is exact) - in SI units when fac and f_ans are
    given (as the answer was worked out); 0 when it cannot be told"""
    def solve(d):
        if fac is not None and f_ans:
            return [x / f_ans for x in _solve_for(eq, {k: v * (fac.get(k) or 1) for k, v in d.items()}, sym)]
        return _solve_for(eq, d, sym)
    try:
        base = [x for x in solve(data) if isinstance(x, (int, float))]
        if not base:
            return 0.0
        total = 0.0
        for k, t in texts.items():
            h = _half_unit(t)
            v = _num(data[k]) if k in data else None
            if not h or not isinstance(v, (int, float)):
                continue
            moved = [x for x in solve({**data, k: v + h}) if isinstance(x, (int, float))]
            if not moved:
                return 0.0
            total += min(abs(x - y) for x in moved for y in base)
        return total
    except Exception:                              # noqa: BLE001
        return 0.0


def _stmt_values(d: dict) -> tuple:
    """{letter: value statement} -> ({letter: value as written}, {letter: value in SI units}, {letter: its unit})"""
    w, si, units = {}, {}, {}
    for k, x in (d or {}).items():
        num = getattr(x, "number", None)
        if num is None or not isinstance(k, sp.Basic):
            continue
        u = getattr(x, "unit", "") or ""
        f = _si(u) if u.strip() and u.strip() != "m" else None
        w[k], si[k], units[k] = num, num * (f or 1), u
    return w, si, units


def _foreign_steps(s, steps: list, foreign: dict, lect_vals: dict, given, far, answers):
    """a computation under a formula, in letters the formula does not have ('f_c = \\frac{1}{2\\pi \\tau} = 207 Hz' under
    '\\tau = R C' and its work, 'P = I^2 R = 127 mW' under 'I = \\frac{V}{|Z|}'): its steps checked with the values known on
    the page - this problem's, the data written before it, values such computations above worked out, the student's
    earlier results (it was checked with none: no verdict - a slip missed); a step wrong only with an earlier
    problem's values is not sure (another part's letters?). Its own value: for such computations below only"""
    g_w, g_si, g_u = _stmt_values(given)
    f_w, f_si, f_u = _stmt_values(far)
    a_w = {sym: v for sym, v, _, _ in answers}
    a_si = {sym: v * ((_si(src.unit) if src is not None and src.kind == "value" else None) or 1) for sym, v, _, src in answers}
    a_u = {sym: src.unit for sym, _, _, src in answers if src is not None and src.kind == "value"}
    o_w, o_si = {k: v[0] for k, v in foreign.items()}, {k: v[1] for k, v in foreign.items()}
    sure_w, sure_si = {**lect_vals, **g_w, **o_w, **a_w}, {**lect_vals, **g_si, **o_si, **a_si}
    all_w, all_si = {**lect_vals, **f_w, **sure_w}, {**lect_vals, **f_si, **sure_si}
    units = {**f_u, **g_u, **a_u}
    st_sure, st_all, derived = [], [], []
    _chain(s, sure_w, st_sure, [], sure_si, _milli_data(sure_si, units))
    _chain(s, all_w, st_all, derived, all_si, _milli_data(all_si, units))
    judged = {t for t, *_ in st_sure}
    for t, ok, va, vb in st_all:
        if not ok and t not in judged and not (isinstance(va, str) and va.startswith(_MAYBE)):
            va = _MAYBE + f"{va if isinstance(va, str) else _fmt(va)} is not {vb if isinstance(vb, str) else _fmt(vb)} - with " \
                          f"values of the work above"
        steps.append((t, ok, va, vb))
    f_d = _last_factor(s)
    for sym, v, _, _ in derived:
        if isinstance(sym, (sp.Symbol, sp.Abs)) and s.sides and s.sides[0] == sym and isinstance(v, (int, float, complex)):
            foreign[sym] = (v, v * (f_d or 1))


def _check_problem_core(no, formula, stmts, alts, others, repairs, given=None, far=None, lect=None):
    checks = []
    eqs = [formula] + [s for s in stmts if s.kind == "formula" and _pure_equation(s)] if _pure_equation(formula) else []
    if len(eqs) >= 2:
        got = _check_system(no, eqs, stmts, given, {**(lect or {}), **(far or {})})
        if got:
            return got
        formula, stmts = eqs[-1], [s for s in stmts if s not in eqs]     # no single solution: the last one alone
    eq = sp.Eq(formula.sides[0], formula.sides[-1])
    if isinstance(eq, sp.logic.boolalg.BooleanAtom):  # 'x^2 + 4 = 0' with x real: False at once - its roots are complex
        eq = sp.Eq(formula.sides[0], formula.sides[-1], evaluate=False)
    letters = _letters(eq)
    data, steps, answers, spans, bnames = {}, [], [], [], {}
    foreign = {}                                   # {letter: (value, in SI)} computations foreign to the formula worked out
    lect_vals = {k: v.number for k, v in (lect or {}).items()}         # the lecturer's data: under the student's own
    for s in stmts:
        if s.kind == "value":
            answers.append((s.symbol, s.number, s.value_text, s))
        elif s.kind == "boolean":
            checks += _boolean_checks(no, s, bnames)
        elif s.kind == "identity":
            known = {**lect_vals, **{sym: v for sym, v, _, _ in answers}}
            checks += _identity_checks(no, s, [known, {**lect_vals, **{sym: v * ((_si(src.unit) if src is not None and src.kind == "value" else None) or 1)
                                                                       for sym, v, _, src in answers}}])
        elif s.kind == "phasor":
            checks += _phasor_checks(no, s)
        elif s.kind == "matrix":
            checks += _matrix_checks(no, s)
        elif s.kind == "unit":
            checks.append(Check(no, "wrong unit", s.calc["msg"], s.conf, False, [s.text], line=s.line))
        elif s.kind in ("chain", "numeric"):
            own = set().union(*(_letters(x) for x in s.sides))
            extra = own - letters
            if extra and s.sides[0].is_Symbol and not any(_letters(x) for x in s.sides[1:]):
                extra = set()                        # 'R = 1 + 1 = 2', 'I = 12 / 2k = 6 mA' under 'R_2 \| R_3 = 1 k\Omega':
                #                                      a new quantity worked out in numbers - its steps checked
            if extra and len(own) == 1:              # a lone foreign letter in a substitution: a misread digit?
                fixed, digit = _digit_elsewhere(s, next(iter(extra)), (others or {}).get(s.piece))
                if fixed is not None:
                    repairs.append(f"{next(iter(extra))} taken as {digit}, as another reading of that piece has it")
                    stmts = [fixed if x is s else x for x in stmts]
                    s, extra = fixed, set()
            if extra and len(own) > 1 and s.sides[0].is_Symbol and s.sides[0] in letters and s.sides[-1] is not None and \
                    not _letters(s.sides[-1]):
                extra = set()                        # 'I = \frac{V}{R} = \frac{12 V}{4.7 k\Omega} = 2.55 mA' under 'P = I^2 R':
                #                                      a letter of the formula worked out by another formula, in numbers
                #                                      at last - its steps checked, its value the formula's (it was 'no
                #                                      value for I')
            if extra:
                if len(own) == 1:
                    checks.append(Check(no, "misread?", f"'{_pretty(s.text)}': {next(iter(extra))} is not in "
                                        f"{_pretty(formula.text)} - a misread digit?", min(s.conf, formula.conf), False,
                                        [formula.text, s.text]))
                elif not s.sides[0].is_Symbol or s.sides[0] not in letters:
                    # another computation in the formula's lines ('V_{out} = V \frac{R_2}{R + R_2} = 12 \cdot \frac{3.3k}{10k
                    # + 3.3k} = 3.87 V' under 'P = I^2 R'): its steps checked (it was skipped - no verdict), its value none
                    # of the formula's
                    k0 = len(steps)
                    _foreign_steps(s, steps, foreign, lect_vals, given, far, answers)
                    spans.append((s, k0, len(steps)))
                continue
            derived = []
            k0 = len(steps)
            latest = {a[0]: a[3] for a in answers}
            first = s.sides[0] if isinstance(s.sides[0], (sp.Symbol, sp.Abs)) else None     # (a size: '|H(j1000)| = 0.995'
            #                                                  after '|H(j100)| = 0.707' - another frequency, not 0.707)
            # ('P = (2 mA)^2 \cdot 2.2 k\Omega = 10 mW', then '... = 8.8 mW': worked out again - its own value, not judged
            # against the one above: a false wrong step, and the formula's answer the first)
            redo = first is not None and latest.get(first) is not None and latest[first].kind in ("chain", "numeric")
            ans = [a for a in answers if not (redo and a[0] == first)]
            d_si = {**lect_vals, **{sym: v * ((_si(src.unit) if src is not None and src.kind == "value" else None) or 1)
                                    for sym, v, _, src in ans}}
            _chain(s, {**lect_vals, **{sym: v for sym, v, _, _ in ans}}, steps, derived, d_si,
                   _milli_data(d_si, {sym: src.unit for sym, _, _, src in ans if src is not None and src.kind == "value"}))
            spans.append((s, k0, len(steps)))
            answers += derived                       # in the order written: a later value overrides it
    _redone(spans, steps, stmts)
    # 'I = \frac{12}{4}', then 'I = 3 A': a computation written a line at a time - its step
    vals = [s for s in stmts if s.kind == "value"]
    for a, b in zip(vals, vals[1:]):
        if a.symbol == b.symbol and _EXPRESSION.search(a.value_text.strip().lstrip("+-")) and \
                not _EXPRESSION.search(b.value_text) and not (a.calc.get("pm") and b.calc.get("pm")):   # (not 'x = \pm 4')
            steps.append((f"{_pretty(a.value_text)} = {_pretty(b.value_text)}",
                          _close(b.number, a.number, b.value_text) or _close(a.number, b.number), a.number, b.number))
    # 'a^2+b^2=c^2' and '4^2+2^2 = \sqrt{20}': a false step whose right side is the root of its left side
    root_pairs = []
    for i, (txt, ok, va, vb) in enumerate(steps):
        if ok:
            continue
        for p in (x for x in formula.sides if isinstance(x, sp.Pow) and x.base.is_Symbol and x.exp.is_number):
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va > 0 and _close(vb, va ** (1 / float(p.exp))):
                root_pairs.append((i, p, va, vb))
    if root_pairs and not any(a[0] in letters for a in answers):
        # the student put his numbers straight into the formula ('4^2+2^2'): its other letters are his numbers,
        # and his value is the root of their sum - right for them, written in a false equality
        i, p, va, vb = root_pairs[-1]
        conf = min(formula.conf, min((s.conf for s in stmts), default=1.0))
        checks.append(Check(no, "right value, wrong equality",
                            f"'{steps[i][0]}': the left side is {_fmt(va)}, not {_fmt(vb)}; {_fmt(va)} is {_pretty(p)} - so "
                            f"{p.base} = {steps[i][0].split(' = ')[-1]} is right, but the equality as written is not",
                            conf, conf >= SURE, [formula.text] + [s.text for s in stmts], expected=[vb],
                            answer=f"{p.base} = {_fmt(vb)}", steps=[(t, ok) for t, ok, _, _ in steps]))
        return checks
    # an equation's roots named apart: 's_1 = -1', 's_2 = -2' - its one letter, or the one its data written before leave
    # open ('R = 2', 'L = 1', 'C = 0.5', then 's^2 + \frac{R}{L}s + \frac{1}{LC} = 0', 's_1 = ...')
    known = {k: v for k, v in (given or {}).items() if k in letters and v.kind == "value"}
    open_ = letters - set(known) - {a[0] for a in answers}
    if len(open_) == 1 and (len(letters) == 1 or (known and not any(a[0] in letters for a in answers))):
        (only,) = open_
        roots = [a for a in answers if isinstance(a[0], sp.Symbol) and re.fullmatch(re.escape(str(only)) + r"_\{?\d\}?", str(a[0]))]
        if roots and not any(a[0] == only for a in answers):
            cplx = any(isinstance(a[1], complex) for a in roots)
            readings = [({k: _as_data(v.number) for k, v in known.items()}, "")]
            fac = {k: _si(v.unit) for k, v in known.items()}
            if any(f not in (None, 1) for f in fac.values()) and all(f is not None or not known[k].unit.strip() for k, f in fac.items()):
                readings.append(({k: _as_data(v.number * (fac[k] or 1)) for k, v in known.items()}, " (in SI units)"))
            got = _named_roots(no, formula, eq, only, roots, readings, cplx,
                               min([formula.conf] + [a[3].conf for a in roots if a[3] is not None] + [v.conf for v in known.values()]),
                               [v.text for v in known.values()])
            if got:
                return checks + got
    mine = [a for a in answers if a[0] in letters]
    size = False                                     # '|Z| = 5' under 'Z = R + j\omega L': the size of what it gives,
    sizes = [k for k, a in enumerate(answers)        # when written last, for a letter with no value of its own
             if isinstance(a[0], sp.Abs) and a[0].args[0] in letters and not any(m[0] == a[0].args[0] for m in mine)]
    if sizes and sizes[-1] > max((k for k, a in enumerate(answers) if a[0] in letters), default=-1):
        a = answers[sizes[-1]]
        mine, size = mine + [(a[0].args[0], a[1], a[2], a[3])], True
    if not mine and formula.kind == "chain" and (isinstance(formula.sides[0], sp.Abs) or str(getattr(formula.sides[0], "func", "")) in
                                                 ("Re", "Im", "Arg")) and formula.sides[0].args and formula.sides[0].args[0] in (given or {}):
        x = formula.sides[0].args[0]                 # '|S| = 50' after 'S = 30 + j40': the size of a value written above
        g = given[x]
        _chain(formula, {x: _as_data(g.number)}, steps, [], {x: _as_data(g.number) * (_si(g.unit) or 1)})
    from_above = False
    if not mine and not steps and given and formula.kind == "formula" and len(formula.sides) == 2 and \
            isinstance(formula.sides[0], sp.Symbol) and formula.sides[0] in given and letters <= set(given) and \
            formula.sides[0] not in _letters(formula.sides[1]) and not isinstance(given[formula.sides[0]].number, complex):
        # the values written first, the formula after them ('V_{rms} = 10 V', 'V_p = 14.1 V', then 'V_p = \sqrt{2} V_{rms}'):
        # the letter it gives judged by the others (it was no verdict); wrong not sure - another part's value?
        g = given[formula.sides[0]]
        mine, from_above = [(formula.sides[0], g.number, g.value_text, g)], True
    if not mine:                                     # nothing answered for the formula: the arithmetic under it
        for txt, ok, va, vb in steps:
            checks.append(_step_check(no, txt, ok, va, vb, formula.conf, [formula.text]))
        return checks
    sym, value, vtext, src = mine[-1]
    if size and sym in letters and sym not in (eq.lhs - eq.rhs).subs(sp.Abs(sym), sp.Dummy("size")).free_symbols:
        eq = eq.xreplace({sp.Abs(sym): sym})         # '|I| = \frac{|V|}{|Z|}', then '|I| = 5 mA': solved for the size (it
        #                                              'cannot be solved' for I itself)
    for a in mine[:-1]:
        if a[0] != sym:
            data[a[0]] = _as_data(a[1])
    size_units = {}                                  # '|S| = 1 kVA' for a formula in |S| only ('P = |S| \cos\phi'): the
    for a in answers:                                # size is the value it needs - it was 'no value for S'
        x = a[0].args[0] if isinstance(a[0], sp.Abs) and a[0].args else None
        if x is None or x not in letters or x == sym or x in data or isinstance(a[1], complex) or not isinstance(a[1], (int, float)):
            continue
        if x in (eq.lhs - eq.rhs).subs(sp.Abs(x), sp.Dummy("size")).free_symbols:
            continue                                 # (Z itself in it too: 'I = V / Z' - its size no value for it)
        data[x] = _as_data(abs(a[1]))
        size_units[x] = a[3].unit if a[3] is not None and not a[3].calc.get("si") else ""
    def_fac, def_used = {}, []                       # 'I = \frac{V}{R}' written under 'P = I^2 R', V and R given: I worked
    vals_all = {a[0]: a for a in answers if isinstance(a[0], sp.Symbol)}     # out of them for the formula (it was 'no
    for d in stmts:                                  # value for I'); in SI units too (its factor: as a unit's)
        if d.kind != "formula" or len(d.sides) != 2 or not d.sides[0].is_Symbol:
            continue
        x, rhs = d.sides[0], d.sides[1]
        need = _letters(rhs)
        if x not in letters or x == sym or x in data or x in need or not need or not need <= set(vals_all):
            continue
        dw = {k: _as_data(vals_all[k][1]) for k in need}
        ds = {k: _as_data(vals_all[k][1]) * ((_si(vals_all[k][3].unit) if vals_all[k][3] is not None and
                                              vals_all[k][3].kind == "value" and vals_all[k][3].unit.strip() != "m" else None) or 1)
              for k in need}
        vw, vs = _num(rhs.subs(dw)), _num(rhs.subs(ds))
        if vw is None or vs is None or vw == 0:
            continue
        data[x], def_fac[x] = _as_data(vw), vs / vw
        def_used += [d] + [vals_all[k][3] for k in need if vals_all[k][3] is not None]
    used = [formula] + [a[3] for a in mine if a[3] is not None and (a[0] in data or a is mine[-1])]
    used += [d for d in def_used if d not in used]
    found = []                 # a missing value in a piece of the formula's line the layout called text ('|r=5')
    missing = letters - set(data) - {sym}
    for v in (v for ln in (formula.line, formula.line + 1) for v in (alts or {}).get(ln, [])):
        if v.symbol in missing and v.symbol not in data:
            data[v.symbol] = _as_data(v.number)
            found.append(v)
    missing -= set(data)
    at_w = src.calc.get("at_w") if src is not None else None
    w_sym = next((x for x in missing if str(x) in ("omega", "w")), None)
    s_sym = next((x for x in missing if str(x) == "s"), None)
    if at_w is not None and w_sym is not None:       # 'H(j\omega) = \frac{10}{10 + j\omega}', then '|H(j10)| = 0.707': at
        data[w_sym] = _as_data(at_w)                 # omega = 10 (it was 'no value for omega')
        missing.discard(w_sym)
    elif at_w is not None and s_sym is not None and len(missing) == 1:     # ('H(s) = \frac{10}{s + 10}': at s = j10)
        data[s_sym] = sp.I * _as_data(at_w)
        missing.discard(s_sym)
    from_given, used_far = {}, False
    for pool, is_far in ((given or {}, False), (far or {}, True), (lect or {}, False)):   # the data written above: just
    #                                                  before, then earlier; then the lecturer's (the exam's own values)
        for v in sorted(missing & set(pool), key=str):
            data[v] = _as_data(pool[v].number)
            used.append(pool[v])
            from_given[v] = pool[v]
            used_far = used_far or is_far
        missing -= set(data)
    used += found
    conf = min([x.conf for x in used] + ([ALT_CONF] if found else []))
    name = _pretty(src.text.split("=")[0]) if src is not None and src.kind == "value" else _shown(sym)
    name = name or str(sym)
    if size:
        name = next((_pretty(t) for t in (src.side_texts if src is not None else []) if "vert" in t or "|" in t), f"|{sym}|")
    answer = f"{name} = {vtext}" if src is None or src.kind != "value" else _pretty(src.text)
    if src is not None and src.kind in ("chain", "numeric") and src.side_texts and vtext in src.side_texts:
        answer = f"{name} = {_pretty(_split_eq(src.text)[src.side_texts.index(vtext)].strip())}"   # (as written: '10 mW')
    step_list = [(t, ok) for t, ok, _, _ in steps]
    if missing and sym is not None:
        # partial fractions with their coefficients found ('\frac{s+3}{(s+1)(s+2)} = \frac{A}{s+1} + \frac{B}{s+2}', A = 2,
        # B = -1): the expansion holds for every s, or it does not
        both = [x.subs({**data, sym: value}) for x in formula.sides]
        if len(missing) == 1 and _partial_fractions(both):
            same = _same_expr(both[0], both[1])
            if same is not None:
                with_vals = ", ".join(f"{_shown(k)}={_fmt_num(v)}" for k, v in {**data, sym: value}.items())
                checks.append(Check(no, "right" if same else "wrong", f"{_pretty(formula.text)} with {with_vals}: "
                                    + ("it holds for every " if same else "it does not hold for every ") + str(next(iter(missing))),
                                    conf, conf >= SURE, [x.text for x in used], answer=answer, steps=step_list))
                return checks
    if missing:
        for txt, ok, va, vb in steps:                # the arithmetic still checks
            checks.append(_step_check(no, txt, ok, va, vb, conf, [x.text for x in used]))
        checks.append(Check(no, "cannot check", f"{_pretty(formula.text)}: no value for {', '.join(sorted(map(_shown, missing)))} "
                            f"- cannot check {answer}", conf, False, [x.text for x in used], answer=answer, steps=step_list))
        return checks
    sols = _solve_for(eq, data, sym, isinstance(value, complex) or size)
    f_si = _si(src.unit) if formula.calc.get("si") and src is not None and src.kind == "value" and src.unit.strip() not in ("", "m") else None
    if sols and not data and f_si not in (None, 1.0):
        # '10 V = I \cdot 2 k\Omega + 4 V' (its numbers put in SI units), then 'I = 3 mA': the answer in SI too (3 is 0.003)
        sols = [x / f_si for x in sols]
    unit_of = {a[0]: a[3].unit for a in mine[:-1] if a[3] is not None and a[3].kind == "value"}
    unit_of.update({v.symbol: v.unit for v in found})
    unit_of.update({k: v.unit for k, v in from_given.items()})
    unit_of.update(size_units)
    fac = {k: _si(unit_of.get(k, "")) for k in data}
    fac.update(def_fac)
    f_ans = _si(src.unit) if src is not None and src.kind == "value" and src.unit.strip() != "m" else None   # a lone 'm':
    #                                                         metres, or milli of what goes unsaid ('22 m' for 22 ms) - as written
    alt, prefixed = [], any(f not in (None, 1) for f in fac.values())
    si_branch = False
    if sols and f_ans is not None and data and all(f is not None for f in fac.values()) and \
            (f_ans != 1 or any(f != 1 for f in fac.values())):
        # every value written with its unit: worked out in SI units and given in the answer's ('R = 5000 Ω' for
        # 10 V / 2 mA is as right as '5 kΩ'; '5 Ω' is not) - the numbers as written would take 2 mA for 2 A
        sols_si = _solve_for(eq, {k: v * fac[k] for k, v in data.items()}, sym, isinstance(value, complex) or size)
        if sols_si:
            sols = [x / f_ans for x in sols_si]
            si_branch = True
    elif sols and data and all(f is not None or not unit_of.get(k, "").strip() for k, f in fac.items()):
        # some values (or all) written bare: in SI units ('T = 2\pi/\omega', omega = 1000 -> 'T = 6.28 ms'), or in
        # the student's own (I = 2, R = 5 -> 'V = 10 mV') - either reading may be his
        a_unit = src.unit.strip() if src is not None and src.kind == "value" else ""
        f_a = f_ans if a_unit else 1.0
        if f_a is not None and (f_a != 1 or prefixed):
            sols_b = _solve_for(eq, {k: v * (fac[k] or 1) for k, v in data.items()}, sym, isinstance(value, complex) or size)
            alt = [x / f_a for x in sols_b]
    data_m = _milli_data({k: v * (fac[k] or 1) for k, v in data.items()}, unit_of)
    a_unit_m = src.unit.strip() if src is not None and src.kind == "value" else ""
    f_am = f_ans if a_unit_m else 1.0
    sols_m = []
    if sols and data_m and f_am is not None:         # 'L = 10m': milli as EE students write it (metres in SI) - either
        sols_m = [x / f_am for x in _solve_for(eq, data_m, sym, isinstance(value, complex) or size)]    # reading his
        alt = alt + sols_m
    if size:                                         # the sizes of what the formula gives
        sols = sorted({round(abs(x), 12) for x in sols})
        alt = sorted({round(abs(x), 12) for x in alt})
    if not sols:
        checks.append(Check(no, "cannot check", f"{_pretty(formula.text)} cannot be solved for {name}", conf, False,
                            [x.text for x in used], answer=answer, steps=step_list))
        return checks
    pos = (_positive(sols) or sols) if data else sols      # an equation's roots: all of them
    with_vals = ", ".join(f"{_shown(k)}={_fmt_num(v)}" for k, v in data.items())
    if found:
        with_vals += " (" + ", ".join(_pretty(v.text) for v in found) + " read from a piece taken for text)"
    if used_far:
        with_vals += " (some from the work above)"
        conf = min(conf, ALT_CONF)
    exs = max((_exp_slack(x, data, {k: v * (fac[k] or 1) for k, v in data.items()}) for x in formula.sides), default=0.0)   # 'I_D =
    #                                                  I_S e^{V_D/V_T}': the exponent

    def near(x):                                     # rounded on the way - else the digits written, or 1 %
        return _close(value, x, vtext) or (exs > 0 and not isinstance(value, complex) and not isinstance(x, complex)
                                           and abs(value - x) <= exs * abs(x))
    ok = any(near(x) for x in sols)
    if ok and isinstance(value, (int, float)) and value < 0 and any(isinstance(x, (int, float)) and x > 0 and near(-x) for x in sols):
        try:                                         # 'c = -5' for 'a^2 + b^2 = c^2': the formula leaves c's sign free -
            free = sp.simplify((eq.lhs - eq.rhs).subs(sym, -sym) - (eq.lhs - eq.rhs)) == 0     # the negative root of a
        except Exception:                          # noqa: BLE001                               size: right, not sure
            free = False
        if free:
            conf = min(conf, ALT_CONF)
            with_vals += " (the negative root: a size, a length?)"
    if not ok and not size and not isinstance(value, complex) and \
            any(abs(complex(v).imag) > 1e-12 for v in data.values() if isinstance(v, (complex, sp.Basic))):
        # complex data, a real answer: the data by their sizes, as power formulas are written ('I = 2\angle 30^\circ',
        # 'P = I^2 R' -> 20, not 20\angle 60^\circ)
        for d in ({k: abs(complex(v)) for k, v in data.items()}, {k: abs(complex(v)) * (fac[k] or 1) for k, v in data.items()}):
            sm = _solve_for(eq, d, sym)
            if f_ans not in (None, 1) and d is not None and any(f not in (None, 1) for f in fac.values()):
                sm += [x / f_ans for x in sm]
            if any(_close(value, x, vtext) for x in sm):
                ok, pos = True, _positive(sm) or sm
                with_vals += " (the complex values by their sizes)"
                break
    if not ok and not size and isinstance(value, (int, float)) and re.search(r"\\cdot\s*0\.010?\s*$", vtext or "") and \
            re.search(r"(?:\\cdot|\\times|\*)\s*100(?![\d.])", formula.text) and any(_close(value * 100, x) for x in sols):
        ok = True                                    # '\eta = \frac{P_o}{P_i} \cdot 100', then '\eta = 50\%': the formula
        #                                              is in percent - 50, not 0.5
    if not ok and not size and src is not None and src.kind == "value" and _is_db(src.unit):
        how = {_db_match(value, x, vtext) for x in sols + alt if isinstance(x, (int, float))}   # 'A_v = 40 dB' for 100
        if how & {10, 20}:
            ok = True
            with_vals += " (in dB: 20 log10)" if 20 in how else f" ({_DB10_NOTE})"
            conf = conf if 20 in how else min(conf, ALT_CONF)
    if not ok and not size and not isinstance(value, complex) and _MIXED.search(vtext or ""):
        prod = _parse(_MIXED.sub(r"\1 \\cdot ", vtext))    # 'x = 3 \frac{1}{4}': the product, if not the mixed number
        if prod is not None and _num(prod) is not None and any(_close(_num(prod), x) for x in sols):
            ok = True
    sols_d, si_d = [], False
    deg_marked = any(re.search(r"\\circ|°", a[2] or "") for a in mine[:-1] if a[0] in data) or \
        any(re.search(r"\\circ|°", getattr(v, "value_text", "") or "") for v in from_given.values())
    # (an angle written with its degree mark is in radians already: 'theta_v = 30^\circ' read in degrees again - cos of
    # 1.05 degrees - made 'P = 10 W' right for 5 W)
    in_rad = src is not None and src.kind == "value" and re.sub(r"[\s{}\\]", "", src.unit or "") == "rad"   # ('= 45 rad')
    if not ok and not size and not deg_marked and not in_rad and not isinstance(value, complex) and eq.has(*_TRIG, *_ATRIG):
        eq_d = _in_degrees(eq)                       # '\varphi = \arctan(\frac{X}{R})' answered '53.13': in degrees
        sols_d = _solve_for(eq_d, data, sym) if eq_d is not None else []
        if eq_d is not None and f_ans is not None and data and (f_ans != 1 or any(f not in (None, 1) for f in fac.values())):
            # ('S = 1 kVA', '\phi = 36.87°', then 'Q = S \sin\phi', 'Q = 600 VAR': in degrees and in SI units both - it was
            # a sure wrong, 1 kVA taken for 1 VA with the angle in degrees, or 1000 VA with 36.87 radians)
            sols_dsi = [x / f_ans for x in _solve_for(eq_d, {k: v * (fac[k] or 1) for k, v in data.items()}, sym)]
            si_d = any(_close(value, x, vtext) for x in sols_dsi)
            sols_d += [x for x in sols_dsi if not any(_close(x, y) for y in sols_d)]
        if any(_close(value, x, vtext) for x in sols_d):
            ok, pos = True, _positive(sols_d) or sols_d
            with_vals += " (angles in degrees)"
            if not all(_degree_like(v) for v in _angle_args(eq.subs({**data, sym: value}))):
                conf = min(conf, ALT_CONF)
                with_vals += f" ({_DEG_NOTE})"
    db_keys = [k for k in data if _is_db(unit_of.get(k, ""))]
    mult_db = [k for k in db_keys if not _additive(eq, k, sym)] if ok and not size and db_keys else []
    if mult_db and not isinstance(value, complex):
        # ('A_v = 40 dB', 'V_o = A_v V_i', 'V_o = 4 V': 40 times 0.1 - a gain in dB multiplied as if a ratio; dB added are
        # right - 'G = G_1 + G_2' - not sure, for a human)
        conf = min(conf, ALT_CONF)
        with_vals += " (" + ", ".join(_shown(k) for k in mult_db) + " in dB, multiplied as a ratio? its ratio is 10^(x/20))"
    if not ok and not size and db_keys and not isinstance(value, complex):
        # 'A_v = 40 dB', then 'V_o = A_v V_i', 'V_o = 10 V': the gain as its ratio (100) - it was 40 times V_i, a sure wrong
        for kk in (20, 10):
            got_db = []
            for dset in (data, {k: v * (fac[k] or 1) for k, v in data.items()}):
                try:
                    got_db += _solve_for(eq, {k: (10 ** (float(v) / kk) if k in db_keys else v) for k, v in dset.items()}, sym)
                except (TypeError, ValueError, OverflowError):
                    pass
            if f_ans not in (None, 1):
                got_db += [x / f_ans for x in got_db]
            got_db = [x for i_, x in enumerate(got_db) if not any(_close(x, y) for y in got_db[:i_])]
            if any(_close(value, x, vtext) for x in got_db):
                ok, pos = True, [x for x in (_positive(got_db) or got_db) if _close(value, x, vtext)] or got_db
                with_vals += " (" + ", ".join(_shown(k) for k in db_keys) + f" from dB: 10^(x/{kk}))"
                conf = conf if kk == 20 else min(conf, ALT_CONF)
                break
    if alt and any(near(x) for x in alt):
        if not ok and not any(near(x) for x in alt if x not in sols_m):
            with_vals += " (" + ", ".join(f"{_shown(k)} in milli" for k in data if (unit_of.get(k) or "").strip() == "m") + ")"
        ok, pos = True, _positive(alt) or alt
    elif ok and alt and prefixed and not si_d:
        # right only with the numbers as written, past a prefix the student wrote ('C = 1 \mu F', omega = 1000 ->
        # 'X_C = 0.001'): the prefix left out on the way? - for a human (not when it holds in SI units with the angles
        # in degrees: 'S = 2 kVA', '\phi = 30', 'P = S \cos\phi = 1.73 kW')
        conf = min(conf, ALT_CONF)
        with_vals += f" (as written; in SI units {name} = {_fmt(alt[0])})"
    elif alt and prefixed and not ok:                # wrong either way: the reading in SI units first
        with_vals += " (" + " or ".join(f"{name} = {_fmt(x)}" for x in pos) + " with the numbers as written)"
        pos = _positive(alt) or alt
    if not ok and not size and isinstance(value, (int, float)) and data and pos and all(isinstance(x, (int, float)) for x in pos):
        # the values used written rounded ('I_C = 11 mA' for 10.99 mA, then 'V_{CE} = V_{CC} - I_C R_C = 1.01 V' - 12 - 11 is
        # 1): how far the answer moves with each of them moved by half a unit in its last digit - an answer within
        # that is no sure wrong (right, not sure, when near; else wrong, not sure)
        # (a value worked out above, as rounded as its digits say; a value written of three digits and more - not
        # '12 V', '3.3 k\Omega', '11 mA' given: exact)
        rounded = {a[0]: a[2] for a in mine[:-1] if a[0] in data and (
            (a[3] is not None and (a[3].kind in ("chain", "numeric") or a[3].calc.get("worked"))) or _rel_half_unit(a[2]))}
        rounded.update({k: getattr(v, "value_text", "") for k, v in from_given.items() if k in data and (
            v.calc.get("worked") or _rel_half_unit(getattr(v, "value_text", "")))})
        moved = _rounding_moves(eq, sym, data, rounded, fac if si_branch else None, f_ans if si_branch else None)
        x0 = min(pos, key=lambda x: abs(x - value))
        if moved and abs(value - x0) <= moved + _half_unit(vtext):
            conf = min(conf, ALT_CONF)
            ok = abs(value - x0) <= 0.03 * abs(x0)
            with_vals += " (within the rounding of the values used)"
    in_deg = bool(re.search(r"\\circ|°", vtext or "")) and all(isinstance(x, (int, float)) for x in pos)
    want = " or ".join(f"{name} = {_fmt(math.degrees(x)) + '°' if in_deg else _fmt(x)}" for x in pos)   # (an answer in degrees:
    #                                                  the values so - 'φ = 53.13°', not 0.9273)
    if not ok and sols_d:                            # wrong either way: the value with the angles in degrees too
        want += " (with the angles in degrees " + " or ".join(f"{name} = {_fmt(x)}" for x in (_positive(sols_d) or sols_d)) + ")"
    head = f"{_pretty(formula.text)}" + (f" with {with_vals}" if with_vals else "") + f": {want}"
    bad_steps = [(t, va, vb) for t, good, va, vb in steps if not good]
    unit_note = ""                                   # 'R = 5 kV' for 10 V / 2 mA: the number right, the unit not
    a_unit = src.unit.strip() if src is not None and src.kind == "value" else ""
    if a_unit and a_unit != "m" and data and not formula.calc.get("si"):     # (its numbers' units gone: no dimension)
        want_dim, got_dim = _result_dimension(eq, sym, {k: unit_of.get(k, "") for k in data}), _dimension(a_unit)
        if want_dim is not None and got_dim is not None and sp.simplify(want_dim / got_dim).free_symbols:
            unit_note = f"its unit: {name} comes out in {_dim_text(want_dim)}, not {a_unit}"
    if ok and unit_note:
        checks.append(Check(no, "right value, wrong unit", f"{head} - written {answer}: the number is right, but {unit_note}",
                            min(conf, ALT_CONF), False, [x.text for x in used], expected=pos, answer=answer, steps=step_list))
    elif ok:
        checks.append(Check(no, "right", f"{head} - written {answer}: right", conf, conf >= SURE,
                            [x.text for x in used], expected=pos, answer=answer, steps=step_list))
    if ok:                                           # a false step beside a right answer ('I = 2', then 'P = 2 \cdot 10 = 30'
        checks += [_step_check(no, t, good, va, vb, conf, [x.text for x in used])   # under 'I = \frac{V}{R}'): its own check -
                   for t, good, va, vb in steps if not good]                     # it was dropped, the problem 'right' 
    else:
        if not a_unit and isinstance(value, (int, float)) and value != 0 and \
                any(isinstance(x, (int, float)) and x != 0 and _close(value, x * 1000.0 ** p, vtext) for x in pos + alt for p in (1, 2, -1, -2)):
            conf = min(conf, ALT_CONF)               # 'I = 5' for 5 mA (0.005 A): a power of 1000 apart, the unit left out -
            head += " (a power of 1000 apart: a unit left out?)"      # wrong, not sure: for a human
        why = "".join(f"; the step '{t}' is false ({_fmt(va)}, not {_fmt(vb)})" for t, va, vb in bad_steps)
        why += f"; {unit_note}" if unit_note else ""
        if from_above:
            conf = min(conf, ALT_CONF)
        checks.append(Check(no, "wrong", f"{head}, but written {answer}: wrong{why}", conf, conf >= SURE,
                            [x.text for x in used], expected=pos, answer=answer, steps=step_list))
    return checks


_SI_BASE = {"V": 1.0, "A": 1.0, "W": 1.0, "Wh": 3600.0, "Ah": 3600.0, "VA": 1.0, "VAR": 1.0, "var": 1.0, "S": 1.0, "Omega": 1.0, "Ω": 1.0, "ohm": 1.0, "F": 1.0, "H": 1.0, "Hz": 1.0, "s": 1.0,
            "J": 1.0, "N": 1.0, "C": 1.0, "Pa": 1.0, "m": 1.0, "g": 1e-3, "dB": 1.0, "bit": 1.0, "bits": 1.0, "bps": 1.0, "baud": 1.0,
            "dBm": 1.0, "dBW": 1.0, "dBi": 1.0, "dBc": 1.0, "T": 1.0, "Wb": 1.0, "eV": 1.602176634e-19, "rad": 1.0}
_SI_PREFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "mu": 1e-6, "m": 1e-3, "c": 1e-2, "k": 1e3, "M": 1e6, "G": 1e9}


def _si(unit: str):
    """a unit as written -> its factor to SI ('mA' 0.001, 'k\\Omega' 1000, 'mm' 0.001, 'kg' 1, 'km/h' 1/3.6,
    '\\frac{g}{cm^3}' 1000), None: no unit / unknown"""
    u = re.sub(r"[\\\s{}]", "", unit or "")
    if not u:
        return None
    if u in _SI_BASE:
        return _SI_BASE[u]
    if u in ("k", "M", "G", "u", "µ", "μ", "mu", "n", "p"):    # a bare prefix ('C = 100n', 'R = 1k', 'C = 10\\mu'): that
        return {"μ": 1e-6}.get(u, _SI_PREFIX.get(u))           # many of the unit left unsaid (100 was taken: false wrongs)
    for p in sorted(_SI_PREFIX, key=len, reverse=True):
        if u.startswith(p) and u[len(p):] in _SI_BASE:
            return _SI_PREFIX[p] * _SI_BASE[u[len(p):]]
    return _si_compound(unit)


_DM, _DL, _DT, _DI, _DK = sp.symbols("Dim_M Dim_L Dim_T Dim_I Dim_K", positive=True)
_OHM = _DM * _DL ** 2 * _DT ** -3 * _DI ** -2
_DIM = {"V": _DM * _DL ** 2 * _DT ** -3 * _DI ** -1, "A": _DI, "W": _DM * _DL ** 2 * _DT ** -3, "Ω": _OHM, "Omega": _OHM,
        "VA": _DM * _DL ** 2 * _DT ** -3, "VAR": _DM * _DL ** 2 * _DT ** -3, "var": _DM * _DL ** 2 * _DT ** -3,
        "Wh": _DM * _DL ** 2 * _DT ** -2,
        "Ah": _DI * _DT,
        "ohm": _OHM, "F": _DM ** -1 * _DL ** -2 * _DT ** 4 * _DI ** 2, "H": _DM * _DL ** 2 * _DT ** -2 * _DI ** -2, "Hz": _DT ** -1,
        "s": _DT, "sec": _DT, "min": _DT, "h": _DT, "hr": _DT, "J": _DM * _DL ** 2 * _DT ** -2, "N": _DM * _DL * _DT ** -2,
        "C": _DT * _DI, "Pa": _DM * _DL ** -1 * _DT ** -2, "m": _DL, "g": _DM, "K": _DK, "rad": sp.Integer(1), "dB": sp.Integer(1),
        "S": _DM ** -1 * _DL ** -2 * _DT ** 3 * _DI ** 2, "L": _DL ** 3, "Wb": _DM * _DL ** 2 * _DT ** -2 * _DI ** -1,
        "Vs": _DM * _DL ** 2 * _DT ** -2 * _DI ** -1,
        "T": _DM * _DT ** -2 * _DI ** -1, "eV": _DM * _DL ** 2 * _DT ** -2}
_DIM_NAME = {"V": "V", "A": "A", "W": "W", "Ω": "Ω", "F": "F", "H": "H", "Hz": "Hz (1/s)", "s": "s", "J": "J", "N": "N", "C": "C",
             "Pa": "Pa", "m": "m", "g": "kg", "S": "S"}


def _dim_atom(a: str):
    m = re.fullmatch(r"(?P<b>[^\^]+)(?:\^(?P<e>-?\d))?", a)
    if not m:
        return None
    b, e = m.group("b"), int(m.group("e") or 1)
    if b in _DIM:
        return _DIM[b] ** e
    for p in sorted(_SI_PREFIX, key=len, reverse=True):
        if b.startswith(p) and b[len(p):] in _DIM:
            return _DIM[b[len(p):]] ** e
    return None


def _dimension(unit: str):
    """a unit as written -> its dimension, a product of powers of mass, length, time, current, temperature ('k\\Omega',
    '\\frac{m}{s^2}', 'kg \\cdot m'); None: no unit, or one not known (a bare prefix, '°')"""
    u = _unwrap_rm(unit or "").replace("\\Omega", "Ω").replace("\\mu", "µ")
    u = re.sub(r"(?<![A-Za-zµ])µ\s+", "µ", re.sub(r"\\(?:cdot|times)", " ", u)).strip()
    if not u:
        return None
    m = re.fullmatch(r"\s*\\frac\s*\{(?P<n>(?:[^{}]|\{[^{}]*\})*)\}\s*\{(?P<d>(?:[^{}]|\{[^{}]*\})*)\}\s*", u)
    num, den = (m.group("n"), m.group("d")) if m else ((u.split("/") + [""])[:2] if u.count("/") <= 1 else (None, None))
    if num is None:
        return None
    d = sp.Integer(1)
    for part, sign in ((num, 1), (den, -1)):
        for a in re.sub(r"\s*\^\s*", "^", re.sub(r"[{}\\]", "", part)).replace("−", "-").split():
            if a == "1":
                continue
            da = _dim_atom(a)
            if da is None:
                return None
            d *= da ** sign
    return d


def _result_dimension(eq, sym, units: dict):
    """the dimension the formula gives its letter from the data's units - None when some letter has no unit known,
    or the formula mixes dimensions / puts them under a function"""
    dims = {}
    for k, u in units.items():
        d = _dimension(u) if u and u.strip() not in ("m",) else (_DIM["m"] if u and u.strip() == "m" else None)
        if d is None:
            return None
        dims[k] = d
    try:
        sols = sp.solve(eq, sym)
    except Exception:                              # noqa: BLE001
        return None
    for s in sols:
        if s.free_symbols - set(dims):
            continue
        got = None                                   # term by term: 'V_{CC} - I_C R_C' is volts less volts - as one
        for t in sp.Add.make_args(sp.expand(s)):     # expression 0, '1 (no unit)': a false 'wrong unit'
            e = sp.simplify(t.xreplace(dims))
            e = sp.powsimp(sp.expand_power_base(e, force=True), force=True)
            if e.is_Add or e.atoms(sp.Function):
                return None
            coeff, rest = e.as_coeff_Mul()
            if not rest.free_symbols <= {_DM, _DL, _DT, _DI, _DK}:
                return None
            if got is None:
                got = rest
            elif sp.simplify(got / rest) != 1:       # (dimensions mixed: none told)
                return None
        if got is not None:
            return got
    return None


def _dim_text(d) -> str:
    for k, v in _DIM.items():
        if k in _DIM_NAME and sp.simplify(d / v) == 1:
            return _DIM_NAME[k]
    return "1 (no unit)" if d == 1 else str(d).replace("Dim_", "").replace("**", "^")


_SI_MORE = {"rad": 1.0, "sec": 1.0, "min": 60.0, "h": 3600.0, "hr": 3600.0, "L": 1e-3, "K": 1.0, "S": 1.0, "Vs": 1.0}
# ('Vs': a mobility's volt-seconds, 'cm^2/Vs')


def _si_atom(a: str):
    m = re.fullmatch(r"(?P<b>[^\^]+)(?:\^(?P<e>-?\d))?", a)
    if not m:
        return None
    b, e = m.group("b"), int(m.group("e") or 1)
    bases = {**_SI_BASE, **_SI_MORE}
    if b in bases:
        return bases[b] ** e
    for p in sorted(_SI_PREFIX, key=len, reverse=True):
        if b.startswith(p) and b[len(p):] in bases:
            return (_SI_PREFIX[p] * bases[b[len(p):]]) ** e
    return None


def _si_compound(unit: str):
    """'\\frac{m}{s^2}', 'km/h', 'm^2', '\\frac{kg \\cdot m}{s^2}' -> the factor to SI; None: not such a unit"""
    u = _unwrap_rm(unit or "").replace("\\Omega", "Ω").replace("\\mu", "µ")
    u = re.sub(r"(?<![A-Za-zµ])µ\s+", "µ", re.sub(r"\\(?:cdot|times)", " ", u))      # '\mu C': one atom
    m = re.fullmatch(r"\s*\\frac\s*\{(?P<n>(?:[^{}]|\{[^{}]*\})*)\}\s*\{(?P<d>(?:[^{}]|\{[^{}]*\})*)\}\s*", u)
    num, den = (m.group("n"), m.group("d")) if m else ((u.split("/") + [""])[:2] if u.count("/") <= 1 else (None, None))
    if num is None:
        return None
    f = 1.0
    for part, sign in ((num, 1), (den, -1)):
        for a in re.sub(r"\s*\^\s*", "^", re.sub(r"[{}]", "", part)).replace("−", "-").split():
            if a == "1":                             # ('\frac{1}{cm^3}')
                continue
            fa = _si_atom(a)
            if fa is None:
                return None
            f *= fa ** sign
    return f


def _cm_factor(unit: str):
    """a unit with lengths in it ('\\mu m', 'nm', 'm^{-3}', '\\frac{cm^2}{s}') -> the factor from it to the same unit in
    centimetres (1e-4 for \\mu m, 1e-6 for m^{-3}); None: no length in it, or not such a unit"""
    u = _unwrap_rm(unit or "").replace("\\Omega", "Ω").replace("\\mu", "µ")
    u = re.sub(r"(?<![A-Za-zµ])µ\s+", "µ", re.sub(r"\\(?:cdot|times)", " ", u))
    m = re.fullmatch(r"\s*\\frac\s*\{(?P<n>(?:[^{}]|\{[^{}]*\})*)\}\s*\{(?P<d>(?:[^{}]|\{[^{}]*\})*)\}\s*", u)
    num, den = (m.group("n"), m.group("d")) if m else ((u.split("/") + [""])[:2] if u.count("/") <= 1 else (None, None))
    if num is None or _si(unit) is None:
        return None
    n = 0
    for part, sign in ((num, 1), (den, -1)):
        for a in re.sub(r"\s*\^\s*", "^", re.sub(r"[{}]", "", part)).replace("−", "-").split():
            ma = re.fullmatch(r"(?P<p>[pnuµckM]|mu|m)?m(?:\^(?P<e>-?\d))?", a)
            if ma:
                n += sign * int(ma.group("e") or 1)
    return _si(unit) / 0.01 ** n if n else None


def _is_substitution(formula, s) -> bool:
    """a one-letter 'formula' under a formula that lacks that letter: a substitution with a misread digit"""
    lets = set().union(*(_letters(x) for x in s.sides))
    return len(lets) == 1 and not (lets & set().union(*(_letters(x) for x in formula.sides)))


def problems(stmts):
    """statements in reading order -> [(formula, [statements under it])]; statements before the first
    formula form a problem without one (formula None)"""
    out, formula, body = [], None, []
    system = set()                                 # the letters of a system of equations being written
    assigned = {}                                  # the values given in this problem so far: {letter: statement}
    feeds = set()                                  # the letters of the formula this one gives a letter of ('P = I^2 R',
    for s in stmts:                                # then 'I = \frac{V}{R}': the data and answers below are both's)
        own = set().union(*(_letters(x) for x in s.sides))
        if s.calc.get("item") and (formula is not None or body):
            out.append((formula, body))            # '3) ...': a new exercise, whatever its first line is
            formula, body, system, assigned = None, [], set(), {}
            feeds = set()
        elif s.kind == "value" and s.symbol is not None and (formula is not None or body) and (
                (s.symbol in assigned and not _same_value(assigned[s.symbol], s) and not _roots_of(formula, assigned[s.symbol], s) and
                 not _answer_again(formula, body, assigned, s)) or
                (formula is not None and formula.kind == "formula" and s.symbol not in _letters_of(formula) | feeds and
                 _letters_of(formula) and _letters_of(formula) | feeds <= set(assigned))):
            # (or a value for a letter foreign to a formula all of whose letters have their values already: 'P = I^2 R',
            # 'I = 2 mA', 'R = 2.2 k\Omega', 'P = 8.8 mW', then 'V = 10 V' - the next part)
            # 'R = 2.2 k\Omega' ... then 'R = 4.7 k\Omega': a letter given again with another value - the next part of
            # the page, its sub-questions unnumbered (it was taken for this problem's: the formula's answer 'R = 4.7 k',
            # a step 'I = V/R' judged with the I worked out above - false sure wrongs)
            body, carried = _repeated_tail(body, s)
            if formula is not None or body:
                out.append((formula, body))
            formula, body, system, assigned = None, list(carried), set(), {t.symbol: t for t in carried}
            feeds = set()
        if s.kind == "value" and s.symbol is not None:
            assigned[s.symbol] = s
        if formula is not None and s.kind == "formula" and not s.calc.get("item") and _uses_defined(formula, body, s):
            body.append(s)                         # "f'(x) = 2x" under 'f(x) = x^2 + 1': the same problem
            continue
        if formula is not None and s.kind == "formula" and _pure_equation(s) and own & system and not s.calc.get("item"):
            body.append(s)                         # '10 = 5I_1 - 2I_2', '0 = -2I_1 + 6I_2': one system, one problem
            system |= own
            continue
        equation = (formula is None and s.kind == "chain" and len(s.sides) == 2      # '2x+3 = 7': an equation to solve
                    and len(own) == 1 and not any(x.is_Symbol for x in s.sides) and not s.calc.get("evaluated"))
        # a computation in letters the formula does not have ('P_3A = -(3 x v_1) = -45' under 'V_1 = V_2 + 9'): the next part
        elsewhere = (formula is not None and s.kind == "chain" and len(own) >= 2
                     and not own & set().union(*(_letters(x) for x in formula.sides)))
        if (s.kind == "formula" and (formula is None or not _is_substitution(formula, s))) or equation or elsewhere:
            feeds = _letters_of(formula) if (formula is not None and formula.kind == "formula" and not body and
                                             s.kind == "formula" and s.sides and s.sides[0].is_Symbol and
                                             s.sides[0] in _letters_of(formula)) else set()
            if formula is not None or body:
                out.append((formula, body))
            formula, body, assigned = s, [], {}
            system = own if s.kind == "formula" and _pure_equation(s) else set()
        else:
            body.append(s)
    if formula is not None or body:
        out.append((formula, body))
    return out


def _roots_of(formula, a, b) -> bool:
    """two values for one letter that are an equation's roots ('x^2 - 5x + 6 = 0', then 'x = 2, x = 3' or on lines of their
    own; 'x = \\pm 3'), not a letter given again in the next part - the same line, both a '\\pm', or under an equation
    with no letter alone on a side (it was split: the second root never checked - 'x = 4' no wrong)"""
    if a.line == b.line and a.line >= 0 or (a.calc.get("pm") and b.calc.get("pm")):
        return True
    return formula is not None and formula.kind in ("formula", "chain") and not any(x.is_Symbol for x in formula.sides)


def _answer_again(formula, body: list, assigned: dict, s) -> bool:
    """'P = I^2 R', 'I = 1 mA', 'R = 10 k\\Omega', 'P = 13 mW', then 'P = 10 mW' at once: the answer written again (the
    first one crossed out - the reader does not see strokes), not the next part - unless the line is a sub-question's"""
    if formula is None or s.calc.get("sub") or s.calc.get("item") or not body or body[-1] is not assigned.get(s.symbol):
        return False
    lets = _letters_of(formula)
    return s.symbol in lets and lets - {s.symbol} <= set(assigned)


def _repeated_tail(body: list, s) -> tuple:
    """the value statements ending a problem that only give its letters their values again ('R = 1 k\\Omega' twice) -
    the next part's first lines, when the next part starts ('R = 1 k\\Omega', 'V = 5 V', 'I = \\frac{V}{R}' after 'P = I^2 R'
    and its data: R stayed with it - no value for R below, a false wrong step) -> (the problem's, the next part's)"""
    k = len(body)
    while k > 0 and body[k - 1].kind == "value" and body[k - 1].symbol is not None and body[k - 1].symbol != s.symbol and \
            any(t.kind == "value" and t.symbol == body[k - 1].symbol and _same_value(t, body[k - 1]) for t in body[:k - 1]):
        k -= 1
    return body[:k], body[k:]


def _letters_of(s) -> set:
    return set().union(*(_letters(x) for x in s.sides)) if s is not None else set()


def _same_value(a, b) -> bool:
    """two value statements give their letter the same value - as written, or in SI units ('R = 2 k\\Omega', 'R = 2000
    \\Omega'), or one a computation the other finishes ('I = \\frac{12}{4}', then 'I = 3 A')"""
    try:
        if _EXPRESSION.search((a.value_text or "").strip().lstrip("+-")) or _EXPRESSION.search((b.value_text or "").strip().lstrip("+-")):
            return True
        va, vb = complex(a.number), complex(b.number)
        fa = _si(a.unit) if a.unit and a.unit.strip() != "m" else 1.0
        fb = _si(b.unit) if b.unit and b.unit.strip() != "m" else 1.0
        if abs(va - vb) <= 0.02 * max(abs(va), abs(vb)):
            return True
        return fa is not None and fb is not None and abs(va * fa - vb * fb) <= 0.02 * max(abs(va * fa), abs(vb * fb))
    except (TypeError, ValueError):
        return True


def _fnames(s):
    """the functions a statement names (f of f(x), and f of its derivative f'(x)), and the letters whose derivative it
    names (y of y')"""
    from sympy.core.function import AppliedUndef
    out = set()
    for x in s.sides:
        for f in x.atoms(AppliedUndef):
            m = _DERIV_NAME.fullmatch(str(f.func))
            out.add(m.group(1) if m else str(f.func))
        for v in x.free_symbols:
            m = _DERIV_NAME.fullmatch(str(v))
            if m:
                out.add(m.group(1))
            elif str(v) in _D_OPAQUE and _D_OPAQUE[str(v)][0].is_Symbol:
                out.add(str(_D_OPAQUE[str(v)][0]))
    return out


def _uses_defined(formula, body, s) -> bool:
    """does s use a function (or a letter's derivative) the problem's formula or its body defines?"""
    used = _fnames(s)
    if not used:
        return False
    defined = set()
    for d in [formula] + [b for b in body if b.kind == "formula"]:
        if len(d.sides) == 2:
            for x in d.sides:
                if x.is_Symbol:
                    defined.add(str(x))
            defined |= {n for n in _fnames(d)}
    return bool(used & defined)


def _pure_equation(s) -> bool:
    """a linear equation in two or more letters with no letter alone on a side ('10 = 5I_1 - 2I_2', '\\frac{V_A - 10}{2} +
    \\frac{V_A}{4} = 0', 'I_1 - I_2 - I_3 = 0') - one of a system; not a definition ('P = V I'), nor a formula used problem
    after problem ('a^2 + b^2 = c^2')"""
    own = set().union(*(_letters(x) for x in s.sides))
    if len(s.sides) != 2 or len(own) < 2 or any(x.is_Symbol for x in s.sides) or any(_unworked(x) for x in s.sides):
        return False
    try:
        return sp.Poly(sp.together(s.sides[0] - s.sides[1]).as_numer_denom()[0], *sorted(own, key=str)).total_degree() == 1
    except Exception:                              # noqa: BLE001
        return False


def _check_system(no, eqs, stmts, given, far):
    """a system of equations and the values written for its letters after it -> [Check] (one per value), or None when it
    has no single solution in them. The other letters' values: the data written before it"""
    letters = set().union(*(_letters(x) for e in eqs for x in e.sides))
    answers = {s.symbol: s for s in stmts if s.kind == "value" and s.symbol in letters}
    data = {k: _as_data(v.number) for pool in (far or {}, given or {}) for k, v in pool.items() if k in letters and k not in answers}
    unknowns = sorted(letters - set(data), key=str)
    if not answers or not 2 <= len(unknowns) <= 4 or len(eqs) < len(unknowns):
        return None
    plain = {x: sp.Symbol(x.name) for x in unknowns}             # phasors: complex unknowns (the letters are real,
    system = [sp.Eq(e.sides[0], e.sides[1]).xreplace(plain).subs(data) for e in eqs]   # and equal to j... False at once)
    if any(isinstance(x, sp.Basic) and x.has(sp.I) for x in system) or any(isinstance(a.number, complex) for a in answers.values()):
        back, targets = {v: k for k, v in plain.items()}, list(plain.values())
    else:
        system = [sp.Eq(e.sides[0], e.sides[1]).subs(data) for e in eqs]
        back, targets = {}, unknowns
    try:
        sols = sp.solve(system, targets, dict=True)
    except Exception:                              # noqa: BLE001
        return None
    sols = [{back.get(k, k): v for k, v in d.items()} for d in sols]
    sols = [d for d in sols if all(k in d and _value(d[k]) is not None for k in unknowns)]
    if len(sols) != 1:
        return None
    sol = {k: _value(v) for k, v in sols[0].items()}
    conf = min([e.conf for e in eqs] + [a.conf for a in answers.values()])
    name = lambda k: _pretty(_split_eq(answers[k].text)[0]) if k in answers else _shown(k)     # noqa: E731 (as written: V_1,
    head = f"the system {'; '.join(_pretty(e.text) for e in eqs)}: " + ", ".join(f"{name(k)} = {_fmt(v)}" for k, v in sol.items())   # not v_1)
    out = []
    for sym, a in answers.items():
        if sym not in sol:
            continue
        x, f_a, c, note = sol[sym], _si(a.unit) if a.unit and a.unit.strip() != "m" else None, conf, ""
        ok = _close(a.number, x, a.value_text)     # in the system's own units (kΩ, so mA - or SI)
        if not ok and isinstance(a.number, (int, float)) and isinstance(x, (int, float)):
            if f_a not in (None, 1.0) and _close(a.number * f_a, x):
                ok = True                          # 'I_1 = 4.36 mA' for a system in SI units (0.00436)
            elif f_a is not None and _close(a.number * f_a * 1e3, x):
                ok, c = True, min(conf, ALT_CONF)  # 'I_1 = 0.00436 A' for a system in kΩ (so in mA: 4.36) - not sure
                note = " (the system in kΩ, so mA?)"
            elif a.number != 0 and x != 0 and any(_close(a.number * (f_a or 1.0), x * 1000.0 ** p) for p in (1, -1, 2, -2)):
                c = min(conf, ALT_CONF)            # a power of 1000 apart - a unit left out, or the system's in kΩ: for a human
                note = " (a power of 1000 apart: a unit left out?)"
        out.append(Check(no, "right" if ok else "wrong", f"{head}{note} - written {_pretty(a.text)}: {'right' if ok else 'wrong'}",
                         c, c >= SURE, [e.text for e in eqs] + [a.text], expected=[x], answer=_pretty(a.text),
                         line=a.line))
    return out or None


_DERIV_NAME = re.compile(r"([A-Za-z])_\{((?:\\prime)+)\}")


def _laplace_pair(name, args, expr, defs, s):
    """a time function defined after its Laplace transform by the same letter (V(s), then v(t); I_L(s), then i_L(t)):
    the inverse transform against it, for t > 0 -> an 'identity' statement; None when there is no such pair"""
    if len(args) != 1 or str(args[0]) != "t":
        return None
    canon = lambda n: re.sub(r"\\iota\s*", "i", n).replace("{", "").replace("}", "").replace(" ", "").lower()   # noqa: E731
    other = next((lam for n, lam in defs.items() if n != name and canon(n) == canon(name) and len(lam.variables) == 1
                  and str(lam.variables[0]) == "s"), None)
    expr = _signals(expr)
    if other is None or other.expr.free_symbols != {other.variables[0]}:
        return None
    try:
        ps = sp.Symbol("s")
        inv = sp.inverse_laplace_transform(other.expr.xreplace({other.variables[0]: ps}), ps, args[0])
    except Exception:                              # noqa: BLE001
        return None
    if inv is None or inv.has(sp.InverseLaplaceTransform, sp.Integral) or _unworked(expr):
        return None
    inv = inv.xreplace({x: sp.Integer(1) for x in inv.atoms(sp.Heaviside)})      # t > 0
    sides = [inv, expr.xreplace({x: sp.Integer(1) for x in expr.atoms(sp.Heaviside)})]
    return Statement("identity", s.text, sides, s.conf, line=s.line, piece=s.piece,
                     side_texts=[f"L^-1{{{canon(name).upper()}(s)}}", s.side_texts[-1] if s.side_texts else str(expr)],
                     calc={"indefinite": [], "may_be_wrong": True, "pairs": [(0, 1)]})


def _canon_name(n: str) -> str:
    """V and v(t), I_L and i_L(t): one quantity's names compared"""
    return re.sub(r"\\iota\s*", "i", n).replace("{", "").replace("}", "").replace(" ", "").lower()


def _phasor_pair(name, args, expr, phasors, s):
    """a time function written after its phasor by the same letter ('I = 2\\angle -53.13^\\circ', then 'i(t) =
    2\\cos(1000t - 53.13^\\circ)'): its amplitude and phase against the phasor's - as a peak or an RMS phasor, with a
    cosine or a sine for reference -> a 'phasor' statement (its verdict in calc), or None"""
    if len(args) != 1 or str(args[0]) != "t" or _canon_name(name) not in phasors:
        return None
    t = args[0]
    trig = [g for g in expr.atoms(sp.cos, sp.sin) if g.has(t)]
    if len(trig) != 1 or len({g.args[0] for g in trig}) != 1:
        return None
    g = trig[0]
    arg = sp.expand(g.args[0])
    try:
        amp = sp.simplify(expr / g)
        phase = arg.subs(t, 0)
        w = sp.diff(arg, t)
    except Exception:                              # noqa: BLE001
        return None
    if amp.has(t) or w.has(t) or amp.free_symbols or phase.free_symbols or _cnum(amp) is None or _num(phase) is None:
        return None
    a, ph = _cnum(amp), _num(phase)
    if abs(a.imag) > 1e-9:
        return None
    mine = a.real * cmath.exp(1j * (ph - (math.pi / 2 if isinstance(g, sp.sin) else 0.0)))    # as a cosine's phasor
    x, src = phasors[_canon_name(name)]
    near = min((x, x * math.sqrt(2)), key=lambda c: abs(mine - c))      # a peak phasor, or an RMS one
    ok = abs(mine - near) <= max(0.015 * abs(near), 1e-9)
    msg = (f"'{_pretty(s.text)}': the phasor {_pretty(src.text)}" + (" (RMS)" if near is not x else "") +
           (" - as written" if ok else f" gives {_fmt(abs(near))}cos(ωt {'+' if cmath.phase(near) >= 0 else '-'} "
                                         f"{_fmt(abs(math.degrees(cmath.phase(near))))}°), not {_fmt(abs(mine))}∠"
                                         f"{_fmt(math.degrees(cmath.phase(mine)))}°"))
    return Statement("phasor", s.text, list(s.sides), min(s.conf, src.conf), line=s.line, piece=s.piece,
                     side_texts=s.side_texts, calc={"ok": ok, "msg": msg})


def _at_jw(s, defs):
    """'H(j\\omega) = \\frac{1}{1 + j\\omega RC}' or 'H(j2) = 0.2 - j0.4' after 'H(s) = \\frac{1}{1 + sRC}': the transfer
    function at s = j\\omega (j2) against it - an 'identity' statement (in letters), a 'phasor' one (a number: within
    its decimals), or None"""
    if s.kind not in ("formula", "value", "chain") or len(s.sides) != 2 or len(s.side_texts) != 2:
        return None
    head = s.side_texts[0].strip()
    phase = _ANGLE_AT.fullmatch(head)
    if phase:                                      # '\angle H(j1000) = -45^\circ': the phase of H at s = j1000
        head = re.sub(r"^\s*(?:\\angle|\\arg|\\measuredangle|\\sphericalangle|\\phase)\s*", "", head)
        head = re.sub(r"^\\left\s*\(|^\(|\\right\s*\)$|\)$", "", head.strip()) if head.count("(") > 1 else head
    db = re.fullmatch(r"20\s*(?:\\cdot\s*)?\\log\s*(?:_\s*\{?\s*10\s*\}?)?\s*(.*)", head)
    if db:                                         # '20\log|H(j1)| = -3': its size in dB
        head = db.group(1).strip()
    bars = re.fullmatch(r"(?:\\lvert|\\left\s*\||\|)\s*(.*?)\s*(?:\\rvert|\\right\s*\||\|)", head)
    if bars:                                       # '|H(j\omega)| = \frac{1}{\sqrt{1 + \omega^2}}': its size
        head = bars.group(1).strip()
    elif db:
        return None
    m = _AT_JW.fullmatch(head)
    if not m:
        return None
    lam = next((v for k, v in defs.items() if _canon_name(k) == _canon_name(m.group(1) + (m.group(2) or ""))
                and len(v.variables) == 1 and str(v.variables[0]) == "s"), None)
    arg = re.search(r"\(\s*j\s*(\\omega|w|ω|\d+(?:\.\d+)?)", head)
    if lam is None or arg is None:
        return None
    a = arg.group(1)
    w = sp.Float(a) if a[0].isdigit() else (sp.Symbol("w", real=True) if a == "w" else sp.Symbol("omega", real=True))
    if phase:
        got, x = _num(s.sides[1]), _cnum(lam(sp.I * w)) if a[0].isdigit() else None
        if got is None or x is None or abs(x) < 1e-12:
            return None
        ph = cmath.phase(x)                         # radians; the student's in degrees if he wrote no mark
        rad = abs((got - ph + math.pi) % (2 * math.pi) - math.pi) <= max(0.0087, REL_TOL * abs(ph))
        deg = abs((got - math.degrees(ph) + 180) % 360 - 180) <= max(_half_unit(s.side_texts[1]), 0.5, REL_TOL * abs(math.degrees(ph)))
        ok = rad or deg
        return Statement("phasor", s.text, list(s.sides), s.conf, line=s.line, piece=s.piece, side_texts=s.side_texts,
                         calc={"ok": ok, "sure_wrong": True, "msg": f"'{_pretty(s.text)}': ∠{m.group(1)}(j{a}) = "
                               f"{_fmt(math.degrees(ph))}°" + ("" if ok else f", not {_pretty(s.side_texts[1])}")})
    try:
        want = sp.simplify(lam(sp.I * w))
        if bars:
            want = sp.simplify(sp.Abs(want))
        if db:
            want = 20 * sp.log(want, 10)
    except Exception:                              # noqa: BLE001
        return None
    mine = s.sides[1]
    shown = f"{m.group(1)}(s→j{'ω' if a in ('\\omega', 'w', 'ω') else a})"
    shown = (f"20log|{shown}|" if db else f"|{shown}|") if bars else shown
    if not _letters(mine) and not _letters(want):  # a number: within its decimals
        got, x = _cnum(mine), _cnum(want)
        if got is None or x is None:
            return None
        ok = _close(got, x, s.side_texts[1])
        return Statement("phasor", s.text, list(s.sides), s.conf, line=s.line, piece=s.piece, side_texts=s.side_texts,
                         calc={"ok": ok, "sure_wrong": True, "msg": f"'{_pretty(s.text)}': {shown} = {_fmt_c(x)}" +
                               ("" if ok else f", not {_fmt_c(got)}")})
    if _unworked(mine):
        return None
    return Statement("identity", s.text, [want, mine], s.conf, line=s.line, piece=s.piece,
                     side_texts=[shown, s.side_texts[1]], calc={"indefinite": [], "may_be_wrong": True, "pairs": [(0, 1)]})


def _phasor_checks(no, s):
    """a time function against its phasor: right (sure), or wrong - not sure: the convention (peak or RMS, cosine or
    sine) may be one the check does not know"""
    ok = s.calc.get("ok")
    sure = (ok or s.calc.get("sure_wrong", False)) and s.conf >= SURE
    return [Check(no, "right" if ok else "wrong", s.calc.get("msg", s.text), s.conf, sure, [s.text], line=s.line)]


def _calls(e, names):
    """'2 u(n - 1)', 'h(n - 1)' as the parser reads them - products, 2*u*(n - 1): a letter before brackets with an expression
    in them is a call only to the parser when the brackets hold a single letter - back to calls, for the names that are
    functions here (u, \\delta - of n, k or t - and the functions the student defines)"""
    if not isinstance(e, sp.Basic) or not e.has(sp.Mul):
        return e

    def fix(m):
        args = list(m.args)
        si = [i for i, a in enumerate(args) if isinstance(a, sp.Symbol) and a.name.lstrip("\\") in names]
        ai = [i for i, a in enumerate(args) if isinstance(a, sp.Add)]
        if len(si) != 1 or len(ai) != 1:
            return m
        name, arg = args[si[0]].name, args[ai[0]]
        if name.lstrip("\\") in ("u", "delta") and (not arg.free_symbols or
                                                    not {str(x) for x in arg.free_symbols} <= {"n", "k", "t", "tau"}):
            return m
        rest = [a for i, a in enumerate(args) if i not in (si[0], ai[0])]
        return sp.Mul(*rest) * sp.Function(name)(arg)
    return e.replace(lambda x: isinstance(x, sp.Mul), fix)


def _steps_at(e, known=lambda name: None):
    """u(2), \\delta(0), \\delta(1) at a number - the unit step and the impulse, Kronecker's for a sequence (u[0] = 1): 'h[n] =
    0.5^n u[n]', then 'h[2] = 0.25' (it got no verdict); not a u the student defines himself"""
    from sympy.core.function import AppliedUndef
    rep = {}
    for f in e.atoms(AppliedUndef):
        name = str(f.func).strip("\\{} ")
        if len(f.args) != 1 or not f.args[0].is_number or known(str(f.func)) is not None:
            continue
        x = _num(f.args[0])
        if x is None:
            continue
        if name in ("u", "H", "Heaviside"):
            rep[f] = sp.Integer(1 if x >= 0 else 0)
        elif name in ("delta", "δ"):
            rep[f] = sp.Integer(1 if abs(x) < 1e-12 else 0)
    return e.xreplace(rep) if rep else e


def _zero_instants(stmts):
    """a value at t = 0 of a function the page never defines ('v_C(0) = 5 V', then 'i(0) = \\frac{12 - v_C(0)}{1 k\\Omega} =
    7 mA'): one name, as 'v_C(0^+)' is (statements()) - it was no value, and the steps with it got no verdict. A function
    the page defines ('v_C(t) = 12 - 7e^{-t/\\tau}', 'f(x) = x^2 + 1', 'H(s) = ...') keeps its value worked out there"""
    from sympy.core.function import AppliedUndef

    def calls(st):
        return [f for x in st.sides if isinstance(x, sp.Basic) for f in x.atoms(AppliedUndef)
                if len(f.args) == 1 and f.args[0] == 0]
    if not any(calls(st) for st in stmts):
        return stmts
    defined = set()
    for st in stmts:
        for x in st.sides:
            if isinstance(x, AppliedUndef) and x.args and all(a.is_Symbol for a in x.args):
                defined.add(str(x.func))             # ('v_C(t) = ...', 'f(x) = ...')
        if len(st.sides) == 2 and isinstance(st.sides[0], sp.Symbol) and isinstance(st.sides[1], sp.Basic) and \
                len(st.sides[1].free_symbols) == 1 and st.kind != "value":
            defined.add(str(st.sides[0]))            # ('y = x^2', then 'y(0)')
    out = []
    for st in stmts:
        names = {str(f.func) for f in calls(st)} - defined - {"u", "delta", "\\delta"}
        if not names or not st.text:
            out.append(st)
            continue

        def one(m):
            sub = (m.group(2) or m.group(3) or "").replace(" ", "")
            name = m.group(1) + (f"_{{{sub}}}" if sub else "")
            return f"{m.group(1)}_{{{sub}QQz}}" if name in names else m.group(0)
        text = re.sub(r"(?<![A-Za-z\\])([A-Za-z])(?:\s*_\s*(?:\{\s*([A-Za-z0-9 ]*?)\s*\}|([A-Za-z0-9])))?\s*(?:\\left\s*)?\(\s*0\s*"
                      r"(?:\\right\s*)?\)", one, st.text)
        got = statements(text, st.conf, st.line, st.piece) if text != st.text else []
        if not got:
            out.append(st)
            continue
        for g in got:
            if st.calc.get("item"):
                g.calc = {**g.calc, "item": st.calc["item"]}
        out += got
    return out


def _functions(stmts):
    """functions the student defines and uses: 'f(x) = x^2 + 1', then 'f(2) = 5' - its value worked out; "f'(x) = 2x",
    "f'(3) = 6", and y' or dy/dx under 'y = x^3' - checked against the derivative itself. A function written again
    with letters ('f(x) = \\sin x' in the next exercise) is defined anew; with a number ('f(x) = 7') it is an equation"""
    from sympy.core.function import AppliedUndef
    defs, letter_defs, out = {}, {}, []
    phasors = {}                                   # 'I = 2\angle -53.13^\circ': for i(t) written after it

    def known(name):
        if name in defs:
            return defs[name]
        if name in letter_defs:                      # 'y = x^2', then 'y(3) = 9'
            (v,) = letter_defs[name].free_symbols
            return sp.Lambda(v, letter_defs[name])
        m = _DERIV_NAME.fullmatch(name)
        if m and m.group(1) in defs and len(defs[m.group(1)].variables) == 1:
            v = defs[m.group(1)].variables[0]
            return sp.Lambda(v, sp.diff(defs[m.group(1)].expr, v, m.group(2).count("prime")))
        return None

    def subs(e):
        e = _calls(e, {n.lstrip("\\") for n in defs} | {"u", "delta"})    # ('h(n - 1)', '2\\delta(n - 1)': calls)
        rep = {}
        for f in e.atoms(AppliedUndef):
            lam = known(str(f.func))
            if lam is not None and len(lam.variables) == len(f.args):
                rep[f] = lam(*f.args)
        for x in e.free_symbols:
            m, base, var, order = _DERIV_NAME.fullmatch(str(x)), None, None, 0
            if m and m.group(1) in letter_defs:                    # y', y''
                base, order = m.group(1), m.group(2).count("prime")
            elif str(x) in _D_OPAQUE and _D_OPAQUE[str(x)][0].is_Symbol and \
                    {str(_D_OPAQUE[str(x)][0]), str(_D_OPAQUE[str(x)][0]).lower()} & set(letter_defs):
                nm = str(_D_OPAQUE[str(x)][0])     # (its letter as written; the definition's may be folded: V as v)
                base, var, order = (nm if nm in letter_defs else nm.lower()), _D_OPAQUE[str(x)][1], _D_OPAQUE[str(x)][2]
            if base is not None and order >= 1:
                expr = letter_defs[base]
                (v,) = expr.free_symbols
                if var is None or str(var) == str(v) or str(var).lower() == str(v):
                    rep[x] = sp.diff(expr, v, order)
        if not rep:
            return _steps_at(e, known)
        e = e.xreplace(rep)
        if e.has(sp.Derivative, sp.Limit):             # ('\lim_{s \to 0} s F(s)' with F(s) put in: worked out)
            try:
                e = e.doit()
            except Exception:                      # noqa: BLE001
                pass
        return _steps_at(e, known)

    for s in stmts:
        sides = list(s.sides)
        defined = False
        if s.calc.get("item"):                       # '2) ...': a new exercise - the phasors above are another's
            phasors = {}
        if s.kind == "formula" and len(sides) == 2:
            for i in (0, 1):
                f, other = sides[i], sides[1 - i]
                if not (isinstance(f, AppliedUndef) and f.args and all(a.is_Symbol for a in f.args)
                        and len(set(f.args)) == len(f.args)):
                    continue
                name = str(f.func)
                if _DERIV_NAME.fullmatch(name) and known(name) is not None:
                    continue                         # f'(x) of a known f: a derivative to check, below
                if known(name) is not None and not _letters(other):
                    continue                         # 'f(x) = 7' of a known f: an equation
                other = subs(other)
                if not other.has(f.func):
                    defs[name] = sp.Lambda(f.args, other)
                    defined = True
                    pair = _laplace_pair(name, f.args, other, defs, s)
                    if pair is not None:             # 'V(s) = \frac{10}{s(s+5)}', then 'v(t) = 2 - 2e^{-5t}'
                        out.append(pair)
                    pair = _phasor_pair(name, f.args, other, phasors, s)
                    if pair is not None:             # 'I = 2\angle -53.13^\circ', then 'i(t) = 2\cos(1000t - 53.13^\circ)'
                        out.append(pair)
                break
            if not defined and sides[0].is_Symbol and len(sides[1].free_symbols) == 1 and not _unworked(sides[1]) \
                    and not _DERIV_NAME.fullmatch(str(sides[0])):
                letter_defs[str(sides[0])] = sides[1]    # 'y = x^3': for y' and dy/dx
        if defined:                                  # a definition: no formula to solve ('f(x) = 2x + 1', then
            out.append(Statement("definition", s.text, sides, s.conf, line=s.line, piece=s.piece,   # 'f(x) = 7')
                                 side_texts=s.side_texts))
            continue
        if s.kind == "value" and s.symbol is not None and s.number is not None and (
                isinstance(s.number, complex) or "angle" in s.value_text):
            phasors[_canon_name(str(s.symbol))] = (complex(s.number), s)
        at = _at_jw(s, defs)                         # 'H(j\omega) = ...' after 'H(s) = ...': s = j\omega put in
        if at is not None:
            out.append(at)
        if s.kind in ("value", "identity"):
            out.append(s)
            continue
        new = [subs(x) for x in sides]
        if all(a == b for a, b in zip(new, sides)):
            out.append(s)
            continue
        kw = dict(line=s.line, piece=s.piece, side_texts=s.side_texts if len(s.side_texts) == len(new) else [str(x) for x in new])
        orig = [None if a != b else b for a, b in zip(new, sides)]     # a side worked out is no definition's letter
        got = _with_identity(s.text, new, orig, kw["side_texts"], s.unit, s.conf, kw,
                             {"indefinite": [], "may_be_wrong": True}) or [s]
        if any(f.args and all(a.is_number for a in f.args) for x in sides for f in x.atoms(AppliedUndef)):
            for g in got:                            # 'v_C(1 ms) = 6.32' with tau known: a function's value at a time
                g.calc = {**g.calc, "evaluated": True}   # worked out - a step to check, no equation to solve for tau
                if g.kind == "formula":                # ('V_0 e^{-1/\tau} = 3.68': its letters' values put in, as a
                    g.kind = "chain"                   # chain's - with two letters it was a formula, never checked)
        out += got                                   # ('f(x) = 7' under 'f(x) = 2x + 1' is one, for x)
    return out


def _pm_swapped(pair):
    """the roots of a "\\pm" pair named the other way round (x_1 the smaller): both answers right"""
    (cp, body), (cm, _) = pair[1], pair[-1]
    wp = [c for c in cp if c.verdict == "wrong"]
    wm = [c for c in cm if c.verdict == "wrong"]
    if len(wp) != 1 or len(wm) != 1:
        return
    vals = {c.answer: next((s for s in body if s.kind == "value" and _pretty(s.text) == c.answer), None) for c in (wp[0], wm[0])}
    a_p, a_m = vals[wp[0].answer], vals[wm[0].answer]
    if a_p is None or a_m is None:
        return
    if any(_close(a_p.number, x, a_p.value_text) for x in wm[0].expected) and \
            any(_close(a_m.number, x, a_m.value_text) for x in wp[0].expected):
        for c in (wp[0], wm[0]):
            c.verdict = "right"
            c.text = c.text.replace(", but written", " - written").replace(": wrong", ": right (the roots named the other way round)")


def _syms(s) -> set:
    out = set()
    for x in list(s.sides) + [s.symbol]:
        if isinstance(x, sp.Basic):
            out |= {y for y in x.free_symbols if isinstance(y, sp.Symbol)}
    return out


_SHOWN: dict = {}                                  # {a folded letter: its name as written} - for messages


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
          "theta": "θ", "vartheta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "rho": "ρ", "sigma": "σ",
          "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
          "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}


def _shown(k) -> str:
    """a letter for a message, as the student wrote it ('omega_n' -> 'ω_n')"""
    t = re.sub(r"^([A-Za-z]+)(?=_|$)", lambda m: _GREEK.get(m.group(1), m.group(1)), _SHOWN.get(k, str(k)))
    return re.sub(r"_\{([^{}]*)\}", r"_\1", t)           # ('V_{CC}' -> 'V_CC', as the formula is shown)


def _fold_case(stmts, alts=None, lect=None):
    """the letters of the statements in one case, as the LaTeX parser used to give them (V for v - a reader's or a
    student's slip), but for a name written in both cases in one statement ('A_v = -\\frac{R_C}{r_e + R_E}', 'T = 2t',
    'x = t/T'): two quantities then (they were merged: false sure wrong verdicts) -> (stmts, alts, lect)"""
    from dataclasses import replace
    from .symbolic import case_map
    groups = [_syms(s) for s in stmts] + [_syms(v) for vs in (alts or {}).values() for v in vs]
    groups.append({k for k in (lect or {}) if isinstance(k, sp.Symbol)})       # the lecturer's names together
    m = case_map(groups, fixed=set(_D_OPAQUE))
    _SHOWN.clear()                                 # (messages name a letter as the student wrote it: 'R=5000', not 'r=5000')
    _SHOWN.update({v: k.name for k, v in m.items()})
    if not m:
        return stmts, alts, lect

    def f(s):
        calc = s.calc
        if calc.get("indefinite"):
            calc = {**calc, "indefinite": [v.xreplace(m) if isinstance(v, sp.Basic) else v for v in calc["indefinite"]]}
        return replace(s, sides=[x.xreplace(m) if isinstance(x, sp.Basic) else x for x in s.sides],
                       symbol=s.symbol.xreplace(m) if isinstance(s.symbol, sp.Basic) else s.symbol, calc=calc)
    stmts = [f(s) for s in stmts]
    alts = {k: [f(v) for v in vs] for k, vs in alts.items()} if alts else alts
    lect = {(k.xreplace(m) if isinstance(k, sp.Basic) else k): f(v) for k, v in lect.items()} if lect else lect
    return stmts, alts, lect


def _chain_value(s):
    """a chain that works a letter out ('\\omega_n = \\sqrt{10} = 3.16', 'R = R_1 + R_2 = 5 k\\Omega') -> a value statement of
    its result, for the formulas after it (they found no value for it: 'cannot check'); None for any other statement"""
    if s.kind not in ("chain", "numeric") or len(s.sides) < 2 or _letters(s.sides[-1]) or not (
            isinstance(s.sides[0], sp.Symbol) or (isinstance(s.sides[0], sp.Abs) and isinstance(s.sides[0].args[0], sp.Symbol))):
        return None                                  # (a size too: '|Z| = \sqrt{R^2 + X_C^2} = 188 \Omega', for 'I = \frac{V}{|Z|}')
    v = _cnum(s.sides[-1])
    if v is None:
        return None
    v = v.real if abs(v.imag) < 1e-12 else v
    vt = s.side_texts[-1] if len(s.side_texts) == len(s.sides) else _fmt(v)
    unit = "" if re.fullmatch(r"\(.*\)\s*\\cdot\s*.+", vt.strip()) else s.unit     # (a last side already put in SI units:
    return Statement("value", f"{s.side_texts[0] if s.side_texts else s.sides[0]} = {vt}", [s.sides[0], s.sides[-1]], s.conf,   # '(20) \cdot 10^{-3}')
                     symbol=s.sides[0], number=v, value_text=vt, unit=unit, line=s.line, piece=s.piece, calc={"worked": True})


def _continued(stmts):
    """'I = \\frac{V}{R}', then '= \\frac{10}{5}' and '= 2 A' on lines of their own: one chain - 'I = \\frac{V}{R} = \\frac{10}{5}
    = 2 A' (a line that begins with '=' was dropped: its steps never checked); a line like that with nothing above it: none"""
    if not any(s.kind == "cont" for s in stmts):
        return stmts
    out = []
    for s in stmts:
        if s.kind != "cont":
            out.append(s)
            continue
        if not out:
            continue
        prev = out[-1]
        if prev.kind == "value" and not _EXPRESSION.search(prev.value_text.strip().lstrip("+-")):
            continue                                 # ('R = 5' just above: a datum - not what the '= 2 A' goes on from)
        try:
            joined = statements(f"{prev.text} {s.text}", conf=min(prev.conf, s.conf), line=s.line, piece=prev.piece)
        except Exception:                          # noqa: BLE001
            joined = []
        joined = [j for j in joined if j.kind != "cont"]
        if joined:
            out[-1:] = joined
    return out


def check_statements(stmts, alts=None, others=None, values=None):
    """statements in reading order -> checks, problem by problem (best readings only). alts: {line:
    [value statements]} read by the formula reader in pieces the layout took for text; others: {piece:
    [texts of its other readings]} (a letter foreign to the formula may be a digit there); values: the problem's
    data as the lecturer gives it ({'R_1': '2k', 'E': '10'}) - for a letter the student writes no value for"""
    checks = _check_statements_once(stmts, alts, others, values)
    return _slash_page(stmts, checks, alts, others, values)


def _slash_statements(s) -> list:
    """a statement read another way at a slash of one of its sides (_slash_alternatives): [(the statement so, how)]"""
    texts = s.side_texts
    if not texts or len(texts) != len(s.sides) or s.kind not in ("formula", "identity", "chain", "numeric", "value"):
        return []
    out = []
    for t in texts:
        if not isinstance(t, str) or "/" not in t or s.text.count(t) != 1:
            continue
        for alt, label in _slash_alternatives(t):
            got = statements(s.text.replace(t, alt), s.conf, s.line, s.piece)
            if len(got) == 1 and got[0].kind == s.kind and len(got[0].sides) == len(s.sides):
                out.append((got[0], label))
    return out[:4]


def _slash_page(stmts, checks, alts, others, values):
    """a page with a sure wrong where a slash leaves a choice ('\\int x dx = 1/2 x^2 + C', 'f(x) = 1/2 x^2' then 'f(2) = 2',
    'y = 1/2 x^2' then "y' = x": a half times x^2 - the parser takes 1/(2x^2), as it must '1/2\\pi f C'): judged again with
    that statement read the other way - kept when its wrongs are fewer and its rights more (the rights it adds not sure:
    which way the student meant it the arithmetic cannot tell); a formula's own answers are its problem's
    (_check_problem: _slash_formulas)"""
    bad = ("wrong", "wrong step")
    if not any(c.verdict in bad and c.sure for c in checks) or not any("/" in (s.text or "") for s in stmts):
        return checks
    tried = 0
    for k, s in enumerate(stmts):
        if "/" not in (s.text or ""):
            continue
        for s2, label in _slash_statements(s):
            if tried >= 6:
                return checks
            tried += 1
            more = _check_statements_once(stmts[:k] + [s2] + stmts[k + 1:], alts, others, values)
            if sum(c.verdict in bad for c in more) < sum(c.verdict in bad for c in checks) and \
                    sum(c.verdict == "right" for c in more) > sum(c.verdict == "right" for c in checks):
                # (a right it adds - not a reading that leaves the line unread: '325/sqrt(2) = 299' lost its wrong step)
                old = {(c.verdict, c.text) for c in checks}
                for c in more:
                    if c.verdict == "right" and (c.verdict, c.text) not in old and "read as" not in c.text:
                        _slash_mark(c, label)
                checks, stmts = more, stmts[:k] + [s2] + stmts[k + 1:]
                break
    return checks


def _data_below(body, snaps) -> list:
    """a chain whose letters get their values only below it ('I = \\frac{V}{R} = 3 mA', then 'V = 12 V', 'R = 4 k\\Omega' - the
    data listed after the result): its steps with them (it had none - no verdict); a wrong one not sure (a value of the
    next part?). snaps: [(chain, the data known at it, in SI units, their units)]"""
    out = []
    where = {id(x): k for k, x in enumerate(body)}
    for s, data, data_si, units in snaps:
        first = s.sides[0] if s.sides and isinstance(s.sides[0], sp.Symbol) else None
        lets = set().union(*(_letters(x) for x in s.sides[1:])) - {first}
        missing = lets - set(data)
        if not missing or _letters(s.sides[-1]) or id(s) not in where:
            continue
        later = {}
        for x in body[where[id(s)] + 1:]:
            if x.kind == "value" and x.symbol in missing and x.symbol not in later and not isinstance(x.number, complex):
                later[x.symbol] = x
            elif x.kind in ("chain", "numeric") and x.sides and x.sides[0] in missing:
                break                                # (worked out below: a chain of its own)
        if set(later) != missing:
            continue
        d, d_si, u = dict(data), dict(data_si), dict(units)
        for k, x in later.items():
            d[k] = _as_data(x.number)
            d_si[k] = d[k] * ((_si(x.unit) if x.unit.strip() and x.unit.strip() != "m" else None) or 1)
            u[k] = x.unit
        got = []
        _chain(s, d, got, [], d_si, _milli_data(d_si, u))
        for t, ok, va, vb in got:
            if not ok and not (isinstance(va, str) and va.startswith(_MAYBE)):
                va = _MAYBE + f"{_fmt(va)} is not {_fmt(vb)} - with values written below it"
            out.append((t, ok, va, vb))
    return out


def _check_statements_once(stmts, alts=None, others=None, values=None):
    """check_statements, the statements read as they are"""
    checks = []
    given, far = {}, {}
    lect = _lecturer_pool(values)
    if any(s.kind == "boolean" for s in stmts):      # a page in Boolean algebra: its plain lines read so too ('F = A + BC',
        stmts = _boolean_names(_boolean_page(stmts))   # 'Y = (A + B)(A + C) = A + BC' - a formula, a chain elsewhere)
    stmts = _continued(stmts)                      # (a chain going on at the next line's '=')
    stmts = _zero_instants(stmts)                  # ('v_C(0) = 5 V' of no function defined: a name - before the case is folded)
    stmts, alts, lect = _fold_case(stmts, alts, lect)
    stmts = _functions(stmts)
    probs = problems(stmts)
    for i, (formula, body) in enumerate(probs):    # 'x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2a}': the data and the
        nxt = probs[i + 1][0] if i + 1 < len(probs) else None     # answers follow the second formula - both use them
        if formula is not None and formula.calc.get("pm") == 1 and not body and nxt is not None and nxt.calc.get("pm") == -1:
            probs[i] = (formula, probs[i + 1][1])
    fed = set()
    for i, (formula, body) in enumerate(probs):    # 'P = I^2 R', then 'I = \frac{V}{R}' and the data: the second formula
        nxt = probs[i + 1] if i + 1 < len(probs) else None     # gives the first its letter - the data and answers below
        if formula is not None and formula.kind == "formula" and not body and nxt is not None and nxt[0] is not None and \
                nxt[0].kind == "formula" and len(nxt[0].sides) == 2 and nxt[0].sides[0].is_Symbol and \
                nxt[0].sides[0] in set().union(*(_letters(x) for x in formula.sides)) and \
                nxt[0].sides[0] not in _letters(nxt[0].sides[1]):
            probs[i] = (formula, [nxt[0]] + nxt[1])     # are both's (the second checked on its own too)
            fed.add(id(nxt[0]))
    pm_checks, last_got = {}, []
    for no, (formula, body) in enumerate(probs, 1):
        if formula is None:                        # arithmetic without a formula: its steps, with the data written
            steps, derived = [], []                  # above them ('a = 3', 'b = 4', then 'c = \sqrt{a^2+b^2} = 5')
            data = {k: _as_data(v.number) for k, v in lect.items()}
            data_si, units, prec = dict(data), {}, {}
            far.update(given)
            given = {}
            implied = {}                             # {letter: (value, chain)} a chain just above solved for it
            worked = set()                           # letters whose value a chain above worked out (not a datum)
            spans = []                               # [(chain, its steps' span)]
            bnames = {}                              # Boolean functions defined: {name: (expression, text)}
            snaps = []                               # [(chain, the data known at it: as written, in SI, units)]
            for s in body:
                if s.kind == "value" and s.symbol is not None:
                    worked.discard(s.symbol)
                    got = implied.get(s.symbol)       # '\frac{1}{R} = \frac{1}{6} + \frac{1}{3} = \frac{1}{2}', then 'R = 3': R = 2
                    f_u = _si(s.unit) if s.unit.strip() and s.unit.strip() != "m" else 1.0
                    if got is not None and not isinstance(s.number, complex) and not isinstance(got[0], complex) and \
                            f_u is not None and not _close(s.number, got[0], s.value_text) and \
                            not _close(s.number * f_u, got[0], s.value_text, f_u):    # (as written, or in SI: 'R = 2 k\Omega')
                        steps.append((f"{_pretty(got[1].text)} → {_pretty(s.text)}", False, got[0], s.number))
                    data[s.symbol] = _as_data(s.number)
                    data_si[s.symbol] = data[s.symbol] * (_si(s.unit) or 1)
                    units[s.symbol] = s.unit
                    given[s.symbol] = s
                    pr = _rel_half_unit(s.value_text, s.number)
                    if pr:
                        prec[s.symbol] = pr
                    implied = {}
                    continue
                implied = {}
                if s.kind in ("numeric", "chain"):
                    derived = []
                    spans.append((s, len(steps), None))
                    again = s.sides[0] if s.sides and isinstance(s.sides[0], sp.Symbol) and s.sides[0] in worked else None
                    kept = None
                    if again is not None:            # 'I = \frac{V}{R} = 6.5 mA', then 'I = \frac{V}{R} = 5 mA' (worked out
                        # again), or 'I = \frac{V}{R_2} = 2 mA' (the next part, its own I): its own value - not judged
                        # against the one above (a false wrong step), and the one the lines below it use
                        kept = (data.pop(again, None), data_si.pop(again, None), units.pop(again, None), prec.pop(again, None))
                        worked.discard(again)
                    _chain(s, dict(data), steps, derived, dict(data_si), _milli_data(data_si, units), prec,
                           {k for k, u in units.items() if _is_db(u)})
                    spans[-1] = (s, spans[-1][1], len(steps))
                    snaps.append((s, dict(data), dict(data_si), dict(units)))
                    f_d = _last_factor(s)
                    for sym, v, _, _ in derived:
                        if isinstance(sym, sp.Symbol) and not any(_letters(x) == {sym} for x in s.sides):
                            continue                 # (solved from a side with other letters, their values put in:
                            # 'V_{out} = V \frac{R_2}{R + R_2} = 5 \cdot \frac{1k}{4.7k + 1k}' with the V and R of a part
                            # above - an R_2 of nothing: a false wrong step at the next line that uses it)
                        if sym not in data:
                            worked.add(sym)
                        data.setdefault(sym, v)
                        if isinstance(sym, sp.Symbol) and not any(x == sym for x in s.sides) and sym not in data_si:
                            implied[sym] = (v, s)    # (from '1/R', 'R^2', '2R' - not a side that is R itself)
                        elif isinstance(sym, (sp.Symbol, sp.Abs)) and s.sides and s.sides[0] == sym and sym not in data_si and \
                                isinstance(v, (int, float, complex)):     # (a size too: '|Z| = ... = 4.14 k\Omega')
                            # 'I = \frac{V}{R_{eq}} = 208 \mu A', then 'P = I^2 R_{eq} = (208 \mu A)^2 \cdot 24 k\Omega': I in SI
                            # units too - the step was judged with I = 208 beside R_eq = 24000 only: a sure wrong
                            data_si[sym] = v * (f_d or 1)
                            units.setdefault(sym, s.unit)
                            pr = _rel_half_unit(s.side_texts[-1] if s.side_texts else "", v)
                            if pr:
                                prec[sym] = pr
                    if again is not None and again not in data and kept[0] is not None:    # (no value of its own: the
                        for d, v in zip((data, data_si, units, prec), kept):                  # one above stays)
                            if v is not None:
                                d[again] = v
                        worked.add(again)
                    cv = _chain_value(s)
                    if cv is not None:               # ('\omega_n = \sqrt{10} = 3.16', then '2\zeta\omega_n = 2': no value
                        given[cv.symbol] = cv        # for omega_n, before)
                elif s.kind == "boolean":
                    checks += _boolean_checks(no, s, bnames)
                elif s.kind == "identity":
                    checks += _identity_checks(no, s, [data, data_si])
                elif s.kind == "phasor":
                    checks += _phasor_checks(no, s)
                elif s.kind == "matrix":
                    checks += _matrix_checks(no, s)
                elif s.kind == "unit":
                    checks.append(Check(no, "wrong unit", s.calc["msg"], s.conf, False, [s.text], line=s.line))
            steps += _data_below(body, snaps)
            for a, b in zip(body, body[1:]):         # 'V = 5 \cdot 2 \Rightarrow V = 10': a line at a time (as under a
                if a.kind == b.kind == "value" and a.symbol == b.symbol and _EXPRESSION.search(a.value_text.strip().lstrip("+-")) and \
                        not _EXPRESSION.search(b.value_text) and not (a.calc.get("pm") and b.calc.get("pm")):   # formula)
                    steps.append((f"{_pretty(a.value_text)} = {_pretty(b.value_text)}",
                                  _close(b.number, a.number, b.value_text) or _close(a.number, b.number), a.number, b.number))
            _redone(spans, steps, body)
            for t, ok, va, vb in steps:
                checks.append(_step_check(no, t, ok, va, vb, min(s.conf for s in body), [s.text for s in body]))
            continue
        pre = []
        if formula.kind in ("chain", "numeric") and len(formula.sides) > 2 and isinstance(formula.sides[0], sp.Symbol) and \
                not _letters(formula.sides[-1]):
            # 'I_0 = \frac{V}{R} = 6.5 mA' starting a problem of its own (none of its letters the formula's above): its
            # steps, with the values known on the page (they were never checked - a slip missed); its value for the
            # lines under it
            got_steps = []
            _foreign_steps(formula, got_steps, {}, {k: v.number for k, v in lect.items()}, given, far, [])
            pre = [_step_check(no, t, ok, va, vb, formula.conf, [formula.text]) for t, ok, va, vb in got_steps]
            cv = _chain_value(formula)
            if cv is not None:
                given = {**given, cv.symbol: cv}
        got = pre + _check_problem(no, formula, body, alts, others, given, far, lect)
        if id(formula) in fed:                     # (a formula that gave the one above it its letter: no 'cannot check' of
            seen = {(c.verdict, c.text) for c in last_got if c.text.startswith("'")}   # its own for the answers that
            got = [c for c in got if c.verdict != "cannot check" and (c.verdict, c.text) not in seen]
            # were the other's - nor a step of the lines below both, judged there already: one slip, one wrong step)
        last_got = got
        if formula.calc.get("pm"):
            pm_checks[formula.calc["pm"]] = (got, body)
            if len(pm_checks) == 2:
                _pm_swapped(pm_checks)
                pm_checks = {}
        checks += got
        far.update(given)                          # later formulas may use them too, and this one's results -
        given = {}                                 # but only as suggestions
        for s in body:
            if s.kind == "value" and s.symbol is not None:
                far[s.symbol] = s
            elif _chain_value(s) is not None:        # a letter worked out under the formula
                far[s.sides[0]] = _chain_value(s)
    return checks


def _same_key(t: str) -> str:
    """two readings that differ only in spaces, \\left / \\right or how items are separated ('|', '\\vert', ',')
    say the same thing"""
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", t)
    t = re.sub(r"\\(?:vert|mid)(?![A-Za-z])|[|,;]", "|", t)
    return re.sub(r"\s+", "", t)


def _mix(first, second, w: float = 0.5):
    """two formula readers as two experts: P(reading) = (1 - w) P_first + w P_second over the readings either
    gave, normalised; readings that say the same (_same_key) are merged - where both readers agree, that
    reading gets both votes. Give each reader's own probabilities of its readings (exp of the log-probability,
    _raw): a reader that returns one reading must not count it as certain just because it is alone."""
    if not second:
        return first
    pool = {}
    for weight, rd in ((1 - w, first), (w, second)):
        for t, p in rd:
            k = _same_key(t)
            text, q = pool.get(k, (t, 0.0))
            pool[k] = (text, q + weight * p)
    z = sum(q for _, q in pool.values()) or 1.0
    return sorted(((t, q / z) for t, q in pool.values()), key=lambda x: -x[1])


def _raw(nbest):
    """[(text, logp)] -> [(text, the reader's probability of that exact reading)]"""
    return [(t, math.exp(min(0.0, lp))) for t, lp in nbest]


def check_pages(pages, nbest: int = NBEST, values: dict | None = None):
    """read_page.read_pages(...) -> the checks of every problem on the pages. The structure (which
    pieces form which problem) comes from the best readings; each problem's verdict is then weighed over
    the n-best readings of its pieces: its confidence is the probability of the readings that give it,
    times the layout confidence of the pieces. values: the problem's data ({'R_1': '2k', ...}) - for the circuits
    drawn (grader.circuit_check.with_values: a drawing that only names its parts is then judged with them) and the
    derivations (a letter the student writes no value for: the exam's own)"""
    from .mixed import readings as nbest_probs
    pieces, alts, line = [], {}, 0
    for page in pages:
        for segs in page["lines"]:
            for seg in sorted(segs, key=lambda s: s.box[0]):
                if seg.kind == "math":
                    pieces.append((line, seg, _mix(_raw(seg.nbest)[:nbest], _raw(seg.math2)[:nbest])
                                   if getattr(seg, "math2", None) else nbest_probs(seg.nbest)[:nbest]))
                elif getattr(seg, "formula_nbest", None):
                    text, p = nbest_probs(seg.formula_nbest)[0]
                    alts.setdefault(line, []).extend(v for v in statements(text, conf=p, line=line) if v.kind == "value")
            line += 1
    best = []
    for k, (ln, seg, rd) in enumerate(pieces):
        best += statements(rd[0][0], conf=1.0, line=ln, piece=k)
    best = _fold_case(best)[0]                     # (the problems found as the checks will see their letters)
    others = {k: [t for t, _ in rd] for k, (ln, seg, rd) in enumerate(pieces)}     # every reading of each piece
    out = []
    for no, (formula, body) in enumerate(problems(best), 1):
        used = sorted({s.piece for s in ([formula] if formula else []) + body})
        layout = min(pieces[k][1].p_kind for k in used) if used else 1.0
        tally, first, total = {}, None, 0.0
        combos = itertools.islice(itertools.product(*[range(len(pieces[k][2])) for k in used]), MAX_COMBOS)
        for combo in combos:
            p = 1.0
            stmts = []
            for k, j in zip(used, combo):
                text, pj = pieces[k][2][j]
                p *= pj
                stmts += statements(text, conf=1.0, line=pieces[k][0], piece=k)
            try:
                res = check_statements(stmts, alts, others, values)
            except Exception as ex:                # noqa: BLE001 - a reading the checks fail on: no verdict from it, and
                sys.stderr.write(f"selfcheck: {type(ex).__name__} on {[x.text for x in stmts]}\n")   # the page goes on
                res = [Check(no, "cannot check", f"the checks failed on this reading ({type(ex).__name__})", 0.0, False)]
            res = res or [Check(no, "nothing", "no problem found in this reading", 1.0, False)]
            main = next((c for c in reversed(res) if c.verdict != "misread?"), res[-1])
            for c in res:                        # the problem's verdict counts by its kind, a single step by its text
                key = c.verdict if c is main else (c.verdict, c.text)
                tally.setdefault(key, [0.0, c])[0] += p
            total += p
            if first is None:
                first, first_main = res, main
        total = total or 1.0
        for c in first or []:                      # the best reading's checks, with the verdict's probability
            if c.verdict == "nothing":
                continue
            key = c.verdict if c is first_main else (c.verdict, c.text)
            prob = 1.0 if c.verdict == "misread?" else tally.get(key, [0.0])[0] / total     # a flag, not a verdict
            c.problem = no
            c.confidence = min(c.confidence, prob * layout)
            c.sure = c.sure and c.confidence >= SURE
            if c.line < 0 and used:
                c.line = max(pieces[k][0] for k in used)
            out.append(c)
    # the equations against the circuits drawn on the pages (Kirchhoff, node voltages, ...): grader/circuit_check.py
    circuits = [c for page in pages for c in page.get("circuits", [])]
    if circuits:
        from .circuit_check import check_against, circuit_statements, with_values
        stmts = []
        for k, (ln, seg, rd) in enumerate(pieces):
            stmts += circuit_statements(rd[0][0], conf=rd[0][1] * seg.p_kind, line=ln, piece=k)
        per = []
        for n, c in enumerate(circuits, 1):
            try:
                per.append(check_against(stmts, with_values(c["netlist"], values), problem=f"circuit {n}"))
            except Exception as ex:                # noqa: BLE001 - (as above: that circuit's checks only are lost)
                sys.stderr.write(f"circuit_check: {type(ex).__name__} on circuit {n}\n")
        out += _across_circuits(per)
    return out


def _across_circuits(per: list) -> list:
    """the checks against each circuit on the pages -> one list: with two or more circuits (the problem's and an
    equivalent the student drew, or two questions' drawings) an equation right against one of them is right - its
    'wrong' against another dropped - and a 'wrong' is sure only when every circuit could judge it"""
    if len(per) <= 1:
        return [c for cs in per for c in cs]
    by = {}
    for cs in per:
        for c in cs:
            by.setdefault(c.pieces[0] if c.pieces else c.text, []).append(c)
    out = []
    for cs in by.values():
        rights = [c for c in cs if c.verdict in ("right", "right value, wrong unit")]
        if rights:                                 # the value right against one of the circuits
            out += rights
            continue
        judged = len({c.problem for c in cs if c.verdict in ("right", "wrong")})
        for c in cs:
            if c.verdict == "wrong":
                c.sure = c.sure and judged == len(per)
            out.append(c)
    return out
