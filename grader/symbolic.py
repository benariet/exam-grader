"""Formula equivalence: Math-Verify only *parses* the LaTeX into SymPy; the comparison is ours.

Why not Math-Verify's verify(): with 'I' on the left it treats any two equations as equal
('I = V/R' == 'I = 5'), because its LaTeX parser turns I into the imaginary unit. In circuit
exams I is the current (the imaginary unit is written j), so we map it back to a variable and
compare: same quantity on the left, right-hand sides equal symbolically or at random points.
"""
from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass

import sympy as sp
from math_verify import parse as _mv_parse

# Math-Verify guards every parse with a timeout: signal.alarm on Unix, but on Windows a new process per parse -
# slow, and it fails outright when the output is piped (WinError 6: every parse returns nothing). No timeout there.
_PARSE_TIMEOUT = None if os.name == "nt" else 5


def _extraction():
    """Math-Verify's default extraction, with its unit stripping off: it drops a trailing c, d, g, h, l, m, o, s or t
    as a unit - 'm g h' came out as m, '\\omega t' as omega, 'v_0 t' as v_0 (our own unit handling stays: '5 s',
    '10 V' are cut by the callers before parsing)"""
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    try:
        from latex2sympy2_extended.math_normalization import NormalizationConfig
    except ImportError:                                    # an older layout of the package
        from latex2sympy2_extended import NormalizationConfig
    return [LatexExtractionConfig(normalization_config=NormalizationConfig(basic_latex=True, units=False, malformed_operators=True,
                                                                           nits=True, boxed="all", equations=False)),
            ExprExtractionConfig()]


_EXTRACTION = None


def _keep_case():
    """the LaTeX parser lowercases every letter by default (latex2sympy's ConversionConfig): 'r_e + R_E' came out as
    2 r_e, 'T = 2t' as T = 2T, 't/T' as 1 - false sure 'wrong's in EE notation, where R_E and r_e, T and t, V and v are
    different quantities. Letters keep their case here; the checks fold it back where a letter is written in one case
    only (case_map) - as forgiving as before of a v read for V"""
    try:
        import math_verify.parser as mvp
        from latex2sympy2_extended.latex2sympy2 import ConversionConfig
        cfg = ConversionConfig(lowercase_symbols=False)
    except Exception:                                      # noqa: BLE001 - another version: its default
        return
    orig = mvp.latex2sympy
    if getattr(orig, "_keeps_case", False):
        return

    def l2s(*a, **k):
        k.setdefault("conversion_config", cfg)
        return orig(*a, **k)
    l2s._keeps_case = True
    mvp.latex2sympy = l2s
    if hasattr(mvp, "parse_latex_cached"):
        mvp.parse_latex_cached.cache_clear()


_keep_case()


def case_map(groups, fixed=()) -> dict:
    """{Symbol: Symbol} that lowercases every letter of the expressions, as the parser used to - but for names written
    in both cases within one group (one statement, one formula): two quantities then ('R_E' and 'r_e', 'T' and 't'),
    each kept as written. The current I (it comes through iota), a capital J (a current density) and the names in
    fixed stay as they are"""
    keep = set(fixed)
    for syms in groups:
        by = {}
        for y in syms:
            by.setdefault(y.name.lower(), set()).add(y.name)
        for names in by.values():
            if len(names) > 1:
                keep |= names
    out = {}
    for syms in groups:
        for y in syms:
            n = y.name
            if n in keep or n == n.lower() or n == "J" or re.match(r"I(?![a-z])", n):
                continue
            out[y] = sp.Symbol(n.lower(), **y.assumptions0)
    return out


def parse(text: str):
    global _EXTRACTION
    if _EXTRACTION is None:
        try:
            _EXTRACTION = _extraction()
        except Exception:                                  # noqa: BLE001 - another Math-Verify version: its defaults
            _EXTRACTION = False
    if _EXTRACTION:
        return _mv_parse(text, extraction_config=_EXTRACTION, parsing_timeout=_PARSE_TIMEOUT)
    return _mv_parse(text, parsing_timeout=_PARSE_TIMEOUT)

from .normalize import normalize_formula_text, strip_math_delims, undecorate


@dataclass
class SymbolicVerdict:
    status: str          # 'equivalent' | 'not_equivalent' | 'parse_error'
    detail: str
    student_expr: str | None = None
    reference_expr: str | None = None


_PLAIN_WORDS = {
    # plain-text spellings that LaTeX mode would split into single-letter symbols
    "arctan": r"\arctan", "arcsin": r"\arcsin", "arccos": r"\arccos", "atan": r"\arctan", "asin": r"\arcsin",
    "acos": r"\arccos", "sinh": r"\sinh", "cosh": r"\cosh", "tanh": r"\tanh",
    "pi": r"\pi", "omega": r"\omega", "tau": r"\tau", "sqrt": r"\sqrt", "alpha": r"\alpha",
    "beta": r"\beta", "theta": r"\theta", "phi": r"\phi", "mu": r"\mu", "sigma": r"\sigma",
    "sin": r"\sin", "cos": r"\cos", "tan": r"\tan", "exp": r"\exp", "ln": r"\ln", "log": r"\log",
}


_PAR_OPERAND = r"(\ue010\d+\ue011|\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)|\\left\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*?\\right\)|\\frac\{[^{}]*\}\{[^{}]*\}|[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?|\d+(?:\.\d+)?(?:\s*[kKM](?![A-Za-z]))?)"
_PAR_OP = r"\s*(?:\\\||\\parallel(?![A-Za-z])|\|\||∥)\s*"


def _parallel(s: str) -> str:
    """'R_1 \\| R_2' (or \\parallel, ||): resistances in parallel -> '\\frac{(R_1)(R_2)}{(R_1)+(R_2)}', left to right
    ('R_1 \\| R_2 \\| R_3'); the LaTeX parser took '2 \\| 3' for 3"""
    held = []
    while True:
        m = re.search(_PAR_OPERAND + _PAR_OP + _PAR_OPERAND, s)
        if not m:
            break
        a, b = m.group(1), m.group(2)
        held.append(f"\\left(\\frac{{({a})({b})}}{{({a})+({b})}}\\right)")
        s = s[:m.start()] + f"\ue010{len(held) - 1}\ue011" + s[m.end():]
    while "\ue010" in s:                                    # the placeholders back, innermost last
        s = re.sub(r"\ue010(\d+)\ue011", lambda m: held[int(m.group(1))], s)
    return s


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _after_groups(s: str) -> str:
    """a product sign after a fraction or a root that a number or another fraction follows ('\\frac{10}{2} 3', '\\sqrt{4}0.25',
    '\\frac{6}{3}\\frac{1}{2}'), and between a bracket and a fraction ('(3)\\frac{1}{2}') - not after an operator's limits
    ('\\lim_{x \\to 0} \\frac{\\sin x}{x}', '\\sin^{2} 30')"""
    if "frac" not in s and "sqrt" not in s:
        return s
    out, i = [], 0
    for m in re.finditer(r"\\(?:[dt]?frac|sqrt)(?![A-Za-z])", s):
        if m.start() < i:
            continue
        j, groups = m.end(), (2 if "frac" in m.group(0) else 1)
        k = j
        while k < len(s) and s[k] == " ":
            k += 1
        if "sqrt" in m.group(0) and k < len(s) and s[k] == "[":      # ('\\sqrt[3]{8}')
            e = s.find("]", k)
            if e < 0:
                continue
            k = e + 1
        ok, first = True, None
        for _ in range(groups):
            while k < len(s) and s[k] == " ":
                k += 1
            if k >= len(s) or s[k] != "{":
                ok = False
                break
            depth = 0
            for q in range(k, len(s)):
                depth += {"{": 1, "}": -1}.get(s[q], 0)
                if depth == 0:
                    first = s[k + 1:q] if first is None else first
                    k = q + 1
                    break
            else:
                ok = False
                break
        if not ok or ("frac" in m.group(0) and re.fullmatch(r"\s*(?:d|\\partial)\s*(?:\^\s*\{?\s*\d\s*\}?)?\s*\\?[A-Za-z]*\s*", first or "")):
            continue                                 # (a derivative's operator: '\\frac{d}{dx}\\frac{d}{dx} x^3')
        rest = s[k:]
        if re.match(r"\s*(?:\d|\\[dt]?frac(?![A-Za-z]))", rest):
            out.append(s[i:k] + " \\cdot ")
            i = k
    out.append(s[i:])
    s = "".join(out)
    return re.sub(r"(\)|\\right\s*\))\s*(?=\\[dt]?frac(?![A-Za-z]))", r"\1 \\cdot ", s)


def _radicals(s: str) -> str:
    """'√2', '√x', '√\\pi', '√(x+1)', '√(30^2 + (80 - 40)^2)' -> '\\sqrt{...}' - brackets nested inside too (they were not:
    '√((1 \\cdot 10^{-3}) \\cdot (1 \\cdot 10^{-6}))', its units put in, was read as 10 - a false sure wrong)"""
    out, i = [], 0
    while True:
        j = s.find("√", i)
        if j < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:j])
        k = j + 1
        while k < len(s) and s[k] == " ":
            k += 1
        if k < len(s) and s[k] in "({":
            depth, e = 0, -1
            for q in range(k, len(s)):
                if s[q] in "({[":
                    depth += 1
                elif s[q] in ")}]":
                    depth -= 1
                    if depth == 0:
                        e = q
                        break
            if e < 0:
                out.append("√")
                i = j + 1
                continue
            out.append("\\sqrt{" + s[k + 1:e] + "}")
            i = e + 1
            continue
        m = re.match(r"\d+(?:\.\d+)?|\\[A-Za-z]+|[A-Za-z](?:_\{[^{}]*\}|_[A-Za-z0-9])?", s[k:])
        if m:
            out.append("\\sqrt{" + m.group(0) + "}")
            i = k + m.end()
        else:
            out.append("√")
            i = j + 1


def _plain_text_math(s: str) -> str:
    """math typed as plain text, as a model solution or a reader may give it: '10⁻⁸', 'V²' -> 10^{-8}, V^{2}; '√2',
    '√(x+1)' -> \\sqrt{...}; 'R2', 'V1', 'i3' (no underscore) -> R_{2}, V_{1}, i_{3}; a number in scientific notation
    one number ('50 / 1.5·10^{-6}' is 50 over 1.5·10^{-6}, not (50 / 1.5)·10^{-6})"""
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^{" + m.group(0).translate(_SUPERSCRIPT) + "}", s)
    # a signed power without braces ('10^-3', 'e^-1000t', 'e^-\alpha t'): its sign, number and letters together - LaTeX
    # takes the sign alone ('2.55 \cdot 10^-3' was read as 10: a false wrong step); after a number only letters next to
    # it ('e^-1000t' - not the unit of '10^-3 A')
    s = re.sub(r"\^\s*([+-])\s*(\d+(?:\.\d+)?(?:(?:\\(?:alpha|beta|gamma|delta|zeta|theta|lambda|mu|omega|sigma|tau|phi|varphi)"
               r"(?![A-Za-z])|[A-Za-z])(?:_\{[^{}]*\}|_[A-Za-z0-9])?)*|(?:\s*(?:\\(?:alpha|beta|gamma|delta|zeta|theta|lambda|mu|"
               r"omega|sigma|tau|phi|varphi)(?![A-Za-z])|[A-Za-z])(?:_\{[^{}]*\}|_[A-Za-z0-9])?)+)",
               lambda m: "^{" + m.group(1) + m.group(2).strip() + "}", s)
    s = s.replace("**", "^")                             # '3**2': a power, as a program writes it (it was 4)
    s = s.replace("½", "\\frac{1}{2}").replace("¼", "\\frac{1}{4}").replace("¾", "\\frac{3}{4}").replace("⅓", "\\frac{1}{3}")
    s = s.replace("Δ", "\\Delta ")                       # 'ΔV' typed: one name, as '\Delta V' (it was Delta times V)
    s = _radicals(s)
    s = re.sub(r"(?<![A-Za-z\\])(log|lg)\s*(10|2)(?=\s*\()", lambda m: "\\log_{" + ("10" if m.group(1) == "lg" else m.group(2)) + "}", s)
    # ('log10(100)', 'log2(8)' typed: the base - it was log of 10 times 100, 3)
    s = re.sub(r"(?<![A-Za-z\\_{^0-9])([RVICLEPZXGYJKTNAQ]|[ivftxwkpzr])(\d{1,2})(?![\d.])", r"\1_{\2}", s)
    # ('f0', 't1', 'T2', 'N1', 'x1', 'w0' too - 'f0/Q' was f times 0: a false wrong step; not j: 'j4' is 4j)
    s = re.sub(r"(?<![\d.}])(\d+(?:\.\d+)?)\s*(?:\\cdot|\\times|\*)\s*10\s*\^\s*(\{\s*[+-]?\s*\d+\s*\}|[+-]?\d)",
               lambda m: m.group(0) if re.search(r"\^\s*\{?\s*[+-]?\s*$", s[:m.start()]) or
               (re.search(r"\(\s*$", s[:m.start()]) and re.match(r"\s*\)", s[m.end():]))     # (in its own brackets
               else f"({m.group(1)} \\cdot 10^{m.group(2)})", s)                     # already: '(2 \cdot 10^{3})')
    # (not a number that is a power itself: '10^3 \cdot 10^{-6}' is 10^3 times 10^-6 - it was read 10^(3 \cdot 10^-6), 1)
    return s


def _inner(t: str) -> str:
    """'{-\\pi/4}' -> '-\\pi/4' (one pair of braces around it all off)"""
    t = t.strip()
    return t[1:-1] if t.startswith("{") and t.endswith("}") else t


_MAG = r"(\d+(?:\.\d+)?|\\sqrt\s*\{[^{}]*\}|\\frac\s*\{[^{}]*\}\s*\{[^{}]*\})"     # a phasor's size ('\sqrt{2} \angle 45')


def _wrap(s: str) -> str:
    s = undecorate(strip_math_delims(s)).strip()          # '\underline{...}', 'I \overset{KVL}{=} ...', 'R := ...'
    typed = "\\" not in re.sub(r"\\cdot\s*10\s*\^", "", s)    # plain text as typed - before the steps below add commands
    #                                                  (a number's prefix put in SI units no command: '2*pi*(1 \cdot 10^{3})')
    # a power inside upright text ('2.25 \cdot 10^4 \mathrm{cm^{-3}}', '5 \mathrm{m^2}'): the text's - the parser dropped all
    # after it and more ('2.25 \cdot 10^4 \mathrm{cm^{-3}}' read as 10: a false wrong step)
    s = re.sub(r"\\(mathrm|text|textrm|rm)\s*\{\s*([^{}^]*?)\s*\^\s*(\{[^{}]*\}|[^{}\s])\s*\}",
               lambda m: "\\" + m.group(1) + "{" + m.group(2) + "}^" + m.group(3), s)
    s = re.sub(r"[·∙⋅•]", r" \\cdot ", s).replace("×", r" \times ").replace("÷", r" \div ").replace("−", "-").replace("–", "-")
    s = _plain_text_math(s)
    s = re.sub(r"\\sqrt\s*(\d+(?:\.\d+)?)", r"\\sqrt{\1}", s)       # '\sqrt 20': √20 - TeX's one token made it √2 · 0
    # 'x_1', '\omega_d t', 'r_e': a subscript without braces is its one character - the parser took '\omega_d t' for
    # omega_{dt} (d as a differential's) and failed on 'r_e', '\omega_e t' (e as Euler's number)
    s = re.sub(r"(?<!\\)_\s*([A-Za-z0-9])", r"_{\1}", s)
    # 'E_1', 'E_{th}', 'e_2': an EMF's, a node's name - the parser took E for Euler's number and dropped it with its
    # subscript ('E_2 + 3' read as 3; 'E_1 - I R_1' not read at all)
    s = re.sub(r"(?<![A-Za-z\\])([Ee])(?=\s*_\s*\{)", r"\\mathrm{E}", s)
    s = _parallel(s)
    # '\sin^2(30^\circ)': the 2 is the function's power, not a factor of '(30)' (the products below)
    s = re.sub(r"(\\(?:sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|ln|log))\s*\^\s*(-?\d)(?!\d)", r"\1^{\2}", s)
    # a lone capital I is the current: the LaTeX parser takes it for sqrt(-1) and loses its powers ('I^2 R' read as
    # I R) - it is parsed as iota, then mapped back (_ece)
    s = re.sub(r"\\Delta\s*([A-Za-z])(?![A-Za-z])(?:\s*_\s*\{?\s*(\w+)\s*\}?)?",
               lambda m: "\\Delta_{" + m.group(1) + (m.group(2) or "") + "}", s)      # '\Delta V': one name, not delta * V
    s = re.sub(r"(?<![A-Za-z\\])(?<!Delta_\{)I(?![A-Za-z])", r"\\iota ", s)
    # '\Gamma' (a reflection coefficient), '\gamma' (a propagation constant): the parser takes either for Euler's constant,
    # 0.577 ('\frac{1 + |\Gamma|}{1 - |\Gamma|}' worked out as 3.73 - a false wrong step), and fails on '\Gamma_L' - they
    # are parsed as \Upsilon and \varpi, then mapped back (_ece); not the gamma function ('\Gamma(5) = 24')
    # (a lowercase gamma before a bracket is a coefficient still - the body effect's '\\gamma(\\sqrt{2|\\Phi_F| + V_{SB}} -
    # \\sqrt{2|\\Phi_F|})' was the gamma function of the bracket: a false sure wrong on every body-effect threshold)
    s = re.sub(r"\\Gamma(?![A-Za-z])(?!\s*(?:\(|\\left\s*\())", r"\\Upsilon ", s)
    s = re.sub(r"\\gamma(?![A-Za-z])", r"\\varpi ", s)
    s = re.sub(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[eE]\s*([-+]?\d+)(?![\d.])",          # '1.5e-3': 1.5 times 10^-3, not e - 3 -
               lambda m: (f"({m.group(1)} \\cdot 10^{{{m.group(2)}}})" if re.search(r"/\s*$", m.string[:m.start()]) or
                          re.match(r"\s*\^", m.string[m.end():])
                          else f"{m.group(1)} \\cdot 10^{{{m.group(2)}}}"), s)          # one number after a slash: '12/4.7e3' is 12
    #                                                  over 4700 (it was (12/4.7) times 1000); before a power: '3.18e3^2' is
    #                                                  3180 squared (it was 3.18 times 10^6)
    s = re.sub(r"\)\s*\(|(?<=\d)(?<![\^_]\d)\s*\((?=[^()]*\))", lambda m: ") \\cdot (" if m.group(0).startswith(")") else " \\cdot (", s)   # '(2)(3)', '2(3)': products
    # (not after a limit or a power's digit: '\int_0^1 (2t)^2 dt' is no product of 1 and (2t))
    s = re.sub(r"\\pi\s*(?=\(|\\left\s*\()", r"\\pi \\cdot ", s)                       # '2\pi(50)': pi times 50
    # a number after a group ('\frac{10}{2} 3', '\frac{10}{1} 0.5', '\frac{1}{2}\sqrt{4} 0.25'), a fraction after a fraction or
    # a bracket ('\frac{6}{3}\frac{1}{2}', '(3)\frac{1}{2}'): products - the parser took 10/2 and 3 for 8 (a mixed number -
    # that is a number before a fraction: '2 \frac{1}{2}'), 6/3 and 1/2 for 2 1/2, and split 0.5 into 0 times .5
    s = _after_groups(s)
    s = re.sub(r"\\right\s*\)\s*\\left\s*\(", r"\\right) \\cdot \\left(", s)            # '\left(50\right)\left(0.1\right)'
    s = re.sub(r"(?<=\d)(?<![\^_]\d)\s*\\left\s*\(", r" \\cdot \\left(", s)              # '2\left(3\right)'
    s = re.sub(_MAG + r"\s*\\angle\s*(\{\s*[+-]?[^{}]*\\pi[^{}]*\}|[+-]?\s*\\frac\s*\{[^{}]*\\pi[^{}]*\}\s*\{[^{}]*\}|"
               r"[+-]?\s*(?:\d+(?:\.\d+)?\s*)?\\pi(?:\s*/\s*\d+)?)",
               lambda m: f"{m.group(1)} \\cdot (\\cos({_inner(m.group(2))}) + j \\sin({_inner(m.group(2))}))", s)
    # ('\sqrt{2} \angle \frac{\pi}{4}': an angle in terms of pi is in radians - it was not read at all)
    s = re.sub(_MAG + r"\s*\\angle\s*(?:\{\s*([+-]?\s*\d+(?:\.\d+)?)\s*(?:\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°)?\s*\}|"
               r"([+-]?\s*\d+(?:\.\d+)?))\s*(?:\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°)?",
               lambda m: f"{m.group(1)} \\cdot (\\cos({(m.group(2) or m.group(3)).replace(' ', '')}^\\circ) + j \\sin({(m.group(2) or m.group(3)).replace(' ', '')}^\\circ))",
               s)                                                           # '2 \angle -53.13^\circ': a phasor, in degrees
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°)",
               r"(\1 \\cdot \\frac{\\pi}{180})", s)                                  # '30^\circ' in radians: sin(30°) = 0.5
    # (a '}' after '^\circ' is consumed only with its own '{': '\frac{10\angle 0^\circ}{2\angle -30^\circ}' lost its
    # numerator's brace and was not read at all)
    s = re.sub(r"(\\(?:sin|cos|tan|cot|sec|csc)(?:\^\{-?\d\})?)\s*(\(\d+(?:\.\d+)? \\cdot \\frac\{\\pi\}\{180\}\))", r"\1(\2)", s)
    # ('\sin 30^\circ', '\sin^2 30^\circ': the angle in degrees is the function's argument)
    if typed:
        # plain-text answer (e.g. from a typed transcription): 'pi' -> '\pi', 'sqrt(x)' -> '\sqrt{x}'; also when the steps
        # above put commands in ('2·sin(30°)': '\cdot', the degrees - its sin was read as letters)
        for w, tex in _PLAIN_WORDS.items():
            s = re.sub(rf"(?<![A-Za-z\\]){w}(?![A-Za-z])", tex.replace("\\", "\\\\"), s)
        s = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"\\sqrt{\1}", s)
    return f"${s}$"


def _first_sympy(parsed):
    for p in parsed:
        if isinstance(p, sp.Basic):
            return p
    return None


CURRENT = sp.Symbol("I")


def _ece(e):
    """ECE convention: 'I' / 'i' is the current, not sqrt(-1) (a lone I comes in as iota: see _wrap)."""
    if not isinstance(e, sp.Basic):
        return e
    back = {x: (CURRENT if x.name == "iota" else sp.Symbol("I" + x.name[4:])) for x in e.free_symbols
            if isinstance(x, sp.Symbol) and x.name.startswith("iota")}
    back.update({x: sp.Symbol(("Gamma" + x.name[7:]) if x.name.startswith("Upsilon") else ("gamma" + x.name[5:]))
                 for x in e.free_symbols if isinstance(x, sp.Symbol) and x.name.startswith(("Upsilon", "varpi"))})   # (_wrap)
    return e.xreplace({sp.I: CURRENT, **back})


def _rhs(e):
    """If the student wrote 'f_c = ...' or 'H(s) = ...' keep only the right-hand side.
    Only when the left-hand side is a bare name — 'lim ... = -∞' stays a full equation."""
    if isinstance(e, sp.Equality) and _is_name(e.lhs):
        return e.rhs
    return e


def _rhs_lhs(e):
    """The name on the left of 'name = ...', or None if e is not such an equation."""
    if isinstance(e, sp.Equality) and _is_name(e.lhs):
        return e.lhs
    return None


def _is_name(x) -> bool:
    """A quantity name on the left of '=': a symbol, f(x), or 'I' - which the LaTeX parser turns
    into the imaginary unit, but in circuit exams is the current."""
    return (isinstance(x, sp.Symbol) or
            (isinstance(x, sp.Function) and isinstance(x.func, sp.core.function.UndefinedFunction)))


def _numeric_equivalent(a: sp.Basic, b: sp.Basic, trials: int = 8, tol: float = 1e-6) -> bool:
    syms = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(0)
    ok = 0
    for _ in range(trials):
        subs = {s: rng.uniform(0.5, 3.0) for s in syms}  # positive values: fine for R, C, L, f...
        try:
            va = complex(a.evalf(subs=subs))
            vb = complex(b.evalf(subs=subs))
        except Exception:  # noqa: BLE001
            return False
        if not all(math.isfinite(v) for v in (va.real, va.imag, vb.real, vb.imag)):
            return False
        if abs(va - vb) <= tol * max(1.0, abs(va), abs(vb)):
            ok += 1
    return ok == trials


def brackets_balanced(s: str) -> bool:
    """Cheap sanity check: a recognizer that emits unbalanced brackets misread something,
    even if the parser manages to make sense of the string."""
    s = s.replace(r"\{", "").replace(r"\}", "")
    for a, b in (("{", "}"), ("(", ")"), ("[", "]")):
        if s.count(a) != s.count(b):
            return False
    # \left ... \right  (not \leftarrow / \rightarrow)
    if len(re.findall(r"\\left(?![A-Za-z])", s)) != len(re.findall(r"\\right(?![A-Za-z])", s)):
        return False
    return True


def compare_formula(student: str, reference: str) -> SymbolicVerdict:
    if not brackets_balanced(strip_math_delims(student)):
        return SymbolicVerdict("parse_error", f"unbalanced brackets in {student!r}")
    try:
        ref_parsed = parse(_wrap(normalize_formula_text(reference)))
        ref_e = _first_sympy(ref_parsed)
        if ref_e is None:
            raise ValueError("reference did not parse")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"bad reference formula {reference!r}: {e}") from e

    try:
        st_parsed = parse(_wrap(normalize_formula_text(student)))
    except Exception as e:  # noqa: BLE001
        return SymbolicVerdict("parse_error", f"could not parse {student!r}: {e}")
    st_e = _first_sympy(st_parsed)
    if st_e is None:
        return SymbolicVerdict("parse_error", f"could not parse {student!r}")

    st_e, ref_e = _ece(st_e), _ece(ref_e)
    fold = case_map([st_e.free_symbols, ref_e.free_symbols]) if isinstance(st_e, sp.Basic) and isinstance(ref_e, sp.Basic) else {}
    if fold:                                       # 'v = IR' for 'V = IR' as before; 't/T' stays two letters
        st_e, ref_e = st_e.xreplace(fold), ref_e.xreplace(fold)

    # same quantity on the left ('I = V/R' must not match 'V = V/R')
    ls, lr = _rhs_lhs(st_e), _rhs_lhs(ref_e)
    if ls is not None and lr is not None and str(ls).lower() != str(lr).lower():
        return SymbolicVerdict("not_equivalent", f"different quantity on the left ({ls} vs {lr})", str(st_e), str(ref_e))
    a, b = _rhs(st_e), _rhs(ref_e)
    if isinstance(a, sp.Equality) or isinstance(b, sp.Equality):
        # e.g. '1 = V/R' (left side is not a name) or 'lim ... = -oo': compare whole equations
        if not (isinstance(a, sp.Equality) and isinstance(b, sp.Equality)):
            return SymbolicVerdict("not_equivalent", "equation vs. expression", str(a), str(b))
        same = False
        try:                                       # side by side first: 'lim ... = -oo' minus itself is oo - oo, not 0
            same = all(x == y or sp.simplify(x - y) == 0 for x, y in ((a.lhs, b.lhs), (a.rhs, b.rhs))) or \
                all(x == y or sp.simplify(x - y) == 0 for x, y in ((a.lhs, b.rhs), (a.rhs, b.lhs)))
        except Exception:  # noqa: BLE001
            pass
        a, b = a.lhs - a.rhs, b.lhs - b.rhs
        try:
            same = same or sp.simplify(a - b) == 0 or sp.simplify(a + b) == 0
        except Exception:  # noqa: BLE001
            pass
        return SymbolicVerdict("equivalent" if same else "not_equivalent",
                               "equations agree" if same else "equations differ", str(st_e), str(ref_e))
    try:
        if sp.simplify(a - b) == 0:
            return SymbolicVerdict("equivalent", "sympy: simplify(a-b)==0", str(a), str(b))
    except Exception:  # noqa: BLE001
        pass
    if a.free_symbols == b.free_symbols and _numeric_equivalent(a, b):
        return SymbolicVerdict("equivalent", "numeric sampling agrees at 8 random points", str(a), str(b))

    missing = b.free_symbols - a.free_symbols
    extra = a.free_symbols - b.free_symbols
    hint = ""
    if missing or extra:
        hint = f" (missing symbols: {sorted(map(str, missing))}, unexpected: {sorted(map(str, extra))})"
    return SymbolicVerdict("not_equivalent", "expressions differ" + hint, str(a), str(b))
