"""The student's equations against the circuit he drew (the netlist of grader/circuit.py, read by
recog/circuit_reader.py) - no answer key needed:

  * Kirchhoff's current law at a node where every current is named: 'I1 + I3 = I2' is checked against the
    arrows drawn (I1 = I2 + I3 there is wrong);
  * a node voltage or a named current written in the parts' names or values ('V_A = R2 (E1 R3 + E2 R1) / (R1 R2 +
    R1 R3 + R2 R3)', 'I_2 = 3 mA') is checked against the circuit's own solution (modified nodal analysis).

An equation in other terms (a loop's voltages, an intermediate step) is not judged here. A circuit whose reading has
problems (Netlist.problems: an unread polarity, a loose wire) gives suggestions for the human who checks, not
verdicts.

    from grader.circuit_check import check_against
    checks = check_against(statements, netlist)          # [grader.selfcheck.Check]
"""
from __future__ import annotations

import re

import sympy as sp

from .circuit import Netlist
from .normalize import undecorate
from .selfcheck import REL_TOL, SURE, Check, _fmt, _num

LABEL_SURE = 0.9                # a value read off the drawing this sure may decide a verdict


def _group(t: str, i: int):
    """the {...} group starting at t[i] (spaces before it skipped) -> (its inside, index after it); a single token
    else - a command ('\\pi') whole"""
    while i < len(t) and t[i] == " ":
        i += 1
    if i < len(t) and t[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", t[i:])
        if m:
            return m.group(0)[1:], i + m.end()
    if i >= len(t) or t[i] != "{":
        return (t[i], i + 1) if i < len(t) else ("", i)
    depth = 0
    for j in range(i, len(t)):
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        if depth == 0:
            return t[i + 1:j], j + 1
    return t[i + 1:], len(t)


def _plain(tex: str) -> str:
    """LaTeX of a circuit equation's side -> Python syntax: '\\frac{R_2 (E_1 R_3)}{R_1 + R_2}' -> '((R_2 (E_1 R_3))/(R_1 + R_2))'"""
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])|\\[,;! ]|\\(?:mathrm|text|rm|mathit)(?=\s*\{)", "", tex)
    t = re.sub(r"\\(tau|omega|alpha|beta|theta|phi|lambda|gamma|delta|sigma|zeta|xi)(?![A-Za-z])", r" \1 ", t)   # names
    t = re.sub(r"\s+(?=[_^])", "", t)                 # the formula reader spaces its tokens: 'I _ { 1 }'
    t = re.sub(r"(?<=[\d.])(?<![\^_]\d)\s+(?=[\d.])", "", t)     # '1 0' -> '10' (not 'x^2 3': a power, then 3)
    t = re.sub(r"\\(?:cdot|times)(?![A-Za-z])", "*", t)
    t = re.sub(r"\\div(?![A-Za-z])|÷", "/", t)
    out, i = [], 0
    while i < len(t):
        if t.startswith("\\frac", i) or t.startswith("\\dfrac", i):
            i += 6 if t.startswith("\\dfrac", i) else 5
            a, i = _group(t, i)
            b, i = _group(t, i)
            out.append(f"(({_plain(a)})/({_plain(b)}))")
        elif t.startswith("\\sqrt", i):                   # \sqrt{x}, \sqrt[3]{x}
            i += 5
            n = None
            if i < len(t) and t[i] == "[":
                j = t.index("]", i) if "]" in t[i:] else len(t)
                n, i = t[i + 1:j], j + 1
            a, i = _group(t, i)
            out.append(f"(({_plain(a)})**(1/({n})))" if n else f"sqrt({_plain(a)})")
        elif t[i] == "_":                                  # a subscript joins its name: R_{2} -> R_2
            g, i = _group(t, i + 1)
            out.append("_" + re.sub(r"\W", "", _plain(g)) + " ")
        elif t[i] == "^":
            g, i = _group(t, i + 1)
            out.append(f"**({_plain(g)})")
        elif t[i] in "{}":
            out.append("(" if t[i] == "{" else ")")
            i += 1
        else:
            out.append(t[i])
            i += 1
    return "".join(out)


_ANGLE = re.compile(r"(?P<amp>\d+(?:\.\d+)?)\s*\\angle\s*(?P<sign>[+-]?)\s*(?P<ang>\d+(?:\.\d+)?)\s*"
                    r"(?:\^\s*\{?\s*\\circ\s*\}?|°)?")


_UNIT_END = re.compile(r"(?:\\[,;:! ]|\\quad|\s)*\[?\s*(?:\\(?:mathrm|text|rm)\s*\{\s*)?(?P<prefix>\\mu|[pnuµmk]?)"
                       r"(?P<unit>V|A)\s*\}?\s*\]?\s*$")
_WHEN = re.compile(r"(?:,|\\quad|\\qquad|\\[,;]|\s)*(?:\\(?:text|mathrm)\s*\{\s*for\s*\}|for)?\s*t\s*"
                   r"(?:>|\\geq?|\\geqslant|≥|>=)\s*0\s*(?:\^\s*\{?\s*\+\s*\}?)?\s*$")


def _timed_text(tex: str) -> tuple[str, float]:
    """a function of time as written after a switching -> (text to parse, the factor its unit gives): 'for t > 0'
    and u(t) (1 after t = 0) dropped, a unit at its end taken off ('5(1 - e^{-t/\\tau}) mA' -> 0.001)"""
    t = _WHEN.sub("", tex)
    t = re.sub(r"(?:\\cdot|\*)?\s*(?:\\mathrm\s*\{\s*u\s*\}|u)\s*(?:\\left)?\s*\(\s*t\s*(?:\\right)?\s*\)", " ", t)
    t = re.sub(r"\\exp(?![A-Za-z])", " exp", t)
    factor = 1.0
    m = _UNIT_END.search(t)
    if m and re.search(r"(?<![A-Za-z\\])t(?![A-Za-z])", t[:m.start()]) and \
            re.search(r"[\d)}]\s*$|\\[,;:! ]\s*$|\s$", t[:m.start()] + " "):
        factor = _PREFIX[m.group("prefix")]
        t = t[:m.start()]
    return t, factor


def _split_names(py: str, known) -> str:
    """'R C' written together ('RC', 'R_1C_1' is fine already) -> 'R C' when each piece names a part"""
    known = {k for k in known if k}

    def split(m):
        w = m.group(0)
        if _canon(w) in known or w in ("exp", "tau", "t", "e", "pi", "cos", "sin", "omega"):
            return w
        out, i = [], 0
        while i < len(w):
            for j in range(len(w), i, -1):                 # the longest piece that names a part
                if _canon(w[i:j]) in known and (j == len(w) or not w[j].isdigit()):
                    out.append(w[i:j])
                    i = j
                    break
            else:
                return w
        return "(" + "*".join(out) + ")"
    return re.sub(r"(?<![A-Za-z_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)?(?![A-Za-z0-9_])", split, py)


_INLINE = re.compile(r"(?<![A-Za-ik-z_\d.\\])(?P<num>\d+(?:\.\d+)?)\s*(?P<rm>\\(?:mathrm|text|rm)\s*\{\s*)?"
                     r"(?P<prefix>\\mu\s*|[pnuµmkKMG])?(?P<unit>\\Omega|Ω|V|A|F|H|W)?(?(rm)\s*\})"
                     r"(?![A-Za-z0-9_])")


def _inline_units(t: str, k_is_name: bool = False) -> str:
    """'\\frac{10 - 0.7}{2k}' -> '\\frac{10 - 0.7}{(2*1000.0)}', '5mA + 2mA': a number with a prefix or a unit
    inside an expression, in the base unit (not s: that is the Laplace variable). k_is_name: a MOSFET's K is in the
    circuit - '2K V_{ov}', '2K(V_{GS} - V_t)' are that K, not kilo"""
    def rep(m):
        if not m.group("prefix") and not m.group("unit"):
            return m.group(0)
        if k_is_name and (m.group("prefix") or "").strip() in ("k", "K") and not m.group("unit") and \
                re.match(r"\s*(?:V|\(|\\left)", t[m.end():]):
            return m.group(0)
        f = _PREFIX[(m.group("prefix") or "").strip()]
        return f"({m.group('num')}*{f!r})" if f != 1.0 else m.group("num")
    return _INLINE.sub(rep, re.sub(r"(?<=\d)\s*\\[,;:! ]\s*", " ", t))


def parse_side(tex: str, ac: bool = False, timed: bool = False, known=(), k_is_name: bool = False, radians: bool = False):
    """one side of a circuit equation -> SymPy, every name a plain symbol (I is a current, E a source, not
    sqrt(-1) or e); None when it cannot be read. ac: a sinusoidal circuit's - j is sqrt(-1), '5 \\angle 30^\\circ' a
    phasor (degrees), \\cos / \\sin of t, e^{j...}. timed: a function of t after a switching - e is e, u(t) is 1,
    \\tau a name. known: the parts' names (canonical), to split 'RC' into R times C"""
    from sympy.parsing.sympy_parser import (implicit_multiplication, parse_expr, standard_transformations)
    factor = 1.0
    tex = tidy(tex)
    if timed:
        tex, factor = _timed_text(tex)
        tex = re.sub(r"(?<![A-Za-z\\\d.])(\d+(?:\.\d+)?)\s*(m|\\mu\s*|µ|u|n)?\s*s(?![A-Za-z])",      # 'e^{-t/1 ms}':
                     lambda m: repr(float(m.group(1)) * {"m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9}.get(           # seconds
                         (m.group(2) or "").replace("\\mu", "µ").strip(), 1.0)), tex)                             # in time
    else:                                                  # a unit after an expression: '3.61\\angle 33.7^\\circ \\Omega'
        m = re.search(r"\s*(?:\\mu\s*|[pnuµmk])?\s*(?:\\Omega|Ω|V|A|W|VA|VAR)\s*$", tex)
        if m and re.search(r"(?:[\d}°)]|\\circ)\s*$", tex[:m.start()]) and \
                re.search(r"\\angle|∠|\bj|[+*/(]", tex[:m.start()]):
            unit = tex[m.start():].strip()
            pre = re.match(r"(\\mu|[pnuµmk])", unit)
            if pre and unit not in ("m",):
                factor = _PREFIX[pre.group(1)]
            tex = tex[:m.start()]
    t = _inline_units(re.sub(r"(?<=\d)\s+(?=[\d.])", "", tex), k_is_name)
    t = re.sub(r"_\s*\{?\s*\\([A-Za-z]+)\s*\}?", r"_{\1}", t)          # r_\pi, r_{\pi}: a name
    if timed and not ac:                                   # cos / sin of t (a second-order response), degrees
        t = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", r" \\pi / 180", t)
        t = re.sub(r"\\(cos|sin|pi)(?![A-Za-z])", r" \1 ", t)
    if ac:                                                 # an angle as a fraction of pi: '\\angle -\\frac{\\pi}{4}', '\\angle \\pi/3'
        deg = lambda m: f"\\angle {m.group(1)}{float(m.group(2) or 1) * 180 / float(m.group(3) or 1):g}^\\circ"   # noqa: E731
        t = re.sub(r"\\angle\s*([+-]?)\s*\\frac\s*\{\s*(\d*(?:\.\d+)?)\s*\\pi\s*\}\s*\{\s*(\d+(?:\.\d+)?)\s*\}", deg, t)
        t = re.sub(r"\\angle\s*([+-]?)\s*(\d*(?:\.\d+)?)\s*\\pi\s*(?:/\s*(\d+(?:\.\d+)?))?(?![A-Za-z])", deg, t)
        t = _ANGLE.sub(lambda m: f"({m.group('amp')} e^{{j*({m.group('sign')}{m.group('ang')})" +
                       ("" if radians and "circ" not in m.group(0) and "°" not in m.group(0) else "*\\pi/180") + "})", t)
        t = re.sub(r"(?<![A-Za-z_\\])j\s*(?=[\d.(])", "j*", t)                  # 'j4' = j times 4
        t = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", r" \\pi / 180", t)            # degrees
        t = re.sub(r"\\(cos|sin|pi)(?![A-Za-z])", r" \1 ", t)
    t = re.sub(r"\\parallel(?![A-Za-z])|\\\||\|\||∥", " ∥ ", t)          # R_C \| R_L: in parallel
    t = re.sub(r"\\pi(?![A-Za-z])", " pi ", t)
    py = _plain(t)
    if "\\" in py:
        return None                                        # a command we do not know
    if known:
        py = _split_names(py, known)
    if "∥" in py:
        operand = r"(?:[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?(?:\*[\d.e+-]+)?|\((?:[^()]|\([^()]*\))*\))"
        while True:
            new = re.sub(rf"({operand})\s*∥\s*({operand})", r"(par(\1,\2))", py, count=1)
            if new == py:
                break
            py = new
        if "∥" in py:
            return None
    names = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", py)) - {"par", "sqrt"}
    local = {n: sp.Symbol(n) for n in names}
    local["par"] = lambda a, b: a * b / (a + b)
    local["sqrt"] = sp.sqrt
    if "pi" in names and not ac:
        local["pi"] = sp.pi
    if ac:
        local.update({"j": sp.I, "e": sp.E, "cos": sp.cos, "sin": sp.sin, "pi": sp.pi, "t": sp.Symbol("t")})
    if timed:
        local.update({"e": sp.E, "exp": sp.exp, "t": sp.Symbol("t"), "cos": sp.cos, "sin": sp.sin, "pi": sp.pi})
    try:
        e = parse_expr(py, local_dict=local, transformations=standard_transformations + (implicit_multiplication,))
    except Exception:                                      # noqa: BLE001
        return None
    if not isinstance(e, sp.Expr):
        return None
    return e * factor if factor != 1.0 else e


def parse_bool(tex: str):
    """a Boolean expression as students write it -> SymPy logic, or None: A B or A \\cdot B (and), A + B (or),
    \\overline{X} / \\bar{X} / X' / \\neg X / !X (not), A \\oplus B (xor), \\land / \\lor / \\wedge / \\vee, 0 / 1; a variable
    is a letter, maybe with a subscript"""
    t = re.sub(r"\\(?:neg|lnot|sim)(?![A-Za-z])|\u00ac|!(?=\s*[A-Za-z01(\\])", "~", tex)   # \neg A, \lnot A, \sim A, !A
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])|\\[,;! ]|\s+", "", t)
    out, i = [], 0
    while i < len(t):                                      # \overline{...} / \bar{...} -> ~( ... )
        if t.startswith("\\overline", i) or t.startswith("\\bar", i):
            i += 9 if t.startswith("\\overline", i) else 4
            g, i = _group(t, i)
            inner = parse_bool(g)
            if inner is None:
                return None
            out.append("~(" + str(inner).replace(" ", "") + ")")
            continue
        out.append(t[i])
        i += 1
    t = "".join(out)
    for a_, b_ in (("\\cdot", "&"), ("\\wedge", "&"), ("\\land", "&"), ("\\times", "&"), ("\\ast", "&"),
                   ("*", "&"), ("\u00b7", "&"), ("\u2227", "&"), ("\\oplus", "^"), ("\u2295", "^"),
                   ("\\vee", "|"), ("\\lor", "|"), ("\u2228", "|"), ("+", "|"), ("{", "("), ("}", ")")):
        t = t.replace(a_, b_)
    if "\\" in t:
        return None
    pat = r"[A-Za-z](?:_(?:\([0-9A-Za-z]+\)|[0-9A-Za-z]))?|[01]|[&|^~()']"
    tokens = re.findall(pat, t)
    if "".join(tokens) != t:
        return None
    operand = lambda x: bool(re.match(r"[A-Za-z01]", x))       # noqa: E731
    res = []
    for tok in tokens:
        if tok == "'":                                     # postfix ' : 'not' around the last operand
            if not res:
                return None
            if res[-1] == ")":
                depth, j = 0, len(res) - 1
                while j >= 0:
                    depth += {")": 1, "(": -1}.get(res[j], 0)
                    if depth == 0:
                        break
                    j -= 1
                res[j:] = ["~("] + res[j:] + [")"]
            else:
                res[-1:] = ["~", res[-1]]
            continue
        if res and (res[-1] == ")" or operand(res[-1])) and (tok in ("(", "~") or operand(tok)):
            res.append("&")                                # AB, A(B+C): an implicit 'and'
        res.append(tok)
    names, parts_ = {}, []
    for tok in res:
        if re.match(r"[A-Za-z]", tok):
            n = re.sub(r"[()]", "", tok)
            names[n] = sp.Symbol(n)
            parts_.append(n)
        elif tok in ("0", "1"):
            parts_.append("false_" if tok == "0" else "true_")
        else:
            parts_.append(tok)
    try:
        e = sp.sympify("".join(parts_), locals=names | {"true_": sp.true, "false_": sp.false}, convert_xor=False)
    except Exception:                                      # noqa: BLE001
        return None
    return e if isinstance(e, sp.logic.boolalg.Boolean) else None


_SMALL_SIGNAL = re.compile(r"r\s*_\s*\{?\s*(?:e|o|d|ds|b|bb|x|\\?pi|π|\\?mu|μ)\s*\}?")


def _canon(name: str) -> str:
    """'I_{1}', 'I_1', 'I1' -> 'i1'; 'V_A' -> 'va'. A small-signal resistance by its lowercase name ('r_e', 'r_\\pi', 'r_o')
    is its own quantity, not the resistor R_E drawn beside it ('r_e = 25 Ω' was judged against R_E = 1 kΩ: wrong)"""
    n = str(name).strip()
    if _SMALL_SIGNAL.fullmatch(n):
        return "ss" + re.sub(r"[_{}\s\\]", "", n).lower()
    return re.sub(r"[_{}\s\\]", "", n).lower()


def _kcl_exprs(net):
    out = {}
    for node, eq in net.kcl().items():
        left, right = eq.split("=")
        loc = {t: sp.Symbol(t) for t in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", eq)}
        out[node] = (eq, sp.sympify(left, locals=loc) - sp.sympify(right, locals=loc))
    return out


_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "\\mu": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6}
# a unit in square brackets, as Israeli exams write it: '3.3 [V]', '2 \\left[ mA \\right]', '[k\\Omega]'
_BRACKET_UNIT = re.compile(r"(?:\\left\s*)?\[\s*(?:\\(?:mathrm|text|rm)\s*\{\s*)?(?P<u>(?:\\mu\s*|[pnuµmkKM])?\s*"
                           r"(?:V\s*A\s*R|V\s*A|var|V|A|W|F|Hz|H|S|J|C|sec|s|\\Omega|Ω|ohm|rad\s*/\s*s|d\s*B))"
                           r"\s*\}?\s*(?:\\right\s*)?\]")
# a numeric expression with its unit after it: '\\frac{10}{3} V', '(10 - 0.7) / 2 mA'
# |V_A|, |Z_{in}|, |I|: a quantity's magnitude (the bars are not statement separators)
_MAGNITUDE = re.compile(r"(?:\\left\s*)?\|\s*(?P<n>[A-Za-z](?:\s*_\s*(?:\{[^{}|]*\}|[A-Za-z0-9]))?(?:\s*\([^()|]*\))?)\s*"
                        r"(?:\\right\s*)?\|")
_EXPR_UNIT = re.compile(r"^\s*(?P<expr>.*?[\d})])\s*(?:\\[,;:! ]|\\quad|\s)*(?P<rm>\\(?:mathrm|text|rm)\s*\{\s*)?"
                        r"(?P<prefix>\\mu|[pnuµmkKM]?)\s*(?P<unit>V\s*A\s*R|V\s*A|var|V|A|W|F|Hz|H|J|\\Omega|Ω|ohm|S|sec|s)"
                        r"\s*\}?\s*$")


def _balanced(t: str) -> bool:
    """braces and parentheses closed in the order opened"""
    stack = []
    for ch in t:
        if ch in "{(":
            stack.append(ch)
        elif ch in "})":
            if not stack or stack.pop() != {"}": "{", ")": "("}[ch]:
                return False
    return not stack


def _bracket_unit(side: str) -> str:
    """a side ending in a unit in square brackets ('3.3 [mA]', '\\frac{E}{R} [A]'): a number or a numeric expression
    keeps its unit after it (read in the base unit: 0.0033); after one in letters the unit is dropped"""
    m = re.search(_BRACKET_UNIT.pattern + r"\s*$", side)
    if not m or not side[:m.start()].strip():
        return side
    body, unit = side[:m.start()].strip(), re.sub(r"\s+", "", m.group("u"))
    plain = re.sub(r"\\(?:d?frac|tfrac|sqrt|cdot|times|left|right|pi|div)|(?<=\d)[eE][-+]?\d", "", body)
    if not re.search(r"[A-Za-z]", plain):
        return _value_side(f"{body} {unit}")               # a number, a numeric expression: in the base unit
    wave = re.sub(r"\\(?:angle|circ|cos|sin|exp|omega)|(?<![A-Za-z])[jte](?![A-Za-z])", "", plain)
    if not re.search(r"[A-Za-z]", wave):
        return f"{body} {unit}"                            # a phasor, a wave in time: its unit read after it
    return body                                            # in the parts' names: the unit says nothing more


_WITH_UNIT = re.compile(r"^\s*(?P<num>-?[\d\s]*\.?[\d\s]+)\s*(?:\\(?:mathrm|text|rm)\s*\{\s*)?"
                        r"(?P<prefix>\\mu|[pnuµmkKM]?)\s*(?P<unit>[VA]\s*/\s*s|d\s*B|V\s*A\s*R|var|V\s*A|V|A|W|F|Hz|H|S|J|C|rad\s*/\s*s|1/s|s|sec|\\Omega|Ω|ohm|℧)"
                        r"\s*\}?\s*$")


_POWER_OF_TEN = re.compile(r"^\s*(?P<num>-?\s*[\d.\s]*(?:\\cdot|\\times|\*)?\s*10\s*\^\s*\{?\s*-?\s*\d+\s*\}?)\s*"
                           r"(?P<prefix>\\mu|[pnuµmkKM]?)\s*(?P<unit>V|A|W|F|Hz|H|J|s|sec|\\Omega|Ω|ohm)\s*$")


def _value_side(text: str) -> str:
    """'1.8 mA' -> '0.0018', '2 k\\Omega' -> '2000.0', '10^{-3} s' -> '0.001': a number with a unit, in the base unit"""
    p10 = _POWER_OF_TEN.match(text)
    if p10:                                                # '2 \cdot 10^{-3} s': the unit after a power of ten
        num = re.sub(r"\\cdot|\\times|\*", "*", p10.group("num"))
        num = re.sub(r"10\s*\^\s*\{?\s*(-?)\s*(\d+)\s*\}?", r"10**(\1\2)", num)
        num = re.sub(r"(?<=\d)\s+(?=[\d.])", "", num).strip()
        try:
            val = float(sp.sympify(num.lstrip("*")))
            return repr(val * _PREFIX[p10.group("prefix")])
        except Exception:                                  # noqa: BLE001
            pass
    m = _WITH_UNIT.match(text)
    if not m:
        e = _EXPR_UNIT.match(text)                         # '\frac{10}{3} V': a numeric expression, then its unit
        if e and re.search(r"\d", e.group("expr")) and _balanced(e.group("expr")) and \
                (e.group("rm") is not None) == text.rstrip().endswith("}") and not re.search(
                r"[A-Za-z]", re.sub(r"\\(?:d?frac|tfrac|sqrt|cdot|times|left|right|pi|div)|(?<=\d)[eE][-+]?\d", "", e.group("expr"))):
            f = _PREFIX[e.group("prefix")]
            return f"({e.group('expr')})" + (f" \\cdot {f!r}" if f != 1.0 else "")
        return text
    if m.group("unit") == "C" and not m.group("prefix"):
        return text                                        # '5 C': five times C, or 5 coulombs - left as written
    return repr(float(re.sub(r"\s+", "", m.group("num"))) * _PREFIX[m.group("prefix")])


_FUNCS = ("cos", "sin", "tan", "exp", "ln", "log", "sqrt", "arctan", "angle")


_AT_ZERO_BAR = re.compile(r"\\(?:[Bb]igg?|right)?\s*\|\s*_\s*\{\s*t\s*=\s*0\s*(?:\^\s*\{?\s*\+\s*\}?|_\s*\{?\s*\+\s*\}?|\+)\s*\}")


def _word(m):
    w = re.sub(r"\s+", "", m.group(1))
    return "\\" + w + " " if w in _FUNCS else w


def _respace(tex: str) -> str:
    """the formula reader's spacing evened out, statements' separators (\\quad, '|', ',') left as they are: its
    spacing commands dropped ('\\!', '\\,', '~'), '\\mid' a bar, the letters it spaces inside \\mathrm / \\text /
    \\operatorname joined ('\\mathrm { c o s }' -> '\\cos', '\\mathrm { m A }' -> 'mA'; '\\mathrm { = }' -> '='), its
    spaced digits joined ('2 . 0 3' -> '2.03'), a subscript's ('Z _ { i n }' -> 'Z_{in}'), a prime ('i _ { L } ^ {
    \\prime }' -> "i_{L}'")"""
    t = re.sub(r"\\[!,;:> ]|~", " ", undecorate(tex))     # (a box, an underline around it: its content)
    t = re.sub(r"\\mid(?![A-Za-z])", "|", t)
    t = re.sub(r"\\s[bp](?![A-Za-z])\s*(?==)", "", t)            # a stray '\sb' (TeX's '_') before '='
    t = re.sub(r"\\(?:mathrm|text|rm|operatorname|mathit)\s*\{([^A-Za-z{}\\]*)\}", r" \1 ", t)
    t = re.sub(r"\\(?:mathrm|text|rm|operatorname|mathit)\s*\{([A-Za-z\s]*)\}", _word, t)
    t = re.sub(r"\\(?:mathrm|text|rm|operatorname|mathit)\s*\{([^{}\\]*)\}",          # '\mathrm { ( l e a d i n g ) }'
               lambda m: " " + re.sub(r"\s+", "", m.group(1)) + " ", t)
    t = re.sub(r"(?<=[\d.])(?<![\^_]\d)\s+(?=[\d.])", "", t)    # (not '\int_0^1 4t^2': a limit, then 4 - it was 14)
    t = re.sub(r"\s*([_^])\s*", r"\1", t)
    t = re.sub(r"_\{\s*([A-Za-z0-9](?:\s*[A-Za-z0-9])*)\s*\}", lambda m: "_{" + re.sub(r"\s+", "", m.group(1)) + "}", t)
    t = re.sub(r"\^\{\s*((?:(?:\\prime|')\s*)+)\}", lambda m: "'" * len(re.findall(r"\\prime|'", m.group(1))), t)
    t = re.sub(r"(?<=[A-Za-z0-9}])\s+(?=')", "", t)                  # "V_{A} '": the prime is the name's
    return re.sub(r"\^\\prime(?![A-Za-z])|\\prime(?![A-Za-z])", "'", t)


# a power factor's load: 'lagging' (inductive, the current behind the voltage) or 'leading' (capacitive)
_CHARACTER = re.compile(r"\s*\(?\s*(?:\\text\s*\{\s*)?(lagging|leading|lag|lead|inductive|capacitive|ind\.?|cap\.?)"
                        r"\s*\}?\s*\)?\s*$", re.IGNORECASE)
_PF_NAMES = {"pf", "fp", "cosphi", "costheta", "cosvarphi"}


# names the reader spaces into letters ('B W'), which would be products: joined when they are a whole side
_WORDS = {"bw", "pf", "gm", "av", "ai", "rin", "rout", "req", "rth", "vth", "zin", "zeq", "zth", "irms", "vrms", "ptotal",
          "ptot", "hfe", "vce", "vbe", "vgs", "vds", "vcb", "vov", "vbb", "rbb"}


def tidy(tex: str) -> str:
    """the formula reader's LaTeX as the checks read it: _respace, and \\quad a space, a spaced unit joined ('H z'
    -> 'Hz')"""
    t = re.sub(r"\\q?quad(?![A-Za-z])", " ", _respace(tex))
    t = re.sub(r"(?<![A-Za-z\\])H\s+z(?![A-Za-z])", "Hz", t)
    t = re.sub(r"(?<![A-Za-z\\])r\s*a\s*d\s*/\s*s(?![A-Za-z])", "rad/s", t)
    t = re.sub(r"\\frac\s*\{\s*r\s*a\s*d\s*\}\s*\{\s*s\s*(?:e\s*c\s*)?\}", "rad/s", t)      # '\\frac{rad}{s}'
    t = re.sub(r"(?<=[\d.}])\s*N\s*p\s*/\s*s(?![A-Za-z])", " 1/s", t)      # nepers a second ('1/s' itself: a step's X(s))
    t = re.sub(r"(?<![A-Za-z\\])V\s+A\s+R(?![A-Za-z])", "VAR", t)
    t = _AT_ZERO_BAR.sub("(0^+)", t)                        # 'dv/dt |_{t=0^+}' - its '=' is no equation's
    t = re.sub(r"\\cos\s*(?:\\left)?\(?\s*\\(var)?(phi|theta)\s*(?:\\right)?\)?", r"\\mathrm{cos\1\2}", t)   # cos(phi): the power factor
    t = re.sub(r"\{\s*=\s*\}", "=", t)                    # 'F { = }'
    t = re.sub(r"\{\s*\}", " ", t)                          # an empty group
    return re.sub(r"\s+", " ", t).strip()


def circuit_statements(tex: str, conf: float = 1.0, line: int = -1, piece: int = -1):
    """a piece of the student's math -> its equations, cut as grader.selfcheck.statements does, sides kept as
    written (the general LaTeX parser takes I for sqrt(-1), E for e and drops subscripts - not here); the formula
    reader's spacing evened out first (tidy)"""
    from .selfcheck import _ITEM, Statement, _top_level_split
    out = []
    tex = _AT_ZERO_BAR.sub("(0^+)", _respace(re.sub(r"\\approx|\\simeq|\\cong|≈|≃|≅", "=", tex)))
    from .selfcheck import _REPEATING, repeating
    tex = _REPEATING.sub(repeating, tex)                   # '3.\overline{3}': 3.333...
    tex = re.sub(r"(\d\.\d+)\s*(?:\.\.\.|…|\\[cl]?dots(?![A-Za-z]))", r"\1", tex)      # '3.333...': the decimals go on
    tex = re.sub(r"\^\s*\{?\s*\(\s*(\d)\s*\)\s*\}?", lambda m: "'" * int(m.group(1)), tex)   # V_A^{(1)}: superposition's
    tex = re.sub(r"(?<![A-Za-z\\])Q\s*(?:_\s*\{?\s*(?:Q|point|op)?\s*\}?)?\s*=?\s*(?:\\left\s*)?\(\s*([^(),=]*?\d[^(),=]*?)\s*,"
                 r"\s*([^(),=]*?\d[^(),=]*?)\s*(?:\\right\s*)?\)", lambda m: f"Q_{{point}} = {m.group(1)}\ue003{m.group(2)}", tex)
    from .selfcheck import _decimal_commas
    tex = _decimal_commas(tex)                             # '7,5': a decimal comma, not two statements; '1 , 000' one
    tex = re.sub(r"(?:\\left)?\|\s*S\s*(?:\\right)?\|", "S", tex)       # |S|: the apparent power - the bars cut nothing
    tex = _spectrum_text(tex)                                    # |H(jw)|, 20 log|H(jw)|, the angle of H(jw)
    tex = _gain_db_text(tex)                                     # 'A_v = 20 dB', '20\log|A_v| = 20': a gain in dB
    tex = re.sub(r"(?<![A-Za-z\\])([Vv])\s*(?:_|\^)\s*(?:\{\s*([+-])\s*\}|([+-])(?=\s*(?:=|,|$|\\quad)))",
                 lambda m: m.group(1) + ("_{p}" if (m.group(2) or m.group(3)) == "+" else "_{n}"), tex)   # an op-amp's V_+, V_-
    tex = _MAGNITUDE.sub(lambda m: "\ue001" + m.group("n") + "\ue002", re.sub(r"\\[lr]?vert(?![A-Za-z])", "|", tex))
    for part in _top_level_split(tex):
        part = tidy(_ITEM.sub("", part.replace("\ue001", "|").replace("\ue002", "|")))
        sides = [_bracket_unit(x.strip()) if k and "\ue003" not in x else x.strip() for k, x in enumerate(part.split("="))]
        m = _CHARACTER.search(sides[-1]) if len(sides) >= 2 else None
        if m and m.start() > 0:                                  # 'pf = 0.8 lagging': the number, its word kept in
            sides[-1] = sides[-1][:m.start()].strip()            # the statement's text
        if re.fullmatch(r"[A-Za-z](?:\s+[A-Za-z])+", sides[0]) and re.sub(r"\s+", "", sides[0]).lower() in _WORDS:
            joined = re.sub(r"\s+", "", sides[0])                # 'B W = 100 rad/s': BW, not B times W
            part, sides[0] = part.replace(sides[0], joined, 1), joined
        if len(sides) >= 2 and all(sides):
            unit = _unit_written(part.split("=")[-1]) if "\ue003" not in part else ""
            sides = sides[:1] + [_divisor_unit(sides[0], x) for x in sides[1:]]
            sides[-1] = _value_side(sides[-1]) if "\ue003" not in sides[-1] else sides[-1]
            out.append(Statement("formula", _as_written(part), [], conf, unit=unit, line=line, piece=piece, side_texts=sides))
    return out


def _divisor_unit(name: str, side: str) -> str:
    """'I = 12 / 6 k\\Omega': the unit after a divisor's number is that number's when the quantity named is of another
    kind (a current is no kΩ) - the side in SI units so ('12 / (6 \\cdot 10^{3})'); it was (12/6) kΩ, a false sure wrong
    against the circuit. 'I = (10 - 0.7) / 2 mA' (a current in mA) stays the whole side's"""
    from .selfcheck import _UNIT, _den_si, _join_digits, _name_unit_differs
    if "\ue003" in side:
        return side
    d = _den_si(side)
    m = _UNIT.match(_join_digits(side)) if d is not None else None
    return d if m and _name_unit_differs(name, m.group("unit")) else side


def _as_written(part: str) -> str:
    """a statement's text for the reader of the verdicts: the markers the parsing put in ('magH(j1000)',
    '\\mathrm{cosphi}') back as an exam writes them"""
    fn = r"([A-Za-z](?:\s*_\s*\{?\s*\w+\s*\}?)?\s*\([^()]*\))"
    t = re.sub(r"(?<![A-Za-z\\])mag" + fn, r"|\1|", part)
    t = re.sub(r"(?<![A-Za-z\\])db" + fn, r"20\\log|\1|", t)
    t = re.sub(r"(?<![A-Za-z\\])arg" + fn, r"\\angle \1", t)
    t = re.sub(r"(?<![A-Za-z\\])dbgain\s*(" + _GAIN + r")\s*=\s*([^=]+)", r"\1 = \2 dB", t)    # 'A_v = 20 dB'
    t = re.sub(r"Q_\{point\}\s*=\s*([^\ue003]*)\ue003(.*)", r"Q = (\1, \2)", t)                 # the operating point
    return re.sub(r"\\mathrm\{cos(var)?(phi|theta)\}|(?<![A-Za-z\\])cos(var)?(phi|theta)(?![A-Za-z])",
                  lambda m: "\\cos\\" + (m.group(1) or m.group(3) or "") + (m.group(2) or m.group(4)), t)


def _states(net):
    """every combination of the switches' states (none: one empty combination); at most 3 switches. A switch
    with two ends is open or closed, a two-way one (three: the common end, then its two throws) at '1' or '2'"""
    import itertools
    sw = [c for c in net.components if c.type == "switch"]
    if len(sw) > 3:
        return []
    choice = [("1", "2") if len(c.nodes) == 3 else ("open", "closed") for c in sw]
    return [dict(zip([c.id for c in sw], combo)) for combo in itertools.product(*choice)]


def _close(x, y, tol: float = REL_TOL) -> bool:
    try:
        d = sp.simplify(x - y)
    except Exception:                                              # noqa: BLE001
        return False
    if d == 0:
        return True
    try:                                                           # numbers, complex ones too (phasors)
        g, a = complex(sp.N(d)), complex(sp.N(y))
    except (TypeError, ValueError):
        return False
    return abs(g) <= tol * max(abs(a), 1e-12)


ROUNDED = 0.05                  # a numeric answer this close may be the circuit's value, rounded on the way


def _digits_slack(s) -> float:
    """half a unit in the last digit of the number a statement ends in, in SI units ('V_A = 3.3 V' -> 0.051, '3.3 mA' ->
    0.000051): a value rounded to the digits written is the circuit's ('3.3' for 3.333 - it was 'rounded?', not sure);
    0: no plain number there"""
    from .selfcheck import _NUM_TEXT, _UNIT, _half_unit, _join_digits, _si
    last = _join_digits(re.sub(r"(?:\\left\s*)?\[\s*([^\[\]]*?)\s*(?:\\right\s*)?\]", r" \1", s.text.split("=")[-1]).strip())
    m = _UNIT.match(last)
    num, unit = (m.group("num").strip(), m.group("unit")) if m else (last, "")
    f = _si(unit) if unit else 1.0
    if f is None or not _NUM_TEXT.fullmatch(re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", num)):
        return 0.0
    return _half_unit(num) * f


def _within_digits(a, b, slack: float) -> bool:
    """the circuit's value a and the student's b apart by no more than the digits written allow (at most ROUNDED)"""
    try:
        x, y = complex(sp.N(a)), complex(sp.N(b))
    except (TypeError, ValueError):
        return False
    return abs(x.imag) < 1e-12 and abs(y.imag) < 1e-12 and abs(x - y) <= min(slack, ROUNDED * abs(x))


def _show(v) -> str:
    """a value as an exam writes it: a number, a phasor '7.07∠-45° (5 - j5)', or the expression"""
    try:
        z = complex(sp.N(v))
    except (TypeError, ValueError):
        return str(sp.simplify(v))
    if abs(z.imag) <= 1e-9 * max(1.0, abs(z)):
        return _fmt(z.real)
    import cmath
    return (f"{_fmt(abs(z))}∠{_fmt(cmath.phase(z) * 180 / cmath.pi)}° "
            f"({_fmt(z.real)} {'+' if z.imag >= 0 else '-'} j{_fmt(abs(z.imag))})")


def _same_wave(f, phasor, w) -> bool:
    """a function of t the student wrote against the circuit's phasor at angular frequency w: Re(P e^{jwt}),
    compared at seven moments of a period"""
    t = sp.Symbol("t")
    try:
        P, W = complex(sp.N(phasor)), float(sp.N(w))
        vals = [(complex(sp.N(f.subs(t, k * 2 * 3.141592653589793 / (7 * W)))).real,
                 (P * complex(__import__("cmath").exp(1j * W * k * 2 * 3.141592653589793 / (7 * W)))).real) for k in range(7)]
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    top = max(abs(b) for _, b in vals) or 1.0
    return all(abs(a - b) <= REL_TOL * top + 1e-12 for a, b in vals)


def _node_v(net, sol, n):
    return 0 if n == net.ground else sol.get(_canon(f"V_{net.node_labels.get(n, n)}"))


def _port_pair(net, ports) -> tuple:
    """a port's two terminals -> ('+' one, '-' one, whether the drawing says which: its '+' / '-' marks, or names
    'a' / 'b' - else the upper one is taken as '+', or the left one)"""
    a, b = ports
    if net.ports_marked and list(ports) == list(net.ports[:2]):
        return a, b, True
    if a in net.node_labels and b in net.node_labels:
        a, b = sorted(ports, key=lambda n: net.node_labels[n].lower())
        return a, b, True
    where = {n: p for n, p in net.terminals}
    if a in where and b in where:
        (xa, ya), (xb, yb) = where[a], where[b]
        if (abs(yb - ya) >= abs(xb - xa) and yb < ya) or (abs(yb - ya) < abs(xb - xa) and xb < xa):
            a, b = b, a
    else:
        a, b = sorted(ports, key=lambda n: net.node_labels.get(n, "~" + str(n)).lower())
    return a, b, False


def _small_signal_names(net, ss: dict) -> dict:
    """Netlist.small_signal's results by the names exam answers use: g_m / r_pi (of one transistor, or g_{mQ1}),
    A_v, A_vs, R_in / R_i / Z_in, R_out / Z_out"""
    out = {}
    qs = [c for c in net.components if c.type in ("npn", "pnp", "nmos", "pmos")]
    for c in qs:
        nm = _canon(c.label or c.id)
        if c.id in ss.get("gm", {}):
            out["gm" + nm] = ss["gm"][c.id]
            if c.id in ss.get("rpi", {}):
                out["rpi" + nm] = ss["rpi"][c.id]
            if len(qs) == 1:
                out["gm"] = ss["gm"][c.id]
                if c.id in ss.get("rpi", {}):          # (r_pi, r_e by their own keys - 're' was the resistor R_E's too)
                    out[_canon("r_pi")], out[_canon("r_e")] = ss["rpi"][c.id], 1 / ss["gm"][c.id]
    for k, names in (("av", ("av", "a", "gain")), ("avs", ("avs",)), ("rin", ("rin", "ri", "zin")),
                     ("rout", ("rout", "zout"))):
        if k in ss:
            for n in names:
                out[n] = ss[k]
    return out


def _base_thevenin(net, st) -> dict:
    """one transistor: what its base sees with the transistor taken out - V_TH / V_BB (the open-circuit voltage
    there) and R_TH / R_BB / R_B (the resistance, the sources set to zero) - the usual first step of a bias problem"""
    qs = [c for c in net.components if c.type in ("npn", "pnp") and "Base" in c.roles]
    if len(qs) != 1:
        return {}
    b = qs[0].roles["Base"]
    rest = Netlist([c for c in net.components if c is not qs[0]], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
    r = rest.solve("dc", st)
    out = {}
    vb = 0 if b == net.ground else (r or {}).get(f"V_{net.node_labels.get(b, b)}")
    if vb is not None:
        out.update({"vth": vb, "vbb": vb})
    rb = rest.resistance_seen(b, net.ground, st) if net.ground is not None else None
    if rb is not None:
        out.update({"rth": rb, "rbb": rb, "rb": rb})
    return out


def _two_port(net, st, mode: str) -> dict:
    """four open terminals (a two-port: port 1 the left pair, port 2 the right one, each's upper terminal its '+')
    -> z11 ... z22 (1 A pushed into one port, the other open, the independent sources set to zero), and from them
    the y, h, g and transmission (t11 = A, t12 = B, t21 = C, t22 = D) parameters"""
    if len(net.terminals) != 4:
        return {}
    by_x = sorted(net.terminals, key=lambda t: t[1][0])                # (node, (x, y)); two may share a node
    (p1, m1), (p2, m2) = [[t[0] for t in sorted(pair, key=lambda t: t[1][1])] for pair in (by_x[:2], by_x[2:])]

    def across(r, a, b):
        va = 0 if a == net.ground else r.get(f"V_{net.node_labels.get(a, a)}")
        vb = 0 if b == net.ground else r.get(f"V_{net.node_labels.get(b, b)}")
        return None if va is None or vb is None else sp.simplify(va - vb)
    z = {}
    for j, (a, b) in ((1, (p1, m1)), (2, (p2, m2))):
        r = net.solve(mode, st, zero_sources=True, test=(a, b))
        if not r:
            return {}
        z[(1, j)], z[(2, j)] = across(r, p1, m1), across(r, p2, m2)
    if any(v is None for v in z.values()):
        return {}
    z11, z12, z21, z22 = z[(1, 1)], z[(1, 2)], z[(2, 1)], z[(2, 2)]
    det = sp.simplify(z11 * z22 - z12 * z21)
    out = {"z11": z11, "z12": z12, "z21": z21, "z22": z22}
    if det != 0:
        out.update({"y11": z22 / det, "y12": -z12 / det, "y21": -z21 / det, "y22": z11 / det})
    if z22 != 0:
        out.update({"h11": det / z22, "h12": z12 / z22, "h21": -z21 / z22, "h22": 1 / z22})
    if z11 != 0:
        out.update({"g11": 1 / z11, "g12": -z12 / z11, "g21": z21 / z11, "g22": det / z11})
    if z21 != 0:
        out.update({"t11": z11 / z21, "t12": det / z21, "t21": 1 / z21, "t22": z22 / z21})
    return {k: sp.simplify(v) for k, v in out.items()}


def _phasor_z(net, st) -> dict:
    """a sinusoidal circuit: Z_in / Z_eq / Z - what the one source sees (the circuit without it) - and Z_th between a
    port's two terminals, the sources set to zero; at the sources' angular frequency (or none needed: parts labelled
    with their impedances)"""
    out = {}
    srcs = [c for c in net.components if c.type in ("voltage_source", "current_source")]
    if len(srcs) == 1 and len(srcs[0].nodes) == 2:
        rest = Netlist([c for c in net.components if c is not srcs[0]], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
        z = _impedance_at(rest, srcs[0].nodes[0], srcs[0].nodes[1], st, net)
        if z is not None:
            for k in ("z", "zin", "zeq", "zi", "zt", "ztotal"):
                out[k] = z
    ports = list(net.ports)
    if len(ports) == 2:
        z = _impedance_at(net, ports[0], ports[1], st, net)
        if z is not None:
            for k in ("zth", "zab", "zn"):
                out[k] = z
    return out


def _impedance_at(n, a, b, st, net):
    from .circuit import ac_phasor
    ws = {ac_phasor(c.value)[1] for c in net.components if c.ac and ac_phasor(c.value)}
    r = n.solve("phasor", st, zero_sources=True, test=(a, b), w=ws.pop() if len(ws) == 1 else None)
    if not r:
        return None
    va = 0 if a == n.ground else r.get(f"V_{n.node_labels.get(a, a)}")
    vb = 0 if b == n.ground else r.get(f"V_{n.node_labels.get(b, b)}")
    return None if va is None or vb is None else sp.simplify(va - vb)


def _aliases(net, st, sol) -> dict:
    """what exam answers call by name though nothing on the drawing says it: between a port's two terminals (the
    open circles a, b) Thevenin's V_th / V_oc, R_th / R_eq / R_ab, Norton's I_N / I_sc; with one op-amp, V_o / V_out
    its output and V_i / V_in / V_s the one independent source, A_v / A / G their ratio"""
    out = {}
    ports = list(net.ports)
    by_label = {lab.lower(): n for n, lab in net.node_labels.items()}
    if len(ports) != 2 and "a" in by_label and "b" in by_label:
        ports = [by_label["a"], by_label["b"]]
    if len(ports) == 2:
        a, b, _ = _port_pair(net, ports)
        va, vb = _node_v(net, sol, a), _node_v(net, sol, b)
        r = net.resistance_seen(a, b, st)
        if va is not None and vb is not None:
            for k in ("vth", "voc", "vab", "vt"):
                out[k] = sp.simplify(va - vb)
        if r is not None:
            for k in ("rth", "req", "rab", "rn", "rt", "rin"):
                out[k] = r
            if va is not None and vb is not None and r != 0:
                for k in ("in", "isc", "inorton"):
                    out[k] = sp.simplify((va - vb) / r)
                if not any(c.ac for c in net.components):   # the most power a load across the port can take
                    for k in ("pmax", "plmax", "pmaxl", "plmaximum"):
                        out[k] = sp.simplify((va - vb) ** 2 / (4 * r))
    amps = [c for c in net.components if c.type == "opamp" and "Out" in c.roles]
    srcs = [c for c in net.components if c.type == "voltage_source"]
    if len(amps) == 1:
        for k, role in (("vp", "In+"), ("vplus", "In+"), ("vn", "In-"), ("vminus", "In-")):
            if role in amps[0].roles and _node_v(net, sol, amps[0].roles[role]) is not None:
                out[k] = _node_v(net, sol, amps[0].roles[role])      # the inputs' voltages, V_+ and V_-
        vo = _node_v(net, sol, amps[0].roles["Out"])
        if vo is not None:
            out["vo"] = out["vout"] = vo
            if len(srcs) == 1:
                vi = net.symbol(srcs[0])
                out["vi"] = out["vin"] = out["vs"] = vi
                if vi != 0:
                    out["av"] = out["a"] = out["g"] = out["gain"] = sp.simplify(vo / vi)
    labelled = [(lab, n) for n, lab in net.node_labels.items() if re.fullmatch(r"[A-Za-z]\d?", lab)][:12]
    for la, na in labelled:                        # V_AB: node A's voltage over node B's (a line voltage, a part's)
        for lb, nb in labelled:
            if na != nb:
                va, vb = _node_v(net, sol, na), _node_v(net, sol, nb)
                if va is not None and vb is not None:
                    out.setdefault(_canon(f"v{la}{lb}"), sp.simplify(va - vb))
    for vol in net.voltages:                       # a voltage marked on the drawing: its '+' node's minus its '-' node's
        vp, vm = _node_v(net, sol, vol.plus), _node_v(net, sol, vol.minus)
        if vp is not None and vm is not None and vol.plus != vol.minus:
            out[_canon(vol.label)] = sp.simplify(vp - vm)
    return out


def _switchings(net, states):
    """the switchings at t = 0 the drawing allows: (before, after) - at most two switches flip together; when some
    switch is labelled 't = 0', exactly the switches so labelled flip"""
    moving = {c.id for c in net.components if c.type == "switch" and (c.value or "").replace(" ", "") == "t=0"}
    out = []
    for p in states:
        for q in states:
            flipped = {k for k in p if p[k] != q[k]}
            if flipped and (flipped == moving if moving else len(flipped) <= 2):
                out.append((p, q))
    return out


def _transients(net):
    """first-order responses to every switching the drawing allows: [(before, after, response, drawn)] - a
    response as Netlist.first_order_response gives it; at most two switches flip at t = 0; drawn: `before` is
    the state the switches are drawn in (when that was read)"""
    states = _states(net)
    if len(states) < 2 or not any(c.type in ("capacitor", "inductor") for c in net.components):
        return []
    shown = {c.id: c.state for c in net.components if c.type == "switch" and c.state}
    memo, out = {}, []
    for p, q in _switchings(net, states):
        r = net.first_order_response(p, q, memo)
        if r:
            out.append((p, q, r, bool(shown) and all(p.get(k) == v for k, v in shown.items())))
    return out


_TYPE_LETTER = {"c": "capacitor", "l": "inductor", "r": "resistor"}


def _pick(k: str, r: dict, net, currents: dict) -> list:
    """the student's name for a quantity ('vc', 'va', 'il', 'i1': canonical) -> what it may be in the response r:
    [(x(t), whether either sign is right: a part's voltage / current, its direction not drawn)] - a node voltage
    (only nodes the drawing names), a current drawn, a part's voltage or current (V_C: the capacitor's, or the
    node's when a node is named C too)"""
    vals = r["values"]
    for vol in net.voltages:                                 # a voltage marked '+' / '-' on the drawing: as marked
        if _canon(vol.label) == k and vol.plus != vol.minus:
            vp, vm = (0 if n == net.ground else vals.get(f"V_{net.node_labels.get(n, n)}") for n in (vol.plus, vol.minus))
            if vp is not None and vm is not None:
                return [(sp.simplify(vp - vm), False)]
    named = {f"V_{lab}" for lab in net.node_labels.values()}
    out = []
    for name, e in vals.items():                             # node voltages and the currents drawn first
        if (name in named or name in currents.values()) and _canon(name) == k:
            out.append((e, False))
    keys = [k] + (["v" + k[1:]] if k[:1] == "u" else [])     # u_C: a voltage, as some books write it
    for key in keys:
        for name, e in vals.items():
            if name[:2] in ("v_", "i_") and _canon(name) == key:
                out.append((e, True))
        if len(key) == 2 and key[0] in "vi" and key[1] in _TYPE_LETTER:      # v_C with one capacitor, i_L ...
            of = [c for c in net.components if c.type == _TYPE_LETTER[key[1]]]
            if len(of) == 1:
                e = vals.get(f"{key[0]}_{re.sub(chr(92) + 'W', '', of[0].label or of[0].id) or of[0].id}")
                if e is not None:
                    out.append((e, True))
    if k == "i":                                             # i(t) in a single loop: the current all its parts carry
        flows = [e for name, e in vals.items() if name.startswith("i_") and sp.simplify(e) != 0]
        if flows and all(sp.simplify(e - flows[0]) == 0 or sp.simplify(e + flows[0]) == 0 for e in flows):
            out.append((flows[0], True))
    return out


def _show_t(e) -> str:
    """x(t) as an exam writes it: '5 - 5 e^(-1000 t)'"""
    try:
        e = sp.N(sp.expand(e), 4)
    except Exception:                                              # noqa: BLE001
        return str(e)
    out = str(e).replace("**", "^").replace("exp(", "e^(").replace("*", " ")
    return re.sub(r"(?<![\d.])(\d+)\.0+(?!\d)", r"\1", out)


def bare_st(st: dict) -> tuple:
    """a state of the switches, without the variants tried beside it (diode models, AC power conventions, which
    resistance R_eq means)"""
    return tuple(sorted((k, v) for k, v in st.items() if k not in ("diodes", "power", "VT", "R_eq", "Q")))


_REQ = ("req", "rin", "rt", "rtotal", "rtot", "rs")


def _what_the_source_sees(net, sols):
    """R_eq / R_in / R_total - the resistance the one independent source sees (the rest of the circuit, its other
    sources set to zero). With terminals drawn as well, R_eq / R_in may mean theirs: both tried then, as variants"""
    srcs = [c for c in net.components if c.type in ("voltage_source", "current_source") and len(set(c.nodes)) == 2]
    if len(srcs) != 1:
        return sols
    rest = Netlist([c for c in net.components if c is not srcs[0]], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
    out = []
    for st, sol, tau in sols:
        r = rest.resistance_seen(srcs[0].nodes[0], srcs[0].nodes[1],
                                 {k: v for k, v in st.items() if k not in ("diodes", "power", "VT")})
        if r is None or r == 0:
            out.append((st, sol, tau))
            continue
        port = sol.get("req")
        mine = dict(sol)
        for k in _REQ:
            mine[k] = r
        if port is None or (_num(port) is not None and _num(r) is not None and _close(port, r)):
            out.append((st, {**sol, **{k: r for k in _REQ if k not in sol or port is not None}}, tau))
        else:
            out.append((dict(st, R_eq="at the terminals"), sol, tau))
            out.append((dict(st, R_eq="what the source sees"), mine, tau))
    return out


def _what_the_storage_sees(net, sols):
    """one capacitor / inductor: R_th - the resistance it sees, the rest of the circuit with its sources set to zero
    (the R of its time constant); R_eq / R_T may mean that too: tried beside what the source sees, as variants"""
    stores = [c for c in net.components if c.type in ("capacitor", "inductor") and len(set(c.nodes)) == 2]
    if len(stores) != 1:
        return sols
    x = stores[0]
    rest = Netlist([c for c in net.components if c is not x], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
    out = []
    for st, sol, tau in sols:
        r = rest.resistance_seen(x.nodes[0], x.nodes[1],
                                 {k: v for k, v in st.items() if k not in ("diodes", "power", "VT", "R_eq")})
        if r is None or r == 0:
            out.append((st, sol, tau))
            continue
        base = dict(sol)
        if not net.ports:
            base.setdefault("rth", r)                      # no terminals drawn: Thevenin's R is what it sees
        src = sol.get("req")
        if src is None or (_num(src) is not None and _num(r) is not None and _close(src, r)):
            out.append((st, {**base, **{k: r for k in ("req", "rt") if k not in base}}, tau))
        else:
            out.append((st, base, tau))
            out.append((dict(st, R_eq=f"what {x.label or x.id} sees"), {**base, "req": r, "rt": r}, tau))
    return out


def _show_wave(e) -> str:
    """a sum of a constant and sinusoids of t as an exam writes it: '5 + 7.07 cos(1000t - 45°)'"""
    import cmath
    t = sp.Symbol("t")
    e = sp.expand(sp.sympify(e))
    const, waves = sp.Integer(0), {}
    for term in sp.Add.make_args(e):
        f = [a for a in term.atoms(sp.cos, sp.sin)]
        if not f:
            const += term
            continue
        fn = f[0]
        w = sp.simplify(fn.args[0] / t)
        coef = sp.simplify(term / fn)
        z = waves.get(w, 0)
        waves[w] = z + (complex(sp.N(coef)) if fn.func == sp.cos else complex(0, 1) * complex(sp.N(coef)))
    out = [_fmt(float(sp.N(const)))] if const != 0 else []
    for w, z in waves.items():                                     # a cos + b sin = |X| cos(wt + arg X), X = a - j b
        X = complex(z.real, -z.imag)
        ang = cmath.phase(X) * 180 / cmath.pi
        out.append(f"{_fmt(abs(X))} cos({_fmt(float(w))}t {'+' if ang >= 0 else '-'} {_fmt(abs(ang))}°)")
    return " + ".join(out) or "0"


def switching_text(p: dict, q: dict) -> str:
    """'S1 open -> closed'"""
    return ", ".join(f"{k} {p[k]} → {q[k]}" for k in p if p[k] != q[k])


def _check_transient(k: str, rhs, trans, net, parts: dict, currents: dict):
    """a function of time the student wrote for the quantity k against the circuit's first-order responses ->
    (right?, the verdict's text, sure?) or None when it cannot be judged. Right: it is x(t) for some switching;
    when only its final value and decay fit, the start may come from an earlier switching or the problem's text
    - not judged sure"""
    t, tau_s = sp.Symbol("t"), sp.Symbol("tau")
    if not all(str(x) in ("t", "tau") or _canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) not in ("t", "tau")})
    cands = []
    for p, q, r, drawn in trans:
        for e, either in _pick(k, r, net, currents):
            if _decidable(mine.subs(tau_s, r["tau"]), [e, r["tau"]]):
                cands.append((p, q, r, drawn, e, either))
    if not cands:
        return None
    syms = set(mine.free_symbols)
    for c in cands:
        syms |= c[4].free_symbols | sp.sympify(c[2]["tau"]).free_symbols
    syms -= {t, tau_s}
    num = {x: 1.0 + 0.37 * i for i, x in enumerate(sorted(syms, key=str))}   # parts named, not valued: any numbers

    def values(expr, tau, ts):
        f = sp.lambdify(t, sp.sympify(expr).subs(tau_s, tau).subs(num), "math")
        return [complex(f(x)).real for x in ts]
    full, shape, steady = [], [], []
    for p, q, r, drawn, e, either in cands:
        try:
            tau = float(sp.N(sp.sympify(r["tau"]).subs(num)))
            if not tau > 0:
                continue
            ts = [tau * m for m in (0, 0.25, 0.5, 1, 2, 3)]
            theirs, me = values(e, tau, ts), values(mine, tau, ts)
            inf = values(e, tau, [60 * tau])[0]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        tol = REL_TOL * max([abs(v) for v in theirs + me] + [1e-12])
        signs = (1, -1) if either else (1,)
        if any(all(abs(a - sign * b) <= tol for a, b in zip(me, theirs)) for sign in signs):
            if not any(f[0] == p and f[1] == q for f in full):
                full.append((p, q, r, drawn))
            continue
        for sign in signs:                                   # its final value and decay only
            model = [sign * inf + (me[0] - sign * inf) * __import__("math").exp(-x / tau) for x in ts]
            if all(abs(a - b) <= tol for a, b in zip(me, model)):
                if max(me) - min(me) <= tol:
                    steady.append(q)                         # a constant: the steady state after it
                else:
                    shape.append((p, q, r, e))
                break
    if full:
        drawn_ok = [h for h in full if h[3]]
        best = drawn_ok or full
        when = "; ".join(dict.fromkeys(switching_text(p, q) for p, q, _, _ in best))
        assumed = any(r["assumed"] for _, _, r, _ in best)
        sure = len(best) == 1 and not assumed and (bool(drawn_ok) or not any(c.state for c in net.components if c.type == "switch"))
        return True, f": as the circuit gives when {when}" + (" (the capacitor taken uncharged before)" if assumed else ""), sure
    if not mine.has(t):
        if steady:
            return True, f": the steady state ({st_text(steady[0])})", False
        return None
    if shape:
        p, q, r, e = shape[0]
        return False, (f": its final value and time constant fit ({st_text(q)}), but not its start - the circuit "
                       f"gives {_show_t(e)} when {switching_text(p, q)}"), False
    shown = "; ".join(dict.fromkeys(f"{_show_t(e)} when {switching_text(p, q)}" for p, q, _, _, e, _ in cands[:4]))
    return False, f": the circuit gives {shown}", True                 # its end / decay: whatever the part held


def _ltransients(net):
    """the Laplace-domain responses (any order) to every switching the drawing allows: [(before, after, response,
    drawn)] as _transients gives them, from Netlist.laplace_response"""
    states = _states(net)
    if len(states) < 2 or not any(c.type in ("capacitor", "inductor") for c in net.components):
        return []
    shown = {c.id: c.state for c in net.components if c.type == "switch" and c.state}
    memo, out = {}, []
    for p, q in _switchings(net, states):
        r = net.laplace_response(p, q, memo)
        if r:
            out.append((p, q, r, bool(shown) and all(p.get(k) == v for k, v in shown.items())))
    return out


def _terms(X, num):
    """X(s) (numbers once `num` is put in) -> its inverse Laplace transform for t > 0 as terms [(c, p, k)]:
    c t^k / k! e^(p t) (poles and residues; impulses at t = 0 left out); None when it cannot be done"""
    from scipy import signal
    s = sp.Symbol("s")
    try:
        e = sp.together(sp.sympify(X).subs(num))
        if e.free_symbols - {s}:
            return None
        n, d = sp.fraction(e)
        b = [float(x) for x in sp.Poly(sp.expand(n), s).all_coeffs()]
        a = [float(x) for x in sp.Poly(sp.expand(d), s).all_coeffs()]
    except Exception:                                              # noqa: BLE001
        return None
    if not any(b):
        return []
    try:
        r, p, _ = signal.residue(b, a)
    except Exception:                                              # noqa: BLE001
        return None
    out, j = [], 0
    while j < len(p):                                              # a pole met m times: t^0 ... t^(m-1) terms
        m = 1
        while j + m < len(p) and abs(p[j + m] - p[j]) <= 1e-3 * max(1.0, abs(p[j])):
            m += 1
        out += [(complex(r[j + k]), complex(p[j]), k) for k in range(m)]
        j += m
    return out


def _at(terms, ts):
    import cmath
    import math
    return [sum((c * t ** k / math.factorial(k) * cmath.exp(p * t)).real for c, p, k in terms) for t in ts]


def _scale(terms) -> float:
    """the time over which the response settles: 1 / the slowest decay (or the slowest oscillation's period)"""
    import math
    poles = [p for _, p, _ in terms if abs(p) > 1e-9]
    if not poles:
        return 1.0
    decay = [abs(p.real) for p in poles if abs(p.real) > 1e-9 * abs(p)]
    return 1.0 / min(decay) if decay else 2 * math.pi / min(abs(p.imag) for p in poles)


def _show_terms(terms) -> str:
    """'10 - 11.55 e^(-50t) sin(86.6t) - 10 e^(-50t) cos(86.6t)'"""
    out = []
    for c, p, k in terms:
        if p.imag < -1e-9 * max(1.0, abs(p)):
            continue                                               # its conjugate says it
        tk = "" if k == 0 else (" t" if k == 1 else f" t^{k}/{__import__('math').factorial(k)}")
        ex = "" if abs(p) <= 1e-9 else f" e^({_fmt(p.real)}t)"
        if abs(p.imag) <= 1e-9 * max(1.0, abs(p)):
            if abs(c.real) > 1e-12:
                out.append(f"{_fmt(c.real)}{tk}{ex}")
        else:                                                      # c e^(pt) + its conjugate: 2 e^(st)(Re c cos wt - Im c sin wt)
            w = _fmt(p.imag)
            if abs(c.real) > 1e-12:
                out.append(f"{_fmt(2 * c.real)}{tk}{ex} cos({w}t)")
            if abs(c.imag) > 1e-12:
                out.append(f"{_fmt(-2 * c.imag)}{tk}{ex} sin({w}t)")
    return " + ".join(out).replace("+ -", "- ") or "0"


def _numbers(exprs, extra=()):
    syms = set(extra)
    for e in exprs:
        syms |= sp.sympify(e).free_symbols
    syms -= {sp.Symbol("t"), sp.Symbol("s"), sp.Symbol("tau")}
    return {x: 1.0 + 0.37 * i for i, x in enumerate(sorted(syms, key=str))}


def _check_laplace_time(k: str, rhs, ltrans, net, parts: dict, currents: dict):
    """a function of time against the Laplace-domain responses (circuits of second order and up): right when it is
    the response to some switching; wrong when its final value is no switching's; when only that fits - not sure"""
    t = sp.Symbol("t")
    if not all(str(x) == "t" or _canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) != "t"})
    cands = [(p, q, r, drawn, e, either) for p, q, r, drawn in ltrans for e, either in _pick(k, r, net, currents)
             if _decidable(mine, [e])]
    if not cands:
        return None
    num = _numbers([c[4] for c in cands] + [mine])
    try:
        f = sp.lambdify(t, mine.subs(num), "math")
    except Exception:                                              # noqa: BLE001
        return None
    full, ends, shown = [], [], []
    for p, q, r, drawn, e, either in cands:
        terms = _terms(e, num)
        if terms is None:
            continue
        T = _scale(terms)
        ts = [T * m for m in (0, 0.02, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)]
        try:
            me = [complex(f(x)).real for x in ts]
            late = complex(f(60 * T)).real
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        theirs, inf = _at(terms, ts), _at(terms, [60 * T])[0]
        tol = REL_TOL * max([abs(v) for v in theirs + me] + [1e-12])
        signs = (1, -1) if either else (1,)
        if any(all(abs(a - sg * b) <= tol for a, b in zip(me, theirs)) for sg in signs):
            if not any(x[0] == p and x[1] == q for x in full):
                full.append((p, q, r, drawn))
            continue
        if any(abs(late - sg * inf) <= tol for sg in signs):
            ends.append((p, q))
        shown.append(f"{_show_terms(terms)} when {switching_text(p, q)}")
    if full:
        drawn_ok = [h for h in full if h[3]]
        best = drawn_ok or full
        when = "; ".join(dict.fromkeys(switching_text(p, q) for p, q, _, _ in best))
        assumed = any(r["assumed"] for _, _, r, _ in best)
        sure = len(best) == 1 and not assumed and (bool(drawn_ok) or not any(c.state for c in net.components if c.type == "switch"))
        return True, f": as the circuit gives when {when}" + (" (what the parts held before taken as nothing)" if assumed else ""), sure
    if not shown:
        return None
    text = "; ".join(dict.fromkeys(shown[:4]))
    if ends:
        return False, f": its final value fits, not the rest - the circuit gives {text}", False
    return False, f": the circuit gives {text}", True


_MOMENT = re.compile(r"\s*([A-Za-z](?:\s*_\s*\{?\s*\w+(?:\s+\w+)*\s*\}?)?)\s*\(\s*(?:0\s*(?:\^\s*\{?\s*([+-])\s*\}?"
                     r"|_\s*\{?\s*([+-])\s*\}?|([+-]))?|(\\infty|∞|t\s*\\(?:to|rightarrow)\s*\\infty))\s*\)\s*")


_N = r"([A-Za-z](?:_\{?\w+\}?)?)"
_AT0 = r"(?:0\^\{?\+\}?|0_\{?\+\}?|0\+)"
_DERIV = [re.compile(p) for p in (                       # dx/dt at 0+, as exams write it (spaces taken out first)
    rf"\\frac\{{d{_N}\({_AT0}\)\}}\{{dt\}}",
    rf"\\frac\{{d{_N}\}}\{{dt\}}(?:\({_AT0}\)|\\(?:[Bb]igg?|left\.)?\|_\{{?(?:t=)?{_AT0}\}}?|\|_\{{?(?:t=)?{_AT0}\}}?)",
    rf"d{_N}/dt\({_AT0}\)",
    rf"{_N}'\({_AT0}\)",
    rf"\\dot\{{?([A-Za-z])\}}?((?:_\{{?\w+\}}?)?)\({_AT0}\)")]


def _derivative_at_0(text: str):
    """'\\frac{dv_C}{dt}(0^+)', 'v_C\'(0^+)', '\\dot{v}_C(0^+)' -> the name ('v_C'), else None"""
    t = re.sub(r"\s+|\\left\.", "", text)
    for k, p in enumerate(_DERIV):
        m = p.fullmatch(t)
        if m:
            return m.group(1) + (m.group(2) if k == 4 else "")
    return None


def _moment_of(X, when: str):
    """x(0+) of X(s) - the initial value theorem, lim s X(s) as s grows - or x(inf), the final one (s X(s) at s = 0),
    or dx/dt at 0+ ('d0+': lim s (s X(s) - x(0+))); None when it has none (an impulse, a pole at 0 left)"""
    s = sp.Symbol("s")
    if when == "d0+":
        x0 = _moment_of(X, "0+")
        return None if x0 is None else _moment_of(sp.together(s * X - x0), "0+")
    num, den = sp.fraction(sp.cancel(sp.together(s * X)))
    try:
        pn, pd = sp.Poly(num, s), sp.Poly(den, s)
    except sp.PolynomialError:
        return None
    if when == "0+":
        if pn.degree() > pd.degree():
            return None
        return sp.Integer(0) if pn.degree() < pd.degree() else sp.simplify(pn.LC() / pd.LC())
    d0 = pd.eval(0)
    return None if d0 == 0 else sp.simplify(pn.eval(0) / d0)


def _check_moment(k: str, when: str, rhs, ltrans, net, parts: dict, currents: dict):
    """x(0-) / x(0+) / x(0) / x(inf) = value against the switchings: just before (the steady state before), just
    after (what C and L held carried over: the initial value theorem on X(s)), long after (the final value), its slope
    just after ('d0+') -> (right?, text, sure?) or None. x(0) without a sign: either moment - sure only when both
    agree"""
    if not all(_canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols})
    got, before = [], {}
    for p, q, r, drawn in ltrans:
        rs = []
        if when in ("0-", "0"):
            key = tuple(sorted(p.items()))
            if key not in before:
                before[key] = net.laplace_response(p, p)          # no switching: its final value is the state before
            if before[key]:
                rs.append((before[key], "inf"))
        if when in ("0+", "0", "inf", "d0+"):
            rs.append((r, {"inf": "inf", "d0+": "d0+"}.get(when, "0+")))
        for resp, at in rs:
            for e, either in _pick(k, resp, net, currents):
                v = _moment_of(e, at)
                if v is not None and _decidable(mine, [v]):
                    got.append((p, q, drawn, v, either))
    if not got:
        return None
    drawn_got = [g for g in got if g[2]] or got
    hit = [g for g in drawn_got if any(_close(mine, sg * g[3]) for sg in ((1, -1) if g[4] else (1,)))]
    sw = {(tuple(sorted(g[0].items())), tuple(sorted(g[1].items()))) for g in drawn_got}
    vals = {_show(g[3]) for g in drawn_got}
    sure = len(sw) == 1 and (when != "0" or len(vals) == 1)
    if hit:
        return True, f": as the circuit gives when {switching_text(hit[0][0], hit[0][1])}", sure
    shown = "; ".join(dict.fromkeys(f"{_show(g[3])} when {switching_text(g[0], g[1])}" for g in drawn_got[:4]))
    return False, f": the circuit gives {shown}", sure


_S_NAMES = {"h", "t", "g", "hs", "a", "av", "z", "zin", "zeq", "zi", "y", "yin", "zth", "zab", "zt", "zout", "zo"}
_FREQ = re.compile(r"\s*(?:\\mathrm\s*\{\s*)?(mag|db|arg)?\s*\}?\s*([A-Za-z](?:\s*_\s*\{?\s*\w+\s*\}?)?)\s*\(\s*"
                   r"(j\s*\\omega|j\s*w|j\s*\\infty|\\infty|∞|0|j\s*[\d.]+(?:\s*(?:\\cdot|\\times)\s*10\s*\^\s*\{?\s*-?\d+\s*\}?"
                   r"|\s*\^\s*\{?\s*\d+\s*\}?)?)\s*\)\s*")
_SPECTRUM = re.compile(r"(20\s*\\log(?:_\s*\{?\s*10\s*\}?)?\s*)?(?:\\left)?\|\s*([A-Za-z](?:_\{?\w+\}?)?\s*\([^()|]*\))\s*"
                       r"(?:\\right)?\|(\s*_\s*\{?\s*(?:\\mathrm\s*\{\s*)?d\s*B\s*\}?\s*\}?)?")


def _spectrum_text(tex: str) -> str:
    """'|H(j\\omega)|' -> '\\mathrm{mag}H(j\\omega)', '20\\log|H(j1000)|' or '|H(j1000)|_{dB}' -> '\\mathrm{db}...',
    '\\angle H(j1000)' -> '\\mathrm{arg}...': the bars no statement's separators"""
    t = _SPECTRUM.sub(lambda m: ("\\mathrm{db}" if m.group(1) or m.group(3) else "\\mathrm{mag}") + m.group(2), tex)
    return re.sub(r"\\(?:angle|arg)\s*(?=[A-Za-z](?:_\{?\w+\}?)?\s*\()", r"\\mathrm{arg}", t)


_GAIN = r"(?:A|G|H|T)(?:\s*_\s*(?:\{\s*[A-Za-z]{1,3}\s*\}|[A-Za-z]))?"     # A, A_v, A_{vs}, G_p, H, T: a gain


def _q_point(text: str):
    """'4.98 V\ue0031.23 mA' (a Q-point's pair, either order) -> (volts, amperes), by the units; None without them"""
    out = {}
    for part in text.split("\ue003"):
        m = re.fullmatch(r"\s*(-?[\d.\s]+(?:\\cdot\s*10\s*\^\s*\{?\s*-?\d+\s*\}?)?)\s*(?:\[\s*)?(?:\\mathrm\s*\{\s*)?"
                         r"(\\mu|[pnuµmk]?)\s*([VA])\s*\}?\s*\]?\s*", part)
        if not m:
            return None
        num = _value_side(f"{m.group(1)} {m.group(2)}{m.group(3)}")
        try:
            out[m.group(3)] = float(sp.sympify(num))
        except (sp.SympifyError, TypeError, ValueError):
            return None
    return (out["V"], out["A"]) if set(out) == {"V", "A"} else None


def _gain_db_text(tex: str) -> str:
    """a gain given in decibels -> '\\mathrm{dbgain}A_v = 20': 'A_v = 20 dB' (or '[dB]'), '20\\log|A_v|', '|A_v|_{dB}'"""
    db = r"\[?\s*(?:\\mathrm\s*\{\s*)?d\s*B\s*\}?\s*\]?"
    t = re.sub(rf"(?<![A-Za-z\\])(?:\\left\s*)?\|?\s*({_GAIN})\s*(?:\\right\s*)?\|?\s*=\s*(-?\s*\d[\d.\s]*?)\s*{db}(?![A-Za-z])",
               r"\\mathrm{dbgain}\1 = \2", tex)
    t = re.sub(rf"20\s*\\log(?:_\s*\{{?\s*10\s*\}}?)?\s*(?:\\left\s*)?(?:\||\()\s*({_GAIN})\s*(?:\\right\s*)?(?:\||\))",
               r"\\mathrm{dbgain}\1", t)
    return re.sub(rf"(?:\\left\s*)?\|\s*({_GAIN})\s*(?:\\right\s*)?\|\s*_\s*\{{?\s*(?:\\mathrm\s*\{{\s*)?d\s*B\s*\}}?\s*\}}?",
                  r"\\mathrm{dbgain}\1", t)


def _freq_point(arg: str):
    """'j1000' -> 1000.0, 'j10^3' / 'j1\\cdot10^{3}' -> 1000.0, '0' -> 0.0, '\\infty' -> inf; 'j\\omega' -> None"""
    a = re.sub(r"\s+", "", arg)
    if a == "0":
        return 0.0
    if "infty" in a or "∞" in a:
        return float("inf")
    if "omega" in a or a == "jw":
        return None
    m = re.fullmatch(r"j([\d.]+)(?:\\(?:cdot|times)10\^\{?(-?\d+)\}?|\^\{?(\d+)\}?)?", a)
    if not m:
        return None
    base = float(m.group(1))
    return base * 10 ** int(m.group(2)) if m.group(2) else base ** int(m.group(3)) if m.group(3) else base


def _check_freq(kind, k: str, point, rhs, net, parts: dict):
    """H(j\\omega) / |H(j\\omega)| / 20 log|H| / the angle of H - at any w (a function of \\omega) or at one (j1000,
    0, \\infty) - against the circuit's H(s) / Z(s) at s = jw -> (right?, text, sure?) or None"""
    import cmath
    import math
    s, w = sp.Symbol("s"), sp.Symbol("omega")
    if not all(str(x) in ("omega", "w") or _canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) not in ("omega", "w")}).subs(sp.Symbol("w"), w)
    if point is not None and mine.has(w):
        return None
    got = []
    for st in _states(net) or [{}]:
        f = _s_functions(net, st).get(k)
        if f is not None and _decidable(mine.subs(w, 1), [f]):
            got.append((st, sp.sympify(f)))
    if not got:
        return None
    num = _numbers([f for _, f in got] + [mine], extra=[])
    num.pop(w, None)
    pts = [point] if point is not None else [0.0137, 0.137, 1.37, 13.7, 137.0, 1370.0, 13700.0, 137000.0]

    def measure(z):
        if kind == "mag":
            return abs(z)
        if kind == "db":
            return 20 * math.log10(abs(z)) if abs(z) > 0 else -math.inf
        if kind == "arg":
            return cmath.phase(z)
        return z

    def same(a, b):
        if kind == "arg":
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi) <= 0.02 * max(1.0, abs(b))
        if kind == "db":
            return abs(a - b) <= 0.15
        return abs(a - b) <= REL_TOL * max(abs(a), abs(b)) + 1e-12

    hit, shown = [], []
    for st, f in got:
        fe = f.subs(num)
        try:
            theirs = []
            for x in pts:
                if x == 0:
                    z = complex(sp.limit(fe, s, 0))
                elif x == float("inf"):
                    z = complex(sp.limit(fe, s, sp.oo))
                else:
                    z = complex(sp.N(fe.subs(s, sp.I * x)))
                theirs.append(measure(z))
            me = [complex(sp.N(mine.subs(num).subs(w, x))) if point is None else complex(sp.N(mine.subs(num))) for x in pts]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if kind:
            me = [z.real for z in me]
        ok = all(same(a, b) for a, b in zip(me, theirs))
        if not ok and kind == "arg" and point is not None and abs(me[0]) > 6.3:   # '-45' meant in degrees
            ok = same(me[0] * math.pi / 180, theirs[0])
        if ok:
            hit.append(st)
        elif point is not None:
            v = theirs[0]
            shown.append((f"{_fmt(v * 180 / math.pi)}°" if kind == "arg" else f"{_fmt(v)} dB" if kind == "db" else
                          _show(sp.nsimplify(v.real, rational=False) + sp.I * v.imag) if isinstance(v, complex) else _fmt(v))
                         + (f" ({st_text(st)})" if st else ""))
    sure = len({bare_st(st) for st, _ in got}) == 1
    if hit:
        return True, ": as the circuit gives" + (f" ({st_text(hit[0])})" if hit[0] else ""), sure
    return False, (f": the circuit gives {'; '.join(shown[:3])}" if shown else
                   ": not the circuit's " + ("magnitude" if kind == "mag" else "gain in dB" if kind == "db" else
                                             "angle" if kind == "arg" else "frequency response")), sure


def _check_laplace_s(k: str, rhs, ltrans, net, parts: dict, currents: dict):
    """a Laplace transform the student wrote ('V_C(s) = 10/s - 10/(s + 500)') against the circuit's, at a few
    real s"""
    s = sp.Symbol("s")
    if not all(str(x) == "s" or _canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) != "s"})
    cands = [(p, q, r, drawn, e, either) for p, q, r, drawn in ltrans for e, either in _pick(k, r, net, currents)
             if _decidable(mine, [e])]
    if not cands:
        return None
    num = _numbers([c[4] for c in cands] + [mine])
    full, shown = [], []
    for p, q, r, drawn, e, either in cands:
        terms = _terms(e, num)
        if terms is None:
            continue
        T = _scale(terms)
        ss = [x / T for x in (0.037, 0.37, 1.13, 2.9, 7.7, 23.0, 61.0, 170.0, 1300.0)]     # over three decades
        try:
            fm = sp.lambdify(s, mine.subs(num), "math")
            fe = sp.lambdify(s, sp.sympify(e).subs(num), "math")
            me, theirs = [complex(fm(x)).real for x in ss], [complex(fe(x)).real for x in ss]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        same = lambda sg: all(abs(a - sg * b) <= REL_TOL * max(abs(a), abs(b)) + 1e-15 for a, b in zip(me, theirs))   # noqa: E731
        if any(same(sg) for sg in ((1, -1) if either else (1,))):
            if not any(x[0] == p and x[1] == q for x in full):
                full.append((p, q, r, drawn))
            continue
        shown.append(f"{sp.nsimplify(sp.simplify(sp.sympify(e).subs(num)), rational=False)} when {switching_text(p, q)}")
    if full:
        drawn_ok = [h for h in full if h[3]]
        best = drawn_ok or full
        when = "; ".join(dict.fromkeys(switching_text(p, q) for p, q, _, _ in best))
        assumed = any(r["assumed"] for _, _, r, _ in best)
        sure = len(best) == 1 and not assumed and (bool(drawn_ok) or not any(c.state for c in net.components if c.type == "switch"))
        return True, f": as the circuit gives when {when}", sure
    if not shown:
        return None
    return False, f": the circuit gives {'; '.join(dict.fromkeys(shown[:3]))}", not any(c[2]["assumed"] for c in cands)


def _s_functions(net, st) -> dict:
    """what exam answers in the Laplace domain name without the drawing saying it (zero initial state): H(s) / T(s) /
    G(s), the output over the one independent source (the output: a node named o / out / Vo, else across the port's
    terminals, else the op-amp's output); Z(s) / Z_in(s), what that source sees - or Z_th(s) between the port's
    terminals, the sources set to zero; Y(s) = 1 / Z(s)"""
    out = {}
    from dataclasses import replace
    if any(c.ac for c in net.components if c.type in ("voltage_source", "current_source")):
        net = replace(net, components=[replace(c, ac=False, value="1") if c.ac and c.type in ("voltage_source", "current_source")
                                       else c for c in net.components])     # an AC source: a unit input (H does not care)
    srcs = [c for c in net.components if c.type in ("voltage_source", "current_source")]
    by_label = {lab.lower().replace("_", ""): n for n, lab in net.node_labels.items()}
    ports = list(net.ports)
    if len(ports) != 2 and "a" in by_label and "b" in by_label:
        ports = [by_label["a"], by_label["b"]]
    if len(srcs) == 1:
        r = net.solve("s", st)
        vin = net.symbol(srcs[0])
        vout = None
        if r:
            outn = next((by_label[x] for x in ("o", "out", "vo", "vout", "y") if x in by_label), None)
            amps = [c for c in net.components if c.type == "opamp" and "Out" in c.roles]
            if outn is not None:
                vout = _node_v(net, {_canon(k): v for k, v in r.items()}, outn)
            elif len(ports) == 2:
                sol = {_canon(k): v for k, v in r.items()}
                a, b, _ = _port_pair(net, ports)
                va, vb = _node_v(net, sol, a), _node_v(net, sol, b)
                vout = None if va is None or vb is None else va - vb
            elif len(amps) == 1:
                vout = _node_v(net, {_canon(k): v for k, v in r.items()}, amps[0].roles["Out"])
        if vout is not None and vin != 0:
            for k in ("h", "t", "g", "hs", "a", "av"):
                out[k] = sp.simplify(vout / vin)
        a, b = srcs[0].nodes if len(srcs[0].nodes) == 2 else (None, None)
        if a is not None:                                  # what the source sees: the circuit without it, between its ends
            rest = Netlist([c for c in net.components if c is not srcs[0]], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
            z = rest._impedance(a, b, st)
            if z is not None:
                for k in ("z", "zin", "zeq", "zi"):
                    out[k] = z
                out["y"] = out["yin"] = sp.simplify(1 / z) if z != 0 else None
    if len(ports) == 2:
        z = net._impedance(ports[0], ports[1], st)
        if z is not None:
            for k in ("zth", "zab", "zt", "zout", "zo"):
                out[k] = z
            out.setdefault("z", z)
    return {k: v for k, v in out.items() if v is not None}


def _check_s_function(k: str, rhs, net, parts: dict):
    """H(s) = ..., Z_in(s) = ... against the circuit (every state of its switches)"""
    s = sp.Symbol("s")
    if not all(str(x) == "s" or _canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) != "s"})
    got = []
    for st in _states(net) or [{}]:
        f = _s_functions(net, st).get(k)
        if f is not None and _decidable(mine, [f]):
            got.append((st, f))
    if not got:
        return None
    num = _numbers([f for _, f in got] + [mine])
    hit = []
    for st, f in got:
        try:
            fm = sp.lambdify(s, mine.subs(num), "math")
            fe = sp.lambdify(s, sp.sympify(f).subs(num), "math")
            ss = (0.0137, 0.137, 1.37, 13.7, 137.0, 1370.0, 13700.0, 137000.0)
            me, theirs = [complex(fm(x)).real for x in ss], [complex(fe(x)).real for x in ss]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if all(abs(a - b) <= REL_TOL * max(abs(a), abs(b)) + 1e-15 for a, b in zip(me, theirs)):
            hit.append(st)
    if hit:
        return True, ": as the circuit gives" + (f" ({st_text(hit[0])})" if hit[0] else ""), len(got) == 1
    shown = "; ".join(f"{sp.simplify(f)}" + (f" ({st_text(st)})" if st else "") for st, f in got[:3])
    return False, f": the circuit gives {shown}", len(got) == 1


def _decidable(mine, theirs) -> bool:
    """whether the drawing can decide: the circuit's expression rests only on names the student's uses too, or the
    student's is a relation between the circuit's quantities with no numbers in it ('V_A = E - R_1 I_1'). When the
    drawing gives a part only a name (R_1) and the student wrote numbers (from the problem's text: 'V_A = 5', or
    'E/2' with R_1 = R_2 given), nothing can be decided from the drawing"""
    need = set()
    for e in theirs:
        need |= {x for x in sp.sympify(e).free_symbols if str(x) not in ("t", "s", "tau")}
    mine = sp.sympify(mine)
    if need <= set(mine.free_symbols):
        return True
    return not {x for x in mine.atoms(sp.Number) if x not in (0, 1, -1)}


_TYPE_ALIAS = {"resistor": "r", "capacitor": "c", "inductor": "l", "diode": "d", "lamp": "lp"}


def _solve_parts(net, st, mode: str):
    """the circuit solved with every two-terminal part's current too -> (the solution as Netlist.solve gives it,
    {canonical name: value} of what exam answers call by a part's name - its voltage v_R1 (first node minus second),
    current i_R1 (first node through it), power p_R1 = v i; in the phasor domain the average power P (half the real
    part of V I*: amplitudes - 'power' 'peak' - or all of it: RMS values - 'rms') and the reactive Q; v_R / i_R / p_R
    with one part of that kind, p_s / i_s with one source) - these in either sign (the part's direction is not drawn)"""
    from .circuit import TWO_TERMINAL, Current
    two = [c for c in net.components if c.type in TWO_TERMINAL and len(c.nodes) == 2]
    probes = [Current(f"__i_{c.id}", c.id, c.nodes[0], c.nodes[1]) for c in two]
    r = Netlist(net.components, net.currents + probes, net.ground, net.node_labels, net.ports, voltages=net.voltages).solve(mode, st)
    if not r:
        return {}, {}
    sol = {k: v for k, v in r.items() if not k.startswith("__")}
    node = lambda n: 0 if n == net.ground else r.get(f"V_{net.node_labels.get(n, n)}")   # noqa: E731
    quant = {}
    count = {}
    for c in two:
        count[c.type] = count.get(c.type, 0) + 1
    srcs = [c for c in two if c.type in ("voltage_source", "current_source")]
    for c in two:
        va, vb = node(c.nodes[0]), node(c.nodes[1])
        i = r.get(f"__i_{c.id}")
        if i is None and ((c.type == "switch" and st.get(c.id) == "open") or (c.type == "capacitor" and mode == "dc")):
            i = sp.Integer(0)
        v = None if va is None or vb is None else sp.simplify(va - vb)
        names = [_canon(c.label or c.id)]
        if count.get(c.type) == 1 and c.type in _TYPE_ALIAS:
            names.append(_TYPE_ALIAS[c.type])
        if len(srcs) == 1 and c is srcs[0]:
            names.append("s")
        for nm in names:
            if v is not None:
                quant.setdefault("v" + nm, v)
            if i is not None:
                quant.setdefault("i" + nm, sp.simplify(i))
            if mode == "dc" and v is not None and c.type == "capacitor":      # energy held: C v^2 / 2
                quant.setdefault("w" + nm, sp.simplify(net.symbol(c) * v ** 2 / 2))
                quant.setdefault("e" + nm, sp.simplify(net.symbol(c) * v ** 2 / 2))
                quant.setdefault("q" + nm, sp.simplify(net.symbol(c) * v))      # the charge it holds: C v
            if mode == "dc" and i is not None and c.type == "inductor":       # L i^2 / 2
                quant.setdefault("w" + nm, sp.simplify(net.symbol(c) * i ** 2 / 2))
                quant.setdefault("e" + nm, sp.simplify(net.symbol(c) * i ** 2 / 2))
            if v is not None and i is not None:
                if mode == "phasor":
                    vi = sp.expand_complex(v * sp.conjugate(i))
                    quant.setdefault("p" + nm, (sp.re(vi), sp.re(vi) / 2))       # (rms, peak)
                    quant.setdefault("q" + nm, (sp.im(vi), sp.im(vi) / 2))
                    if c.type in ("resistor", "capacitor", "inductor") and sp.simplify(i) != 0:
                        z = sp.simplify(sp.expand_complex(v / i))                # its impedance Z_L = jwL, its
                        sol.setdefault("Z_" + nm, z)                            # reactance X_L (either sign:
                        quant.setdefault("x" + nm, sp.im(z))                    # X_C = 1/wC or -1/wC by the course)
                else:
                    quant.setdefault("p" + nm, sp.simplify(v * i))
    if mode == "dc" and count.get("capacitor") == 1 and "qc" in quant:
        quant.setdefault("q", quant["qc"])                 # one capacitor: Q is its charge (no reactive power in DC)
    flows = [quant["i" + _canon(c.label or c.id)] for c in two if "i" + _canon(c.label or c.id) in quant]
    flows = [f for f in flows if sp.simplify(f) != 0]
    if flows and all(sp.simplify(f - flows[0]) == 0 or sp.simplify(f + flows[0]) == 0 for f in flows):
        quant.setdefault("i", flows[0])                    # one loop: I, the current every part carries
    if mode == "dc" and srcs:                              # the power the sources give together
        given = []
        for c in srcs:
            va, vb, i = node(c.nodes[0]), node(c.nodes[1]), r.get(f"__i_{c.id}")
            if va is None or vb is None or i is None:
                given = []
                break
            given.append(-(va - vb) * i)
        if given:
            for k in ("p", "pt", "ptotal", "ptot"):
                quant.setdefault(k, sp.simplify(sum(given)))
    if mode == "phasor" and srcs:                          # what the sources give together (all three phases)
        vis = []
        for c in srcs:
            va, vb = node(c.nodes[0]), node(c.nodes[1])
            i = r.get(f"__i_{c.id}")
            if va is None or vb is None or i is None:
                vis = []
                break
            vis.append(sp.expand_complex(-(va - vb) * sp.conjugate(i)))      # given: minus what it takes in
        if vis:
            vi = sp.expand_complex(sum(vis))
            for k in ("p", "pt", "ptotal", "pavg", "pav"):
                quant.setdefault(k, (sp.re(vi), sp.re(vi) / 2))      # the average power: (all of Re V I*, half of it)
            for k in ("q", "qt", "qtotal"):
                quant.setdefault(k, (sp.im(vi), sp.im(vi) / 2))
            quant.setdefault("s", (vi, vi / 2))                     # S = P + jQ ('S = 10 VA': its size)
            if sp.simplify(vi) != 0:
                pf = sp.simplify(sp.Abs(sp.re(vi)) / sp.Abs(vi))
                for k in ("pf", "fp", "cosphi", "costheta", "cosvarphi"):
                    quant.setdefault(k, pf)
                re_ = _num(sp.re(vi))
                if re_ is not None and re_ != 0:                    # the load's angle: Z's, V over I
                    d = -vi if re_ < 0 else vi
                    for k in ("phi", "theta", "varphi", "phiz", "thetaz"):
                        quant.setdefault(k, sp.atan2(sp.im(d), sp.re(d)))
        if len(srcs) == 1 and "vs" in quant and "is" in quant:
            for k, x in (("irms", quant["is"]), ("vrms", quant["vs"]), ("ieff", quant["is"]), ("veff", quant["vs"])):
                quant.setdefault(k, (sp.Abs(x), sp.Abs(x) / sp.sqrt(2)))    # the phasors RMS values / amplitudes
    return sol, quant


def _diode_parts(net, st, dstate, vd) -> dict:
    """the parts' own voltages, currents and powers with each diode as it turned out - on: a source of its drop, in
    breakdown: of V_Z, off: open - and the power the real sources give in all"""
    from .circuit import Component, zener_voltage
    if not dstate:
        return {}
    comps, fake = [], set()
    for c in net.components:
        if c.type not in ("diode", "zener"):
            comps.append(c)
            continue
        anode = next((n for n in c.nodes if n != c.plus), None)
        if anode is None:
            return {}
        if dstate.get(c.id) == "on":
            comps.append(Component(c.id, "voltage_source", [anode, c.plus], plus=anode, value=repr(float(vd))))
            fake.add(c.id)
        elif dstate.get(c.id) == "breakdown":
            vz = zener_voltage(c.value)
            if vz is None:
                return {}
            comps.append(Component(c.id, "voltage_source", [anode, c.plus], plus=c.plus, value=repr(float(vz))))
            fake.add(c.id)
    _, quant = _solve_parts(Netlist(comps, net.currents, net.ground, net.node_labels, net.ports, voltages=net.voltages), st, "dc")
    for k in ("p", "pt", "ptotal", "ptot", "s", "vs", "is", "ps"):
        quant.pop(k, None)                                 # the diodes stood in by sources are no sources
    real = [c for c in net.components if c.type in ("voltage_source", "current_source")]
    given = [quant.get("p" + _canon(c.label or c.id)) for c in real]
    if given and all(g is not None for g in given):
        for k in ("p", "pt", "ptotal", "ptot"):
            quant[k] = sp.simplify(-sum(given))            # what they give: minus what they take in
    return {k: v for k, v in quant.items() if not any(k.endswith(_canon(f)) and k[:-len(_canon(f))] in ("v", "i", "p")
                                                       for f in fake)}


def _check_superposed(k: str, rhs_text: str, net, parts: dict, currents: dict, rlc, states):
    """V_A', I_2'' (or V_A^{(1)}): what one source gives alone, the others set to zero (superposition) - right when
    it is one source's share (which one the student numbered first is his: not sure), else wrong (not sure either:
    he may have grouped sources) -> (right?, text, sure?) or None"""
    srcs = [c for c in net.components if c.type in ("voltage_source", "current_source")]
    if len(srcs) < 2 or len(states) != 1:
        return None
    rhs = parse_side(rhs_text, known=rlc, k_is_name="k" in parts)
    if rhs is None or not all(_canon(x) in parts for x in rhs.free_symbols):
        return None
    mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols})
    shares = []
    for c in srcs:
        r = net.solve("dc", states[0], keep={c.id})
        if not r:
            return None
        sol = {_canon(n): v for n, v in r.items()}
        key = k if k in sol else (_canon(currents[k]) if k in currents and _canon(currents[k]) in sol else None)
        if key is None:
            return None
        shares.append((c.label or c.id, sol[key]))
    if not _decidable(mine, [v for _, v in shares]):
        return None
    hit = [name for name, v in shares if _close(mine, v)]
    shown = "; ".join(f"{_show(v)} ({name} alone)" for name, v in shares)
    return (bool(hit), f": as {hit[0]} alone gives (superposition)" if hit else f": one source alone gives {shown}",
            False)


def _resonance(net) -> dict:
    """one inductor and one capacitor: w_0 = 1 / sqrt(L C), f_0 = w_0 / 2 pi"""
    ls = [c for c in net.components if c.type == "inductor"]
    cs = [c for c in net.components if c.type == "capacitor"]
    if len(ls) != 1 or len(cs) != 1:
        return {}
    w0 = 1 / sp.sqrt(net.symbol(ls[0]) * net.symbol(cs[0]))
    return {"w0": w0, "omega0": w0, "wo": w0, "omegao": w0, "wr": w0, "f0": w0 / (2 * sp.pi), "fo": w0 / (2 * sp.pi),
            "fr": w0 / (2 * sp.pi)}


_ANGLE_NAMES = {"phi", "theta", "varphi", "phiz", "thetaz"}      # shown in degrees
_RESONANT = {"w0", "omega0", "wo", "omegao", "wr", "f0", "fo", "fr"}


def _pure_resonance(net, st) -> bool:
    """at w_0 = 1/sqrt(LC) what the one source sees is real - a plain series or parallel RLC: there w_0 is the
    resonance by every definition (unity power factor, the peak); elsewhere (a coil's resistance beside a capacitor)
    a book's resonance may differ"""
    st = {k: v for k, v in st.items() if k not in ("diodes", "power", "VT", "R_eq")}
    z, r = _s_functions(net, st).get("z"), _resonance(net)
    if z is None or not r:
        return False
    num = _numbers([z, r["w0"]])
    try:
        val = complex(sp.N(sp.sympify(z).subs(num).subs(sp.Symbol("s"), sp.I * sp.sympify(r["w0"]).subs(num))))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return abs(val.imag) <= 1e-6 * max(1.0, abs(val))
_MOMENT_FREE = {"alpha", "omegan", "wn", "zeta", "xi", "omegad", "wd", "bw", "w0", "omega0", "wo", "omegao", "wr",
                "f0", "fo", "fr", "omegac", "omega3db", "omegab", "fc", "f3db", "fb"}   # the circuit's, not a moment's


def _second_order(net, st) -> dict:
    """two capacitors / inductors: the natural response's s^2 + 2 alpha s + w_n^2 (Netlist.natural_polynomial) ->
    alpha (the neper frequency), w_n (the undamped natural frequency), zeta = alpha / w_n (the damping ratio), w_d =
    sqrt(w_n^2 - alpha^2) (the damped frequency, when it rings), the bandwidth 2 alpha (rad/s)"""
    if sum(c.type in ("capacitor", "inductor") for c in net.components) != 2:
        return {}
    co = net.natural_polynomial(st)
    if co is None or len(co) != 3:
        return {}
    a1, a0 = co[1], co[2]
    wn, alpha = sp.sqrt(a0), a1 / 2
    out = {"alpha": alpha, "omegan": wn, "wn": wn, "zeta": sp.simplify(alpha / wn), "xi": sp.simplify(alpha / wn),
           "bw": a1}
    out["omegad"] = out["wd"] = sp.sqrt(sp.simplify(a0 - alpha ** 2))
    return out


def _guess_ground(net):
    """the reference node of a drawing without a ground symbol: where most sources' '-' ends meet (the bottom wire of
    a textbook circuit), else the node the most parts touch"""
    minus, deg = {}, {}
    for c in net.components:
        for n in c.nodes:
            deg[n] = deg.get(n, 0) + 1
        if c.type in ("voltage_source", "current_source", "dependent_voltage_source", "dependent_current_source") \
                and len(c.nodes) == 2 and c.plus in c.nodes:
            m = next(n for n in c.nodes if n != c.plus)
            minus[m] = minus.get(m, 0) + 1
    if minus:
        return max(minus, key=lambda n: (minus[n], deg.get(n, 0)))
    return max(deg, key=deg.get) if deg else None


def _givens(stmts, net):
    """the values the student writes for parts the drawing only names ('R_1 = 10 k\\Omega', 'E = 12 V', '\\beta = 100')
    - the problem's data, copied before the solution -> ({part id: value text}, the statements that gave them)"""
    from .circuit import transistor_beta
    named = {}
    for c in net.components:
        if isinstance(net.symbol(c), sp.Symbol) and c.type not in ("switch", "meter", "opamp", "transformer", "and", "or",
                                                                  "not", "nand", "nor", "xor", "ic", "other"):
            if re.sub(r"[_{}\s]", "", (c.label or "")).lower() not in ("rl", "zl"):   # a load is often what is asked
                named[_canon(c.label or c.id)] = c
    qs = [c for c in net.components if c.type in ("npn", "pnp") and transistor_beta(c.value).free_symbols]
    got, used = {}, []
    for s in stmts:
        texts = s.side_texts or []
        if len(texts) != 2:
            continue
        lhs, rhs = parse_side(texts[0]), parse_side(texts[1])
        if lhs is None or rhs is None or not lhs.is_Symbol or rhs.free_symbols:
            continue
        try:
            v = float(sp.N(rhs))
        except (TypeError, ValueError):
            continue
        k = _canon(lhs)
        if k in named and named[k].id not in got:
            got[named[k].id] = repr(v)
            used.append(s)
        elif k in ("beta", "hfe") and len(qs) == 1 and qs[0].id not in got:
            got[qs[0].id] = f"β={v:g}"
            used.append(s)
    return got, used


def with_values(net, values: dict):
    """the problem's data, as the lecturer gives it once for the question ({'R_1': '2k', 'E': '10', 'V_s': '10cos(1000t)',
    'beta': '100'}): each part the drawing names takes its value by its name (a value the drawing already writes
    stays) -> a new Netlist; the student's answers are then judged against them, sure"""
    from dataclasses import replace

    from .circuit import transistor_beta
    vals = {_canon(k): str(v) for k, v in (values or {}).items()}
    if not vals:
        return net
    comps = []
    for c in net.components:
        v = vals.get(_canon(c.label or "")) or vals.get(_canon(c.id))
        if c.type in ("npn", "pnp") and transistor_beta(c.value).free_symbols:
            b = vals.get("beta") or vals.get("hfe")
            if b is not None:
                c = replace(c, value=f"β={b}")
        elif v is not None and isinstance(net.symbol(c), sp.Symbol):
            ac = bool(re.search(r"cos|sin|∠|\\angle", v)) or c.ac
            c = replace(c, value=v, ac=ac) if c.type in ("voltage_source", "current_source") else replace(c, value=v)
        comps.append(c)
    return replace(net, components=comps)


def check_against(stmts, net, conf: float = 1.0, problem=0) -> list[Check]:
    """stmts: circuit_statements(...) of the student's work; net: the circuit's Netlist. With switches, every state
    they may be in is tried; with a capacitor or an inductor, the steady state (C open, L short) and the time
    constant (tau = R C or L / R, R what the part sees) - an equation that fits none of those is not judged wrong
    then (it may be about another moment). A drawing that only names its parts: the values the student writes for
    them first ('R_1 = 10k') are used for the rest - suggestions then, not verdicts (those values come from a text
    the grader does not see)."""
    net = _distinct_nodes(net)
    variants = _ac_orientations(net)
    if variants:                                   # an AC source whose polarity is not drawn: each way round it
        return _across_orientations([_check_given(stmts, v, conf, problem) for v in variants])
    return _check_given(stmts, net, conf, problem)


_KIND_OF = {"v": ("V", "a voltage"), "e": ("V", "a voltage"), "u": ("V", "a voltage"), "i": ("A", "a current"),
            "p": ("W", "a power"), "r": ("Ω", "a resistance"), "z": ("Ω", "an impedance")}


_NUM_RE = r"[-+]?\s*\d+(?:[.,]\d+)?"
_VALUE_UNIT = re.compile(r"\s*(?:" + _NUM_RE + r"\s*\\angle\s*\{?\s*" + _NUM_RE + r"\s*(?:\^\s*(?:\{\s*\\circ\s*\}|\\circ)|°)?\s*\}?|"   # polar,
                         r"\(?\s*" + _NUM_RE + r"\s*[-+]\s*(?:j\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*j)\s*\)?|"     # rectangular,
                         r"[-+]?\s*\d+(?:[.,]\d+)?\s*(?:(?:\\cdot|\\times|\*)\s*10\s*\^\s*\{?\s*[-+]?\s*\d+\s*\}?)?)"    # a plain number
                         r"(?:\s|\\[,;:! ]|~)*(?:\\left\s*)?(?:\[\s*)?(?:\\(?:mathrm|text|rm)\s*\{\s*)?(?:\\mu\s*|[pnuµmkMG])?\s*"
                         r"(?P<unit>V\s*A\s*R|V\s*A|var|V|A|W|\\Omega|Ω|ohm|F|H|Hz)\s*\}?\s*(?:\\right\s*)?(?:\]\s*)?")


def _unit_written(side: str) -> str:
    """the unit after a plain number ('7.5 A', '2.25 \\cdot 10^{-3} [V]', '1 k\\Omega' -> 'A', 'V', 'Ω'), else ''.
    A numeric expression ('\\frac{12}{4 k\\Omega}') carries its unit inside, not after: ''"""
    m = _VALUE_UNIT.fullmatch(side)
    if not m:
        return ""
    unit = re.sub(r"\s+", "", m.group("unit"))
    return {"\\Omega": "Ω", "ohm": "Ω"}.get(unit, unit)


def _unit_mismatch(s):
    """'V_A = 7.5 A', 'I_1 = 2.25 mV', 'P_{R1} = 10 mA': a value written with a unit of another kind than the
    quantity its letter names (V a voltage, I a current, P a power, R / Z a resistance) - a note, or None"""
    texts = s.side_texts or []
    if len(texts) != 2 or not s.unit:
        return None
    name = re.sub(r"[\\{}\s]", "", texts[0])
    if not re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)*(?:\((?:[st]|0\^?[+-]?|infty|∞)\))?", name) or name[0].lower() not in _KIND_OF:
        return None                                # (v_C(t), i_L(0^+), v_C(\infty) too)
    if re.fullmatch(r"[Pp]_?[fF]", name):
        return None                                # pf: the power factor
    from .selfcheck import _dimension
    want = _dimension(_KIND_OF[name[0].lower()][0])
    got = _dimension({"VAR": "W", "var": "W", "VA": "W"}.get(s.unit, s.unit))
    if want is None or got is None or sp.simplify(want / got) == 1:
        return None
    return f"{texts[0].strip()} is {_KIND_OF[name[0].lower()][1]}, and {s.unit} is not its unit"


_PORT_NAMES = {"vth", "voc", "vt", "rth", "req", "rab", "rn", "rt", "rin", "in", "isc", "inorton", "pmax", "plmax",
               "pmaxl", "plmaximum"}


def _distinct_nodes(net):
    """a node with no letter whose own name reads like another's label or a part's name (node 'a' beside a node
    labelled 'A', node 'c' beside a capacitor C): once names are compared without case both were 'V_A' - one value hid
    the other (a voltage marked across a part came out 0, a false 'wrong') - renamed so they stay apart"""
    from dataclasses import replace
    taken = {_canon(lab) for lab in net.node_labels.values()}
    taken |= {_canon(x) for c in net.components for x in (c.id, c.label) if x}
    nodes = {n for c in net.components for n in c.nodes} | {n for n, _ in net.terminals} | set(net.ports)
    clash = sorted(n for n in nodes if n is not None and n != net.ground and n not in net.node_labels and _canon(n) in taken)
    if not clash:
        return net
    ren = {n: f"#{k}" for k, n in enumerate(clash)}
    r = lambda n: ren.get(n, n)                    # noqa: E731
    comps = [replace(c, nodes=[r(n) for n in c.nodes], plus=r(c.plus), roles={k: r(v) for k, v in c.roles.items()})
             for c in net.components]
    return replace(net, components=comps, currents=[replace(x, frm=r(x.frm), to=r(x.to)) for x in net.currents],
                   node_labels={r(k): v for k, v in net.node_labels.items()}, ports=[r(n) for n in net.ports],
                   terminals=[(r(n), xy) for n, xy in net.terminals],
                   voltages=[replace(v, plus=r(v.plus), minus=r(v.minus)) for v in net.voltages])


def _ac_orientations(net) -> list:
    """an AC source drawn without '+' / '-' (a circle with a sine in it has no polarity): the netlists with each way
    round it may be - [] when none is open; one source: both ways; several meeting at one node (a wye): all with '-'
    there, or all with '+'; else every way (three sources at most)"""
    import itertools
    from dataclasses import replace
    open_ = [c for c in net.components if c.type in ("voltage_source", "current_source") and c.ac
             and len(c.nodes) == 2 and c.plus not in c.nodes]
    if not open_:
        return []
    shared = set(open_[0].nodes)
    for c in open_[1:]:
        shared &= set(c.nodes)
    if len(open_) >= 2 and len(shared) == 1:
        n0 = shared.pop()
        combos = [tuple(next(x for x in c.nodes if x != n0) for c in open_), tuple(n0 for _ in open_)]
    elif len(open_) <= 3:
        combos = list(itertools.product(*[c.nodes for c in open_]))
    else:
        return []
    out = []
    for combo in combos:
        plus = {c.id: p for c, p in zip(open_, combo)}
        out.append(replace(net, components=[replace(c, plus=plus[c.id]) if c.id in plus else c for c in net.components]))
    return out


def _across_orientations(per: list) -> list:
    """the checks against each way round the AC sources (their polarity not drawn) -> one list: right when the
    equation holds one way - sure only when it holds every way (a power, a magnitude, an impedance) -, wrong when it
    holds none (sure when every way judged it so, sure)"""
    order, by = [], {}
    for cs in per:
        seen = {}
        for c in cs:
            base = c.pieces[0] if c.pieces else c.text
            key = (base, seen.get(base, 0))
            seen[base] = key[1] + 1
            if key not in by:
                by[key] = []
                order.append(key)
            by[key].append(c)
    out = []
    for key in order:
        cs = by[key]
        for verdict in ("right", "right value, wrong unit", "wrong"):
            same = [c for c in cs if c.verdict == verdict]
            if same:
                c = same[0]
                c.sure = c.sure and len(same) == len(per) and all(x.sure for x in same)
                if verdict == "right" and len(same) < len(per):
                    c.text += " - with the AC source's '+' taken at one end (its polarity is not drawn)"
                out.append(c)
                break
        else:
            out.append(cs[0])
    return out


def _is_name(text: str) -> bool:
    """a side that is one quantity's name ('V_A', 'I_{R1}', 'v_C(t)', 'R_{eq}'), no operator outside its brackets"""
    t = text
    for _ in range(4):
        t = re.sub(r"\{[^{}]*\}|\([^()]*\)", "", t)
    return bool(re.fullmatch(r"\s*\\?[A-Za-z][^+\-*/=\\]*", t)) and "\\frac" not in text and "\\cdot" not in text


def _bare_steps(text: str, texts: list) -> list:
    """'P_{R1} = 2^2 \\cdot 4 = 16 mW': a step in bare numbers, in the units an EE student keeps (mA, kΩ - so mW), that
    is the last value in its unit written ('16 mW') - that step in SI units too (16 against the circuit's 0.016 W was a
    false sure wrong); a step in SI units stays ('0.002^2 \\cdot 4000 = 16 mW')"""
    from .selfcheck import _UNIT, _close, _join_digits, _num, _parse, _si
    m = _UNIT.match(_join_digits(text.split("=")[-1].strip()))
    f = _si(m.group("unit")) if m else None
    try:
        last = float(texts[-1])
    except ValueError:
        return texts
    if f in (None, 1.0):
        return texts
    out = list(texts)
    for k in range(1, len(texts) - 1):
        if re.search(r"[A-Za-z]", re.sub(r"\\(?:frac|cdot|times|sqrt|left|right|pi)(?![A-Za-z])", "", texts[k])):
            continue
        e = _parse(texts[k])
        v = _num(e) if e is not None else None
        if v is not None and _close(v * f, last) and not _close(v, last):
            out[k] = f"({texts[k]}) \\cdot {f!r}"
    return out


def _chain_pairs(stmts) -> tuple:
    """a chain 'V_A = \\frac{E R_2}{R_1 + R_2} = 3.33' -> equalities checked one by one: each later side against the
    first when that is one quantity's name, else each side against the next ('I_1 + I_2 = I_3 = 5 mA') -> (the
    statements, {a pair's text: its chain's text})"""
    from dataclasses import replace
    out, of = [], {}
    for s in stmts:
        texts = s.side_texts or []
        if len(texts) <= 2:
            out.append(s)
            continue
        texts = _bare_steps(s.text, texts)
        pairs = [(texts[0], t) for t in texts[1:]] if _is_name(texts[0]) else list(zip(texts, texts[1:]))
        for k, (a, b) in enumerate(pairs):
            pt = f"{a} = {b}"
            of.setdefault(pt, s.text)                    # the unit written after the last side: the last pair's
            out.append(replace(s, text=pt, side_texts=[a, b], unit=s.unit if k == len(pairs) - 1 else ""))
    return out, of


def _merge_chains(checks: list, of: dict) -> list:
    """the checks of a chain's equalities -> one for the chain: wrong when one of them is (which one said), right
    when the rest are right (sure when every one is)"""
    if not of:
        return checks
    out, groups = [], {}
    for c in checks:
        chain = of.get(c.pieces[0]) if c.pieces else None
        if chain is None:
            out.append(c)
            continue
        if chain not in groups:
            groups[chain] = []
            out.append(chain)
        groups[chain].append(c)
    merged = []
    for c in out:
        if not isinstance(c, str):
            merged.append(c)
            continue
        cs = groups[c]
        wrong = [x for x in cs if x.verdict == "wrong"]
        right = [x for x in cs if x.verdict == "right"]
        unit = [x for x in cs if x.verdict == "right value, wrong unit"]
        if wrong or unit:                          # a false equality, else a right value with a unit of another kind
            w = (wrong or unit)[0]
            pair = w.pieces[0]
            msg = w.text[len(f"'{pair}'"):].lstrip(": ") if w.text.startswith(f"'{pair}'") else w.text
            w.text, w.pieces = f"'{c}': '{pair}': {msg}", [c]
            merged.append(w)
        elif right:
            r = right[0]
            r.sure = all(x.sure for x in right)
            r.confidence = min(x.confidence for x in right)
            r.text, r.pieces = f"'{c}': as the circuit gives", [c]
            merged.append(r)
        else:
            for x in cs:
                x.pieces = [c]
                merged.append(x)
    return merged


def _check_given(stmts, net, conf: float = 1.0, problem=0) -> list[Check]:
    """check_against for one netlist: the values the student wrote for the parts it only names used, if any; a
    chain of equalities checked one equality at a time"""
    stmts, of = _chain_pairs(stmts)
    return _merge_chains(_check_valued(stmts, net, conf, problem), of)


def _check_valued(stmts, net, conf: float = 1.0, problem=0) -> list[Check]:
    givens, used = _givens(stmts, net)
    if not givens:
        return _check_against(stmts, net, conf, problem)
    from dataclasses import replace
    valued = replace(net, components=[replace(c, value=givens[c.id]) if c.id in givens else c for c in net.components])
    out = _check_against([s for s in stmts if s not in used], valued, conf, problem)
    for c in out:
        c.sure = False
        c.text += " (with the values the student wrote for the parts)"
    return out


def _check_against(stmts, net, conf: float = 1.0, problem=0) -> list[Check]:
    trusted = not net.problems()
    guessed = net.ground is None and bool(net.components)
    if guessed:                                    # no ground drawn: node voltages from a guessed reference
        from dataclasses import replace
        net = replace(net, ground=_guess_ground(net))
    node_v = {_canon(f"V_{lab}") for lab in net.node_labels.values()}
    island_v = {_canon(f"V_{net.node_labels.get(n, n)}") for group, _ in net.islands() for n in group}
    # the circuit's solution rests on every value read off the drawing ('10k' by R1): to decide, all those readings
    # must be sure (Kirchhoff's law at a node rests on the arrows only)
    values_ok = not any(c.value and c.conf < LABEL_SURE and not isinstance(net.symbol(c), sp.Symbol)
                        for c in net.components)
    currents = {_canon(c.label): c.label for c in net.currents}
    parts = {}                                     # a part's name -> its value: a number written by it, or a symbol
    for c in net.components:
        v = net.symbol(c)
        parts[_canon(c.label or c.id)] = v
        if isinstance(v, sp.Symbol):
            parts[_canon(v)] = v
    nodev = {}                                     # node voltages by name: 'V_1' of a node labelled 1
    for n, lab in net.node_labels.items():
        nodev.setdefault(_canon(f"V_{lab}"), []).append(n)
    ambiguous_names = {k for k, ns in nodev.items() if len(ns) > 1}     # two nodes labelled 'A' and 'a': one name, once
    for c in net.components:                                            # names are compared without case
        k = _canon(c.id)
        if c.label is None and k in nodev and k in parts and parts[k] == net.symbol(c):
            del parts[k]                           # an unlabelled part's own id ('V1') is no name on the drawing - node 1's
        elif c.label and _canon(c.label) in nodev:  # V_1 is; a source labelled V_1 beside a node labelled 1: either
            ambiguous_names.add(_canon(c.label))
    for kind, alias in (("resistor", "r"), ("capacitor", "c"), ("inductor", "l")):   # 'tau = R C' with one of each
        same = [c for c in net.components if c.type == kind]
        if len(same) == 1 and alias not in parts and not isinstance(net.symbol(same[0]), sp.Symbol):
            parts[alias] = net.symbol(same[0])
    from .circuit import transistor_beta
    betas = {transistor_beta(c.value) for c in net.components if c.type in ("npn", "pnp")}
    from .circuit import fet_params
    fets = [fet_params(c) for c in net.components if c.type in ("nmos", "pmos")]
    if len(fets) == 1 and fets[0] is not None:     # K and V_t as names
        parts["k"], parts["vt"], parts["vth"], parts["vtn"] = fets[0][0], fets[0][1], fets[0][1], fets[0][1]
    if len(betas) == 1:                            # beta / h_FE: the transistors' current gain, as written by them
        parts["beta"] = parts["hfe"] = betas.pop()
    rlc = {_canon(c.label or c.id) for c in net.components if c.type in ("resistor", "capacitor", "inductor")}
    kcl = _kcl_exprs(net)
    logic = {}                                     # a logic circuit: each gate output's Boolean function, by its name
    if any(c.type in ("and", "or", "not", "nand", "nor", "xor") for c in net.components):
        for n, f in net.logic().items():
            if n in net.node_labels:
                logic[_canon(net.node_labels[n])] = f
    states = _states(net)
    ac = any(c.ac for c in net.components if c.type in ("voltage_source", "current_source"))
    storage = not ac and any(c.type in ("capacitor", "inductor") for c in net.components)
    w = None
    from .circuit import ac_phasor
    if ac:                                         # a sinusoidal circuit: phasors at the sources' angular frequency
        ws = {ac_phasor(c.value)[1] for c in net.components if c.ac and ac_phasor(c.value)}
        w = ws.pop() if len(ws) == 1 else None
    mixed = ac and (len({ac_phasor(c.value)[1] for c in net.components if c.ac and ac_phasor(c.value)}) > 1 or
                    any(not c.ac for c in net.components if c.type in ("voltage_source", "current_source")))
    sup = None                                     # DC and AC together: each group of sources alone, added up
    sols = None
    either = set()                                 # names a part's direction leaves the sign of open (v_R1, i_R1, P_R1)
    marked = {_canon(v.label) for v in net.voltages}   # voltages the drawing marks with '+' / '-': their sign is set
    unsure_marks = {_canon(v.label) for v in net.voltages if v.conf < LABEL_SURE}
    if len(net.ports) == 2 and not _port_pair(net, net.ports)[2]:
        either |= {"vth", "voc", "vab", "vt", "in", "isc", "inorton"}   # which terminal is the port's '+': not drawn
    pending = []                                   # equations in the student's own unknowns, solved together after
    trans = None                                   # first-order responses to the switchings, worked out when asked
    ltrans = None                                  # the same in the Laplace domain, any order
    out = []

    stmt_names = set()

    port_guessed = len(net.ports) != 2 and {"a", "b"} <= {lab.lower() for lab in net.node_labels.values()}
    ab = [n for n, lab in net.node_labels.items() if lab.lower() in ("a", "b")]
    ab_loaded = port_guessed and len(ab) == 2 and any(set(ab) <= set(c.nodes) for c in net.components)

    def verdict(ok, text, s, sure=True):
        p = min(conf, s.conf)
        if (guessed and stmt_names & node_v) or stmt_names & island_v:
            sure = False                           # a node's voltage, from a reference the drawing does not mark
        if stmt_names & unsure_marks:
            sure = False                           # a voltage marked on the drawing whose name or marks are not read sure
        if not ok and stmt_names & {"vp", "vn", "vplus", "vminus"}:
            sure = False                           # V_+ / V_- may be the op-amp's supply rails, not its inputs
        if stmt_names & ambiguous_names:
            sure = False                           # a name two things on the drawing answer to
        if not ok and port_guessed and (stmt_names & _PORT_NAMES or (ab_loaded and "vab" in stmt_names)):
            sure = False                           # R_eq / V_th taken between nodes A and B - no terminals drawn: the
            #                                        student's R_eq may be some part of the circuit's; V_ab with a part
            #                                        between A and B: maybe with that load taken out (V_th)
        note = _unit_mismatch(s) if ok else None
        if note:                                   # 'V_A = 7.5 A': the number the circuit gives, the unit of another kind
            out.append(Check(problem, "right value, wrong unit", text + f" - but {note}", min(p, 0.5), False, [s.text], line=s.line))
            return
        out.append(Check(problem, "right" if ok else "wrong", text, p, trusted and sure and p >= SURE, [s.text], line=s.line))
    for s in stmts:
        texts = s.side_texts or []
        if len(texts) < 2:
            continue
        if logic and len(texts) == 2 and re.fullmatch(r"\s*[A-Za-z](?:\s*_\s*\{?\s*\w+(?:\s+\w+)*\s*\}?)?\s*", texts[0]) \
                and _canon(texts[0]) in logic:                                  # 'F = A B + C\'' against the gates
            mine = parse_bool(texts[1])
            if mine is not None:
                from sympy.logic.boolalg import Xor
                from sympy.logic.inference import satisfiable
                up = lambda e: e.xreplace({x: sp.Symbol(str(x).upper()) for x in e.free_symbols})   # noqa: E731
                theirs = logic[_canon(texts[0])]
                ok = not satisfiable(Xor(up(mine), up(theirs)))
                verdict(ok, f"'{s.text}'" + (": as the gates give" if ok else f": the gates give {texts[0].strip()} = {theirs}"), s)
                continue
        sp_m = re.fullmatch(r"\s*([A-Za-z](?:\s*_\s*\{?\s*[A-Za-z0-9]+\s*\}?)?)\s*('+)\s*", texts[0])
        if sp_m and len(texts) == 2 and not ac and not storage:
            res = _check_superposed(_canon(sp_m.group(1)), texts[1], net, parts, currents, rlc, states)
            if res is not None:                                 # V_A', I_2'': one source's share (superposition)
                ok, text, sure = res
                verdict(ok, f"'{s.text}'" + text, s, values_ok and sure)
                continue
        fr = _FREQ.fullmatch(texts[0])                              # 'H(j\omega) = ...', '|H(j1000)| = 0.707'
        if fr and _canon(fr.group(2)) in _S_NAMES:
            rhs = parse_side(texts[-1], ac=True, known=rlc, k_is_name="k" in parts)
            if rhs is not None:
                res = _check_freq(fr.group(1), _canon(fr.group(2)), _freq_point(fr.group(3)), rhs, net, parts)
                if res is not None:
                    ok, text, sure = res
                    verdict(ok, f"'{s.text}'" + text, s, sure and values_ok)
            continue
        moment = None if ac else _MOMENT.fullmatch(texts[0])           # 'v_C(0^+) = 5', 'i_L(\infty) = 2 mA'
        slope = None if ac or moment else _derivative_at_0(texts[0])    # 'dv_C/dt(0^+) = -1000'
        if moment or slope:
            rhs = parse_side(texts[-1], known=rlc, k_is_name="k" in parts)
            if rhs is not None:
                if ltrans is None:
                    ltrans = _ltransients(net)
                if moment:
                    sign = moment.group(2) or moment.group(3) or moment.group(4)
                    when = "inf" if moment.group(5) else {"+": "0+", "-": "0-"}.get(sign, "0")
                name = moment.group(1) if moment else slope
                res = _check_moment(_canon(name), when if moment else "d0+", rhs, ltrans, net, parts, currents) \
                    if ltrans else None
                if res is not None:
                    ok, text, sure = res
                    verdict(ok, f"'{s.text}'" + text, s, sure and values_ok)
            continue
        magnitude = re.fullmatch(r"\s*(?:\||\\left\s*\[|\[)\s*([A-Za-z][^\[\]|]*?)\s*(?:\||\\right\s*\]|\])\s*", texts[0])
        if magnitude and "|" not in texts[0] and not re.fullmatch(r"\s*[-\d.\s]+(?:\\cdot\s*10\s*\^\s*\{?\s*-?\d+\s*\}?)?\s*",
                                                                   texts[-1]):
            magnitude = None                               # '[R] = \Omega' is about units; '[Z] = 1.41': |Z| misread
        if magnitude:
            texts = [magnitude.group(1)] + list(texts[1:])
        qpoint = None                                      # 'Q = (4.98 V, 1.23 mA)': a transistor's operating point
        if len(texts) == 2 and _canon(texts[0]) == "qpoint" and "\ue003" in texts[1]:
            qpoint = _q_point(texts[1])
            if qpoint is None:
                continue
            texts = ["Q_{point}", "0"]
        in_db = re.fullmatch(r"\s*(?:\\mathrm\s*\{\s*)?dbgain\s*\}?\s*(.+?)\s*", texts[0])   # 'A_v = 20 dB'
        if in_db:
            texts = [in_db.group(1)] + list(texts[1:])
        texts = [re.sub(r"\\left\s*\(\s*([st])\s*\\right\s*\)", r"(\1)", texts[0])] + list(texts[1:])   # 'H \left( s \right)'
        timed = re.fullmatch(r"\s*([A-Za-z](?:\s*_\s*\{?\s*\w+(?:\s+\w+)*\s*\}?)?)\s*\(\s*t\s*\)\s*", texts[0])   # 'v_A(t) = ...'
        sdom = re.fullmatch(r"\s*([A-Za-z](?:\s*_\s*\{?\s*\w+(?:\s+\w+)*\s*\}?)?)\s*\(\s*s\s*\)\s*", texts[0])   # 'V_C(s)'
        if timed or sdom:
            texts = [(timed or sdom).group(1)] + list(texts[1:])
        sides = [parse_side(t, ac, timed=bool(timed), known=rlc | ({"s"} if sdom else set()), k_is_name="k" in parts)
                 for t in texts]
        if any(x is None for x in sides):
            continue
        stmt_names = {_canon(x) for side in sides for x in side.free_symbols} | {_canon(texts[0])}
        if len(sides) > 2:                                 # a chain 'V_A = ... = ...': its last two sides
            sides = sides[-2:]
        lhs, rhs = sides
        syms = {_canon(x): x for x in (lhs - rhs).free_symbols}
        if len(syms) >= 2 and all(n in currents for n in syms):                 # Kirchhoff's current law
            e = (lhs - rhs).subs({x: sp.Symbol(currents[n]) for n, x in syms.items()})
            names = {str(v) for v in e.free_symbols}
            at_node = False
            for node, (eq, k) in kcl.items():
                if {str(v) for v in k.free_symbols} == names:
                    ok = sp.simplify(e - k) == 0 or sp.simplify(e + k) == 0
                    verdict(ok, f"'{s.text}': Kirchhoff's current law at node {node}" + ("" if ok else f" gives {eq}"), s)
                    at_node = True
                    break
            if at_node:
                continue
            # no node has just these currents: a loop's equation in them ('-12 + 2k I_1 + 4k I_2 = 0') - checked below
            # against the circuit's solution like any other (it got no verdict at all); not holding, it is no sure
            # 'wrong': the student's I_1, I_2 may be mesh currents of his own, named like the drawing's arrows
            loop_in_currents = True
        else:
            loop_in_currents = False
        if sdom and ac:                                    # an AC-fed circuit: its H(s), Z(s) - the source a unit input
            res = _check_s_function(_canon(lhs), rhs, net, parts) if lhs.is_Symbol else None
            if res is not None:
                ok, text, sure = res
                verdict(ok, f"'{s.text}'" + text, s, sure and values_ok)
            continue
        if (timed or sdom) and not ac and lhs.is_Symbol:   # after a switching: x(t), or its transform X(s)
            res = None
            if timed:
                if trans is None:
                    trans = _transients(net)               # first order: exact, in the parts' names
                res = _check_transient(_canon(lhs), rhs, trans, net, parts, currents) if trans else None
            if res is None and (sdom or not trans):
                if ltrans is None:
                    ltrans = _ltransients(net)             # any order: the Laplace domain
                if ltrans:
                    res = (_check_laplace_s if sdom else _check_laplace_time)(_canon(lhs), rhs, ltrans, net, parts, currents)
            if res is None and sdom:                       # H(s), Z_in(s): the circuit's own functions of s
                res = _check_s_function(_canon(lhs), rhs, net, parts)
            if res is not None:
                ok, text, sure = res
                verdict(ok, f"'{s.text}'" + text, s, sure and values_ok)
            continue
        if timed and not ac:
            continue
        if timed and mixed and lhs.is_Symbol:              # DC and AC sources (or several frequencies): superposed
            if sup is None:
                sup = [(st, net.superposed(st)) for st in states]
                sup = [(st, f) for st, f in sup if f]
            if not sup or not all(str(x) == "t" or _canon(x) in parts for x in rhs.free_symbols):
                continue
            mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) != "t"})
            got = []
            for st, f in sup:
                waves = {n: Netlist.wave(v) for n, v in f.items()}
                for e, either_sign in _pick(_canon(lhs), {"values": waves}, net, currents):
                    if _decidable(mine, [e]):
                        got.append((st, f, e, either_sign))
            if not got:
                continue
            ws = [ac_phasor(c.value)[1] for c in net.components if c.ac and ac_phasor(c.value)]
            T = 2 * 3.141592653589793 / float(min(ws)) if ws else 1.0
            ts = [T * m / 7.3 for m in range(15)]
            tt = sp.Symbol("t")
            hit = []
            for st, f, e, either_sign in got:
                try:
                    fm, fe = sp.lambdify(tt, mine, "math"), sp.lambdify(tt, e, "math")
                    a_, b_ = [float(fm(x)) for x in ts], [float(fe(x)) for x in ts]
                except (TypeError, ValueError, ZeroDivisionError):
                    continue
                tol = REL_TOL * max([abs(v) for v in a_ + b_] + [1e-12])
                if any(all(abs(x - sg * y) <= tol for x, y in zip(a_, b_)) for sg in ((1, -1) if either_sign else (1,))):
                    hit.append(st)
            shown = "; ".join(dict.fromkeys(_show_wave(e) + (f" ({st_text(st)})" if st else "") for st, _, e, _ in got[:3]))
            verdict(bool(hit), f"'{s.text}'" + (": as the circuit gives (each source alone, added up)" if hit else
                                                f": the circuit gives {shown}"), s, len({bare_st(st) for st, *_ in got}) == 1)
            continue
        if sols is None:                                   # the circuit solved once per switch state
            sols = []
            diodes = not ac and any(c.type in ("diode", "zener") for c in net.components)
            bjts = any(c.type in ("npn", "pnp", "nmos", "pmos") for c in net.components)   # an amplifier's source is AC
            for st in states:
                if bjts:                                   # each transistor in the region that fits
                    r, regions = net.solve_transistors(st)
                    if r:
                        sol = {_canon(k): v for k, v in r.items()}
                        sol.update({k_: v_ for k_, v_ in _aliases(net, st, sol).items() if k_ not in sol})   # a node's own name first
                        for k, v in _base_thevenin(net, st).items():
                            sol.setdefault(k, v)
                        for k in ("ib", "ic", "ie", "vce", "vbe", "vcb", "id", "vds", "vgs", "vov", "vec", "veb", "vsd", "vsg"):
                            if k in sol:                   # the operating point Q: I_CQ, V_CEQ, I_DQ ... (one transistor)
                                sol.setdefault(k + "q", sol[k])
                        for k in sol:
                            if re.fullmatch(r"i[bced].*", k):
                                either.add(k)
                        variants = []                      # the amplifier at mid-band, V_T as the course takes it
                        has_bjt = any(c.type in ("npn", "pnp") for c in net.components)
                        for vt, name in ((0.025, "25 mV"), (0.026, "26 mV")) if has_bjt else ((0.025, None),):
                            ss = net.small_signal(vt, st)          # MOSFETs alone: no V_T in it
                            if ss:
                                variants.append((name, dict(_small_signal_names(net, ss), **({"vt": sp.nsimplify(vt)} if name else {}))))
                        for name, extra in variants or [(None, {})]:
                            v_sol = dict(sol)
                            for k, v in extra.items():
                                v_sol.setdefault(k, v)
                            sols.append((dict(st, **regions, **({"VT": name} if name else {})), v_sol, None))
                    continue
                if diodes:                                 # each diode model a problem may mean: ideal, 0.7 V drop
                    for vd, model in ((0.0, "ideal"), (0.7, "0.7 V")):
                        r, dstate = net.solve_diodes(st, vd)
                        if r:
                            sol = {_canon(k): v for k, v in r.items()}
                            sol.update({k_: v_ for k_, v_ in _aliases(net, st, sol).items() if k_ not in sol})   # a node's own name first
                            for n_, v_ in _diode_parts(net, st, dstate, vd).items():   # P_R1, I of one loop ...
                                sol.setdefault(n_, v_)
                                either.add(n_)
                            sols.append((dict(st, diodes=model), sol, None))
                    continue
                r, quant = _solve_parts(net, st, "phasor" if ac else "dc")
                if r:
                    sol = {_canon(k): v for k, v in r.items()}
                    sol.update({k_: v_ for k_, v_ in _aliases(net, st, sol).items() if k_ not in sol})   # a node's own name first
                    sol.update(_resonance(net))
                    for k, v in _second_order(net, st).items():
                        sol.setdefault(k, v)
                    tau_ = net.time_constant(st)
                    if tau_ is not None and tau_ != 0:     # one C or L: a first-order filter's corner, 1 / tau
                        for k in ("omegac", "omega3db", "omegab"):
                            sol.setdefault(k, sp.simplify(1 / tau_))
                        for k in ("fc", "f3db", "fb"):
                            sol.setdefault(k, sp.simplify(1 / (2 * sp.pi * tau_)))
                    if ac:
                        for k, v in _phasor_z(net, st).items():
                            sol.setdefault(k, v)
                        if w is not None:                  # the sources' angular frequency, its frequency
                            sol.setdefault("omega", sp.Float(w) if not isinstance(w, sp.Basic) else w)
                            sol.setdefault("f", sol["omega"] / (2 * sp.pi))
                    if len(net.terminals) == 4:            # a two-port: its parameters (jw, s with C / L, else plain)
                        sol.update(_two_port(net, st, "phasor" if ac else ("s" if storage else "dc")))
                    tau = net.time_constant(st)
                    if ac:                                 # powers: amplitudes or RMS values, as the course has it -
                        amplitudes = all(re.search(r"\\?(?:cos|sin)", str(c.value or "")) for c in net.components if c.ac)
                        so = _second_order(net, st)        # an RLC: 'Q' may be its quality factor, not the reactive power
                        qf = sp.simplify(so["wn"] / so["bw"]) if so and sp.simplify(so["bw"]) != 0 else None
                        for conv, k in (("peak", 1),) if amplitudes else (("peak", 1), ("rms", 0)):   # '10cos(1000t)
                            v_sol = dict(sol)
                            for n, v in quant.items():
                                part = v[k] if isinstance(v, tuple) else v
                                v_sol.setdefault(n, part)
                                either.add(n)
                            sols.append((dict(st, power=conv) if any(isinstance(v, tuple) for v in quant.values()) else st, v_sol, tau))
                            if qf is not None:
                                sols.append((dict(st, power=conv, Q="as the quality factor"), dict(v_sol, q=qf), tau))
                            if not any(isinstance(v, tuple) for v in quant.values()):
                                break
                    else:
                        for n, v in quant.items():
                            sol.setdefault(n, v)
                            either.add(n)
                        sols.append((st, sol, tau))
            if not ac:
                sols = _what_the_source_sees(net, sols)
                if storage:
                    sols = _what_the_storage_sees(net, sols)
        if not sols:
            continue
        if qpoint is not None:                             # the operating point: |V_CE| (|V_DS|) and |I_C| (|I_D|)
            got = []
            for st, sol, _ in sols:
                vk = next((k for k in ("vce", "vds") if k in sol), None)
                ik = next((k for k in ("ic", "id") if k in sol), None)
                if vk and ik and not sol[vk].free_symbols and not sol[ik].free_symbols:
                    got.append((st, abs(float(sp.N(sol[vk]))), abs(float(sp.N(sol[ik])))))
            if not got:
                continue
            hit = [st for st, v_, i_ in got if _close(abs(qpoint[0]), v_) and _close(abs(qpoint[1]), i_)]
            near = not hit and any(_close(abs(qpoint[0]), v_, ROUNDED) and _close(abs(qpoint[1]), i_, ROUNDED) for _, v_, i_ in got)
            shown = "; ".join(dict.fromkeys(f"({_fmt(v_)} V, {_fmt(i_ * 1000)} mA)" for _, v_, i_ in got[:3]))
            verdict(bool(hit) or near, f"'{s.text}': " + ("as the circuit gives" if hit else f"the circuit gives Q = {shown}"
                                                          + (" (rounded?)" if near else "")), s,
                    values_ok and not near and (len({(round(v_, 9), round(i_, 12)) for _, v_, i_ in got}) == 1 or not hit))
            continue
        if in_db:                                          # a gain in decibels: 20 log10 |A| (a power gain: 10 log10)
            k = _canon(lhs) if lhs.is_Symbol else None
            if k is None or rhs.free_symbols:
                continue
            per = 10 if k in ("ap", "gp") else 20
            got = []
            for st, sol, _ in sols:
                if k in sol and sp.N(sp.Abs(sol[k])).is_number and sp.N(sp.Abs(sol[k])) > 0:
                    got.append((st, per * sp.log(sp.Abs(sp.N(sol[k])), 10)))
            if not got:
                continue
            hit = [st for st, v in got if abs(float(sp.N(rhs)) - float(v)) <= 0.1 + 0.005 * abs(float(v))]
            shown = "; ".join(dict.fromkeys(f"{float(v):.2f} dB" for _, v in got[:3]))
            verdict(bool(hit), f"'{s.text}': " + ("as the circuit gives" if hit else f"the circuit gives {shown}"), s,
                    values_ok and (len({f"{float(v):.2f}" for _, v in got}) == 1 or not hit))
            continue
        if magnitude:                                      # |V_A|, |Z_in|, |I|: the size of the circuit's value
            k = _canon(lhs) if lhs.is_Symbol else None
            if k is None or timed or rhs.free_symbols:
                continue
            got = [(st, sp.Abs(sp.N(sol[k] if k in sol else sol[_canon(currents[k])]))) for st, sol, _ in sols
                   if k in sol or (k in currents and _canon(currents[k]) in sol)]
            got = [(st, v) for st, v in got if v.is_number]
            if not got:
                continue
            hit = [st for st, v in got if _close(sp.Abs(sp.N(rhs)), v)]
            near = not hit and any(_close(sp.Abs(sp.N(rhs)), v, ROUNDED) for _, v in got)
            shown = "; ".join(dict.fromkeys(f"{_show(v)}" + (f" ({st_text(st)})" if st else "") for st, v in got[:3]))
            verdict(bool(hit) or near, f"'{s.text}': " + ("as the circuit gives" if hit else
                                                          f"the circuit gives |{texts[0]}| = {shown}" + (" (rounded?)" if near else "")),
                    s, values_ok and not near and (len({_show(v) for _, v in got}) == 1 or not hit))
            continue
        if timed:                                          # a function of time against the phasor: Re(P e^{jwt})
            k = _canon(lhs)
            got = [(st, sol) for st, sol, _ in sols if (k in sol or (k in currents and _canon(currents[k]) in sol))]
            if not ac or w is None or not got or not all(str(x) == "t" or _canon(x) in parts for x in rhs.free_symbols):
                continue
            mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols if str(x) != "t"})
            got = [(st, sol) for st, sol in got if _decidable(mine, [sol[k] if k in sol else sol[_canon(currents[k])]])]
            if not got:
                continue
            hit = [st for st, sol in got if any(_same_wave(mine, sg * (sol[k] if k in sol else sol[_canon(currents[k])]), w)
                                                for sg in ((1, -1) if k in either and k not in marked else (1,)))]
            st0, sol0 = got[0]
            P = sol0[k] if k in sol0 else sol0[_canon(currents[k])]
            mag, ang = abs(complex(sp.N(P))), __import__("cmath").phase(complex(sp.N(P))) * 180 / 3.141592653589793
            verdict(bool(hit), f"'{s.text}'" + (": as the circuit gives" if hit else
                                                f": the circuit gives {_fmt(mag)} cos({_fmt(float(w))}t {'+' if ang >= 0 else '-'} {_fmt(abs(ang))}°)"),
                    s, len({bare_st(st) for st, _ in got}) == 1)
            continue
        names = {_canon(x): x for x in (lhs - rhs).free_symbols}
        if "tau" in names and (lhs.is_Symbol or rhs.is_Symbol) and len(names) >= 1:      # the time constant
            mine = rhs if lhs.is_Symbol and _canon(lhs) == "tau" else lhs
            if not all(_canon(x) in parts for x in mine.free_symbols):
                continue
            mine = mine.subs({x: parts[_canon(x)] for x in mine.free_symbols})
            taus = [(st, t) for st, _, t in sols if t is not None and _decidable(mine, [t])]
            if not taus:
                continue
            hit = [st for st, t in taus if _close(mine, t)]
            plain = {}                                     # one line a switching (the variants beside it aside)
            for st, t in taus:
                st0 = {k: v for k, v in st.items() if k not in ("diodes", "power", "VT", "R_eq", "Q")}
                plain.setdefault((tuple(sorted(st0.items())), str(t)), (st0, t))
            taus = list(plain.values())
            shown = ", ".join(f"{_fmt(_num(t)) if _num(t) is not None else t}" + (f" ({st_text(st)})" if st else "")
                              for st, t in taus)
            verdict(bool(hit), f"'{s.text}': the time constant the circuit gives: {shown}", s,
                    values_ok and len(taus) == 1)
            continue
        if ac and re.fullmatch(r"\s*(?:\|\s*S\s*\||S)\s*", texts[0]) and rhs.is_number and rhs.is_real \
                and any("s" in sol for _, sol, _ in sols):     # the apparent power |S| ('S = 10 VA': a size, not S)
            got = [(st, sp.Abs(sp.N(sol["s"]))) for st, sol, _ in sols if "s" in sol]
            hit = [st for st, v in got if _close(sp.Abs(rhs), v)]
            shown = "; ".join(dict.fromkeys(f"{_show(v)}" + (f" ({st_text(st)})" if st else "") for st, v in got[:3]))
            verdict(bool(hit), f"'{s.text}': " + ("as the circuit gives" if hit else f"the circuit gives |S| = {shown}"), s,
                    values_ok and (len(got) == 1 or len({_show(v) for _, v in got}) == 1 or not hit))
            continue
        ch = _CHARACTER.search(s.text)
        if ch and lhs.is_Symbol and _canon(lhs) in _PF_NAMES and rhs.is_number and ac:
            # 'pf = 0.8 lagging': the power factor and the load's character - lagging when the sources give reactive
            # power (an inductive load), leading when they take it in (capacitive)
            got = []
            for st, sol, _ in sols:
                q = sol.get("qt", sol.get("qtotal"))
                if "pf" in sol and q is not None and _num(q) is not None:
                    got.append((st, sol["pf"], "unity" if abs(_num(q)) < 1e-12 else "lagging" if _num(q) > 0 else "leading"))
            if got:
                said = "lagging" if ch.group(1).lower().startswith(("lag", "ind")) else "leading"
                hit = [st for st, pf, kind in got if _close(rhs, pf) and kind in (said, "unity")]
                shown = "; ".join(dict.fromkeys(f"{_show(pf)} {kind}" for _, pf, kind in got))
                verdict(bool(hit), f"'{s.text}': " + ("as the circuit gives" if hit else f"the circuit gives {shown}"), s,
                        values_ok and len({bare_st(st) for st, _, _ in got}) == 1)
                continue
        if lhs.is_Symbol and _canon(lhs) in _MOMENT_FREE:           # alpha, w_0, zeta, a corner: no moment to them
            k = _canon(lhs)
            if not all(_canon(x) in parts for x in rhs.free_symbols):
                continue
            mine = rhs.subs({x: parts[_canon(x)] for x in rhs.free_symbols})
            got = [(st, sol[k]) for st, sol, _ in sols if k in sol and _decidable(mine, [sol[k]])]
            if not got:
                continue
            hit = [st for st, v in got if _close(mine, v)]
            plain = [({x: y for x, y in st.items() if x not in ("diodes", "power", "VT", "R_eq")}, v) for st, v in got]
            shown = ", ".join(dict.fromkeys(f"{_show(v)}" + (f" ({st_text(st)})" if st else "") for st, v in plain[:3]))
            near = any(_close(mine, v, ROUNDED) for _, v in got)
            book = hit or k not in _RESONANT or all(_pure_resonance(net, st) for st, _ in got)   # w_0 by every definition
            verdict(bool(hit) or near,
                    f"'{s.text}': " + ("as the circuit gives" if hit else
                                       f"the circuit gives {shown}" + (" (rounded?)" if near else
                                                                         "" if book else " - 1/sqrt(LC); not a plain series / parallel RLC, so its resonance by another definition may differ")),
                    s, values_ok and len({bare_st(st) for st, _ in got}) == 1 and bool(hit or not near) and book)
            continue
        if lhs.is_Symbol and _canon(lhs) in ("rl", "rload") and _canon(lhs) not in parts and len(net.ports) == 2 \
                and not rhs.free_symbols:                  # 'R_L = 3k' with no load drawn: the one taking the most power
            seen = [r_ for r_ in (net.resistance_seen(*net.ports[:2], st) for st in states) if r_ is not None and not r_.free_symbols]
            if any(_close(rhs, r_) for r_ in seen):
                verdict(True, f"'{s.text}': the resistance the port's terminals see - the load that takes the most power", s, False)
            continue
        if lhs.is_Symbol and isinstance(parts.get(_canon(lhs)), sp.Symbol) and not rhs.free_symbols:
            # a part whose value the drawing leaves open, worked out ('R_L = 2 k'): the load that takes the most power
            # is the resistance it sees - that is judged; anything else is about the problem's text, not the drawing
            part = next((c for c in net.components if _canon(c.label or c.id) == _canon(lhs)), None)
            seen = []
            if part is not None and part.type == "resistor" and len(part.nodes) == 2:
                rest = Netlist([c for c in net.components if c is not part], [], net.ground, net.node_labels, net.ports, voltages=net.voltages)
                for st in states:
                    r_ = rest.resistance_seen(part.nodes[0], part.nodes[1], st)
                    if r_ is not None and not r_.free_symbols:
                        seen.append(r_)
            if any(_close(rhs, r_) for r_ in seen):
                verdict(True, f"'{s.text}': the resistance {lhs} sees - the load that takes the most power", s, False)
            continue
        # any equation in the parts, the node voltages and the named currents: the circuit's solution must satisfy it
        known = lambda n, sol: n in parts or n in sol or (n in currents and _canon(currents[n]) in sol)   # noqa: E731
        fits, misses = [], []
        # a part's voltage / current / power by its name ('V_{R1}', 'I_s'): its direction is not drawn - any choice of
        # the student's reference directions that makes the equation hold
        flip = [x for n, x in names.items() if n in either and n not in parts and n not in currents and n not in marked][:4]
        import itertools
        rhs_alts = [rhs]                                   # '2\angle -0.927': an angle with no degree mark may be
        if ac and re.search(r"\\angle|∠", texts[-1]) and not re.search(r"\\circ|°|deg", texts[-1]):   # in radians
            r_rad = parse_side(texts[-1], ac=True, known=rlc, k_is_name="k" in parts, radians=True)
            if r_rad is not None and not _close(r_rad, rhs):
                rhs_alts.append(r_rad)
        radians = False
        slack = _digits_slack(s) if lhs.is_Symbol else 0.0       # 'V_A = 3.3 V' for 3.333: rounded to its digits
        for attempt, rhs in enumerate(rhs_alts):
            fits, misses = [], []
            undecided, near = [], []
            for st, sol, _ in sols:
                if not names or not all(known(n, sol) for n in names):
                    continue
                if st.get("Q") and re.search(r"V\s*A\s*R|var|V\s*A(?![A-Za-z])", s.text):
                    continue                                   # 'Q = 8 VAR': the reactive power, by its unit
                sub = {x: parts[n] if n in parts else sol[n] if n in sol else sol[_canon(currents[n])] for n, x in names.items()}
                mine_ = (lhs - rhs).subs({x: parts[n] for n, x in names.items() if n in parts})
                if not _decidable(mine_, [sub[x] for n, x in names.items() if n not in parts]):
                    undecided.append((st, sol))                # it rests on values the drawing does not give
                    continue
                try:
                    ok = rounded = False
                    for signs in itertools.product((1, -1), repeat=len(flip)):
                        sub2 = dict(sub)
                        for x, sg in zip(flip, signs):
                            sub2[x] = sg * sub[x]
                        a_, b_ = lhs.subs(sub2), rhs.subs(sub2)
                        if _close(a_, b_) or (slack and _within_digits(a_, b_, slack)):
                            ok = True
                            break
                        rounded = rounded or _close(a_, b_, ROUNDED)
                    (fits if ok else misses).append((st, sol))
                    if not ok and rounded:
                        near.append((st, sol))
                except Exception:                                              # noqa: BLE001
                    continue
            if fits or attempt == len(rhs_alts) - 1:
                radians = attempt > 0 and bool(fits)
                break
        if not fits and not misses:
            unknown = {n for n in names if not any(known(n, sol) for _, sol, _ in sols)}
            if unknown and not undecided and all(re.fullmatch(r"[ivuj][a-z0-9]{0,3}", n) for n in unknown):   # J_1: a mesh current
                pending.append((s, lhs, rhs, names, unknown))   # the student's own unknowns: mesh currents, node voltages
            if undecided and lhs.is_Symbol:                # for the human: what the circuit gives, in its names
                st, sol = undecided[0]
                k = _canon(lhs)
                k = k if k in sol else (_canon(currents[k]) if k in currents else None)
                if k in sol:
                    out.append(Check(problem, "cannot check", f"'{s.text}': the circuit gives {lhs} = {_show(sol[k])} - "
                                     "its values are not on the drawing", min(conf, s.conf), False, [s.text], line=s.line))
            continue
        one = len({bare_st(st) for st, _ in fits + misses}) == 1         # one state of the switches (variants aside)
        if not fits and lhs.is_Symbol and _canon(lhs) in ("p", "pt", "ptotal", "ptot") and rhs.is_number:
            # 'P = 2 W' that is not what the sources give in all may be one part's power the question asked for
            # ('the power R2 takes'): right then, but not sure
            pk = {"p" + _canon(c.label or c.id): (c.label or c.id) for c in net.components}
            hit = [(st, pk[k], sol[k]) for st, sol in misses for k in pk
                   if k in sol and sol[k].is_number and _close(sp.Abs(rhs), sp.Abs(sol[k]))]
            if hit:
                st0, name, _ = hit[0]
                verdict(True, f"'{s.text}': as the power of {name} " + (f"({st_text(st0)}) " if st0 else "") +
                        "- if that part is meant (the sources give " +
                        "; ".join(dict.fromkeys(_show(sol[_canon(lhs)]) for st, sol in misses if _canon(lhs) in sol)) +
                        " in all)", s, False)
                continue
        if fits:
            agreed = not misses                            # every diode model / power convention gives it
            shown_st = [st if not agreed else {k: v for k, v in st.items() if k not in ("diodes", "power", "VT", "R_eq", "Q")}
                        for st, _ in fits]
            when = list(dict.fromkeys(st_text(st) for st in shown_st if st)) + (["in the steady state"] if storage else [])
            verdict(True, f"'{s.text}': as the circuit gives" + (f" ({', '.join(when)})" if when else "") +
                    (" - its angle read in radians" if radians else ""), s,
                    values_ok and one and not storage and agreed and not radians)
        elif near:                                         # within a few percent: rounded on the way, most likely
            st0 = near[0][0]
            verdict(True, f"'{s.text}': close to what the circuit gives (rounded?)" +
                    (f" ({st_text(st0)})" if st0 else ""), s, False)
        elif not storage:                                  # a moment of a capacitor's / inductor's circuit: not judged
            scale, vals = [], []
            if lhs.is_Symbol:                              # what the circuit gives for it, in each state / diode model
                for st, sol in misses[:4]:
                    k = _canon(lhs)
                    k = k if k in sol else (_canon(currents[k]) if k in currents else None)
                    if k in sol:
                        v = sol[k]
                        if k in either and k not in marked and v.is_real is not False and _num(v) is not None and _num(v) < 0:
                            v = -v                             # its direction is the student's choice
                        vals.append((st, f"{_fmt(float(sp.N(v)) * 180 / 3.141592653589793)}°"
                                     if k in _ANGLE_NAMES and _num(v) is not None else _show(v)))
                if len({v for _, v in vals}) == 1:         # the same in every variant: said once
                    scale = [vals[0][1]]
                else:
                    scale = [v + (f" ({st_text(st)})" if st else "") for st, v in vals]
            verdict(False, f"'{s.text}'" + (f": the circuit gives {lhs} = {'; '.join(scale)}" if scale else
                                            ": the circuit does not satisfy it" + (" in any state of its switches" if len(misses) > 1 else ""))
                    + (" - if these are the currents drawn (mesh currents of his own?)" if loop_in_currents else ""),
                    s, values_ok and one and not loop_in_currents)
    if pending and sols:                           # mesh / nodal analysis in the student's own unknowns
        out += _check_systems(pending, sols, parts, conf, problem, trusted and values_ok and not storage)
    return out


def _check_systems(pending, sols, parts: dict, conf: float, problem, can_be_sure: bool) -> list:
    """equations in unknowns the drawing does not name - mesh currents I_1, I_2 (circular arrows the reader does not
    take), node voltages V_1, V_2 of nodes it does not label: solved together, as many equations as unknowns; right
    when every unknown comes out as one of the circuit's currents / voltages (a branch's current, either way round:
    a mesh current is an outer branch's), wrong (a suggestion) when none does - an inner mesh's current is no
    branch's, so some unmatched is not judged"""
    import itertools
    groups = []
    for item in pending:                                   # equations sharing unknowns form one system
        hit = [g for g in groups if g[1] & item[4]]
        merged = ([item], set(item[4]))
        for g in hit:
            merged[0].extend(g[0])
            merged[1].update(g[1])
            groups.remove(g)
        groups.append(merged)
    out = []
    for items, unknown in groups:
        items = sorted(items, key=lambda it: next(i for i, p in enumerate(pending) if p is it))   # as written
        if len(items) < len(unknown) or len(unknown) > 6:
            continue
        if len(items) == 1 and items[0][1].is_Symbol and {_canon(items[0][1])} == unknown:
            continue                                       # 'V_TH = ...': a name for a step's value, not a system
        verdicts = []
        for st, sol, _ in sols:
            eqs, syms = [], {}
            for s, lhs, rhs, names, _ in items:
                sub = {}
                for n, x in names.items():
                    if n in unknown:
                        syms.setdefault(n, sp.Symbol(f"__u_{n}"))
                        sub[x] = syms[n]
                    elif n in parts:
                        sub[x] = parts[n]
                    elif n in sol:
                        sub[x] = sol[n]
                    else:
                        sub = None
                        break
                if sub is None:
                    break
                eqs.append(sp.expand((lhs - rhs).subs(sub)))
            else:
                try:
                    res = sp.solve(eqs, list(syms.values()), dict=True)
                except Exception:                                      # noqa: BLE001
                    continue
                if len(res) != 1 or set(res[0]) != set(syms.values()):
                    continue
                got = {n: res[0][x] for n, x in syms.items()}
                matched = {}
                for n, v in got.items():
                    pool = [sol[k] for k in sol if k[:1] == ("i" if n[0] in "ij" else "v")]   # J_1: a current too
                    if any(not set(sp.sympify(c).free_symbols) <= set(sp.sympify(v).free_symbols) for c in pool):
                        matched = None                         # the circuit's values are names the student's are not
                        break
                    matched[n] = any(_close(v, c) or _close(-v, c) for c in pool)
                if matched:
                    verdicts.append((st, got, matched))
        if not verdicts:
            continue
        st, got, matched = next((v for v in verdicts if all(v[2].values())), verdicts[0])
        shown = ", ".join(f"{n.upper()[0]}_{n[1:]} = {_show(v)}" for n, v in sorted(got.items()))
        texts = [it[0].text for it in items]
        p_ = min([conf] + [it[0].conf for it in items])
        last = max(it[0].line for it in items)
        if all(matched.values()):
            out.append(Check(problem, "right", f"{'; '.join(texts)}: their unknowns ({shown}) are the circuit's own "
                             "currents / voltages", p_, can_be_sure and p_ >= SURE, texts, line=last))
        elif not any(any(m.values()) for _, _, m in verdicts):
            out.append(Check(problem, "wrong", f"{'; '.join(texts)}: their unknowns ({shown}) are none of the circuit's "
                             "currents / voltages", p_, False, texts, line=last))
    return out


def st_text(st: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in st.items())
