"""Numeric final-answer checking with units and tolerance.

Returns a structured verdict so the rubric layer can give partial credit, e.g.
"right number, wrong unit prefix" is a different outcome from "wrong number".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pint

from .normalize import normalize_numeric_text

_ureg = pint.UnitRegistry()
_ureg.define("percent = 0.01 = %")
try:                                       # reactive volt-amperes ('Q = 600 var', '0.6 kVAR'): pint knows VA only - a
    _ureg.Quantity(1, "var")               # rubric's key in var was a bad reference (its question could not be graded)
except Exception:  # noqa: BLE001
    _ureg.define("volt_ampere_reactive = volt * ampere = var = VAR = VAr")
Q_ = _ureg.Quantity

_NUM_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$")


_UNIT_WORDS = {"volt": "V", "amp": "A", "ampere": "A", "watt": "W", "ohm": "ohm"}   # '24 Volt', '3 Amps'


class ParseError(ValueError):
    pass


def _respell(unit: str) -> str:
    """a unit as students miswrite its case: '3 ma' (mA - pint reads milli-years), '50 hz', '40 db', '4.7 Kohm' (a capital
    K for kilo), '20 msec' -> its spelling ('mA', 'Hz', 'dB', 'kohm', 'ms'); the unit itself when none of these"""
    m = re.fullmatch(r"([kKmunpMG\u00b5\u03bcu]?)(a|amps?|hz|db|ohms?|sec|v|w)", unit or "")
    if not m:
        return unit
    pre = {"K": "k"}.get(m.group(1), m.group(1))
    base = {"a": "A", "amp": "A", "amps": "A", "hz": "Hz", "db": "dB", "ohm": "ohm", "ohms": "ohm", "sec": "s", "v": "V",
            "w": "W"}[m.group(2)]
    return pre + base


@dataclass
class NumericAnswer:
    value: float
    unit: str  # '' when dimensionless
    raw: str

    @property
    def quantity(self):
        return Q_(self.value, self.unit or "dimensionless")


_PREFIX_ONLY = {"k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "m": 1e-3, "u": 1e-6, "\u00b5": 1e-6, "\u03bc": 1e-6, "n": 1e-9, "p": 1e-12}


def _israeli(s: str) -> str:
    """how answers are written here: a unit in square brackets ('3.3 [V]', '\\left[ mA \\right]'), '\\%', a degree
    sign ('45^\\circ'), a repeating decimal ('3.\\overline{3}') -> what the parser takes"""
    s = re.sub(r"\\left\s*\[|\\right\s*\]", lambda m: "[" if "[" in m.group(0) else "]", s)
    s = re.sub(r"\[\s*([^\[\]]{1,12}?)\s*\]\s*$", r" \1", s)
    s = s.replace("\\%", "%")
    s = re.sub(r"\s*(?:\^\s*\{?\s*\\circ\s*\}?|\u00b0)", " deg", s)
    from .selfcheck import _REPEATING, repeating
    return _REPEATING.sub(repeating, s)


def _numeric_expression(s: str):
    """'10\\sqrt{2} V', '\\frac{10}{3} V' -> (value, unit): the number worked out, the unit after it; None if not"""
    m = re.match(r"^(.*?)\s*((?:[kmunpMGµμ]|u)?(?:V|A|W|ohm|Hz|F|H|s|J|C|S|VA|VAR|var|rad/s|deg|%)|)\s*$", s)
    if not m or not m.group(1).strip():
        return None
    from .selfcheck import _parse
    e = _parse(m.group(1).replace("*", " \\cdot "))
    if e is None or e.free_symbols:
        return None
    try:
        return float(e), m.group(2)
    except (TypeError, ValueError):
        return None


def parse_numeric(text: str) -> NumericAnswer:
    """Parse '4.7 kohm', '3,3 mA', '1.2e3', '10^{-3} F', 'f_c = 159 Hz' ..."""
    s = normalize_numeric_text(_israeli(text))
    s = s.replace("\\approx", "=").replace("\u2248", "=")          # 'V \approx 5 V' is a final answer too
    if re.search(r"[<>]|\\(?:le|ge|leq|geq|neq|ne)\b", s):
        raise ParseError(f"not a final answer (relation other than '='): {text!r}")
    # allow a leading 'name =' (e.g. 'f_c = 159 Hz' or 'R_{th} = 4.7 kohm'); the value is the last side
    # that is a number ('5 V =' and '5 V = V' come from right-to-left lines such as 'המתח = 5V')
    sides = [x.strip().lstrip("≈~ ") for x in s.split("=")]       # also tolerate a leading '≈' / '~'
    s = next((x for x in reversed(sides) if _NUM_RE.match(x)), sides[-1])
    m = _NUM_RE.match(s)
    compound = _pint_unit(m.group(2)) if m and m.group(2).strip() else None     # 'm/s^2', '\frac{kg}{m^3}'
    if not m or (m.group(2).strip() and compound is None and re.search(r"\\(?:sqrt|frac|pi)|[/*^]\s*\d", m.group(2))):
        ex = _numeric_expression(s)                    # '\frac{10}{3} V', '10\sqrt{2} V': the number worked out
        if ex is None:
            raise ParseError(f"not a number: {text!r}")
        return NumericAnswer(value=ex[0], unit=ex[1], raw=text)
    value = float(m.group(1))
    unit = compound if compound is not None and re.search(r"[/*]", compound) else m.group(2).strip().replace(" ", "")
    if unit in _PREFIX_ONLY and not _is_unit(unit):    # '5k': five thousand of the unit the question means
        return NumericAnswer(value=value * _PREFIX_ONLY[unit], unit="", raw=text)
    if unit in _PREFIX_ONLY:                           # 'k' (Boltzmann's), 'm' (metres): decided against the question
        return NumericAnswer(value=value, unit=unit, raw=text)
    if unit:
        try:
            Q_(1, unit)
        except Exception as e:  # noqa: BLE001
            fixed = _UNIT_WORDS.get(unit.lower().rstrip("s"), unit) if len(unit) > 2 else unit
            fixed = re.sub(r"^([kmunpMG\u00b5\u03bc]?)([vw])$", lambda m: m.group(1) + m.group(2).upper(), fixed)
            #   '5 v', '3 mv', '45 w', '2 kw': volts / watts written lowercase
            if fixed == unit:                          # '50 hz', '4.7 Kohm' (a capital K)
                fixed = _respell(unit)
            if fixed == unit:
                raise ParseError(f"unknown unit {unit!r} in {text!r}: {e}") from e
            unit = fixed
    return NumericAnswer(value=value, unit=unit, raw=text)


def _pint_unit(rest: str):
    """a compound unit as students write it -> pint's spelling: '\\frac{m}{s^2}' -> '(m)/(s**2)', 'g/cm^3' -> 'g/cm**3',
    'N \\cdot m' -> 'N*m', '\\frac{\\mu C}{m^2}'; None: not a unit (a number, an expression)"""
    u = re.sub(r"\\(?:mathrm|text|rm)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", rest.strip())
    u = re.sub(r"\\left\s*\[|\\right\s*\]|^\[|\]$", "", u).strip()
    for _ in range(2):
        u = re.sub(r"\\[dt]?frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"(\1)/(\2)", u)
    u = re.sub(r"\^\s*\{\s*(-?\d)\s*\}|\^\s*(-?\d)", lambda m: "**" + (m.group(1) or m.group(2)), u)
    u = re.sub(r"\\(?:cdot|times)|[·⋅]", "*", u).replace("\\Omega", "ohm").replace("Ω", "ohm")
    u = re.sub(r"\\mu\s*|[µμ]", "u", u)
    u = re.sub(r"\s*([*/()])\s*", r"\1", u).strip()
    u = re.sub(r"^\(1\)/", "1/", u)                 # ('\frac{1}{cm^3}': per cubic centimetre)
    if not u or not re.search(r"[A-Za-z]", u) or re.search(r"\\|[{}]", u) or \
            re.search(r"(?<!\*\*)(?<!\*\*-)\d", re.sub(r"^1/", "/", u)):
        return None
    try:
        Q_(1, u)
    except Exception:  # noqa: BLE001
        return None
    return u


def unit_given(student: str, reference: str) -> bool:
    """did the student write the answer's unit? ('5k' for 5 kΩ: only a prefix - no; '5 m' for 5 mA: no; '5 m' for 5
    metres: yes)"""
    st = parse_numeric(student)
    if not st.unit:
        return False
    if st.unit in _PREFIX_ONLY:
        try:
            return st.quantity.dimensionality == parse_numeric(reference).quantity.dimensionality
        except Exception:  # noqa: BLE001
            return False
    return True


def _is_unit(u: str) -> bool:
    try:
        Q_(1, u)
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class NumericVerdict:
    status: str          # 'correct' | 'prefix_error' | 'factor_2pi' | 'unit_mismatch' | 'wrong' | 'parse_error'
    rel_error: float | None
    detail: str


_COMPLEX_MARK = re.compile(r"\\angle|∠|(?<![A-Za-z\\])j\s*\d|\dj(?![A-Za-z])|(?<![A-Za-z\\])j\s*\\?\(|\d\s*\\?cdot\s*j(?![A-Za-z])")
_C_TAIL = re.compile(r"^(?P<num>.*?)\s*(?:\\left\s*)?(?:\[\s*)?(?P<rm>\\(?:mathrm|text|rm)\s*\{\s*)?"
                     r"(?P<unit>(?:[kmunpMGµμ]|\\mu\s*)?(?:V\s*A\s*R|V\s*A|var|V|A|W|\\Omega|Ω|ohm|S|Hz))?\s*(?(rm)\})\s*"
                     r"(?:\\right\s*)?(?:\]\s*)?$")


def _complex_answer(text: str):
    """a phasor or an impedance as an answer - '2\\angle -53.13^\\circ A', '1.2 - j1.6 A', '(3 + j4) \\Omega',
    'I = 2∠-53.13°' -> (its complex value, its unit in pint's spelling, '' for none); None: no j or angle in it"""
    s = re.split(r"(?<![<>!])=", text)[-1].strip()
    if not _COMPLEX_MARK.search(s):
        return None
    s = re.sub(r"((?:\\angle|∠)\s*\{?\s*[+-]?\s*\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|°)(?![A-Za-z])", r"\1^\\circ", s)   # '10 \angle 30 deg'
    m = _C_TAIL.match(s.replace("∠", "\\angle "))
    num, unit = (m.group("num"), m.group("unit") or "") if m else (s, "")
    from .selfcheck import _cnum, _parse
    e = _parse(num)
    v = _cnum(e) if e is not None and not e.free_symbols else None
    if v is None:
        return None
    unit = re.sub(r"\\mu|μ|µ", "u", re.sub(r"\s+", "", unit).replace("\\Omega", "ohm").replace("Ω", "ohm"))
    unit = re.sub(r"(VAR|var|VA)$", "V*A", unit)                  # volt-amperes (reactive too): pint's V*A
    return v, unit


def _compare_complex(student: str, reference: str, rel_tol: float):
    """a phasor answer against a phasor key (or one of them a plain number) -> NumericVerdict, or None when neither is
    complex. A plain number for a phasor key: 'magnitude_only' when it is the key's size; the angle's sign flipped:
    'conjugate'; the size right, the angle not: 'angle_wrong' - for a human, all three"""
    rc, sc = _complex_answer(reference), _complex_answer(student)
    if rc is None and sc is None:
        return None
    if rc is None:
        try:
            ref = parse_numeric(reference)
        except ParseError:
            return None
        rc = (complex(ref.value), ref.unit)
    st_real = None
    if sc is None:
        try:
            st_real = parse_numeric(student)
        except ParseError as e:
            return NumericVerdict("parse_error", None, str(e))
        sc = (complex(st_real.value), st_real.unit)
    (r, ru), (v, su) = rc, sc
    if su and ru and su != ru:
        try:
            v = v * Q_(1, su).to(Q_(1, ru).units).magnitude
        except Exception:  # noqa: BLE001
            return NumericVerdict("unit_mismatch", None, f"units are incompatible: student {su!r} vs reference {ru!r}")
    if abs(r.imag) <= 1e-12 * max(1.0, abs(r)) and abs(v.imag) <= 1e-12 * max(1.0, abs(v)):
        if st_real is None:                        # '10\\angle 0^\\circ V' for 10 V: the usual comparison of its value
            return compare_numeric(f"{v.real!r} {ru or su}".strip(), reference, rel_tol)
        return None
    size = abs(r) or 1.0
    rel = abs(v - r) / size
    if rel <= rel_tol:
        return NumericVerdict("correct", rel, f"within {rel_tol:.0%} as a complex number (rel. error {rel:.2%})")
    same_size = abs(abs(v) - abs(r)) <= rel_tol * size
    if st_real is not None and abs(r.imag) > 1e-12 * size:
        if same_size:
            return NumericVerdict("magnitude_only", rel, "the size of the phasor, without its angle")
        return NumericVerdict("wrong", rel, f"the key is the phasor {r:.4g}; written {v.real:.4g}")
    if same_size and abs(v - r.conjugate()) <= rel_tol * size:
        return NumericVerdict("conjugate", rel, "the angle's sign flipped (the conjugate)")
    if same_size:
        return NumericVerdict("angle_wrong", rel, "the size right, the angle not")
    if v != 0 and r != 0:
        ratio = v / r
        for p in (1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 10, 0.1, 100, 0.01):
            if abs(ratio / p - 1) <= rel_tol:
                return NumericVerdict("prefix_error", rel, f"magnitude off by ×{p:g} — likely SI-prefix or decimal-point error")
    return NumericVerdict("wrong", rel, f"rel. error {rel:.1%} exceeds {rel_tol:.0%} as a complex number")


def _within(s_val: float, r_val: float, rel_tol: float) -> bool:
    return abs(s_val - r_val) <= rel_tol * abs(r_val) if r_val else abs(s_val) <= 1e-12


def compare_numeric(student: str, reference: str, rel_tol: float = 0.02) -> NumericVerdict:
    cv = _compare_complex(student, reference, rel_tol)   # phasors, impedances: 'I = 2\angle -53.13^\circ A'
    if cv is not None:
        return cv
    try:
        ref = parse_numeric(reference)
    except ParseError as e:
        raise ValueError(f"bad reference answer: {e}") from e
    try:
        st = parse_numeric(student)
    except ParseError as e:
        return NumericVerdict("parse_error", None, str(e))

    rq, sq = ref.quantity, st.quantity
    if rq.dimensionality != sq.dimensionality and st.unit in _PREFIX_ONLY:
        sq = Q_(st.value * _PREFIX_ONLY[st.unit], (1 * rq.units).to_base_units().units)   # '5 m' for 5 mA, '5k' for 5 kΩ: a prefix
        #                                                                     of the unit meant, not metres / Boltzmann's k
    if rq.dimensionality != sq.dimensionality:
        # Student gave a dimensionless number for a quantity with units (or vice versa):
        # compare magnitudes as if the reference unit was implied.
        if not st.unit:
            sq = Q_(st.value, ref.unit or "dimensionless")
            si = Q_(st.value, (1 * rq.units).to_base_units().units)       # or in SI units: '0.02553' for 25.53 mA
            if _within(si.to(rq.units).magnitude, rq.magnitude, rel_tol) and not _within(sq.magnitude, rq.magnitude, rel_tol):
                sq = si
        else:
            again = _respell(st.unit)                  # '3 ma' for 3 mA (pint read milli-years), '40 db' for 40 dB
            try:
                sq2 = Q_(st.value, again) if again != st.unit else None
            except Exception:  # noqa: BLE001
                sq2 = None
            if sq2 is None or sq2.dimensionality != rq.dimensionality:
                return NumericVerdict(
                    "unit_mismatch", None,
                    f"units are incompatible: student {st.unit!r} vs reference {ref.unit!r}")
            sq = sq2

    s_val = sq.to(rq.units).magnitude
    r_val = rq.magnitude
    if r_val == 0:
        rel = abs(s_val - r_val)
    else:
        rel = abs(s_val - r_val) / abs(r_val)

    if rel <= rel_tol:
        return NumericVerdict("correct", rel, f"within {rel_tol:.0%} (rel. error {rel:.2%})")

    # Off by a power of 1000 (or 10) -> almost always an SI-prefix slip (k/m/u) or an OCR k↔m confusion.
    if s_val != 0 and r_val != 0:
        ratio = abs(s_val / r_val)
        # ECE classic: angular vs. cyclic frequency (ω = 2πf), or 2π dropped from a formula.
        for p in (2 * math.pi, 1 / (2 * math.pi)):
            if abs(ratio / p - 1) <= rel_tol:
                return NumericVerdict(
                    "factor_2pi", rel,
                    "off by a factor of 2π — rad/s vs Hz, or 2π dropped")
        for p in (1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 10, 0.1, 100, 0.01):
            if abs(ratio / p - 1) <= rel_tol:
                return NumericVerdict(
                    "prefix_error", rel,
                    f"magnitude off by ×{p:g} — likely SI-prefix or decimal-point error")
    return NumericVerdict("wrong", rel, f"rel. error {rel:.1%} exceeds {rel_tol:.0%}")


def is_finite(x: float) -> bool:
    return math.isfinite(x)
