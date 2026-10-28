"""Boolean algebra as students write it in a digital-systems exam: 'F = A\\bar{B} + \\bar{A}B = A \\oplus B', 'A + AB = A',
'\\overline{A + B} = \\bar{A} \\bar{B}', 'F = (A + B)(A + C)' then 'F = A + BC' - each side read into a logic expression;
two sides are equal when their truth tables are (every row, up to 10 letters).

Expressions are tuples: ('var', name), ('const', 0 | 1), ('not', e), ('and', [e...]), ('or', [e...]), ('xor', [e...]),
('xnor', [a, b])."""
import itertools
import re

# a line in Boolean algebra: a complement or a logic operator in it ('A + AB = A' alone is no sure sign - an equation in
# algebra too; a page's other lines may tell, check_statements decides)
MARK = re.compile(r"\\(?:bar|overline|oplus|odot|neg|lnot|vee|wedge|land|lor)(?![A-Za-z])|[⊕¬∨∧]")
_PRIMED = re.compile(r"(?:[A-Za-z](?:_\{?\w+\}?|\d{1,2})?|\))\s*'")


def primed(tex: str) -> bool:
    """complements written as primes inside a side ("A' + B' = (AB)'", "F = XY + X'Z"): a sign of Boolean algebra - not a
    lone first side ("V' = V + IR": a name), nor a derivative ("f'(x)"), nor with digits other than 0 and 1"""
    sides = [x for x in re.split(r"(?<![<>!])=", tex)]
    if len(sides) < 2 or re.search(r"[2-9]|\\frac|\\sqrt|\\int|/", tex) or re.search(r"[A-Za-z]\s*'\s*\(", tex):
        return False
    sides = [re.sub(r"\\[A-Za-z]+", " ", x) for x in sides]       # (command names no letters: '\\prime')
    first, rest = sides[0].strip(), sides[1:]
    if any(_PRIMED.search(x) for x in rest):
        return True
    return bool(_PRIMED.search(first)) and not re.fullmatch(r"[A-Za-z](?:_\{?\w+\}?)?\s*'", first)


_NAME = re.compile(r"\s*([A-Za-z])(?:\s*_\s*\{?\s*[A-Za-z0-9]+\s*\}?|\d{1,2}(?![\d.]))?\s*(?:\(\s*[A-Za-z](?:\s*,\s*[A-Za-z])*\s*\))?\s*")
_OR = (r"\vee", r"\lor", "∨", "+", "|")
_AND = (r"\cdot", r"\wedge", r"\land", "∧", "·", "*", "&", r"\times")
_XOR = (r"\oplus", "⊕")
_XNOR = (r"\odot", "⊙")
_NOT = (r"\neg", r"\lnot", "¬", "~", "!")


class _Parser:
    def __init__(self, s: str):
        self.s, self.i, self.ops = s, 0, 0

    def skip(self):
        while True:
            m = re.match(r"\s+|\\[,;:! ]|\\left(?![A-Za-z])|\\right(?![A-Za-z])|\\big[lr]?(?![A-Za-z])|\\Big[lr]?(?![A-Za-z])", self.s[self.i:])
            if not m or not m.group(0):
                return
            self.i += len(m.group(0))

    def peek(self, toks) -> str | None:
        self.skip()
        for t in toks:
            if self.s.startswith(t, self.i) and not (t[0] == "\\" and self.s[self.i + len(t):self.i + len(t) + 1].isalpha()):
                return t
        return None

    def eat(self, toks) -> bool:
        t = self.peek(toks)
        if t is None:
            return False
        self.i += len(t)
        return True

    def group(self) -> str:
        """the content of a braced group at i (after its '{')"""
        self.skip()
        if not self.s.startswith("{", self.i):
            raise ValueError("no group")
        depth, j = 0, self.i
        while j < len(self.s):
            if self.s[j] == "{":
                depth += 1
            elif self.s[j] == "}":
                depth -= 1
                if depth == 0:
                    inner = self.s[self.i + 1:j]
                    self.i = j + 1
                    return inner
            j += 1
        raise ValueError("unclosed group")

    def or_(self):
        terms = [self.xor_()]
        while self.eat(_OR):
            self.ops += 1
            terms.append(self.xor_())
        return terms[0] if len(terms) == 1 else ("or", terms)

    def xor_(self):
        terms = [self.and_()]
        while True:
            if self.eat(_XOR):
                self.ops += 1
                terms.append(self.and_())
            elif self.eat(_XNOR):
                self.ops += 1
                a = terms.pop() if len(terms) == 1 else ("xor", terms)
                terms = [("xnor", [a, self.and_()])]
            else:
                break
        return terms[0] if len(terms) == 1 else ("xor", terms)

    def starts_atom(self) -> bool:
        self.skip()
        if self.i >= len(self.s):
            return False
        c = self.s[self.i]
        if c.isalpha() or c in "(01[" or c in "¬~!":
            return True
        return bool(re.match(r"\\(?:bar|overline|neg|lnot)(?![A-Za-z])", self.s[self.i:]))

    def and_(self):
        terms = [self.unary()]
        while True:
            if self.eat(_AND):
                self.ops += 1
                terms.append(self.unary())
            elif self.starts_atom():
                self.ops += 1
                terms.append(self.unary())
            else:
                break
        return terms[0] if len(terms) == 1 else ("and", terms)

    def unary(self):
        if self.eat(_NOT):
            return ("not", self.unary())
        e = self.atom()
        while True:
            self.skip()
            if self.s.startswith("'", self.i):
                self.i += 1
                e = ("not", e)
            elif re.match(r"\^\s*\{?\s*\\prime\s*\}?", self.s[self.i:]):
                self.i += len(re.match(r"\^\s*\{?\s*\\prime\s*\}?", self.s[self.i:]).group(0))
                e = ("not", e)
            else:
                return e

    def atom(self):
        self.skip()
        s, i = self.s, self.i
        if i >= len(s):
            raise ValueError("end")
        m = re.match(r"\\(bar|overline)(?![A-Za-z])", s[i:])
        if m:
            self.i += len(m.group(0))
            self.skip()
            if self.s.startswith("{", self.i):
                inner = _Parser(self.group())
                e = inner.or_()
                inner.skip()
                if inner.i != len(inner.s):
                    raise ValueError("trailing")
                self.ops += inner.ops
                return ("not", e)
            return ("not", self.atom())
        if s[i] in "([":
            close = ")" if s[i] == "(" else "]"
            self.i += 1
            e = self.or_()
            self.skip()
            if not self.s.startswith(close, self.i):
                raise ValueError("unclosed")
            self.i += 1
            return e
        if s[i] in "01" and not (i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] == ".")):
            self.i += 1
            return ("const", int(s[i]))
        if s[i].isalpha():
            # (a letter with its index typed after it - 'Y1', 'A0 + A1', a circuit's nodes and a multiplexer's inputs: Y_1,
            # not Y and 1 - 'Y1 = 0' was a false sure wrong step, 'not the same for Y=1')
            m = re.match(r"([A-Za-z])(?:\s*_\s*(?:\{\s*([A-Za-z0-9]+)\s*\}|([A-Za-z0-9]))|(\d{1,2})(?![\d.]))?", s[i:])
            self.i += len(m.group(0))
            sub = m.group(2) or m.group(3) or m.group(4)
            return ("var", m.group(1) + (f"_{sub}" if sub else ""))
        raise ValueError(f"unexpected {s[i]!r}")


def parse(tex: str):
    """a side in Boolean algebra -> (expression, its binary operators' count); None when it is no such side"""
    try:
        p = _Parser(tex)
        e = p.or_()
        p.skip()
        if p.i != len(p.s):
            return None
        return e, p.ops
    except (ValueError, IndexError, AttributeError, RecursionError):
        return None


def variables(e) -> set:
    if e[0] == "var":
        return {e[1]}
    if e[0] == "const":
        return set()
    if e[0] == "not":
        return variables(e[1])
    return set().union(*(variables(x) for x in e[1]))


def value(e, row: dict) -> int:
    k = e[0]
    if k == "var":
        return row[e[1]]
    if k == "const":
        return e[1]
    if k == "not":
        return 1 - value(e[1], row)
    vals = [value(x, row) for x in e[1]]
    if k == "and":
        return int(all(vals))
    if k == "or":
        return int(any(vals))
    if k == "xor":
        return sum(vals) % 2
    return int(vals[0] == vals[1])                   # xnor


def substitute(e, defs: dict, depth: int = 0):
    """the letters defined as expressions put in ('X + C' with X = A + B -> A + B + C), definitions inside them too (a
    few deep - a definition through itself goes no further)"""
    k = e[0]
    if k == "var":
        return substitute(defs[e[1]], defs, depth + 1) if e[1] in defs and depth < 4 else e
    if k == "const":
        return e
    if k == "not":
        return ("not", substitute(e[1], defs, depth))
    return (k, [substitute(x, defs, depth) for x in e[1]])


def equivalent(a, b):
    """-> (True, None), or (False, a row where they differ: {letter: 0/1}); None when there are too many letters"""
    names = sorted(variables(a) | variables(b))
    if len(names) > 10:
        return None
    for bits in itertools.product((0, 1), repeat=len(names)):
        row = dict(zip(names, bits))
        if value(a, row) != value(b, row):
            return False, row
    return True, None


def name_side(tex: str):
    """'F', 'F(A,B,C)', 'f(x, y)', 'F_1': a function's name - the letter, else None"""
    m = _NAME.fullmatch(tex)
    return m.group(1) if m else None


def pretty(tex: str) -> str:
    """for a message: complements as primes ('\\bar{A}' -> "A'", '\\overline{A + B}' -> "(A+B)'"), operators as signs"""
    t = tex
    for _ in range(4):
        t = re.sub(r"\\overline\s*\{([^{}]*)\}", lambda m: (m.group(1).strip() + "'") if re.fullmatch(r"\s*[A-Za-z](?:_\w+|\d{1,2})?\s*", m.group(1))
                   else "(" + m.group(1).strip() + ")'", t)
        t = re.sub(r"\\bar\s*\{([^{}]*)\}", lambda m: (m.group(1).strip() + "'") if re.fullmatch(r"\s*[A-Za-z](?:_\w+|\d{1,2})?\s*", m.group(1))
                   else "(" + m.group(1).strip() + ")'", t)
    t = re.sub(r"\\bar\s*([A-Za-z])", r"\1'", t)
    for a, b in ((r"\oplus", "⊕"), (r"\odot", "⊙"), (r"\cdot", "·"), (r"\vee", "∨"), (r"\wedge", "∧"), (r"\lor", "∨"), (r"\land", "∧"),
                 (r"\neg", "¬"), (r"\lnot", "¬"), (r"\left", ""), (r"\right", "")):
        t = t.replace(a, b)
    return re.sub(r"\s+", "", t)


def row_text(row: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(row.items()))
