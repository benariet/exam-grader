"""Grade a whole page against a rubric: find each item's answer among the page's segments, grade
it over the readings' probabilities (grader.core), add up, and say what a human has to check.

Rubric (JSON, see data/my_scans/silly_test_rubric.json):

    {"name": "...",
     "items": [
       {"id": "1",  "points": 8, "arithmetic": "1+5"},        # the student writes '1+5 = ...'
       {"id": "Q1", "points": 8, "open": "how many legs does a dog have?"}
     ]}

arithmetic: the exercise identifies its line - the formula segment whose left side reads as the
            exercise (over all n-best readings; an item number in front, like '2)', is allowed).
            The value of the exercise is the reference answer; the student's answer is the right
            side. If no segment carries the exercise with probability >= 0.5, the item goes to a
            human ("not found"); so does a segment the layout was unsure about.
            A factor of 10 / 1000 off earns nothing in arithmetic; "prefix_credit": 0.5 gives half points
            for it (an SI-prefix slip, mA for A, in an answer with units).
value:      a named result, as in a circuits exam: {"id": "a-i", "points": 5, "value": "i", "answer": "-4 A"}.
            The line is found by its left side: a formula segment whose left side reads as the name
            ('i', 'V_1', 'P_{3A}': case, braces, accents and font commands do not matter; '(b)' or a
            circled letter in front is allowed) with probability >= 0.5 over the n-best. A frame the
            student drew around the answer (read_page: .boxed) is preferred: if a boxed segment
            carries the name, only boxed segments count. The value is the last side that is a number
            ('P = -(3 x 15) = -45 W'); a fraction is evaluated ('i = -5/2 A' -> -2.5 A); units are
            checked (grader.numeric), "rel_tol" defaults to 0.02. "page": n (1 = first) looks only on
            that page - two questions may both ask for 'i'. Not found -> a human.
open:       open answers (sentences, proofs) always go to a human for now.
"""
from __future__ import annotations

import ast
import json
import operator
import re
from dataclasses import dataclass, field

from .core import Reading, grade_question
from .mixed import readings as nbest_probs
from .normalize import undecorate
from .rubric import Criterion, Question

FOUND_MIN = 0.5            # probability that a segment holds the exercise, to grade it there
LAYOUT_MIN = 0.8           # the segment's layout confidence (as grader.mixed)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_MARKER = re.compile(r"^(?:\(?\d{1,2}\)|\d{1,2}\.(?=\d))")      # '2)', '(2)', or '1.' read off a circled 1


def safe_eval(expr: str):
    """value of + - * / arithmetic on numbers, else None"""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(n)
    try:
        return float(ev(ast.parse(expr, mode="eval")))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return None


def norm_math(t: str) -> str:
    """'7 \\times 7', '7x7', '7 × 7' -> '7*7';  '70 \\div 7', '70:7', '\\frac{70}{7}' -> '70/7'"""
    t = t.replace("\\times", "*").replace("×", "*").replace("\\cdot", "*").replace("·", "*")
    t = t.replace("\\div", "/").replace("÷", "/")
    t = re.sub(r"\\frac\s*\{\s*([\d\s.]+)\}\s*\{\s*([\d\s.]+)\}", r"\1/\2", t)     # \frac{70}{7} -> 70/7
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = t.replace("−", "-").replace("–", "-")
    t = re.sub(r"\\left|\\right|\\[,;!]|\\q?quad|\s", "", t)
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"(?<=\d)[xX](?=\d)", "*", t)
    t = re.sub(r"(?<=\d):(?=\d)", "/", t)
    return t


def lhs_rhs(text: str):
    """-> ({left side, left side without an item number in front}, right side) at the first '='"""
    t = norm_math(re.sub(r"^\s*\\textcircled\s*\{\s*\d+\s*\}", "", undecorate(text).strip()))
    if "=" not in t:
        return set(), None
    lhs, rhs = t.split("=", 1)
    return {lhs, _MARKER.sub("", lhs)}, rhs


@dataclass
class ItemResult:
    id: str
    points: float
    kind: str
    route: str                     # 'auto' | 'review'
    score: float | None = None     # most likely score (auto items; a guess for located review items)
    expected: float | None = None
    confidence: float | None = None
    reason: str = ""
    found: str = ""                # what the tool read there
    readings: list = field(default_factory=list)


_FONT = re.compile(r"\\(?:mathrm|mathbf|mathit|mathtt|mathsf|boldsymbol|bf|rm|it|dot|ddot|hat|bar|tilde|vec|widehat|overline|text)\b")
_LEAD = re.compile(r"^(?:\\textcircled\{[a-z0-9]\}|\([a-z0-9]\)|[a-z0-9]\))")       # '(b)', circled b, 'b)'


def var_key(t: str) -> str:
    """'P_{3A}', '\\mathrm{P}_{3 A}', 'p3a' -> 'p3a';  '\\dot{\\imath}_1', 'i_1' -> 'i1'"""
    t = re.sub(r"^\s*\\textcircled\s*\{[^{}]*\}", "", t)
    t = _FONT.sub("", t).replace("\\imath", "i").replace("\\jmath", "j").replace("ı", "i")
    t = re.sub(r"\\[A-Za-z]+|[{}_^\s\\,;:.*]", "", t).lower()
    return _LEAD.sub("", t)


_ARROW = re.compile(r"\\(?:Longrightarrow|Rightarrow|implies|therefore|quad|qquad)|=>")


_CLAUSE = re.compile(r"(?<!\\)[;,](?=\s*\\?[A-Za-z])")   # "V_3 = 235, V_4 = ..." - not "\," (a thin space)


def named_clause(text: str, key: str):
    """the value side of the LAST statement 'key = ...' in a reading, else None. A line holds several
    statements: '... => 6i = -15 => i = -5/2 A', 'V_3 = 235, V_4 = 10'."""
    t = re.sub(r"\{([^{}=]*)=\}", r"{\1}=", undecorate(text))     # 'P_{9V=}' (the reader put '=' inside);
    out = None                                     # 'V \overset{KVL}{=} ...', ':=' an equals sign
    for clause in (c for part in _ARROW.split(t) for c in _CLAUSE.split(part)):
        if "=" in clause:
            lhs, rhs = clause.split("=", 1)
            if var_key(lhs) == key and rhs.strip():
                out = rhs
    return out


def _named_lhs(text: str) -> str:
    """the name on the left of the first statement (kept for callers and tests)"""
    t = re.sub(r"\{([^{}=]*)=\}", r"{\1}=", text)
    if "=" not in t:
        return ""
    return var_key(_ARROW.split(t.split("=", 1)[0])[-1])


_ARITH = re.compile(r"^([-+]?\(?[\d.]+\)?(?:[*/][-+]?\(?[\d.]+\)?)+)(.*)$")


def simplify_value(text: str) -> str:
    """'-5/2A' -> '-2.5A', '=-117/2=-58.5W' unchanged (its last side is a number): the last side, if it
    is plain arithmetic followed by a unit, is evaluated"""
    sides = text.split("=")
    m = _ARITH.match(sides[-1].strip())
    if m:
        v = safe_eval(m.group(1))
        if v is not None:
            sides[-1] = f"{v:g}{m.group(2)}"
    return "=".join(sides)


def grade_value(item: dict, segments) -> ItemResult:
    pts, name = float(item["points"]), str(item["value"])
    key = var_key(name)
    found = []                                     # (probability, segment, matching readings, boxed)
    for s in segments:
        if s.kind != "math":
            continue
        if "page" in item and getattr(s, "page", None) not in (None, int(item["page"])):
            continue
        rs = nbest_probs(s.nbest)
        match = [(simplify_value(norm_math(v)), p) for v, p in ((named_clause(t, key), p) for t, p in rs)
                 if v is not None]
        m = sum(p for _, p in match)
        if m > 0:                                  # (a frame the layout saw, or one the formula reader wrote)
            framed = sum(p for t, p in rs if re.search(r"\\(?:boxed|fbox|framebox)(?![A-Za-z])", t)) >= 0.5
            found.append((m, s, match, bool(getattr(s, "boxed", False)) or framed))
    if any(f[3] for f in found):
        found = [f for f in found if f[3]]
    best = max(found, key=lambda f: f[0], default=None)
    if best is None or best[0] < FOUND_MIN:
        guess = f" (best: {best[0]:.0%})" if best else ""
        return ItemResult(item["id"], pts, "value", "review", reason=f"{name} = ... not found on the page{guess}")
    m, seg, match, boxed = best
    cands = [Reading(t, p) for t, p in match] + ([Reading("?", 1 - m)] if m < 0.999 else [])
    q = Question(item["id"], name, [Criterion("value", "final_numeric", pts, str(item["answer"]),
                                              rel_tol=float(item.get("rel_tol", 0.02)),
                                              prefix_error_credit=float(item.get("prefix_credit", 0.0)))])
    g = grade_question(q, cands)
    route, reason = g.route, g.reason
    if seg.p_kind < LAYOUT_MIN and route == "auto":
        route, reason = "review", f"unsure this part is a formula ({seg.p_kind:.0%})"
    top = max(nbest_probs(seg.nbest), key=lambda tp: tp[1])[0]
    return ItemResult(item["id"], pts, "value", route, g.most_likely_score, g.expected_score, g.confidence,
                      reason + (" (boxed answer)" if boxed else ""), found=top,
                      readings=[(r.text, round(r.prob, 3)) for r in cands])


def grade_item(item: dict, segments) -> ItemResult:
    pts = float(item["points"])
    if "open" in item:
        return ItemResult(item["id"], pts, "open", "review", reason="open answer: graded by a human")
    if "value" in item:
        return grade_value(item, segments)
    expr = str(item["arithmetic"])
    value = safe_eval(norm_math(expr))
    if value is None:
        raise ValueError(f"item {item['id']}: cannot evaluate {expr!r}")
    target = norm_math(expr)
    best = None                                    # (found probability, segment, matching readings)
    for s in segments:
        if s.kind != "math":
            continue
        rs = nbest_probs(s.nbest)                      # [(text, logp)] -> [(text, p)] over the n-best
        match = [(lhs_rhs(t)[1], p) for t, p in rs if target in lhs_rhs(t)[0]]
        m = sum(p for _, p in match)
        if m > 0 and (best is None or m > best[0]):
            best = (m, s, match)
    if best is None or best[0] < FOUND_MIN:
        guess = f" (best: {best[0]:.0%})" if best else ""
        return ItemResult(item["id"], pts, "arithmetic", "review",
                          reason=f"exercise {expr} not found on the page{guess}")
    m, seg, match = best
    cands = [Reading(rhs, p) for rhs, p in match] + ([Reading("?", 1 - m)] if m < 0.999 else [])
    q = Question(item["id"], expr, [Criterion("value", "final_numeric", pts, repr(value), rel_tol=0.005,
                                                   prefix_error_credit=float(item.get("prefix_credit", 0.0)))])
    g = grade_question(q, cands)
    route, reason = g.route, g.reason
    if seg.p_kind < LAYOUT_MIN and route == "auto":
        route, reason = "review", f"unsure this part is a formula ({seg.p_kind:.0%})"
    top = max(nbest_probs(seg.nbest), key=lambda tp: tp[1])[0]
    return ItemResult(item["id"], pts, "arithmetic", route, g.most_likely_score, g.expected_score, g.confidence,
                      reason, found=top, readings=[(r.text, round(r.prob, 3)) for r in cands])


def grade_page(rubric: dict, segments) -> dict:
    items = [grade_item(it, segments) for it in rubric["items"]]
    auto = [r for r in items if r.route == "auto"]
    review = [r for r in items if r.route != "auto"]
    return {
        "name": rubric.get("name", ""),
        "max_points": sum(r.points for r in items),
        "auto_points": sum(r.score for r in auto),
        "auto_max": sum(r.points for r in auto),
        "review_points": sum(r.points for r in review),
        "items": [r.__dict__ for r in items],
    }


def load_rubric(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
