"""Grading answers that mix Hebrew text and math.

The answer arrives as segments (recog/segment.py), each with its kind (hebrew / english / math), the
layout classifier's probability for that kind, and an n-best list of readings with log-probs.
Each rubric criterion looks at the relevant segments and yields a probability distribution
over its points; criteria are combined (assumed independent) into a distribution over the
total score -> expected score, most likely score, confidence, route.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

from .core import grade_criterion
from .numeric import parse_numeric
from .rubric import Criterion
from .symbolic import compare_formula

HEB_PREFIXES = "והבכלמש"

# Handwritten I, l, 1 and | are often indistinguishable. A reading such as 'l = V/R' is then
# counted as a *partial* match for 'I = V/R' (weight below), so the system becomes unsure
# instead of confidently wrong.
CONFUSABLE_WEIGHT = 0.5
_CONFUSABLE_LHS = re.compile(r"^\s*(?:l|1|\\vert|\|)\s*=")


def _confusable_variant(t: str):
    return _CONFUSABLE_LHS.sub("I =", t, count=1) if _CONFUSABLE_LHS.match(t) else None


@dataclass
class MixedCriterion:
    id: str
    type: str                     # "formula_any" | "numeric_final" | "keyword"
    points: float
    references: list = field(default_factory=list)   # formulas | [number with unit] | Hebrew keywords
    rel_tol: float = 0.02
    prefix_error_credit: float = 0.5
    description: str = ""


@dataclass
class CriterionOutcome:
    id: str
    description: str
    points: float
    distribution: dict            # score -> probability
    evidence: str                 # full detail (all readings)
    short: str = ""               # one-line summary for reports

    @property
    def p_full(self):
        return self.distribution.get(self.points, 0.0)


@dataclass
class MixedResult:
    expected_score: float
    most_likely_score: float
    confidence: float
    max_points: float
    route: str
    reason: str
    distribution: dict
    criteria: list


def readings(nbest):
    """[(text, logp), ...] -> [(text, p)] normalised over the n-best list."""
    m = max(lp for _, lp in nbest)
    w = [(t, math.exp(lp - m)) for t, lp in nbest]
    z = sum(p for _, p in w)
    return [(t, p / z) for t, p in w]


def keyword_present(text: str, kw: str) -> bool:
    if kw.isascii():                                   # an English keyword: whole words, any case
        max_ed = 1 if len(kw) >= 5 else 0
        words = re.findall(r"[A-Za-z]+", text.lower())
        return any(Levenshtein.distance(w, kw.lower()) <= max_ed for w in words)
    max_ed = 1 if len(kw) >= 4 else 0
    for tok in text.replace(",", " ").replace(".", " ").split():
        cands = [tok] + [tok[i:] for i in range(1, 3) if all(c in HEB_PREFIXES for c in tok[:i])]
        if any(Levenshtein.distance(c, kw) <= max_ed for c in cands):
            return True
    return False


def _outcome(c, segments) -> CriterionOutcome:
    maths = [s for s in segments if s.kind == "math"]
    hebs = [s for s in segments if s.kind in ("hebrew", "english")]        # text segments

    if c.type == "formula_any":
        p_none, ev, best = 1.0, [], (0.0, None)
        for s in maths:
            def matches(t):
                return any(compare_formula(t, ref).status == "equivalent" for ref in c.references)
            hits, p_ok = [], 0.0
            for t, p in readings(s.nbest):
                if matches(t):
                    hits.append((t, p, False)); p_ok += p
                else:
                    v = _confusable_variant(t)
                    if v and matches(v):                    # 'l = V/R' might be 'I = V/R'
                        hits.append((t, p * CONFUSABLE_WEIGHT, True)); p_ok += p * CONFUSABLE_WEIGHT
            p_none *= 1 - p_ok
            ev.append(f"{readings(s.nbest)[0][0]} (P match={p_ok:.2f})")
            if hits and p_ok > best[0]:
                best = (p_ok, max(hits, key=lambda h: h[1]))
        p = 1 - p_none
        if best[1]:
            t, _, confusable = best[1]
            short = (f"{t.replace(' ', '')}  P={best[0]:.2f}" + ("   (read l or 1 where I is expected: half weight)"
                                                                if confusable else ""))
        else:
            short = "no matching formula read"
        return CriterionOutcome(c.id, c.description, c.points, {c.points: p, 0.0: 1 - p},
                                "; ".join(ev) or "no math found", short)

    if c.type == "numeric_final":
        if not maths:
            return CriterionOutcome(c.id, c.description, c.points, {0.0: 1.0}, "no math found")
        # the final answer = the last formula (reading order) that actually contains a number
        def p_numeric(seg):
            ok = 0.0
            for t, p in readings(seg.nbest):
                try:
                    parse_numeric(t); ok += p
                except Exception:  # noqa: BLE001
                    pass
            return ok
        numeric = [s for s in maths if p_numeric(s) >= 0.5]
        s = numeric[-1] if numeric else maths[-1]
        crit = Criterion(c.id, "final_numeric", c.points, c.references[0], rel_tol=c.rel_tol,
                         prefix_error_credit=c.prefix_error_credit)
        dist, ev = defaultdict(float), []
        for t, p in readings(s.nbest):
            r = grade_criterion(crit, t)
            dist[round(r.score, 3)] += p
            ev.append(f"{t} -> {r.status} (p={p:.2f})")
        t0, p0 = readings(s.nbest)[0]
        short = f"{t0.replace(' ', '')}  ->  {grade_criterion(crit, t0).status}  (p={p0:.2f})"
        return CriterionOutcome(c.id, c.description, c.points, dict(dist), "; ".join(ev), short)

    if c.type == "keyword":
        p_none, ev = 1.0, []
        for s in hebs:
            p_ok = sum(p for t, p in readings(s.nbest) if any(keyword_present(t, k) for k in c.references))
            p_none *= 1 - p_ok
            ev.append(f"{readings(s.nbest)[0][0]} (P={p_ok:.2f})")
        p = 1 - p_none
        short = f"P(found '{c.references[0]}') = {p:.2f}"
        return CriterionOutcome(c.id, c.description, c.points, {c.points: p, 0.0: 1 - p},
                                "; ".join(ev) or "no text found", short)

    raise ValueError(f"unknown criterion type {c.type}")


def grade_mixed(criteria, segments, *, threshold: float = 0.85, min_layout_conf: float = 0.8) -> MixedResult:
    outs = [_outcome(c, segments) for c in criteria]
    total = {0.0: 1.0}
    for o in outs:                                               # convolution of independent criteria
        nxt = defaultdict(float)
        for a, pa in total.items():
            for b, pb in o.distribution.items():
                nxt[round(a + b, 3)] += pa * pb
        total = dict(nxt)
    best, conf = max(total.items(), key=lambda kv: kv[1])
    expected = sum(s * p for s, p in total.items())
    weak = [s for s in segments if s.p_kind < min_layout_conf]
    if weak:
        route, reason = "review", f"unsure what {len(weak)} chunk(s) are (Hebrew / English / math)"
    elif conf < threshold:
        route, reason = "review", f"readings disagree on the score (confidence {conf:.0%})"
    else:
        route, reason = "auto", f"confidence {conf:.0%}"
    return MixedResult(round(expected, 2), best, round(conf, 3), sum(c.points for c in criteria), route, reason,
                       {k: round(v, 4) for k, v in sorted(total.items())}, outs)
