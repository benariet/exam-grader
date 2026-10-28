"""Grading core: criterion-level scoring + confidence from n-best OCR readings.

Design:
  * The *grading* of a number or formula is deterministic and explainable.
  * Almost all uncertainty comes from *reading* the handwriting, so the OCR /
    VLM layer passes in several candidate readings with probabilities
    (an n-best list, or the empirical distribution over N sampled transcriptions).
  * We grade every candidate, aggregate into a distribution over scores, and
    report: expected score, most likely score, confidence = P(most likely score),
    and a routing decision (auto-accept vs. send to a human).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .numeric import compare_numeric, parse_numeric, unit_given
from .rubric import Criterion, Question
from .symbolic import compare_formula


@dataclass
class Reading:
    """One candidate transcription of the student's answer and its probability."""
    text: str
    prob: float = 1.0


@dataclass
class CriterionResult:
    criterion_id: str
    score: float
    max_points: float
    status: str
    explanation: str


def grade_criterion(c: Criterion, answer_text: str) -> CriterionResult:
    if c.type == "final_numeric":
        v = compare_numeric(answer_text, c.reference, rel_tol=c.rel_tol)
        if v.status == "correct":
            score = c.points
            try:
                if c.missing_unit_penalty and not unit_given(answer_text, c.reference):
                    score -= c.missing_unit_penalty
                    v.detail += "; unit missing"
            except Exception:  # noqa: BLE001
                pass
        elif v.status in ("prefix_error", "factor_2pi"):
            score = c.points * c.prefix_error_credit
        else:
            score = 0.0
        return CriterionResult(c.id, max(score, 0.0), c.points, v.status, v.detail)

    if c.type == "final_formula":
        v = compare_formula(answer_text, c.reference)
        score = c.points if v.status == "equivalent" else 0.0
        return CriterionResult(c.id, score, c.points, v.status, v.detail)

    raise ValueError(f"unknown criterion type {c.type}")


@dataclass
class GradeResult:
    question_id: str
    expected_score: float
    most_likely_score: float
    confidence: float                       # P(score == most_likely_score) over readings
    max_points: float
    route: str                              # 'auto' | 'review'
    reason: str
    score_distribution: dict[float, float] = field(default_factory=dict)
    per_reading: list[dict] = field(default_factory=list)


def grade_question(
    q: Question,
    readings: list[Reading],
    *,
    confidence_threshold: float = 0.85,
    review_on_parse_error: bool = True,
) -> GradeResult:
    """Grade a question given n-best readings of the student's answer(s).

    `readings[i].text` is a *single* answer text per question for this prototype
    (one final answer). Probabilities are renormalized.
    """
    total_p = sum(r.prob for r in readings) or 1.0
    dist: dict[float, float] = defaultdict(float)
    per_reading = []
    unparsed_mass = 0.0   # probability mass of readings we could not grade at all
    partly = {}           # a phasor answer partly right ('magnitude_only', 'conjugate', 'angle_wrong'): its detail

    for r in readings:
        p = r.prob / total_p
        crit_results = [grade_criterion(c, r.text) for c in q.criteria]
        score = round(sum(cr.score for cr in crit_results), 3)
        if any(cr.status in ("parse_error", "magnitude_only", "conjugate", "angle_wrong") for cr in crit_results):
            unparsed_mass += p                     # (a phasor's size only, its conjugate, its angle wrong: partly
            #                                        right - how much is the lecturer's call: to a human)
            for cr in crit_results:
                if cr.status in ("magnitude_only", "conjugate", "angle_wrong"):
                    partly[cr.explanation] = partly.get(cr.explanation, 0.0) + p
        else:
            dist[score] += p
        per_reading.append({
            "text": r.text, "prob": round(p, 3), "score": score,
            "criteria": [cr.__dict__ for cr in crit_results],
        })

    if not dist:  # nothing parsed
        why = (f"partly right, for a human: {max(partly, key=partly.get)}" if partly
               else "no candidate reading could be parsed")
        return GradeResult(q.id, 0.0, 0.0, 0.0, q.max_points, "review", why, {}, per_reading)

    most_likely = max(dist.items(), key=lambda kv: kv[1])
    expected = sum(s * p for s, p in dist.items()) / (1 - unparsed_mass)
    confidence = most_likely[1]   # unparsed mass counts against confidence

    if review_on_parse_error and unparsed_mass > 1 - confidence_threshold:
        route, reason = "review", (f"partly right, for a human: {max(partly, key=partly.get)}" if partly else
                                   f"{unparsed_mass:.0%} of the reading probability is unparseable")
    elif confidence < confidence_threshold:
        route, reason = "review", f"readings disagree on the score (confidence {confidence:.0%})"
    else:
        route, reason = "auto", f"confidence {confidence:.0%}"

    return GradeResult(
        question_id=q.id,
        expected_score=round(expected, 3),
        most_likely_score=most_likely[0],
        confidence=round(confidence, 3),
        max_points=q.max_points,
        route=route,
        reason=reason,
        score_distribution={k: round(v, 3) for k, v in sorted(dist.items())},
        per_reading=per_reading,
    )
