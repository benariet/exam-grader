"""Rubric model: a question = list of criteria, each worth points, each checkable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CriterionType = Literal["final_numeric", "final_formula"]


@dataclass
class Criterion:
    id: str
    type: CriterionType
    points: float
    reference: str                 # reference answer (number+unit, or LaTeX / plain formula)
    rel_tol: float = 0.02          # numeric only
    prefix_error_credit: float = 0.5   # fraction of points if only the SI prefix / decimal point is off
    missing_unit_penalty: float = 0.0  # points to deduct if the student omitted the unit
    description: str = ""


@dataclass
class Question:
    id: str
    text: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def max_points(self) -> float:
        return sum(c.points for c in self.criteria)
