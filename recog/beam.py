"""Model-agnostic beam search returning an n-best list with *un-normalised* log-probs.

The grading layer turns the n-best list into a probability distribution over readings,
so we need the total log-probability of each sequence (sum of token log-probs), not the
length-normalised score that is only meant for ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Hypothesis:
    tokens: list[int]
    logp: float                                  # sum of token log-probs (EOS included)
    token_logps: list[float] = field(default_factory=list)

    def score(self, length_penalty: float) -> float:   # ranking score (HF convention)
        return self.logp / (max(1, len(self.token_logps)) ** length_penalty)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def beam_search(
    step_logits: Callable[[np.ndarray], np.ndarray],
    *,
    start_id: int,
    eos_id: int,
    num_beams: int = 3,
    max_new_tokens: int = 160,
    length_penalty: float = 1.0,
    allowed: Callable[[list[int]], np.ndarray] | None = None,
) -> list[Hypothesis]:
    """`step_logits(input_ids[b, t]) -> logits[b, vocab]` for the last position.
    allowed(tokens after the start token) -> the token ids that may come next (a grammar: e.g. only
    '<question number> <letter>' readings); the others are never chosen. The log-probabilities stay
    the model's own (not renormalised over the allowed tokens)."""
    alive = [Hypothesis([start_id], 0.0)]
    finished: list[Hypothesis] = []

    for _ in range(max_new_tokens):
        ids = np.array([h.tokens for h in alive], dtype=np.int64)
        lp = log_softmax(step_logits(ids).astype(np.float64))        # [b, vocab]
        if allowed is not None:
            for b, h in enumerate(alive):
                ok = allowed(h.tokens[1:])
                row = np.full(lp.shape[1], -np.inf)
                if len(ok):
                    row[ok] = lp[b, ok]
                lp[b] = row
        cands = []
        k = 2 * num_beams
        for b, h in enumerate(alive):
            top = np.argpartition(-lp[b], k)[:k]
            for t in top:
                if not np.isfinite(lp[b, t]):
                    continue
                cands.append((h.logp + lp[b, t], b, int(t), float(lp[b, t])))
        cands.sort(key=lambda c: -c[0])

        new_alive = []
        for total, b, t, tlp in cands:
            h = alive[b]
            nh = Hypothesis(h.tokens + [t], total, h.token_logps + [tlp])
            if t == eos_id:
                finished.append(nh)
            else:
                new_alive.append(nh)
            if len(new_alive) == num_beams:
                break
        alive = new_alive

        if len(finished) >= num_beams:
            worst_kept = sorted(h.score(length_penalty) for h in finished)[-num_beams]
            best_alive = max((h.score(length_penalty) for h in alive), default=-np.inf)
            if best_alive < worst_kept:
                break
        if not alive:
            break

    pool = finished if len(finished) >= num_beams else finished + alive
    pool.sort(key=lambda h: -h.score(length_penalty))
    return pool[:num_beams]
