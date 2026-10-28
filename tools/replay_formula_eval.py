"""The real-formula evaluation (eval_real.py) again from the readings it saved - no recognizer, seconds on a CPU:
checks that a change to the parser or the grader moves none of the 140 grading decisions on the 70 real handwritten
formulas (data/handwritten_formulas) the wrong way.

    python tools/replay_formula_eval.py            # the readings in data/handwritten_formulas/pix2text_readings.json
    python tools/replay_formula_eval.py --readings results/formula_eval.json

Scenarios as in eval_real.py: A) the reference is the ground truth (the student is right), B) the reference is
another formula (the student is wrong). Exit status 1 when a grade is auto-accepted and wrong.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def replay(readings_path=None, threshold=0.85):
    """-> {scenario: {'n', 'auto', 'auto_correct', 'auto_wrong', 'review'}}, [the wrong auto-accepted ones]"""
    from grader import Criterion, Question, grade_question
    from recog import readings_from_candidates
    from recog.formula import Candidate
    data = os.path.join(HERE, "data", "handwritten_formulas")
    labels = {os.path.basename(k): v for k, v in json.load(open(os.path.join(data, "labels.json"), encoding="utf-8")).items()}
    path = readings_path or os.path.join(data, "pix2text_readings.json")
    cached = json.load(open(path, encoding="utf-8"))["readings"]
    files = sorted(f for f in labels if f in cached)
    norm = lambda s: s.replace(" ", "")                                       # noqa: E731
    out, bad = {}, []
    for scenario in ("A:correct", "B:wrong"):
        out[scenario] = {"n": 0, "auto": 0, "auto_correct": 0, "auto_wrong": 0, "review": 0}
    for i, f in enumerate(files):
        cands = [Candidate(c["text"], c["logp"], c["min_token_p"]) for c in cached[f]]
        readings = readings_from_candidates(cands)
        gt = labels[f]
        other = next(labels[g] for g in sorted(labels)[i + 1:] + sorted(labels)[:i] if norm(labels[g]) != norm(gt))
        for scenario, ref, truth in (("A:correct", gt, 1.0), ("B:wrong", other, 0.0)):
            q = Question(f, "", [Criterion("f", "final_formula", 1, ref)])
            r = grade_question(q, readings, confidence_threshold=threshold)
            s = out[scenario]
            s["n"] += 1
            if r.route == "auto":
                s["auto"] += 1
                if r.most_likely_score == truth:
                    s["auto_correct"] += 1
                else:
                    s["auto_wrong"] += 1
                    bad.append((f, scenario, cands[0].text, ref))
            else:
                s["review"] += 1
    return out, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", default=None)
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()
    summary, bad = replay(a.readings, a.threshold)
    for sc, s in summary.items():
        print(f"{sc:10} n={s['n']}  auto={s['auto']}  auto&correct={s['auto_correct']}  auto&WRONG={s['auto_wrong']}  review={s['review']}")
    for f, sc, read, ref in bad:
        print(f"  WRONG {f} {sc}\n     read: {read}\n     ref:  {ref}")
    sys.exit(1 if bad else 0)
