"""Why the solution's results are not found in the work (tools/recheck_exams.py's coverage): for each result not
reached, where its number is in the readings, if anywhere - counts only (no student's values or readings printed).

    python tools/coverage_gap.py <out folder> --pages pages.json --solution solution_math_q.json [--grades grades.xlsx]

Each result not reached is put in the first class that holds it:
  near      a result of the work within 2-6 % of it (rounded otherwise: 0.69 for 0.7)
  best      its number in the best reading of a math piece, not as a result (a chain the checks could not take)
  other     its number in another reading of a math piece (an n-best's other line, the second reader's)
  text      its number in a piece read as Hebrew or English
  scaled    its number at a thousand times another power (its unit's prefix read or written otherwise)
  nowhere   none of these: not read, or not written
and by the lecturer's sheet: in a question nothing was taken off (the work reaches the result: what is missed is the
reading's or the matching's), or one points were taken off (the student may not have reached it)."""
import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.recheck_exams import (_best_statements, _close, _result, checkpoints, grades, load,  # noqa: E402
                                 solution_text)

CLASSES = ("near", "best", "other", "text", "scaled", "nowhere")
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> list:
    """the numbers written in a reading ('69.8', '0,32' -> 0.32; a thousands' run of digits as one)"""
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group().replace(",", ".")))
        except ValueError:
            pass
    return out


def _near(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def mantissa(written: str):
    """the number a result is written with ('69.8 ps' -> 69.8; '1.38 \\cdot 10^{-12}' -> 1.38)"""
    ns = numbers(written)
    return ns[0] if ns else None


def classify(value: float, written: str, best_results: list, pieces: list, factor: float = 1.0) -> str:
    """pieces: [(kind, best reading, other readings)] of the question's pages -> the first class that holds the result
    (factor: the result moved by it - a control, how often a number not the solution's is found so)"""
    value = value * factor
    if any(r and _near(abs(r), abs(value), 0.06) for r in best_results):
        return "near"
    m = mantissa(written)
    m = m * factor if m is not None else None
    if m is None or m == 0:
        return "nowhere"
    if any(_near(x, m) for kind, best, _ in pieces if kind == "math" for x in numbers(best)):
        return "best"
    if any(_near(x, m) for kind, _, others in pieces if kind == "math" for t in others for x in numbers(t)):
        return "other"
    if any(_near(x, m) for kind, best, others in pieces if kind != "math" for t in [best] + others for x in numbers(t)):
        return "text"
    scales = [10.0 ** k for k in (-12, -9, -6, -3, 3, 6, 9, 12)]
    if any(_near(x * f, m) for _, best, others in pieces for t in [best] + others for x in numbers(t) for f in scales):
        return "scaled"
    return "nowhere"


def pieces_of(path: str, pnos: list) -> list:
    """the question's pieces as read (not crossed out): (kind, best reading, [its other readings])"""
    out = []
    raw = json.load(open(path, encoding="utf-8"))
    for k in pnos:
        for segs in raw[k - 1]["lines"]:
            for s in segs:
                if (s.get("crossed") or {}).get("share", 0) >= 0.6:
                    continue
                nb = [t for t, _ in s.get("nbest") or []]
                others = nb[1:] + [t for key in ("math2", "formula_nbest") for t, _ in (s.get(key) or [])]
                if s.get("line_text"):
                    others.append(s["line_text"])
                out.append((s["kind"], nb[0] if nb else "", others))
    return out


def main():
    from grader.selfcheck import statements
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pages", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--grades", default="")
    ap.add_argument("--control", type=float, default=1.0, help="every result moved by this factor (1.13: how often a "
                    "number that is not the solution's is found where it is looked for - by chance)")
    a = ap.parse_args()
    qpages = json.load(open(a.pages, encoding="utf-8"))
    marks = grades(a.grades) if a.grades else {}
    sol = json.load(open(a.solution, encoding="utf-8"))
    cps = {}
    for q in sorted({o["q"] for o in sol}):
        st = []
        for k, o in enumerate(x for x in sol if x["q"] == q):
            st += statements(solution_text(o["tex"]), line=k, piece=k)
        cps[str(q)] = checkpoints(st)
    files = sorted((f for f in os.listdir(a.out) if re.fullmatch(r"\d+\.json", f)), key=lambda f: int(f[:-5]))
    tot = {"full": Counter(), "off": Counter(), "unknown": Counter()}
    reached = Counter()
    per_exam = {}
    for f in files:
        name = f[:-5]
        path = os.path.join(a.out, f)
        pages = load(path)
        groups = {}
        for pno in range(1, len(pages) + 1):
            q = (qpages.get(name, {}).get(str(pno)) or {}).get("question")
            if q is not None:
                groups.setdefault(str(q), []).append(pno)
        row = Counter()
        for q, pnos in groups.items():
            if not cps.get(q):
                continue
            best = _best_statements([pages[k - 1] for k in pnos])
            results = [_result(s) for s in best]
            vals = [r for r in results if r]
            off = (marks.get(name) or {}).get(int(q))
            side = "unknown" if off is None else "full" if off == 0 else "off"
            pcs = pieces_of(path, pnos)
            for n, v, t in cps[q]:
                if any(_close(v * a.control, w) for w in vals):  # (reached: as recheck_exams.coverage)
                    reached[side] += 1
                    continue
                c = classify(v, t, results, pcs, a.control)
                tot[side][c] += 1
                row[c] += 1
        per_exam[name] = row
    print("results not reached, by where their number is in the readings (counts):")
    print("exam\t" + "\t".join(CLASSES))
    for name, row in per_exam.items():
        print(name + "\t" + "\t".join(str(row[c]) for c in CLASSES))
    for side, label in (("full", "questions with nothing taken off"), ("off", "questions with points taken off"),
                        ("unknown", "questions not on the sheet")):
        if sum(tot[side].values()) or reached[side]:
            print(f"{label}: reached {reached[side]}, not reached {sum(tot[side].values())}: " +
                  ", ".join(f"{c} {tot[side][c]}" for c in CLASSES))


if __name__ == "__main__":
    main()
