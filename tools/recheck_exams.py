"""Judge saved readings again (tools/read_exams.py's <n>.json) - no reading: the derivation checks of every page of
each exam, as text (only the math and the verdicts: no Hebrew text, no images).

    python tools/recheck_exams.py <out folder> [n ...]
    python tools/recheck_exams.py <out folder> --pages <out folder>/pages.json --grades "grades_moed C.xlsx"

--pages (tools/exam_questions.py): each question's pages are judged together (a value from its first page used on the
next), and each verdict says its question; --grades: per student and question, the points the lecturer took off beside
the verdicts' counts - a question with nothing taken off and a sure wrong is a false wrong to look at (or a misreading);
one with points off and no wrong, a slip the checks missed; --solution (the official solution's math, per question:
[{"q": 1, "tex": "..."}]): each value the student names that the solution names too, against the solution's - a value
worked on from a wrong start ('Y_2 = 1.5 V' where the solution has 1.151) is right step by step, wrong against the key"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def load(path):
    from recog.segment import Segment
    pages = []
    for p in json.load(open(path, encoding="utf-8")):
        lines = []
        for segs in p["lines"]:
            row = []
            for s in segs:
                seg = Segment(kind=s["kind"], box=tuple(s["box"]), p_kind=s["p_kind"], nbest=[(t, lp) for t, lp in s["nbest"]])
                for k in ("math2", "formula_nbest"):         # (what the heavy readers added - read_exams.py keeps it)
                    if s.get(k):
                        setattr(seg, k, [(t, lp) for t, lp in s[k]])
                seg.note = s.get("note", "")
                seg.boxed = bool(s.get("boxed", False))
                row.append(seg)
            lines.append(row)
        pages.append({"page": None, "info": p["info"], "lines": lines, "figures": p.get("figures", []), "circuits": []})
    return pages


def _best_statements(pages) -> list:
    """the pages' statements from each math piece's best reading (as check_pages finds the problems)"""
    from grader.selfcheck import NBEST, _mix, _raw, statements
    from grader.mixed import readings
    out, line, k = [], 0, 0
    for page in pages:
        for segs in page["lines"]:
            for seg in sorted(segs, key=lambda s: s.box[0]):
                if seg.kind == "math" and seg.nbest:
                    rd = _mix(_raw(seg.nbest)[:NBEST], _raw(seg.math2)[:NBEST]) if getattr(seg, "math2", None) \
                        else readings(seg.nbest)[:NBEST]
                    out += statements(rd[0][0], conf=1.0, line=line, piece=k)
                    k += 1
            line += 1
    return out


def grades(path: str) -> dict:
    """the lecturer's sheet -> {student: {question: points taken off}} (the first column 'n/n', columns 'Question k')"""
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = next((i for i, r in enumerate(rows) if any(isinstance(c, str) and re.fullmatch(r"\s*Question\s*1\s*", c) for c in r)), None)
    if head is None:
        raise SystemExit(f"{path}: no 'Question 1' column")
    cols = {int(re.search(r"\d+", c).group()): j for j, c in enumerate(rows[head]) if isinstance(c, str) and re.fullmatch(r"\s*Question\s*\d+\s*", c)}
    out = {}
    for r in rows[head + 1:]:
        if not r or r[0] is None:
            continue
        m = re.match(r"\s*(\d+)", str(r[0]))
        if not m:
            continue
        out[m.group(1)] = {q: (float(r[j]) if isinstance(r[j], (int, float)) else None) for q, j in cols.items()}
    return out


_ABS = re.compile(r"\\[lr]vert|\\left\s*\||\\right\s*\||\\mid|\|")


def key_values(stmts) -> dict:
    """the values a page names -> {name (one case, no braces; |x| as x): [(the value in SI units, as shown, |x| written)]}
    - a letter's value ('Y_2 = 1.151 V') or a chain's result ('V_{BS} = V_{DD} - Y_2 = 0.649 V'); not 0/1 logic values"""
    from grader.selfcheck import _canon_name, _last_unit, _num, _pretty, _si
    import sympy as sp
    out = {}
    for s in stmts:
        last = re.split(r"(?<![<>!])=|\\approx", s.text or "")[-1].strip()      # the result as written ('21.6 \mu A')
        if s.kind == "value" and s.symbol is not None and isinstance(s.number, (int, float)):
            name, v, unit = (s.side_texts[0] if s.side_texts else str(s.symbol)), float(s.number), s.unit or ""
            if v in (0.0, 1.0) and unit in ("", "S", "W"):
                continue                                 # (a node's state, an input's value)
            f = _si(unit) if unit else None
            v = v * f if f else v
        elif s.kind == "chain" and s.side_texts and len(s.sides) >= 2 and isinstance(s.sides[-1], sp.Basic) and \
                not s.sides[-1].free_symbols and re.fullmatch(r"\s*(?:\\[lr]vert|\|)?\s*\\?[A-Za-z]+(?:_\{[^{}]*\}|_\w)?(?:\([^()]*\))?\s*(?:\\[lr]vert|\|)?\s*",
                                                              s.side_texts[0]):
            name, v = s.side_texts[0], _num(s.sides[-1])
            if v is None:
                continue
            if "10^{" not in (s.side_texts[-1] if len(s.side_texts) == len(s.sides) else ""):
                unit = s.unit or _last_unit(s.text or "")
                f = _si(unit) if unit else None          # (a result in the units written: '5.65 \mu A')
                v = v * f if f else v
        else:
            continue
        absolute = bool(_ABS.search(name))
        key = _canon_name(_ABS.sub("", name)).replace("\\", "")
        out.setdefault(key, []).append((v, _pretty(name), absolute, _pretty(last)))
        alias = re.fullmatch(r"t_p?d?,(.+)", key)          # ('t_{pd,NAND}' as 't_{NAND}' too: a delay's name either way)
        if alias:
            out.setdefault("t_" + alias.group(1), []).append((v, _pretty(name), absolute, _pretty(last)))
    return out


def solution_text(tex: str) -> str:
    """a line of the official solution as its document converter wrote it -> plain: '{= t}_{p0}' -> '= t_{p0}', a qualifier
    inside a name's subscript ('t_{pd,NAND;S = 1}': its sizing) left out"""
    tex = re.sub(r"\{=\s*([A-Za-z])\}_", r"= \1_", tex)
    return re.sub(r";\s*S\s*=\s*\d+", "", tex)


def against_key(student: dict, key: dict) -> list:
    """-> [(name as shown, the student's value, the solution's values, agrees)] for each name both give"""
    def close(a, b):
        return abs(a - b) <= 0.02 * max(abs(a), abs(b)) + 1e-15
    def swapped(n):
        """'v_sb' -> 'v_bs': a voltage between two nodes named the other way round (its sign the other)"""
        m = re.fullmatch(r"([vu])_([a-z])([a-z])(\d*)", n)
        return f"{m.group(1)}_{m.group(3)}{m.group(2)}{m.group(4)}" if m and m.group(2) != m.group(3) else None
    out = []
    for name, vals in student.items():
        sol, sign = key.get(name), 1
        if sol is None and "qq" in name:
            sol = key.get(re.sub(r"qq[pmiz]", "", name))     # ('Y_2(0^-)' against the solution's 'Y_2')
        if sol is None and swapped(name):
            sol, sign = key.get(swapped(name)), -1          # ('V_{SB} = -0.649' against the solution's 'V_{BS} = 0.649')
        if sol is None:                                     # ('I_D' against the solution's 'I_D(0)' and 'I_D(t_{pHL})': any)
            sol = [x for k, xs in key.items() if k.startswith(name + "(") for x in xs] or None
        if not sol:
            continue
        for v, shown, ab, written in vals:
            agree = any(close(abs(v), abs(w)) if (ab or sab) else close(sign * v, w) for w, _, sab, _ in sol)
            out.append((shown, written, sorted({("-" if sign < 0 else "") + t for _, _, _, t in sol}), agree))
    return out


def main():
    from grader.selfcheck import check_pages
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("students", nargs="*")
    ap.add_argument("--pages", default="", help="tools/exam_questions.py's pages.json: judge each question's pages together")
    ap.add_argument("--grades", default="", help="the lecturer's sheet (.xlsx): points off per question, beside the verdicts")
    ap.add_argument("--quiet", action="store_true", help="the table only, not every verdict")
    ap.add_argument("--solution", default="", help="the official solution's math per question (JSON): the values against it")
    args = ap.parse_args()
    want = set(args.students)
    qpages = json.load(open(args.pages, encoding="utf-8")) if args.pages else {}
    marks = grades(args.grades) if args.grades else {}
    keys = {}
    if args.solution:
        from grader.selfcheck import statements
        sol = json.load(open(args.solution, encoding="utf-8"))
        for q in sorted({o["q"] for o in sol}):
            st = []
            for k, o in enumerate(x for x in sol if x["q"] == q):
                st += statements(solution_text(o["tex"]), line=k, piece=k)
            keys[q] = key_values(st)
    files = [f for f in glob.glob(os.path.join(args.out, "*.json")) if re.fullmatch(r"\d+\.json", os.path.basename(f))]
    files.sort(key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
    table = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        if want and name not in want:
            continue
        pages = load(f)
        groups = {}
        for pno in range(1, len(pages) + 1):
            q = (qpages.get(name, {}).get(str(pno)) or {}).get("question") if qpages else None
            groups.setdefault(q if qpages else f"p{pno}", []).append(pno)
        counts = {}
        for key, pnos in groups.items():
            if qpages and key is None:
                continue                                 # (a cover, the instructions: no question's)
            checks = check_pages([pages[k - 1] for k in pnos])
            tag = f"q{key}" if qpages else key
            c = counts.setdefault(key, {"right": 0, "wrong_sure": 0, "wrong_unsure": 0, "other": 0, "differs": 0, "agrees": 0})
            if keys and qpages and key in keys:          # the values named against the official solution's
                for shown, v, sol, agree in against_key(key_values(_best_statements([pages[k - 1] for k in pnos])), keys[key]):
                    c["agrees" if agree else "differs"] += 1
                    if not args.quiet:
                        print(f"{name}\t{tag}\tp{','.join(map(str, pnos))}\tkey\t{'agrees' if agree else 'DIFFERS'}\t"
                              f"{shown} = {v}; the solution: {' or '.join(sol)}")
            for ch in checks:
                if ch.verdict == "right":
                    c["right"] += 1
                elif ch.verdict.startswith("wrong"):
                    c["wrong_sure" if ch.sure else "wrong_unsure"] += 1
                else:
                    c["other"] += 1
                if not args.quiet:
                    print(f"{name}\t{tag}\tp{','.join(map(str, pnos))}\t{ch.verdict}\t{'sure' if ch.sure else 'unsure'}\t{ch.text[:220]}")
        if qpages:
            for q in sorted(k for k in counts if k is not None):
                off = marks.get(name, {}).get(int(q)) if marks else None
                table.append((name, q, off, counts[q]))
    if table:
        print("\nstudent\tquestion\tpoints off\tright\twrong (sure)\twrong (not sure)\tother\tsolution: agrees\tdiffers\tlook at")
        for name, q, off, c in table:
            look = ""
            if off is not None and off == 0 and c["wrong_sure"]:
                look = "nothing off, a sure wrong: a false wrong or a misreading?"
            elif off is not None and off >= 5 and not (c["wrong_sure"] or c["wrong_unsure"] or c["differs"]):
                look = "points off, nothing found"
            elif off is not None and off == 0 and c["differs"]:
                look = "nothing off, a value unlike the solution's: a misreading?"
            print(f"{name}\t{q}\t{'' if off is None else f'{off:g}'}\t{c['right']}\t{c['wrong_sure']}\t{c['wrong_unsure']}\t{c['other']}\t"
                  f"{c['agrees']}\t{c['differs']}\t{look}")


if __name__ == "__main__":
    main()
