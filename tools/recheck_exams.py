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


CROSSED = 0.6           # a piece this much under the student's crossing-out (tools/find_crossed.py) is not judged


def load(path):
    """the saved pages as the checks take them; pieces the student crossed out left out (counted: "crossed_out")"""
    from recog.segment import Segment
    pages = []
    for p in json.load(open(path, encoding="utf-8")):
        lines = []
        gone = 0
        for segs in p["lines"]:
            row = []
            for s in segs:
                if (s.get("crossed") or {}).get("share", 0) >= CROSSED:
                    gone += 1
                    continue
                seg = Segment(kind=s["kind"], box=tuple(s["box"]), p_kind=s["p_kind"], nbest=[(t, lp) for t, lp in s["nbest"]])
                for k in ("math2", "formula_nbest"):         # (what the heavy readers added - read_exams.py keeps it)
                    if s.get(k):
                        setattr(seg, k, [(t, lp) for t, lp in s[k]])
                seg.note = s.get("note", "")
                seg.boxed = bool(s.get("boxed", False))
                row.append(seg)
            if row:
                lines.append(row)
        pages.append({"page": None, "info": p["info"], "lines": lines, "figures": p.get("figures", []), "circuits": [],
                      "crossed_out": gone})
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
_NAME_SIDE = re.compile(r"\s*(?:\\[lr]vert|\|)?\s*\\?[A-Za-z]+(?:_\{[^{}]*\}|_\w)?(?:\([^()]*\))?\s*(?:\\[lr]vert|\|)?\s*")


def _delay_name(t: str) -> str:
    """'t P_{HL}', 'tp_{LH}' (a delay's p read apart from its t) -> 't_{pHL}'"""
    return re.sub(r"^\s*t\s*[pP]\s*_\s*\{\s*(HL|LH|d)\s*\}", lambda m: "t_{p" + m.group(1) + "}", t)


def _name_sides(s) -> list:
    """the chain's names: its sides before the first that is no name ('t_{PD} = t_{PHL} = C_L ... = 51.86 ps': both),
    past a misread first side ('t\\Omega_2 = t P_{HL} = ...': t_{pHL})"""
    texts = s.side_texts if len(s.side_texts) == len(s.sides) else s.side_texts[:1]
    out, started = [], False
    for t in (texts[:-1] if len(texts) == len(s.sides) else texts):
        t = _delay_name(t)
        if _NAME_SIDE.fullmatch(t):
            out.append(t)
            started = True
        elif started:
            break
    return out


def key_values(stmts) -> dict:
    """the values a page names -> {name (one case, no braces; |x| as x): [(the value in SI units, as shown, |x| written)]}
    - a letter's value ('Y_2 = 1.151 V') or a chain's result ('V_{BS} = V_{DD} - Y_2 = 0.649 V'); not 0/1 logic values"""
    from grader.selfcheck import _canon_name, _last_unit, _num, _pretty, _si
    import sympy as sp
    out = {}
    for s in stmts:
        last = re.split(r"(?<![<>!])=|\\approx", s.text or "")[-1].strip()      # the result as written ('21.6 \mu A')
        if s.kind == "value" and s.symbol is not None and isinstance(s.number, (int, float)):
            name, v, unit = (_delay_name(s.side_texts[0]) if s.side_texts else str(s.symbol)), float(s.number), s.unit or ""
            names = [name]
            if v in (0.0, 1.0) and unit in ("", "S", "W"):
                continue                                 # (a node's state, an input's value)
            f = _si(unit) if unit else None
            v = v * f if f else v
        elif s.kind == "chain" and s.side_texts and len(s.sides) >= 2 and isinstance(s.sides[-1], sp.Basic) and \
                not s.sides[-1].free_symbols and _name_sides(s):
            names, v = _name_sides(s), _num(s.sides[-1])
            name = names[0]
            if v is None:
                continue
            if "10^{" not in (s.side_texts[-1] if len(s.side_texts) == len(s.sides) else ""):
                unit = s.unit or _last_unit(s.text or "")
                f = _si(unit) if unit else None          # (a result in the units written: '5.65 \mu A')
                v = v * f if f else v
        else:
            continue
        for name in names:
            absolute = bool(_ABS.search(name))
            key = _canon_name(_ABS.sub("", name)).replace("\\", "")
            out.setdefault(key, []).append((v, _pretty(name), absolute, _pretty(last)))
            alias = re.fullmatch(r"t_p?d?,(.+)", key)      # ('t_{pd,NAND}' as 't_{NAND}' too: a delay's name either way)
            if alias:
                out.setdefault("t_" + alias.group(1), []).append((v, _pretty(name), absolute, _pretty(last)))
    return out


def _result(s):
    """a statement's number in SI units: a value's ('Y_2 = 1.151 V') or a chain's last side, named or not
    ('= C_L \\frac{\\Delta V}{I} = 69.7 ns') -> float or None"""
    from grader.selfcheck import _last_unit, _num, _si
    import sympy as sp
    if s.kind == "value" and isinstance(s.number, (int, float)):
        f = _si(s.unit) if s.unit else None
        return float(s.number) * f if f else float(s.number)
    if s.kind == "cont" and s.side_texts:                  # ('= 69.8 ns' under the line it goes on: its last side alone)
        from grader.selfcheck import statements
        got = statements("x = " + s.side_texts[-1])
        return _result(got[0]) if len(got) == 1 and got[0].kind == "value" else None
    if s.kind == "chain" and len(s.sides) >= 2 and isinstance(s.sides[-1], sp.Basic) and not s.sides[-1].free_symbols:
        v = _num(s.sides[-1])
        if v is None:
            return None
        if "10^{" not in (s.side_texts[-1] if len(s.side_texts) == len(s.sides) else ""):
            unit = s.unit or _last_unit(s.text or "")
            f = _si(unit) if unit else None
            v = v * f if f else v
        return v
    return None


def checkpoints(stmts) -> list:
    """the results a question's solution works out -> [(its name as shown, value in SI, as written)]: each chain's
    result once (a value given in the question, 'C_L = 1 pF', is no result; 0 and 1 are too common to look for)"""
    from grader.selfcheck import _pretty
    out = []
    for s in stmts:
        if s.kind != "chain":
            continue
        v = _result(s)
        if v is None or abs(v) in (0.0, 1.0) or any(_close(v, w) for _, w, _ in out):
            continue
        last = re.split(r"(?<![<>!])=|\\approx", s.text or "")[-1].strip()
        out.append((_pretty(s.side_texts[0]) if s.side_texts else "", v, _pretty(last)))
    return out


def _close(a, b) -> bool:
    return abs(abs(a) - abs(b)) <= 0.02 * max(abs(a), abs(b))


def coverage(cps: list, stmts) -> tuple:
    """which of the solution's results (checkpoints) the student's work reaches, by value under any name (a student
    calls the solution's t_{pd,NAND} 't_{p21}' or nothing at all) -> (reached [(name, as written)], missing ...)"""
    vals = [v for v in (_result(s) for s in stmts) if v]
    hit = [any(_close(v, w) for w in vals) for _, v, _ in cps]
    return [(n, t) for (n, _, t), h in zip(cps, hit) if h], [(n, t) for (n, _, t), h in zip(cps, hit) if not h]


def solution_text(tex: str) -> str:
    """a line of the official solution as its document converter wrote it -> plain: '{= t}_{p0}' -> '= t_{p0}', a qualifier
    inside a name's subscript ('t_{pd,NAND;S = 1}': its sizing) left out"""
    tex = re.sub(r"\{=\s*([A-Za-z])\}_", r"= \1_", tex)
    return re.sub(r";\s*S\s*=\s*\d+", "", tex)


def against_key(student: dict, key: dict) -> list:
    """-> [(name as shown, the student's value, the solution's values, agrees)] for each name both give; a delay the
    solution names otherwise ('t_{p21}' for its 't_{NAND}') against all its delays: agrees True when like one of them
    (a unitless value at any prefix), None - like none of them (a value to look at)"""
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
        loose = False
        if sol is None:                                     # ('V_T' against the solution's 'V_{Tp}', 'V_{Tn}': any - shown 'V_T?')
            sol = [x for k, xs in key.items() if k.startswith(name) and re.fullmatch(r"[a-z]{1,2}", k[len(name):])
                   for x in xs] or None
            loose = bool(sol)
        family = False
        if sol is None and re.fullmatch(r"t_\S+", name):     # ('t_{p21}', a delay the solution names t_{NAND}: held
            sol = [x for k, xs in key.items() if k[0] == "t" and k[1:2] in ("_", "") for x in xs if x[0]] or None
            family = bool(sol)                              # against all the solution's delays - like one, or none)
        if not sol:
            continue
        if family:
            for v, shown, ab, written in vals:
                scales = (1,) if re.search(r"\d\s*\\?[A-Za-zµ]", written) else (1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15)
                like = [t for w, _, _, t in sol for f in scales if close(abs(f * v), abs(w))]
                out.append((f"{shown} (as {like[0]}?)" if like else shown, written,
                            sorted({t for _, _, _, t in sol}), True if like else None))
            continue
        for v, shown, ab, written in vals:
            agree = any(close(abs(v), abs(w)) if (ab or sab) else close(sign * v, w) for w, _, sab, _ in sol)
            if not agree and re.search(r"\d\s*\\?[A-Za-zµ]", written):  # a value with a unit 1000 times off the key's,
                for f in (1e3, 1e-3):                       # near it otherwise: a prefix slip ('51.86 ps' for 51.86 ns)
                    if any(abs(f * v - w) <= 0.3 * abs(w) for w, _, _, _ in sol if w):
                        written += f" (x{f:g} is near the solution: a unit prefix slip?)"
                        break
            out.append((shown + ("?" if loose else ""), written, sorted({("-" if sign < 0 else "") + t for _, _, _, t in sol}), agree))
    seen = {w.split(" (")[0] for _, w, _, a in out if a is not None}   # (one value under two names, 't_{PD} = t_{PHL}
    return [o for o in out if o[3] is not None or o[1].split(" (")[0] not in seen]   # = 51.86 ps': once)


PER_FINDING = 2.0     # points off estimated per finding - fixed before the first blind test, not fitted to an exam


def estimate(c: dict, full: float | None) -> tuple:
    """a question's counts -> (estimated points off or None when nothing in it could be checked, findings, checked):
    a finding is a sure wrong step, a value unlike the solution's or like none of its values - PER_FINDING points each,
    at most the question's points; not-sure verdicts are listed, not counted"""
    checked = c["right"] + c["wrong_sure"] + c["wrong_unsure"] + c["agrees"] + c["differs"] + c["unlike"]
    findings = c["wrong_sure"] + c["differs"] + c["unlike"]
    if not checked:
        return None, findings, checked
    off = findings * PER_FINDING
    return (min(off, full) if full else off), findings, checked


def main():
    from grader.selfcheck import check_pages
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("students", nargs="*")
    ap.add_argument("--pages", default="", help="tools/exam_questions.py's pages.json: judge each question's pages together")
    ap.add_argument("--grades", default="", help="the lecturer's sheet (.xlsx): points off per question, beside the verdicts")
    ap.add_argument("--quiet", action="store_true", help="the table only, not every verdict")
    ap.add_argument("--solution", default="", help="the official solution's math per question (JSON): the values against it")
    ap.add_argument("--remarks", action="store_true", help="per question, remarks in Hebrew as a lecturer writes them "
                    "(-> <out>/<n>_auto_remarks.txt): a value unlike the solution's, a result not found, a sure wrong step")
    ap.add_argument("--model", default="", help="tools/fit_estimate.py's fitted estimate (JSON): an estimate v2 table")
    ap.add_argument("--points", default="", help="each question's points, '1:50,2:25,3:25': an estimate of the points off")
    args = ap.parse_args()
    want = set(args.students)
    qpages = json.load(open(args.pages, encoding="utf-8")) if args.pages else {}
    marks = grades(args.grades) if args.grades else {}
    full = {int(q): float(v) for q, v in (x.split(":") for x in args.points.split(",") if x)} if args.points else {}
    keys, cps = {}, {}
    if args.solution:
        from grader.selfcheck import statements
        sol = json.load(open(args.solution, encoding="utf-8"))
        for q in sorted({o["q"] for o in sol}):
            st = []
            for k, o in enumerate(x for x in sol if x["q"] == q):
                st += statements(solution_text(o["tex"]), line=k, piece=k)
            keys[q] = key_values(st)
            cps[q] = checkpoints(st)
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
            c = counts.setdefault(key, {"right": 0, "wrong_sure": 0, "wrong_unsure": 0, "other": 0, "differs": 0, "agrees": 0,
                                          "unlike": 0, "remarks": []})
            gone = sum(pages[k - 1].get("crossed_out", 0) for k in pnos)
            if gone:
                c["crossed_out"] = gone
                c["remarks"].append(f"מה שהסטודנט מחק לא נבדק ({gone} קטעים).")
            if keys and qpages and key in keys:          # the values named against the official solution's
                best = _best_statements([pages[k - 1] for k in pnos])
                if cps.get(key):                          # which of the solution's results the work reaches at all
                    got, miss = coverage(cps[key], best)
                    c["reached"], c["results"] = len(got), len(cps[key])
                    if miss:
                        c["remarks"].append("לא נמצא בדף ערך שתואם לפתרון עבור: " + ", ".join(f"{n} (בפתרון {t})" for n, t in miss) + ".")
                    if not args.quiet:
                        print(f"{name}\t{tag}\tp{','.join(map(str, pnos))}\tcoverage\t{len(got)} of {len(cps[key])}\t"
                              f"reached: {', '.join(n for n, _ in got) or '-'}; not found: {', '.join(n for n, _ in miss) or '-'}")
                for shown, v, sol, agree in against_key(key_values(best), keys[key]):
                    if agree is False:
                        slip = " (x1000 is near" in v
                        c["remarks"].append(f"{shown}: אצלך {v.split(' (')[0]}, ובפתרון {' או '.join(sol[:3])}." +
                                            (" ייתכן בלבול בקידומת היחידה (פי 1000)." if slip else ""))
                    elif agree is None:
                        c["remarks"].append(f"{shown} = {v}: לא דומה לאף ערך מתאים בפתרון ({', '.join(sol[:6])}).")
                    c["agrees" if agree else "differs" if agree is False else "unlike"] += 1
                    if not args.quiet:
                        word = "agrees" if agree else "DIFFERS" if agree is False else "UNLIKE ANY"
                        print(f"{name}\t{tag}\tp{','.join(map(str, pnos))}\tkey\t{word}\t"
                              f"{shown} = {v}; the solution: {' or '.join(sol[:6])}{' ...' if len(sol) > 6 else ''}")
            for ch in checks:
                if ch.verdict == "right":
                    c["right"] += 1
                elif ch.verdict.startswith("wrong"):
                    c["wrong_sure" if ch.sure else "wrong_unsure"] += 1
                    if ch.sure:
                        c["remarks"].append(f"צעד שגוי: {ch.text[:160]}")
                else:
                    c["other"] += 1
                if not args.quiet:
                    print(f"{name}\t{tag}\tp{','.join(map(str, pnos))}\t{ch.verdict}\t{'sure' if ch.sure else 'unsure'}\t{ch.text[:220]}")
        if qpages:
            for q in sorted(k for k in counts if k is not None):
                off = marks.get(name, {}).get(int(q)) if marks else None
                table.append((name, q, off, counts[q]))
    if table and full:
        print("\nstudent\tquestion\tpoints\testimated off (findings x %g)\tfindings\tchecks\tsolution results reached\t"
              "lecturer took off" % PER_FINDING)
        for name, q, off, c in table:
            est, nf, nc = estimate(c, full.get(int(q)))
            pts = full.get(int(q))
            reach = f"{c['reached']} of {c['results']}" if c.get("results") else ""
            print(f"{name}\t{q}\t{'' if pts is None else f'{pts:g}'}\t{'not checked' if est is None else f'{est:g}'}\t{nf}\t{nc}\t"
                  f"{reach}\t{'' if off is None else f'{off:g}'}")
    if table and full and args.model:                    # the estimate fitted on graded exams (tools/fit_estimate.py)
        from tools.fit_estimate import predict
        model = json.load(open(args.model, encoding="utf-8"))
        print(f"\nestimate v2 (fitted on exams {', '.join(model.get('fitted_on', []))}: per question off = "
              f"{model['alpha']:.2f} x points x missing + {model['beta']:.2f} x findings + {model['gamma']:.2f} x points "
              f"/ (1 + checks); without the solution's results: the mean share off)")
        print("student	question	points	estimated off v2	lecturer took off")
        tot = {}
        for name, q, off, c in table:
            pts = full.get(int(q))
            if pts is None:
                continue
            _, nf, nc = estimate(c, pts)
            r = {"exam": name, "q": str(q), "points": pts, "findings": nf, "checks": nc, "reached": c.get("reached"),
                 "results": c.get("results")}
            v = predict(r, model)
            t = tot.setdefault(name, [0.0, 0.0, True])
            t[0] += v
            if off is None:
                t[2] = False
            else:
                t[1] += off
            print(f"{name}	{q}	{pts:g}	{v:.1f}	{'' if off is None else f'{off:g}'}")
        total = sum(full.values())
        print("student	estimated grade v2	grade")
        for name, (v, o, known) in tot.items():
            print(f"{name}	{total - v:.0f}	{f'{total - o:g}' if known and marks else ''}")
    if table and args.remarks:
        for name in dict.fromkeys(t[0] for t in table):
            lines = []
            for n, q, off, c in table:
                if n != name:
                    continue
                lines.append(f"שאלה {q} - הערות אוטומטיות")
                checked = c["right"] + c["wrong_sure"] + c["wrong_unsure"] + c["agrees"] + c["differs"] + c["unlike"]
                manual = [] if c.get("results") else ["בפתרון אין כאן ערכים מחושבים (ציור, טבלת אמת, בחירה או הסבר): לבדיקה ידנית."]
                rem = manual + c["remarks"] or ["לא נמצאה טעות בחישובים שנקראו." if checked else "לא נקראו כאן חישובים."]
                lines += [f"  - {r}" for r in rem]
            open(os.path.join(args.out, f"{name}_auto_remarks.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
            print(f"\n{name}: automatic remarks\n" + "\n".join(lines))
    if table:
        print("\nstudent\tquestion\tpoints off\tright\twrong (sure)\twrong (not sure)\tother\tsolution: agrees\tdiffers\t"
              "like none\tlook at")
        for name, q, off, c in table:
            look = ""
            if off is not None and off == 0 and c["wrong_sure"]:
                look = "nothing off, a sure wrong: a false wrong or a misreading?"
            elif off is not None and off >= 5 and not (c["wrong_sure"] or c["wrong_unsure"] or c["differs"] or c["unlike"]):
                look = "points off, nothing found"
            elif off is not None and off == 0 and c["differs"]:
                look = "nothing off, a value unlike the solution's: a misreading?"
            print(f"{name}\t{q}\t{'' if off is None else f'{off:g}'}\t{c['right']}\t{c['wrong_sure']}\t{c['wrong_unsure']}\t{c['other']}\t"
                  f"{c['agrees']}\t{c['differs']}\t{c['unlike']}\t{look}")


if __name__ == "__main__":
    main()
