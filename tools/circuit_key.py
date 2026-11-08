"""The circuits students drew (tools/read_exam_circuits.py) against the solution's: for a design question - Moed C
2.1, 'S = A + B + C with the fewest transistors' - does a node of the drawing give the function (switch level:
static CMOS, pass transistors, transmission gates; grader.circuit.Netlist.switch_logic), with how many transistors;
beside the points the lecturer took off the question.

    python tools/circuit_key.py <circuits folder> --key key.json [--pages pages.json] [--grades grades.xlsx]
        key.json: [{"q": 2, "function": "S = A + B + C", "inputs": "ABC", "fewest": 4}]
        -> the counts (how many drawings give the function, how many were read with none ...) and, with --grades,
           the mean points off in each group; <circuits folder>/circuit_key.json per exam (stays with the exams);
           --remarks: <n>_circuit_remarks.txt per exam, in Hebrew; --per-exam: a line per exam printed too
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def question_of_pages(name, scan, qpages, by_booklet) -> dict:
    """{scan page: question} - the exam's own page map (tools/exam_questions.py), else the booklet's page, when the
    scan is the booklet page for page"""
    own = qpages.get(name)
    if own:
        return {int(p): v.get("question") for p, v in own.items() if v.get("question") is not None}
    return {p["page"]: by_booklet.get(p["booklet"]) for p in scan["pages"] if p.get("booklet") in by_booklet}


def judge(net, want, inputs):
    """-> (verdict, transistors, the node functions read): 'right' (a node gives it), 'other' (functions, none it),
    'none' (transistors, no function: a network read wrong or drawn wrong)"""
    import sympy as sp
    from sympy.logic.boolalg import Xor
    fets = sum(1 for c in net.components if c.type in ("nmos", "pmos", "fet"))
    if not fets:
        return None, 0, {}
    funcs = net.switch_logic(inputs=inputs)
    up = lambda e: e.xreplace({x: sp.Symbol(str(x).upper()) for x in e.free_symbols})   # noqa: E731
    if any(not sp.satisfiable(Xor(up(f), want)) for f in funcs.values()):
        return "right", fets, funcs
    return ("other" if funcs else "none"), fets, funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits")
    ap.add_argument("--key", required=True)
    ap.add_argument("--pages", default="")
    ap.add_argument("--grades", default="")
    ap.add_argument("--remarks", action="store_true")
    ap.add_argument("--per-exam", action="store_true")
    a = ap.parse_args()
    from grader.circuit import Netlist
    from grader.circuit_check import parse_bool
    keys = json.load(open(a.key, encoding="utf-8"))
    qpages = json.load(open(a.pages, encoding="utf-8")) if a.pages else {}
    votes = {}
    for pp in qpages.values():                         # the booklet's pages' questions, as the mapped exams have them
        for v in pp.values():
            if v.get("booklet") is not None and v.get("question") is not None:
                votes.setdefault(v["booklet"], Counter())[v["question"]] += 1
    by_booklet = {b: c.most_common(1)[0][0] for b, c in votes.items()}
    marks = {}
    if a.grades:
        from tools.recheck_exams import grades
        marks = grades(a.grades)
    files = sorted(glob.glob(os.path.join(a.circuits, "*.circuits.json")),
                   key=lambda f: int(re.sub(r"\D", "", os.path.basename(f).split(".")[0]) or 0))
    results, remarks = {}, {}
    for k in keys:
        q = int(k["q"])
        want = parse_bool(k["function"].split("=", 1)[-1])
        rows = []
        for f in files:
            scan = json.load(open(f, encoding="utf-8"))
            name = scan["exam"]
            qmap = question_of_pages(name, scan, qpages, by_booklet)
            best = None
            n_draw = 0
            for p in scan["pages"]:
                if qmap.get(p["page"]) != q:
                    continue
                for c in p["circuits"]:
                    net = Netlist.from_dict(c["netlist"])
                    verdict, fets, funcs = judge(net, want, k.get("inputs") or None)
                    if verdict is None:
                        continue
                    n_draw += 1
                    rank = {"right": 3, "other": 2, "none": 1}[verdict]
                    if best is None or (rank, -fets) > (best[0], -best[2]):
                        best = (rank, verdict, fets, {n: str(f) for n, f in funcs.items()}, p["page"])
            off = marks.get(name, {}).get(q) if marks else None
            row = {"exam": name, "q": q, "drawings": n_draw, "verdict": best[1] if best else "no drawing",
                   "transistors": best[2] if best else None, "functions": best[3] if best else {},
                   "page": best[4] if best else None, "off": off, "mapped": bool(qmap)}
            rows.append(row)
            if a.per_exam:
                print(f"{name}\tq{q}\t{row['verdict']}\t{row['transistors'] or ''}\t{'' if off is None else off}")
        results[q] = rows
        cnt = Counter(r["verdict"] for r in rows)
        print(f"question {q} ({k['function']}, inputs {k.get('inputs') or 'as read'}): {len(rows)} exams - "
              + ", ".join(f"{v}: {cnt[v]}" for v in ("right", "other", "none", "no drawing")))
        right = [r for r in rows if r["verdict"] == "right"]
        if right and k.get("fewest"):
            print(f"  of those right: with the fewest transistors ({k['fewest']}): "
                  f"{sum(1 for r in right if r['transistors'] == k['fewest'])}, more: "
                  f"{sum(1 for r in right if r['transistors'] > k['fewest'])}")
        if marks:
            for v in ("right", "other", "none", "no drawing"):
                offs = [r["off"] for r in rows if r["verdict"] == v and r["off"] is not None]
                if offs:
                    print(f"  {v}: mean points off question {q} {sum(offs) / len(offs):.1f} (n={len(offs)}; "
                          f"none off: {sum(1 for o in offs if o == 0)})")
        if a.remarks:
            for r in rows:
                txt = {"right": f"המעגל שצויר נותן {k['function']} ({r['transistors']} טרנזיסטורים"
                                + (", המספר המינימלי)." if r["transistors"] == k.get("fewest") else
                                   f"; בפתרון {k['fewest']})." if k.get("fewest") else ")."),
                       "other": "המעגל שצויר (כפי שנקרא) נותן פונקציה אחרת: " +
                                "; ".join(f"{n} = {f}" for n, f in r["functions"].items()) + " - לבדיקה ידנית.",
                       "none": "המעגל שצויר נקרא, אך לא נמצאה בו פונקציה לוגית (צומת צף או קצר בחלק מהמצבים, או קריאה "
                               "שגויה) - לבדיקה ידנית.",
                       "no drawing": "לא נמצא מעגל טרנזיסטורים מצויר בדפי השאלה - לבדיקה ידנית."}[r["verdict"]]
                remarks.setdefault(r["exam"], []).append(f"שאלה {q} - מעגל: {txt}")
    for name, lines in remarks.items():
        open(os.path.join(a.circuits, f"{name}_circuit_remarks.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(results, open(os.path.join(a.circuits, "circuit_key.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
