"""Grade a scanned or photographed answer page against a rubric - fully local.

    python grade_page.py page.jpg rubric.json          # -> results/<name>_grade.txt (+ .json)
    python grade_page.py page.jpg rubric2.json --reuse # grade again with a changed rubric: the page's
                                                       # reading (results/<name>_reading.json) is reused

The page is read exactly like read_page.py (a phone photo is cut out and flattened first), then
grader/page.py finds every rubric item's answer and grades it over the readings' probabilities.
The report gives the points the tool could grade on its own, and the items a human has to look
at, with the reason - an exercise the tool cannot find or read confidently, or an open answer.
See grader/page.py for the rubric format (data/my_scans/silly_test_rubric.json is an example).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fmt(x):
    return f"{x:g}" if x is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("rubric")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--photo", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--language", choices=("hebrew", "english"), default="hebrew",
                    help="the exam's language: hebrew (Hebrew + English + math) or english (English + math)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--reuse", action="store_true", help="reuse results/<name>_reading.json instead of reading again")
    args = ap.parse_args()

    from types import SimpleNamespace
    from demo_mixed import pretty
    from grader.page import grade_page, load_rubric

    rubric = load_rubric(args.rubric)
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.path))[0]
    cache = os.path.join(args.out, f"{name}_reading.json")
    if args.reuse and os.path.isfile(cache):
        with open(cache, encoding="utf-8") as fh:
            saved = json.load(fh)
    else:
        from read_page import read_pages
        pages = read_pages(args.path, args.dpi, photo=args.photo, scan=args.scan, language=args.language)
        saved = [{"kind": s.kind, "box": [int(v) for v in s.box], "p_kind": float(s.p_kind),
                  "note": getattr(s, "note", ""), "boxed": bool(getattr(s, "boxed", False)),
                  "nbest": [[t, float(lp)] for t, lp in s.nbest]}
                  | {"page": pno}
                 for pno, p in enumerate(pages, 1) for line in p["lines"] for s in line]
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(saved, fh, ensure_ascii=False, indent=0)
    segments = [SimpleNamespace(**s) for s in saved]
    res = grade_page(rubric, segments)

    out = [f"{res['name']}  ({os.path.basename(args.path)})", ""]
    for r in res["items"]:
        found = f"  read: {pretty(r['found'])}" if r["found"] else ""
        if r["route"] == "auto":
            out.append(f"  {r['id']:>4}  {fmt(r['score']):>3} / {fmt(r['points'])}   auto, confidence {r['confidence']:.0%}{found}")
        else:
            guess = f"  (tool's guess {fmt(r['score'])})" if r["score"] is not None else ""
            out.append(f"  {r['id']:>4}    ? / {fmt(r['points'])}   to a human: {r['reason']}{guess}{found}")
    out += ["", f"graded by the tool: {fmt(res['auto_points'])} of {fmt(res['auto_max'])} points",
            f"for a human:        {fmt(res['review_points'])} points "
            f"({', '.join(r['id'] for r in res['items'] if r['route'] != 'auto')})",
            f"total so far:       {fmt(res['auto_points'])} + what the human gives, out of {fmt(res['max_points'])}"]
    print("\n".join(out))
    with open(os.path.join(args.out, f"{name}_grade.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    with open(os.path.join(args.out, f"{name}_grade.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("wrote", os.path.relpath(os.path.join(args.out, f"{name}_grade.txt"), HERE))


if __name__ == "__main__":
    main()
