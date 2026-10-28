"""How much of a hand-transcribed page the page reader gets right - words and formulas, regardless
of how the page was cut into lines and pieces.

    python tools/score_page.py data/my_scans/tal_notebook.pdf      # truth: data/my_scans/tal_notebook_truth.json

Truth file (JSON): {"hebrew": [lines], "english": [lines], "math": [formulas, LaTeX]}.
Hebrew / English: the words of every piece read as that language (best reading, punctuation off,
English in lower case) against the truth's words, as multisets - a word counts once per time it is
written. Math: every truth formula must appear (spaces, braces and \\left/\\right ignored, each piece
of the page used once) inside one of the pieces read as math. Prints the scores and what was missed.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

_HEB_WORD = re.compile(r"[א-ת]+")
_EN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def norm_tex(t: str) -> str:
    t = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", t)
    t = re.sub(r"\\(?:vert|mid)\b", "|", t).replace("\\pi", "π").replace("\\sqrt", "√")
    return re.sub(r"[\s{}$]", "", t)


def words(texts, pattern, lower=False):
    return Counter(w.lower() if lower else w for t in texts for w in pattern.findall(t))


def score(pages, truth):
    by_kind = {"hebrew": [], "english": [], "math": []}
    review = total = 0
    for p in pages:
        for segs in p["lines"]:
            for s in segs:
                by_kind[s.kind].append(s.nbest[0][0] if s.nbest else "")
                total += 1
                review += int(s.p_kind < 0.8)
    out = {"pieces": total, "pieces_flagged_for_review": review}
    for kind, pat, lower in (("hebrew", _HEB_WORD, False), ("english", _EN_WORD, True)):
        want, got = words(truth.get(kind, []), pat, lower), words(by_kind[kind], pat, lower)
        hit = sum((want & got).values())
        out[kind] = {"words_right": hit, "words": sum(want.values()),
                     "missed": sorted((want - got).elements()), "extra": sorted((got - want).elements())}
    pieces = [norm_tex(t) for t in by_kind["math"]]
    found, missed = 0, []
    for f in truth.get("math", []):
        key = norm_tex(f)
        for i, p in enumerate(pieces):
            if key and key in p:
                pieces[i] = p.replace(key, "\0", 1)            # each place on the page counts once
                found += 1
                break
        else:
            missed.append(f)
    out["math"] = {"formulas_right": found, "formulas": len(truth.get("math", [])), "missed": missed}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--truth", default=None, help="default: <page>_truth.json")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    truth_path = a.truth or os.path.splitext(a.page)[0] + "_truth.json"
    truth = json.load(open(truth_path, encoding="utf-8"))
    from read_page import read_pages
    res = score(read_pages(a.page, a.dpi), truth)
    h, e, m = res["hebrew"], res["english"], res["math"]
    print(f"Hebrew words right:  {h['words_right']}/{h['words']}   missed: {' '.join(h['missed'])}")
    print(f"English words right: {e['words_right']}/{e['words']}   missed: {' '.join(e['missed'])}")
    print(f"formulas right:      {m['formulas_right']}/{m['formulas']}   missed: {', '.join(m['missed'])}")
    print(f"pieces flagged for review: {res['pieces_flagged_for_review']}/{res['pieces']}")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
