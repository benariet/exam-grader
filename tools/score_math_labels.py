"""Score the formula readers against a person's labels (tools/label_math.py -> math_labels.json): per reader, the share
of readable lines read right (the same once both are written plainly: no spaces, braces, \\left/\\right, \\cdot as ·,
\\sqrt as √ ...) and the mean character error; the unreadable lines apart.

    python tools/score_math_labels.py <math_labels.json> [--merged <exam>=<merged json> ...]"""
import argparse
import json
import re
import sys
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def plain(t: str) -> str:
    """a reading or a person's correction -> one plain form to compare"""
    t = t or ""
    t = re.sub(r"\\(?:left|right|displaystyle|mathrm|mathit|mathbf|text|operatorname)\b", "", t)
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", t)
    for a, b in (("\\cdot", "·"), ("\\times", "·"), ("*", "·"), ("\\sqrt", "√"), ("sqrt", "√"), ("\\mu", "µ"), ("μ", "µ"),
                 ("\\Delta", "Δ"), ("\\approx", "≈"), ("\\rightarrow", "→"), ("\\to", "→"), ("\\Rightarrow", "⇒"),
                 ("\\gamma", "γ"), ("\\phi", "φ"), ("\\Phi", "Φ"), ("\\Omega", "Ω"), ("\\lvert", "|"), ("\\rvert", "|")):
        t = t.replace(a, b)
    t = re.sub(r"e([+-]?\d+)", r"·10^\1", t)            # (1.5e-12 as typed = 1.5·10^-12 as read)
    return re.sub(r"[\s{}\\]", "", t)


def cer(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b) / max(1, len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--merged", action="append", default=[], help="<exam>=<merged json>: also score what the merge chose")
    args = ap.parse_args()
    labels = json.load(open(args.labels, encoding="utf-8"))
    merged = {}
    for m in args.merged:
        ex, path = m.split("=", 1)
        merged[ex] = json.load(open(path, encoding="utf-8"))
    readable = [x for x in labels if not x["unreadable"] and not x.get("crossed") and x.get("truth")]
    crossed = sum(1 for x in labels if x.get("crossed"))
    print(f"{len(labels)} labelled lines: {len(readable)} readable, {crossed} crossed out, "
          f"{sum(1 for x in labels if x['unreadable'])} unreadable")
    rows = {"qwen": [], "mumer": [], "merge": []}
    for x in readable:
        truth = plain(x["truth"])
        qwen, mumer = (x["a"], x["b"]) if x["a_is"] == "qwen" else (x["b"], x["a"])
        rows["qwen"].append((plain(qwen) == truth, cer(plain(qwen), truth)))
        rows["mumer"].append((plain(mumer) == truth, cer(plain(mumer), truth)))
        data = merged.get(str(x["exam"]))
        if data:
            seg = next((s for segs in data[x["page"] - 1]["lines"] for s in segs if s["box"] == x["box"]), None)
            if seg and seg.get("math2"):
                pick = seg["math2"][0][0]
                rows["merge"].append((plain(pick) == truth, cer(plain(pick), truth)))
    for name, r in rows.items():
        if r:
            print(f"{name:6s}: right {sum(a for a, _ in r)} of {len(r)} ({100 * sum(a for a, _ in r) / len(r):.0f}%), "
                  f"mean character error {100 * sum(c for _, c in r) / len(r):.0f}%")
    by_exam = {}
    for x in labels:
        by_exam.setdefault(x["exam"], [0, 0, 0])[2 if x["unreadable"] else 1 if x.get("crossed") else 0] += 1
    print("per exam (readable, crossed out, unreadable):", by_exam)


if __name__ == "__main__":
    main()
