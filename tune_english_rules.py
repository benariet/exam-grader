"""Tune the Hebrew / English / math chunk decision of recog/mixed_reader.py offline - on development
sets only, never on the test sets (data/mixed_eval_en2, data/mixed_eval3).

    python tools_extract_chunks.py mixed_dev_en  results/chunks_mixed_dev_en.json   # slow: all three readers
    python tools_extract_chunks.py mixed_eval_en results/chunks_mixed_eval_en.json
    python tools_extract_chunks.py mixed_eval    results/chunks_mixed_eval.json
    python tune_english_rules.py mixed_dev_en=results/chunks_mixed_dev_en.json \\
           mixed_eval_en=results/chunks_mixed_eval_en.json mixed_eval=results/chunks_mixed_eval.json

Add --quick to score only the rule sets V1 / V3 / SHIPPED (seconds instead of minutes).
The chunk dumps of the three development sets are kept in results/ (regenerate with the first commands).

decide_line() mirrors MixedReader.layout_checked up to the whole-run check (MixedReader._whole_runs,
not simulated: it needs new readings, and a run it changes always goes to review); a line's layout is judged exactly like
eval_mixed.py (by ink: every true part's segment holds >= 90 % of its ink and <= 10 % of any other
part's), and it goes to review if any segment's confidence is < 0.8.

Objective - safety first: a line whose layout is wrong and that is NOT sent to review costs 5, a line
sent to review costs 1. Ties: more lines split right, then more English chunks found.
"""
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
HEB = re.compile("[א-ת]")
CAP = 0.5                     # recog.mixed_reader.CONFLICT_CAP
# None = rule off. V1 / V3: the first and the third rule set measured during development (V1 as
# reconstructed here - its code differed in details; V3 is exact)
V1 = {"tiny": 0.6, "econf": 0.80, "econf_doubt": None, "short_ok": True, "agree": 0.5, "alone": 0.9,
      "pen_min": 0.2, "doubt": 0.5, "context": None, "boundary": None, "pk": "classifier", "tiny_en": None,
      "reader_min": 1.0}
V3 = dict(V1, econf=0.90, agree=0.3, pen_min=None, context=0.7, pk="combined")
V4 = dict(V3, econf_doubt=0.8, boundary=0.3, tiny_en="absorb", reader_min=0.0)
SHIPPED = V4                  # = recog/mixed_reader.py
GRID = {"econf": (0.8, 0.9), "econf_doubt": (None, 0.7, 0.8), "short_ok": (True, False), "agree": (0.3, 0.5),
        "alone": (0.9, None), "pen_min": (0.2, None), "doubt": (0.3, 0.5, None), "context": (0.5, 0.7, None),
        "boundary": (None, 0.1, 0.2, 0.3), "pk": ("classifier", "combined"), "tiny_en": (None, "flag", "absorb"),
        "reader_min": (1.0, 0.1, 0.0)}


def prepare(c):
    """static per-chunk signals"""
    toks = c["f_text"].split()
    m_raw = c["f_plausible"] and ("=" in c["f_text"] or len(toks) >= 2) and c["f_conf"] >= 0.9 and c["f_min"] >= 0.5
    h_raw = bool(HEB.search(c["h_text"])) and c["h_conf"] >= 0.85
    c["_m_raw"], c["_m"], c["_h"] = m_raw, m_raw and not h_raw, h_raw and not m_raw
    c["_sure_en"] = c["e_conf"] >= 0.9 and c["e_score"] >= 0.9 and not c["f_plausible"]
    return c


def decide_line(chunks, P):
    """-> [[kind, p_kind, tiny]] per chunk, left to right."""
    out = []
    for c in chunks:
        p = c["p_smooth"]
        kind, pk, noted = ("math" if p >= 0.5 else "hebrew"), max(p, 1 - p), False
        if kind == "hebrew" and c["_m"]:
            noted = True
            kind = "math" if p > 0.15 else kind
        elif kind == "math" and c["_h"]:
            noted = True
            kind = "hebrew" if p < 0.85 else kind
        tiny = c["width"] < P["tiny"] * c["ink_h"]
        # the English reader is slow: the code runs it only where English is possible
        reader_ran = not tiny and (p >= 0.5 or c["p_en"] >= min(P["agree"], P["reader_min"]))
        reader_ok = reader_ran and c["e_min"] >= 0.3 and not c["_m_raw"]
        eng = reader_ok and c["e_conf"] >= P["econf"] and c["e_looks"]
        eng_short = reader_ok and c["e_conf"] >= P["econf"] and c["e_looks_short"]
        eng_doubt = (P["econf_doubt"] is not None and reader_ok and c["e_conf"] >= P["econf_doubt"]
                     and (c["e_looks"] or c["e_looks_short"]))
        if not noted and not tiny:
            pe = c["p_en"]
            if kind == "math":
                if eng:
                    kind, noted = "english", not c["_sure_en"]
                elif eng_doubt:
                    noted = True
            elif kind == "hebrew" and not c["_h"]:
                if pe >= P["agree"] and (eng or (P["short_ok"] and eng_short)):
                    # how sure: the classifier's own probability, or classifier and reader combined as
                    # if independent (the reader writes fluent English for almost anything, so they are not)
                    kind, pk = "english", (pe if P["pk"] == "classifier" else 1 - (1 - pe) * (1 - P["econf"]))
                elif P["alone"] is not None and pe >= P["alone"]:
                    kind, noted = "english", True
                elif P["pen_min"] is not None and eng and pe >= P["pen_min"]:
                    kind, noted = "english", True
                elif (P["doubt"] is not None and pe >= P["doubt"]) or eng_doubt:
                    noted = True
        out.append([kind, min(pk, CAP) if noted else pk, tiny, c["_h"], noted])
    english_at = lambda j: 0 <= j < len(out) and out[j][0] == "english" and not out[j][2]   # noqa: E731
    if P["context"] is not None:          # an unsure chunk next to an English word is English too
        for i, c in enumerate(chunks):
            o = out[i]
            if o[0] == "hebrew" and not o[2] and not o[3] and c["p_en"] >= P["context"] and (english_at(i - 1) or english_at(i + 1)):
                o[0], o[1], o[4] = "english", min(o[1], CAP), True
    if P["tiny_en"] is not None:          # tiny chunks (a letter, a dot) next to English words
        for i in range(len(out)):
            o = out[i]
            if not o[2] or o[0] == "english":
                continue
            left = next((j for j in range(i - 1, -1, -1) if not out[j][2]), None)
            right = next((j for j in range(i + 1, len(out)) if not out[j][2]), None)
            le = left is not None and out[left][0] == "english"
            ri = right is not None and out[right][0] == "english"
            if P["tiny_en"] == "absorb" and le and ri:      # inside an English phrase: part of it
                o[0], o[1] = "english", min(out[left][1], out[right][1], o[1] if o[4] else 1.0)
            elif le or ri:                                    # at its edge: could belong to either side
                o[1], o[4] = min(o[1], CAP), True
    if P["boundary"] is not None:         # a chunk next to an English word that might be English too -> review
        for i, c in enumerate(chunks):
            o = out[i]
            if o[0] != "english" and not o[2] and c["p_en"] >= P["boundary"] and (english_at(i - 1) or english_at(i + 1)):
                o[1], o[4] = min(o[1], CAP), True
    return [o[:3] for o in out]


class Line:
    def __init__(self, page_ink_cols, spans, kinds, chunks):
        self.cum = np.concatenate([[0], np.cumsum(page_ink_cols)])
        self.spans, self.kinds, self.chunks = spans, kinds, chunks
        self.totals = [max(1, self.ink(a, b)) for a, b in spans]

    def ink(self, a, b):
        a, b = max(0, int(a)), min(len(self.cum) - 1, int(b))
        return int(self.cum[b] - self.cum[a]) if b > a else 0

    def judge(self, dec):
        """-> (layout right, sent to review) exactly like eval_mixed.py"""
        segs = []
        for c, (kind, pk, _) in zip(self.chunks, dec):
            x0, x1 = c["x"]
            if segs and segs[-1][0] == kind:
                segs[-1][2] = x1; segs[-1][3] = min(segs[-1][3], pk)
            else:
                segs.append([kind, x0, x1, pk])
        flagged = any(s[3] < 0.8 for s in segs)
        pred = segs[::-1]                               # reading order: right to left, like the truth
        if len(pred) != len(self.spans):
            return False, flagged

        def share(i, s):
            return self.ink(max(s[1], self.spans[i][0]), min(s[2], self.spans[i][1])) / self.totals[i]
        ok = all(s[0] == k and share(i, s) >= 0.9 and all(share(j, s) <= 0.1 for j in range(len(self.spans)) if j != i)
                 for i, (s, k) in enumerate(zip(pred, self.kinds)))
        return ok, flagged


def load(name, path):
    D = os.path.join(HERE, "data", name)
    truth = {t["file"]: t for t in json.load(open(os.path.join(D, "truth.json"), encoding="utf-8"))}
    by_file = defaultdict(list)
    for c in json.load(open(path, encoding="utf-8")):
        by_file[c["file"]].append(prepare(c))
    lines = []
    for f, chunks in sorted(by_file.items()):
        ink = np.asarray(Image.open(os.path.join(D, f)).convert("L")) < 128
        t = truth[f]
        lines.append(Line(ink.sum(0), [s["x"] for s in t["segments"]], [s["kind"] for s in t["segments"]],
                          sorted(chunks, key=lambda c: c["x"][0])))
    return lines


def score(sets, P):
    cost, stats = 0, {}
    for name, lines in sets.items():
        right = danger = reviewed = en_ok = en_n = false_en = n_other = 0
        for ln in lines:
            dec = decide_line(ln.chunks, P)
            ok, flagged = ln.judge(dec)
            right += ok; reviewed += flagged; danger += (not ok) and (not flagged)
            for c, (k, _, tiny) in zip(ln.chunks, dec):
                if tiny:
                    continue
                if c["true"] == "english":
                    en_n += 1; en_ok += k == "english"
                elif c["true"] in ("hebrew", "math"):
                    n_other += 1; false_en += k == "english"
        cost += 5 * danger + reviewed
        n = len(lines)
        stats[name] = {"lines": n, "layout_right": round(right / n, 3), "wrong_not_reviewed": round(danger / n, 3),
                       "reviewed": round(reviewed / n, 3), "english_recall": round(en_ok / max(1, en_n), 3),
                       "false_english": round(false_en / max(1, n_other), 3), "_right": right}
    return cost, stats


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    if not args:
        sys.exit(__doc__)
    sets = {a.split("=", 1)[0]: load(*a.split("=", 1)) for a in args}
    show = lambda st: {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in st.items()}  # noqa: E731
    for name, P in (("v1", V1), ("v3", V3), ("v4 = shipped", SHIPPED)):
        c, st = score(sets, P)
        print(f"{name}:", c, json.dumps(show(st)))
    if "--quick" in sys.argv:
        return
    res = []
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        P = dict(SHIPPED, **dict(zip(keys, vals)))
        cost, st = score(sets, P)
        right = sum(s["_right"] for s in st.values())
        recall = sum(s["english_recall"] for s in st.values())
        res.append((cost, -right, -recall, P, st))
    res.sort(key=lambda r: r[:3])
    for cost, _, _, P, st in res[:12]:
        print(cost, {k: P[k] for k in keys}, json.dumps(show(st)))


if __name__ == "__main__":
    main()
