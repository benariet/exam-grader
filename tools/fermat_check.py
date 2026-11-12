"""Does the vision model find what is wrong in a handwritten solution? Measured on FERMAT (ai4bharat/FERMAT on Hugging
Face, CC BY 4.0; arXiv 2501.07244): 2,244 photos of math solutions written out by hand - the question at the top, then
a solution - 1,904 of them with a planted mistake (1 computational: a wrong number, a slip in a step, a miscopied
expression; 2 conceptual: a wrong rule, formula or fact; 3 notation: a wrong operator, a lost bracket; 4
presentation: wrong units, a wrong term, an answer not in the asked form) and 340 changed only on the surface (5: a
variable renamed, a step left out, a needless sentence added) - each with the correct solution and the mistake's
place marked. Public data (not our students'): read on the GPU server.

    python tools/fermat_check.py pick --data <the parquet files' folder> --out pick.json [--n 340] [--seed 1]
        -> the rows checked: every one without a mistake and as many with one, at random over the kinds
    python tools/fermat_check.py run --data <folder> --pick pick.json --out runs.json [--shard 0/8]
                                     [--modes key,nokey,read,text]
        key:   the photo, the question and a correct solution (as a grader with the key would) -> mistake or not, where
        nokey: the photo and the question only
        read:  the photo -> the solution written out (LaTeX)
        text:  that writing (no photo), the question and the correct solution -> mistake or not, where
    python tools/fermat_check.py score runs.json [more.json ...] --data <folder>
        -> per way: right / wrong against the marks, by kind of mistake; how sure a 'no mistake' is (the share of them
           really without one); the mistake's place found; the writing against the marked solution; and with a
           different kind of judge - the final answer of the writing against the correct solution's by the symbolic /
           numeric checker - what both ways and it agree on (decided) and what is left to a human
"""
import argparse
import collections
import difflib
import glob
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

META = ["new_custom_id", "grade", "domain_code", "subdomain_code", "handwriting_style", "image_quality", "rotation",
        "new_pert_id", "orig_q", "orig_a", "pert_a", "pert_reasoning", "has_error"]
KINDS = {1: "computational", 2: "conceptual", 3: "notation", 4: "presentation", 5: "no mistake (surface change)"}

ASK = ('Answer with one line of JSON: {{"mistake": true or false, "where": "<the wrong part, as the student wrote it; '
       'empty if none>"}}. Write nothing else.')
KEY_PROMPT = ("The image is a student's handwritten solution to a math question (the question may be written at the "
              "top).\nThe question: {q}\nA correct solution: {a}\nCheck the student's solution against the correct "
              "one. Is there a mathematical mistake in it - a wrong number, sign, operator, step, formula, fact, unit "
              "or final answer? A different variable name, a different but equivalent form, a skipped obvious step or "
              "a different correct method is not a mistake. " + ASK)
NOKEY_PROMPT = ("The image is a student's handwritten solution to a math question (the question may be written at the "
                "top).\nThe question: {q}\nCheck the student's solution carefully, step by step. Is there a "
                "mathematical mistake in it - a wrong number, sign, operator, step, formula, fact, unit or final "
                "answer? A different variable name, a different but equivalent form, a skipped obvious step or a "
                "different correct method is not a mistake. " + ASK)
READ_PROMPT = ("The image is a student's handwritten solution to a math question; the question is written at the top. "
               "Write out the student's solution (not the question) exactly as written, the mathematics in LaTeX. Do "
               "not correct anything. Write nothing else.")
TEXT_PROMPT = ("A student's handwritten solution to a math question, as transcribed:\n{t}\nThe question: {q}\nA correct "
               "solution: {a}\nCheck the student's solution against the correct one. Is there a mathematical mistake "
               "in it - a wrong number, sign, operator, step, formula, fact, unit or final answer? A different variable "
               "name, a different but equivalent form, a skipped obvious step, a different correct method or a "
               "transcription that is merely unclear is not a mistake. " + ASK)
THINK_PROMPT = ("A student's handwritten solution to a math question, as transcribed:\n{t}\nThe question: {q}\nA "
                "correct solution: {a}\nCheck the student's computations, the rules and facts used and the final "
                "answer against the correct solution (a different variable name, an equivalent form, a skipped obvious "
                "step, a different correct method or an unclear transcription is not a mistake). Write at most 5 short "
                "lines - only what is wrong or doubtful, not every line - then, on the last line, only the JSON "
                '{{"mistake": true or false, "where": "<the first wrong part, as the student wrote it; empty if none>"}}.')
VERDICT = re.compile(r'"?mistake"?\s*:\s*"?(true|false|yes|no)"?', re.I)
WHERE = re.compile(r'"where"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)
RED = re.compile(r"\\textcolor\{red\}\{")


def kind(pert_id) -> int:
    """FERMAT's perturbation id (1.1 .. 5.3) -> its kind (1..5)"""
    try:
        return int(float(pert_id))
    except (TypeError, ValueError):
        return 0


def verdict(text: str):
    """the model's line -> (mistake: True/False/None, where); the last such line counts (the 'think' way writes its
    checking first)"""
    ms = list(VERDICT.finditer(text or ""))
    v = None if not ms else ms[-1].group(1).lower() in ("true", "yes")
    ws = list(WHERE.finditer(text or ""))
    return v, (ws[-1].group(1).replace('\\"', '"').strip() if ws else "")


def red_parts(pert_a: str):
    """what FERMAT marked as changed: the contents of every \\textcolor{red}{...} (braces matched)"""
    out = []
    for m in RED.finditer(pert_a or ""):
        i, depth = m.end(), 1
        while i < len(pert_a) and depth:
            depth += {"{": 1, "}": -1}.get(pert_a[i], 0)
            i += 1
        out.append(pert_a[m.end():i - 1])
    return out


def unmark(pert_a: str) -> str:
    """the marked solution as written (the red marks' wrappers taken off)"""
    s = pert_a or ""
    for part in red_parts(s):
        s = s.replace("\\textcolor{red}{" + part + "}", part, 1)
    return s


LATEX_WORDS = re.compile(r"\\(?:textcolor|text|mathrm|left|right|cdot|times|frac|dfrac|tfrac|sqrt|quad|qquad|,|;|!|"
                         r"displaystyle|begin|end|aligned|align|array|boxed|red|limits)")


def core(s: str) -> str:
    """for comparing what is written: letters, digits and operators only (LaTeX commands, spaces, brackets dropped)"""
    s = LATEX_WORDS.sub("", (s or "").lower().replace("\\times", "*").replace("\\cdot", "*").replace("×", "*")
                        .replace("−", "-").replace("÷", "/"))
    s = re.sub(r"\\[a-z]+", lambda m: m.group()[1:], s)
    return re.sub(r"[^0-9a-z+\-*/=^<>.%]", "", s)


def found_place(where: str, pert_a: str) -> bool | None:
    """the model's 'where' against the marked change: one of the red parts in it (or it in a red part, when it says
    enough - 3 marks or more). None: nothing marked or nothing said."""
    reds = [c for c in (core(p) for p in red_parts(pert_a)) if c]
    reds = [c for c in reds if len(c) >= 2] or reds  # (a lone letter is in almost anything said)
    w = core(where)
    if not reds or not w:
        return None
    return any(r in w or (len(w) >= 3 and w in r) for r in reds)


def similarity(read: str, pert_a: str) -> float:
    """the writing against the marked solution (both cut to letters, digits and operators): 0..1"""
    a, b = core(read), core(unmark(pert_a))
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


MATH = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|\$(.*?)\$|\\begin\{(?:align\*?|aligned|equation\*?)\}"
                  r"(.*?)\\end\{(?:align\*?|aligned|equation\*?)\}", re.S)
NOISE = re.compile(r"\\(?:text|mathrm|textbf|operatorname)\s*\{[^{}]*\}|₹|\bRs\.?|\\quad|\\qquad|\\[,;!]|&|\\\\|"
                   r"\\(?:therefore|because|implies|Rightarrow|Longrightarrow|hline)")


def final_of(text: str) -> str:
    """a solution's final answer, roughly: the last piece of mathematics that says something, the side after its last
    '=' (words, units and money signs off); '' when there is none"""
    segs = [next(g for g in m.groups() if g is not None) for m in MATH.finditer(text or "")]
    for seg in reversed(segs):
        s = NOISE.sub(" ", seg)
        s = s.split("=")[-1] if "=" in s else s
        s = re.sub(r"\\(?:approx|simeq)", "", s).strip().rstrip(".,;: ").strip()
        if re.search(r"[0-9a-zA-Z]", s) and not re.search(r"\\(?:begin|end)\b", s):
            return s
    return ""


def final_check(read: str, solution: str) -> str:
    """the written-out solution's final answer against the correct solution's, by the symbolic / numeric checker (a
    different kind of judge than the vision model): 'same' | 'differ' | 'unknown'"""
    from grader.numeric import compare_numeric
    from grader.symbolic import compare_formula
    st, ref = final_of(read), final_of(solution)
    if not st or not ref:
        return "unknown"
    try:
        v = compare_numeric(st, ref, rel_tol=0.005)
        if v.status == "correct":
            return "same"
        if v.status in ("wrong", "prefix_error", "factor_2pi"):
            return "differ"
    except Exception:  # noqa: BLE001
        pass
    def formula(s):
        try:
            return compare_formula(s, ref).status
        except Exception:  # noqa: BLE001
            return "parse_error"
    got = formula(st)
    if got == "not_equivalent":                       # (the student's own letters for the key's, used throughout:
        for s in renamed(st, ref):                    #  'u' for 'x' - not a mistake)
            if formula(s) == "equivalent":
                return "same"
    return {"equivalent": "same", "not_equivalent": "differ"}.get(got, "unknown")


LETTER = re.compile(r"(?<![\\a-zA-Z])[a-zA-Z](?![a-zA-Z])")


def renamed(st: str, ref: str):
    """the student's final answer with its single letters put for the key's - every one-to-one way, when both use the
    same number of letters (at most 3) and not the same ones"""
    import itertools
    a, b = sorted(set(LETTER.findall(st))), sorted(set(LETTER.findall(ref)))
    if not a or len(a) != len(b) or len(a) > 3 or a == b:
        return []
    out = []
    for perm in itertools.permutations(b):
        table = dict(zip(a, perm))
        out.append(LETTER.sub(lambda m: table.get(m.group(), m.group()), st))
    return out


def load_meta(data: str):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(os.path.join(data, "*.parquet"))):
        for r in pq.read_table(f, columns=META).to_pylist():
            r["_file"] = os.path.basename(f)
            rows.append(r)
    return rows


def pick(a):
    rows = load_meta(a.data)
    rnd = random.Random(a.seed)
    clean = [r for r in rows if not r["has_error"]]
    wrong = [r for r in rows if r["has_error"]]
    n = min(a.n, len(clean), len(wrong))
    chosen = rnd.sample(clean, n) + rnd.sample(wrong, n)
    ids = sorted(r["new_custom_id"] for r in chosen)
    json.dump(ids, open(a.out, "w"))
    print(len(ids), "rows:", collections.Counter(KINDS[kind(r["new_pert_id"])] for r in chosen))


def images_of(data: str, ids):
    """{id: PIL image} for the rows asked for (the photos read file by file)"""
    import pyarrow.parquet as pq
    from PIL import Image
    want, out = set(ids), {}
    for f in sorted(glob.glob(os.path.join(data, "*.parquet"))):
        t = pq.read_table(f, columns=["new_custom_id", "image"]).to_pylist()
        for r in t:
            if r["new_custom_id"] in want:
                out[r["new_custom_id"]] = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
    return out


def shrink(img, side: int):
    if max(img.size) > side:
        img = img.copy()
        img.thumbnail((side, side))
    return img


def run(a):
    import torch
    from PIL import Image
    from recog.hebrew_vlm import HebrewRecognizer
    ids = json.load(open(a.pick))
    k, n = (int(x) for x in a.shard.split("/"))
    ids = ids[k::n]
    meta = {r["new_custom_id"]: r for r in load_meta(a.data)}
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    if a.reuse and os.path.isfile(a.reuse):            # (what an earlier run has: the writing out is not asked again)
        for i, res in json.load(open(a.reuse, encoding="utf-8")).items():
            if i in ids:
                out[i] = {**res, **out.get(i, {})}
    modes = [m for m in a.modes.split(",") if m]
    need = {"key": ["key"], "nokey": ["nokey"], "read": ["read"], "text": ["read", "text"], "think": ["read", "think"]}
    todo = [i for i in ids if any(n not in out.get(i, {}) for m in modes for n in need[m])]
    if not todo:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print("all done", flush=True)
        return
    photo = [i for i in todo if any(m not in out.get(i, {}) for m in ("key", "nokey", "read") if m in
                                    {n for x in modes for n in need[x]})]
    imgs = images_of(a.data, photo) if photo else {}
    rd = HebrewRecognizer(model_dir=a.vlm, dtype=torch.bfloat16 if torch.cuda.is_available() and
                          torch.cuda.is_bf16_supported() else None)
    blank = Image.new("RGB", (32, 32), "white")      # (the 'text' way is asked about text: an empty picture with it)

    def ask(img, prompt, tokens):
        return rd.recognize(img, n_best=1, max_new_tokens=tokens, prompt=prompt)[0].text

    for i in todo:
        r, res = meta[i], dict(out.get(i, {}))
        img = shrink(imgs[i], a.side) if i in imgs else None
        q, sol = r["orig_q"].strip(), r["orig_a"].strip()
        if "key" in modes and "key" not in res:
            res["key"] = ask(img, KEY_PROMPT.format(q=q, a=sol), 120)
        if "nokey" in modes and "nokey" not in res:
            res["nokey"] = ask(img, NOKEY_PROMPT.format(q=q), 120)
        if any(m in modes for m in ("read", "text", "think")) and "read" not in res:
            res["read"] = ask(img, READ_PROMPT, a.tokens)
        if "text" in modes and "text" not in res:
            res["text"] = ask(blank, TEXT_PROMPT.format(t=res["read"], q=q, a=sol), 120)
        if "think" in modes and "think" not in res:
            res["think"] = ask(blank, THINK_PROMPT.format(t=res["read"], q=q, a=sol), a.think_tokens)
        out[i] = res
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(i, {m: verdict(v)[0] for m, v in res.items() if m != "read"}, flush=True)


def tally(rows, runs, mode):
    """one way's verdicts against the marks -> counts"""
    c = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    for i, res in runs.items():
        if mode not in res or i not in rows:
            continue
        r = rows[i]
        v, where = verdict(res[mode])
        truth = bool(r["has_error"])
        key = "none" if v is None else ("tp" if v and truth else "fn" if truth else "fp" if v else "tn")
        c[key] += 1
        by_kind[kind(r["new_pert_id"])][key] += 1
        if v and truth:
            f = found_place(where, r["pert_a"])
            c["place_" + ("none" if f is None else "yes" if f else "no")] += 1
    return c, by_kind


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "-"


def combine(rows, runs, second):
    """key and a second way (text / think) together; then with the final answer's check (the symbolic / numeric
    checker: a different kind of judge than the vision model) -> (agreement counts, decided counts)"""
    both, fin = collections.Counter(), collections.Counter()
    for i, res in runs.items():
        if not ("key" in res and second in res):
            continue
        truth = bool(rows[i]["has_error"])
        v1, v2 = verdict(res["key"])[0], verdict(res[second])[0]
        if v1 is not None and v1 == v2:
            both[("mistake" if v1 else "clean") + ("_right" if v1 == truth else "_wrong")] += 1
        else:
            both["differ"] += 1
        if "read" not in res:
            continue
        f = final_check(res["read"], rows[i]["orig_a"])
        fin["final_" + f + ("_mistake" if truth else "_clean")] += 1
        if v1 is False and v2 is False and f == "same":
            fin["clean_" + ("wrong" if truth else "right")] += 1
        elif v1 and v2 and f == "differ":
            fin["mistake_" + ("right" if truth else "wrong")] += 1
        else:
            fin["left"] += 1
    return both, fin


def score(a):
    runs = {}
    for f in a.runs:
        runs.update(json.load(open(f, encoding="utf-8")))
    rows = {r["new_custom_id"]: r for r in load_meta(a.data)}
    n_true = sum(1 for i in runs if rows[i]["has_error"])
    print(f"{len(runs)} solutions: {n_true} with a mistake, {len(runs) - n_true} without")
    report = {}
    for mode in ("key", "nokey", "text", "think"):
        c, by_kind = tally(rows, runs, mode)
        if not sum(c[k] for k in ("tp", "fn", "fp", "tn", "none")):
            continue
        tp, fn, fp, tn, none = (c[k] for k in ("tp", "fn", "fp", "tn", "none"))
        print(f"\n== {mode}")
        print("  right:", pct(tp + tn, tp + fn + fp + tn + none), "| no answer:", none)
        print("  mistakes found:", pct(tp, tp + fn), "| clean ones called clean:", pct(tn, tn + fp))
        print("  'no mistake' said - really none:", pct(tn, tn + fn), "| 'mistake' said - really one:", pct(tp, tp + fp))
        print("  its place named (of the mistakes found, where it said something):",
              pct(c["place_yes"], c["place_yes"] + c["place_no"]), "| place not said:", c["place_none"])
        for k in sorted(by_kind):
            b = by_kind[k]
            got = b["tp"] + b["tn"]
            print(f"    {KINDS.get(k, k)}: right {pct(got, sum(b.values()))}")
        report[mode] = dict(c)
    both, fin = {}, {}
    for second in ("text", "think"):
        if not any(second in res for res in runs.values()):
            continue
        bo, fi = combine(rows, runs, second)
        both[second], fin[second] = dict(bo), dict(fi)
        print(f"\n== key and {second} agreeing (the same model both times: a hint only, not a decision)")
        print("  both 'no mistake' - really none:", pct(bo["clean_right"], bo["clean_right"] + bo["clean_wrong"]))
        print("  both 'mistake' - really one:", pct(bo["mistake_right"], bo["mistake_right"] + bo["mistake_wrong"]))
        print("  they differ:", pct(bo["differ"], sum(bo.values())))
        n = sum(fi[k] for k in ("clean_right", "clean_wrong", "mistake_right", "mistake_wrong", "left"))
        print("== and the final answer (written out, by the symbolic checker) against the correct solution's")
        for t in ("mistake", "clean"):
            print(f"  with a {t}: same {fi[f'final_same_{t}']}, differ {fi[f'final_differ_{t}']}, "
                  f"unknown {fi[f'final_unknown_{t}']}")
        print(f"  decided 'no mistake' (key, {second}: none; the final answer the same) - really none:",
              pct(fi["clean_right"], fi["clean_right"] + fi["clean_wrong"]))
        print(f"  decided 'mistake' (key, {second}: one; the final answer differs) - really one:",
              pct(fi["mistake_right"], fi["mistake_right"] + fi["mistake_wrong"]))
        print("  left to a human:", pct(fi["left"], n))
    sims = sorted(similarity(res["read"], rows[i]["pert_a"]) for i, res in runs.items() if "read" in res)
    if sims:
        print(f"\n== the writing against the marked solution: median {sims[len(sims) // 2]:.2f}, "
              f"0.9 or more in {pct(sum(s >= 0.9 for s in sims), len(sims))}, under 0.6 in "
              f"{pct(sum(s < 0.6 for s in sims), len(sims))}")
    if a.out:
        json.dump({"modes": report, "both": both, "final": fin, "similarity": sims}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=340)
    p.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--pick", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--modes", default="key,nokey,read,text")
    r.add_argument("--side", type=int, default=1600, help="the photo's longer side at most (pixels)")
    r.add_argument("--tokens", type=int, default=700, help="the writing out's length at most")
    r.add_argument("--think-tokens", type=int, default=350, help="the 'think' way's length at most")
    r.add_argument("--reuse", default="", help="an earlier run's results: what it has is not asked again")
    r.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="")
    a = ap.parse_args()
    {"pick": pick, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
