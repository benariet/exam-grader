"""Grade the simulated students (forms/simulate.py) and compare with what they really wrote.

    python -m forms.evaluate sim_dev --sweep   # development set: thresholds compared
    python -m forms.evaluate sim_test          # the test set, measured once -> <sim>/evaluation.json

Measured per cell (the reader's character vs the written one), per answer (the whole string), per
graded item (the tool's automatic score vs the true score: an automatic grade that differs is the
error that matters - 'student loses points' or 'student gains points'), per student (final grade
when nothing was left for a human), and the share of items sent to a human, with the reasons.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from forms.read_form import NUM_RE, FormGrader  # noqa: E402

COURSE_DIR = os.path.join(HERE, "data", "course_elec101")


def true_score(q, written, rel_tol):
    if not written:
        return 0.0
    if q["kind"] == "choice":
        return float(q["points"]) if written == str(q["answer"]) else 0.0
    if not NUM_RE.match(written):
        return 0.0
    v, a = float(written), float(q["answer"])
    return float(q["points"]) if abs(v - a) <= rel_tol * abs(a) + 1e-12 else 0.0


def read_all(fg, sim, course_dir=COURSE_DIR, limit=None):
    """read every simulated student once: [(truth, exam id, pages, cells, problems)]"""
    truth = json.load(open(os.path.join(course_dir, sim, "truth.json"), encoding="utf-8"))[:limit]
    data, t0 = [], time.time()
    for k, t in enumerate(truth):
        imgs = [Image.open(os.path.join(course_dir, f)) for f in t["files"]]
        ex_id, pages, cells, problems = fg.read(imgs)
        data.append((t, ex_id, {p: None for p in pages}, cells, problems))
        if (k + 1) % 20 == 0:
            print(f"read {k + 1}/{len(truth)} students, {time.time() - t0:.0f}s", flush=True)
    return data, (time.time() - t0) / max(1, len(truth))


def metrics(fg, data, auto_conf=None, roster=None):
    """roster: IDs to match the students against (None: the grader's own, [] : none)"""
    import forms.read_form as R
    old, old_roster = R.AUTO_CONF, fg.roster
    if auto_conf is not None:
        R.AUTO_CONF = auto_conf
    if roster is not None:
        fg.roster = roster
    try:
        return _metrics(fg, data)
    finally:
        R.AUTO_CONF = old
        fg.roster = old_roster


def sim_roster(data, extra=80, seed=0):
    """a class roster for a simulated set: every student's ID and `extra` students who did not come"""
    import random
    from forms.read_form import id_valid
    rng = random.Random(seed)
    ids = [t["id"] for t, *_ in data]
    while len(ids) < len(data) + extra:
        s = "".join(str(rng.randrange(10)) for _ in range(9))
        if id_valid(s) and s not in ids:
            ids.append(s)
    return ids


def _metrics(fg, data):
    classes = fg.reader.classes
    rel_tol = float(fg.course.get("rel_tol", 0.01))
    C = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    reasons = collections.Counter()
    errors, students = [], []
    for t, ex_id, pages, cells, problems in data:
        res = fg.score(ex_id, pages, cells, problems)
        keys = (t["exam"], t["mode"])
        C["students"] += 1
        if res["exam"] != t["exam"] or len(res.get("pages_read", [])) != len(t["files"]):
            C["pages_not_identified"] += 1
            for kk in keys:
                by[kk]["pages_not_identified"] += 1
            continue
        ex = fg.exams[t["exam"]]
        sid = res.get("student_id", {})
        C["id_accepted"] += bool(sid.get("valid"))
        C["id_right"] += sid.get("id") == t["id"] and bool(sid.get("valid"))
        C["id_wrong_but_accepted"] += bool(sid.get("valid")) and sid.get("id") != t["id"]
        if "id" in cells:
            P = cells["id"]["P"]
            for i, ch in enumerate(t["id"]):
                C["cells"] += 1
                C["cells_right"] += classes[int(P[i].argmax())] == ch
        tot_true, tot_tool, complete = 0.0, 0.0, True
        for q in ex["questions"]:
            ti = t["items"][q["id"]]
            r = next(x for x in res["items"] if x["id"] == q["id"])
            if q["kind"] == "open":
                complete = False
                continue
            written = ti["written"]
            ts = true_score(q, written, rel_tol)
            tot_true += ts
            if q["id"] in cells and not ti.get("crossed"):
                P = cells[q["id"]]["P"]
                for i in range(len(P)):
                    want = written[i] if i < len(written) else " "
                    got = classes[int(P[i].argmax())]
                    C["cells"] += 1
                    C["cells_right"] += got == want
                    if got != want:
                        C[f"cell_err:{want}->{got}"] += 1
            C["items"] += 1
            C["answer_read_right"] += (r["read"] == written) and not ti.get("crossed")
            for kk in keys:
                by[kk]["items"] += 1
            if r["route"] == "auto":
                C["auto"] += 1
                for kk in keys:
                    by[kk]["auto"] += 1
                tot_tool += r["score"]
                if abs(r["score"] - ts) > 1e-9:
                    kind = "student_loses" if r["score"] < ts else "student_gains"
                    C["auto_wrong"] += 1
                    C[kind] += 1
                    for kk in keys:
                        by[kk]["auto_wrong"] += 1
                    errors.append({"student": t["student"], "item": q["id"], "written": written, "read": r["read"],
                                   "tool": r["score"], "true": ts, "conf": r["confidence"], "mode": t["mode"],
                                   "crossed": bool(ti.get("crossed"))})
            else:
                complete = False
                C["review"] += 1
                for kk in keys:
                    by[kk]["review"] += 1
                reason = r["reason"].split(" (")[0]
                if re.match(r"\d+% of the reading probability", reason):
                    reason = "part of the reading probability is not a number"
                if reason.startswith("'"):
                    reason = "the reading is not a number"
                if ti.get("crossed"):
                    reason = "crossed-out box: " + reason
                reasons[reason] += 1
                tot_tool += ts
        students.append({"student": t["student"], "true": tot_true, "auto_points": res.get("auto_points"),
                         "auto_max": res.get("auto_max"), "complete_numeric": complete or ex["questions"][-1]["kind"] == "open"})
    n_items = max(1, C["items"])
    return {"counts": dict(C), "by": {k: dict(v) for k, v in by.items()}, "review_reasons": dict(reasons),
            "auto_errors": errors, "students": students,
            "rates": {"cell_accuracy": C["cells_right"] / max(1, C["cells"]),
                      "answer_read_right": C["answer_read_right"] / n_items,
                      "auto_share": C["auto"] / n_items,
                      "auto_wrong": C["auto_wrong"], "student_loses": C["student_loses"], "student_gains": C["student_gains"],
                      "auto_wrong_share_of_auto": C["auto_wrong"] / max(1, C["auto"]),
                      "id_right": C["id_right"] / max(1, C["students"]),
                      "id_wrong_but_accepted": C["id_wrong_but_accepted"]}}


def evaluate(sim="sim_test", course_dir=COURSE_DIR, limit=None, sweep=False):
    fg = FormGrader(course_dir)
    data, sec = read_all(fg, sim, course_dir, limit)
    out = metrics(fg, data)
    out["seconds_per_student"] = sec
    if sweep:
        out["sweep"] = {str(c): metrics(fg, data, c)["rates"] for c in (0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995)}
    with open(os.path.join(course_dir, sim, "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    sim = sys.argv[1] if len(sys.argv) > 1 else "sim_test"
    o = evaluate(sim, sweep="--sweep" in sys.argv)
    print(json.dumps({k: o[k] for k in ("rates", "counts", "review_reasons")}, ensure_ascii=False, indent=1))
    print("automatic errors:", json.dumps(o["auto_errors"][:20], ensure_ascii=False, indent=1))
