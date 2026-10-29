"""Fit the points-off estimate to exams whose grades are known (the dev set) - then held fixed for exams judged blind:

    python tools/fit_estimate.py <recheck table.txt> [more tables ...] --out results/estimate_v2.json [--exclude 7,8,9]

A table is tools/recheck_exams.py's output with --grades and --points (its "student  question  points  estimated off
... lecturer took off" rows). The model, per question:

    with the solution's results to reach (coverage):  off = min(points, alpha * points * missing + beta * findings
                                                                       + gamma * points * thin)
    without them (a drawing, a choice, an explanation): off = prior[q] * points  (the dev set's mean share off)

missing: the share of the solution's results not found in the work; findings: sure wrong steps and values unlike the
solution's; thin: 1 / (1 + the checks made) - little work read (most of exam 6's points were lost on work not done). alpha and beta by least squares; reported with each exam left out in turn (leave-one-out) against the
plain baseline - the dev set's mean share off for that question - so that a model no better than the mean says so."""
import argparse
import json
import re
import sys


def rows(path: str) -> list:
    """the estimate table's rows: [{"exam", "q", "points", "findings", "reached", "results", "off"}]"""
    out, on = [], False
    for line in open(path, encoding="utf-8"):
        if line.startswith("student\tquestion\tpoints\testimated"):
            on = True
            continue
        if on:
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or not f[0].strip():
                on = False
                continue
            m = re.match(r"(\d+) of (\d+)", f[6])
            out.append({"exam": f[0], "q": f[1], "points": float(f[2]), "findings": int(f[4]), "checks": int(f[5]),
                        "reached": int(m.group(1)) if m else None, "results": int(m.group(2)) if m else None,
                        "off": float(f[7]) if f[7] else None})
    return out


FEATURES = ("alpha", "beta", "gamma")


def features(r: dict) -> list:
    return [r["points"] * (1 - r["reached"] / r["results"]), r["findings"], r["points"] / (1 + r.get("checks", 0))]


def predict(r: dict, model: dict) -> float:
    if r["results"]:
        return min(r["points"], sum(model[k] * x for k, x in zip(FEATURES, features(r))))
    return model["prior"].get(r["q"], model["prior_all"]) * r["points"]


def fit(data: list, use=FEATURES) -> dict:
    """least squares for the weights in `use` over the questions with coverage, none negative (a negative weight -
    more of it, fewer points off - is noise: left out and fitted again); prior: the mean share off per question"""
    import numpy as np
    cov = [r for r in data if r["results"] and r["off"] is not None]
    model = {k: 0.0 for k in FEATURES}
    use = list(use)
    while cov and use:
        A = np.array([[x for k, x in zip(FEATURES, features(r)) if k in use] for r in cov])
        y = np.array([r["off"] for r in cov])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        if (sol >= 0).all():
            model.update({k: float(v) for k, v in zip(use, sol)})
            break
        use.pop(int(np.argmin(sol)))
    by_q = {}
    for r in data:
        if r["off"] is not None:
            by_q.setdefault(r["q"], []).append(r["off"] / r["points"])
    model["prior"] = {q: sum(v) / len(v) for q, v in by_q.items()}
    allv = [x for v in by_q.values() for x in v]
    model["prior_all"] = sum(allv) / len(allv) if allv else 0.3
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", default="")
    ap.add_argument("--exclude", default="", help="exams never used (the blind ones)")
    ap.add_argument("--without", nargs="*", default=[], choices=FEATURES, help="weights left out of the model")
    args = ap.parse_args()
    skip = {x.strip() for x in args.exclude.split(",") if x.strip()}
    data = [r for t in args.tables for r in rows(t) if r["exam"] not in skip and r["off"] is not None]
    exams = sorted({r["exam"] for r in data}, key=int)
    if len(exams) < 2:
        sys.exit("need two exams or more")
    use = [k for k in FEATURES if k not in args.without]
    model = fit(data, use)
    err_m = err_b = 0.0
    tot_m, tot_b = {}, {}
    for e in exams:                                      # each exam left out: fitted on the others, judged on it
        train = [r for r in data if r["exam"] != e]
        m = fit(train, use)
        base = {q: sum(r["off"] / r["points"] for r in train if r["q"] == q) / max(1, sum(1 for r in train if r["q"] == q))
                for q in {r["q"] for r in train}}
        for r in data:
            if r["exam"] == e:
                pm, pb = predict(r, m), base.get(r["q"], m["prior_all"]) * r["points"]
                err_m += abs(pm - r["off"])
                err_b += abs(pb - r["off"])
                tot_m[e] = tot_m.get(e, 0) + pm
                tot_b[e] = tot_b.get(e, 0) + pb
    n = len(data)
    print(f"fitted on exams {', '.join(exams)} ({n} questions): alpha {model['alpha']:.2f}, beta {model['beta']:.2f}, "
          f"gamma {model['gamma']:.2f}, "
          f"prior {', '.join(f'q{q} {v:.0%}' for q, v in sorted(model['prior'].items()))}")
    print(f"leave-one-out mean error per question: model {err_m / n:.1f} points, baseline (mean share off) {err_b / n:.1f}")
    for e in exams:
        off = sum(r["off"] for r in data if r["exam"] == e)
        print(f"  exam {e}: off {off:g}; model {tot_m[e]:.1f}, baseline {tot_b[e]:.1f}")
    model.update({"fitted_on": exams, "loo_error_model": round(err_m / n, 2), "loo_error_baseline": round(err_b / n, 2)})
    if args.out:
        json.dump(model, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
