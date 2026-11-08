"""Why the reader missed on tools/cmos_synth.py's drawings: each drawing's truth boxes (the symbols and letters as
stamped) against what tools/circuit_eval.py --details read - the transistors found (and as what), the supply arrow,
the ground, each letter's reading, the transistors read with an end missing.

    python tools/synth_failures.py --report cmos_eval.json --synth ~/eg-data/cmos_synth
"""
import argparse
import json
import os
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--synth", required=True)
ap.add_argument("--long-side", type=int, default=1600)
a = ap.parse_args()


def iou(p, q):
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0])) * max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - ix
    return ix / u if u > 0 else 0.0


def main():
    rep = json.load(open(a.report, encoding="utf-8"))
    rows = rep.get("transistor_circuits", {}).get("rows", []) + rep.get("rows", [])
    fet_as, rail, gnd, letters, short, causes = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    misread = Counter()
    per_gate = defaultdict(lambda: [0, 0])
    for r in rows:
        tf = os.path.join(a.synth, r["image"] + ".json")
        if not os.path.isfile(tf) or "boxes" not in r:
            continue
        t = json.load(open(tf, encoding="utf-8"))
        k = a.long_side / max(t["page"])
        tb = [(w, [v * k for v in b]) for w, b in t["boxes"]]
        rb, rt = r["boxes"], r.get("texts", [])
        why = []
        n_missed = 0
        for w, b in tb:
            if w.startswith("fet"):
                m = max(rb, key=lambda x: iou(x[1], b), default=None)
                got = m[0] if m and iou(m[1], b) > 0.3 else "missed"
                fet_as[got] += 1
                n_missed += got not in ("fet",)
            elif w in ("voltage_rail", "gnd"):
                m = max(rb, key=lambda x: iou(x[1], b), default=None)
                got = m[0] if m and iou(m[1], b) > 0.2 else "missed"
                (rail if w == "voltage_rail" else gnd)[got] += 1
                if w == "gnd" and got != w:
                    why.append(f"ground as {got}")
                elif w == "voltage_rail" and not any(typ == "voltage_source" for _, typ, _n, _r in r["read"]):
                    why.append(f"no supply source (the arrow boxed as {got})")   # (a 'ground' fed from below is one)
        if n_missed:
            why.append(f"{n_missed} transistor(s) not boxed as such")
        reads = defaultdict(set)
        for w, b in tb:
            if w.startswith("text"):
                L = w.split()[1]
                m = max(rt, key=lambda x: iou(x[1], b), default=None)
                got = m[0] if m and iou(m[1], b) > 0.2 else None
                ok = "missed" if got is None else ("right" if got.strip().upper() == L else
                                                   "case only" if got.strip().lower() == L.lower() else "wrong")
                letters[ok] += 1
                if ok == "wrong":
                    misread[f"{L}->{got}"] += 1
                reads[L].add((got or "").strip().upper())
        clash = [L for L, s in reads.items() if len(s) > 1] + \
                [f"{L1}={L2}" for L1 in reads for L2 in reads if L1 < L2 and reads[L1] & reads[L2]]
        if clash:
            why.append("letters read inconsistently: " + ",".join(clash))
        ns = sum(1 for _, typ, nodes, _r in r["read"] if typ in ("nmos", "pmos", "fet", "bjt", "npn", "pnp") and len(nodes) < 3)
        if ns:
            short[ns] += 1
            why.append(f"{ns} transistor(s) with an end missing")
        per_gate[t["gate"]][0] += bool(r["topology_exact"])
        per_gate[t["gate"]][1] += 1
        if not r["topology_exact"]:
            for w in why or ["(none of these)"]:
                causes[w.split(":")[0] if w.startswith("letters") else w] += 1
            print(r["image"], "|", "; ".join(why) or "(none of these)")
    print("\nexact by gate:", {g: f"{x}/{n}" for g, (x, n) in per_gate.items()})
    print("transistors boxed as:", dict(fet_as))
    print("supply arrow boxed as:", dict(rail), " ground boxed as:", dict(gnd))
    print("letters:", dict(letters), " misread:", misread.most_common(12))
    print("drawings with transistors' ends missing (how many each):", dict(short))
    print("causes in the drawings not read exactly:", causes.most_common())


if __name__ == "__main__":
    main()
