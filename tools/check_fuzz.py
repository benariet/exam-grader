"""Correct answers written many ways must never be judged wrong - a regression check for the checkers (no models, a
couple of minutes on a CPU):

    python tools/check_fuzz.py

  circuits, DC: every node voltage / current / part's V, I, P of two circuits in many formats (rounded, prefixes and
     units, a decimal comma, powers of ten, fractions)
  circuits, AC: phasors in polar / rectangular / time form, three circuits
  circuits, a switching: first-order responses written many ways
  derivations (grader/selfcheck.py, no key): 142 kinds of correct derivations - none may be judged wrong - and
     27 wrong ones that must be caught; and the correct ones again as the formula reader spaces its tokens
Exit status 1 when anything fails.
"""
import cmath
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def circuits_dc():
    import sympy as sp
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values

    tal = with_values(Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                               C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                               C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")],
                              [Current("I1", "R1", "n1", "a"), Current("I2", "R2", "a", "0"), Current("I3", "R3", "n3", "a")],
                              ground="0", node_labels={"a": "A", "n1": "B", "n3": "D"}),
                      {"E1": "12", "E2": "6", "R1": "2k", "R2": "4k", "R3": "4k"})
    wheat = Netlist([C("V1", "voltage_source", ["t", "0"], plus="t", value="10"), C("R1", "resistor", ["t", "a"], value="1k"),
                     C("R2", "resistor", ["t", "b"], value="2k"), C("R3", "resistor", ["a", "0"], value="3k"),
                     C("R4", "resistor", ["b", "0"], value="1k"), C("R5", "resistor", ["a", "b"], value="5k")], ground="0",
                    node_labels={"a": "A", "b": "B", "t": "T"})


    def formats(v, unit):
        """a value (base units) written as students write it"""
        out = []
        for sig in (3, 4):
            out.append(f"{v:.{sig}g}")
        pre = [("m", 1e-3), ("\\mu ", 1e-6), ("k", 1e3)]
        for p, f in pre:
            x = v / f
            if 1 <= abs(x) < 1000:
                out.append(f"{x:.4g} {p}{unit}")
                out.append(f"{x:.3g}\\,\\mathrm{{{p.strip()}{unit}}}")
                if "." in f"{x:.3g}":
                    out.append(f"{x:.3g}".replace(".", ",") + f" {p}{unit}")
        if v != 0:
            e = int(math.floor(math.log10(abs(v))))
            out.append(f"{v / 10 ** e:.3g} \\cdot 10^{{{e}}} {unit}")
        fr = sp.nsimplify(v, rational=True, tolerance=1e-12)
        if fr.q != 1 and fr.q < 50:
            out.append(f"\\frac{{{fr.p}}}{{{fr.q}}}")
        return out


    units = {"v": "V", "i": "A", "p": "W"}
    bad = total = 0
    for name, net in (("tal", tal), ("wheatstone", wheat)):
        r = net.solve("dc", {})
        quants = {}
        for k, v in r.items():
            if k.startswith("__"):
                continue
            nm = k.replace("V_", "V_{") + "}" if k.startswith("V_") else k.replace("I", "I_{", 1) + "}"
            quants[nm] = (float(sp.N(v)), "V" if k.startswith("V_") else "A")
        for c in net.components:
            if c.type == "resistor":
                va = 0 if c.nodes[0] == "0" else float(sp.N(r[f"V_{net.node_labels.get(c.nodes[0], c.nodes[0])}"]))
                vb = 0 if c.nodes[1] == "0" else float(sp.N(r[f"V_{net.node_labels.get(c.nodes[1], c.nodes[1])}"]))
                R = float(sp.N(net.symbol(c)))
                quants[f"V_{{{c.id}}}"] = (va - vb, "V")
                quants[f"I_{{{c.id}}}"] = ((va - vb) / R, "A")
                quants[f"P_{{{c.id}}}"] = ((va - vb) ** 2 / R, "W")
        for q, (v, unit) in quants.items():
            for f in formats(v, unit):
                tex = f"{q} = {f}"
                got = [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
                total += 1
                if any(v_ == "wrong" for v_, _ in got):
                    bad += 1
                    print("WRONG?", name, tex, got)
        for c in net.components:                     # a part's current as V / R with the unit after the divisor's number
            if c.type != "resistor":                 # ('I_{R1} = 4 / 2 k\Omega = 2 mA'), and doubled (must be wrong)
                continue
            va = 0 if c.nodes[0] == "0" else float(sp.N(r[f"V_{net.node_labels.get(c.nodes[0], c.nodes[0])}"]))
            vb = 0 if c.nodes[1] == "0" else float(sp.N(r[f"V_{net.node_labels.get(c.nodes[1], c.nodes[1])}"]))
            R = float(sp.N(net.symbol(c)))
            i_ma = (va - vb) / R * 1e3
            p_mw = i_ma ** 2 * R / 1e3
            for tex, right in ((f"I_{{{c.id}}} = {va - vb:.4g} / {R / 1e3:.4g} k\\Omega", True),
                               (f"I_{{{c.id}}} = {va - vb:.4g} / {R / 1e3:.4g} k\\Omega = {i_ma:.4g} mA", True),
                               (f"I_{{{c.id}}} = {va - vb:.4g} / {R / 1e3:.4g} k\\Omega = {2 * i_ma:.4g} mA", False),
                               # (a power in bare numbers, mA and kΩ, then mW: 'P_{R1} = 2^2 \\cdot 4 = 16 mW')
                               (f"P_{{{c.id}}} = {abs(i_ma):.6g}^2 \\cdot {R / 1e3:.4g} = {p_mw:.4g} mW", True),
                               (f"P_{{{c.id}}} = {abs(i_ma):.6g}^2 \\cdot {R / 1e3:.4g} = {1.5 * p_mw:.4g} mW", False)):
                if abs(i_ma) < 1e-9:
                    continue
                got = [(x.verdict, x.sure) for x in check_against(circuit_statements(tex), net)]
                total += 1
                if any(v_ == "wrong" for v_, _ in got) == right:
                    bad += 1
                    print("WRONG?" if right else "MISSED", name, tex, got)
    return total, bad


def circuits_ac():
    import sympy as sp
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements

    nets = {
        "rl": Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10cos(1000t)", ac=True),
                       C("R1", "resistor", ["a", "b"], value="3"), C("L1", "inductor", ["b", "0"], value="4m")],
                      [Current("I1", "R1", "a", "b")], ground="0", node_labels={"b": "B", "a": "A"}),
        "rlc": Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="20cos(500t + 30°)", ac=True),
                        C("R1", "resistor", ["a", "b"], value="10"), C("L1", "inductor", ["b", "c"], value="40m"),
                        C("C1", "capacitor", ["c", "0"], value="100u")], [Current("I1", "R1", "a", "b")], ground="0",
                       node_labels={"a": "A", "b": "B", "c": "D"}),
        "phasor": Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10∠0°"), C("Z1", "resistor", ["a", "b"], value="3+j4Ω"),
                           C("C1", "capacitor", ["b", "0"], value="-j2Ω")], [Current("I1", "Z1", "a", "b")], ground="0",
                          node_labels={"a": "A", "b": "B"}),
    }


    def forms(z, w, name):
        m, th = abs(z), math.degrees(cmath.phase(z))
        out = []
        for sig in (3, 4):
            out.append(f"{m:.{sig}g}\\angle {th:.{sig}g}^\\circ")
            out.append(f"{m:.{sig}g} \\angle {th:.{sig}g}°")
            re_, im_ = z.real, z.imag
            sign = "+" if im_ >= 0 else "-"
            out.append(f"{re_:.{sig}g} {sign} j{abs(im_):.{sig}g}")
            out.append(f"{re_:.{sig}g} {sign} {abs(im_):.{sig}g}j")
        if w is not None and not name.startswith(("Z", "z")):
            sign = "+" if th >= 0 else "-"
            out.append(f"{name.lower()}(t) = {m:.4g}\\cos({w:g}t {sign} {abs(th):.4g}^\\circ)")
        return out


    bad = total = 0
    for nm, net in nets.items():
        r = net.solve("phasor", {})
        from grader.circuit import ac_phasor
        ws = {ac_phasor(c.value)[1] for c in net.components if c.ac and ac_phasor(c.value)}
        w = float(ws.pop()) if len(ws) == 1 else None
        for k, v in r.items():
            if k.startswith("__"):
                continue
            z = complex(sp.N(v))
            q = ("V_{" + k[2:] + "}") if k.startswith("V_") else ("I_{" + k[1:] + "}")
            for f in forms(z, w, q.replace("{", "").replace("}", "")):
                tex = f if "(t)" in f else f"{q} = {f}"
                got = [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
                total += 1
                if any(v_ == "wrong" for v_, _ in got):
                    bad += 1
                    print("WRONG?", nm, tex, got)
    return total, bad


def circuits_forms():
    """answers about magnitudes, H(jw) / H(s), dB, phasors, a voltage marked on the drawing - each written several ways
    (bars as |..|, \\left| \\right|, \\lvert \\rvert; units plain, in brackets, upright; '\\approx'; spaced as the
    formula reader writes): the correct ones never judged wrong, the wrong ones always"""
    from grader.circuit import Component as C, Netlist, Voltage
    from grader.circuit_check import check_against, circuit_statements
    div = Netlist([C("E", "voltage_source", ["a", "0"], value="10V", plus="a", label="E"), C("R1", "resistor", ["a", "m"], value="2k", label="R_1"),
                   C("R2", "resistor", ["m", "0"], value="1k", label="R_2")], ground="0", node_labels={"m": "A"},
                  voltages=[Voltage("V_x", "a", "m", "R1")])
    lp = Netlist([C("V1", "voltage_source", ["a", "0"], value="10cos(1000t)", plus="a", ac=True),
                  C("R1", "resistor", ["a", "m"], value="1k"), C("C1", "capacitor", ["m", "0"], value="1u")], ground="0", node_labels={"m": "o"})
    inv = Netlist([C("V1", "voltage_source", ["i", "0"], value="1V", plus="i"), C("R1", "resistor", ["i", "m"], value="1k", label="R_1"),
                   C("R2", "resistor", ["m", "o"], value="10k", label="R_f"),
                   C("OA1", "opamp", ["0", "m", "o"], roles={"In+": "0", "In-": "m", "Out": "o"})], ground="0", node_labels={"o": "o"})
    right = [("V_A = 3.33 V", div), ("V_x = 6.67 V", div), ("I = 3.33 mA", div), ("|V_o| = 7.07 V", lp), ("|Z| = 1.41 k\\Omega", lp),
             ("|H(j1000)| = 0.707", lp), ("20 \\log |H(j1000)| = -3 dB", lp), ("H(s) = \\frac{1}{1 + 0.001 s}", lp),
             ("V_o = 7.07 \\angle -45^\\circ V", lp), ("V_o = 5 - j5 V", lp), ("A_v = 20 dB", inv), ("20\\log|A_v| = 20", inv), ("A_v = -10", inv)]
    wrong = [("V_x = -6.67 V", div), ("V_A = 4 V", div), ("|V_o| = 5 V", lp), ("H(s) = \\frac{1}{1 + s}", lp), ("A_v = 26 dB", inv)]
    from grader.circuit import Current
    from grader.circuit_check import with_values
    tal = with_values(Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                               C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                               C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")],
                              [Current("I1", "R1", "n1", "a"), Current("I2", "R2", "a", "0"), Current("I3", "R3", "n3", "a")],
                              ground="0", node_labels={"a": "A", "n1": "B", "n3": "D"}),
                      {"E1": "12", "E2": "6", "R1": "2k", "R2": "4k", "R3": "4k"})
    right += [(t, tal) for t in ("-12 + 2k I_1 + 4k I_2 = 0", "12 = 2000 I_1 + 4000 I_2", "2 \\cdot 10^3 I_1 + 4 \\cdot 10^3 I_2 - 12 = 0",
                                 "-6 + 4k I_3 + 4k I_2 = 0", "-12 + 2k I_1 - 4k I_3 + 6 = 0", "E_1 = R_1 I_1 + R_2 I_2", "I_1 + I_3 = I_2",
                                 "\\frac{V_A - 12}{2k} + \\frac{V_A}{4k} + \\frac{V_A - 6}{4k} = 0")]
    wrong += [("I_1 + I_2 = I_3", tal), ("\\frac{V_A - 12}{2k} + \\frac{V_A}{4k} = 0", tal)]
    right += [("\\frac{V_o - 10}{1k} + \\frac{V_o}{-j1k} = 0", lp), ("Z_C = -j1k", lp), ("Z = 1k - j1k", lp)]
    wrong += [("\\frac{V_o - 10}{1k} + \\frac{V_o}{j1k} = 0", lp)]

    def variants(t):
        out = {t, t.replace("=", "\\approx", 1) if t.count("=") == 1 else t}
        out.add(re.sub(r"\|([^|]+)\|", r"\\left| \1 \\right|", t))
        out.add(re.sub(r"\|([^|]+)\|", r"\\lvert \1 \\rvert", t))
        m = re.search(r"(?<=\d) (V|mA|k\\Omega|dB)$", t)
        if m:
            out.add(t[:m.start()] + f" [{m.group(1)}]")
            out.add(t[:m.start()] + f" \\, \\mathrm{{{m.group(1)}}}")
        out.add(spaced(t))
        return sorted(out)
    total = bad = 0
    for cases, should in ((right, False), (wrong, True)):
        for t, net in cases:
            for v in variants(t):
                got = [c.verdict for c in check_against(circuit_statements(v), net)]
                total += 1
                if ("wrong" in got) != should:
                    bad += 1
                    print("WRONG?" if not should else "MISSED", v, got)
    return total, bad


def circuits_random(n_nets: int = 10, seed: int = 3):
    """random ladder / bridge networks with tricky names - an internal node 'a' beside a node labelled 'A', 'c' beside
    a capacitor-like name, labels '1', 'o', 'x', an unlabelled source whose id is V1 beside node 1: every labelled node's
    voltage and every resistor's V / I / P never judged wrong for sure (ground truth: the network solved with every
    internal node renamed to a name nothing collides with)"""
    import random
    import sympy as sp
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    labels_pool = ["A", "B", "C", "D", "E", "a", "b", "o", "x", "1", "2", "3", "out", "N1"]
    internal_pool = ["a", "b", "c", "d", "m", "n1", "n2", "t", "x", "r1", "e", "o", "q"]
    rng = random.Random(seed)
    total = bad = 0
    for _ in range(n_nets):
        nodes = rng.sample(internal_pool, rng.randint(3, 5))
        comps = [C("V1", "voltage_source", [nodes[0], "0"], plus=nodes[0], value=str(rng.choice([5, 10, 12, 9])))]
        for i in range(1, len(nodes)):
            comps.append(C(f"R{len(comps)}", "resistor", [nodes[i - 1], nodes[i]], value=f"{rng.choice([1, 2, 3, 4, 5, 10])}k"))
            if rng.random() < 0.8:
                comps.append(C(f"R{len(comps)}", "resistor", [nodes[i], "0"], value=f"{rng.choice([1, 2, 3, 4, 6, 10])}k"))
        labels = {}
        for nd in nodes:
            if rng.random() < 0.7:
                labels[nd] = rng.choice([x for x in labels_pool if x.lower() not in {v.lower() for v in labels.values()}])
        net = Netlist(comps, [], ground="0", node_labels=labels)
        ren = {n: f"zz{k}" for k, n in enumerate(nodes)}
        good = Netlist([C(c.id, c.type, [ren.get(x, x) for x in c.nodes], plus=ren.get(c.plus, c.plus) if c.plus else None, value=c.value)
                        for c in comps], [], ground="0", node_labels={ren[k]: v for k, v in labels.items()})
        sol = good.solve("dc", {})
        if not sol:
            continue
        volt = lambda n: 0.0 if n == "0" else float(sp.N(sol[f"V_{good.node_labels.get(ren[n], ren[n])}"]))   # noqa: E731
        answers = [f"V_{{{lab}}} = {volt(nd):.4g}" for nd, lab in labels.items()]
        for c in comps[1:]:
            r_ = float(c.value[:-1]) * 1000
            dv = volt(c.nodes[0]) - volt(c.nodes[1])
            answers += [f"V_{{{c.id}}} = {abs(dv):.4g}", f"I_{{{c.id}}} = {abs(dv) / r_ * 1000:.4g} mA", f"P_{{{c.id}}} = {dv * dv / r_ * 1000:.4g} mW"]
        for a in answers:
            got = [(x.verdict, x.sure) for x in check_against(circuit_statements(a), net)]
            total += 1
            if any(v == "wrong" and sure for v, sure in got):
                bad += 1
                print("WRONG?", a, got, [(c.id, c.nodes) for c in comps], labels)
    return total, bad


def circuits_random_labelled(n_nets: int = 8, seed: int = 5):
    """random networks, harder: a current source or a labelled voltage source, resistors labelled out of order ('R_1' on
    the part the reader called R3), currents drawn on some of them (I_1 ...), answers by the labels - right ones never a
    sure 'wrong', wrong ones (30 % off) always caught. -> (right answers, sure wrongs among them, wrong answers, missed)"""
    import random
    import sympy as sp
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rng = random.Random(seed)
    total = bad = missed = n_wrong = 0
    for _ in range(n_nets):
        n = rng.randint(3, 4)
        nodes = [f"n{k}" for k in rng.sample(range(1, 9), n)]
        if rng.random() < 0.7:
            comps = [C("V1", "voltage_source", [nodes[0], "0"], plus=nodes[0], value=str(rng.choice([5, 10, 12])), label=rng.choice(["E", "V_s", "E_1"]))]
        else:
            comps = [C("I1", "current_source", ["0", nodes[0]], plus=nodes[0], value=f"{rng.choice([1, 2, 5])}m", label=rng.choice(["I_s", "J"]))]
        pairs = [p for i in range(1, n) for p in ((nodes[i - 1], nodes[i]), (nodes[i], "0"))]
        order = list(range(1, len(pairs) + 1))
        rng.shuffle(order)
        rs = [C(f"R{k + 1}", "resistor", [a, b], value=f"{rng.choice([1, 2, 3, 4, 5])}k", label=f"R_{order[k]}") for k, (a, b) in enumerate(pairs)]
        comps += rs
        curs = [Current(f"I_{k + 1}", r.id, r.nodes[0], r.nodes[1]) for k, r in enumerate(rng.sample(rs, min(3, len(rs))))]
        labels = {nd: lab for nd, lab in zip(nodes, rng.sample(["A", "B", "C", "D"], n)) if rng.random() < 0.8}
        net = Netlist(comps, curs, ground="0", node_labels=labels)
        sol = net.solve("dc", {})
        if not sol:
            continue
        volt = lambda nd: 0.0 if nd == "0" else float(sp.N(sol[f"V_{labels.get(nd, nd)}"]))   # noqa: E731
        right = [f"V_{lab} = {volt(nd):.4g}" for nd, lab in labels.items()]
        wrong = [f"V_{lab} = {volt(nd) * 1.3 + 0.5:.4g}" for nd, lab in labels.items()]
        for r in rs:
            dv, r_ = volt(r.nodes[0]) - volt(r.nodes[1]), float(r.value[:-1]) * 1000
            right += [f"I_{{{r.label}}} = {abs(dv) / r_ * 1000:.4g} mA", f"P_{{{r.label}}} = {dv * dv / r_ * 1000:.4g} mW"]
        for cu in curs:
            r = next(x for x in rs if x.id == cu.through)
            i = (volt(cu.frm) - volt(cu.to)) / (float(r.value[:-1]) * 1000)
            right.append(f"{cu.label} = {i * 1000:.4g} mA")
            wrong.append(f"{cu.label} = {i * 1000 * 1.3 + 0.3:.4g} mA")
        for a in right:
            got = [(x.verdict, x.sure) for x in check_against(circuit_statements(a), net)]
            total += 1
            if any(v == "wrong" and sure for v, sure in got):
                bad += 1
                print("WRONG?", a, got, [(c.id, c.label, c.nodes) for c in comps], labels)
        for a in wrong:
            got = [x.verdict for x in check_against(circuit_statements(a), net)]
            n_wrong += 1
            if "wrong" not in got:
                missed += 1
                print("MISSED", a, got)
    return total, bad, n_wrong, missed


def circuits_random_ac(n_nets: int = 8, seed: int = 7):
    """random AC ladders (R, L, C by a source Vm cos(wt + phi)), nodes named 'c' beside C1, 'l' beside L1: node phasors
    (rectangular, polar in degrees), magnitudes, a current drawn - right answers never a sure 'wrong', wrong ones (30 %
    off in size) always caught -> (right answers, sure wrongs among them, wrong answers, missed)"""
    import random
    import sympy as sp
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rng = random.Random(seed)
    total = bad = missed = n_wrong = 0
    kinds = {"resistor": ["1k", "2k", "500"], "inductor": ["100m", "1", "500m"], "capacitor": ["1u", "2u", "10u"]}
    pref = {"resistor": "R", "inductor": "L", "capacitor": "C"}
    for _ in range(n_nets):
        w, n = rng.choice([100, 500, 1000, 2000]), rng.randint(2, 3)
        nodes = rng.sample(["a", "b", "c", "l", "m", "o", "x"], n)
        vm, ph = rng.choice([5, 10, 20]), rng.choice([0, 30, -45, 60])
        comps = [C("V1", "voltage_source", [nodes[0], "0"], plus=nodes[0], ac=True,
                   value=f"{vm}cos({w}t {'+' if ph > 0 else '-'} {abs(ph)}°)" if ph else f"{vm}cos({w}t)")]
        cnt = dict.fromkeys(kinds, 0)
        for i in range(1, n):
            for a, b in ((nodes[i - 1], nodes[i]), (nodes[i], "0")):
                t = rng.choice(list(kinds))
                cnt[t] += 1
                comps.append(C(f"{pref[t]}{cnt[t]}", t, [a, b], value=rng.choice(kinds[t])))
        curs = [Current("I_1", comps[1].id, comps[1].nodes[0], comps[1].nodes[1])]
        labels = dict(zip(nodes, rng.sample(["A", "B", "D", "o", "x"], n)))
        net = Netlist(comps, curs, ground="0", node_labels=labels)
        sol = net.solve("phasor", {})
        try:
            vals = {lab: complex(sp.N(sol[f"V_{lab}"])) for lab in labels.values()}
        except (TypeError, KeyError):
            continue
        right, wrong = [], []
        for lab, z in vals.items():
            m, th = abs(z), math.degrees(cmath.phase(z))
            right += [f"V_{lab} = {z.real:.4g} {'+' if z.imag >= 0 else '-'} j{abs(z.imag):.4g}", f"V_{lab} = {m:.4g} \\angle {th:.4g}^\\circ",
                      f"|V_{lab}| = {m:.4g}"]
            wrong.append(f"|V_{lab}| = {m * 1.3 + 0.2:.4g}")
        if "I_1" in sol:
            zi = complex(sp.N(sol["I_1"]))
            right.append(f"I_1 = {abs(zi) * 1000:.4g} \\angle {math.degrees(cmath.phase(zi)):.4g}^\\circ mA")
            wrong.append(f"I_1 = {abs(zi) * 1300:.4g} \\angle {math.degrees(cmath.phase(zi)):.4g}^\\circ mA")
        for a in right:
            got = [(x.verdict, x.sure) for x in check_against(circuit_statements(a), net)]
            total += 1
            if any(v == "wrong" and sure for v, sure in got):
                bad += 1
                print("WRONG?", a, got, [(c.id, c.nodes, c.value) for c in comps], labels)
        for a in wrong:
            n_wrong += 1
            if "wrong" not in [x.verdict for x in check_against(circuit_statements(a), net)]:
                missed += 1
                print("MISSED", a, [(c.id, c.nodes, c.value) for c in comps], labels)
    return total, bad, n_wrong, missed


def circuits_switching():
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    # switch closes at t = 0: tau = 1 ms, v_C: 0 -> 5 V; i_C: 5 mA * e^{-1000t}; V_A = v_C
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"], state="open"),
                  C("R1", "resistor", ["n1", "a"], value="2k"), C("R2", "resistor", ["a", "0"], value="2k"),
                  C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})
    vc = ["5(1 - e^{-1000t})", "5 - 5e^{-1000t}", "5(1 - e^{-t/0.001})", "5(1 - e^{-t/10^{-3}})", "5\\left(1 - e^{-\\frac{t}{0.001}}\\right)",
          "5 - 5 e^{-t / 1 ms}", "5(1-e^{-1000t}) V", "5(1-e^{-1000t})u(t)", "5(1 - e^{-1000 t}) \\quad t > 0", "5(1 - e^{-1000t}), t \\geq 0",
          "5 - 5\\exp(-1000t)", "5(1-e^{-10^3 t})", "5.0(1 - e^{-1000t})"]
    ic = ["5e^{-1000t} mA", "0.005e^{-1000t}", "5 \\cdot 10^{-3} e^{-1000t}", "\\frac{5}{1000}e^{-1000t}", "5 mA \\cdot e^{-t/1ms}"]
    bad = total = 0
    for name, forms in (("v_C(t)", vc), ("V_A(t)", vc), ("i_C(t)", ic)):
        for f in forms:
            tex = f"{name} = {f}"
            got = [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), rc)]
            total += 1
            flag = any(v == "wrong" for v, _ in got)
            bad += flag
            if flag:
                print("WRONG?", tex, got)
    return total, bad



RIGHT_DERIVATIONS = [
    ["V = I R \\vert I = 2 A, R = 5 \\Omega", "V = 10 V"],
    ["V = I R \\vert I = 2 mA, R = 5 k\\Omega", "V = 10 V"],
    ["V = I R \\vert I = 2 [mA], R = 5 [k\\Omega]", "V = 10 [V]"],
    ["V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R}", "I = 2 mA"],
    ["V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R} = 2 mA"],
    ["V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R} = \\frac{10}{5000} = 0.002 A"],
    ["V = 10", "R = 5k", "I = \\frac{V}{R} = 2 m"],
    ["I = \\frac{V}{R} = \\frac{10 V}{5 k\\Omega} = 2 mA"],
    ["P = I^2 R \\vert I = 2 A, R = 5 \\Omega", "P = 20 W"],
    ["P = I^{2} R \\vert I = 0.1 A, R = 100 \\Omega", "P = 1 W"],
    ["P = \\frac{V^2}{R} \\vert V = 10 V, R = 2 k\\Omega", "P = 50 mW"],
    ["P = \\frac{V^2}{R} \\vert V = 10 V, R = 2 k\\Omega", "P = 0.05 W"],
    ["P = V I = 10 V \\cdot 2 mA = 20 mW"],
    ["V_{out} = V_{in} \\frac{R_2}{R_1 + R_2} \\vert V_{in} = 12 V, R_1 = 1 k\\Omega, R_2 = 2 k\\Omega", "V_{out} = 8 V"],
    ["V_{out} = V_{in} \\frac{R_2}{R_1 + R_2} \\vert V_{in} = 12, R_1 = 1k, R_2 = 2k", "V_{out} = 8"],
    ["R_{eq} = R_1 + R_2 \\vert R_1 = 2 k\\Omega, R_2 = 3 k\\Omega", "R_{eq} = 5 k\\Omega"],
    ["R_{eq} = R_1 \\| R_2 \\vert R_1 = 2, R_2 = 3", "R_{eq} = 1.2"],
    ["R_{eq} = 2 \\| 3 = 1.2"],
    ["R_{eq} = \\frac{R_1 R_2}{R_1 + R_2} = \\frac{2 \\cdot 3}{2 + 3} = 1.2"],
    ["\\tau = R C \\vert R = 1 k\\Omega, C = 1 \\mu F", "\\tau = 1 ms"],
    ["\\tau = R C \\vert R = 1 k\\Omega, C = 1 \\mu F", "\\tau = 0.001 s"],
    ["\\tau = R C \\vert R = 2.2 k\\Omega, C = 10 \\mu F", "\\tau = 22 m"],
    ["f = \\frac{1}{T} \\vert T = 2 ms", "f = 500 Hz"],
    ["f = \\frac{1}{T} \\vert T = 2 ms", "f = 0.5 kHz"],
    ["\\omega = 2 \\pi f \\vert f = 50 Hz", "\\omega = 314.16"],
    ["X_C = \\frac{1}{2 \\pi f C} \\vert f = 50 Hz, C = 10 \\mu F", "X_C = 318 \\Omega"],
    ["X_L = 2\\pi f L \\vert f = 50, L = 0.1", "X_L = 31.4"],
    ["X_L = 2\\pi(50)(0.1) = 31.4"],
    ["C = \\frac{Q}{V} \\vert Q = 10 \\mu C, V = 5 V", "C = 2 \\mu F"],
    ["W = \\frac{1}{2} C V^2 \\vert C = 10 \\mu F, V = 10 V", "W = 0.5 mJ"],
    ["\\eta = \\frac{P_{out}}{P_{in}} \\vert P_{out} = 80 W, P_{in} = 100 W", "\\eta = 80\\%"],
    ["\\eta = \\frac{P_{out}}{P_{in}} \\vert P_{out} = 80 W, P_{in} = 100 W", "\\eta = 0.8"],
    ["E = m g h \\vert m = 2 kg, g = 10, h = 5 m", "E = 100 J"],
    ["E = \\frac{1}{2} m v^2 \\vert m = 2 kg, v = 3", "E = 9 J"],
    ["x = x_0 + v t \\vert x_0 = 1 m, v = 2, t = 3 s", "x = 7 m"],
    ["v = \\sqrt{2 g h} \\vert g = 10, h = 5", "v = 10"],
    ["a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c = 5"],
    ["a = 3", "b = 4", "c = \\sqrt{a^2 + b^2} = 5"],
    ["\\sin(30^\\circ) = 0.5"], ["\\cos(60^{\\circ}) = 0.5"],
    ["P = V I \\cos(\\phi) \\vert V = 10, I = 2, \\phi = 60^\\circ", "P = 10"],
    ["\\theta = \\arctan(\\frac{X}{R}) \\vert X = 1, R = 1", "\\theta = 45^\\circ"],
    ["\\theta = \\arctan(\\frac{X}{R}) \\vert X = 1, R = 1", "\\theta = 0.785"],
    ["x = 3.\\overline{3}", "y = 3x = 10"],
    ["R = \\frac{\\Delta V}{\\Delta I} \\vert \\Delta V = 2 V, \\Delta I = 4 mA", "R = 500 \\Omega"],
    ["x = 1.5e-3 \\cdot 2 = 3e-3"],
    ["2x+3=7", "x=2"],
    ["7 \\times 8 = 56"], ["2 \\cdot 10^{3} \\cdot 5 \\cdot 10^{-3} = 10"], ["\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}"],
    ["A = \\pi r^{2} \\vert r = 5", "A = 78.54"], ["A \\approx \\pi r^{2} \\vert r = 5", "A \\approx 78.5"],
    ["Z = \\sqrt{R^2 + X^2} \\vert R = 3, X = 4", "Z = 5"],
    ["I_{rms} = \\frac{I_m}{\\sqrt{2}} \\vert I_m = 10", "I_{rms} = 7.07"],
    ["V = 10 V", "I = 2 mA", "R = \\frac{V}{I}", "R = 5 k\\Omega", "P = I^2 R", "P = 20 mW"],
    # calculus, functions, logs, roots, units as fractions, unicode signs, words between the math
    ['\\frac{d}{dx} x^3 = 3x^2'],
    ['\\frac{d}{dx}(x^2 + 3x) = 2x + 3'],
    ['\\int_0^2 x^2 dx = \\frac{8}{3}'],
    ['\\int_{0}^{2} x^{2} \\, dx = \\left[ \\frac{x^3}{3} \\right]_0^2 = \\frac{8}{3}'],
    ['\\int_0^1 e^x dx = e - 1'],
    ['\\int_{0}^{\\pi} \\sin x \\, dx = 2'],
    ['\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1'],
    ['f(x) = x^2 + 1', 'f(2) = 5'],
    ['f(x) = 3x - 1', 'f(1) = 2', 'f(2) = 5'],
    ['f(x) = x^2', "f'(x) = 2x"],
    ['y = x^3', "y' = 3x^2"],
    ['f(t) = e^{-2t}', 'f(0) = 1'],
    ['\\ln(e^2) = 2'],
    ['\\log_{10}(1000) = 3'],
    ['\\log_2 8 = 3'],
    ['e^{-1} = 0.368'],
    ['e^0 = 1'],
    ['10^{-3} = 0.001'],
    ['20\\log_{10}(10) = 20 dB'],
    ['20 \\log(100) = 40'],
    ['\\ln 2 = 0.693'],
    ['2x + 3 = 7 \\Rightarrow x = 2'],
    ['2x + 3 = 7 \\implies 2x = 4 \\implies x = 2'],
    ['3x = 12 \\rightarrow x = 4'],
    ['x^2 = 9 \\Rightarrow x = \\pm 3'],
    ['x^2 - 5x + 6 = 0', 'x_1 = 2, x_2 = 3'],
    ['x^2 - 5x + 6 = 0', 'x_{1,2} = \\frac{5 \\pm 1}{2}'],
    ['x_{1,2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\vert a = 1, b = -5, c = 6', 'x_1 = 3', 'x_2 = 2'],
    ['s^2 + 3s + 2 = 0', 's = -1, s = -2'],
    ['(x - 1)(x + 2) = 0', 'x = 1 \\text{ or } x = -2'],
    ['v = \\frac{d}{t} \\vert d = 100 m, t = 10 s', 'v = 10 \\frac{m}{s}'],
    ['a = \\frac{\\Delta v}{\\Delta t} \\vert \\Delta v = 20 \\frac{m}{s}, \\Delta t = 4 s', 'a = 5 \\frac{m}{s^2}'],
    ['F = m a \\vert m = 2 kg, a = 3 \\frac{m}{s^2}', 'F = 6 N'],
    ['v = 72 \\frac{km}{h} = 20 \\frac{m}{s}'],
    ['p = \\frac{F}{A} \\vert F = 100 N, A = 0.5 m^2', 'p = 200 Pa'],
    ['\\rho = \\frac{m}{V} \\vert m = 2 kg, V = 0.001 m^3', '\\rho = 2000 \\frac{kg}{m^3}'],
    ['v = 10 m/s', 't = 3 s', 'd = v t = 30 m'],
    ['Q = C V \\vert C = 100 nF, V = 5 V', 'Q = 500 nC'],
    ['Q = C V \\vert C = 100 nF, V = 5 V', 'Q = 0.5 \\mu C'],
    ['R = \\rho \\frac{L}{A} \\vert \\rho = 2, L = 3, A = 0.5', 'R = 12'],
    ['2 \\times 10^{3} \\times 3 \\times 10^{-3} = 6'],
    ['(2 \\cdot 10^3)(3 \\cdot 10^{-3}) = 6'],
    ['4^{\\frac{1}{2}} = 2'],
    ['\\sqrt[3]{27} = 3'],
    ['8^{1/3} = 2'],
    ['2^{10} = 1024'],
    ['\\frac{6}{4} = 1.5 = \\frac{3}{2}'],
    ['\\frac{1}{4} = 0.25 = 25\\%'],
    ['\\sin^2(30^\\circ) + \\cos^2(30^\\circ) = 1'],
    ['\\tan(45^\\circ) = 1'],
    ['\\sin(\\frac{\\pi}{6}) = 0.5'],
    ['\\cos(\\pi) = -1'],
    ['\\arcsin(0.5) = 30^\\circ'],
    ['\\arccos(0) = \\frac{\\pi}{2}'],
    ['\\sin(90°) = 1'],
    ['(1 + j)(1 - j) = 2'],
    ['\\frac{1}{j} = -j'],
    ['|4 + 3j| = 5'],
    ['j^2 = -1'],
    ['G = \\frac{V_o}{V_i} \\vert V_o = 10, V_i = 0.1', 'G = 100', 'G_{dB} = 20\\log(100) = 40 dB'],
    ['f_c = \\frac{1}{2 \\pi R C} \\vert R = 1 k\\Omega, C = 1 \\mu F', 'f_c = 159 Hz'],
    ['\\omega_0 = \\frac{1}{\\sqrt{L C}} \\vert L = 1 mH, C = 1 \\mu F', '\\omega_0 = 31623 \\frac{rad}{s}'],
    ['T = \\frac{2 \\pi}{\\omega} \\vert \\omega = 1000', 'T = 6.28 ms'],
    ['I = \\frac{V}{R} = \\frac{5}{1k} = 5 mA'],
    ['2 · 3 = 6'],
    ['6 ÷ 2 = 3'],
    ['4 × 5 = 20'],
    ['5 − 3 = 2'],
    ['2x = 4 \\text{ ולכן } x = 2'],
    ['\\frac{5!}{2! \\cdot 3!} = 10'],
    ['\\binom{5}{2} = 10'],
    ['d = 5 km = 5000 m'],
    ['m = 500 g = 0.5 kg'],
    ['C = 100 nF = 0.1 \\mu F'],
    ['I_C = \\beta I_B = 100 \\cdot 20 \\mu A = 2 mA'],
    ['g_m = \\frac{I_C}{V_T} = \\frac{1 mA}{25 mV} = 40 \\frac{mA}{V}'],
    ['A = 1 + \\frac{R_2}{R_1} = 1 + \\frac{9k}{1k} = 10'],
    ['f_0 = \\frac{1}{2\\pi\\sqrt{LC}} = \\frac{1}{2\\pi\\sqrt{10 mH \\cdot 1 \\mu F}} = 1.59 kHz'],
    ['v = \\omega r = 10 \\cdot 0.5 = 5 \\frac{m}{s}'],
    ['W = F d \\cos\\theta = 10 \\cdot 2 \\cdot \\cos 60^\\circ = 10 J'],
    ['\\frac{1}{R} = \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2}', 'R = 2'],
    ['I = \\frac{V}{R}', 'I = \\frac{12}{4}', 'I = 3 A'],
    ["V_A' = 3", "V_A'' = 2", "V_A = V_A' + V_A'' = 5"],
    ['x = 2', '\\frac{dy}{dx} = 3x^2 = 12'],
    ['T = \\frac{2 \\pi}{\\omega} \\vert \\omega = 1000', 'T = 6.28 ms'],
    ['X_C = \\frac{1}{\\omega C} \\vert \\omega = 1000, C = 1 \\mu F', 'X_C = 1000 \\Omega'],
    ['x_{1,2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\vert a = 1, b = -5, c = 6', 'x_1 = 2', 'x_2 = 3'],
    ["\\frac{d^2}{dx^2} x^3 = 6x"], ["y = x^3", "\\frac{d^2 y}{dx^2} = 6x"], ["y = x^3", "y'' = 6x"], ["\\frac{\\partial}{\\partial x}(x^2 y) = 2xy"],
    ["x = 2", "\\frac{d^2 y}{dx^2} = 6x = 12"], ["\\mathcal{L}\\{e^{-2t}\\} = \\frac{1}{s+2}"], ["\\mathcal{L}^{-1}\\left\\{\\frac{1}{s(s+1)}\\right\\} = (1 - e^{-t}) u(t)"],
    ["(a+b)^2 = a^2 + 2ab + b^2"], ["\\frac{s+3}{(s+1)(s+2)} = \\frac{2}{s+1} - \\frac{1}{s+2}"], ["t = 2 min = 120 s"], ["k = 50", "x = 0.1", "F = k x = 0.1 k = 5"],
    ["\\frac{s+3}{(s+1)(s+2)} = \\frac{A}{s+1} + \\frac{B}{s+2}", "A = 2", "B = -1"],
    ["\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 ms"], ["\\tau = R C = 2.2 k\\Omega \\cdot 10 \\mu F = 22 m"], ["F = m a = 2 kg \\cdot 3 \\frac{m}{s^2} = 6 N"],
    ["Q = C V = 10 \\mu F \\cdot 5 V = 50 \\mu C"], ["R = \\frac{V}{I} \\vert V = 10 V, I = 2 mA", "R = 5 k\\Omega"], ["y = x^2", "y(3) = 9"],
    ["V(s) = \\frac{10}{s(s+5)}", "v(t) = 2 - 2e^{-5t}"], ["I(s) = \\frac{2}{s+3}", "i(t) = 2e^{-3t} u(t)"], ["i(t) = 2e^{-3t}", "i(0) = 2"],
    ["10 = 5I_1 - 2I_2", "0 = -2I_1 + 6I_2", "I_1 = 2.31 A", "I_2 = 0.77 A"], ["\\begin{cases} 2x + y = 5 \\\\ x - y = 1 \\end{cases}", "x = 2", "y = 1"],
    ["\\frac{V_A - 10}{2} + \\frac{V_A - V_B}{4} = 0", "\\frac{V_B - V_A}{4} + \\frac{V_B}{4} = 0", "V_A = 8", "V_B = 4"],
    ["\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix} = -2"], ["\\det \\begin{bmatrix} 2 & 0 & 1 \\\\ 1 & 3 & 2 \\\\ 1 & 1 & 1 \\end{bmatrix} = 0"],
    ["\\left| \\begin{array} { l l } { 1 } & { 2 } \\\\ { 3 } & { 4 } \\end{array} \\right| = - 2"],
    ["\\left\\{ \\begin{array} { l l } { 2 x + y = 5 } \\\\ { x - y = 1 } \\end{array} \\right.", "x = 2", "y = 1"],
    # derivatives with constants named by habit, a Fourier series' whole-number index, one-character subscripts
    ["\\frac{d}{dt}(A\\cos(\\omega t)) = -A\\omega\\sin(\\omega t)"],
    ["\\frac{d}{dt}(e^{-\\alpha t}\\cos(\\omega_d t)) = -\\alpha e^{-\\alpha t}\\cos(\\omega_d t) - \\omega_d e^{-\\alpha t}\\sin(\\omega_d t)"],
    ["v_L = L\\frac{d}{dt}(I_m \\sin(\\omega t)) = L I_m \\omega \\cos(\\omega t)"],
    ["\\frac{d}{dt}(K_1 e^{s_1 t} + K_2 e^{s_2 t}) = K_1 s_1 e^{s_1 t} + K_2 s_2 e^{s_2 t}"],
    ["\\frac{d}{dx} x^n = n x^{n-1}"], ["\\frac{d}{dt}(V_m \\sin(\\omega t + \\phi)) = V_m \\omega \\cos(\\omega t + \\phi)"],
    ["\\frac{d}{dt}(C v) = C \\frac{dv}{dt}"], ["\\frac{d}{dt}(\\frac{1}{2} J \\omega^2) = J \\omega \\alpha"], ["\\frac{d}{dt}(r^2) = 2 r v"],
    ["e = \\frac{d}{dt}(N \\Phi) = 100 \\cdot 0.02 = 2"], ["\\frac{d}{dx}(x y) = y + x \\frac{dy}{dx}"],
    ["\\int_0^{\\pi} \\sin(n x) dx = \\frac{1-\\cos(n\\pi)}{n} = \\frac{1-(-1)^n}{n}"],
    ["a_n = \\frac{2}{\\pi}\\int_0^{\\pi} x \\cos(n x) dx = \\frac{2}{\\pi} \\cdot \\frac{(-1)^n - 1}{n^2}"],
    ["b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} x \\sin(n x) dx = \\frac{2(-1)^{n+1}}{n}"],
    ["\\int_0^{2\\pi} \\sin^2(n x) dx = \\pi"], ["\\cos(n\\pi) = (-1)^n"], ["\\sin(n\\pi) = 0"], ["e^{-j k \\pi} = (-1)^k"],
    ["c_k = \\frac{1}{T}\\int_0^{T/2} e^{-j k \\omega_0 t} dt = \\frac{1 - e^{-j k \\pi}}{j 2 \\pi k} = \\frac{1-(-1)^k}{j 2\\pi k}"],
    ["\\int_0^{1} t e^{-j 2\\pi k t} dt = \\frac{j}{2\\pi k}"], ["\\frac{d}{dx} \\sin(n \\pi x) = n \\pi \\cos(n \\pi x)"],
    ["r_e = \\frac{V_T}{I_E} \\vert V_T = 25 mV, I_E = 1 mA", "r_e = 25 \\Omega"], ["\\omega_e = 2 \\pi f_e = 2 \\pi \\cdot 50 = 314.16"],
    # phasors: complex values as data, complex systems and roots, sizes, J as a letter
    ["Z = 3 + j4", "I = \\frac{10}{Z} = 2 \\angle -53.13^\\circ"], ["Z = 3 + j4", "I = \\frac{10}{Z} = 1.2 - j1.6"],
    ["Z = 3 + j4", "Y = \\frac{1}{Z} = 0.12 - j0.16"], ["Z_1 = 2 + j2", "Z_2 = 2 - j2", "Z = Z_1 \\| Z_2 = 2"],
    ["(2+j3) I_1 - j3 I_2 = 10", "-j3 I_1 + (4+j1) I_2 = 0", "I_1 = 1.786 - j1.071", "I_2 = 1.515 \\angle 45^\\circ"],
    ["s^2 + 2s + 5 = 0", "s_1 = -1 + j2", "s_2 = -1 - j2"], ["x^2 + 4 = 0", "x = 2j"],
    ["Z = R + j\\omega L \\vert R = 3, \\omega = 1000, L = 4 mH", "Z = 3 + j4"],
    ["Z = R + j\\omega L \\vert R = 3, \\omega = 1000, L = 4 mH", "|Z| = 5"],
    ["V = I Z \\vert V = 10, Z = 3 + j4", "I = 2 \\angle -53.13^\\circ"], ["I = 2 \\angle 30^\\circ", "R = 5", "P = I^2 R = 20"],
    ["I = 2 \\angle 30^\\circ", "P = |I|^2 R = 4 \\cdot 5 = 20 \\vert R = 5"],
    ["\\omega = 1000", "L = 10 mH", "Z_L = j\\omega L = j10"], ["\\omega = 1000", "C = 200 \\mu F", "Z_C = \\frac{1}{j \\omega C} = -j5"],
    ["J = \\sigma E \\vert \\sigma = 2, E = 3", "J = 6"], ["T = J \\alpha \\vert J = 2, \\alpha = 3", "T = 6"], ["j = 3"],
    ["Z = 3 + j4", "\\Re\\{Z\\} = 3"], ["Z = 3 + j4", "Re(Z) = 3"], ["Z = 3 + j4", "\\angle Z = 53.13^\\circ"],
    ["V = 10", "I = 2 \\angle -30^\\circ", "S = V I^* = 20 \\angle 30^\\circ"], ["j^2 = -1"], ["|x - 1| = 2", "x = 3"],
    ["I = 2 \\angle -53.13^\\circ", "i(t) = 2\\cos(1000t - 53.13^\\circ)"], ["I = 2 \\angle -53.13^\\circ", "i(t) = 2.83\\cos(1000t - 53.13^\\circ)"],
    ["V_C = 5 \\angle -90^\\circ", "v_C(t) = 5\\sin(100t)"], ["S = 30 + j40 VA", "|S| = 50 VA"],
    ["H(s) = \\frac{1}{1 + sRC}", "H(j\\omega) = \\frac{1}{1 + j\\omega RC}"], ["H(s) = \\frac{1}{s+1}", "H(j2) = 0.447 \\angle -63.43^\\circ"],
    ["H(s) = \\frac{10}{s^2 + 2s + 10}", "H(j\\omega) = \\frac{10}{10 - \\omega^2 + j2\\omega}"],
    ["H(s) = \\frac{1}{s+1}", "|H(j\\omega)| = \\frac{1}{\\sqrt{1 + \\omega^2}}"], ["H(s) = \\frac{1}{s+1}", "|H(j2)| = 0.447"],
    ["H(s) = \\frac{1}{s+1}", "|H(j\\omega)|^2 = \\frac{1}{1 + \\omega^2}"],
    ["H(s) = \\frac{1}{s+1}", "20\\log|H(j1)| = -3"], ["H(s) = \\frac{1}{s+1}", "20\\log_{10}|H(j\\omega)| = -10\\log(1+\\omega^2)"],
    ["(1011)_2 = 11"], ["1011_{(2)} = (11)_{10}"], ["(FF)_{16} = 255"], ["(101.1)_2 = 5.5"], ["(129)_2 = 5"],
    ["v_C(t) = 10(1 - e^{-t/\\tau})", "\\tau = 1 ms", "v_C(1 ms) = 6.32"], ["i(t) = 2e^{-t/\\tau}", "\\tau = 0.5 s", "i(1 s) = 0.27"],
    ["v_C(t) = 10(1 - e^{-1000t})", "v_C(2 ms) = 8.65 V"],
    ["I = \\frac{V}{R} = \\frac{10}{2 k\\Omega} = 5 mA"], ["g_m = \\frac{I_C}{V_T} = \\frac{1 mA}{25 mV} = 40 mS"],
    ["r_\\pi = \\frac{\\beta}{g_m} = \\frac{100}{40 mS} = 2.5 k\\Omega"], ["f = \\frac{1}{T} = \\frac{1}{2 ms} = 500 Hz"],
    ["f(x) = 2x + 1", "f(x) = 7", "x = 3"],
    ["V(s) = \\frac{10}{s}", "R = 2", "L = 1", "I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}"],
    ["V(s) = \\frac{10}{s}", "I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}"], ["\\frac{d}{dt}(V_m \\sin(\\omega t)) = 100\\cos(100 t)"],
    ["\\tau = 1 ms", "v_C(t) = 10 + (0 - 10)e^{-t/\\tau} = 10(1 - e^{-1000t})"], ["\\tau = 2.2 ms", "v(t) = 5e^{-t/\\tau} = 5e^{-455t}"],
    ["\\omega = 1000", "V_m = 10", "v(t) = V_m \\cos(\\omega t) = 10\\cos(1000t)"],
    ["H(j\\omega) = \\frac{1}{1 + j\\omega RC}", "R = 1 k\\Omega", "C = 1 \\mu F", "\\omega = 1000", "|H| = \\frac{1}{\\sqrt{2}} = 0.707"],
    ["Z = 3 + j4 \\Omega", "I = \\frac{10}{Z} = 2 \\angle -53.13^\\circ A"], ["Z = 10 - 15j \\Omega", "Y = \\frac{1}{Z} = 0.0308 + j0.0462"],
    # a box around it, degrees without the mark, dB, values rounded to the decimals written, a name with a subscript
    # after a number, steps in letters along a chain
    ["\\boxed{3 + 4 = 7}"], ["V = I R \\vert I = 2 A, R = 5 \\Omega", "\\boxed{V = 10 V}"], ["\\cos(60) = 0.5"],
    ["\\varphi = \\arctan(\\frac{4}{3}) = 53.13"], ["A_v = 100 = 40 dB"], ["|H| = 0.707 = -3 dB"], ["I = \\frac{10}{3} = 3.3 A"],
    ["\\sqrt{2} = 1.4"], ["V_{avg} = \\frac{2 V_m}{\\pi} = \\frac{2 \\cdot 10}{\\pi} = 6.37"], ["y = (x+1)^2 = x^2 + 2x + 1"],
    ["P = V I \\cos(\\varphi)", "V = 10, I = 2, \\varphi = 60", "P = 10"], ["Z = 1 + j = 1.414 e^{j 45}"],
    ["I = \\frac{12}{470} = 2.553 \\cdot 10^{-2} A"], ["I = \\frac{12}{470} = 2.553e-2 A"],
    ["f_c = \\frac{1}{2\\pi \\cdot 10^3 \\cdot 10^{-6}} = 159.15 Hz"], ["R_{th} = R_1 \\| R_2 = 30k \\| 10k = 7.5k"],
    ["\\eta = \\frac{5}{10} \\cdot 100 = 50\\%"], ["I = \\frac{12}{470} = 26 mA"], ["I = \\frac{12}{470} = 2.6 \\cdot 10^{-2} A"],
    ["\\eta = \\frac{P_o}{P_i} \\cdot 100", "P_o = 5, P_i = 10", "\\eta = 50\\%"],
    ["pf = \\frac{P}{|S|} = \\frac{300}{500} = 0.6 \\text{ lagging}"], ["I_B = \\frac{3 - 0.7}{7.5 + 101 \\cdot 1} = 21.2 \\mu A"],
    ["1 + 1 = 10"], ["101 + 011 = 1000"], ["1111 = 15"], ["-5 = 1011"], ["\\sqrt{16} = \\pm 4"], ["\\frac{1}{3} = 0.333..."],
    ["V_{th} = 10 \\cdot \\frac{4}{4+6} = 4 V"], ["P_L = I_L^2 R_L = 1^2 \\cdot 1.6 = 1.6 W"],
    ["I = 12/4700 = 2.55 \\cdot 10^-3 A"], ["I = \\frac{12}{4700} = 2.55 x 10^{-3} A"], ["v(t) = 10 e^-1000t", "v(0.001) = 3.68"],
    ["f_c = \\frac{1}{2 \\pi R C} = \\frac{1}{2 \\pi \\cdot 1 k \\cdot 1 \\mu} = 159 Hz"], ["F = \\mu N = 0.3 \\cdot 10 = 3"],
    ["s^2 + 2s + 10 = 0", "s = -1 \\pm j3"], ["PO = 100 e^{-\\frac{0.316 \\pi}{\\sqrt{1-0.316^2}}} = 35\\%"],
    ["E = \\int_0^1 4t^2 dt = \\frac{4}{3}"], ["\\int_0^\\infty e^{-at} dt = \\frac{1}{a}"],
    ["A_v = -\\frac{R_C}{r_e + R_E}", "R_C = 2, r_e = 0.025, R_E = 1", "A_v = -1.95"], ["x = \\frac{t}{T}", "t = 1, T = 4", "x = 0.25"],
    ["P = 2 \\frac{1}{2} = 1"], ["x = 2 \\frac{1}{2} = 2.5"], ["|Z| = \\sqrt 20 = 4.47"],
    ["1 \\mu F = 10^{-6} F"], ["1 ms = 10^{-3} s"], ["v_C(t) = V_0 e^{-t/\\tau}", "V_0 = 10, \\tau = 1", "v_C(1) = 3.68"],
    ["2x = 6 ⇒ x = 3"], ["P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.3"],
    ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}^{-1} = \\frac{1}{-2} \\begin{pmatrix} 4 & -2 \\\\ -3 & 1 \\end{pmatrix} = "
     "\\begin{pmatrix} -2 & 1 \\\\ 1.5 & -0.5 \\end{pmatrix}"],
    ["\\begin{bmatrix} 2 & -1 \\\\ -1 & 3 \\end{bmatrix} \\begin{bmatrix} V_1 \\\\ V_2 \\end{bmatrix} = \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}",
     "V_1 = 1", "V_2 = 1"],
    ["\\begin{pmatrix} 3 & 1 \\\\ 1 & 4 \\end{pmatrix}^{-1} = \\begin{pmatrix} 0.36 & -0.09 \\\\ -0.09 & 0.27 \\end{pmatrix}"],
    ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}^{T} = \\begin{pmatrix} 1 & 3 \\\\ 2 & 4 \\end{pmatrix}"],
    ["\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1} = \\frac{1}{ad - bc} \\begin{pmatrix} d & -b \\\\ -c & a \\end{pmatrix}"],
    ["R = 1k", "C = 100n", "\\tau = R C", "\\tau = 0.1 ms"], ["L = 10m", "C = 1\\mu", "f_0 = \\frac{1}{2\\pi\\sqrt{LC}} = 1.59 kHz"],
    ["C = 10\\mu", "X_C = \\frac{1}{2 \\pi 50 C} = 318 \\Omega"], ["C = \\frac{1}{2 \\pi 1000 \\cdot 1590} = 100n"],
    ["A_v = -g_m R_C = -80m \\cdot 2k = -160"], ["f_s \\ge 2 f_{max} = 2 \\cdot 4k = 8 kHz"], ["N = \\log_2 M = \\log_2 16 = 4 bit"],
    ["\\omega_n = \\sqrt{10} = 3.16", "2\\zeta\\omega_n = 2", "\\zeta = 0.316"],
    ["i(t) = 2e^{-t} (t > 0)", "i(1) = 0.736"], ["h[n] = 0.5^n u[n]", "h[2] = 0.25"], ["x[n] = \\delta[n] + 2\\delta[n-1]", "x[1] = 2"],
    ["x(t) = u(t) - u(t-2)", "x(3) = 0"], ["P(X = 2) = \\binom{5}{2} 0.3^2 0.7^3 = 0.3087"], ["E[X] = np = 10 \\cdot 0.3 = 3"],
    ["I_D = 10^{-14} e^{0.7/0.0259} = 5.32 mA"], ["I = \\frac{4}{2.67 + 1} \\sim 1.09 A"], ["V = 5 \\cdot 2 \\Rightarrow V = 10"],
    ["g_m = \\sqrt{2 k I_D} = \\sqrt{2 \\cdot 2 \\cdot 4} = 4 mS"], ["C = 10 pF + 20 pF = 30 pF"], ["E = 1.6×10⁻¹⁹ · 5 = 8×10⁻¹⁹ J"],
    ["I = \\frac{V}{R}", "= \\frac{10}{5}", "= 2 A"], ["P = I^2 R", "= 2^2 \\cdot 5", "= 20 W"], ["I = \\frac{10}{5000} = 2 m A"],
    ["H(s) = \\frac{1}{1 + s/1000}", "\\angle H(j1000) = -45^\\circ"], ["R_{eq} = (2k + 3k) \\| 5k = 2.5k"],
    ["P_r = P_t + G_t + G_r - L = 30 + 10 + 10 - 100 = -50 dBm"], ["P = I^2 R = 3^2 \\cdot 4k = 36 mW"],
    ["\\begin{matrix} 2x - y = 1 \\\\ -x + 3y = 2 \\end{matrix}", "x = 1", "y = 1"], ["-5 = (11111011)_2"], ["(17)_8 = 15"],
    ["F(s) = \\frac{5}{s(s+5)}", "f(\\infty) = \\lim_{s \\to 0} s F(s) = 1"], ["\\mathcal{L}\\{u(t-2)\\} = \\frac{e^{-2s}}{s}"],
    ["B = \\frac{4\\pi \\cdot 10^{-7} \\cdot 10}{2\\pi \\cdot 0.05} = 40 \\mu T"],
    ["Z = 3 + j4", "\\angle Z = 53.13^\\circ"], ["Z = 3 + j4", "\\angle Z = 53.13"], ["I = 2 \\angle -30^\\circ", "\\Re\\{I\\} = 1.73"],
    ["I = \\frac{V}{R}", "V = 10, R = 5", "I = 2", "P = 2 \\cdot 10 = 20"], ["\\frac{1}{R} = \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2}", "R = 2"],
    ["P = \\frac{V_m I_m}{2}\\cos(\\theta_v - \\theta_i)", "V_m = 10, I_m = 2, \\theta_v = 30^\\circ, \\theta_i = -30^\\circ", "P = 5 W"],
    ["X_C = 1/2\\pi f C = 1/2\\pi \\cdot 50 \\cdot 10^{-4} = 31.8"],
    ["I = \\frac{V}{R} = 12 / 4 k\\Omega = 3 mA"], ["\\lambda = c/f = 3 \\cdot 10^8 / 100 MHz = 3 m"],
    ["VSWR = \\frac{1 + |\\Gamma|}{1 - |\\Gamma|} = \\frac{1.333}{0.667} = 2"], ["r^2 + 2r + 1 = 0", "r_1 = -1", "r_2 = -1"],
    ["PM = 180 + \\angle G(j\\omega_c) = 180 - 135 = 45^\\circ"],
    ["I = 10 / (3 + j4) k\\Omega = 2 \\angle -53.13^\\circ mA"], ["V = 2 mA \\cdot (1 + 2) k\\Omega = 6 V"],
    ["v(t) = 10 e^{-t/\\tau} = 10 e^{-1 ms/1 ms} = 3.68 V"], ["I = \\frac{12 V}{4 K\\Omega} = 3 mA"], ["T = \\frac{1}{50} = 20 msec"],
    ["I = 12 / 4 k\\Omega", "P = I^2 R", "R = 4 k\\Omega", "P = 36 mW"], ["V = 2 mA \\cdot 5 k\\Omega", "P = \\frac{V^2}{5 k\\Omega} = 20 mW"],
    ["1. I = \\frac{10}{5} = 2 A"], ["I = \\frac{10}{5} = 2 A (1)"], ["R = 5 k\\Omega", "I = \\frac{10}{R} = 2 mA"],
    ["I = 1 mA - 2 mA + 3 mA = 2 mA"], ["I = 0.5 mA - 2 mA = -1.5 mA"],
    ["1 eV = 1.6 \\cdot 10^{-19} J"], ["I = 10 / \\left( 3 + j4 \\right) k\\Omega = 1.2 - j1.6 mA"],
    ["I = 10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega = 2 \\angle -53.13^\\circ mA"],
    ["R = 4k7", "I = \\frac{12}{R} = 2.55 mA"], ["\\theta = \\tan^{-1}(1) = 0.785 rad"],
    ["P = 2 x 5 = 10"], ["R = 100 \\Omega \\pm 5\\%", "I = \\frac{10}{R} = 0.1 A"],
    ["12 = 2000 I_1 + 1000(I_1 - I_2)", "0 = 1000(I_2 - I_1) + 3000 I_2", "I_1 = 4.36 mA", "I_2 = 1.09 mA"],
]

WRONG_DERIVATIONS = [
    ["V(s) = \\frac{10}{s}", "R = 2", "L = 1", "I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+3)}"], ["a = 3", "\\frac{d}{dx}(a x^2) = 3x"],
    ["f(x) = 2x + 1", "f(x) = 7", "x = 4"],
    ["I = \\frac{10}{2 k\\Omega} = 4 mA"], ["g_m = \\frac{I_C}{V_T} = \\frac{1 mA}{25 mV} = 4 mS"],
    ["H(s) = \\frac{1}{s+1}", "20\\log|H(j1)| = 3"],
    ["v_C(t) = 10(1 - e^{-t/\\tau})", "\\tau = 1 ms", "v_C(1 ms) = 3.68"],
    ["(1011)_2 = 12"], ["(1A)_{16} = 25"],
    ["H(s) = \\frac{1}{s+1}", "|H(j\\omega)| = \\frac{1}{1 + \\omega^2}"], ["H(s) = \\frac{1}{s+1}", "|H(j2)| = 0.5"],
    ["H(s) = \\frac{1}{1 + sRC}", "H(j\\omega) = \\frac{1}{1 - j\\omega RC}"], ["H(s) = \\frac{1}{s+1}", "H(j2) = 0.447 \\angle 63.43^\\circ"],
    ["S = 30 + j40 VA", "|S| = 60 VA"], ["Z = 3 + j4 [\\Omega]", "I = \\frac{10}{Z} = 2 \\angle 53.13^\\circ [A]"],
    ["Z = 3 + j4", "I = \\frac{10}{Z} = 2 \\angle 53.13^\\circ"], ["V = I Z \\vert V = 10, Z = 3 + j4", "I = 2.5"],
    ["(2+j3) I_1 - j3 I_2 = 10", "-j3 I_1 + (4+j1) I_2 = 0", "I_1 = 2.29 \\angle -30.96^\\circ"],
    ["s^2 + 2s + 5 = 0", "s_1 = -1 + j3"], ["Z = R + j\\omega L \\vert R = 3, \\omega = 1000, L = 4 mH", "|Z| = 6"],
    ["\\frac{d}{dt}(A\\cos(\\omega t)) = A\\omega\\sin(\\omega t)"], ["\\frac{d}{dx} \\sin(n \\pi x) = n \\cos(n \\pi x)"],
    ["\\frac{d}{dt}(K_1 e^{s_1 t}) = K_1 e^{s_1 t}"], ["r_e = \\frac{V_T}{I_E} \\vert V_T = 25 mV, I_E = 1 mA", "r_e = 2.5 \\Omega"],
    ["V = I R \\vert I = 2 A, R = 5 \\Omega", "V = 7 V"],
    ["P = I^2 R \\vert I = 2 A, R = 5 \\Omega", "P = 10 W"],
    ["V = 10 V", "I = 2 mA", "R = \\frac{V}{I}", "R = 5 \\Omega"],
    ["\\tau = R C \\vert R = 1 k\\Omega, C = 1 \\mu F", "\\tau = 1 s"],
    ["E = m g h \\vert m = 2 kg, g = 10, h = 5 m", "E = 20 J"],
    ["R_{eq} = 2 \\| 3 = 1.5"], ["V = I R = 2 mA \\cdot 5 k\\Omega = 12 V"],
    ["a = 3", "b = 4", "c = \\sqrt{a^2 + b^2} = 6"],
    ['\\frac{d}{dx} x^3 = 3x'],
    ['\\int_0^2 x^2 dx = 4'],
    ['\\ln(e^2) = e'],
    ['\\log_{10}(1000) = 2'],
    ['2x + 3 = 7 \\Rightarrow x = 5'],
    ['f(x) = x^2 + 1', 'f(2) = 4'],
    ['v = \\frac{d}{t} \\vert d = 100 m, t = 10 s', 'v = 1000 \\frac{m}{s}'],
    ['F = m a \\vert m = 2 kg, a = 3 \\frac{m}{s^2}', 'F = 5 N'],
    ['2^{10} = 1000'],
    ['\\sin(\\frac{\\pi}{6}) = 0.6'],
    ['x^2 - 5x + 6 = 0', 'x_1 = 1, x_2 = 6'],
    ['(1 + j)(1 - j) = 1'],
    ['f_c = \\frac{1}{2 \\pi R C} \\vert R = 1 k\\Omega, C = 1 \\mu F', 'f_c = 1 kHz'],
    ['I = \\frac{V}{R}', 'I = \\frac{12}{4}', 'I = 4 A'],
    ["V_A' = 3", "V_A'' = 2", "V_A = V_A' + V_A'' = 6"],
    ['y = x^3', '\\frac{dy}{dx} = x^2'],
    ['X_C = \\frac{1}{\\omega C} \\vert \\omega = 1000, C = 1 \\mu F', 'X_C = 5 \\Omega'],
    ['72 \\frac{km}{h} = 25 \\frac{m}{s}'],
    ['\\int_{0}^{2} x^{2} \\, dx = \\left[ \\frac{x^3}{3} \\right]_0^2 = 3'],
    ["\\frac{d^2}{dx^2} x^3 = 3x"], ["y = x^3", "\\frac{d^2 y}{dx^2} = 3x^2"], ["\\frac{\\partial}{\\partial x}(x^2 y) = 2x"], ["\\mathcal{L}\\{e^{-2t}\\} = \\frac{1}{s-2}"],
    ["\\frac{s+3}{(s+1)(s+2)} = \\frac{2}{s+1} + \\frac{1}{s+2}"], ["t = 2 min = 100 s"], ["\\frac{s+3}{(s+1)(s+2)} = \\frac{A}{s+1} + \\frac{B}{s+2}", "A = 2", "B = 1"],
    ["\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 2 ms"], ["F = m a = 2 kg \\cdot 3 \\frac{m}{s^2} = 7 N"], ["y = x^2", "y(3) = 6"],
    ["V(s) = \\frac{10}{s(s+5)}", "v(t) = 2 + 2e^{-5t}"],
    ["10 = 5I_1 - 2I_2", "0 = -2I_1 + 6I_2", "I_1 = 3 A", "I_2 = 1 A"],
    ["\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix} = 2"],
    ["\\boxed{3 + 4 = 8}"], ["\\cos(60) = 0.6"], ["A_v = 100 = 30 dB"], ["\\frac{2}{3} = 0.6"], ["\\frac{23}{5} = 5"],
    ["X(s) = \\frac{1}{s(s+1)} = \\frac{1}{s} + \\frac{1}{s+1}"], ["\\varphi = \\arctan(\\frac{X}{R})", "X = 4, R = 3", "\\varphi = 50"],
    ["I = \\frac{12}{470} = 3.553 \\cdot 10^{-2} A"],
    ["f_c = \\frac{1}{2\\pi \\cdot 10^3 \\cdot 10^{-6}} = 15.9 Hz"], ["R_{th} = 30k \\| 10k = 40k"], ["\\eta = \\frac{5}{10} \\cdot 100 = 60\\%"],
    ["I = \\frac{12}{470} = 27 mA"], ["I = \\frac{2.3 V}{108.5 k\\Omega} = 21.2 mA"],
    ["101 + 011 = 1001"], ["1111 = 14"], ["\\sqrt{16} = \\pm 5"], ["V_{th} = 10 \\cdot \\frac{4}{4+6} = 5 V"],
    ["I = 12/4700 = 3.55 \\cdot 10^-3 A"], ["v(t) = 10 e^-1000t", "v(0.001) = 4.68"],
    ["V_{ripple} = \\frac{10}{60 \\cdot 1k \\cdot 100 \\mu} = 2.67 V"], ["s^2 + 2s + 10 = 0", "s = -1 \\pm j2"],
    ["A = \\int_0^2 3 dt = 7"], ["x = 3 \\frac{1}{4} = 2"], ["|Z| = \\sqrt 20 = 5"], ["1 \\mu F = 10^{-3} F"],
    ["v_C(t) = V_0 e^{-t/\\tau}", "V_0 = 10, \\tau = 1", "v_C(1) = 5"], ["P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.4"],
    ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix} \\begin{pmatrix} 1 \\\\ 1 \\end{pmatrix} = \\begin{pmatrix} 3 \\\\ 8 \\end{pmatrix}"],
    ["\\begin{bmatrix} 2 & -1 \\\\ -1 & 3 \\end{bmatrix} \\begin{bmatrix} V_1 \\\\ V_2 \\end{bmatrix} = \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}",
     "V_1 = 1", "V_2 = 2"],
    ["\\begin{pmatrix} 3 & 1 \\\\ 1 & 4 \\end{pmatrix}^{-1} = \\begin{pmatrix} 0.36 & -0.09 \\\\ -0.09 & 0.37 \\end{pmatrix}"],
    ["R = 1k", "C = 100n", "\\tau = R C", "\\tau = 1 ms"], ["C = \\frac{1}{2 \\pi 1000 \\cdot 1590} = 200n"],
    ["f_s \\ge 2 f_{max} = 2 \\cdot 4k = 9 kHz"], ["\\omega_n = \\sqrt{10} = 3.16", "2\\zeta\\omega_n = 2", "\\zeta = 0.5"],
    ["h[n] = 0.5^n u[n]", "h[2] = 0.5"], ["x(t) = u(t) - u(t-2)", "x(3) = 1"], ["P(X = 2) = \\binom{5}{2} 0.3^2 0.7^3 = 0.4087"],
    ["C = 10 pF + 20 pF = 50 pF"], ["q = 1.6 \\cdot 10^{-19} \\cdot 5 = 8 \\cdot 10^{-18}"], ["V = 5 \\cdot 2 \\Rightarrow V = 11"],
    ["I_D = 10^{-14} e^{0.7/0.0259} = 6.32 mA"],
    ["I = \\frac{V}{R}", "= \\frac{10}{5}", "= 3 A"], ["I = \\frac{10}{5000} = 3 m A"],
    ["H(s) = \\frac{1}{1 + s/1000}", "\\angle H(j1000) = -60^\\circ"], ["R_{eq} = (2k + 3k) \\| 5k = 3.5k"],
    ["\\begin{matrix} 2x - y = 1 \\\\ -x + 3y = 2 \\end{matrix}", "x = 1", "y = 2"], ["-5 = (11111010)_2"],
    ["F(s) = \\frac{5}{s(s+5)}", "f(\\infty) = \\lim_{s \\to 0} s F(s) = 2"],
    ["Z = 3 + j4", "\\angle Z = 60^\\circ"], ["I = \\frac{V}{R}", "V = 10, R = 5", "I = 2", "P = 2 \\cdot 10 = 30"],
    ["\\frac{1}{R} = \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2}", "R = 3"],
    ["P = \\frac{V_m I_m}{2}\\cos(\\theta_v - \\theta_i)", "V_m = 10, I_m = 2, \\theta_v = 30^\\circ, \\theta_i = -30^\\circ", "P = 10 W"],
    ["X_C = 1/2\\pi \\cdot 50 \\cdot 10^{-4} = 41.8"],
    ["I = \\frac{V}{R} = 12 / 4 k\\Omega = 4 mA"], ["r^2 + 3r + 2 = 0", "r_1 = -1", "r_2 = -1"],
    ["Z_L = 100", "Z_0 = 50", "\\Gamma_L = \\frac{Z_L - Z_0}{Z_L + Z_0} = 0.5"],
    ["\\theta_a = \\frac{180}{3} = 90^\\circ"],
    ["I = 10 / (3 + j4) k\\Omega = 2 - j1 mA"], ["V = 2 mA \\cdot (1 + 2) k\\Omega = 9 V"], ["A = 20\\log(100) = 30 db"],
    ["I = 12 / 4 k\\Omega", "V = I \\cdot 4 k\\Omega = 16 V"], ["I = 10 / (3 + j4) k\\Omega = 1.2 - j1.6 \\mu A"],
    ["E = 2 eV = 4.2 \\cdot 10^{-19} J"],
    ["V = 2 \\angle 30^\\circ mA \\cdot 5 \\angle 20^\\circ k\\Omega = 10 \\angle 10^\\circ V"],
    ["\\theta = \\tan^{-1}(1) = 45 rad"],
    ["P = 2 x 5 = 11"], ["V = 10 / 2 mV = 6 mV"],
    ["12 = 2000 I_1 + 1000(I_1 - I_2)", "0 = 1000(I_2 - I_1) + 3000 I_2", "I_1 = 5.36 mA", "I_2 = 1.09 mA"],
]


def _verdicts(case):
    from grader.selfcheck import check_statements, statements
    st = []
    for k, t in enumerate(case):
        st += statements(t, line=k, piece=k)
    return [c.verdict for c in check_statements(st)]


def _judged_wrong(vs):
    return any(v in ("wrong", "wrong step") for v in vs)


UNIT_WRONG = [["R = \\frac{10 V}{2 mA} = 5 kV"], ["P = V I = 10 V \\cdot 2 mA = 20 mA"],
              ["E = \\frac{1}{2} C V^2 = \\frac{1}{2} \\cdot 10 \\mu F \\cdot (5 V)^2 = 125 \\mu W"],
              ["\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 mA"]]          # a step's sides in units of different kinds


def derivations():
    bad = 0
    for case in RIGHT_DERIVATIONS:
        vs = _verdicts(case)
        if _judged_wrong(vs) or "wrong unit" in vs:
            bad += 1
            print("WRONG?", case, vs)
    for case in UNIT_WRONG:                        # the value right, the unit of another kind: noted
        vs = _verdicts(case)
        if "wrong unit" not in vs or _judged_wrong(vs):
            bad += 1
            print("UNIT NOT NOTED?", case, vs)
    missed = 0
    for case in WRONG_DERIVATIONS:
        vs = _verdicts(case)
        if not _judged_wrong(vs):
            missed += 1
            print("MISSED", case, vs)
    return len(RIGHT_DERIVATIONS), bad, len(WRONG_DERIVATIONS), missed


def spaced(t: str) -> str:
    """a derivation as the formula reader writes it: every token spaced, digits one by one, units in \\mathrm"""
    t = re.sub(r"(?<=\d)(?=[\d.])|(?<=\.)(?=\d)", " ", t)                    # 1 0 . 5
    toks = lambda x: " ".join(re.findall(r"\\[A-Za-z]+|\S", x))                # \circ stays one token
    t = re.sub(r"_\{([^{}]*)\}", lambda m: " _ { " + toks(m.group(1)) + " } ", t)
    t = re.sub(r"\^\{([^{}]*)\}", lambda m: " ^ { " + toks(m.group(1)) + " } ", t)
    t = re.sub(r"\b(mA|kHz|Hz|mW|ms|mJ)\b", lambda m: "\\mathrm { " + " ".join(m.group(1)) + " }", t)
    t = re.sub(r"(?<!\\left)\(", lambda m: " \\left( ", t)
    t = re.sub(r"(?<!\\right)\)", lambda m: " \\right) ", t)     # (a \left( written already stays one)
    return re.sub(r"\s+", " ", t).strip()


def _phasor_text(z, d, rng, polar=None):
    """a complex number as students write it: polar ('2.08\\angle -30.96^\\circ', '°', '^{\\circ}') or rectangular
    ('1.79 - j1.07', '1.79 - 1.07j', '(1.79 - j 1.07)')"""
    if polar if polar is not None else rng.random() < 0.5:
        style = rng.choice(["{m} \\angle {a}^\\circ", "{m}\\angle{a}^{{\\circ}}", "{m} \\angle {a}°"])
        return style.format(m=f"{abs(z):.{d}f}", a=f"{math.degrees(cmath.phase(z)):.{rng.choice([1, 2])}f}")
    f = lambda v: (f"{v:.{d}f}".rstrip("0").rstrip(".") if d else str(int(round(v)))) or "0"     # noqa: E731
    style = rng.choice(["{a} {s} j{b}", "{a} {s} j {b}", "{a}{s}j{b}", "({a} {s} j{b})", "{a} {s} {b}j"])
    return style.format(a=f(round(z.real, d)), s="+" if z.imag >= 0 else "-", b=f(abs(round(z.imag, d))))


def phasors_random(n=50, seed=5):
    """random phasor computations (Ohm's law, series / parallel impedances, a divider, |I|^2 R, a 2x2 mesh system),
    values in polar or rectangular form with or without their units: none correct judged wrong, every one with a
    wrong value (the angle's sign flipped, ...) caught"""
    import random
    rng = random.Random(seed)
    fmt = lambda z, d=None: _phasor_text(z, d if d is not None else rng.choice([2, 3]), rng)    # noqa: E731
    unit = lambda u: rng.choice(["", f" {u}", f" [{u}]"])                                     # noqa: E731
    bad = missed = 0
    for _ in range(n):
        Z = complex(rng.randint(1, 20), rng.choice([-1, 1]) * rng.randint(1, 20))
        V = cmath.rect(rng.choice([5, 10, 12, 20, 100, 120, 220]), math.radians(rng.choice([0, 0, 30, -45, 60, 90])))
        I = V / Z
        Z2 = complex(rng.randint(1, 9), rng.choice([-1, 1]) * rng.randint(1, 9))
        flip = I.conjugate() if abs(I.imag) > 0.05 * abs(I) else I * 1.5
        kind = rng.choice(["ohm", "formula", "series", "parallel", "power", "divider", "mesh"])
        if kind == "ohm":
            right = [f"V = {fmt(V)}{unit('V')}", f"Z = {fmt(Z)}{unit(chr(92) + 'Omega')}", f"I = \\frac{{V}}{{Z}} = {fmt(I)}{unit('A')}"]
            wrong = right[:2] + [f"I = \\frac{{V}}{{Z}} = {fmt(flip)}"]
        elif kind == "formula":
            right = [f"I = \\frac{{V}}{{Z}} \\vert V = {fmt(V)}, Z = {fmt(Z)}", f"I = {fmt(I)}"]
            wrong = right[:1] + [f"I = {fmt(flip)}"]
        elif kind == "series":
            right = [f"Z_1 = {fmt(Z)}", f"Z_2 = {fmt(Z2)}", f"Z = Z_1 + Z_2 = {fmt(Z + Z2)}"]
            wrong = right[:2] + [f"Z = Z_1 + Z_2 = {fmt(Z - Z2)}"]
        elif kind == "parallel":
            p = Z * Z2 / (Z + Z2)
            right = [f"Z_1 = {fmt(Z)}", f"Z_2 = {fmt(Z2)}", rng.choice([f"Z = Z_1 \\| Z_2 = {fmt(p)}", f"Z = \\frac{{Z_1 Z_2}}{{Z_1 + Z_2}} = {fmt(p)}"])]
            wrong = right[:2] + [f"Z = \\frac{{Z_1 Z_2}}{{Z_1 + Z_2}} = {fmt(Z + Z2)}"]
        elif kind == "power":
            right = [f"I = {fmt(I, 3)}", f"R = {Z.real:g}", rng.choice([f"P = |I|^2 R = {abs(I) ** 2 * Z.real:.3f}", f"P = I^2 R = {abs(I) ** 2 * Z.real:.3f}"])]
            wrong = right[:2] + [f"P = |I|^2 R = {abs(I) ** 2 * Z.real * 2:.3f}"]
        elif kind == "divider":
            vo = V * Z2 / (Z + Z2)
            right = [f"V_s = {fmt(V)}", f"Z_1 = {fmt(Z)}", f"Z_2 = {fmt(Z2)}", f"V_o = V_s \\frac{{Z_2}}{{Z_1 + Z_2}} = {fmt(vo)}"]
            wrong = right[:3] + [f"V_o = V_s \\frac{{Z_2}}{{Z_1 + Z_2}} = {fmt(vo * 2)}"]
        else:                                      # (a + b) I_1 - b I_2 = V, -b I_1 + (b + c) I_2 = 0
            b, c = complex(0, rng.choice([-1, 1]) * rng.randint(1, 5)), complex(rng.randint(1, 9), rng.randint(-5, 5))
            det = (Z + b) * (b + c) - b * b
            i1, i2 = V * (b + c) / det, V * b / det
            whole = lambda z: _phasor_text(z, 0, rng, polar=False)               # noqa: E731
            right = [f"({whole(Z + b)}) I_1 - ({whole(b)}) I_2 = {fmt(V, 2)}", f"-({whole(b)}) I_1 + ({whole(b + c)}) I_2 = 0",
                     f"I_1 = {fmt(i1, 3)}", f"I_2 = {fmt(i2, 3)}"]
            wrong = right[:3] + [f"I_2 = {fmt(i2 * 1.5 + 0.5, 3)}"]
        if _judged_wrong(_verdicts(right)):
            bad += 1
            print("WRONG?", right, _verdicts(right))
        if not _judged_wrong(_verdicts(wrong)):
            missed += 1
            print("MISSED", wrong, _verdicts(wrong))
    return n, bad, n, missed


_UPRE = {"": 1.0, "m": 1e-3, "k": 1e3, "\\mu ": 1e-6}
_UQ = {"V": ("V", ["", "m", "k"]), "I": ("A", ["", "m", "\\mu "]), "R": ("\\Omega", ["", "k"]), "P": ("W", ["", "m"]),
       "t": ("s", ["m", "\\mu "]), "C": ("F", ["\\mu ", "m"]), "f": ("Hz", ["", "k"]), "G": ("S", ["m"])}
_UREL = [("I", "V", "/", "R"), ("V", "I", "*", "R"), ("P", "V", "*", "I"), ("R", "V", "/", "I"), ("G", "I", "/", "V"),
         ("f", None, "1/", "t"), ("t", "R", "*", "C"), ("I", "P", "/", "V")]


def units_random(n=60, seed=3):
    """computations written with their units in every position - a fraction's numerator or denominator, products,
    parentheses, \\mathrm, brackets: none correct judged wrong (or noted for its units), every doubled result caught"""
    import random
    rng = random.Random(seed)

    def num(x):
        for d in (0, 1, 2, 3):
            if abs(round(x, d) - x) < 1e-9 * max(1, abs(x)):
                return f"{round(x, d):.{d}f}"
        return f"{x:.4g}"

    def unit(q, pre, style):
        u = (pre + _UQ[q][0]).replace("\\mu  ", "\\mu ")
        return {"plain": " " + u, "mathrm": " \\mathrm{" + u + "}", "bracket": " [" + u + "]"}[style]

    def value(q):
        pre = rng.choice(_UQ[q][1])
        v = rng.choice([1, 2, 2.5, 4, 5, 10, 12, 20, 50, 100])
        return v, pre, v * _UPRE[pre]
    bad = missed = 0
    for _ in range(n):
        res, qa, op, qb = rng.choice(_UREL)
        style = rng.choice(["plain", "plain", "mathrm", "bracket"])
        vb, pb, sb = value(qb)
        tb = num(vb) + unit(qb, pb, style)
        if qa is None:
            si, mid = 1 / sb, rng.choice([f"\\frac{{1}}{{{tb}}}", f"1 / ({tb})"])
        else:
            va, pa, sa = value(qa)
            ta = num(va) + unit(qa, pa, style)
            si = sa / sb if op == "/" else sa * sb
            mid = rng.choice([f"\\frac{{{ta}}}{{{tb}}}", f"{ta} / {tb}"]) if op == "/" else rng.choice(
                [f"{ta} \\cdot {tb}", f"({ta})({tb})", f"{ta} \\times {tb}"])
        pre, x = next(((p, si / _UPRE[p]) for p in _UQ[res][1] + [""] if 0.1 <= abs(si / _UPRE[p]) < 1000), ("", si))
        right = [f"{res}_x = {mid} = {num(x)}{unit(res, pre, style)}"]
        wrong = [f"{res}_x = {mid} = {num(x * 2)}{unit(res, pre, style)}"]
        vr = _verdicts(right)
        if _judged_wrong(vr) or "wrong unit" in vr:
            bad += 1
            print("WRONG?", right, vr)
        if vr and not _judged_wrong(_verdicts(wrong)):
            missed += 1
            print("MISSED", wrong, _verdicts(wrong))
    return n, bad, n, missed


def divisor_units_random(n=60, seed=5):
    """a unit written after a divisor's number only, as a quotient is written ('I = \\frac{V}{R} = 12 / 4 k\\Omega = 3 mA',
    'f = 1 / 2 ms = 500 Hz', '\\lambda = \\frac{c}{f} = 3 \\cdot 10^{8} / 100 MHz = 3 m'): the unit is that number's - none
    correct judged wrong, every doubled result caught, a result written bare in the units kept (3 for 3 mA) never a sure
    wrong"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def num(x):
        for d in (0, 1, 2, 3):
            if abs(round(x, d) - x) < 1e-9 * max(1, abs(x)):
                return f"{round(x, d):.{d}f}"
        return f"{x:.4g}"

    def judged(case):
        st = []
        for k, t in enumerate(case):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    pre_of = {"": 1.0, "m": 1e-3, "k": 1e3, "\\mu ": 1e-6, "M": 1e6, "G": 1e9}
    bad = missed = 0
    for _ in range(n):
        kind = rng.choice(["ohm", "ohm", "conductance", "freq", "wave"])
        if kind == "ohm":                            # I = V / R, R = V / I
            res, qb, ub, rb = rng.choice([("I", "R", "\\Omega", ["k"]), ("R", "I", "A", ["m", "\\mu "]), ("I", "R", "\\Omega", ["k", ""])])
            va = rng.choice([5, 10, 12, 15, 20, 24])
            numer, name = num(va), f"\\frac{{V}}{{{qb}}}"
        elif kind == "conductance":                  # P / V
            res, qb, ub, rb = "I", "V", "V", ["m", "k"]
            va = rng.choice([2, 5, 10, 50])
            numer, name = num(va), "\\frac{P}{V}"
        elif kind == "freq":                         # f = 1 / T
            res, qb, ub, rb = "f", "T", "s", ["m", "\\mu "]
            va, numer, name = 1, "1", "\\frac{1}{T}"
        else:                                        # lambda = c / f
            res, qb, ub, rb = "\\lambda", "f", "Hz", ["M", "k", "G"]
            va, numer, name = 3e8, "3 \\cdot 10^{8}", "\\frac{c}{f}"
        pb = rng.choice(rb)
        vb = rng.choice([1, 2, 2.5, 4, 5, 10, 20, 50, 100])
        si = va / (vb * pre_of[pb])
        res_units = {"I": ("A", ["", "m", "\\mu "]), "R": ("\\Omega", ["", "k", "M"]), "f": ("Hz", ["", "k", "M"]),
                     "\\lambda": ("m", [""])}[res]
        pre, x = next(((p, si / pre_of[p]) for p in res_units[1] if 0.1 <= abs(si / pre_of[p]) < 1000), ("", si))
        sep = rng.choice([" / ", "/", " / "])
        mid = f"{numer}{sep}{num(vb)} {pb}{ub}".replace("\\mu  ", "\\mu ")
        u = f" {pre}{res_units[0]}".replace("\\mu  ", "\\mu ")
        right, wrong = [f"{res} = {name} = {mid} = {num(x)}{u}"], [f"{res} = {name} = {mid} = {num(x * 2)}{u}"]
        vr = judged(right)
        if any(v in ("wrong", "wrong step") for v, _ in vr) or not vr:
            bad += 1
            print("WRONG?", right, vr)
        if not any(v in ("wrong", "wrong step") for v, _ in judged(wrong)):
            missed += 1
            print("MISSED", wrong, judged(wrong))
        if pre:                                      # the result bare, in the units kept: never a sure wrong
            bare = [f"{res} = {name} = {mid} = {num(x)}"]
            vb_ = judged(bare)
            if any(v in ("wrong", "wrong step") and sure for v, sure in vb_):
                bad += 1
                print("SURE WRONG (bare)?", bare, vb_)
    return n, bad, n, missed


def systems_random(n=40, seed=7):
    """two mesh equations as students write them - numbers in kΩ (so the currents in mA), in ohms, or with their units
    ('12 V = 2 k\\Omega \\cdot I_1 + ...') - and the currents answered in mA, in amperes or bare: none correct judged wrong
    for sure, every wrong one (x1.3) caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def judged(case):
        st = []
        for k, t in enumerate(case):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    def g(x):
        return format(x, ".4g") if abs(x) < 1000 else format(x, ".10g")     # (an ohm's value whole: '10000', not '1e+04')
    bad = missed = 0
    for _ in range(n):
        e1, e2 = rng.choice([5, 9, 10, 12, 15, 24]), rng.choice([0, 3, 5, 6])
        r1, r2, r3 = (rng.choice([1, 2, 3, 4.7, 5.6, 10]) for _ in range(3))
        det = (r1 + r2) * (r2 + r3) - r2 * r2                 # (r1+r2) I1 - r2 I2 = e1; -r2 I1 + (r2+r3) I2 = -e2 (kΩ, mA)
        i1, i2 = (e1 * (r2 + r3) - e2 * r2) / det, ((r1 + r2) * -e2 + r2 * e1) / det
        style = rng.choice(["kohm", "si", "units"])
        if style == "kohm":
            eqs = [f"{e1} = {g(r1)} I_1 + {g(r2)}(I_1 - I_2)", f"{-e2} = {g(r2)}(I_2 - I_1) + {g(r3)} I_2"]
        elif style == "si":
            eqs = [f"{e1} = {g(r1 * 1000)} I_1 + {g(r2 * 1000)}(I_1 - I_2)", f"{-e2} = {g(r2 * 1000)}(I_2 - I_1) + {g(r3 * 1000)} I_2"]
        else:
            eqs = [f"{e1} V = {g(r1)} k\\Omega \\cdot I_1 + {g(r2)} k\\Omega (I_1 - I_2)",
                   f"{-e2} V = {g(r2)} k\\Omega (I_2 - I_1) + {g(r3)} k\\Omega \\cdot I_2"]
        ans = rng.choice(["mA", "A", "bare"])
        for f, right in ((1.0, True), (1.3, False)):
            a1, a2 = i1 * f, i2 * f
            if ans == "mA":
                vals = [f"I_1 = {g(a1)} mA", f"I_2 = {g(a2)} mA"]
            elif ans == "A":
                vals = [f"I_1 = {g(a1 / 1000)} A", f"I_2 = {g(a2 / 1000)} A"]
            else:
                vals = [f"I_1 = {g(a1 if style == 'kohm' else a1 / 1000)}", f"I_2 = {g(a2 if style == 'kohm' else a2 / 1000)}"]
            if abs(a2) < 1e-6:
                vals = vals[:1]
            got = judged(eqs + vals)
            if right and any(v in ("wrong", "wrong step") and sure for v, sure in got):
                bad += 1
                print("WRONG?", eqs + vals, got)
            if not right and not any(v in ("wrong", "wrong step") for v, _ in got):
                missed += 1
                print("MISSED", eqs + vals, got)
    return n, bad, n, missed


def ee_random(n=400, seed=13):
    """EE computations across templates - Ohm's law, powers, time constants, cut-off and resonance frequencies, series and
    parallel resistances, reactances, a divider, energies, a charge, a capacitor's voltage at a time, a gain in dB - each
    written many ways (units with and without a space, in brackets, in \\mathrm, as students spell them, bare prefixes;
    fractions or slashes; products as \\cdot, \\times, x or brackets) with its result in a prefixed unit: none judged
    wrong for sure, every result x1.3 caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    def gen():
        PRE = {"": 1.0, "k": 1e3, "M": 1e6, "m": 1e-3, "\\mu ": 1e-6, "n": 1e-9, "p": 1e-12}
        UNIT = {"V": "V", "A": "A", "R": "\\Omega", "C": "F", "L": "H", "f": "Hz", "t": "s", "P": "W", "E": "J", "Q": "C", "dB": "dB"}
        PREF = {"V": ["", "m", "k"], "A": ["", "m", "\\mu "], "R": ["", "k", "M"], "C": ["\\mu ", "n", "p"], "L": ["m", "\\mu ", ""],
                "f": ["", "k", "M"], "t": ["m", "\\mu ", "n", ""], "P": ["", "m", "k"], "E": ["", "m", "\\mu "], "Q": ["\\mu ", "n", "m"], "dB": [""]}


        def num(x):
            for d in (0, 1, 2, 3):
                if abs(round(x, d) - x) < 1e-9 * max(1, abs(x)):
                    return f"{round(x, d):.{d}f}"
            return f"{x:.4g}"


        def leaf(q):
            pre = rng.choice(PREF[q])
            v = rng.choice([1, 1.5, 2, 2.2, 3.3, 4.7, 5, 10, 12, 22, 47, 100])
            style = rng.choice(["plain", "plain", "nospace", "bracket", "mathrm", "bare", "spelled"])
            u = UNIT[q]
            if style == "spelled":                           # 'K\\Omega' (a capital K), 'kohm', 'sec', 'hz'
                u2 = {"\\Omega": rng.choice(["\\Omega", "ohm"]), "s": rng.choice(["s", "sec"]), "Hz": rng.choice(["Hz", "hz"])}.get(u, u)
                p2 = "K" if pre == "k" and u == "\\Omega" and rng.random() < 0.5 else pre
                return v * PRE[pre], f"{num(v)} {p2}{u2}".replace("\\mu  ", "\\mu ")
            if style == "bare" and pre in ("k", "M", "\\mu ", "n", "p") and q in ("R", "C", "L"):     # (a bare m: metres or milli)
                txt = f"{num(v)}{pre.strip()}"
            elif style == "nospace":
                txt = f"{num(v)}{pre}{u}".replace("\\mu  ", "\\mu ")
            elif style == "bracket":
                txt = f"{num(v)} [{pre}{u}]".replace("\\mu  ", "\\mu ")
            elif style == "mathrm":
                txt = f"{num(v)} \\mathrm{{{pre.strip()}{u}}}"
            else:
                txt = f"{num(v)} {pre}{u}".replace("\\mu  ", "\\mu ")
            return v * PRE[pre], txt


        def div(a, b):
            return rng.choice([f"\\frac{{{a}}}{{{b}}}", f"{a} / {b}", f"{a}/{b}"])


        def mul(a, b):
            return rng.choice([f"{a} \\cdot {b}", f"{a} \\times {b}", f"({a})({b})", f"{a} x {b}"])


        TEMPLATES = []


        def t_ohm_i():
            (v, tv), (r, tr) = leaf("V"), leaf("R")
            return "I", v / r, "A", [div(tv, tr)]


        def t_ohm_v():
            (i, ti), (r, tr) = leaf("A"), leaf("R")
            return "V", i * r, "V", [mul(ti, tr)]


        def t_power():
            k = rng.choice(["vi", "i2r", "v2r"])
            if k == "vi":
                (v, tv), (i, ti) = leaf("V"), leaf("A")
                return "P", v * i, "P", [mul(tv, ti)]
            if k == "i2r":
                (i, ti), (r, tr) = leaf("A"), leaf("R")
                return "P", i * i * r, "P", [mul(f"({ti})^2", tr)]
            (v, tv), (r, tr) = leaf("V"), leaf("R")
            return "P", v * v / r, "P", [div(f"({tv})^2", tr)]


        def t_tau():
            if rng.random() < 0.5:
                (r, tr), (c, tc) = leaf("R"), leaf("C")
                return "\\tau", r * c, "t", [mul(tr, tc)]
            (l, tl), (r, tr) = leaf("L"), leaf("R")
            return "\\tau", l / r, "t", [div(tl, tr)]


        def t_freq():
            k = rng.choice(["rc", "T", "lc"])
            if k == "rc":
                (r, tr), (c, tc) = leaf("R"), leaf("C")
                return "f_c", 1 / (2 * math.pi * r * c), "f", [rng.choice([f"\\frac{{1}}{{2\\pi \\cdot {tr} \\cdot {tc}}}", f"\\frac{{1}}{{2\\pi {tr} {tc}}}"])]
            if k == "T":
                (t, tt) = leaf("t")
                return "f", 1 / t, "f", [div("1", tt)]
            (l, tl), (c, tc) = leaf("L"), leaf("C")
            return "\\omega_0", 1 / math.sqrt(l * c), "f", [f"\\frac{{1}}{{\\sqrt{{{tl} \\cdot {tc}}}}}"]


        def t_req():
            k = rng.choice(["sum", "par", "sumpar"])
            (a, ta), (b, tb) = leaf("R"), leaf("R")
            if k == "sum":
                return "R_{eq}", a + b, "R", [f"{ta} + {tb}"]
            if k == "par":
                return "R_{eq}", a * b / (a + b), "R", [rng.choice([f"{ta} \\| {tb}", f"\\frac{{{ta} \\cdot {tb}}}{{{ta} + {tb}}}"])]
            (c, tc) = leaf("R")
            return "R_{eq}", (a + b) * c / (a + b + c), "R", [f"({ta} + {tb}) \\| {tc}"]


        def t_react():
            if rng.random() < 0.5:
                (f, tf), (c, tc) = leaf("f"), leaf("C")
                return "X_C", 1 / (2 * math.pi * f * c), "R", [f"\\frac{{1}}{{2\\pi \\cdot {tf} \\cdot {tc}}}"]
            (f, tf), (l, tl) = leaf("f"), leaf("L")
            return "X_L", 2 * math.pi * f * l, "R", [f"2\\pi \\cdot {tf} \\cdot {tl}"]


        def t_divider():
            (v, tv), (a, ta), (b, tb) = leaf("V"), leaf("R"), leaf("R")
            return "V_{out}", v * b / (a + b), "V", [rng.choice([f"{tv} \\cdot \\frac{{{tb}}}{{{ta} + {tb}}}", f"\\frac{{{tv} \\cdot {tb}}}{{{ta} + {tb}}}"])]


        def t_energy():
            if rng.random() < 0.5:
                (c, tc), (v, tv) = leaf("C"), leaf("V")
                return "E", 0.5 * c * v * v, "E", [rng.choice([f"\\frac{{1}}{{2}} \\cdot {tc} \\cdot ({tv})^2", f"0.5 \\cdot {tc} \\cdot ({tv})^2"])]
            (l, tl), (i, ti) = leaf("L"), leaf("A")
            return "E", 0.5 * l * i * i, "E", [f"\\frac{{1}}{{2}} \\cdot {tl} \\cdot ({ti})^2"]


        def t_charge():
            (c, tc), (v, tv) = leaf("C"), leaf("V")
            return "Q", c * v, "Q", [mul(tc, tv)]


        def t_transient():
            (v, tv), (t, tt), (tau, ttau) = leaf("V"), leaf("t"), leaf("t")
            x = v * (1 - math.exp(-t / tau))
            if not 0.01 < t / tau < 5:
                return t_charge()
            return "v_C", x, "V", [rng.choice([f"{tv} (1 - e^{{-{tt} / {ttau}}})", f"{tv} \\cdot \\left(1 - e^{{-\\frac{{{tt}}}{{{ttau}}}}}\\right)"])]


        def t_db():
            (a, ta), (b, tb) = leaf("V"), leaf("V")
            if a <= 0 or b <= 0:
                return t_charge()
            return "A_{dB}", 20 * math.log10(a / b), "dB", [f"20 \\log\\left(\\frac{{{ta}}}{{{tb}}}\\right)"]


        TEMPLATES = [t_ohm_i, t_ohm_v, t_power, t_tau, t_freq, t_req, t_react, t_divider, t_energy, t_charge, t_transient, t_db]


        def result(x, q):
            u = UNIT[q]
            if q == "dB":
                return f"{num(round(x, 2))} dB", ""
            for p in ["", "k", "M", "G", "m", "\\mu ", "n", "p"]:
                f = {"G": 1e9}.get(p, PRE.get(p))
                if 1 <= abs(x / f) < 1000:
                    return f"{num(x / f)} {p}{u}".replace("\\mu  ", "\\mu "), p
            return f"{x:.4g} {u}", ""





        return TEMPLATES, result
    templates, result = gen()
    bad = missed = 0
    for _ in range(n):
        name, x, q, mids = rng.choice(templates)()
        if abs(x) < 1e-30:
            continue
        res, _ = result(x, q)
        rx, _ = result(x * 1.3, q)
        if name == "\\omega_0":
            res, rx = res.replace("Hz", "rad/s"), rx.replace("Hz", "rad/s")
        right, wrong = f"{name} = {mids[0]} = {res}", f"{name} = {mids[0]} = {rx}"
        vr = verdicts([right])
        if any(v in ("wrong", "wrong step") and s for v, s in vr):
            bad += 1
            print("WRONG?", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts([wrong])):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def formulas_random(n=200, seed=17):
    """a formula, its data and the answer as students write them - the data with units (prefixed, in brackets, bare
    prefixes, powers of ten) or bare in SI, before or after the formula; the answer in a prefixed unit, in SI, or bare in
    SI: none judged wrong for sure, every answer x1.3 caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    def gen():
        PRE = {"": 1.0, "k": 1e3, "M": 1e6, "m": 1e-3, "\\mu ": 1e-6, "n": 1e-9, "p": 1e-12}
        UNIT = {"V": "V", "A": "A", "R": "\\Omega", "C": "F", "f": "Hz", "t": "s", "P": "W", "E": "J"}
        PREF = {"V": ["", "m"], "A": ["m", "\\mu ", ""], "R": ["", "k", "M"], "C": ["\\mu ", "n", "p"], "f": ["", "k"], "t": ["m", "\\mu "],
                "P": ["", "m"], "E": ["m", "\\mu "]}
        # name, formula text, variables (name -> quantity), result quantity, function
        FORMULAS = [
            ("I", "I = \\frac{V}{R}", {"V": "V", "R": "R"}, "A", lambda d: d["V"] / d["R"]),
            ("P", "P = I^2 R", {"I": "A", "R": "R"}, "P", lambda d: d["I"] ** 2 * d["R"]),
            ("\\tau", "\\tau = R C", {"R": "R", "C": "C"}, "t", lambda d: d["R"] * d["C"]),
            ("f_c", "f_c = \\frac{1}{2\\pi R C}", {"R": "R", "C": "C"}, "f", lambda d: 1 / (2 * math.pi * d["R"] * d["C"])),
            ("X_C", "X_C = \\frac{1}{2\\pi f C}", {"f": "f", "C": "C"}, "R", lambda d: 1 / (2 * math.pi * d["f"] * d["C"])),
            ("V_{out}", "V_{out} = V_{in} \\frac{R_2}{R_1 + R_2}", {"V_{in}": "V", "R_1": "R", "R_2": "R"}, "V",
             lambda d: d["V_{in}"] * d["R_2"] / (d["R_1"] + d["R_2"])),
            ("E", "E = \\frac{1}{2} C V^2", {"C": "C", "V": "V"}, "E", lambda d: 0.5 * d["C"] * d["V"] ** 2),
        ]


        def num(x):
            for d in (0, 1, 2, 3):
                if abs(round(x, d) - x) < 1e-9 * max(1, abs(x)):
                    return f"{round(x, d):.{d}f}"
            return f"{x:.4g}"


        def datum(name, q):
            pre = rng.choice(PREF[q])
            v = rng.choice([1, 1.5, 2, 2.2, 3.3, 4.7, 5, 10, 12, 22, 47, 100])
            u = UNIT[q]
            style = rng.choice(["plain", "plain", "bracket", "bare", "sci", "si"])
            if style == "si":
                return v * PRE[pre], f"{name} = {v * PRE[pre]:.6g}"
            if style == "bare" and pre in ("k", "M", "\\mu ", "n", "p"):
                txt = f"{num(v)}{pre.strip()}"
            elif style == "bracket":
                txt = f"{num(v)} [{pre}{u}]".replace("\\mu  ", "\\mu ")
            elif style == "sci" and pre:
                e = round(math.log10(PRE[pre]))
                txt = f"{num(v)} \\cdot 10^{{{e}}} {u}"
            else:
                txt = f"{num(v)} {pre}{u}".replace("\\mu  ", "\\mu ")
            return v * PRE[pre], f"{name} = {txt}"


        def answer(name, x, q):
            u = UNIT[q]
            style = rng.choice(["prefixed", "prefixed", "si", "si-bare"])
            if style == "si":
                return f"{name} = {x:.4g} {u}"
            if style == "si-bare":
                return f"{name} = {x:.4g}"
            for p in ["", "k", "M", "m", "\\mu ", "n", "p"]:
                if 1 <= abs(x / PRE[p]) < 1000:
                    return f"{name} = {num(x / PRE[p])} {p}{u}".replace("\\mu  ", "\\mu ")
            return f"{name} = {x:.4g} {u}"


        return FORMULAS, datum, answer
    formulas, datum, answer = gen()
    bad = missed = 0
    for _ in range(n):
        name, ftxt, vars_, q, fn = rng.choice(formulas)
        data, lines = {}, []
        for vn, vq in vars_.items():
            v, line = datum(vn, vq)
            data[vn] = v
            lines.append(line)
        x = fn(data)
        case = lines + [ftxt] if rng.random() < 0.5 else [ftxt] + lines
        right, wrong = case + [answer(name, x, q)], case + [answer(name, x * 1.3, q)]
        vr = verdicts(right)
        if any(v in ("wrong", "wrong step") and s for v, s in vr):
            bad += 1
            print("WRONG?", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts(wrong)):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def phasor_units_random(n=300, seed=19):
    """phasor computations with their units written every way - a complex value's unit unbracketed ('Z = 3 + j4 \\Omega',
    '\\frac{12 + j5 V}{4 + j4 k\\Omega}', '6 + j 8 k\\Omega'), on each part ('3 \\Omega - j4 \\Omega'), a reactance alone
    ('\\frac{10 V}{-j4 \\Omega}'), in kΩ and mA, apparent power in VA and kVA, products of phasors with units: none judged
    wrong for sure, none without a verdict, every result x1.3 caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    OHM = "\\Omega"

    def num(x, d=2):
        t = f"{x:.{d}f}"
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        return t if t not in ("-0", "") else "0"

    def polar(z, d=2, unit=""):
        style = rng.choice(["{m} \\angle {a}^\\circ", "{m}\\angle{a}^{{\\circ}}", "{m} \\angle {a}°"])
        return style.format(m=num(abs(z), d), a=num(math.degrees(cmath.phase(z)), 2)) + (f" {unit}" if unit else "")

    def rect(z, d=2, unit=""):
        a, b = round(z.real, d), round(z.imag, d)
        sg = "+" if b >= 0 else "-"
        t = rng.choice([f"{num(a, d)} {sg} j{num(abs(b), d)}", f"({num(a, d)} {sg} j{num(abs(b), d)})"])
        return t + (f" {unit}" if unit else "")

    def phasor(z, d=2, unit=""):
        return polar(z, d, unit) if rng.random() < 0.5 else rect(z, d, unit)

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = silent = 0
    for _ in range(n):
        kind = rng.choice(["ohm", "zsum", "reac", "spow", "vi", "series", "zc", "divider"])
        kilo = rng.random() < 0.5
        if kind == "ohm":
            V = complex(rng.choice([10, 12, 20]), rng.choice([0, 0, 5, -8]))
            Z = complex(rng.choice([3, 4, 6, 8]), rng.choice([-4, 3, 4, 6]))
            I = V / Z
            if kilo:
                head = f"I = \\frac{{{rect(V, 0, 'V')}}}{{{rect(Z, 0, 'k' + OHM)}}}"
                right, wrong = f"{head} = {phasor(I, 3, 'mA')}", f"{head} = {phasor(I * 1.3, 3, 'mA')}"
            else:
                head = f"I = \\frac{{V}}{{Z}} = \\frac{{{rect(V, 0, 'V')}}}{{{rect(Z, 0, OHM)}}}"
                right, wrong = f"{head} = {phasor(I, 3, 'A')}", f"{head} = {phasor(I * 1.3, 3, 'A')}"
        elif kind == "zsum":
            R, X = rng.choice([3, 6, 8, 12, 2.2]), rng.choice([4, -4, 8, -5, 1.5])
            u = rng.choice([OHM, "k" + OHM, "\\mathrm{k" + OHM + "}", "Ω"])
            sg = "+" if X > 0 else "-"
            z = rng.choice(["{R} {sg} j{X} {u}", "{R} {u} {sg} j{X} {u}", "({R} {sg} j{X}) {u}", "{R} {sg} j {X} {u}",
                            "{R} {sg} j \\cdot {X} {u}"]).format(R=num(R), sg=sg, X=num(abs(X)), u=u)
            right, wrong = f"Z = {z} = {polar(complex(R, X), 2, u)}", f"Z = {z} = {polar(complex(R, X) * 1.3, 2, u)}"
        elif kind == "reac":
            V, X = rng.choice([5, 10, 12]), rng.choice([2, 4, 5, 8]) * rng.choice([1, -1])
            zt = f"{'-' if X < 0 else ''}j{abs(X)} {'k' if kilo else ''}{OHM}"
            I = V / complex(0, X)
            iu = "mA" if kilo else "A"
            right, wrong = f"I = \\frac{{{V} V}}{{{zt}}} = {phasor(I, 3, iu)}", f"I = \\frac{{{V} V}}{{{zt}}} = {phasor(I * 1.3, 3, iu)}"
        elif kind == "spow":
            P, Q = rng.choice([120, 300, 1.2, 40]), rng.choice([90, -160, 0.9, 30])
            u = rng.choice(["VA", "kVA"]) if P < 10 else "VA"
            sg = "+" if Q > 0 else "-"
            st = rng.choice([f"{num(P)} {sg} j{num(abs(Q))} {u}", f"({num(P)} {sg} j{num(abs(Q))}) {u}"])
            right, wrong = f"S = P + jQ = {st} = {polar(complex(P, Q), 2, u)}", f"S = P + jQ = {st} = {polar(complex(P, Q) * 1.3, 2, u)}"
        elif kind == "vi":
            Im, Ia, Zm, Za = rng.choice([2, 1.5, 3]), rng.choice([0, 30, -45]), rng.choice([5, 10, 4.7]), rng.choice([53.13, -30, 90])
            Vv = cmath.rect(Im, math.radians(Ia)) * cmath.rect(Zm, math.radians(Za))
            it, zt = f"{num(Im)} \\angle {num(Ia)}° {'mA' if kilo else 'A'}", f"{num(Zm)} \\angle {num(Za)}° {'k' if kilo else ''}{OHM}"
            right, wrong = f"V = I Z = ({it})({zt}) = {polar(Vv, 2, 'V')}", f"V = I Z = ({it})({zt}) = {polar(Vv * 1.3, 2, 'V')}"
        elif kind == "series":
            Z1, Z2 = complex(rng.choice([3, 4, 10]), rng.choice([4, -2, 6])), complex(rng.choice([2, 6, 1]), rng.choice([-3, 1, -8]))
            u = "k" + OHM if kilo else OHM
            a, b = rect(Z1, 0), rect(Z2, 0)
            a, b = (a if a.startswith("(") else f"({a})"), (b if b.startswith("(") else f"({b})")
            mid = rng.choice([f"{a} {u} + {b} {u}", f"{a} + {b}"])
            right = f"Z_{{eq}} = Z_1 + Z_2 = {mid} = {phasor(Z1 + Z2, 2, u)}"
            wrong = f"Z_{{eq}} = Z_1 + Z_2 = {mid} = {phasor((Z1 + Z2) * 1.3, 2, u)}"
        elif kind == "zc":
            f, C = rng.choice([50, 1000, 5000]), rng.choice([1e-6, 10e-6, 100e-9])
            Xc = 1 / (2 * math.pi * f * C)
            ct = {1e-6: "1 \\mu F", 10e-6: "10 \\mu F", 100e-9: "100 nF"}[C]
            zt = (lambda x: f"-j{num(x, 1)} {OHM}" if Xc < 1000 else f"-j{num(x / 1000, 3)} k{OHM}")
            head = f"Z_C = \\frac{{1}}{{j\\omega C}} = \\frac{{1}}{{j \\cdot 2\\pi \\cdot {f} \\cdot {ct}}}"
            right, wrong = f"{head} = {zt(Xc)}", f"{head} = {zt(Xc * 1.3)}"
        else:
            V = complex(rng.choice([10, 20]), 0)
            Z1, Z2 = complex(rng.choice([3, 4]), rng.choice([4, -2])), complex(rng.choice([2, 6]), rng.choice([-3, 1]))
            Vo = V * Z2 / (Z1 + Z2)
            head = f"V_o = V \\frac{{Z_2}}{{Z_1 + Z_2}} = {num(V.real, 0)} \\cdot \\frac{{{rect(Z2, 0)}}}{{{rect(Z1, 0)} + {rect(Z2, 0)}}}"
            right, wrong = f"{head} = {phasor(Vo, 2, 'V')}", f"{head} = {phasor(Vo * 1.3, 2, 'V')}"
        vr = verdicts([right])
        if not vr:
            silent += 1
            print("NO VERDICT", right)
        if any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG?", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts([wrong])):
            missed += 1
            print("MISSED", wrong)
    return n, bad + silent, n, missed


def semiconductor_random(n=200, seed=23):
    """semiconductor physics as it is computed - in centimetres (densities in cm^{-3}, D in cm^2/s, lengths in cm or
    micrometres, resistivities in \\Omega \\cdot cm), the units spelt every way ('cm^{-3}', '\\mathrm{cm^{-3}}',
    '\\frac{1}{cm^3}', '\\frac{cm^2}{s}', 'S/cm'): none judged wrong for sure, none without a verdict, every result x1.3
    caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def sci(x, d=3):
        if 0.01 <= abs(x) < 10000:
            return f"{x:.{d}g}"
        e = int(math.floor(math.log10(abs(x))))
        return rng.choice(["{m} \\cdot 10^{{{e}}}", "{m} \\times 10^{{{e}}}"]).format(m=f"{x / 10 ** e:.{d}g}", e=e)

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    dens = ["cm^{-3}", "\\mathrm{cm^{-3}}", "\\frac{1}{cm^3}", "\\mathrm{cm}^{-3}"]
    bad = missed = 0
    for _ in range(n):
        kind = rng.choice(["n", "rho", "sigma", "D", "L", "W", "Vbi", "J", "conv"])
        ni, Nd, mu = rng.choice([1.5e10, 1e10, 9.65e9]), rng.choice([1e15, 1e16, 5e16, 1e17, 2e15]), rng.choice([1350, 480, 1000, 400])
        unit = None
        if kind == "n":
            x, head, unit = ni ** 2 / Nd, f"n = \\frac{{n_i^2}}{{N_A}} = \\frac{{({sci(ni)})^2}}{{{sci(Nd)}}}", rng.choice(dens)
        elif kind == "rho":
            x = 1 / (1.6e-19 * Nd * mu)
            head = f"\\rho = \\frac{{1}}{{q N \\mu_n}} = \\frac{{1}}{{1.6 \\cdot 10^{{-19}} \\cdot {sci(Nd)} \\cdot {mu}}}"
            unit = rng.choice(["\\Omega \\cdot cm", "\\Omega cm", "Ω \\cdot cm"])
        elif kind == "sigma":
            x = 1.6e-19 * Nd * mu
            head, unit = f"\\sigma = q N \\mu_n = 1.6 \\cdot 10^{{-19}} \\cdot {sci(Nd)} \\cdot {mu}", rng.choice(["S/cm", "\\frac{S}{cm}"])
        elif kind == "D":
            vt = rng.choice(["0.0259", "0.026", "25.9 mV", "26 mV"])
            x = mu * {"0.0259": 0.0259, "0.026": 0.026, "25.9 mV": 0.0259, "26 mV": 0.026}[vt]
            head, unit = f"D_n = \\mu_n V_T = {mu} \\cdot {vt}", rng.choice(["cm^2/s", "\\frac{cm^2}{s}", "cm^{2}/s", "\\mathrm{cm^2/s}"])
        elif kind in ("L", "W"):
            if kind == "L":
                D, tau = rng.choice([34.97, 12.4, 25.9, 10.4]), rng.choice([1e-6, 1e-7, 1e-5])
                x, head = math.sqrt(D * tau), f"L_n = \\sqrt{{D_n \\tau_n}} = \\sqrt{{{D} \\cdot {sci(tau)}}}"
            else:
                V = rng.choice([0.7, 0.8, 5, 10])
                x = math.sqrt(2 * 1.04e-12 * V / (1.6e-19 * Nd))
                head = (f"W = \\sqrt{{\\frac{{2 \\epsilon_s V}}{{q N}}}} = \\sqrt{{\\frac{{2 \\cdot 1.04 \\cdot 10^{{-12}} \\cdot {V}}}"
                        f"{{1.6 \\cdot 10^{{-19}} \\cdot {sci(Nd)}}}}}")
            if rng.random() < 0.5:
                unit = "cm"
            else:
                x, unit = x * 1e4, "\\mu m"                  # (worked out in cm, written in micrometres: right, not sure)
        elif kind == "Vbi":
            Na = rng.choice([1e16, 1e17, 1e18])
            x = 0.0259 * math.log(Na * Nd / ni ** 2)
            head = f"V_{{bi}} = V_T \\ln\\frac{{N_A N_D}}{{n_i^2}} = 0.0259 \\ln\\frac{{{sci(Na)} \\cdot {sci(Nd)}}}{{({sci(ni)})^2}}"
            unit = "V"
        elif kind == "J":
            v = rng.choice([1e5, 1.35e5, 5e4])
            x, head = 1.6e-19 * Nd * v, f"J = q n v_d = 1.6 \\cdot 10^{{-19}} \\cdot {sci(Nd)} \\cdot {sci(v)}"
            unit = rng.choice(["A/cm^2", "\\frac{A}{cm^2}", "A/cm^{2}"])
        else:
            x, head, unit = Nd * 1e6, f"N_D = {sci(Nd)} cm^{{-3}}", "m^{-3}"
        right, wrong = f"{head} = {sci(x)} {unit}", f"{head} = {sci(x * 1.3)} {unit}"
        vr = verdicts([right])
        if not vr or any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG? / NO VERDICT", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts([wrong])):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def trig_random(n=200, seed=29):
    """angles in power and phasor work - 'P = S \\cos\\phi', 'Q = S \\sin\\phi', 'P = V I \\cos\\theta', 'X = |Z| \\sin\\phi',
    '\\phi = \\arctan\\frac{X}{R}' - as a formula with its data (prefixed: kVA, mA, kΩ; the angle in degrees marked, bare,
    or in radians) and the answer in a prefixed unit, or as a chain ('2000 \\cos(45) = 1.414 kW'): none judged wrong for
    sure, none without a verdict, every result x1.3 caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    pre = {"": 1.0, "k": 1e3, "m": 1e-3}

    def num(x):
        for d in (0, 1, 2, 3):
            if abs(round(x, d) - x) < 1e-9 * max(1, abs(x)):
                return f"{round(x, d):.{d}f}"
        return f"{x:.4g}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    def angle(deg):
        return rng.choice([f"{num(deg)}°", f"{num(deg)}^\\circ", num(deg), f"{math.radians(deg):.4f} rad"])

    def qty(v, unit, pres):
        p = rng.choice(pres)
        return f"{num(v / pre[p])} {p}{unit}"

    def ans(x, unit, pres):
        p = rng.choice(pres)
        return f"{x / pre[p]:.4g} {p}{unit}"

    bad = missed = 0
    for _ in range(n):
        kind = rng.choice(["P", "Q", "Pvi", "X", "phi"])
        deg = rng.choice([30, 36.87, 45, 53.13, 60])
        chain = rng.random() < 0.35 and kind != "phi"
        if kind in ("P", "Q"):
            S = rng.choice([1000, 2000, 500, 1500])
            f, fn_, u = ("\\cos", math.cos, "W") if kind == "P" else ("\\sin", math.sin, rng.choice(["VAR", "var"]))
            x = S * fn_(math.radians(deg))
            if chain:
                lines = [f"{kind} = S {f}\\phi = {qty(S, 'VA', ['', 'k'])} \\cdot {f}({rng.choice([num(deg), num(deg) + '°'])}) = "]
            else:
                lines = [f"{kind} = S {f}\\phi", f"S = {qty(S, 'VA', ['', 'k'])}", f"\\phi = {angle(deg)}", f"{kind} = "]
            res = (u, ["", "k"])
        elif kind == "Pvi":
            V, I = rng.choice([230, 120, 10]), rng.choice([5, 0.5, 0.002])
            x = V * I * math.cos(math.radians(deg))
            if chain:
                lines = [f"P = V I \\cos\\theta = {num(V)} \\cdot {num(I)} \\cdot \\cos({rng.choice([num(deg), num(deg) + '°'])}) = "]
            else:
                lines = ["P = V I \\cos\\theta", f"V = {qty(V, 'V', [''])}", f"I = {qty(I, 'A', ['', 'm'])}", f"\\theta = {angle(deg)}", "P = "]
            res = ("W", ["", "m", "k"])
        elif kind == "X":
            Z = rng.choice([10, 50, 4700, 2200])
            x = Z * math.sin(math.radians(deg))
            if chain:
                lines = [f"X = |Z| \\sin\\phi = {num(Z)} \\sin({rng.choice([num(deg), num(deg) + '°'])}) = "]
            else:
                lines = ["X = |Z| \\sin\\phi", f"|Z| = {qty(Z, chr(92) + 'Omega', ['', 'k'])}", f"\\phi = {angle(deg)}", "X = "]
            res = ("\\Omega", ["", "k"])
        else:
            R, X = rng.choice([(3, 4), (4, 3), (1, 1), (2.2, 1.5)])
            a = math.degrees(math.atan2(X, R))
            lines = ["\\phi = \\arctan\\frac{X}{R}", f"X = {num(X)} \\Omega", f"R = {num(R)} \\Omega", "\\phi = "]
            style = rng.choice(["{:.2f}°", "{:.2f}^\\circ", "{:.2f}"])
            right, wrong = lines[:-1] + [lines[-1] + style.format(a)], lines[:-1] + [lines[-1] + style.format(a * 1.3)]
            kind = None
        if kind:
            p = rng.choice(res[1])
            right = lines[:-1] + [lines[-1] + f"{x / pre[p]:.4g} {p}{res[0]}"]
            wrong = lines[:-1] + [lines[-1] + f"{x * 1.3 / pre[p]:.4g} {p}{res[0]}"]
        vr = verdicts(right)
        if not vr or any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG? / NO VERDICT", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts(wrong)):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def flow_random(n=150, seed=31):
    """work over several lines whose values flow on by name ('R_{eq} = ... = 1.25 k\\Omega', 'I = \\frac{V}{R_{eq}} = 15.7 mA',
    'P = I^2 R_{eq} = 305 mW'), prefixed, with bare prefixes, rounded on the way (the next line worked out with the value
    unrounded): none judged wrong for sure; with one slip (x1.3) in a line, that line caught and no other judged wrong"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    pre = {"": 1.0, "k": 1e3, "m": 1e-3, "\\mu ": 1e-6}

    def g(x):
        return f"{x:.3g}"

    def val(x, unit, pres, bare_ok=True):
        for p in pres:
            if 1 <= abs(x / pre[p]) < 1000:
                if bare_ok and p == "k" and rng.random() < 0.33:
                    return f"{g(x / pre[p])}k"
                return f"{g(x / pre[p])} {p}{unit}".replace("\\mu  ", "\\mu ")
        return f"{g(x)} {unit}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    om = "\\Omega"
    bad = missed = slips = 0
    for _ in range(n):
        R1, R2 = rng.choice([1e3, 2e3, 2.2e3, 4.7e3, 10e3]), rng.choice([1e3, 3e3, 3.3e3, 6.8e3, 22e3])
        V, par = rng.choice([5, 9, 10, 12, 15]), rng.random() < 0.5
        Req = R1 * R2 / (R1 + R2) if par else R1 + R2
        slip = rng.choice([None, 1, 2, 3])
        slips += slip is not None
        Req_w = Req * (1.3 if slip == 1 else 1)
        I_w = V / Req_w * (1.3 if slip == 2 else 1)
        P_w = I_w ** 2 * Req_w * (1.3 if slip == 3 else 1)
        op = "\\|" if par else "+"
        l1 = f"R_{{eq}} = R_1 {op} R_2 = {val(R1, om, ['k'])} {op} {val(R2, om, ['k'])} = {val(Req_w, om, ['k', ''])}"
        l2 = (f"I = \\frac{{V}}{{R_{{eq}}}} = \\frac{{{V}}}{{{val(Req_w, om, ['k', ''], False)}}} = {val(I_w, 'A', ['m', '', '\\mu '])}"
              if rng.random() < 0.5 else f"I = \\frac{{V}}{{R_{{eq}}}} = {val(I_w, 'A', ['m', '', '\\mu '])}")
        l3 = (f"P = I^2 R_{{eq}} = ({val(I_w, 'A', ['m', '\\mu '], False)})^2 \\cdot {val(Req_w, om, ['k', ''], False)} = "
              f"{val(P_w, 'W', ['m', '', '\\mu '])}" if rng.random() < 0.5 else f"P = I^2 R_{{eq}} = {val(P_w, 'W', ['m', '', '\\mu '])}")
        lines = [f"V = {V} V", l1, l2, l3]
        vr = verdicts(lines)
        wrongs = [v for v, s_ in vr if v in ("wrong", "wrong step") and s_]
        if (slip is None and wrongs) or (slip is not None and len(wrongs) > 1):
            bad += 1
            print("WRONG?", slip, lines, vr)
        if slip is not None and not any(v in ("wrong", "wrong step") for v, _ in vr):
            missed += 1
            print("MISSED", slip, lines)
    return n, bad, slips, missed


def formula_flow_random(n=150, seed=37):
    """a formula whose data include a value worked out on its own line - a series / parallel combination ('R = R_1 \\| R_2
    = 2.2k \\| 6.8k = 1.66 k\\Omega'), a current by Ohm's law - prefixed, bare prefixes, rounded; the answer in a prefixed
    unit: none judged wrong for sure, every answer x1.3 caught (the worked-out line was skipped before: no verdict)"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    pre = {"": 1.0, "k": 1e3, "m": 1e-3, "\\mu ": 1e-6, "n": 1e-9}
    om = "\\Omega"

    def g(x):
        return f"{x:.3g}"

    def val(x, unit, pres, bare=False):
        for p in pres:
            if 1 <= abs(x / pre[p]) < 1000:
                if bare and p == "k" and rng.random() < 0.4:
                    return f"{g(x / pre[p])}k"
                return f"{g(x / pre[p])} {p}{unit}".replace("\\mu  ", "\\mu ")
        return f"{g(x)} {unit}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = 0
    for _ in range(n):
        R1, R2 = rng.choice([1e3, 2.2e3, 4.7e3, 10e3]), rng.choice([1e3, 3.3e3, 6.8e3, 22e3])
        par = rng.random() < 0.5
        op = "\\|" if par else "+"
        Req = R1 * R2 / (R1 + R2) if par else R1 + R2
        req_line = f"R = R_1 {op} R_2 = {val(R1, om, ['k'], True)} {op} {val(R2, om, ['k'], True)} = {val(Req, om, ['k', ''])}"
        kind = rng.choice(["fc", "tau", "P", "Pv"])
        if kind == "fc":
            C = rng.choice([1e-9, 10e-9, 100e-9, 1e-6])
            x, lines, ans = 1 / (2 * math.pi * Req * C), ["f_c = \\frac{1}{2\\pi R C}", req_line, f"C = {val(C, 'F', ['n', '\\mu '])}"], \
                (lambda v: f"f_c = {val(v, 'Hz', ['', 'k', 'M'])}")
        elif kind == "tau":
            C = rng.choice([1e-6, 10e-6, 100e-9])
            x, lines, ans = Req * C, ["\\tau = R C", req_line, f"C = {val(C, 'F', ['n', '\\mu '])}"], \
                (lambda v: f"\\tau = {val(v, 's', ['m', '\\mu ', ''])}")
        elif kind == "P":
            V = rng.choice([5, 10, 12])
            I = V / Req
            x = I ** 2 * Req
            lines = ["P = I^2 R", req_line, f"I = \\frac{{V}}{{R}} = \\frac{{{V}}}{{{val(Req, om, ['k', ''])}}} = {val(I, 'A', ['m', '\\mu '])}"]
            ans = lambda v: f"P = {val(v, 'W', ['m', '\\mu ', ''])}"
        else:
            V = rng.choice([5, 10, 12])
            x, lines, ans = V ** 2 / Req, ["P = \\frac{V^2}{R}", req_line, f"V = {V} V"], (lambda v: f"P = {val(v, 'W', ['m', '\\mu ', ''])}")
        right, wrong = lines + [ans(x)], lines + [ans(x * 1.3)]
        vr = verdicts(right)
        if not vr or any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG? / NO VERDICT", right, vr)
        if not any(v in ("wrong", "wrong step") for v, _ in verdicts(wrong)):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def two_formulas_random(n=150, seed=41):
    """two formulas, the second defining a letter of the first ('P = I^2 R', 'I = \\frac{V}{R}'; 'f_c = \\frac{1}{2\\pi R C}',
    'R = \\frac{R_1 R_2}{R_1 + R_2}'), the data below both (prefixed, bare prefixes, in any order), the first's answer and
    sometimes the second's: none judged wrong for sure, none without a verdict, every answer x1.3 caught (the second
    formula took the data and answers: 'cannot check' before)"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    pre = {"": 1.0, "k": 1e3, "m": 1e-3, "\\mu ": 1e-6, "n": 1e-9}
    om = "\\Omega"

    def g(x):
        return f"{x:.3g}"

    def val(x, unit, pres, bare=False):
        p = rng.choice([q for q in pres if 1 <= abs(x / pre[q]) < 10000] or [pres[0]])
        if bare and p == "k" and rng.random() < 0.4:
            return f"{g(x / pre[p])}k"
        return f"{g(x / pre[p])} {p}{unit}".replace("\\mu  ", "\\mu ")

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = 0
    for _ in range(n):
        kind = rng.choice(["P", "fc", "tau", "Vo"])
        R1, R2 = rng.choice([1e3, 2.2e3, 4.7e3, 10e3]), rng.choice([1e3, 3.3e3, 6.8e3, 22e3])
        V = rng.choice([5, 10, 12])
        if kind == "P":
            I = V / R1
            x, f1, f2 = I ** 2 * R1, "P = I^2 R", "I = \\frac{V}{R}"
            data, mid, ans = [f"V = {V} V", f"R = {val(R1, om, ['k', ''], True)}"], ("I", I, "A", ["m", "\\mu "]), ("P", "W", ["m", ""])
        elif kind == "fc":
            C, R = rng.choice([1e-9, 10e-9, 100e-9]), R1 * R2 / (R1 + R2)
            x, f1, f2 = 1 / (2 * math.pi * R * C), "f_c = \\frac{1}{2\\pi R C}", "R = \\frac{R_1 R_2}{R_1 + R_2}"
            data = [f"R_1 = {val(R1, om, ['k'], True)}", f"R_2 = {val(R2, om, ['k'], True)}", f"C = {val(C, 'F', ['n', '\\mu '])}"]
            mid, ans = ("R", R, om, ["k", ""]), ("f_c", "Hz", ["", "k"])
        elif kind == "tau":
            C, R = rng.choice([1e-6, 10e-6, 100e-9]), R1 + R2
            x, f1, f2 = R * C, "\\tau = R C", "R = R_1 + R_2"
            data = [f"R_1 = {val(R1, om, ['k'], True)}", f"R_2 = {val(R2, om, ['k'], True)}", f"C = {val(C, 'F', ['n', '\\mu '])}"]
            mid, ans = ("R", R, om, ["k", ""]), ("\\tau", "s", ["m", "\\mu "])
        else:
            I = V / (R1 + R2)
            x, f1, f2 = I * R2, "V_o = I R_L", "I = \\frac{V_s}{R_s + R_L}"
            data = [f"V_s = {V} V", f"R_s = {val(R1, om, ['k', ''], True)}", f"R_L = {val(R2, om, ['k', ''], True)}"]
            mid, ans = ("I", I, "A", ["m", "\\mu "]), ("V_o", "V", ["", "m"])
        rng.shuffle(data)
        mid_line = [f"{mid[0]} = {val(mid[1], mid[2], mid[3])}"] if rng.random() < 0.4 else []
        right = [f1, f2] + data + mid_line + [f"{ans[0]} = {val(x, ans[1], ans[2])}"]
        wrong = [f1, f2] + data + mid_line + [f"{ans[0]} = {val(x * 1.3, ans[1], ans[2])}"]
        vr = verdicts(right)
        if not any(v == "right" for v, _ in vr) or any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG? / NO VERDICT", right, vr)
        if not any(v in ("wrong", "wrong step") for v, _ in verdicts(wrong)):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def db_random(n=200, seed=43):
    """gains in dB: given in dB and used as its ratio ('A_v = 40 dB', 'V_o = A_v V_i = 100 \\cdot 0.1 = 10 V', or as a formula's
    data), beside its ratio ('A_v = 40 dB = 100'), added along a cascade, a ratio turned into dB, dBm: none judged wrong for
    sure, every result x1.3 caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def g(x):
        return f"{x:.3g}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = 0
    for _ in range(n):
        kind = rng.choice(["use", "use_f", "cascade", "todb", "fromdb", "dbm"])
        G = rng.choice([20, 40, 6, 34])
        r = 10 ** (G / 20)
        if kind == "use":
            Vi = rng.choice([0.1, 0.05, 0.002])
            vi_t = rng.choice([f"{g(Vi)} V", f"{g(Vi * 1000)} mV"])
            mid = rng.choice([f" = {g(r)} \\cdot {g(Vi)}", ""])
            right, wrong = ([f"A_v = {G} dB", f"V_i = {vi_t}", f"V_o = A_v V_i{mid} = {g(f * r * Vi)} V"] for f in (1, 1.3))
        elif kind == "use_f":
            Vi = rng.choice([0.1, 0.05])
            right, wrong = (["V_o = A_v V_i", f"A_v = {G} dB", f"V_i = {g(Vi)} V", f"V_o = {g(f * r * Vi)} V"] for f in (1, 1.3))
        elif kind == "cascade":
            G1, G2 = rng.choice([10, 20, 15]), rng.choice([6, 20, 30])
            right, wrong = ([f"G_1 = {G1} dB", f"G_2 = {G2} dB", f"G = G_1 + G_2 = {G1} + {G2} = {v} dB"]
                            for v in (G1 + G2, round((G1 + G2) * 1.3)))
        elif kind == "todb":
            Vo, Vi = rng.choice([5, 2, 10]), rng.choice([0.05, 0.02, 0.1])
            x = 20 * math.log10(Vo / Vi)
            right, wrong = ([f"A_v = 20 \\log\\frac{{V_o}}{{V_i}} = 20 \\log\\frac{{{Vo}}}{{{Vi}}} = {g(f * x)} dB"] for f in (1, 1.3))
        elif kind == "fromdb":
            head = rng.choice([f"A_v = 10^{{{G}/20}}", f"A_v = {G} dB", f"A_v = 10^{{\\frac{{{G}}}{{20}}}}"])
            right, wrong = [f"{head} = {g(r)}"], [f"{head} = {g(r * 1.3)}"]
        else:
            P = rng.choice([1e-3, 10e-3, 0.1, 2e-3])
            x = 10 * math.log10(P / 1e-3)
            pt = rng.choice([f"{g(P * 1000)} mW", f"{g(P)} W"])
            head = f"P_{{dBm}} = 10 \\log\\frac{{P}}{{1 mW}} = 10 \\log\\frac{{{pt}}}{{1 mW}}"
            right, wrong = [f"{head} = {g(x)} dBm"], [f"{head} = {g(x * 1.3 + 1)} dBm"]
        vr = verdicts(right)
        if any(v in ("wrong", "wrong step") and s_ for v, s_ in vr):
            bad += 1
            print("WRONG?", right, vr)
        if vr and not any(v in ("wrong", "wrong step") for v, _ in verdicts(wrong)):
            missed += 1
            print("MISSED", wrong)
    return n, bad, n, missed


def page_random(n=120, seed=47):
    """a page of two or three sub-problems one after another, unnumbered or numbered, reusing the letters (R, I, V, P, C,
    f) with their own values or the same ones (a part done again, its result put right): nothing of one part may make
    a verdict on another wrong; one slip in most pages - caught, and the only sure wrong"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "\\Omega"

    def g(x):
        return f"{x:.3g}"

    def kval(x, unit):
        for p_, f_ in (("k", 1e3), ("", 1.0), ("m", 1e-3), ("\\mu ", 1e-6)):
            if 1 <= abs(x / f_) < 1000:
                return f"{g(x / f_)} {p_}{unit}"
        return f"{g(x)} {unit}"

    def sub(slip):
        kind = rng.choice(["ohm_chain", "power_formula", "rc_formula", "divider_chain", "flow", "fed"])
        R = rng.choice([1e3, 2.2e3, 4.7e3, 10e3])
        V = rng.choice([5, 10, 12])
        f = 1.3 if slip else 1.0
        if kind == "ohm_chain":
            return [f"R = {kval(R, om)}", f"V = {V} V", f"I = \\frac{{V}}{{R}} = \\frac{{{V}}}{{{kval(R, om)}}} = {kval(f * V / R, 'A')}"]
        if kind == "power_formula":
            I = rng.choice([2e-3, 5e-3, 1e-3])
            return ["P = I^2 R", f"I = {kval(I, 'A')}", f"R = {kval(R, om)}", f"P = {kval(f * I * I * R, 'W')}"]
        if kind == "rc_formula":
            C = rng.choice([1e-6, 100e-9, 10e-9])
            return ["f = \\frac{1}{2\\pi R C}", f"R = {kval(R, om)}", f"C = {kval(C, 'F')}", f"f = {kval(f / (2 * math.pi * R * C), 'Hz')}"]
        if kind == "divider_chain":
            R2 = rng.choice([1e3, 3.3e3])
            return [f"V_{{out}} = V \\frac{{R_2}}{{R + R_2}} = {V} \\cdot \\frac{{{kval(R2, om)}}}{{{kval(R, om)} + {kval(R2, om)}}} = "
                    f"{g(f * V * R2 / (R + R2))} V"]
        if kind == "fed":
            I = V / R
            return ["P = I^2 R", "I = \\frac{V}{R}", f"V = {V} V", f"R = {kval(R, om)}", f"I = {kval(I, 'A')}", f"P = {kval(f * I * I * R, 'W')}"]
        I = V / R
        return [f"V = {V} V", f"R = {kval(R, om)}", f"I = \\frac{{V}}{{R}} = {kval(I, 'A')}", f"P = V I = {kval(f * V * I, 'W')}"]

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = n_slip = 0
    for _ in range(n):
        parts = rng.choice([2, 3])
        slip_at = rng.choice([None] + list(range(parts)))
        page, numbered = [], rng.random() < 0.3
        for i in range(parts):
            lines = sub(i == slip_at)
            if numbered:
                lines[0] = f"{i + 1}) " + lines[0]
            page += lines
        vr = verdicts(page)
        wrongs = [v for v in vr if v[0] in ("wrong", "wrong step") and v[1]]
        if (slip_at is None and wrongs) or (slip_at is not None and len(wrongs) > 1):
            bad += 1
            print("WRONG?", slip_at, page, vr)
        if slip_at is not None:
            n_slip += 1
            if not any(v[0] in ("wrong", "wrong step") for v in vr):
                missed += 1
                print("MISSED", slip_at, page, vr)
    return n, bad, n_slip, missed


def exam_random(n=100, seed=53):
    """one question answered in sub-questions marked every way ('a)', '(b)', 'c.', 'א)', 'Part d:', '\\text{a)}' or none),
    each working out a value from the data and the parts before it - with numbers put in, without, or the formula on a
    line of its own - in a series circuit, an RC circuit, an AC circuit; one slip in most, the parts after it carrying
    the slipped value on: never a sure wrong but the slip's, the slip caught, and a correct answer never without a
    'right'"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "\\Omega"
    marks = [lambda i: f"{'abcd'[i]})", lambda i: f"({'abcd'[i]})", lambda i: f"{'abcd'[i]}.", lambda i: f"{'אבגד'[i]})",
             lambda i: f"Part {'abcd'[i]}:", lambda i: f"\\text{{{'abcd'[i]})}}", lambda i: ""]

    def g(x):
        return f"{x:.3g}"

    def val(x, unit):
        for p_, f_ in (("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("\\mu ", 1e-6), ("n", 1e-9)):
            if 1 <= abs(x / f_) < 1000:
                return f"{g(x / f_)} {p_}{unit}"
        return f"{g(x)} {unit}"

    def question():
        kind = rng.choice(["dc", "rc", "ac"])
        if kind == "dc":
            vs, r1, r2 = rng.choice([5, 10, 12]), rng.choice([1e3, 2.2e3, 4.7e3]), rng.choice([1e3, 3.3e3, 10e3])
            return [f"V_s = {vs} V", f"R_1 = {val(r1, om)}", f"R_2 = {val(r2, om)}"], [
                ("R_{eq}", "R_1 + R_2", lambda v: f"{val(r1, om)} + {val(r2, om)}", lambda v: r1 + r2, om),
                ("I", "\\frac{V_s}{R_{eq}}", lambda v: f"\\frac{{{vs}}}{{{val(v['R_{eq}'], om)}}}", lambda v: vs / v["R_{eq}"], "A"),
                ("V_2", "I R_2", lambda v: f"{val(v['I'], 'A')} \\cdot {val(r2, om)}", lambda v: v["I"] * r2, "V"),
                ("P_2", "I^2 R_2", lambda v: f"({val(v['I'], 'A')})^2 \\cdot {val(r2, om)}", lambda v: v["I"] ** 2 * r2, "W")]
        if kind == "rc":
            r, c, vv = rng.choice([1e3, 4.7e3, 10e3]), rng.choice([1e-6, 100e-9, 10e-6]), rng.choice([5, 10])
            return [f"R = {val(r, om)}", f"C = {val(c, 'F')}", f"V = {vv} V"], [
                ("\\tau", "R C", lambda v: f"{val(r, om)} \\cdot {val(c, 'F')}", lambda v: r * c, "s"),
                ("f_c", "\\frac{1}{2\\pi \\tau}", lambda v: f"\\frac{{1}}{{2\\pi \\cdot {val(v['\\tau'], 's')}}}",
                 lambda v: 1 / (2 * math.pi * v["\\tau"]), "Hz"),
                ("I_0", "\\frac{V}{R}", lambda v: f"\\frac{{{vv}}}{{{val(r, om)}}}", lambda v: vv / r, "A"),
                ("E", "\\frac{1}{2} C V^2", lambda v: f"\\frac{{1}}{{2}} \\cdot {val(c, 'F')} \\cdot {vv}^2", lambda v: 0.5 * c * vv * vv, "J")]
        r, c, f, vv = rng.choice([100, 220, 1e3]), rng.choice([1e-6, 10e-6]), rng.choice([50, 1e3]), rng.choice([10, 230])
        return [f"R = {val(r, om)}", f"C = {val(c, 'F')}", f"f = {val(f, 'Hz')}", f"V = {vv} V"], [
            ("X_C", "\\frac{1}{2\\pi f C}", lambda v: f"\\frac{{1}}{{2\\pi \\cdot {val(f, 'Hz')} \\cdot {val(c, 'F')}}}",
             lambda v: 1 / (2 * math.pi * f * c), om),
            ("|Z|", "\\sqrt{R^2 + X_C^2}", lambda v: f"\\sqrt{{{g(r)}^2 + {g(v['X_C'])}^2}}", lambda v: math.hypot(r, v["X_C"]), om),
            ("I", "\\frac{V}{|Z|}", lambda v: f"\\frac{{{vv}}}{{{val(v['|Z|'], om)}}}", lambda v: vv / v["|Z|"], "A"),
            ("P", "I^2 R", lambda v: f"({val(v['I'], 'A')})^2 \\cdot {val(r, om)}", lambda v: v["I"] ** 2 * r, "W")]

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = n_slip = 0
    for _ in range(n):
        slip_at = rng.choice([None, 0, 1, 2, 3])
        given, parts = question()
        mark = rng.choice(marks)
        lines, v = list(given), {}
        for i, (name, formula, numbers, fn, unit) in enumerate(parts):
            x = fn(v) * (1.3 if i == slip_at else 1.0)
            v[name] = x
            m = (mark(i) + " ") if mark(i) else ""
            style = rng.choice(["full", "full", "bare", "split"])
            if style == "full":
                lines.append(f"{m}{name} = {formula} = {numbers(v)} = {val(x, unit)}")
            elif style == "bare":
                lines.append(f"{m}{name} = {formula} = {val(x, unit)}")
            else:
                lines += [f"{m}{name} = {formula}", f"{name} = {numbers(v)} = {val(x, unit)}"]
        vr = verdicts(lines)
        wrongs = [w for w in vr if w[0] in ("wrong", "wrong step") and w[1]]
        if (slip_at is None and (wrongs or not any(w[0] == "right" for w in vr))) or (slip_at is not None and len(wrongs) > 1):
            bad += 1
            print("WRONG?", slip_at, lines, vr)
        if slip_at is not None:
            n_slip += 1
            if not any(w[0] in ("wrong", "wrong step") for w in vr):
                missed += 1
                print("MISSED", slip_at, lines, vr)
    return n, bad, n_slip, missed


def redo_random(n=80, seed=59):
    """a line done again just below itself, its result put right (the first try crossed out - the reader does not see
    strokes): a computation, a formula's answer, a formula's worked-out line; and an equation's roots one after another:
    the first try never a sure wrong, the second (and the lines using it) right; a wrong root caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "\\Omega"

    def g(x):
        return f"{x:.3g}"

    def kval(x, unit):
        for p_, f_ in (("k", 1e3), ("", 1.0), ("m", 1e-3), ("\\mu ", 1e-6)):
            if 1 <= abs(x / f_) < 1000:
                return f"{g(x / f_)} {p_}{unit}"
        return f"{g(x)} {unit}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = n_roots = 0
    for _ in range(n):
        kind = rng.choice(["chain", "used", "answer", "formula_chain", "roots"])
        r, vv = rng.choice([1e3, 2.2e3, 4.7e3, 10e3]), rng.choice([5, 10, 12])
        i_ = vv / r
        if kind == "roots":
            a, b = rng.sample([-3, -2, -1, 1, 2, 3, 4, 5], 2)
            wrong_b = b + rng.choice([1, 2, -1])
            if wrong_b == a:
                wrong_b += 3
            eq = f"x^2 {'-' if a + b > 0 else '+'} {abs(a + b)}x {'+' if a * b > 0 else '-'} {abs(a * b)} = 0"
            n_roots += 1
            for xs, ok in (([a, b], True), ([a, wrong_b], False), ([wrong_b, a], False)):
                lines = [eq] + (rng.choice([[f"x = {xs[0]}", f"x = {xs[1]}"], [f"x = {xs[0]}, x = {xs[1]}"]]))
                vr = verdicts(lines)
                if ok and any(w[0] in ("wrong", "wrong step") and w[1] for w in vr):
                    bad += 1
                    print("WRONG?", lines, vr)
                if not ok and not any(w[0] == "wrong" for w in vr):
                    missed += 1
                    print("MISSED", lines, vr)
            continue
        if kind == "chain":
            lines = [f"V = {vv} V", f"R = {kval(r, om)}", f"I = \\frac{{V}}{{R}} = \\frac{{{vv}}}{{{kval(r, om)}}} = {kval(1.3 * i_, 'A')}",
                     f"I = \\frac{{V}}{{R}} = \\frac{{{vv}}}{{{kval(r, om)}}} = {kval(i_, 'A')}"]
        elif kind == "used":
            lines = [f"V = {vv} V", f"R = {kval(r, om)}", f"I = \\frac{{V}}{{R}} = {kval(1.3 * i_, 'A')}", f"I = \\frac{{V}}{{R}} = {kval(i_, 'A')}",
                     f"P = V I = {kval(vv * i_, 'W')}"]
        elif kind == "answer":
            ia = rng.choice([2e-3, 5e-3, 1e-3])
            lines = ["P = I^2 R", f"I = {kval(ia, 'A')}", f"R = {kval(r, om)}", f"P = {kval(1.3 * ia * ia * r, 'W')}", f"P = {kval(ia * ia * r, 'W')}"]
        else:
            ia = rng.choice([2e-3, 5e-3, 1e-3])
            sub = f"({kval(ia, 'A')})^2 \\cdot {kval(r, om)}"
            lines = ["P = I^2 R", f"I = {kval(ia, 'A')}", f"R = {kval(r, om)}", f"P = {sub} = {kval(1.3 * ia * ia * r, 'W')}",
                     f"P = {sub} = {kval(ia * ia * r, 'W')}"]
        vr = verdicts(lines)
        if any(w[0] in ("wrong", "wrong step") and w[1] for w in vr) or not any(w[0] == "right" and w[1] for w in vr):
            bad += 1
            print("WRONG?", lines, vr)
        if not any(w[0] in ("wrong", "wrong step") for w in vr):
            missed += 1
            print("MISSED", lines, vr)
    return n, bad, n_roots, missed


def electronics_random(n=100, seed=61):
    """electronics answers in sub-questions (BJT bias with \\beta and V_{BE} = 0.7 V, op-amp gains, a diode's current,
    Thevenin with '\\|' for parallel), values carried on, rounded on the way as students round: never a sure wrong but a
    slip's, the slip caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "\\Omega"

    def g(x):
        return f"{x:.3g}"

    def val(x, unit):
        for p_, f_ in (("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("\\mu ", 1e-6), ("n", 1e-9)):
            if 1 <= abs(x / f_) < 1000:
                return f"{g(x / f_)} {p_}{unit}"
        return f"{g(x)} {unit}"

    def bjt():
        vbb, vcc = rng.choice([5, 3, 10]), rng.choice([12, 15, 10])
        rb, rc, beta = rng.choice([100e3, 47e3, 220e3]), rng.choice([1e3, 2.2e3, 4.7e3]), rng.choice([100, 150, 200])
        return ([f"V_{{BB}} = {vbb} V", f"V_{{CC}} = {vcc} V", f"R_B = {val(rb, om)}", f"R_C = {val(rc, om)}", f"\\beta = {beta}",
                 "V_{BE} = 0.7 V"],
                [("I_B", "\\frac{V_{BB} - V_{BE}}{R_B}", lambda v: f"\\frac{{{vbb} - 0.7}}{{{val(rb, om)}}}", lambda v: (vbb - 0.7) / rb, "A"),
                 ("I_C", "\\beta I_B", lambda v: f"{beta} \\cdot {val(v['I_B'], 'A')}", lambda v: beta * v["I_B"], "A"),
                 ("V_{CE}", "V_{CC} - I_C R_C", lambda v: f"{vcc} - {val(v['I_C'], 'A')} \\cdot {val(rc, om)}",
                  lambda v: vcc - v["I_C"] * rc, "V")],
                lambda v: v["V_{CE}"] > 0.3)

    def opamp():
        rin, rf, vin, rl = rng.choice([1e3, 10e3, 4.7e3]), rng.choice([10e3, 47e3, 100e3]), rng.choice([0.1, 0.05, 0.2]), rng.choice([1e3, 2e3])
        inv = rng.random() < 0.5
        first = (("A_v", "-\\frac{R_f}{R_{in}}", lambda v: f"-\\frac{{{val(rf, om)}}}{{{val(rin, om)}}}", lambda v: -rf / rin, "") if inv else
                 ("A_v", "1 + \\frac{R_f}{R_{in}}", lambda v: f"1 + \\frac{{{val(rf, om)}}}{{{val(rin, om)}}}", lambda v: 1 + rf / rin, ""))
        return ([f"R_{{in}} = {val(rin, om)}", f"R_f = {val(rf, om)}", f"V_{{in}} = {val(vin, 'V')}", f"R_L = {val(rl, om)}"],
                [first, ("V_{out}", "A_v V_{in}", lambda v: f"{g(v['A_v'])} \\cdot {val(vin, 'V')}", lambda v: v["A_v"] * vin, "V"),
                 ("I_L", "\\frac{V_{out}}{R_L}", lambda v: f"\\frac{{{val(v['V_{out}'], 'V')}}}{{{val(rl, om)}}}", lambda v: v["V_{out}"] / rl, "A")],
                lambda v: abs(v["V_{out}"]) < 12)

    def diode():
        vs, r = rng.choice([5, 10, 12]), rng.choice([1e3, 2.2e3, 470])
        return ([f"V_s = {vs} V", f"R = {val(r, om)}", "V_D = 0.7 V"],
                [("I_D", "\\frac{V_s - V_D}{R}", lambda v: f"\\frac{{{vs} - 0.7}}{{{val(r, om)}}}", lambda v: (vs - 0.7) / r, "A"),
                 ("P_D", "V_D I_D", lambda v: f"0.7 \\cdot {val(v['I_D'], 'A')}", lambda v: 0.7 * v["I_D"], "W")], lambda v: True)

    def thevenin():
        vs, r1, r2, rl = rng.choice([10, 12, 9]), rng.choice([1e3, 2.2e3, 4.7e3]), rng.choice([1e3, 3.3e3, 10e3]), rng.choice([1e3, 5e3])
        return ([f"V_s = {vs} V", f"R_1 = {val(r1, om)}", f"R_2 = {val(r2, om)}", f"R_L = {val(rl, om)}"],
                [("V_{th}", "V_s \\frac{R_2}{R_1 + R_2}", lambda v: f"{vs} \\cdot \\frac{{{val(r2, om)}}}{{{val(r1, om)} + {val(r2, om)}}}",
                  lambda v: vs * r2 / (r1 + r2), "V"),
                 ("R_{th}", "R_1 \\| R_2", lambda v: f"{val(r1, om)} \\| {val(r2, om)}", lambda v: r1 * r2 / (r1 + r2), om),
                 ("I_L", "\\frac{V_{th}}{R_{th} + R_L}", lambda v: f"\\frac{{{val(v['V_{th}'], 'V')}}}{{{val(v['R_{th}'], om)} + {val(rl, om)}}}",
                  lambda v: v["V_{th}"] / (v["R_{th}"] + rl), "A")], lambda v: True)

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = n_slip = 0
    done = 0
    while done < n:
        given, parts, ok_fn = rng.choice([bjt, opamp, diode, thevenin])()
        slip_at = rng.choice([None] + list(range(len(parts))))
        mark = rng.choice([lambda i: f"{'abcd'[i]}) ", lambda i: f"({'abcd'[i]}) ", lambda i: "", lambda i: f"{'אבגד'[i]}. "])
        lines, v = list(given), {}
        for i, (name, formula, numbers, fn, unit) in enumerate(parts):
            x = fn(v) * (1.3 if i == slip_at else 1.0)
            v[name] = x
            shown = val(x, unit) if unit else g(x)
            style = rng.choice(["full", "full", "bare", "split"])
            if style == "full":
                lines.append(f"{mark(i)}{name} = {formula} = {numbers(v)} = {shown}")
            elif style == "bare":
                lines.append(f"{mark(i)}{name} = {formula} = {shown}")
            else:
                lines += [f"{mark(i)}{name} = {formula}", f"{name} = {numbers(v)} = {shown}"]
        if not ok_fn(v):
            continue
        done += 1
        vr = verdicts(lines)
        wrongs = [w for w in vr if w[0] in ("wrong", "wrong step") and w[1]]
        if (slip_at is None and (wrongs or not any(w[0] == "right" for w in vr))) or (slip_at is not None and len(wrongs) > 1):
            bad += 1
            print("WRONG?", slip_at, lines, vr)
        if slip_at is not None:
            n_slip += 1
            if not any(w[0] in ("wrong", "wrong step") for w in vr):
                missed += 1
                print("MISSED", slip_at, lines, vr)
    return n, bad, n_slip, missed


def boolean_random(n=150, seed=67):
    """Boolean algebra as written in a digital-systems exam: known identities (absorption, De Morgan, distribution,
    consensus, XOR) and the classic wrong ones, letters renamed, complements as \\bar{X}, \\overline{X} or X', AND side by
    side or \\cdot, a function's name or none, on a line or simplified over lines: a right one never judged wrong and
    never without a 'right' (when it has a complement or XOR - else it is algebra as much), a wrong one caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    right = ["A + AB = A", "A(A + B) = A", "A + ~AB = A + B", "~(A + B) = ~A~B", "~(AB) = ~A + ~B", "(A + B)(A + C) = A + BC",
             "AB + ~AC + BC = AB + ~AC", "A ^ B = A~B + ~AB", "~(A ^ B) = AB + ~A~B", "A + 1 = 1", "AB + A~B = A",
             "(A + B)(~A + C) = AC + ~AB", "A~A = 0", "A + ~A = 1", "~~A = A"]
    wrong = ["~(A + B) = ~A + ~B", "~(AB) = ~A~B", "A + AB = B", "A + ~AB = B", "(A + B)(A + C) = AB + C", "A ^ B = AB + ~A~B",
             "AB + A~B = B", "A~A = 1", "~(A ^ B) = A~B + ~AB", "AB + ~AC + BC = AB"]

    def render(t, names, comp, cdot):
        out, i = [], 0
        while i < len(t):
            c = t[i]
            if c == "~" and t[i + 1] == "(":
                depth, j = 0, i + 1
                while True:
                    depth += {"(": 1, ")": -1}.get(t[j], 0)
                    if depth == 0:
                        break
                    j += 1
                inner = render(t[i + 2:j], names, comp, cdot)
                out.append(f"({inner})'" if comp == "prime" else f"\\overline{{{inner}}}")
                i = j + 1
                continue
            if c == "~" and t[i + 1] == "~":
                out.append(names[t[i + 2]] + "''" if comp == "prime" else "\\overline{\\overline{" + names[t[i + 2]] + "}}")
                i += 3
                continue
            prev_atom = bool(out) and (out[-1][-1:].isalpha() or out[-1][-1:] in "})'")
            if c == "~":
                x = names[t[i + 1]]
                if cdot and prev_atom:
                    out.append(" \\cdot ")
                out.append({"bar": f"\\bar{{{x}}}", "overline": f"\\overline{{{x}}}", "prime": f"{x}'"}[comp])
                i += 2
                continue
            if c in names or c == "(":
                if cdot and prev_atom:
                    out.append(" \\cdot ")
                out.append(names.get(c, c))
            elif c == "^":
                out.append(" \\oplus ")
            else:
                out.append(c)
            i += 1
        return "".join(out)

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = n_wrong = 0
    for _ in range(n):
        ok = rng.random() < 0.6
        t = rng.choice(right if ok else wrong)
        names = dict(zip("ABC", rng.choice([("A", "B", "C"), ("X", "Y", "Z"), ("P", "Q", "R"), ("x", "y", "z")])))
        comp, cdot = rng.choice(["bar", "overline", "prime"]), rng.random() < 0.3
        left, rt = [render(x.strip(), names, comp, cdot) for x in t.split("=")]
        fn = rng.choice(["F", "G", "Y"])
        style = rng.choice(["plain", "named", "func", "lines"])
        if style == "plain":
            lines = [f"{left} = {rt}"]
        elif style == "named":
            lines = [f"{fn} = {left} = {rt}"]
        elif style == "func":
            lines = [f"{fn}({','.join(names[c] for c in 'ABC' if c in t)}) = {left} = {rt}"]
        else:
            lines = [f"{fn} = {left}", f"{fn} = {rt}"]
        signed = "~" in t or "^" in t
        vr = verdicts(lines)
        if ok and (any(w[0] in ("wrong", "wrong step") for w in vr) or (signed and not any(w[0] == "right" for w in vr))):
            bad += 1
            print("WRONG?", lines, vr)
        if not ok:
            n_wrong += 1
            if any(w[0] == "right" for w in vr) or (signed and not any(w[0] in ("wrong", "wrong step") for w in vr)):
                missed += 1
                print("MISSED", lines, vr)
    return n, bad, n_wrong, missed


def bases_random(n=80, seed=71):
    """numbers in bases 2, 8 and 16 written every way ('(1011)_2', '1011_2', '1011_B', '0b1011', '(1F)_{16}', '1F_{16}', '0x1F',
    '(1F)_H', '37_8', '37_O'), against their decimal value, right and one off: none right judged wrong, none wrong passed"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]

    bad = missed = 0
    for _ in range(n):
        v = rng.randint(2, 4000)
        base = rng.choice([2, 8, 16])
        digits = format(v, {2: "b", 8: "o", 16: "X"}[base])
        if base == 16 and not any(c.isalpha() for c in digits) and rng.random() < 0.5:
            v, digits = 255, "FF"
        forms = {2: [f"({digits})_2", f"{digits}_2", f"{digits}_B", f"0b{digits}", f"({digits})_{{2}}"],
                 8: [f"({digits})_8", f"{digits}_8", f"{digits}_O"],
                 16: [f"({digits})_{{16}}", f"{digits}_{{16}}", f"0x{digits}", f"({digits})_H"]}[base]
        f = rng.choice(forms)
        if base == 2 and f == f"{digits}_B" and len(digits) < 2:
            continue
        for value, ok in ((v, True), (v + 1, False)):
            t = rng.choice([f"{f} = {value}", f"{value} = {f}", f"{f} = ({value})_{{10}}"])
            vr = verdicts(t)
            if ok and vr != [("right", True)]:
                bad += 1
                print("WRONG?", t, vr)
            if not ok and not any(w[0] == "wrong step" for w in vr):
                missed += 1
                print("MISSED", t, vr)
    return n, bad, n, missed


def response_random(n=100, seed=73):
    """a frequency response and its size / angle at frequencies written in it ('H(j\\omega) = \\frac{10}{10 + j\\omega}', '|H(j10)| =
    0.707', '\\angle H(j10) = -45^\\circ'; low-pass, high-pass, with R and C given), one to three frequencies: none right judged
    wrong or left without a 'right', a size x1.3 or an angle 15° off caught"""
    import cmath
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def g(x):
        return f"{x:.3g}"

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]

    bad = missed = 0
    for _ in range(n):
        a = rng.choice([10, 100, 1000, 50])
        kind = rng.choice(["lp", "hp", "lp_k"])
        k = rng.choice([1, 2, 10])
        if kind == "lp":
            head, h = f"H(j\\omega) = \\frac{{{a}}}{{{a} + j\\omega}}", (lambda w: a / (a + 1j * w))
        elif kind == "hp":
            head, h = f"H(j\\omega) = \\frac{{j\\omega}}{{{a} + j\\omega}}", (lambda w: 1j * w / (a + 1j * w))
        else:
            head, h = f"H(j\\omega) = \\frac{{{k * a}}}{{{a} + j\\omega}}", (lambda w: k * a / (a + 1j * w))
        ws = rng.sample([a / 10, a, a * 10, a * 2], rng.choice([1, 2, 3]))
        slip = rng.choice([None] + list(range(len(ws))))
        lines = [head]
        for i, w in enumerate(ws):
            ws_t = f"{w:.0f}" if w >= 1 else g(w)
            if rng.random() < 0.6:
                lines.append(f"|H(j{ws_t})| = {g(abs(h(w)) * (1.3 if i == slip else 1))}")
            else:
                ang = math.degrees(cmath.phase(h(w))) + (15 if i == slip else 0)
                lines.append(f"\\angle H(j{ws_t}) = {g(ang)}^\\circ")
        vr = verdicts(lines)
        wrongs = [w for w in vr if w[0] in ("wrong", "wrong step")]
        if (slip is None and (wrongs or not any(w[0] == "right" for w in vr))) or (slip is not None and len([w for w in wrongs if w[1]]) > 1):
            bad += 1
            print("WRONG?", lines, vr)
        if slip is not None and not wrongs:
            missed += 1
            print("MISSED", lines, vr)
    return n, bad, n, missed


def algebra_random(n=150, seed=79):
    """expansions and factorings written along a chain ('y = (x+2)(x-3) = x^2 - x - 6', '(a+b)^2', 'k(x + c)'), right and with
    a slip in one coefficient, and equations along a chain that hold at one point only ('V_A = 10 - 2I = 4I + 2', 'V = 3(4 -
    I) = 2I + 2'): an expansion right never judged wrong, a slip caught, an equation never judged"""
    import random
    import sympy as sp
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]

    def tex(e):
        return sp.latex(e).replace("\\left", "").replace("\\right", "")

    bad = missed = 0
    x, y = sp.symbols("x y")
    for _ in range(n):
        kind = rng.choice(["two", "square", "scale", "two_var", "equation"])
        if kind == "equation":
            a, b, c, d = rng.sample([2, 3, 4, 5, 6, 10, 12], 4)
            v = rng.choice(["I", "x", "t"])           # (not V: '2V' is 2 volts)
            t = rng.choice([f"V_A = {a} - {b}{v} = {c}{v} + {d}", f"V = {b}({a} - {v}) = {c}{v} + {d}"])
            vr = verdicts(t)
            if any(w[0].startswith("wrong") for w in vr):
                bad += 1
                print("WRONG?", t, vr)
            continue
        if kind == "two":
            p_, q = rng.choice([1, 2, 3, 4, -1, -2, -5]), rng.choice([1, 3, 5, -3, -4, 6])
            f = (x + p_) * (x + q)
        elif kind == "square":
            p_ = rng.choice([1, 2, 3, -1, -4])
            f = (x + p_) ** 2
        elif kind == "scale":
            k, c = rng.choice([2, 3, 5, -2]), rng.choice([1, 3, 4, -6])
            f = sp.Mul(k, x + c, evaluate=False)     # (as written: '3(x + 4)' - not expanded on the way)
        else:
            f = (x + y) ** 2 if rng.random() < 0.5 else (x - y) * (x + y)
        e = sp.expand(f)
        terms = list(sp.Add.make_args(e))
        lead = max(terms, key=lambda t_: sp.Poly(t_, *sorted(e.free_symbols, key=str)).total_degree())
        rest = [t_ for t_ in terms if t_ != lead]
        slip = e + (rng.choice([1, -1, 2]) * (rest[0] / abs(sp.Poly(rest[0], *sorted(e.free_symbols, key=str)).coeffs()[0]) if rest else 1))
        if slip == e:
            slip = e + 1
        name = rng.choice(["y", "f", "g"])
        for rhs, ok in ((e, True), (slip, False)):
            t = f"{name} = {tex(f)} = {tex(rhs)}" if rng.random() < 0.7 else f"{name} = {tex(rhs)} = {tex(f)}"
            vr = verdicts(t)
            if ok and any(w[0].startswith("wrong") for w in vr):
                bad += 1
                print("WRONG?", t, vr)
            if not ok and not any(w[0].startswith("wrong") for w in vr):
                missed += 1
                print("MISSED", t, vr)
    return n, bad, n, missed


def slash_random(n=150, seed=83):
    """formulas written with slashes inline ('1/2 C V^2', 'R_2/(R_1+R_2) V_{in}', '1/2\\pi\\sqrt{LC}', '(R)/(2L)', 'V^2/R',
    '20 \\log(V_o/V_i)'), the data before or after them, the answer below or on the line: a right one never a sure wrong,
    one x1.3 never right"""
    import math
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "\\Omega"

    def pre(x, unit):
        for p, f in (("k", 1e3), ("", 1.0), ("m", 1e-3), ("\\mu ", 1e-6), ("n", 1e-9)):
            if 1 <= abs(x / f) < 1000:
                return f"{x / f:.3g} {p}{unit}"
        return f"{x:.3g} {unit}"
    forms = [
        ("E", "J", ["1/2 C V^2", "(1)/(2) C V^2", "C V^2/2"], [("C", "F", [1e-6, 10e-6, 100e-9]), ("V", "V", [5, 10, 12])],
         lambda d: 0.5 * d["C"] * d["V"] ** 2),
        ("E", "J", ["1/2 L I^2", "L I^2/2"], [("L", "H", [1e-3, 10e-3, 0.1]), ("I", "A", [0.5, 2, 3])], lambda d: 0.5 * d["L"] * d["I"] ** 2),
        ("V_o", "V", ["R_2/(R_1+R_2) V_{in}", "V_{in} R_2/(R_1+R_2)", "(R_2)/(R_1 + R_2) V_{in}"],
         [("R_1", om, [1e3, 2.2e3, 10e3]), ("R_2", om, [1e3, 4.7e3]), ("V_{in}", "V", [5, 9, 12])],
         lambda d: d["V_{in}"] * d["R_2"] / (d["R_1"] + d["R_2"])),
        ("f", "Hz", ["1/2\\pi\\sqrt{LC}", "1/(2\\pi\\sqrt{LC})"], [("L", "H", [1e-3, 10e-3]), ("C", "F", [1e-6, 100e-9])],
         lambda d: 1 / (2 * math.pi * math.sqrt(d["L"] * d["C"]))),
        ("\\alpha", "", ["R/2L", "(R)/(2L)"], [("R", om, [10, 100]), ("L", "H", [1e-3, 0.1])], lambda d: d["R"] / (2 * d["L"])),
        ("P", "W", ["V^2/R", "(V^2)/(R)"], [("V", "V", [5, 10]), ("R", om, [10, 100])], lambda d: d["V"] ** 2 / d["R"]),
        ("A_v", "dB", ["20 \\log(V_o/V_i)", "20 \\log_{10}(V_o/V_i)"], [("V_o", "V", [5, 2]), ("V_i", "V", [50e-3, 10e-3])],
         lambda d: 20 * math.log10(d["V_o"] / d["V_i"])),
        ("X_C", om, ["1/2\\pi f C", "1/(2\\pi f C)"], [("f", "Hz", [50, 1e3]), ("C", "F", [1e-6, 100e-9])],
         lambda d: 1 / (2 * math.pi * d["f"] * d["C"])),
    ]

    def page(f):
        name, unit, fs, spec, fn = rng.choice(forms)
        form = rng.choice(fs)
        d = {k: rng.choice(vs) for k, u, vs in spec}
        val = fn(d) * f
        data = [f"{k} = {pre(d[k], u)}" for k, u, _ in spec]
        ans = f"{val:.3g} dB" if unit == "dB" else (f"{val:.3g}" if not unit else pre(val, unit))
        style = rng.choice(["before", "after", "chain"])
        if style == "before":
            return data + [f"{name} = {form}", f"{name} = {ans}"]
        if style == "after":
            return [f"{name} = {form}"] + data + [f"{name} = {ans}"]
        return data + [f"{name} = {form} = {ans}"]

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    bad = missed = 0
    for _ in range(n):
        state = rng.getstate()
        right = page(1.0)
        rng.setstate(state)
        wrong = page(1.3)
        vr, vw = verdicts(right), verdicts(wrong)
        if any(v[0].startswith("wrong") and v[1] for v in vr) or not any(v[0] == "right" for v in vr):
            bad += 1
            print("WRONG?", right, vr)
        if any(v[0] == "right" and v[1] for v in vw) or not any(v[0].startswith("wrong") for v in vw):
            missed += 1
            print("MISSED", wrong, vw)
    return n, bad, n, missed


def juxtaposed_random(n=400, seed=89):
    """products of numbers, fractions, roots, powers, brackets, \\pi and e-powers written side by side (no sign, a space,
    \\cdot, \\times) - parsed as their product (a number before a fraction, a mixed number, and a digit after a digit, one
    number, left out)"""
    import math
    import random
    from grader.selfcheck import _num, _parse
    rng = random.Random(seed)

    def atom():
        k = rng.choice(["int", "dec", "frac", "sqrt", "pow", "par", "pi", "exp"])
        if k == "int":
            v = rng.choice([2, 3, 4, 5, 10, 12, 20])
            return str(v), float(v)
        if k == "dec":
            v = rng.choice([0.5, 1.5, 2.2, 4.7, 0.25])
            return str(v), v
        if k == "frac":
            a, b = rng.choice([1, 2, 3, 6, 10]), rng.choice([2, 3, 4, 5])
            return f"\\frac{{{a}}}{{{b}}}", a / b
        if k == "sqrt":
            c = rng.choice([2, 4, 9, 10])
            return f"\\sqrt{{{c}}}", math.sqrt(c)
        if k == "pow":
            b, e = rng.choice([2, 3, 10]), rng.choice([2, 3])
            return f"{b}^{{{e}}}", float(b ** e)
        if k == "par":
            a, b = rng.choice([1, 2, 3]), rng.choice([1, 2, 4])
            return f"({a} + {b})", float(a + b)
        if k == "pi":
            return "\\pi", math.pi
        c = rng.choice([1, 2])
        return f"e^{{-{c}}}", math.exp(-c)
    bad = tried = 0
    while tried < n:
        atoms = [atom() for _ in range(rng.choice([2, 3]))]
        seps = [rng.choice([" ", "", " \\cdot ", " \\times "]) for _ in atoms[1:]]
        txt, val, doubt = atoms[0][0], atoms[0][1], False
        for (t, v), sep in zip(atoms[1:], seps):
            if sep in (" ", "") and t[0].isdigit() and (txt[-1].isdigit() or txt[-1] == "."):
                doubt = True                         # (one number, by the readers' spacing)
            if sep in (" ", "") and t.startswith("\\frac") and txt[-1].isdigit():
                doubt = True                         # (a mixed number)
            txt, val = txt + sep + t, val * v
        if doubt:
            continue
        tried += 1
        e = _parse(txt)
        got = _num(e) if e is not None else None
        if not isinstance(got, (int, float)) or abs(got - val) > 1e-6 * max(1.0, abs(val)):
            bad += 1
            print("MISREAD", txt, e, val)
    return n, bad, 0, 0


def typed_random(n=150, seed=97):
    """EE computations typed as plain text ('I = V/R = 12/4k = 3mA', 'f = 1/(2*pi*R*C) = 1/(2*pi*1k*1u) = 159 Hz', 'Z =
    sqrt(R^2 + X^2) = ...', 'tau = L/R = 10m/10 = 1ms'): a right one judged right, never a sure wrong; x1.3 caught"""
    import math
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def g(x):
        return f"{x:.3g}"

    def pre(x):
        for p_, f in (("M", 1e6), ("k", 1e3), ("", 1.0), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9)):
            if 1 <= abs(x / f) < 1000:
                return f"{g(x / f)}{p_}"
        return g(x)

    def page(f):
        kind = rng.choice(["ohm", "power", "rc", "tau", "imped", "xc", "rms", "divider", "energy"])
        star = rng.choice(["*", " * "])
        if kind == "ohm":
            V, R = rng.choice([5, 9, 12]), rng.choice([1e3, 2.2e3, 4.7e3])
            return f"I = V/R = {V}/{pre(R)} = {pre(V / R * f)}A"
        if kind == "power":
            I, R = rng.choice([1e-3, 2e-3, 5e-3]), rng.choice([1e3, 2.2e3])
            return f"P = I^2{star}R = ({pre(I)})^2{star}{pre(R)} = {pre(I ** 2 * R * f)}W"
        if kind == "rc":
            R, C = rng.choice([1e3, 10e3]), rng.choice([1e-6, 100e-9])
            return f"f = 1/(2{star}pi{star}R{star}C) = 1/(2{star}pi{star}{pre(R)}{star}{pre(C)}) = {pre(f / (2 * math.pi * R * C))} Hz"
        if kind == "tau":
            L, R = rng.choice([10e-3, 1e-3, 0.1]), rng.choice([10, 100, 1e3])
            return f"tau = L/R = {pre(L)}/{pre(R)} = {pre(L / R * f)}s"
        if kind == "imped":
            R, X = rng.choice([(300, 400), (30, 40), (5, 12)])
            return f"Z = sqrt(R^2 + X^2) = sqrt({R}^2 + {X}^2) = {g(math.hypot(R, X) * f)} ohm"
        if kind == "xc":
            w, C = rng.choice([314, 1000]), rng.choice([10e-6, 1e-6])
            return f"XC = 1/(w{star}C) = 1/({w}{star}{pre(C)}) = {g(f / (w * C))} ohm"
        if kind == "rms":
            Vp = rng.choice([10, 5, 325])
            return f"Vrms = Vp/sqrt(2) = {Vp}/sqrt(2) = {g(Vp / math.sqrt(2) * f)} V"
        if kind == "divider":
            V, R1, R2 = rng.choice([9, 12]), rng.choice([1e3, 2e3]), rng.choice([2e3, 3e3])
            return f"Vo = V{star}R2/(R1+R2) = {V}{star}{pre(R2)}/({pre(R1)}+{pre(R2)}) = {g(V * R2 / (R1 + R2) * f)}V"
        C, V = rng.choice([10e-6, 1e-6]), rng.choice([5, 10])
        return f"E = 0.5{star}C{star}V^2 = 0.5{star}{pre(C)}{star}{V}^2 = {pre(0.5 * C * V ** 2 * f)}J"

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    bad = missed = 0
    for _ in range(n):
        state = rng.getstate()
        right = page(1.0)
        rng.setstate(state)
        wrong = page(1.3)
        vr, vw = verdicts(right), verdicts(wrong)
        if any(v[0].startswith("wrong") and v[1] for v in vr) or not any(v[0] == "right" for v in vr):
            bad += 1
            print("WRONG?", right, vr)
        if not any(v[0].startswith("wrong") for v in vw):
            missed += 1
            print("MISSED", wrong, vw)
    return n, bad, n, missed


def hebrew_random(n=150, seed=101):
    """pages with Hebrew words around the math ('נתון: V = 12 V, R = 4 k\\Omega', 'לכן I = ...', 'נקבל ...', 'I = 3 mA :תשובה',
    '\\text{ו-}', 'סעיף ב: ...', a sentence's period): a right page never a sure wrong nor silent, a slip (x1.3) caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    om = "k\\Omega"

    def page(f):
        V, R = rng.choice([5, 9, 12, 10]), rng.choice([1, 2, 4, 5])
        I = V / R
        style = rng.choice(["colon", "given", "plain", "text", "notes"])
        data = {"colon": [f"נתון: V = {V} V, R = {R} {om}"], "given": [f"נתונים V = {V} V , R = {R} {om}"],
                "plain": [f"V = {V} V", f"R = {R} {om}"], "text": [f"\\text{{נתון }} V = {V} V \\text{{ ו- }} R = {R} {om}"],
                "notes": [f"V = {V} V \\quad \\text{{מתח המקור}}", f"R = {R} {om} \\quad \\text{{הנגד}}"]}[style]
        lead = rng.choice(["לכן ", "מכאן: ", "נקבל ", "ולכן: ", "", "\\text{לכן } "])
        tail = rng.choice(["", " (תשובה)", " \\text{ תשובה סופית}", " :תשובה", "."])
        work = [f"{lead}I = \\frac{{V}}{{R}} = \\frac{{{V}}}{{{R}k}} = {I * f:.3g} mA{tail}"]
        if rng.random() < 0.5:
            work.append(f"{rng.choice(['סעיף ב: ', 'ב) ', '', 'נציב: '])}P = V I = {V} \\cdot {I:.3g} m = {V * I:.3g} mW")
        return data + work

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    bad = missed = 0
    for _ in range(n):
        state = rng.getstate()
        right = page(1.0)
        rng.setstate(state)
        wrong = page(1.3)
        vr, vw = verdicts(right), verdicts(wrong)
        if any(v[0].startswith("wrong") and v[1] for v in vr) or not any(v[0] == "right" for v in vr):
            bad += 1
            print("WRONG?", right, vr)
        if not any(v[0].startswith("wrong") for v in vw):
            missed += 1
            print("MISSED", wrong, vw)
    return n, bad, n, missed


def digital_random(n=130, seed=103):
    """pages of a digital-circuits (CMOS) exam, as its official solution writes them: the nodes' states ('A = 1, B = 0',
    'V_x = 1S', '\\bar{V_x} = 0S', 'A = B = V_{DD}'), indexed names ('D0', 'A_1', 'Y1'), nodes defined then used, the
    inputs' values put in ('Y = \\overline{AB} = \\overline{1 \\cdot 0} = 1'), a multiplexer at its select's value, a
    decoder; CMOS numbers - the body effect (gamma in square-root volts), the square law, delays (bare prefixes: '2f \\cdot
    5k'), noise margins, a switching threshold, an Elmore delay, logical effort along a path, power, values at an instant ('V_{out}(0^+)').
    A right page never a sure wrong (nor without a 'right', but for the states'), a slip caught"""
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def g(x):
        return f"{x:.3g}"


    def bar(x):
        return rng.choice([f"\\bar{{{x}}}", f"\\overline{{{x}}}", f"{x}'"]) if not x.startswith("\\") else f"\\overline{{{x}}}"


    def idx(base, i):
        return rng.choice([f"{base}{i}", f"{base}_{i}", f"{base}_{{{i}}}"])


    def states_page(slip):
        """node states over a page with a Boolean line on it - no identities to judge; a slip: the Boolean line wrong"""
        a, b = rng.choice([("A", "B"), ("X", "Y"), ("P", "Q")])
        lines = [f"F = {bar(a)} + {b}" if not slip else f"F = \\overline{{{a} + {b}}} = {bar(a)} + {bar(b)}"]
        pool = [f"{a} = {rng.choice('01')}, {b} = {rng.choice('01')}",
                f"V_{{x}} = {rng.choice(['0S', '1S', '0W', '1W', '0', '1'])}",
                f"V_{{x}} = {rng.choice(['0S', '1S'])},\\ {bar('V_{x}')} = {rng.choice(['0S', '1S', 'V_{DD}', '0'])}",
                f"{a} = {b} = V_{{DD}} \\Rightarrow V_{{x}} = 0S",
                f"{idx('Y', rng.choice([1, 2]))} = {rng.choice(['0S', '1S', '0', '1'])}",
                f"{a} = 0 \\Rightarrow V_{{out}} = 1S",
                f"V_{{x}} = 1S", f"V_{{x}} = 0S"]
        lines += rng.sample(pool, rng.randint(2, 5))
        return lines, ("wrong" if slip else None)


    def adder_page(slip):
        """a full adder through its propagate and generate: nodes defined, then put in"""
        cin = rng.choice(["C_{in}", "C_{i}", "C"])
        p, gg = rng.choice([("P", "G"), ("X", "Z")])
        lines = [f"{p} = A \\oplus B", f"{gg} = AB", f"S = {p} \\oplus {cin}"]
        tail = f"AB + (A \\oplus B){cin}" if not slip else f"AB + A{cin}"
        lines.append(f"C_{{out}} = {gg} + {p}{cin} = {tail}")
        if rng.random() < 0.5:
            lines.append(f"S = A \\oplus B \\oplus {cin}")
        return lines, ("wrong" if slip else None)


    def mux_page(slip):
        """a 2:1 multiplexer with indexed inputs at a select value"""
        d0, d1 = rng.choice([("D0", "D1"), ("D_0", "D_1"), ("I_0", "I_1"), ("A0", "A1")])
        s = rng.choice(["S", "S_0", "Sel"]) if False else rng.choice(["S", "S_0"])
        y = rng.choice(["Y", "F", "Y_1"])
        sv = rng.choice([0, 1])
        lines = [f"{y} = {bar(s)}{d0} + {s}{d1}", f"{s} = {sv}"]
        picked = d1 if sv else d0
        other = d0 if sv else d1
        res = picked if not slip else other
        lines.append(f"{y} = \\bar{{{sv}}}{d0} + {sv} \\cdot {d1} = {res}")
        return lines, ("wrong" if slip else None)


    def eval_page(slip):
        """a gate's output at the inputs' values: put in, then worked out"""
        a, b = rng.choice([("A", "B"), ("A_1", "A_0"), ("X", "W")])
        va, vb = rng.choice([0, 1]), rng.choice([0, 1])
        kind = rng.choice(["nand", "nor", "xor", "aoi"])
        if kind == "nand":
            e, pv, val = f"\\overline{{{a} {b}}}", f"\\overline{{{va} \\cdot {vb}}}", 1 - (va & vb)
        elif kind == "nor":
            e, pv, val = f"\\overline{{{a} + {b}}}", f"\\overline{{{va} + {vb}}}", 1 - (va | vb)
        elif kind == "xor":
            e, pv, val = f"{a} \\oplus {b}", f"{va} \\oplus {vb}", va ^ vb
        else:
            c, vc = "C", rng.choice([0, 1])
            e, pv, val = f"\\overline{{{a}{b} + C}}", f"\\overline{{{va} \\cdot {vb} + {vc}}}", 1 - ((va & vb) | vc)
        out = val if not slip else 1 - val
        data = f"{a} = {va}, {b} = {vb}" + (f", C = {vc}" if kind == "aoi" else "")
        style = rng.random()
        if style < 0.4:
            lines = [data, f"Y = {e} = {pv} = {out}"]
        elif style < 0.7:
            lines = [f"Y = {e}", data, f"Y = {pv} = {out}"]
        else:
            lines = [f"Y = {e}", data, f"Y = {out}"]
        return lines, ("wrong" if slip else None)


    def body_page(slip):
        """the body effect: the threshold at a source-body voltage"""
        vt0, gam, phi2, vsb = rng.choice([0.43, 0.4, 0.45, 0.5]), rng.choice([0.4, 0.3, 0.5]), rng.choice([0.6, 0.7, 0.8]), rng.choice([0.5, 1.0, 1.151, 0.9])
        vth = vt0 + gam * ((phi2 + vsb) ** 0.5 - phi2 ** 0.5)
        w = vth * (1.3 if slip else 1)
        phis = rng.choice([f"2|\\Phi_F|", f"2\\phi_F", f"2 \\mid \\Phi_{{F}} \\mid", f"|2\\phi_F|"])
        form = f"V_{{TH}} = V_{{T0}} + \\gamma\\left( \\sqrt{{{phis} + V_{{SB}}}} - \\sqrt{{{phis}}} \\right)"
        if rng.random() < 0.5:
            form = form.replace("\\left(", "(").replace("\\right)", ")")
        data = [f"V_{{T0}} = {vt0} V", f"\\gamma = {gam} V^{{1/2}}" if rng.random() < 0.3 else f"\\gamma = {gam}", f"{phis} = {phi2} V", f"V_{{SB}} = {vsb} V"]
        rng.shuffle(data)
        lines = [form] + data
        if rng.random() < 0.5:
            lines.append(f"V_{{TH}} = {vt0} + {gam}(\\sqrt{{{phi2} + {vsb}}} - \\sqrt{{{phi2}}}) = {w:.3f} V")
        else:
            lines.append(f"V_{{TH}} = {w:.3f} V")
        return lines, ("wrong" if slip else None)


    def square_page(slip):
        """the square law, a current at its gate voltage"""
        k, vgs, vt = rng.choice([30, 50, 100, 200]), rng.choice([1.2, 1.8, 1.5, 1.151]), rng.choice([0.4, 0.43, 0.537, 0.5])
        i = k / 2 * (vgs - vt) ** 2                                       # µA when k in µA/V²
        w = i * (1.3 if slip else 1)
        kn = rng.choice(["k_n", "k_n'", "\\beta_n", "k"])
        lines = [f"{kn} = {k} \\mu A/V^2" if rng.random() < 0.5 else f"{kn} = {k} \\frac{{\\mu A}}{{V^2}}",
                 f"I_{{D}} = \\frac{{{kn}}}{{2}}(V_{{GS}} - V_{{T}})^2"]
        if rng.random() < 0.5:
            lines.append(f"I_{{D}} = \\frac{{{k}}}{{2}}({vgs} - {vt})^{{2}} \\approx {g(w)} \\mu A")
        else:
            lines += [f"V_{{GS}} = {vgs} V", f"V_{{T}} = {vt} V", f"I_{{D}} = {g(w)} \\mu A"]
        return lines, ("wrong" if slip else None)


    def delay_page(slip):
        """an RC delay, the average of the two, energy and dynamic power"""
        r, c = rng.choice([5, 10, 2.5, 8]), rng.choice([2, 10, 50, 1])     # kΩ, fF
        tp = 0.69 * r * c                                                  # ps
        w = tp * (1.3 if slip else 1)
        lines = [f"R_{{eq}} = {r} k\\Omega", f"C_{{L}} = {c} fF"]
        style = rng.random()
        if style < 0.5:
            lines.append(f"t_{{p}} = 0.69 R_{{eq}} C_{{L}} = 0.69 \\cdot {r} k\\Omega \\cdot {c} fF = {g(w)} ps")
        else:
            lines.append(f"t_{{pHL}} = 0.69 \\cdot {r}k \\cdot {c}f = {g(w)} ps" if rng.random() < 0.0 else
                         f"t_{{pHL}} = 0.69 R_{{eq}} C_{{L}} = {g(w)} ps")
        if rng.random() < 0.5:
            vdd, f, a = rng.choice([1.8, 1.2, 1.0, 3.3]), rng.choice([100, 500, 1]), rng.choice([0.25, 0.1, 0.5])
            unit = "MHz" if f != 1 else "GHz"
            fhz = f * (1e6 if unit == "MHz" else 1e9)
            cl = rng.choice([1, 2, 0.5])                                   # pF
            p = a * cl * 1e-12 * vdd ** 2 * fhz
            e = cl * vdd ** 2                                              # pJ
            lines += [f"E = C_{{tot}} V_{{DD}}^2 = {cl} pF \\cdot {vdd}^2 = {g(e)} pJ"]
            pw = p * 1e6                                                   # µW
            lines += [f"P_{{dyn}} = \\alpha C_{{tot}} V_{{DD}}^2 f = {a} \\cdot {cl} pF \\cdot {vdd}^2 \\cdot {f} {unit} = {g(pw)} \\mu W"]
        return lines, ("wrong" if slip else None)


    def instant_page(slip):
        """values at an instant: a node discharging through a transistor"""
        vdd, vt = rng.choice([1.8, 1.2, 2.5]), rng.choice([0.649, 0.4, 0.5])
        v0 = vdd - vt
        w = v0 * (1.3 if slip else 1)
        y = rng.choice(["Y_{2}", "V_{out}", "V_{x}", "v_C"])
        t = rng.choice(["0^{-}", "0^-", " 0^{-} ", "0^{+}"])
        lr = rng.random() < 0.5
        arg = f"\\left( {t} \\right)" if lr else f"({t})"
        lines = [f"V_{{DD}} = {vdd} V", f"V_{{Tn}} = {vt} V", f"{y}{arg} = V_{{DD}} - V_{{Tn}} = {vdd} - {vt} = {g(w)} V"]
        if rng.random() < 0.5:
            lines.append(f"V_{{BS}} = V_{{DD}} - {y}{arg} = {vdd} - {g(v0)} = {g(vdd - v0)} V")
        return lines, ("wrong" if slip else None)


    def decoder_page(slip):
        """a 2-to-4 decoder's output at its inputs, indexed names every way"""
        a1, a0 = rng.choice([("A_1", "A_0"), ("A1", "A0"), ("A_{1}", "A_{0}"), ("S_1", "S_0")])
        v1, v0 = rng.choice([0, 1]), rng.choice([0, 1])
        k = rng.choice([0, 1, 2, 3])
        t1 = a1 if k & 2 else f"\\bar{{{a1}}}"
        t0 = a0 if k & 1 else f"\\bar{{{a0}}}"
        p1 = f"{v1}" if k & 2 else f"\\bar{{{v1}}}"
        p0 = f"{v0}" if k & 1 else f"\\bar{{{v0}}}"
        val = int(((v1 == 1) == bool(k & 2)) and ((v0 == 1) == bool(k & 1)))
        out = val if not slip else 1 - val
        d = idx("D", k)
        if rng.random() < 0.5:
            lines = [f"{a1} = {v1}, {a0} = {v0}", f"{d} = {t1} {t0} = {p1} \\cdot {p0} = {out}"]
        else:
            lines = [f"{a1} = {v1}, {a0} = {v0} \\Rightarrow {d} = {t1}{t0} = {out}"]
        return lines, ("wrong" if slip else None)


    def margins_page(slip):
        """logic levels and noise margins"""
        vdd = rng.choice([1.8, 1.2, 2.5, 3.3, 5])
        vih, vil = round(vdd * rng.choice([0.6, 0.65, 0.7]), 2), round(vdd * rng.choice([0.3, 0.35, 0.4]), 2)
        voh, vol = vdd, 0
        nmh, nml = voh - vih, vil - vol
        w = nmh * (1.3 if slip else 1)
        lines = [f"V_{{DD}} = {vdd} V", f"V_{{OH}} = V_{{DD}} = {vdd} V", f"V_{{OL}} = {vol} V", f"V_{{IH}} = {vih} V", f"V_{{IL}} = {vil} V",
                 f"NM_{{H}} = V_{{OH}} - V_{{IH}} = {vdd} - {vih} = {g(w)} V", f"NM_{{L}} = V_{{IL}} - V_{{OL}} = {g(nml)} V"]
        return lines, ("wrong" if slip else None)


    def threshold_page(slip):
        """a CMOS inverter's switching threshold"""
        vdd, vtn, vtp = rng.choice([1.8, 2.5, 1.2]), rng.choice([0.4, 0.43, 0.5]), rng.choice([0.4, 0.45, 0.5])
        r = rng.choice([1, 2, 4, 2.5])
        vm = (vdd - vtp + vtn * r ** 0.5) / (1 + r ** 0.5)
        w = vm * (1.3 if slip else 1)
        lines = [f"r = \\frac{{k_n}}{{k_p}} = {r}" if rng.random() < 0.5 else f"r = {r}",
                 f"V_{{M}} = \\frac{{V_{{DD}} - |V_{{Tp}}| + V_{{Tn}}\\sqrt{{r}}}}{{1 + \\sqrt{{r}}}}",
                 f"V_{{DD}} = {vdd} V", f"V_{{Tn}} = {vtn} V", f"|V_{{Tp}}| = {vtp} V", f"V_{{M}} = {w:.3f} V"]
        return lines, ("wrong" if slip else None)


    def elmore_page(slip):
        """an Elmore delay of an RC ladder, bare prefixes as the official solution writes them"""
        r1, r2, c1, c2 = rng.choice([1, 2, 5]), rng.choice([1, 2, 3]), rng.choice([10, 20, 5]), rng.choice([10, 30, 15])  # kΩ, fF
        tau = (r1 * c1 + (r1 + r2) * c2)                                  # ps
        w = tau * (1.3 if slip else 1)
        if rng.random() < 0.5:
            lines = [f"\\tau = R_1 C_1 + (R_1 + R_2) C_2 = {r1}k \\cdot {c1}f + ({r1}k + {r2}k) \\cdot {c2}f = {g(w)} ps"]
        else:
            lines = [f"R_1 = {r1} k\\Omega", f"R_2 = {r2} k\\Omega", f"C_1 = {c1} fF", f"C_2 = {c2} fF",
                     f"\\tau_{{D}} = R_1 C_1 + (R_1 + R_2) C_2 = {g(w)} ps"]
        if rng.random() < 0.5:
            lines.append(f"t_{{p}} = 0.69 \\tau = {g(0.69 * tau)} ps" if not slip else f"t_{{p}} = 0.69 \\cdot {g(tau)} = {g(0.69 * tau)} ps")
        return lines, ("wrong" if slip else None)


    def effort_page(slip):
        """logical effort: a stage's delay in units of tau"""
        gg, h, p = rng.choice([1, 4 / 3, 5 / 3, 2]), rng.choice([1, 2, 4, 3]), rng.choice([1, 2, 3])
        gtxt = {1: "1", 4 / 3: "\\frac{4}{3}", 5 / 3: "\\frac{5}{3}", 2: "2"}[gg]
        d = gg * h + p
        w = d * (1.3 if slip else 1)
        lines = [f"g = {gtxt}", f"h = {h}", f"p = {p}", f"d = gh + p = {gtxt} \\cdot {h} + {p} = {g(w)}"]
        return lines, ("wrong" if slip else None)


    def path_page(slip):
        """logical effort along a path: G, H, F = GBH, the optimal stage effort F^{1/N}, the delay; a stage's input capacitance"""
        gs = rng.sample([1, 4 / 3, 5 / 3, 2], rng.choice([2, 3]))
        gt = {1: "1", 4 / 3: "\\frac{4}{3}", 5 / 3: "\\frac{5}{3}", 2: "2"}
        n = len(gs)
        G = 1.0
        for x in gs:
            G *= x
        cin, cout = rng.choice([2, 4, 5]), rng.choice([64, 80, 100, 128])
        H = cout / cin
        F = G * H
        f = F ** (1 / n)
        P = rng.choice([3, 4, 5, 6])
        D = n * f + P
        tp0 = rng.choice([6.9, 5, 10])
        k = rng.choice([0, 1, 2, 3]) if slip else None
        s = lambda x: x * 1.3
        dot = " \\cdot "
        lines = [f"G = {dot.join(gt[x] for x in gs)} = {g(G)}",
                 f"H = \\frac{{C_{{out}}}}{{C_{{in}}}} = \\frac{{{cout}}}{{{cin}}} = {g(s(H) if k == 0 else H)}",
                 f"F = GH = {g(G)} \\cdot {g(H)} = {g(s(F) if k == 1 else F)}",
                 rng.choice([f"\\hat{{f}} = F^{{1/{n}}} = {g(F)}^{{1/{n}}} = {g(s(f) if k == 2 else f)}",
                             f"\\hat{{f}} = \\sqrt[{n}]{{{g(F)}}} = {g(s(f) if k == 2 else f)}"]),
                 f"D = N\\hat{{f}} + P = {n} \\cdot {g(f)} + {P} = {g(s(D) if k == 3 else D)}"]
        if rng.random() < 0.5:
            lines.append(f"t_{{pd}} = t_{{p0}} D = {tp0} ps \\cdot {g(D)} = {g(tp0 * D)} ps")
        return lines, ("wrong" if slip else None)


    KINDS = [states_page, adder_page, mux_page, eval_page, body_page, square_page, delay_page, instant_page,
             decoder_page, margins_page, threshold_page, elmore_page, effort_page, path_page]

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    bad = missed = 0
    for m in range(n):
        kind = KINDS[m % len(KINDS)]
        state = rng.getstate()
        right, _ = kind(False)
        rng.setstate(state)
        wrong, _ = kind(True)
        vr, vw = verdicts(right), verdicts(wrong)
        if any(v[0].startswith("wrong") and v[1] for v in vr) or (kind is not states_page and not any(v[0] == "right" for v in vr)):
            bad += 1
            print("WRONG?", right, vr)
        if not any(v[0].startswith("wrong") for v in vw):
            missed += 1
            print("MISSED", wrong, vw)
    return n, bad, n, missed


def instants_random(n=60, seed=107):
    """switched first-order circuits with values at instants, as students write them: 'v_C(0^-) = V_1 \\frac{R_2}{R_1 + R_2}
    = 4 V', 'v_C(0^+) = v_C(0^-) = 4 V', 'i_C(0^+) = \\frac{V_s - v_C(0^+)}{R} = 4 mA', 'v_C(\\infty) = 12 V', '\\tau = RC',
    'v_C(1 ms) = 12 + (4 - 12) e^{-0.5} = 7.15 V'; an inductor's the same way. A right page never a sure wrong nor silent,
    a slip (one result x1.3) caught"""
    import math
    import random
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)
    OM = "k\\Omega"

    def g(x):
        return f"{x:.3g}"


    def inst(name, t):
        t = {"0-": rng.choice(["0^-", "0^{-}", "0^{-} "]), "0+": rng.choice(["0^+", "0^{+}"]), "inf": "\\infty"}[t]
        return f"{name}\\left( {t} \\right)" if rng.random() < 0.25 else f"{name}({t})"


    def rc_page(slip_at):
        v1, vs = rng.choice([6, 9, 12]), rng.choice([10, 12, 15, 20])
        r1, r2, r = rng.choice([1, 2, 3]), rng.choice([1, 2, 4]), rng.choice([1, 2, 5])        # kΩ
        c = rng.choice([1, 2, 0.5])                                                             # µF
        v0 = v1 * r2 / (r1 + r2)
        i0 = (vs - v0) / r                                                                      # mA
        tau = r * c                                                                             # ms
        t1 = rng.choice([0.5, 1, 2]) * tau
        vt = vs + (v0 - vs) * math.exp(-t1 / tau)
        s = lambda k, x: x * (1.3 if slip_at == k else 1)
        vc = rng.choice(["v_C", "V_C", "v_c"])
        lines = [f"V_1 = {v1} V", f"R_1 = {r1} {OM}", f"R_2 = {r2} {OM}", f"V_s = {vs} V", f"R = {r} {OM}", f"C = {c} \\mu F"]
        lines.append(f"{inst(vc, '0-')} = V_1 \\frac{{R_2}}{{R_1 + R_2}} = {v1} \\cdot \\frac{{{r2}}}{{{r1} + {r2}}} = {g(s(0, v0))} V")
        lines.append(f"{inst(vc, '0+')} = {inst(vc, '0-')} = {g(v0)} V")
        lines.append(f"{inst('i_C', '0+')} = \\frac{{V_s - {inst(vc, '0+')}}}{{R}} = \\frac{{{vs} - {g(v0)}}}{{{r}k}} = {g(s(1, i0))} mA")
        lines.append(f"{inst(vc, 'inf')} = V_s = {vs} V")
        lines.append(f"\\tau = R C = {r} {OM} \\cdot {c} \\mu F = {g(s(2, tau))} ms")
        if rng.random() < 0.5:
            lines.append(f"{vc}(t) = {inst(vc, 'inf')} + ({inst(vc, '0+')} - {inst(vc, 'inf')}) e^{{-t/\\tau}}")
        lines.append(f"{vc}({g(t1)} ms) = {vs} + ({g(v0)} - {vs}) e^{{-{g(t1 / tau)}}} = {g(s(3, vt))} V")
        return lines


    def rl_page(slip_at):
        v, r1, r2 = rng.choice([10, 12, 20]), rng.choice([1, 2, 4]), rng.choice([1, 2, 3])     # V, kΩ
        l = rng.choice([1, 2, 10])                                                             # mH
        i0 = v / r1                                                                            # mA
        vl0 = -i0 * r2                                                                         # V (the inductor's current through R2)
        tau = l / r2                                                                           # µs (mH / kΩ)
        s = lambda k, x: x * (1.3 if slip_at == k else 1)
        il = rng.choice(["i_L", "I_L"])
        lines = [f"V = {v} V", f"R_1 = {r1} {OM}", f"R_2 = {r2} {OM}", f"L = {l} mH"]
        lines.append(f"{inst(il, '0-')} = \\frac{{V}}{{R_1}} = \\frac{{{v}}}{{{r1}k}} = {g(s(0, i0))} mA")
        lines.append(f"{inst(il, '0+')} = {inst(il, '0-')} = {g(i0)} mA")
        lines.append(f"{inst('v_L', '0+')} = -{inst(il, '0+')} R_2 = -{g(i0)} mA \\cdot {r2} {OM} = {g(s(1, vl0))} V")
        lines.append(f"{inst(il, 'inf')} = 0")
        lines.append(f"\\tau = \\frac{{L}}{{R_2}} = \\frac{{{l} mH}}{{{r2} {OM}}} = {g(s(2, tau))} \\mu s")
        return lines

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    bad = missed = 0
    for m in range(n):
        kind = rc_page if m % 2 == 0 else rl_page
        state = rng.getstate()
        right = kind(None)
        rng.setstate(state)
        wrong = kind(rng.choice([0, 1, 2, 3] if kind is rc_page else [0, 1, 2]))
        vr, vw = verdicts(right), verdicts(wrong)
        if any(v[0].startswith("wrong") and v[1] for v in vr) or not any(v[0] == "right" for v in vr):
            bad += 1
            print("WRONG?", right, vr)
        if not any(v[0].startswith("wrong") for v in vw):
            missed += 1
            print("MISSED", wrong, vw)
    return n, bad, n, missed


def line_breaks_random(n=80, seed=109):
    """two lines of work in one reading, a line break between them ('V = 12 V \\\\ R = 4 k\\Omega', as a formula reader
    writes a crop of two lines; a chain going on, '\\\\ = 3 mA'): the same verdicts as the lines apart - right or with a
    slip (x1.3)"""
    import random
    from collections import Counter
    from grader.selfcheck import check_statements, statements
    rng = random.Random(seed)

    def page(f):
        V, R, vs = rng.choice([5, 9, 12, 10]), rng.choice([1, 2, 4, 5]), rng.choice([10, 12, 15])
        I = V / R
        v0 = rng.choice([2, 4, 6])
        kind = rng.randrange(4)
        if kind == 0:
            return [f"V = {V} V", f"R = {R} k\\Omega", f"I = \\frac{{V}}{{R}} = {I * f:.3g} mA"]
        if kind == 1:
            return [f"v_C(0^-) = {v0} V", f"v_C(0^+) = v_C(0^-) = {v0} V",
                    f"i_C(0^+) = \\frac{{{vs} - v_C(0^+)}}{{{R}k}} = {(vs - v0) / R * f:.3g} mA"]
        if kind == 2:
            return [f"I = \\frac{{V}}{{R}}", f"= \\frac{{{V}}}{{{R}k}}", f"= {I * f:.3g} mA"]
        return [f"I = {I:.3g} mA", f"R = {R} k\\Omega", f"P = I^2 R = {I * I * R * f:.3g} mW"]

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return Counter((c.verdict, c.sure) for c in check_statements(st))
    bad = 0
    for m in range(n):
        lines = page(1.0 if m % 2 == 0 else 1.3)
        k = rng.randrange(len(lines) - 1)
        merged = lines[:k] + [lines[k] + " \\\\ " + lines[k + 1]] + lines[k + 2:]
        a, b = verdicts(lines), verdicts(merged)
        if a != b:
            bad += 1
            print("DIFFERENT", lines, dict(a), "merged:", dict(b))
    return n, bad


def angles_db_random(n=80, seed=11):
    """angles and decibels as students write them: a trig function of degrees with the mark left out ('\\cos(60) =
    0.5'), an arctan's value in degrees ('= 53.13'), a ratio in dB ('A_v = 100 = 40 dB'), a phasor as e^{j45} (degrees)
    or \\angle 0.927 (radians), a power factor's angle given bare - none correct judged wrong, every wrong one caught"""
    import random
    rng = random.Random(seed)
    ratio_angles = [(p, q) for p in range(1, 10) for q in range(1, 10) if math.gcd(p, q) == 1]
    fns = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
    bad = missed = n_wrong = 0
    for _ in range(n):
        kind = rng.choice(["trig", "trig", "atan", "db", "exp", "rad", "pf"])
        if kind == "trig":
            fn = rng.choice(list(fns))
            if rng.random() < 0.5:
                a = 15 * rng.randint(1, 23)
                at = str(a)
            else:
                p, q = rng.choice(ratio_angles)
                a = math.degrees(math.atan2(p, q)) + rng.choice([0, 90, 180])
                at = f"{a:.2f}"
            v = fns[fn](math.radians(float(at)))
            if abs(v) < 0.05 or abs(v) > 50:
                continue
            right = [f"\\{fn}({at}) = {v:.4f}"]
            wrong = [f"\\{fn}({at}) = {v * 1.25 + 0.1:.4f}"]
        elif kind == "atan":
            p, q = rng.choice(ratio_angles)
            d = math.degrees(math.atan2(p, q))
            right = [f"\\varphi = \\arctan(\\frac{{{p}}}{{{q}}}) = {d:.2f}"]
            wrong = [f"\\varphi = \\arctan(\\frac{{{p}}}{{{q}}}) = {d + 7:.2f}"]
        elif kind == "db":
            r = rng.choice([2, 5, 10, 20, 50, 100, 200, 1000, 0.5, 0.1, 0.707, 0.01, 3, 7])
            g = 20 * math.log10(r)
            dec = rng.choice([0, 1, 2])
            right = [f"A_v = {r} = {g:.{dec}f} dB"]
            wrong = [f"A_v = {r} = {g + 6:.{dec}f} dB"]
        elif kind in ("exp", "rad"):
            p, q = rng.choice(ratio_angles)
            z = complex(q, p) * rng.choice([1, 2, 5])
            m, ang = abs(z), math.degrees(cmath.phase(z))
            form = f"{m:.3f} e^{{j {ang:.2f}}}" if kind == "exp" else f"{m:.3f} \\angle {math.radians(ang):.4f}"
            wrong_form = (f"{m:.3f} e^{{j {ang + 20:.2f}}}" if kind == "exp" else f"{m * 2:.3f} \\angle {math.radians(ang):.4f}")
            right = [f"Z = {int(z.real)} + j{int(z.imag)} = {form}"]
            wrong = [f"Z = {int(z.real)} + j{int(z.imag)} = {wrong_form}"]
        else:
            a = rng.choice([30, 45, 60, 36.87, 53.13, 25.84])
            vv, ii = rng.choice([10, 100, 230]), rng.choice([2, 5, 10])
            pw = vv * ii * math.cos(math.radians(a))
            right = ["P = V I \\cos(\\varphi)", f"V = {vv}, I = {ii}, \\varphi = {a}", f"P = {pw:.1f}"]
            wrong = ["P = V I \\cos(\\varphi)", f"V = {vv}, I = {ii}, \\varphi = {a}", f"P = {pw * 1.3:.1f}"]
        vr = _verdicts(right)
        if _judged_wrong(vr) or not vr:
            bad += 1
            print("WRONG?" if vr else "NO VERDICT?", right, vr)
        n_wrong += 1
        if not _judged_wrong(_verdicts(wrong)):
            missed += 1
            print("MISSED", wrong, _verdicts(wrong))
    return n_wrong, bad, n_wrong, missed


_GARBAGE_TOK = ["x", "y", "t", "s", "f", "V", "I", "R", "k", "m", "A", "B", "C", "a", "b", "c", "n", "1", "2", "3", "0", "5", "10", "0.5", "1k", "2m",
       "+", "-", "=", "=", "=", "\\cdot", "\\times", "/", "^", "_", "(", ")", "{", "}", "[", "]", "\\{", "\\}", "|", "'", "''", ",", ";",
       "\\frac", "\\sqrt", "\\int", "\\int_0^1", "\\sum_{k=1}^{n}", "\\lim_{x \\to 0}", "\\frac{d}{dx}", "\\frac{d^2}{dx^2}", "\\frac{dy}{dx}",
       "\\partial", "\\mathcal{L}", "\\mathcal{L}^{-1}", "\\pm", "\\left[", "\\right]", "\\left(", "\\right)", "\\left.", "\\right|",
       "\\sin", "\\cos", "\\ln", "\\log", "e", "\\pi", "\\infty", "\\angle", "^\\circ", "j", "\\Omega", "mA", "V", "kHz", "\\frac{m}{s}",
       "\\text{ ולכן }", "\\text{or}", "\\rightarrow", "\\Rightarrow", "\\vert", "x_{1,2}", "f(x)", "f'(x)", "y'", "u(t)", "\\delta(t)",
       "\\%", "\\approx", "·", "−", "÷", "\\|", "dB", "min", "h", "s^2", "\\overline{3}",
       "\\begin{cases}", "\\end{cases}", "\\\\", "&", "\\begin{vmatrix}", "\\end{vmatrix}", "\\begin{array}{cc}", "\\end{array}",
       "\\begin{pmatrix}", "\\end{pmatrix}", "\\begin{bmatrix}", "\\end{bmatrix}", "^{-1}", "^T", "\\det",
       "V(s)", "v(t)", "I(s)", "i(t)", "H(s)", "I_1", "I_2", "2k", "4k", "x_1", "\\mathcal{L}^{-1}\\{", "\\frac{1}{s+1}", "kg", "\\frac{m}{s^2}", "kV",
       "3 + j4", "2\\angle -53.13^\\circ", "(1011)_2", "(FF)_{16}", "1011_{(2)}", "H(j\\omega)", "H(j2)", "|H(j\\omega)|", "20\\log",
       "v_C(1 ms)", "i(2 \\mu s)", "\\Re\\{Z\\}", "I^*", "\\overline{I}", "J", "Z", "\\omega_d t", "r_e", "(-1)^n", "\\cos(n\\pi)",
       "VA", "var", "\\Omega", "e^{j\\omega t}", "\\sin(n x)", "A\\cos(\\omega t)", "\\frac{d}{dt}"]


def garbage_random(n=400, seed=7, limit=20):
    """random LaTeX-ish lines (every notation the checks read, in any order) through the derivation checks: no exception
    may escape and no page may take over `limit` seconds -> (pages, errors, slow ones)"""
    import random
    import signal
    import traceback
    from grader.selfcheck import check_statements, statements

    class Slow(Exception):
        pass

    def alarm(*_):
        raise Slow()
    old = signal.signal(signal.SIGALRM, alarm)
    rng = random.Random(seed)
    errors = slow = 0
    try:
        for _ in range(n):
            lines = [" ".join(rng.choice(_GARBAGE_TOK) for _ in range(rng.randint(3, 14))) for _ in range(rng.randint(1, 3))]
            signal.alarm(limit)
            try:
                st = []
                for k, t in enumerate(lines):
                    st += statements(t, line=k, piece=k)
                check_statements(st)
            except Slow:
                slow += 1
                print("SLOW", lines)
            except Exception:                      # noqa: BLE001
                errors += 1
                print("ERROR", lines)
                traceback.print_exc(limit=3)
            finally:
                signal.alarm(0)
    finally:
        signal.signal(signal.SIGALRM, old)
    return n, errors, slow


def course_solutions():
    """the model solutions of the course's exams (data/course_elec101/E*.json - typed as plain text: '10⁻⁸', '√2', '½',
    'R2 ∥ R3', '12 / 2k = 6 mA', '20 ms', 'kWh'): none judged wrong, each with a verdict where it has math; its last
    number made 1.5 times bigger: caught -> (solutions, judged wrong, without a verdict, wrong ones, missed)"""
    import glob
    import json
    bad = none = missed = n_wrong = 0
    files = sorted(glob.glob(os.path.join(HERE, "data", "course_elec101", "E*.json")))
    sols = [q["solution"] for f in files for q in json.load(open(f, encoding="utf-8"))["questions"]
            if q.get("solution") and q.get("kind") != "choice" and re.search(r"\d\s*[=/·*+-]|=\s*\d", q["solution"])]
    for sol in sols:
        vs = _verdicts([sol])
        if _judged_wrong(vs) or "wrong unit" in vs:
            bad += 1
            print("WRONG?", sol, vs)
        if not vs:
            none += 1
            print("NO VERDICT", sol)
            continue
        m = list(re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?=[^\d=]*$)", sol))
        if not m:
            continue
        x = m[-1]
        wrong = sol[:x.start()] + f"{float(x.group(1)) * 1.5:g}" + sol[x.end():]
        n_wrong += 1
        if not _judged_wrong(_verdicts([wrong])):
            missed += 1
            print("MISSED", wrong, _verdicts([wrong]))
    return len(sols), bad, none, n_wrong, missed


def derivations_spaced():
    """the correct derivations again as the reader spaces them: none may be judged wrong (a verdict that differs from
    the clean one is printed, not counted)"""
    bad = 0
    for case in RIGHT_DERIVATIONS:
        clean, sp_case = _verdicts(case), [spaced(t) for t in case]
        got = _verdicts(sp_case)
        if _judged_wrong(got):
            bad += 1
            print("WRONG?", sp_case, got)
        elif got != clean:
            print("differs", sp_case, clean, "->", got)
    return len(RIGHT_DERIVATIONS), bad


if __name__ == "__main__":
    fails = 0
    for name, fn in (("circuits, DC", circuits_dc), ("circuits, AC", circuits_ac), ("circuits, a switching", circuits_switching)):
        total, bad = fn()
        print(f"{name}: {total} correct answers, judged wrong: {bad}")
        fails += bad
    total, bad = circuits_random()
    print(f"circuits, random networks with tricky names: {total} correct answers, judged wrong for sure: {bad}")
    fails += bad
    total, bad, n_wrong, missed = circuits_random_labelled()
    print(f"circuits, random networks labelled out of order, currents drawn: {total} correct answers, judged wrong for sure: "
          f"{bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = circuits_random_ac()
    print(f"circuits, random AC ladders: {total} correct answers, judged wrong for sure: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad = circuits_forms()
    print(f"circuits, magnitudes / H / dB / a marked voltage / loop and node equations: {total} answers, misjudged: {bad}")
    fails += bad
    n_right, bad, n_wrong, missed = derivations()
    print(f"derivations: {n_right} correct, judged wrong: {bad}; {n_wrong} wrong ones, missed: {missed}")
    total, bad2 = derivations_spaced()
    print(f"derivations as the formula reader spaces them: {total} correct, judged wrong: {bad2}")
    fails += bad + missed + bad2
    total, bad, none, n_wrong, missed = course_solutions()
    print(f"the course's model solutions (plain text): {total}, judged wrong: {bad}, without a verdict: {none}; "
          f"{n_wrong} made wrong, missed: {missed}")
    fails += bad + missed
    total, errors, slow = garbage_random()
    print(f"random garbage through the derivation checks: {total} pages, errors: {errors}, over 20 s: {slow}")
    fails += errors + slow
    total, bad, n_wrong, missed = units_random()
    print(f"derivations with units in every position, random: {total} correct, judged wrong: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = divisor_units_random()
    print(f"a unit after a divisor's number, random: {total} correct, judged wrong: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = systems_random()
    print(f"mesh equations in kΩ, in ohms or with units, random: {total} correct, judged wrong for sure: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = ee_random()
    print(f"EE computations across templates and renderings, random: {total} correct, judged wrong for sure: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = formulas_random()
    print(f"a formula, its data and the answer in units every way, random: {total} correct, judged wrong for sure: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = phasors_random()
    print(f"derivations with phasors, random: {total} correct, judged wrong: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = db_random()
    print(f"gains in dB (used as ratios, beside ratios, added, dBm), random: {total} correct, judged wrong for sure: {bad}; "
          f"{n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = two_formulas_random()
    print(f"two formulas, the second defining a letter of the first, data below both, random: {total} correct, judged wrong "
          f"for sure or no 'right': {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = formula_flow_random()
    print(f"a formula with a value worked out on its own line (a combination, Ohm's law), random: {total} correct, judged wrong "
          f"for sure or no verdict: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = flow_random()
    print(f"work over several lines, values flowing on by name (prefixed, rounded on the way), random: {total} pages, judged "
          f"wrong for sure where right (or a second line wrong): {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = trig_random()
    print(f"angles in power and phasor work (a formula with prefixed data or a chain; degrees marked, bare or radians), random: "
          f"{total} correct, judged wrong for sure or no verdict: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = semiconductor_random()
    print(f"semiconductor physics in centimetres (cm^{{-3}}, cm^2/s, \\Omega cm, µm), random: {total} correct, judged wrong for "
          f"sure or no verdict: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = phasor_units_random()
    print(f"phasors with their units every way (unbracketed, kΩ and mA, VA), random: {total} correct, judged wrong for sure or "
          f"no verdict: {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = exam_random()
    print(f"a question answered in sub-questions marked every way, values carried from part to part, random: {total} answers, "
          f"a sure wrong not the slip's (or no 'right'): {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = redo_random()
    print(f"a line done again below itself, put right (a computation, an answer), and an equation's roots, random: {total} "
          f"pages, a sure wrong for a first try or no sure 'right': {bad}; {n_wrong} with roots, a first try or a wrong root "
          f"missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = electronics_random()
    print(f"electronics answers (BJT bias, op-amps, diodes, Thevenin with parallel), rounded on the way, random: {total} answers, "
          f"a sure wrong not the slip's (or no 'right'): {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = boolean_random()
    print(f"Boolean algebra (identities and the classic wrong ones, complements every way, a function's name, over lines), "
          f"random: {total} pages, a right one judged wrong or without a 'right': {bad}; {n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = bases_random()
    print(f"numbers in bases 2, 8, 16 written every way ('1011_B', '0x1F', '1F_{{16}}', '(1F)_H'), random: {total} correct, judged "
          f"otherwise: {bad}; {n_wrong} one off, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = response_random()
    print(f"a frequency response's size and angle at the frequencies written ('|H(j10)| = 0.707', '\\angle H(j10) = -45°'), "
          f"random: {total} pages, a right one judged wrong or without a 'right': {bad}; slips (x1.3, +15°) missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = algebra_random()
    print(f"expansions and factorings along a chain, right and with a slip, and equations along a chain, random: {total} pages, "
          f"a right one or an equation judged wrong: {bad}; slips missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = juxtaposed_random()
    print(f"numbers, fractions, roots, powers, brackets side by side ('\\frac{{2}}{{3}} 12', '\\frac{{6}}{{3}}\\frac{{1}}{{2}}'), random: "
          f"{total} products, read otherwise: {bad}")
    fails += bad + missed
    total, bad, n_wrong, missed = hebrew_random()
    print(f"pages with Hebrew words around the math ('נתון:', 'לכן', 'תשובה:', a period), random: {total} pages, a right one judged "
          f"wrong for sure or without a 'right': {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad = line_breaks_random()
    print(f"two lines of work in one reading, a line break between them, random: {total} pages, verdicts unlike the lines "
          f"apart: {bad}")
    fails += bad
    total, bad, n_wrong, missed = instants_random()
    print(f"switched circuits with values at instants ('v_C(0^-)', 'v_C(0^+) = v_C(0^-)', 'i_C(0^+)', 'v_C(\\infty)', then v_C at a "
          f"time), random: {total} pages, a right one judged wrong for sure or without a 'right': {bad}; {n_wrong} with a slip, "
          f"missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = digital_random()
    print(f"digital-circuits pages (nodes' states, indexed names, nodes defined, inputs' values put in, a multiplexer, a decoder; "
          f"CMOS: the body effect, the square law, delays, noise margins, power), random: {total} pages, a right one judged wrong "
          f"for sure or without a 'right': {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = typed_random()
    print(f"EE computations typed as plain text ('1/(2*pi*1k*1u)', 'sqrt(300^2 + 400^2)', '10m/10 = 1ms'), random: {total} correct, "
          f"judged wrong for sure or without a 'right': {bad}; {n_wrong} wrong ones (x1.3), missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = slash_random()
    print(f"formulas with slashes inline ('1/2 C V^2', 'R_2/(R_1+R_2) V_{{in}}', '1/2\\pi\\sqrt{{LC}}'), random: {total} pages, "
          f"a right one judged wrong for sure or without a 'right': {bad}; x1.3 judged right for sure or not wrong: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = page_random()
    print(f"pages of two or three sub-problems reusing letter names (unnumbered or numbered, a part done again), random: "
          f"{total} pages, a sure wrong not the slip's: {bad}; {n_wrong} with a slip, missed: {missed}")
    fails += bad + missed
    total, bad, n_wrong, missed = angles_db_random()
    print(f"angles in degrees unmarked, dB, phasor angles, random: {total} correct, judged wrong or no verdict: {bad}; "
          f"{n_wrong} wrong ones, missed: {missed}")
    fails += bad + missed
    from replay_formula_eval import replay                 # the 70 real handwritten formulas, from their saved readings
    summary, wrong = replay()
    auto = sum(x["auto"] for x in summary.values())
    print(f"real handwritten formulas (saved readings): {auto} of {sum(x['n'] for x in summary.values())} grades automatic, "
          f"wrong: {len(wrong)}")
    for w in wrong:
        print("  WRONG", w)
    fails += len(wrong)
    sys.exit(1 if fails else 0)
