"""Circuits read from a drawing: the netlist (components, the nodes they join, the currents the student
named), how well a reading matches a hand transcription, and what the circuit itself says - Kirchhoff's
current law at every node the student labelled currents at, and the node voltages / branch currents
solved symbolically (modified nodal analysis, SymPy). With these the grader can check a student's
'I1 + I3 = I2' or 'V_A = ...' against the circuit he drew, without a key.

    net = Netlist.from_json(reading)                 # e.g. recog/circuit_vlm.py's JSON
    net.kcl()                                        # {'A': 'I1 + I3 = I2'}
    net.solve()                                      # {'V_A': ..., 'I1': ..., ...} in E1, R1, ... (or numbers)
    compare(net, Netlist.from_json(truth))           # components / topology / labels / polarity / currents
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field

import sympy as sp

TWO_TERMINAL = {"resistor", "voltage_source", "current_source", "capacitor", "inductor", "diode", "zener", "wire",
                "dependent_voltage_source", "dependent_current_source", "switch", "meter", "lamp"}
_TYPES = {"r": "resistor", "res": "resistor", "resistor": "resistor", "v": "voltage_source", "vs": "voltage_source",
          "voltage_source": "voltage_source", "voltage source": "voltage_source", "battery": "voltage_source",
          "dc_source": "voltage_source", "source": "voltage_source", "i": "current_source",
          "current_source": "current_source", "current source": "current_source", "c": "capacitor",
          "capacitor": "capacitor", "l": "inductor", "inductor": "inductor", "d": "diode", "diode": "diode", "zener": "zener"}


@dataclass
class Component:
    id: str
    type: str
    nodes: list
    label: str | None = None
    value: str | None = None
    plus: str | None = None                  # sources: the node at the '+' (current source: the arrow's head)
    ac: bool = False                         # an AC source (no polarity drawn)
    conf: float = 1.0                        # how sure the reading of its name / value is (a drawing's labels)
    roles: dict = field(default_factory=dict)   # a multi-terminal part's legs: {'In+': node, ...}, {'Base': node, ...}
    params: dict = field(default_factory=dict)  # a transistor's parameters as written by it: {'V_t': '1V', 'K': '0.5mA/V^2'}
    state: str | None = None                 # a switch: the state it is drawn in, i.e. before t = 0 - 'open' / 'closed',
    #                                          a two-way switch (nodes: common, throw 1, throw 2) '1' / '2'


@dataclass
class Current:
    label: str
    through: str | None                      # component id
    frm: str | None                          # node the arrow comes from
    to: str | None                           # node it points to


@dataclass
class Voltage:
    label: str                               # 'V_x', 'v_o', 'V_{R1}': a voltage marked on the drawing with '+' and '-'
    plus: str | None                         # node at its '+'
    minus: str | None                        # node at its '-'
    across: str | None = None                # the component it is marked across, when both marks sit by its ends
    conf: float = 1.0                        # how sure the reading of its name and marks is


@dataclass
class Netlist:
    components: list = field(default_factory=list)
    currents: list = field(default_factory=list)
    ground: str | None = None
    node_labels: dict = field(default_factory=dict)     # node -> letter written there ('A')
    ports: list = field(default_factory=list)           # nodes drawn as open terminals (a port: a, b)
    terminals: list = field(default_factory=list)       # every open terminal drawn: (its node, (x, y)) - two-ports
    notes: list = field(default_factory=list)           # what the reading itself found doubtful ('ink like a part's
    #                                                     symbol in a wire at (x, y): a part not detected?')
    voltages: list = field(default_factory=list)        # voltages marked with '+' / '-' and a name (Voltage)
    ports_marked: bool = False                          # the port's '+' / '-' drawn by its terminals: ports[0] is '+'

    @staticmethod
    def from_json(d) -> "Netlist":
        """a dict, or text holding one (a model's answer with prose around its JSON)"""
        if isinstance(d, str):
            d = parse_json(d)
        comps = []
        for i, c in enumerate(d.get("components", []) or []):
            t = str(c.get("type", "other")).strip().lower()
            nodes = [str(n) for n in (c.get("nodes") or []) if n is not None]
            comps.append(Component(str(c.get("id") or c.get("label") or f"X{i + 1}"), _TYPES.get(t, t), nodes,
                                   c.get("label"), None if c.get("value") in (None, "") else str(c.get("value")),
                                   None if c.get("plus") is None else str(c.get("plus")), bool(c.get("ac", False))))
        curs = [Current(str(c.get("label")), c.get("through"), None if c.get("from") is None else str(c.get("from")),
                        None if c.get("to") is None else str(c.get("to")))
                for c in (d.get("currents", []) or []) if c.get("label")]
        g = d.get("ground")
        labels = {str(k): str(v) for k, v in (d.get("node_labels") or {}).items() if v}
        vols = [Voltage(str(v.get("label")), None if v.get("plus") is None else str(v.get("plus")),
                        None if v.get("minus") is None else str(v.get("minus")), v.get("across"))
                for v in (d.get("voltages", []) or []) if v.get("label")]
        return Netlist(comps, curs, None if g in (None, "") else str(g), labels, voltages=vols)

    # -- structure --------------------------------------------------------------------------------
    def nodes(self) -> list:
        return sorted({n for c in self.components for n in c.nodes})

    def problems(self) -> list[str]:
        """what cannot be right in a reading (each needs a human, or a second reading)"""
        out = []
        for c in self.components:
            if c.type in TWO_TERMINAL and len(c.nodes) != 2 and not (c.type == "switch" and len(c.nodes) == 3):
                out.append(f"{c.id}: {len(c.nodes)} ends")
            elif c.type == "transformer" and c.roles and (c.roles.get("P+") == c.roles.get("P-") or
                                                          c.roles.get("S+") == c.roles.get("S-")):
                out.append(f"{c.id}: a coil's two ends on the same node (shorted)")
            elif c.type != "transformer" and len(set(c.nodes)) < len(c.nodes):
                out.append(f"{c.id}: both ends on the same node (shorted)")
            if c.type in ("voltage_source", "current_source", "dependent_voltage_source", "dependent_current_source") \
                    and not c.ac and c.plus not in c.nodes:
                out.append(f"{c.id}: polarity not read")
            if c.type.startswith("dependent") and not _DEP.match(c.value or ""):
                out.append(f"{c.id}: what this dependent source follows was not read")
            if c.type == "transformer" and c.state != "dots read":
                out.append(f"{c.id}: its dots were not read (taken at the top of both coils)")
        ends = {}
        for c in self.components:
            for n in c.nodes:
                ends[n] = ends.get(n, 0) + 1
        out += [f"node {n}: only one component touches it (a loose wire?)" for n, k in ends.items()
                if k == 1 and n not in self.ports and n not in self.node_labels]   # a terminal / a named input is meant to be
        ids = {c.id for c in self.components}
        for cur in self.currents:
            comp = next((c for c in self.components if c.id == cur.through), None)
            if cur.through not in ids:
                out.append(f"{cur.label}: through an unknown component {cur.through!r}")
            elif {cur.frm, cur.to} != set(comp.nodes):
                out.append(f"{cur.label}: its direction does not run along {cur.through}")
        if self.ground is not None and self.ground not in ends:
            out.append(f"ground {self.ground!r} is not a node of any component")
        for vol in self.voltages:
            if vol.plus is None or vol.minus is None or vol.plus == vol.minus:
                out.append(f"{vol.label}: its '+' and '-' marks are not on two different nodes")
        return out + list(self.notes)

    # -- what the circuit says -------------------------------------------------------------------
    def kcl(self) -> dict:
        """{node: 'I1 + I3 = I2'} at every node where each component touching it carries a named current"""
        out = {}
        for n in self.nodes():
            if n == self.ground:
                continue
            touching = [c for c in self.components if n in c.nodes]
            named = {cur.through: cur for cur in self.currents}
            if not touching or any(c.id not in named for c in touching):
                continue
            inn = [named[c.id].label for c in touching if named[c.id].to == n]
            out_ = [named[c.id].label for c in touching if named[c.id].frm == n]
            if inn and out_:
                out[self.node_labels.get(n, n)] = f"{' + '.join(inn)} = {' + '.join(out_)}"
        return out

    def islands(self) -> list:
        """groups of nodes joined to each other but not to ground - the far side of a transformer (its coils join
        nothing between them), a drawing in two pieces: [(nodes, the one taken as its 0 V)] - its voltages measured
        from there"""
        adj = {}

        def join(a, b):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        for c in self.components:
            if c.type == "transformer" and {"P+", "P-", "S+", "S-"} <= set(c.roles):
                join(c.roles["P+"], c.roles["P-"])
                join(c.roles["S+"], c.roles["S-"])
                continue
            ns = list(c.roles.values()) if c.type == "opamp" and c.roles else list(c.nodes)
            if c.type == "opamp" and self.ground is not None:
                ns.append(self.ground)                             # its output is driven against ground
            for a in ns:
                for b in ns:
                    if a != b:
                        join(a, b)
                adj.setdefault(a, set())
        seen, out = set(), []
        for start in sorted(adj, key=lambda n: (n != self.ground, str(n))):
            if start in seen:
                continue
            group, todo = set(), [start]
            while todo:
                n = todo.pop()
                if n not in group:
                    group.add(n)
                    todo += list(adj[n] - group)
            seen |= group
            if self.ground in group:
                continue
            minus = [c.roles["S-"] for c in self.components if c.type == "transformer" and c.roles.get("S-") in group]
            out.append((group, minus[0] if minus else sorted(group, key=str)[0]))
        return out

    def symbol(self, c: Component):
        """the component's value: the number written (k, m, M prefixes), an impedance / phasor written ('3+j4',
        '10∠30°'), or a symbol named by its label"""
        if c.value:
            z = complex_value(c.value)
            if z is not None:
                return z
            v = _number(c.value)
            if v is not None:
                return v
        return sp.Symbol(re.sub(r"\W", "", c.label or c.id) or c.id)

    def __post_init__(self):
        for c in self.components:                  # a source labelled as a phasor ('120∠0°', '5+j5') is an AC one
            if c.type in ("voltage_source", "current_source") and not c.ac and complex_value(c.value) is not None:
                c.ac = True

    def solve(self, mode: str = "dc", switches: dict | None = None, zero_sources: bool = False, test=None,
              w=None, keep=None, bjt: dict | None = None, fet: dict | None = None, accept=None) -> dict:
        """node voltages and the named currents, by modified nodal analysis (ground = 0 V). {} when the circuit
        cannot be solved (no ground, a part it cannot take, a polarity / a leg / what a dependent source follows not
        read, a switch whose state is not given).
          mode 'dc': steady state - a capacitor is open, an inductor a short; mode 's': the Laplace domain - sC, 1/(sL);
          mode 'phasor': sinusoidal steady state - jwC, 1/(jwL), the AC sources as phasors ('10cos(1000t+30°)' ->
          10 at 30° with w = 1000; a 'sin' is 90° behind), w = the sources' own (or `w`, or the symbol omega)
          switches: {switch id: 'open' | 'closed'}
          zero_sources: independent sources set to zero (V -> short, I -> open), for what a part 'sees'
          test: (a, b) - 1 A pushed into node a out of node b (with zero_sources: V(a) - V(b) is the resistance)
          keep: the ids of the independent sources left on (the others set to zero) - superposition
          bjt: DC only - {transistor id: 'active' | 'saturation' | 'cutoff'}: an NPN / PNP transistor whose legs were
            read, as that region has it (V_BE 0.7 V; active: I_C = beta I_B, beta from its label 'β=100', else the
            symbol beta; saturation: V_CE 0.2 V); solve_transistors finds the regions that fit
          fet: DC only - {MOSFET id: 'saturation' | 'triode' | 'cutoff'}: I_D = K (V_GS - V_t)^2 or K (2 V_ov V_DS -
            V_DS^2) with K and V_t from its labels (fet_params) - several solutions then: `accept` picks one"""
        SOURCES = ("voltage_source", "current_source", "dependent_voltage_source", "dependent_current_source")
        s_ = sp.Symbol("s")
        switches = switches or {}
        indep = [c for c in self.components if c.type in ("voltage_source", "current_source")]
        on = [c for c in indep if not zero_sources and (keep is None or c.id in keep)]
        if mode != "phasor" and any(c.ac for c in on):
            return {}                                              # an AC source has no DC steady state
        jw = None
        if mode == "phasor":
            if any(not c.ac for c in on):
                return {}                                          # DC and AC together: superposition (superposed)
            ws = {ac_phasor(c.value)[1] for c in on if ac_phasor(c.value)}
            if len(ws) > 1:
                return {}
            jw = sp.I * (w if w is not None else ws.pop() if ws else sp.Symbol("omega"))
        parts = []
        for c in self.components:
            t = c.type
            if t == "switch":
                state = switches.get(c.id)
                if state is None:
                    return {}                                      # its state at the time asked is not known
                if len(c.nodes) == 3:                              # a two-way switch: common end to throw 1 or 2
                    if state not in ("1", "2"):
                        return {}
                    parts.append((Component(c.id, "wire", [c.nodes[0], c.nodes[int(state)]]), "wire"))
                elif state == "closed":
                    parts.append((c, "wire"))
                continue
            if t == "meter":                                       # an ammeter is a short, a voltmeter open
                kind = (c.label or c.value or "").strip().upper()[:1]
                if kind == "A":
                    parts.append((c, "wire"))
                elif kind != "V":
                    return {}
                continue
            if t in ("capacitor", "inductor") and mode != "phasor" and complex_value(c.value) is not None:
                return {}                                          # labelled with its impedance: phasors only
            if t == "capacitor" and mode == "dc":
                continue                                           # open in the steady state
            if t == "inductor" and mode == "dc":
                parts.append((c, "wire"))
                continue
            if t in ("resistor", "lamp", "wire", "capacitor", "inductor", "opamp") + SOURCES:
                parts.append((c, t))
                continue
            if t == "transformer" and {"P+", "P-", "S+", "S-"} <= set(c.roles) and turns_ratio(c.value) is not None:
                parts.append((c, t))                               # ideal: its turns ratio read
                continue
            if t == "transformer" and {"P+", "P-", "S+", "S-"} <= set(c.roles) and coupling(c) is not None:
                if mode == "dc":                                   # coupled coils: in DC two shorts, no coupling
                    parts.append((Component(c.id + "_p", "wire", [c.roles["P+"], c.roles["P-"]]), "wire"))
                    parts.append((Component(c.id + "_s", "wire", [c.roles["S+"], c.roles["S-"]]), "wire"))
                else:
                    parts.append((c, "coupled"))
                continue
            if t == "vccs" and {"c+", "c-", "o+", "o-"} <= set(c.roles):
                parts.append((c, t))                               # g (V(c+) - V(c-)) from o+ to o-: a small-signal model
                continue
            if t in ("nmos", "pmos") and fet is not None and c.id in fet and mode == "dc" and \
                    {"Gate", "Drain", "Source"} <= set(c.roles) and fet_params(c) is not None:
                parts.append((c, "mosfet"))
                continue
            if t in ("npn", "pnp") and bjt is not None and c.id in bjt and mode == "dc" and \
                    {"Base", "Collector", "Emitter"} <= set(c.roles):
                parts.append((c, "bjt"))
                continue
            return {}                                              # transistors otherwise, diodes, logic ...
        if self.ground is None:
            return {}
        for c, t in parts:
            if t == "opamp" and not {"In+", "In-", "Out"} <= set(c.roles):
                return {}                                          # an op-amp whose legs were not all read
            if t == "opamp":                                       # fed back to In+ only (a Schmitt trigger) or not
                o_, p_, m_ = c.roles["Out"], c.roles["In+"], c.roles["In-"]   # at all (a comparator): not linear
                joins = lambda a, b: any({a, b} <= set(x.nodes) for x in self.components if x is not c)   # noqa: E731
                if (joins(o_, p_) and not joins(o_, m_)) or (not joins(o_, m_) and o_ != m_ and not joins(o_, p_)
                                                              and not any(o_ in x.nodes for x in self.components
                                                                          if x is not c and m_ in x.nodes)):
                    if not (o_ == m_):                     # a follower joins Out and In- by a wire: same node
                        return {}
            if t not in ("opamp", "transformer", "bjt", "vccs", "mosfet", "coupled") and len(c.nodes) != 2:
                return {}
            if t in SOURCES and c.plus not in c.nodes:
                return {}                                          # a source whose polarity was not read
        nodes = sorted({n for c in self.components for n in (c.roles.values() if c.type in ("opamp", "transformer", "npn", "pnp", "nmos", "pmos", "vccs") and c.roles
                                                             else c.nodes)} | {self.ground})
        # the unknowns are named apart from anything a part's label may be called ('V_i' the source, node i's voltage)
        V = {n: (0 if n == self.ground else sp.Symbol(f"__node{k}")) for k, n in enumerate(nodes)}
        shown = {V[n]: f"V_{self.node_labels.get(n, n)}" for n in nodes if n != self.ground}
        eqs, unknowns, leave = [], [v for v in V.values() if v != 0], {n: 0 for n in V}
        through = {}                                               # part id -> its current (as an expression)
        bjt_currents = {}                                          # transistor id -> (I_B, I_C) into it (NPN)
        fet_currents = {}                                          # MOSFET id -> I_D (drain to source, N channel)
        vs_current = {c.id: sp.Symbol(f"__I_{c.id}") for c, t in parts if t in ("voltage_source", "dependent_voltage_source")}

        def follows(c):
            """a dependent source's value: its gain times the voltage or current it follows (None: not read)"""
            m = _DEP.match(c.value or "")
            if not m:
                return None
            g = m.group("gain")
            gain = 1 if not g else (_number(g) if _number(g) is not None else sp.Symbol(re.sub(r"\W", "", g)))
            var = re.sub(r"[_{}\s]", "", m.group("var"))
            if var[0] in "Vv":                                     # a voltage marked '+' / '-' (V_x), or a node's: V_A
                vol = next((x for x in self.voltages if re.sub(r"[_{}\s]", "", x.label).lower() == var.lower()), None)
                if vol is not None:
                    return None if vol.plus not in V or vol.minus not in V else gain * (V[vol.plus] - V[vol.minus])
                node = next((n for n, lab in self.node_labels.items() if lab.lower() == var[1:].lower()), None)
                return None if node is None or node not in V else gain * V[node]
            cur = next((x for x in self.currents if re.sub(r"[_{}\s]", "", x.label).lower() == var.lower()), None)
            part = cur and next((x for x in self.components if x.id == cur.through), None)
            if part is None or cur.frm not in V or cur.to not in V:
                return None
            if part.type in ("resistor", "lamp"):                  # the current a resistor carries, as drawn
                return gain * (V[cur.frm] - V[cur.to]) / self.symbol(part)
            if part.id in vs_current:                              # a voltage source's: out of its '+'
                out_of_plus = vs_current[part.id]
                return gain * (out_of_plus if cur.frm == part.plus else -out_of_plus)
            return None
        for c, t in parts:
            if t.startswith("dependent") and follows(c) is None:
                return {}                                          # what it follows was not read
        for k, (c, t) in enumerate(parts):
            if t == "opamp":                                       # ideal: no current in, V+ = V-, any current out
                io = sp.Symbol(f"__I_{c.id}_out")
                unknowns.append(io)
                eqs.append(sp.Eq(V[c.roles["In+"]], V[c.roles["In-"]]))
                leave[c.roles["Out"]] -= io
                continue
            if t == "vccs":
                g = sp.sympify(c.value)
                i = g * (V[c.roles["c+"]] - V[c.roles["c-"]])
                leave[c.roles["o+"]] += i
                leave[c.roles["o-"]] -= i
                through[c.id] = (c.roles["o+"], i)
                continue
            if t == "mosfet":                                      # N / P channel in the region asked
                region = fet[c.id]
                if region == "cutoff":
                    continue
                K, vt = fet_params(c)
                sg = 1 if c.type == "nmos" else -1
                G, D, S_ = c.roles["Gate"], c.roles["Drain"], c.roles["Source"]
                vov = sg * (V[G] - V[S_]) - vt
                vds = sg * (V[D] - V[S_])
                idd = K * vov ** 2 if region == "saturation" else K * (2 * vov * vds - vds ** 2)
                leave[D] += sg * idd
                leave[S_] -= sg * idd
                fet_currents[c.id] = idd
                continue
            if t == "bjt":                                         # NPN / PNP in the region asked
                region = bjt[c.id]
                if region == "cutoff":
                    continue
                sg = 1 if c.type == "npn" else -1
                B, C_, E = c.roles["Base"], c.roles["Collector"], c.roles["Emitter"]
                ib = sp.Symbol(f"__IB_{c.id}")
                unknowns.append(ib)
                eqs.append(sp.Eq(sg * (V[B] - V[E]), sp.Rational(7, 10)))
                if region == "active":
                    ic = transistor_beta(c.value) * ib
                else:
                    ic = sp.Symbol(f"__IC_{c.id}")
                    unknowns.append(ic)
                    eqs.append(sp.Eq(sg * (V[C_] - V[E]), sp.Rational(1, 5)))
                leave[B] += sg * ib
                leave[C_] += sg * ic
                leave[E] -= sg * (ib + ic)
                bjt_currents[c.id] = (ib, ic)
                continue
            if t == "coupled":                                     # v_P = k (L1 i_P + M i_S), v_S = k (M i_P + L2 i_S), k = jw / s
                L1, L2, M = coupling(c)
                k_ = jw if jw is not None else s_
                ip, is_ = sp.Symbol(f"__I_{c.id}_p"), sp.Symbol(f"__I_{c.id}_s")
                unknowns += [ip, is_]
                r_ = c.roles
                eqs.append(sp.Eq(V[r_["P+"]] - V[r_["P-"]], k_ * (L1 * ip + M * is_)))
                eqs.append(sp.Eq(V[r_["S+"]] - V[r_["S-"]], k_ * (M * ip + L2 * is_)))
                leave[r_["P+"]] += ip
                leave[r_["P-"]] -= ip
                leave[r_["S+"]] += is_
                leave[r_["S-"]] -= is_
                through[c.id] = (r_["P+"], ip)
                continue
            if t == "transformer":                                 # ideal, 1 : n: v_S = n v_P, i_P = -n i_S (both into the dots)
                n_ = turns_ratio(c.value)
                ip = sp.Symbol(f"__I_{c.id}_p")
                unknowns.append(ip)
                r_ = c.roles
                eqs.append(sp.Eq(V[r_["S+"]] - V[r_["S-"]], n_ * (V[r_["P+"]] - V[r_["P-"]])))
                leave[r_["P+"]] += ip
                leave[r_["P-"]] -= ip
                leave[r_["S+"]] += -ip / n_
                leave[r_["S-"]] -= -ip / n_
                through[c.id] = (r_["P+"], ip)                     # the primary's current, in at its dot
                continue
            a, b = c.nodes
            if t in ("resistor", "lamp", "capacitor", "inductor"):  # an admittance (dc: only resistors come here)
                k_ = jw if jw is not None else s_
                z = complex_value(c.value)
                if z is not None:                                  # labelled with its impedance ('j4Ω', '3-j2')
                    if jw is None:
                        return {}                                  # a phasor-domain drawing: no DC / s meaning
                    y = 1 / z
                else:
                    y = {"resistor": 1 / self.symbol(c), "lamp": 1 / self.symbol(c),
                         "capacitor": k_ * self.symbol(c), "inductor": 1 / (k_ * self.symbol(c))}[t]
                i = (V[a] - V[b]) * y
                through[c.id] = (a, i)
                leave[a] += i
                leave[b] -= i
            elif t == "wire":                                      # a wire, a closed switch, an ammeter, an L in dc
                iw = sp.Symbol(f"__I_w{k}")
                eqs.append(sp.Eq(V[a], V[b]))
                unknowns.append(iw)
                through[c.id] = (a, iw)
                leave[a] += iw
                leave[b] -= iw
            elif t in ("voltage_source", "dependent_voltage_source"):
                p = c.plus
                m = b if p == a else a
                val = (self.source_value(c, jw) if c in on else 0) if t == "voltage_source" else follows(c)
                eqs.append(sp.Eq(V[p] - V[m], val))
                iv = vs_current[c.id]                           # flowing out of its + terminal into the circuit
                unknowns.append(iv)
                through[c.id] = (m, iv)
                leave[p] -= iv
                leave[m] += iv
            else:                                                  # current sources
                p = c.plus
                m = b if p == a else a
                val = (self.source_value(c, jw) if c in on else 0) if t == "current_source" else follows(c)
                through[c.id] = (m, val)
                leave[p] -= val
                leave[m] += val
        if test is not None:
            ta, tb = test
            if ta not in V or tb not in V:
                return {}
            leave[ta] -= 1
            leave[tb] += 1
        eqs += [sp.Eq(leave[n], 0) for n in V if n != self.ground]
        if any(c.type == "transformer" for c, _ in parts):
            eqs += [sp.Eq(V[ref], 0) for _, ref in self.islands() if ref in V]   # a transformer's far side: from its S-
        try:
            sol = sp.solve(eqs, unknowns, dict=True)
        except Exception:                                          # noqa: BLE001
            return {}
        if not sol:
            return {}
        for one in sol:                                            # nonlinear parts (MOSFETs): the solution that fits
            if any(v not in one for v in unknowns if v in V.values()):
                continue                                           # a floating part: not determined
            if fet_currents and any(not sp.sympify(x).is_real for x in one.values() if not sp.sympify(x).free_symbols):
                continue
            out = {shown[v]: sp.simplify(one.get(v, v)) for v in V.values() if v != 0}
            for cur in self.currents:                              # the named currents, in their drawn direction
                if cur.through in through and cur.frm in V and cur.to in V:
                    start, i = through[cur.through]                # i flows from `start` through the part
                    out[cur.label] = sp.simplify((i if cur.frm == start else -i).subs(one))
            for k, (ib, ic) in bjt_currents.items():               # a transistor's base / collector current
                out[f"__IB_{k}"] = sp.simplify(sp.sympify(ib).subs(one))
                out[f"__IC_{k}"] = sp.simplify(sp.sympify(ic).subs(one))
            for k, idd in fet_currents.items():
                out[f"__ID_{k}"] = sp.simplify(sp.sympify(idd).subs(one))
            if accept is None or accept(out):
                return out
        return {}

    def small_signal(self, vt: float = 0.025, switches: dict | None = None) -> dict:
        """a transistor amplifier at mid-band: the DC bias (solve_transistors, every transistor active), then each
        transistor its hybrid-pi model - g_m = I_C / V_T, r_pi = beta / g_m - the capacitors shorts, the inductors open,
        the DC sources set to zero, the one AC source the input -> {'gm': {id: g_m}, 'rpi': {id: r_pi}, 'av': v_o / v_i,
        'avs': v_o / v_s, 'rin': what the input sees (after a source resistance), 'rout': what the output sees (the
        load taken off)}; the output: a node named o / out / Vo, else the load R_L's, else the rightmost terminal.
        {} when it cannot be worked out"""
        sol, regions = self.solve_transistors(switches)
        if not sol or any(r not in ("active", "saturation") or (r == "saturation" and c.type in ("npn", "pnp"))
                          for c in self.components for k, r in regions.items() if c.id == k):
            return {}                                              # amplifying: BJTs active, MOSFETs in saturation
        qs = [c for c in self.components if c.type in ("npn", "pnp")]
        ms = [c for c in self.components if c.type in ("nmos", "pmos")]
        acs = [c for c in self.components if c.type in ("voltage_source", "current_source") and c.ac]
        gm, rpi, comps = {}, {}, []
        for c in self.components:
            if c in qs:
                nm = re.sub(r"\W", "", c.label or c.id) or c.id
                g = sp.nsimplify(sp.Abs(sol[f"I_C_{nm}"]) / sp.nsimplify(vt))
                if g == 0:
                    return {}
                gm[c.id], rpi[c.id] = g, transistor_beta(c.value) / g
                B, C_, E = c.roles["Base"], c.roles["Collector"], c.roles["Emitter"]
                comps.append(Component(f"__rpi_{c.id}", "resistor", [B, E], value=None, label=f"__rpi{len(rpi)}"))
                v = Component(f"__gm_{c.id}", "vccs", [C_, E], value=str(g))
                v.roles = {"c+": B, "c-": E, "o+": C_, "o-": E}
                comps.append(v)
            elif c in ms:                                                     # g_m = 2 K V_ov, the gate open
                nm = re.sub(r"\W", "", c.label or c.id) or c.id
                g = sp.nsimplify(2 * fet_params(c)[0] * sol[f"V_ov_{nm}"])
                if g == 0:
                    return {}
                gm[c.id] = g
                G, D, S_ = c.roles["Gate"], c.roles["Drain"], c.roles["Source"]
                v = Component(f"__gm_{c.id}", "vccs", [D, S_], value=str(g))
                v.roles = {"c+": G, "c-": S_, "o+": D, "o-": S_}
                comps.append(v)
            elif c.type in ("voltage_source", "current_source") and not c.ac:
                if c.type == "voltage_source":
                    comps.append(Component(c.id, "wire", list(c.nodes)))      # a DC source: no signal across it
            elif c.type in ("voltage_source", "current_source"):
                comps.append(Component(c.id, c.type, list(c.nodes), label="__vs", plus=c.plus or c.nodes[0]))
            elif c.type == "capacitor":
                comps.append(Component(c.id, "wire", list(c.nodes)))          # mid-band: a short
            elif c.type == "inductor":
                continue                                                      # a choke: open
            else:
                comps.append(c)
        subs = {sp.Symbol(f"__rpi{k + 1}"): r for k, r in enumerate(rpi.values())}
        out = {"gm": gm, "rpi": rpi}
        if len(acs) != 1:
            return out
        src = acs[0]
        sig = Netlist(comps, [], self.ground, self.node_labels, self.ports, self.terminals, voltages=self.voltages)
        r = sig.solve("dc", switches)
        if not r:
            return out
        node = lambda res, n: 0 if n == self.ground else sp.sympify(res.get(f"V_{self.node_labels.get(n, n)}")).subs(subs)   # noqa: E731
        vs = sp.Symbol("__vs")
        plus = src.plus or src.nodes[0]
        inp = plus                                                 # the amplifier's input: after a source resistance
        touching = [c for c in self.components if plus in c.nodes and c is not src]
        if len(touching) == 1 and touching[0].type == "resistor":
            inp = next(n for n in touching[0].nodes if n != plus)
        by_label = {lab.lower().replace("_", ""): n for n, lab in self.node_labels.items()}
        outn = next((by_label[x] for x in ("o", "out", "vo", "vout", "y") if x in by_label), None)
        load = next((c for c in self.components if c.type == "resistor" and re.sub(r"[_{}\s]", "", (c.label or "")).lower() == "rl"), None)
        if outn is None and load is not None:
            outn = next((n for n in load.nodes if n != self.ground), None)
        if outn is None and self.terminals:
            right = max(self.terminals, key=lambda t: t[1][0])
            outn = right[0] if right[0] != self.ground else None
        vi = node(r, inp)
        if outn is not None:
            vo = node(r, outn)
            out["avs"] = sp.simplify(vo / vs)
            if vi != 0:
                out["av"] = sp.simplify(vo / vi)
        rest = Netlist([c for c in comps if c.id != src.id], [], self.ground, self.node_labels, voltages=self.voltages)
        if touching and inp != plus:                               # what the amplifier's input sees (R_s left out)
            rest = Netlist([c for c in rest.components if c.id != touching[0].id], [], self.ground, self.node_labels, voltages=self.voltages)
        rin = rest.solve("dc", switches, test=(inp, self.ground))
        if rin:
            out["rin"] = sp.simplify(node(rin, inp))
        if outn is not None:
            zeroed = [Component(c.id, "wire", list(c.nodes)) if c.id == src.id and c.type == "voltage_source" else c
                      for c in comps if not (c.id == src.id and c.type == "current_source") and c is not load]
            ro = Netlist(zeroed, [], self.ground, self.node_labels, voltages=self.voltages).solve("dc", switches, test=(outn, self.ground))
            if ro:
                out["rout"] = sp.simplify(node(ro, outn))
        return out

    def solve_transistors(self, switches: dict | None = None) -> tuple:
        """a DC circuit with transistors whose legs were read - NPN / PNP (V_BE 0.7 V; active: I_C = beta I_B, I_B >= 0,
        V_CE >= 0.2 V; saturation: V_CE 0.2 V, I_C <= beta I_B; cutoff: V_BE < 0.7 V) and N / P channel MOSFETs
        (saturation: I_D = K V_ov^2, V_DS >= V_ov; triode: V_DS < V_ov; cutoff: V_ov < 0) - each in the region that fits
        -> (the solution with 'I_B_Q1', 'I_C_Q1', 'I_E_Q1', 'V_BE_Q1', 'V_CE_Q1' ..., 'I_D_M1', 'V_GS_M1', 'V_DS_M1',
        'V_ov_M1' - I_B, I_C, I_D ... with one transistor -, {id: region}) or ({}, None). beta not written: the symbol
        beta, the active region assumed (its numbers not decided); a MOSFET without K and V_t is not solved"""
        qs = [c for c in self.components if c.type in ("npn", "pnp")]
        ms = [c for c in self.components if c.type in ("nmos", "pmos")]
        if not qs and not ms or len(qs) + len(ms) > 4:
            return {}, None
        if any(not {"Base", "Collector", "Emitter"} <= set(c.roles) for c in qs) or \
                any(not {"Gate", "Drain", "Source"} <= set(c.roles) or fet_params(c) is None for c in ms):
            return {}, None
        known_beta = all(not transistor_beta(c.value).free_symbols for c in qs)
        choices = [("active", "saturation", "cutoff") if known_beta else ("active",)] * len(qs) + \
            [("saturation", "triode", "cutoff")] * len(ms)
        dc = {c.id for c in self.components if c.type in ("voltage_source", "current_source") and not c.ac}
        tol = 1e-9

        def node(r, n):
            return 0 if n == self.ground else r.get(f"V_{self.node_labels.get(n, n)}")

        def fits_fets(r, modes):
            for c in ms:
                K, vt = fet_params(c)
                sg = 1 if c.type == "nmos" else -1
                G, D, S_ = (node(r, c.roles[k]) for k in ("Gate", "Drain", "Source"))
                if None in (G, D, S_):
                    return False
                try:
                    vov, vds = float(sg * (G - S_) - vt), float(sg * (D - S_))
                except TypeError:
                    return False
                region = modes[c.id]
                if region == "saturation" and not (vov >= -tol and vds >= vov - tol):
                    return False
                if region == "triode" and not (vov >= -tol and -tol <= vds <= vov + tol):
                    return False
                if region == "cutoff" and not vov <= tol:
                    return False
            return True
        for combo in itertools.product(*choices):
            modes = dict(zip([c.id for c in qs + ms], combo))
            r = self.solve("dc", switches, keep=dc, bjt={k: v for k, v in modes.items() if k in {c.id for c in qs}},
                           fet={k: v for k, v in modes.items() if k in {c.id for c in ms}},
                           accept=(lambda r_, m=modes: fits_fets(r_, m)) if ms else None)
            if not r:
                continue
            ok, extra = True, {}
            for c in qs:
                sg = 1 if c.type == "npn" else -1
                B, C_, E = (node(r, c.roles[k]) for k in ("Base", "Collector", "Emitter"))
                if None in (B, C_, E):
                    ok = False
                    break
                ib = r.get(f"__IB_{c.id}", sp.Integer(0))
                ic = r.get(f"__IC_{c.id}", sp.Integer(0))
                vbe, vce = sg * (B - E), sg * (C_ - E)
                if known_beta and not (vbe.free_symbols or vce.free_symbols or sp.sympify(ib).free_symbols):
                    beta = transistor_beta(c.value)
                    if modes[c.id] == "active":
                        ok = float(ib) >= -tol and float(vce) >= 0.2 - tol
                    elif modes[c.id] == "saturation":
                        ok = float(ib) > tol and -tol <= float(ic) <= float(beta * ib) + tol
                    else:
                        ok = float(vbe) < 0.7 + tol
                if not ok:
                    break
                nm = re.sub(r"\W", "", c.label or c.id) or c.id      # currents as they flow (into an NPN's base)
                extra.update({f"I_B_{nm}": ib, f"I_C_{nm}": ic, f"I_E_{nm}": sp.simplify(ib + ic),
                              f"V_BE_{nm}": sp.simplify(B - E), f"V_CE_{nm}": sp.simplify(C_ - E),
                              f"V_CB_{nm}": sp.simplify(C_ - B), f"V_EB_{nm}": sp.simplify(E - B),
                              f"V_EC_{nm}": sp.simplify(E - C_)})
            if not ok:
                continue
            for c in ms:
                K, vt = fet_params(c)
                G, D, S_ = (node(r, c.roles[k]) for k in ("Gate", "Drain", "Source"))
                sg = 1 if c.type == "nmos" else -1
                nm = re.sub(r"\W", "", c.label or c.id) or c.id
                extra.update({f"I_D_{nm}": r.get(f"__ID_{c.id}", sp.Integer(0)), f"V_GS_{nm}": sp.simplify(G - S_),
                              f"V_DS_{nm}": sp.simplify(D - S_), f"V_SG_{nm}": sp.simplify(S_ - G),
                              f"V_SD_{nm}": sp.simplify(S_ - D), f"V_ov_{nm}": sp.simplify(sg * (G - S_) - vt)})
            out = {k: v for k, v in r.items() if not k.startswith("__")}
            out.update(extra)
            for group, names in ((qs, ("I_B", "I_C", "I_E", "V_BE", "V_CE", "V_CB", "V_EB", "V_EC")),
                                 (ms, ("I_D", "V_GS", "V_DS", "V_SG", "V_SD", "V_ov"))):
                if len(group) == 1:
                    nm = re.sub(r"\W", "", group[0].label or group[0].id) or group[0].id
                    for q in names:
                        out.setdefault(q, extra[f"{q}_{nm}"])
            return out, modes
        return {}, None

    def source_value(self, c: Component, jw=None):
        """an independent source's value: in the phasor domain, the phasor its label gives ('10cos(1000t+30°)'),
        else the number written or its name"""
        if jw is not None:
            ph = ac_phasor(c.value)
            if ph is not None:
                return ph[0]
            z = complex_value(c.value)                             # '120∠-120°', '5+j5'
            if z is not None:
                return z
            v = _number(c.value) if c.value else None             # '10V': a phasor at 0°
            if v is not None:
                return v
            return sp.Symbol(re.sub(r"\W", "", c.label or c.id) or c.id)
        return self.symbol(c)

    def superposed(self, switches: dict | None = None) -> dict:
        """DC and AC sources together (AC ones at several frequencies too), in the steady state -> {name: (its DC
        value, {w: its phasor at w})}: the node voltages ('V_A'), the named currents, each two-terminal part's
        voltage ('v_R1') and current ('i_R1') - the sum of what each group of sources gives alone (the DC ones: C
        open, L short; those at one angular frequency w: X, i.e. Re(X e^(jwt))). {} when it cannot be worked out"""
        indep = [c for c in self.components if c.type in ("voltage_source", "current_source")]
        groups, dc = {}, {c.id for c in indep if not c.ac}
        for c in indep:
            if c.ac:
                ph = ac_phasor(c.value)
                if ph is None:
                    return {}
                groups.setdefault(ph[1], set()).add(c.id)
        two = [c for c in self.components if c.type in TWO_TERMINAL and len(c.nodes) == 2]
        probes = [Current(f"__i_{c.id}", c.id, c.nodes[0], c.nodes[1]) for c in two]
        here = Netlist(self.components, self.currents + probes, self.ground, self.node_labels, self.ports, voltages=self.voltages)
        found = []
        if dc:
            found.append((here.solve("dc", switches, keep=dc), None))
        for w, ids in groups.items():
            found.append((here.solve("phasor", switches, keep=ids, w=w), w))
        if not found or any(not r for r, _ in found):
            return {}

        def node(r, n):
            return 0 if n == self.ground else r.get(f"V_{self.node_labels.get(n, n)}")
        out = {}
        for k in set.intersection(*[set(r) for r, _ in found]):
            if not k.startswith("__"):
                out[k] = (sum((r[k] for r, w in found if w is None), sp.Integer(0)), {w: r[k] for r, w in found if w is not None})
        for c in two:
            nm = re.sub(r"\W", "", c.label or c.id) or c.id
            vs = [(node(r, c.nodes[0]), node(r, c.nodes[1]), w) for r, w in found]
            if all(a is not None and b is not None for a, b, _ in vs):
                out[f"v_{nm}"] = (sum((a - b for a, b, w in vs if w is None), sp.Integer(0)), {w: a - b for a, b, w in vs if w is not None})
            open_ = (c.type == "switch" and (switches or {}).get(c.id) == "open") or \
                (c.type == "meter" and (c.label or c.value or "").strip().upper()[:1] == "V")
            ii = [(r.get(f"__i_{c.id}", sp.Integer(0) if open_ or (w is None and c.type == "capacitor") else None), w)
                  for r, w in found]                       # a capacitor carries no direct current
            if all(i is not None for i, _ in ii):
                out[f"i_{nm}"] = (sum((i for i, w in ii if w is None), sp.Integer(0)), {w: i for i, w in ii if w is not None})
        return out

    @staticmethod
    def wave(dc_and_phasors):
        """(DC value, {w: phasor}) -> the function of t: DC + sum of Re(X e^(jwt))"""
        t = sp.Symbol("t")
        d, ph = dc_and_phasors
        return d + sum((sp.re(X) * sp.cos(w * t) - sp.im(X) * sp.sin(w * t) for w, X in ph.items()), sp.Integer(0))

    def first_order_response(self, before: dict, after: dict, _memo: dict | None = None) -> dict:
        """a first-order circuit (one capacitor or inductor, DC sources) switched at t = 0 from the switch states
        `before` to `after` -> {'values': {name: x(t)}, 'tau': tau, 'assumed': bool, 'part': the C / L's id}, every
        x(t) = x(inf) + (x(0+) - x(inf)) e^(-t / tau): the node voltages ('V_A'), the named currents ('I1'), and each
        two-terminal part's voltage ('v_R1': its first node's minus its second's) and current ('i_R1': from its
        first node through it). The capacitor's voltage / the inductor's current carry over the switching, from the
        steady state before it (a capacitor cut off from everything before: taken uncharged - 'assumed'). {} when
        it cannot be worked out (more than one C / L, AC sources, a part the solver does not take ...).
        _memo: a dict shared between calls on the same circuit (each state's steady state solved once)"""
        store = [c for c in self.components if c.type in ("capacitor", "inductor")]
        if len(store) != 1 or len(store[0].nodes) != 2 or any(c.ac for c in self.components):
            return {}
        x = store[0]
        a, b = x.nodes
        memo = {} if _memo is None else _memo
        two = [c for c in self.components if c.type in TWO_TERMINAL and len(c.nodes) == 2]
        probes = [Current(f"__i_{c.id}", c.id, c.nodes[0], c.nodes[1]) for c in two]

        def steady(st):
            key = ("dc", tuple(sorted(st.items())))
            if key not in memo:
                here = Netlist(self.components, self.currents + probes, self.ground, self.node_labels, self.ports, voltages=self.voltages)
                memo[key] = here.solve("dc", st)
            return memo[key]

        def tau_of(st):
            key = ("tau", tuple(sorted(st.items())))
            if key not in memo:
                memo[key] = self.time_constant(st)
            return memo[key]
        pre, post, tau = steady(before), steady(after), tau_of(after)
        if not post or tau is None:
            return {}
        node = lambda r, n: 0 if n == self.ground else r.get(f"V_{self.node_labels.get(n, n)}")   # noqa: E731
        assumed = False
        if x.type == "capacitor":
            va, vb = (node(pre, a), node(pre, b)) if pre else (None, None)
            hold = None if va is None or vb is None else va - vb
            held = Component("__held", "voltage_source", [a, b], label="__hold", plus=a)
        else:
            hold = pre.get(f"__i_{x.id}") if pre else None
            held = Component("__held", "current_source", [a, b], label="__hold", plus=b)
        if hold is None:                                            # cut off before the switching: no charge
            hold, assumed = sp.Integer(0), True
        rest = [c for c in self.components if c is not x] + [held]
        probes0 = [Current(p.label, "__held" if p.through == x.id else p.through, p.frm, p.to) for p in probes]
        at0 = Netlist(rest, self.currents + probes0, self.ground, self.node_labels, self.ports, voltages=self.voltages).solve("dc", after)
        if not at0:                                                 # t = 0+: the C / L holds what it had
            return {}
        at0 = {k: sp.sympify(v).subs(sp.Symbol("__hold"), hold) for k, v in at0.items()}
        decay = sp.exp(-sp.Symbol("t") / tau)
        resp = lambda v0, vinf: sp.simplify(vinf + (v0 - vinf) * decay)               # noqa: E731
        values, at = {}, {self.ground: sp.Integer(0)}
        for n in self.nodes():
            k = f"V_{self.node_labels.get(n, n)}"
            if n != self.ground and k in at0 and k in post:
                at[n] = values[k] = resp(at0[k], post[k])
        for cur in self.currents:
            if cur.label in at0 and cur.label in post:
                values[cur.label] = resp(at0[cur.label], post[cur.label])

        def open_in(c, st):                                         # carries no current in that state
            return (c.type == "switch" and st.get(c.id) == "open") or \
                   (c.type == "meter" and (c.label or c.value or "").strip().upper()[:1] == "V")
        for c in two:
            nm = re.sub(r"\W", "", c.label or c.id) or c.id
            n0, n1 = c.nodes
            if n0 in at and n1 in at:
                values[f"v_{nm}"] = sp.simplify(at[n0] - at[n1])
            k = f"__i_{c.id}"
            i0 = at0.get(k, sp.Integer(0) if open_in(c, after) else None)
            iinf = post.get(k, sp.Integer(0) if c.type == "capacitor" or open_in(c, after) else None)
            if i0 is not None and iinf is not None:
                values[f"i_{nm}"] = resp(i0, iinf)
        return {"values": values, "tau": tau, "assumed": assumed, "part": x.id}

    def laplace_response(self, before: dict, after: dict, _memo: dict | None = None) -> dict:
        """a circuit of any order (capacitors, inductors, DC sources) switched at t = 0 from the switch states `before`
        to `after`, in the Laplace domain -> {'values': {name: X(s)}, 'assumed': bool}: the node voltages, the named
        currents, each two-terminal part's voltage and current (named as first_order_response names them). Every
        capacitor's voltage and inductor's current carries over from the steady state before (a capacitor cut off
        then: taken uncharged - 'assumed'): sC beside a source of C v(0-), 1/(sL) beside one of i(0-)/s; the DC
        sources are steps, V/s. {} when it cannot be worked out."""
        store = [c for c in self.components if c.type in ("capacitor", "inductor")]
        if not store or any(c.ac for c in self.components) or any(len(c.nodes) != 2 for c in store):
            return {}
        memo = {} if _memo is None else _memo
        two = [c for c in self.components if c.type in TWO_TERMINAL and len(c.nodes) == 2]
        probes = [Current(f"__i_{c.id}", c.id, c.nodes[0], c.nodes[1]) for c in two]
        key = ("dc", tuple(sorted(before.items())))
        if key not in memo:
            memo[key] = Netlist(self.components, self.currents + probes, self.ground, self.node_labels,
                                self.ports, voltages=self.voltages).solve("dc", before)
        pre = memo[key]
        node = lambda r, n: 0 if n == self.ground else r.get(f"V_{self.node_labels.get(n, n)}")   # noqa: E731
        s_ = sp.Symbol("s")
        comps, subs, assumed = [], {}, False
        for c in self.components:
            if c.type in ("voltage_source", "current_source"):     # a DC source switched on long ago: a step
                k = sp.Symbol(f"__src{len(subs)}")
                comps.append(Component(c.id, c.type, list(c.nodes), label=str(k), plus=c.plus))
                subs[k] = self.symbol(c) / s_
            else:
                comps.append(c)
        for c in store:                                            # what each one holds at t = 0
            a, b = c.nodes
            k = sp.Symbol(f"__ic{len(subs)}")
            if c.type == "capacitor":
                va, vb = (node(pre, a), node(pre, b)) if pre else (None, None)
                v0 = None if va is None or vb is None else va - vb
                if v0 is None:
                    v0, assumed = sp.Integer(0), True
                comps.append(Component(f"__ic_{c.id}", "current_source", [a, b], label=str(k), plus=a))
                subs[k] = self.symbol(c) * v0                      # C v(0-), from b to a beside sC
            else:
                i0 = pre.get(f"__i_{c.id}") if pre else None
                if i0 is None:
                    i0, assumed = sp.Integer(0), True
                comps.append(Component(f"__ic_{c.id}", "current_source", [a, b], label=str(k), plus=b))
                subs[k] = i0 / s_                                  # i(0-)/s, from a to b beside 1/(sL)
        extra = [Current(f"__j_{c.id}", f"__ic_{c.id}", c.nodes[0], c.nodes[1]) for c in store]
        res = Netlist(comps, self.currents + probes + extra, self.ground, self.node_labels, self.ports, voltages=self.voltages).solve("s", after)
        if not res:
            return {}
        res = {k: sp.sympify(v).subs(subs) for k, v in res.items()}
        values, at = {}, {self.ground: sp.Integer(0)}
        for n in self.nodes():
            k = f"V_{self.node_labels.get(n, n)}"
            if n != self.ground and k in res:
                at[n] = values[k] = res[k]
        ids = {c.id: c for c in store}
        for cur in self.currents:                                  # through a C / L: with what it held
            if cur.label in res:
                v = res[cur.label]
                if cur.through in ids and f"__j_{cur.through}" in res:
                    v = v + (1 if cur.frm == ids[cur.through].nodes[0] else -1) * res[f"__j_{cur.through}"]
                values[cur.label] = v

        def open_in(c, st):
            return (c.type == "switch" and st.get(c.id) == "open") or \
                   (c.type == "meter" and (c.label or c.value or "").strip().upper()[:1] == "V")
        for c in two:
            nm = re.sub(r"\W", "", c.label or c.id) or c.id
            n0, n1 = c.nodes
            if n0 in at and n1 in at:
                values[f"v_{nm}"] = at[n0] - at[n1]
            i = res.get(f"__i_{c.id}", sp.Integer(0) if open_in(c, after) else None)
            if i is not None and c.id in ids:
                i = i + res.get(f"__j_{c.id}", 0)
            if i is not None:
                values[f"i_{nm}"] = i
        return {"values": values, "assumed": assumed}

    def logic(self) -> dict:
        """a logic circuit: {node: its Boolean function} for every gate's output, in the inputs the drawing names
        (a node's letter, else the node) - gates whose output and inputs were read; a loop (a latch) is cut"""
        from sympy.logic.boolalg import And, Nand, Nor, Not, Or, Xor
        ops = {"and": And, "or": Or, "nand": Nand, "nor": Nor, "xor": Xor, "not": lambda *x: Not(x[0])}
        driven = {c.roles["out"]: c for c in self.components if c.type in ops and "out" in c.roles}
        memo, busy = {}, set()

        def f(n):
            if n in memo:
                return memo[n]
            if n not in driven or n in busy:
                return sp.Symbol(self.node_labels.get(n, n))
            busy.add(n)
            c = driven[n]
            ins = [f(c.roles[k]) for k in sorted(c.roles, key=lambda k: (len(k), k)) if k.startswith("in")]
            busy.discard(n)
            memo[n] = ops[c.type](*ins) if ins else sp.Symbol(self.node_labels.get(n, n))
            return memo[n]
        return {n: f(n) for n in driven}

    def resistance_seen(self, a: str, b: str, switches: dict | None = None):
        """the resistance between nodes a and b with the independent sources set to zero (Thevenin's), or None"""
        r = self.solve("dc", switches, zero_sources=True, test=(a, b))
        if not r:
            return None
        va = 0 if a == self.ground else r.get(f"V_{self.node_labels.get(a, a)}")
        vb = 0 if b == self.ground else r.get(f"V_{self.node_labels.get(b, b)}")
        return None if va is None or vb is None else sp.simplify(va - vb)

    def solve_diodes(self, switches: dict | None = None, vd: float = 0.0):
        """a DC circuit with diodes: each diode on (a source of vd from its anode to its cathode: 0 ideal, 0.7 the
        constant-drop model) or off (open); a Zener diode also in breakdown (a source of V_Z from its cathode to its
        anode, V_Z from its label '5.1V' / 'V_Z = 5.1V') - the one way they can all be: an on diode carries current
        from anode to cathode, one in breakdown from cathode to anode, an off one has between -V_Z and vd across it ->
        (the solution as solve gives it, with each diode's current 'I_D1' (anode to cathode) and voltage 'V_D1' (anode
        minus cathode), {diode id: 'on' / 'off' / 'breakdown'}), or ({}, None). Values must be numbers (whether a diode
        conducts depends on them); at most 6 diodes; a Zener without V_Z is not solved."""
        diodes = [c for c in self.components if c.type in ("diode", "zener")]
        if not diodes or len(diodes) > 6 or any(len(c.nodes) != 2 or c.plus not in c.nodes for c in diodes):
            return {}, None
        if any(isinstance(self.symbol(c), sp.Symbol) for c in self.components
               if c.type in ("resistor", "voltage_source", "current_source", "lamp")):
            return {}, None
        vz = {c.id: zener_voltage(c.value) for c in diodes if c.type == "zener"}
        if any(v is None for v in vz.values()):
            return {}, None
        anode = {c.id: next(n for n in c.nodes if n != c.plus) for c in diodes}
        tol = 1e-9
        choices = [("on", "off", "breakdown") if c.type == "zener" else ("on", "off") for c in diodes]
        for combo in itertools.product(*choices):
            state = dict(zip([c.id for c in diodes], combo))
            comps = []
            for c in self.components:
                if c.type not in ("diode", "zener"):
                    comps.append(c)
                elif state[c.id] == "on":
                    comps.append(Component(c.id, "voltage_source", [anode[c.id], c.plus], plus=anode[c.id], value=repr(float(vd))))
                elif state[c.id] == "breakdown":                   # held at V_Z, cathode above anode
                    comps.append(Component(c.id, "voltage_source", [anode[c.id], c.plus], plus=c.plus, value=repr(float(vz[c.id]))))
            probes = [Current(f"__d_{c.id}", c.id, anode[c.id], c.plus) for c in diodes if state[c.id] != "off"]
            r = Netlist(comps, self.currents + probes, self.ground, self.node_labels, self.ports, voltages=self.voltages).solve("dc", switches)
            if not r:
                continue
            node = lambda n: 0 if n == self.ground else r.get(f"V_{self.node_labels.get(n, n)}")   # noqa: E731
            ok = True
            for c in diodes:
                if state[c.id] == "on":
                    i = r.get(f"__d_{c.id}")
                    ok = i is not None and float(sp.N(i)) >= -tol
                elif state[c.id] == "breakdown":
                    i = r.get(f"__d_{c.id}")
                    ok = i is not None and float(sp.N(i)) <= tol    # its current runs cathode to anode
                else:
                    va, vk = node(anode[c.id]), node(c.plus)
                    ok = va is not None and vk is not None and \
                        -float(vz.get(c.id, float("inf"))) - tol <= float(sp.N(va - vk)) <= vd + tol
                if not ok:
                    break
            if ok:
                out = {k: v for k, v in r.items() if not k.startswith("__")}
                for c in diodes:
                    nm = re.sub(r"\W", "", c.label or c.id) or c.id
                    out[f"I_{nm}"] = r.get(f"__d_{c.id}", sp.Integer(0))
                    va, vk = node(anode[c.id]), node(c.plus)
                    if va is not None and vk is not None:
                        out[f"V_{nm}"] = sp.nsimplify(va - vk) if va != vk else sp.Integer(0)
                if len(diodes) == 1:                       # one diode: I_D, V_D (a Zener: I_Z, V_Z too, cathode to anode)
                    c = diodes[0]
                    nm = re.sub(r"\W", "", c.label or c.id) or c.id
                    out.setdefault("I_D", out[f"I_{nm}"])
                    if f"V_{nm}" in out:
                        out.setdefault("V_D", out[f"V_{nm}"])
                    if c.type == "zener":
                        out.setdefault("I_Z", -out[f"I_{nm}"])
                        if f"V_{nm}" in out:
                            out.setdefault("V_Z", -out[f"V_{nm}"])
                return out, state
        return {}, None

    def _impedance(self, a: str, b: str, switches: dict | None = None):
        """the impedance between nodes a and b in the Laplace domain, the independent sources set to zero"""
        r = self.solve("s", switches, zero_sources=True, test=(a, b))
        if not r:
            return None
        va = 0 if a == self.ground else r.get(f"V_{self.node_labels.get(a, a)}")
        vb = 0 if b == self.ground else r.get(f"V_{self.node_labels.get(b, b)}")
        return None if va is None or vb is None else sp.simplify(va - vb)

    def natural_polynomial(self, switches: dict | None = None):
        """the characteristic polynomial of the natural response - the independent sources set to zero - as its
        coefficients, highest power first, the first 1: from a capacitor's or inductor's own equation with the rest
        of the circuit, 1/(sC) + Z(s) = 0 or sL + Z(s) = 0 (the part that gives the highest order); None when there is
        no capacitor or inductor, or it cannot be solved"""
        store = [c for c in self.components if c.type in ("capacitor", "inductor") and len(set(c.nodes)) == 2]
        s = sp.Symbol("s")
        best = None
        for x in store:
            rest = Netlist([c for c in self.components if c is not x], [], self.ground, self.node_labels, self.ports, voltages=self.voltages)
            z = rest._impedance(x.nodes[0], x.nodes[1], switches)
            if z is None or z.has(sp.zoo, sp.oo, sp.nan):
                continue
            own = 1 / (s * self.symbol(x)) if x.type == "capacitor" else s * self.symbol(x)
            num = sp.expand(sp.fraction(sp.together(own + z))[0])
            try:
                poly = sp.Poly(num, s)
            except sp.PolynomialError:
                continue
            if poly.degree() >= 1 and (best is None or poly.degree() > best.degree()):
                best = poly
        if best is None:
            return None
        coeffs = best.all_coeffs()
        return [sp.simplify(c / coeffs[0]) for c in coeffs]

    def time_constant(self, switches: dict | None = None):
        """a first-order circuit's time constant: R C or L / R, R the resistance the one capacitor / inductor sees
        (sources set to zero); None when there is not exactly one of them, or it cannot be solved"""
        store = [c for c in self.components if c.type in ("capacitor", "inductor")]
        if len(store) != 1:
            return None
        x = store[0]
        rest = Netlist([c for c in self.components if c is not x], self.currents, self.ground, self.node_labels, self.ports, voltages=self.voltages)
        r = rest.resistance_seen(x.nodes[0], x.nodes[1], switches)
        if r is None or r == 0:
            return None
        return sp.simplify(r * self.symbol(x) if x.type == "capacitor" else self.symbol(x) / r)


_DEP = re.compile(r"^\s*(?P<gain>\d+(?:[.,]\d+)?|[A-Za-zβαµμ]{1,4}(?:_?\{?[0-9A-Za-z]{1,2}\}?)?)?\s*[*·]?\s*"
                  r"(?P<var>[IiVv]_?\{?[0-9A-Za-z]{1,3}\}?)\s*$")       # a dependent source: '2 I_x', 'g_m V_A'


_AC = re.compile(r"^\s*(?P<amp>\d+(?:[.,]\d+)?)?\s*[*·]?\s*\\?(?P<fn>cos|sin)\s*\(\s*(?P<w>\d+(?:[.,]\d+)?)?\s*"
                 r"(?P<pi>\\?pi)?\s*[*·]?\s*t\s*(?:(?P<sign>[+-])\s*(?P<ph>\d+(?:[.,]\d+)?)\s*"
                 r"(?P<deg>°|\^\s*\{?\s*\\circ\s*\}?|deg)?)?\s*\)")


_RATIO = re.compile(r"^\s*(?P<a>\d+(?:[.,]\d+)?|[A-Za-z](?:_?\{?\w{1,2}\}?)?)\s*:\s*"
                    r"(?P<b>\d+(?:[.,]\d+)?|[A-Za-z](?:_?\{?\w{1,2}\}?)?)\s*$")


def zener_voltage(text):
    """a Zener diode's breakdown voltage from its label: '5.1V', 'V_Z = 5.1V', '5V1' -> 5.1; None if not there"""
    t = str(text or "").replace(",", ".")
    m = re.search(r"(?:V_?\{?Z\}?\s*=\s*)?(\d+(?:\.\d+)?)\s*V(\d*)", t)
    if not m:
        return None
    return sp.nsimplify(float(m.group(1) + ("." + m.group(2) if m.group(2) else "")))


_BETA = re.compile(r"^\s*(?:\\?beta|β|h_?\{?FE\}?|hfe)\s*=\s*(?P<v>\d+(?:[.,]\d+)?)\s*$", re.I)


def transistor_beta(text):
    """a transistor's label 'β=100' / 'h_FE = 150' -> beta (a number); else the symbol beta"""
    m = _BETA.match(str(text or ""))
    return sp.nsimplify(float(m.group("v").replace(",", "."))) if m else sp.Symbol("beta")


_UNIT_SCALE = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "\\mu": 1e-6, "n": 1e-9}


def fet_params(c):
    """a MOSFET's K (I_D = K (V_GS - V_t)^2; written k_n / k' / mu C W/L it is twice that) and V_t (positive: its
    size, for a P channel too) from its labels ('V_t = 1V', 'K = 0.5mA/V^2', 'k_n = 1 mA/V²') -> (K, V_t) or None"""
    K = vt = None
    for key, text in (c.params or {}).items():
        m = re.match(r"\s*([-+]?\d+(?:[.,]\d+)?)\s*(\\mu|[munµ])?", str(text))
        if not m:
            continue
        v = float(m.group(1).replace(",", "."))
        k = re.sub(r"[_{}\s\\]", "", key)
        if re.fullmatch(r"V(?:t|T|th|TH|tn|TN|tp|TP|t0|T0|to)", k):
            vt = sp.nsimplify(abs(v))
        elif k == "K":
            K = sp.nsimplify(v * _UNIT_SCALE.get(m.group(2) or "", 1.0))
        elif re.fullmatch(r"k(?:n|p|'|'n|'p|n'|p')?|mu[np]?Cox.*", k):
            K = sp.nsimplify(v * _UNIT_SCALE.get(m.group(2) or "", 1.0) / 2)
    return None if K is None or vt is None else (K, vt)


_PARAM = re.compile(r"^\s*(?P<key>V_?\{?(?:t|T|th|TH|tn|TN|tp|TP|t0|T0|to)\}?|K|k_?\{?[np]?\}?'?|k'_?\{?[np]?\}?)\s*=\s*"
                    r"(?P<val>[-+]?\d+(?:[.,]\d+)?\s*\S*)\s*$")


def coupling(c):
    """coupled coils' L1, L2 and M (henries) from their labels ('L_1 = 4H', 'L_2 = 9H', 'M = 2H' or 'k = 0.5': M = k
    sqrt(L1 L2)) -> (L1, L2, M), or None"""
    got = {}
    for key, text in (c.params or {}).items():
        k = re.sub(r"[_{}\s\\]", "", key)
        v = _number(re.sub(r"\s*H\s*$", "", str(text)))
        if v is not None and k in ("L1", "L2", "M", "k"):
            got[k] = v
    if "L1" not in got or "L2" not in got:
        return None
    if "M" not in got and "k" in got:
        got["M"] = got["k"] * sp.sqrt(got["L1"] * got["L2"])
    return (got["L1"], got["L2"], got["M"]) if "M" in got else None


def turns_ratio(text):
    """a transformer's label 'N1 : N2' ('1:2', '1:n', 'N_1:N_2') -> n = N2 / N1 (a number or an expression), or None"""
    m = _RATIO.match(str(text or ""))
    if not m:
        return None

    def one(x):
        v = _number(x.replace(",", "."))
        return v if v is not None else sp.Symbol(re.sub(r"\W", "", x))
    a, b = one(m.group("a")), one(m.group("b"))
    return None if a == 0 else sp.nsimplify(b / a) if not (a.free_symbols or b.free_symbols) else b / a


def ac_phasor(text):
    """'10cos(1000t+30°)' -> (10 e^{j 30°}, 1000) - the phasor (cosine reference: a sine is 90° behind) and w;
    None when the text is not such a sinusoid"""
    m = _AC.match(str(text or ""))
    if not m:
        return None
    amp = sp.nsimplify(float(m.group("amp").replace(",", "."))) if m.group("amp") else sp.Integer(1)
    w = sp.nsimplify(float(m.group("w").replace(",", "."))) if m.group("w") else sp.Integer(1)
    if m.group("pi"):
        w = w * sp.pi
    ph = sp.Integer(0)
    if m.group("ph"):
        ph = sp.nsimplify(float(m.group("ph").replace(",", ".")))
        ph = ph * sp.pi / 180 if m.group("deg") else ph
        ph = -ph if m.group("sign") == "-" else ph
    if m.group("fn") == "sin":
        ph = ph - sp.pi / 2
    return amp * sp.exp(sp.I * ph), w


_POLAR = re.compile(r"(?P<amp>\d+(?:\.\d+)?)\s*(?:∠|\\angle)\s*(?P<ang>[-+]?\s*\d+(?:\.\d+)?)\s*"
                    r"(?P<deg>°|\^\s*\{?\s*\\circ\s*\}?|deg)?")


def complex_value(text):
    """a phasor or an impedance as labels write it - 'j4', '-j2Ω', '3+j4 Ω', '4j', '5∠53.13°', '10\\angle 30^\\circ',
    '120∠-120° V' - as a complex number; None for anything else (a plain '10Ω' too)"""
    t = re.sub(r"\\Omega|Ω|ohm|\\,|\s|\\mathrm\{[^}]*\}|\$", "", str(text or ""))
    t = re.sub(r"(?<=\d)[VA]$", "", t)
    m = _POLAR.fullmatch(t) or _POLAR.match(t)
    if m and m.end() >= len(t) - 1:
        ang = sp.nsimplify(float(m.group("ang").replace(" ", "")))
        return sp.nsimplify(float(m.group("amp"))) * sp.exp(sp.I * ang * sp.pi / 180)
    if "j" not in t:
        return None
    t = re.sub(r"(?<![\d.])j(\d+(?:\.\d+)?)", r"\1*I", t)          # j4 -> 4*I
    t = re.sub(r"(\d+(?:\.\d+)?)j", r"\1*I", t)                       # 4j -> 4*I
    t = re.sub(r"(?<![\w*])j(?![\w])", "I", t)                         # a lone j
    if not re.fullmatch(r"[-+]?[\d.]*(?:\*I)?(?:[-+][\d.]*(?:\*I)?)?|[-+]?I", t):
        return None
    try:
        v = sp.nsimplify(sp.sympify(t.replace("*I", "*I")), rational=True)
    except (sp.SympifyError, TypeError, ValueError):
        return None
    return v if v.has(sp.I) else None


def _number(text: str):
    m = re.match(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*([pnuµmkKMG]?)", str(text))
    if not m:
        return None
    mult = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}.get(m.group(2), 1)
    return sp.nsimplify(float(m.group(1).replace(",", ".")) * mult)


def parse_json(text: str) -> dict:
    """the first JSON object in a model's answer (```json fences, prose around it, trailing commas)"""
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(t)):
        depth += {"{": 1, "}": -1}.get(t[i], 0)
        if depth == 0:
            chunk = t[start:i + 1]
            for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                try:
                    return json.loads(attempt)
                except ValueError:
                    continue
            break
    return {}


def compare(read: Netlist, truth: Netlist) -> dict:
    """how a reading matches the transcription, under the renaming of its nodes that matches best:
    components (type + the two nodes), polarity of the sources (those whose transcription has one and that are not
    drawn as AC sources), the diodes' cathodes, the letters at nodes, ground, currents"""
    rn, tn = read.nodes(), truth.nodes()
    best = None
    pad = rn + [None] * max(0, len(tn) - len(rn))
    for perm in itertools.permutations(tn + [None] * max(0, len(rn) - len(tn)), len(pad)):
        mp = {r: t for r, t in zip(pad, perm) if r is not None}
        matched, used = [], set()
        for tc in truth.components:
            for i, rc in enumerate(read.components):
                if i in used or rc.type != tc.type or sorted(mp.get(n) or "?" for n in rc.nodes) != sorted(tc.nodes):
                    continue
                used.add(i)
                matched.append((tc, rc))
                break
        # a symmetric circuit fits under several renamings: then the one under which the polar parts agree
        score = (len(matched), mp.get(read.ground) == truth.ground,
                 sum(read.node_labels.get(r) == truth.node_labels.get(t) for r, t in mp.items() if t in truth.node_labels),
                 sum(tc.plus is not None and mp.get(rc.plus) == tc.plus for tc, rc in matched))
        if best is None or score > best[0]:
            best = (score, mp, matched)
    _, mp, matched = best
    polar = {tc.id for tc, rc in matched if tc.type in ("voltage_source", "current_source") and tc.plus is not None
             and not rc.ac}                    # an AC symbol drawn has no polarity to read
    pol = [tc.id for tc, rc in matched if tc.id in polar and mp.get(rc.plus) == tc.plus]
    sources = [tc.id for tc in truth.components if tc.type in ("voltage_source", "current_source") and tc.plus is not None
               and (tc.id in polar or tc.id not in {t.id for t, _ in matched})]
    diodes = [tc.id for tc in truth.components if tc.type in ("diode", "zener") and tc.plus is not None]
    diodes_ok = [tc.id for tc, rc in matched if tc.id in diodes and mp.get(rc.plus) == tc.plus]
    labels_ok = [lab for t, lab in truth.node_labels.items()
                 if any(mp.get(r) == t and read.node_labels.get(r) == lab for r in rn)]
    by_truth = {tc.id: rc for tc, rc in matched}
    cur_ok, cur_dir = [], []
    for tcur in truth.currents:
        rcur = next((c for c in read.currents if c.label.replace(" ", "").lower() == tcur.label.lower()), None)
        rc = by_truth.get(tcur.through)
        if rcur is None or rc is None or rcur.through != rc.id:
            continue
        cur_ok.append(tcur.label)
        if mp.get(rcur.frm) == tcur.frm and mp.get(rcur.to) == tcur.to:
            cur_dir.append(tcur.label)
    kcl_read, kcl_true = read.kcl(), truth.kcl()
    kcl_ok = [n for n, e in kcl_true.items() if _same_kcl(kcl_read.get(n), e)]
    return {"components": f"{len(matched)}/{len(truth.components)}", "extra_components": len(read.components) - len(matched),
            "topology_exact": len(matched) == len(truth.components) == len(read.components),
            "ground": mp.get(read.ground) == truth.ground, "polarity": f"{len(pol)}/{len(sources)}",
            "diode_cathodes": f"{len(diodes_ok)}/{len(diodes)}",
            "node_letters": f"{len(labels_ok)}/{len(truth.node_labels)}",
            "currents_on_the_right_part": f"{len(cur_ok)}/{len(truth.currents)}",
            "current_directions": f"{len(cur_dir)}/{len(truth.currents)}",
            "kcl": f"{len(kcl_ok)}/{len(kcl_true)}", "problems": read.problems()}


def _same_kcl(a: str | None, b: str) -> bool:
    if not a:
        return False

    def expr(s):
        left, right = s.split("=")
        return sp.sympify(left) - sp.sympify(right)
    try:
        d = sp.simplify(expr(a) - expr(b))
        s = sp.simplify(expr(a) + expr(b))
        return d == 0 or s == 0
    except Exception:                                              # noqa: BLE001
        return False


def truth_netlist(path: str) -> Netlist:
    d = json.load(open(path, encoding="utf-8"))
    return Netlist.from_json({"components": [{**c, "label": c.get("label_on_page") or c["id"]} for c in d["components"]],
                              "currents": d.get("currents", []), "ground": d.get("ground"),
                              "node_labels": d.get("node_labels", {})})
