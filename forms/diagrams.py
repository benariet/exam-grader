"""Circuit diagrams for the exam forms (schemdraw -> SVG text)."""
from __future__ import annotations

import schemdraw
import schemdraw.elements as elm

schemdraw.use("svg")


def _svg(d) -> str:
    return d.get_imagedata("svg").decode("utf-8") if isinstance(d.get_imagedata("svg"), bytes) else d.get_imagedata("svg")


def series_parallel() -> str:
    """E with R1 in series, then R2 || R3"""
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.5, fontsize=13)
        src = d.add(elm.SourceV().up().label("E"))
        d.add(elm.Resistor().right().label("R1"))
        d.add(elm.Dot())
        d.push()
        d.add(elm.Resistor().down().label("R2", loc="bottom"))
        d.add(elm.Dot())
        d.pop()
        d.add(elm.Line().right(2))
        d.add(elm.Resistor().down().label("R3", loc="bottom"))
        d.add(elm.Line().left(2))
        d.add(elm.Line().to(src.start))
        return _svg(d)


def divider() -> str:
    """E across R1 (top) and R2 (bottom), Vout across R2"""
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=2.2, fontsize=13)
        src = d.add(elm.SourceV().up().label("E").length(4.4))
        d.add(elm.Line().right(2.5))
        d.add(elm.Resistor().down().label("R1", loc="bottom"))
        d.add(elm.Dot())
        d.push()
        d.add(elm.Line().right(1.5))
        d.add(elm.Dot(open=True).label("+", loc="right"))
        d.pop()
        r2 = d.add(elm.Resistor().down().label("R2", loc="bottom"))
        d.add(elm.Dot())
        d.push()
        d.add(elm.Line().right(1.5))
        d.add(elm.Dot(open=True).label("-", loc="right"))
        d.pop()
        d.add(elm.Line().to(src.start))
        d.add(elm.Label().at((r2.end[0] + 2.4, (r2.start[1] + r2.end[1]) / 2)).label("Vout"))
        return _svg(d)


def kirchhoff() -> str:
    """E1 - R1 - node A - R3 - E2, R2 from A to ground; I1 -> through R1, I3 <- through R3, I2 down R2"""
    with schemdraw.Drawing(show=False) as d:
        d.config(unit=3.2, fontsize=13)
        e1 = d.add(elm.SourceV().up().label("E1"))
        r1 = d.add(elm.Resistor().right().label("R1", loc="bottom"))
        d.add(elm.CurrentLabel(top=True, length=1.2).at(r1).label("I1"))
        d.add(elm.Dot().label("A", loc="top", ofst=0.25))
        d.push()
        r2 = d.add(elm.Resistor().down().label("R2", loc="bottom"))
        d.add(elm.CurrentLabel(top=False, length=1.2).at(r2).label("I2"))
        d.add(elm.Dot())
        d.pop()
        r3 = d.add(elm.Resistor().right().label("R3", loc="bottom"))
        d.add(elm.CurrentLabel(top=True, length=1.2, reverse=True).at(r3).label("I3"))
        d.add(elm.SourceV().down().reverse().label("E2", loc="bottom"))
        d.add(elm.Line().left())
        d.add(elm.Line().to(e1.start))
        d.add(elm.Ground().at(r2.end))
        return _svg(d)


DIAGRAMS = {"series_parallel": series_parallel, "divider": divider, "kirchhoff": kirchhoff}

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/claude-0/diag", exist_ok=True)
    for k, f in DIAGRAMS.items():
        open(f"/tmp/claude-0/diag/{k}.svg", "w").write(f())
        print(k, "ok")
