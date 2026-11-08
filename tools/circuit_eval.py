"""Whole netlists read from hand-drawn circuits, against the circuits they were drawn from: CGHD drafter 1's photos
(a test drafter: his hand was never seen in training) whose twelve circuits come with LTspice schematics, and
Tal's drawing (data/my_scans/tal_circuit_truth.json).

    python tools/circuit_eval.py                  # the full reader (detector + strokes + wiring + polarity)
    python tools/circuit_eval.py --true-boxes     # the true component boxes instead of the detector's: wiring alone

A netlist is exact when every component joins the same nodes as in the schematic (nodes renamed as fits best);
also counted: components found, sources' polarity. -> results/circuit_eval.json
"""
import argparse
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--cghd", default=os.path.join(HERE, "data", "circuits_raw", "cghd"))
ap.add_argument("--true-boxes", action="store_true")
ap.add_argument("--detector", default=None, help="the detector's folder (default models/circuit-detector)")
ap.add_argument("--limit", type=int, default=0, help="at most this many photos (a quick try)")
ap.add_argument("--report", default=os.path.join(HERE, "results", "circuit_eval.json"))
ap.add_argument("--only", default="", help="photos whose names start so, comma-separated ('C10,C3_D1_P2')")
ap.add_argument("--synth", default="", help="a folder of drawings with their netlists (<name>.png + <name>.json: "
                "tools/cmos_synth.py's CMOS gates) - measured by the full reader alone")
ap.add_argument("--no-cghd", action="store_true", help="only the --synth drawings")
ap.add_argument("--letters", default="", help="the letters the drawings' question names ('ABCS'): a letter read as "
                "none of them ('3' for a 'B') re-read as one (recog.circuit_reader.hinted_letters)")
ap.add_argument("--inputs", default="", help="the circuits' inputs as the question names them ('ABC'): the Boolean "
                "functions read in them (tools/ptl_synth.py's pass-transistor networks), a gate's letter one of them")
ap.add_argument("--details", action="store_true", help="each row with the netlist read, its boxes and labels")
ap.add_argument("--via-page", action="store_true", help="read as an exam page is (read_page.find_circuits, then each "
                "drawing found cropped and read) - the largest drawing found is the netlist")
ap.add_argument("--pictures", default="", help="a folder: what the reader saw on each photo not read exactly "
                "(boxes, wire pieces by node, the netlist read and the schematic's)")
a = ap.parse_args()


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def main():
    from PIL import Image, ImageOps

    from circuit_data import CGHD_MAP
    from grader.circuit import compare, truth_netlist
    from ltspice import asc_netlist
    from read_page import load_pages
    from recog.circuit_wiring import netlist_from_boxes
    from recog.strokes import scale_to
    reader = None
    if not a.true_boxes:
        from recog.circuit_reader import CircuitDetector, CircuitReader
        reader = CircuitReader(CircuitDetector(a.detector) if a.detector else None, letters=a.letters, inputs=a.inputs)
    from recog.circuit_reader import LONG_SIDE
    from recog.circuit_wiring import strokes as classical
    stroke_model = None
    if a.true_boxes:
        from recog.strokes import DEFAULT_DIR, StrokeModel
        if os.path.isfile(os.path.join(DEFAULT_DIR, "model.safetensors")):
            stroke_model = StrokeModel()
    rows = []
    base = os.path.join(a.cghd, "drafter_1")
    if not glob.glob(os.path.join(base, "spice", "*.asc")):          # the schematics: a few kB, not fetched before
        from huggingface_hub import snapshot_download
        snapshot_download("lowercaseonly/cghd", repo_type="dataset", local_dir=a.cghd, allow_patterns=["drafter_1/spice/*"])
    todo = [(os.path.basename(x)[:-4], x, base, None) for x in sorted(glob.glob(os.path.join(base, "annotations", "*.xml")))]
    # circuits of other test drafters transcribed by hand (data/circuit_truth/drafter_<n>/<circuit>.json: drafter 10's
    # logic-gate diagrams) - every photo of each
    for tf in sorted(glob.glob(os.path.join(HERE, "data", "circuit_truth", "drafter_*", "*.json"))):
        d = os.path.join(a.cghd, os.path.basename(os.path.dirname(tf)))
        circ = os.path.basename(tf)[:-5]
        todo += [(os.path.basename(x)[:-4], x, d, tf) for x in sorted(glob.glob(os.path.join(d, "annotations", circ + "_*.xml")))]
    if a.no_cghd:
        todo = []
    if a.synth and not a.true_boxes:                       # (drawings with no boxes: the full reader only)
        todo += [(os.path.basename(x)[:-5], None, a.synth, x) for x in sorted(glob.glob(os.path.join(a.synth, "*.json")))]
    for name, xml, folder, truth_file in todo[: a.limit or None]:
        if a.only and not any(name.startswith(x) for x in a.only.split(",") if x):
            continue
        if truth_file:
            truth = truth_netlist(truth_file)
        else:
            truth = asc_netlist(os.path.join(base, "spice", name.split("_")[0] + ".asc"))
            if truth.unknown_symbols or any(c.type == "other" for c in truth.components):
                continue                                   # an IC, a switch: not in the schematics' reader
        transistors = sum(1 for c in truth.components if c.roles)
        path = os.path.join(folder, name + ".png") if xml is None else glob.glob(os.path.join(folder, "images", name + ".*"))[0]
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        dbg = None
        if reader is not None and a.via_page:                 # as an exam page is read: the drawing found, cropped
            from read_page import find_circuits, read_circuits
            regions = find_circuits(img, reader)
            circs = read_circuits(img, regions, reader) if regions else []
            if circs:
                net = max(circs, key=lambda c: len(c["netlist"].components))["netlist"]
            else:
                from grader.circuit import Netlist
                net = Netlist()
        elif reader is not None:
            if a.pictures or a.details:
                net, _, dbg = reader.read(img, return_debug=True)
            else:
                net, _ = reader.read(img)
        else:
            small = scale_to(img, LONG_SIDE)
            s = LONG_SIDE / max(img.size)
            comps, texts = [], []
            for o in ET.parse(xml).getroot().findall("object"):
                b = o.find("bndbox")
                box = tuple(float(b.findtext(k)) * s for k in ("xmin", "ymin", "xmax", "ymax"))
                cls = CGHD_MAP.get(o.findtext("name"), "other")
                if cls is None:
                    continue
                (texts.append((box, o.findtext("text"))) if cls == "text" else comps.append((cls, box)))
            trs = [b for c, b in comps if c == "transformer"]  # (CGHD boxes a transformer's coils too: the part's)

            def inside(b, t):
                ix = max(0.0, min(b[2], t[2]) - max(b[0], t[0])) * max(0.0, min(b[3], t[3]) - max(b[1], t[1]))
                return ix >= 0.7 * (b[2] - b[0]) * (b[3] - b[1])
            comps = [(c, b) for c, b in comps if not (c == "inductor" and any(inside(b, t) for t in trs))]
            mask = stroke_model(small) if stroke_model else classical(small)
            net = netlist_from_boxes(mask, comps, texts)
        c = compare(net, truth)
        if a.pictures and dbg is not None and not c["topology_exact"]:
            os.makedirs(a.pictures, exist_ok=True)
            k = LONG_SIDE / max(img.size)
            true_boxes = [(CGHD_MAP.get(o.findtext("name"), "other") or o.findtext("name"),
                           [float(o.find("bndbox").findtext(v)) * k for v in ("xmin", "ymin", "xmax", "ymax")])
                          for o in ET.parse(xml).getroot().findall("object")] if xml else []
            picture(img, net, dbg, os.path.join(a.pictures, name + ".png"),
                    ["schematic:"] + [f"{x.id} {x.type} {x.nodes}" for x in truth.components], true_boxes)
        rows.append({"image": name, "transistors": transistors,
                     **{k: c[k] for k in ("components", "extra_components", "topology_exact", "polarity", "legs", "kinds")
                        if k in c}})
        hint = a.inputs or None                            # (the question's inputs: 'ABC')
        tf = truth.switch_logic(inputs=hint) if any(x.type in ("nmos", "pmos") for x in truth.components) else {}
        if tf:                                             # a CMOS gate: the Boolean function read, whatever the layout
            import sympy as sp
            up = lambda e: e.xreplace({x: sp.Symbol(str(x).upper()) for x in e.free_symbols})   # noqa: E731
            rf = [up(f) for f in net.switch_logic(inputs=hint).values()]
            rows[-1]["function_right"] = all(any(not sp.satisfiable(sp.Xor(up(f), g)) for g in rf) for f in tf.values())
        if a.details:                                      # (what was read: to sort the failures by their cause)
            rows[-1]["read"] = [(x.id, x.type, x.nodes, x.roles) for x in net.components]
            rows[-1]["labels"] = net.node_labels
            rows[-1]["truth"] = [(x.id, x.type, x.nodes) for x in truth.components]
            if dbg is not None:
                rows[-1]["boxes"] = [(cls, [round(v) for v in b]) for cls, b in dbg["comps"]]
                rows[-1]["texts"] = [(item[1], [round(v) for v in item[0]]) for item in dbg["texts"]]
        say(name, c["topology_exact"], c["components"], "polarity", c["polarity"],
            *(("legs", c["legs"], "kinds", c["kinds"]) if transistors else ()))
    with_t = [r for r in rows if r["transistors"]]
    rows_t, rows = with_t, [r for r in rows if not r["transistors"]]
    exact = sum(r["topology_exact"] for r in rows)
    pol = [r for r in rows if r["topology_exact"]]
    pol_ok = sum(1 for r in pol if r["polarity"].split("/")[0] == r["polarity"].split("/")[1])
    ex_t = [r for r in rows_t if r["topology_exact"]]

    def frac(key):
        return f"{sum(int(r[key].split('/')[0]) for r in ex_t)}/{sum(int(r[key].split('/')[1]) for r in ex_t)}"
    out = {"mode": "true boxes" if a.true_boxes else "full reader", "images": len(rows), "exact_netlists": exact,
           "polarity_right_of_exact": f"{pol_ok}/{len(pol)}", "rows": rows,
           "transistor_circuits": {"images": len(rows_t), "exact_netlists": len(ex_t),
                                   "legs_right_of_exact": frac("legs") if ex_t else "0/0",
                                   "kinds_right_of_exact": frac("kinds") if ex_t else "0/0", "rows": rows_t}}
    tal = os.path.join(HERE, "data", "my_scans", "tal_circuit.pdf")
    if reader is not None and os.path.isfile(tal):
        page = load_pages(tal, 150)[0]
        net, found, dbg = reader.read(page, return_debug=True)
        out["tal"] = {"compare": compare(net, truth_netlist(os.path.join(HERE, "data", "my_scans", "tal_circuit_truth.json"))),
                      "found": [(c, [round(v) for v in b], round(s, 2)) for c, b, s in found],
                      "netlist": [(x.id, x.type, x.nodes, x.plus, x.label, x.value) for x in net.components],
                      "currents": [(c.label, c.through, c.frm, c.to) for c in net.currents],
                      "node_labels": net.node_labels, "kcl": net.kcl(), "problems": net.problems()}
        say("Tal's drawing:", json.dumps(out["tal"]["compare"], ensure_ascii=False))
        picture(page, net, dbg, os.path.join(HERE, "results", "tal_circuit_read.png"))
    json.dump(out, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say(f"exact netlists {exact}/{len(rows)}, polarity right in {pol_ok}/{len(pol)} of them ({out['mode']})")
    tc = out["transistor_circuits"]
    say(f"with transistors or gates: exact netlists {tc['exact_netlists']}/{tc['images']}, legs right {tc['legs_right_of_exact']}, "
        f"kinds right {tc['kinds_right_of_exact']} (of the exact ones)")
    fr = [r["function_right"] for r in rows_t if "function_right" in r]
    if fr:
        say(f"transistor networks (CMOS, pass-transistor) whose Boolean function was read right: {sum(fr)}/{len(fr)}")


def picture(page, net, dbg, path, extra=(), true_boxes=()):
    """what the reader saw: wire pieces coloured by node, the boxes found, the current arrows, the netlist"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.strokes import scale_to
    from recog.circuit_reader import LONG_SIDE
    small = scale_to(page.convert("RGB"), LONG_SIDE)
    pic = Image.blend(small, Image.new("RGB", small.size, "white"), 0.6)
    lab = dbg["labels"]
    rng = np.random.default_rng(7)
    col = rng.integers(0, 200, size=(lab.max() + 1, 3)).astype(np.uint8)
    arr = np.asarray(pic).copy()
    arr[lab > 0] = col[lab[lab > 0]]
    pic = Image.fromarray(arr)
    d = ImageDraw.Draw(pic)
    for cls, b in true_boxes:                      # (the annotation's boxes, green, under the reader's)
        if cls != "text":
            d.rectangle(b, outline=(0, 170, 0), width=1)
            d.text((b[0], b[3] + 2), cls, fill=(0, 140, 0))
    for cls, b in dbg["comps"]:
        d.rectangle(b, outline=(220, 0, 0), width=3)
        d.text((b[0], b[1] - 12), cls, fill=(220, 0, 0))
    for item in dbg["texts"]:                      # (box, text) or (box, text, how sure)
        b, t = item[0], item[1]
        d.rectangle(b, outline=(0, 0, 220), width=2)
        d.text((b[0], b[3] + 2), t or "?", fill=(0, 0, 220))
    for tip, tail in dbg.get("arrows", []):
        d.line([tail, tip], fill=(200, 0, 200), width=5)
        d.ellipse([tip[0] - 9, tip[1] - 9, tip[0] + 9, tip[1] + 9], outline=(200, 0, 200), width=4)
    lines = [f"{c.id} {c.type} {c.nodes}" + (f" +{c.plus}" if c.plus else "") + (f" '{c.label}'" if c.label else "")
             for c in net.components]
    lines += [f"{c.label}: {c.through} {c.frm}->{c.to}" for c in net.currents] + [f"nodes {net.node_labels}"]
    lines += list(extra)
    for k, t in enumerate(lines):
        d.text((10, 10 + 14 * k), t, fill=(0, 0, 0))
    pic.save(path)


if __name__ == "__main__":
    main()
