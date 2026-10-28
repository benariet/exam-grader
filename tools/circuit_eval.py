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
        reader = CircuitReader(CircuitDetector(a.detector) if a.detector else None)
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
    for xml in sorted(glob.glob(os.path.join(base, "annotations", "*.xml")))[: a.limit or None]:
        name = os.path.basename(xml)[:-4]
        truth = asc_netlist(os.path.join(base, "spice", name.split("_")[0] + ".asc"))
        if truth.unknown_symbols or any(c.type == "other" for c in truth.components):
            continue                                       # transistors / ICs: their pins' roles are not read yet
        img = ImageOps.exif_transpose(Image.open(glob.glob(os.path.join(base, "images", name + ".*"))[0])).convert("RGB")
        if reader is not None:
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
            mask = stroke_model(small) if stroke_model else classical(small)
            net = netlist_from_boxes(mask, comps, texts)
        c = compare(net, truth)
        rows.append({"image": name, **{k: c[k] for k in ("components", "extra_components", "topology_exact", "polarity")}})
        say(name, c["topology_exact"], c["components"], "polarity", c["polarity"])
    exact = sum(r["topology_exact"] for r in rows)
    pol = [r for r in rows if r["topology_exact"]]
    pol_ok = sum(1 for r in pol if r["polarity"].split("/")[0] == r["polarity"].split("/")[1])
    out = {"mode": "true boxes" if a.true_boxes else "full reader", "images": len(rows), "exact_netlists": exact,
           "polarity_right_of_exact": f"{pol_ok}/{len(pol)}", "rows": rows}
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


def picture(page, net, dbg, path):
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
    for k, t in enumerate(lines):
        d.text((10, 10 + 14 * k), t, fill=(0, 0, 0))
    pic.save(path)


if __name__ == "__main__":
    main()
