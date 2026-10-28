"""Hand-drawn circuit datasets -> one COCO detection set for the circuit detector (tools/train_circuit_detector.py).

    python tools/circuit_data.py                 # downloads both sets into data/circuits_raw, writes data/circuits
    python tools/circuit_data.py --cghd <dir> --dhcd <dir>     # already downloaded

Sources (both open; cite them when publishing):
  CGHD - Thoma, Bayer, Li, Dengel (DFKI), ICDAR 2021; Hugging Face lowercaseonly/cghd, CC BY-SA 3.0:
         3,293 annotated phone photos of 34 drafters (paper of every kind, light, angle), 59 classes, text labels
  Digitize-HCD - Ahmed et al., Data in Brief 59 (2025); Mendeley Data rngcz5wtv8, CC BY 4.0:
         1,277 scans by 150+ volunteers, 17 component classes (current sources too), transcribed value labels
Split - CGHD by drafter: a test drafter's hand is never seen in training (test: drafters 1 5 10 15 20 25 30 -
        drafter 1's circuits have LTspice netlists, for measuring whole netlists -, val: 3 13 23); Digitize-HCD has
        no writer ids: by image number (test: n % 10 == 0, val: n % 10 == 5)
Classes - UNIFIED below: what a BSc in electrical engineering draws - circuits (R, C, L, independent and dependent
        sources, switches, transformers, meters), electronics (diodes, Zener diodes, BJT, FET, op-amps, thyristors, ICs), digital
        logic (AND OR NOT NAND NOR XOR), terminals (a port's open circles), grounds, crossovers, labels. Junction dots
        are left to the wiring stage, which finds them in the wire strokes; what is rare goes to 'other'.
Dependent sources - drawn as diamonds in exams, absent from both collections: SYNTH_DEP of the round DC sources
        (CGHD 'voltage.dc', Digitize-HCD V-DC / I-DC) are redrawn as diamonds in their place - the circle erased into
        the paper around it, a diamond in the drafter's pen colour and width, '+ -' (a dependent voltage source)
        or an arrow (a dependent current source) inside - and labelled dep_voltage / dep_current.
Images - longest side scaled to 1600 px (CGHD photos are up to 5800 px), JPEG 92.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
import zlib

from PIL import Image, ImageDraw, ImageOps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED = ["resistor", "capacitor", "inductor", "voltage_dc", "voltage_rail", "voltage_ac", "current_dc",
           "current_ac", "dep_voltage", "dep_current", "diode", "zener", "bjt", "fet", "opamp", "thyristor", "switch",
           "transformer", "meter", "lamp", "terminal", "and", "or", "not", "nand", "nor", "xor", "ic",
           "gnd", "crossover", "text", "other"]
CGHD_MAP = {"resistor": "resistor", "resistor.adjustable": "resistor", "resistor.photo": "resistor", "varistor": "resistor",
            "capacitor.unpolarized": "capacitor", "capacitor.polarized": "capacitor", "capacitor.adjustable": "capacitor",
            "inductor": "inductor", "inductor.ferrite": "inductor", "inductor.coupled": "transformer",
            "transformer": "transformer",
            "voltage.dc": "voltage_dc", "voltage.battery": "voltage_dc", "voltage.ac": "voltage_ac",
            "diode": "diode", "diode.light_emitting": "diode", "diode.zener": "zener", "diode.thyrector": "diode",
            "transistor.bjt": "bjt", "transistor.photo": "bjt", "transistor.fet": "fet",
            "operational_amplifier": "opamp", "operational_amplifier.schmitt_trigger": "opamp",
            "thyristor": "thyristor", "triac": "thyristor", "diac": "thyristor", "switch": "switch",
            "probe": "meter", "probe.current": "meter", "probe.voltage": "meter", "lamp": "lamp",
            "terminal": "terminal", "and": "and", "or": "or", "not": "not", "nand": "nand", "nor": "nor", "xor": "xor",
            "integrated_circuit": "ic", "integrated_circuit.ne555": "ic", "integrated_circuit.voltage_regulator": "ic",
            "gnd": "gnd", "vss": "gnd", "crossover": "crossover", "text": "text",
            "junction": None, "explanatory": None, "unknown": None,
            "mechanical": None, "magnetic": None, "optical": None}          # parts of other symbols / not components
SYNTH_DEP = 0.35               # this share of the round DC sources is redrawn as dependent sources (diamonds)
DHCD_MAP = {"Resistor": "resistor", "Capacitor": "capacitor", "Inductor": "inductor", "V-DC": "voltage_dc",
            "V-DC (one port)": "voltage_rail", "V-AC": "voltage_ac", "I-DC": "current_dc", "I-AC": "current_ac",
            "Diode": "diode", "Zener Diode": "zener", "BJT-NPN": "bjt", "BJT-PNP": "bjt", "MOSFET-N": "fet",
            "MOSFET-P": "fet", "Op-Amp": "opamp", "GND": "gnd", "Wire Crossover": "crossover"}
CGHD_TEST, CGHD_VAL = {1, 5, 10, 15, 20, 25, 30}, {3, 13, 23}     # drafter 1: its 12 circuits come with LTspice netlists
MAX_SIDE = 1600


def download(raw):
    """CGHD from Hugging Face, Digitize-HCD from Mendeley Data (only what the detector needs)"""
    import shutil
    import urllib.request
    import zipfile

    from huggingface_hub import snapshot_download
    cghd = os.path.join(raw, "cghd")
    if not os.path.isdir(os.path.join(cghd, "drafter_1")):
        print("downloading CGHD (Hugging Face lowercaseonly/cghd, ~5 GB) ...", flush=True)
        snapshot_download("lowercaseonly/cghd", repo_type="dataset", local_dir=cghd,
                          allow_patterns=["*/annotations/*", "*/images/*", "*.json", "README.md"])
    dhcd = os.path.join(raw, "dhcd")
    base = os.path.join(dhcd, "Digitize-HCD Dataset", "Component Symbol and Text Label Data")
    if not os.path.isfile(os.path.join(base, "component_annotations.json")):
        os.makedirs(dhcd, exist_ok=True)
        z = os.path.join(dhcd, "dhcd.zip")
        url = "https://data.mendeley.com/public-files/datasets/rngcz5wtv8/files/f64e02f0-743c-4967-afa5-29fbdc06f73b/file_downloaded"
        try:
            if not os.path.isfile(z):
                print("downloading Digitize-HCD (Mendeley Data, 2.1 GB) ...", flush=True)
                with urllib.request.urlopen(url) as r, open(z + ".part", "wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 20)
                os.replace(z + ".part", z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dhcd, [n for n in zf.namelist() if n.startswith("Digitize-HCD Dataset/Component Symbol and Text Label Data/")])
        except Exception as e:                           # noqa: BLE001  (go on with CGHD alone)
            # Mendeley may refuse a script's download (403): then fetch it in a browser and put it at z
            print(f"WARNING: Digitize-HCD not available ({type(e).__name__}: {e}) - going on with CGHD only. "
                  f"To add it, download {url} in a browser and save it as {z}", flush=True)
            return cghd, None
    return cghd, base


def load_image(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")       # phone photos: the way they were held
    return im


def save_scaled(im, out_path):
    s = min(1.0, MAX_SIDE / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(out_path, quality=92)
    return s, im.size


def _to_diamond(name, k):
    """deterministic: does object k of image `name` become a dependent source?"""
    return zlib.crc32(f"{name}:{k}".encode()) % 1000 < SYNTH_DEP * 1000


def draw_diamond(img, box, kind, axis_vertical, plus_first, seed):
    """a hand-drawn dependent source over the round source in box (img's pixels): the round symbol erased into the
    paper around it, a diamond with its corners on the box's sides in the drafter's pen colour and width, '+ -' (kind
    'dep_voltage'; the '+' towards the top / left when plus_first) or an arrow ('dep_current', pointing that way)"""
    import numpy as np
    rnd = random.Random(seed)
    W, H = img.size
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 12 or bh < 12:
        return img
    arr = np.asarray(img, dtype=np.float32)
    g = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sub = g[y0:y1, x0:x1]
    paper = float(np.percentile(sub, 90))
    ink = sub < paper - 50
    if ink.sum() > 20:
        pen = tuple(int(v) for v in np.median(arr[y0:y1, x0:x1][ink], axis=0))
        from scipy import ndimage as ndi
        d = ndi.distance_transform_edt(ink)
        width = float(np.clip(2 * np.median(d[d >= ndi.maximum_filter(d, 3)][d[d >= ndi.maximum_filter(d, 3)] > 0]), 1.5, 9))
    else:
        pen, width = (45, 45, 55), max(2.0, 0.025 * max(bw, bh))
    # the symbol erased: its ink (and a pen width around it) replaced by the paper as it lies there - the box's
    # strokes removed by a maximum filter, smoothed (the photo's light gradient kept), with the paper's grain
    from scipy import ndimage as ndi
    m = max(1, int(round(width)))                            # the box's rim stays (the leads end there)
    pad = int(3 * width) + 4
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    reg = arr[Y0:Y1, X0:X1]
    lum = reg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # the paper under the strokes: the average of the paper around them (normalised convolution, ink left out)
    clear = ~ndi.binary_dilation(lum < paper - 30, iterations=int(width / 2) + 2)
    sig = 2.5 * width + 2
    den = ndi.gaussian_filter(clear.astype(np.float32), sig)
    bg = np.stack([ndi.gaussian_filter(reg[..., c] * clear, sig) for c in range(3)], -1) / np.maximum(den, 1e-3)[..., None]
    fine = lum - ndi.gaussian_filter(lum, 2.0)                  # the paper's grain, without its light gradient
    grain = float(np.clip(fine[lum > paper - 25].std(), 0.3, 4.0)) if (lum > paper - 25).sum() > 30 else 1.5
    fill = bg + np.random.default_rng(seed).normal(0, grain, bg.shape[:2])[..., None]
    inner = np.zeros(reg.shape[:2], bool)
    inner[y0 - Y0 + m:y1 - Y0 - m, x0 - X0 + m:x1 - X0 - m] = True
    # only the ink (and a little around it) is painted over: the paper elsewhere keeps its own grain
    strokes_ = ndi.binary_dilation((lum < paper - 30) & inner, iterations=int(width / 2) + 2) & inner
    alpha = np.minimum(1.0, ndi.gaussian_filter(strokes_.astype(np.float32), 1.0) * 1.5) * inner
    arr[Y0:Y1, X0:X1] = np.clip(reg * (1 - alpha[..., None]) + fill * alpha[..., None], 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    j = lambda s: rnd.uniform(-s, s)                         # noqa: E731
    if axis_vertical:                                        # corners: on the axis at the box's ends, across a bit in
        a, b = (cx + j(0.03 * bw), y0 + m), (cx + j(0.03 * bw), y1 - m)
        c, e = (x0 + rnd.uniform(0.0, 0.12) * bw, cy + j(0.04 * bh)), (x1 - rnd.uniform(0.0, 0.12) * bw, cy + j(0.04 * bh))
    else:
        a, b = (x0 + m, cy + j(0.03 * bh)), (x1 - m, cy + j(0.03 * bh))
        c, e = (cx + j(0.04 * bw), y0 + rnd.uniform(0.0, 0.12) * bh), (cx + j(0.04 * bw), y1 - rnd.uniform(0.0, 0.12) * bh)
    size = max(bw, bh)

    def stroke(p, q, w=width):
        pts = [p]
        for t in (0.25, 0.5, 0.75):                          # a hand's wobble
            pts.append((p[0] + t * (q[0] - p[0]) + j(0.008 * size), p[1] + t * (q[1] - p[1]) + j(0.008 * size)))
        pts.append(q)
        d.line(pts, fill=pen, width=max(1, int(round(w * rnd.uniform(0.85, 1.15)))), joint="curve")
    for p, q in ((a, c), (c, b), (b, e), (e, a)):
        stroke(p, q)
    first, second = (a, b) if plus_first else (b, a)         # the '+' end / the arrow's head
    ux, uy = (second[0] - first[0]) / max(1.0, math.dist(first, second)), (second[1] - first[1]) / max(1.0, math.dist(first, second))
    L = math.dist(a, b)
    sm = 0.13 * L * rnd.uniform(0.8, 1.2)
    if kind == "dep_voltage":
        px_, py_ = first[0] + ux * 0.3 * L, first[1] + uy * 0.3 * L
        stroke((px_ - sm / 2, py_), (px_ + sm / 2, py_), 0.8 * width)
        stroke((px_, py_ - sm / 2), (px_, py_ + sm / 2), 0.8 * width)
        mx_, my_ = first[0] + ux * 0.7 * L, first[1] + uy * 0.7 * L
        stroke((mx_ - sm / 2, my_), (mx_ + sm / 2, my_), 0.8 * width)          # a minus is level, as written
    else:
        tail = (second[0] - ux * 0.75 * L, second[1] - uy * 0.75 * L)
        head = (second[0] - ux * 0.25 * L, second[1] - uy * 0.25 * L)
        stroke(tail, head, 0.8 * width)
        hl = 0.12 * L
        for sgn in (1, -1):
            stroke(head, (head[0] - ux * hl - sgn * uy * hl * 0.6, head[1] - uy * hl + sgn * ux * hl * 0.6), 0.8 * width)
    return img


def cghd_items(cghd):
    for xml in sorted(glob.glob(os.path.join(cghd, "drafter_*", "annotations", "*.xml"))):
        drafter = int(re.search(r"drafter_(-?\d+)", xml).group(1))
        root = ET.parse(xml).getroot()
        name = os.path.splitext(os.path.basename(xml))[0]
        imgs = [p for p in glob.glob(os.path.join(os.path.dirname(os.path.dirname(xml)), "images", name + ".*"))]
        if not imgs:
            continue
        objs, draws = [], []
        for k, o in enumerate(root.findall("object")):
            cls = o.findtext("name")
            uni = CGHD_MAP.get(cls, "other")
            if uni is None:
                continue
            b = o.find("bndbox")
            box = [float(b.findtext(k_)) for k_ in ("xmin", "ymin", "xmax", "ymax")]
            if cls == "voltage.dc" and b.findtext("rotation") is not None and _to_diamond(name, k):
                rot = float(b.findtext("rotation")) % 360            # its '+' up at 0, turned counter-clockwise
                vertical = min(abs(rot - 0), abs(rot - 180), abs(rot - 360)) < 45
                plus_first = (rot < 45 or rot >= 315) if vertical else (45 <= rot < 135)
                uni = "dep_voltage"
                draws.append((box, uni, vertical, plus_first))
            objs.append((uni, box, o.findtext("text") if uni == "text" else None))
        split = "test" if drafter in CGHD_TEST else "val" if drafter in CGHD_VAL else "train"
        yield split, f"cghd_d{drafter}_{name}", imgs[0], objs, {"source": "cghd", "drafter": drafter}, draws


def dhcd_items(base):
    comp = json.load(open(os.path.join(base, "component_annotations.json"), encoding="utf-8"))
    texts = {d["file_name"]: d["instances"] for d in json.load(open(os.path.join(base, "text_annotations.json"), encoding="utf-8"))["data_list"]}
    cats = {c["id"]: c["name"] for c in comp["categories"]}
    by_img, draws = {}, {}
    for k, a in enumerate(comp["annotations"]):
        x, y, w, h = a["bbox"]
        uni = DHCD_MAP[cats[a["category_id"]]]
        if uni in ("voltage_dc", "current_dc") and _to_diamond(str(a["image_id"]), k):
            uni = "dep_voltage" if uni == "voltage_dc" else "dep_current"
            draws.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], uni, h >= w, bool(k % 2)))
        by_img.setdefault(a["image_id"], []).append((uni, [x, y, x + w, y + h], None))
    for im in comp["images"]:
        objs = by_img.get(im["id"], []) + [("text", list(map(float, t["bbox"])), t.get("text")) for t in texts.get(im["file_name"], [])]
        n = int(re.search(r"(\d+)", im["file_name"]).group(1))
        split = "test" if n % 10 == 0 else "val" if n % 10 == 5 else "train"
        yield (split, f"dhcd_{os.path.splitext(im['file_name'])[0]}", os.path.join(base, "Circuit Diagram Images", im["file_name"]),
               objs, {"source": "dhcd"}, draws.get(im["id"], []))


def _convert(job):
    """one image: load, scale, redraw the chosen sources as diamonds, save -> (job index, scale, (w, h)) or (job
    index, None, error)"""
    k, path, out_path, draws = job
    try:
        im = load_image(path)
        sc = min(1.0, MAX_SIDE / max(im.size))
        if sc < 1.0:
            im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
        for n, (box, kind, vertical, plus_first) in enumerate(draws):
            im = draw_diamond(im, [v * sc for v in box], kind, vertical, plus_first, seed=zlib.crc32(f"{out_path}:{n}".encode()))
        im.save(out_path, quality=92)
        return k, sc, im.size
    except Exception as e:                               # noqa: BLE001  (a broken file: skipped)
        return k, None, str(e)


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "data", "circuits_raw"))
    ap.add_argument("--cghd", default=None)
    ap.add_argument("--dhcd", default=None, help="the 'Component Symbol and Text Label Data' folder")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "circuits"))
    ap.add_argument("--limit", type=int, default=0, help="at most this many images per source (a quick try)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--check", action="store_true", help="exit 0 if --out was made with these classes, else 1")
    a = ap.parse_args()
    stamp = {"classes": UNIFIED, "synthetic_dependent_sources": SYNTH_DEP}
    stamp_path = os.path.join(a.out, "classes.json")
    if a.check:
        ok = os.path.isfile(os.path.join(a.out, "test.json")) and os.path.isfile(stamp_path) and \
            json.load(open(stamp_path, encoding="utf-8")) == stamp
        print("data/circuits is up to date" if ok else "data/circuits must be prepared (again)")
        sys.exit(0 if ok else 1)
    cghd, dhcd = (a.cghd, a.dhcd) if a.cghd and a.dhcd else download(a.raw)
    jobs = []
    for items in (cghd_items(cghd) if cghd else [], dhcd_items(dhcd) if dhcd else []):
        for n, item in enumerate(items):
            if a.limit and n >= a.limit:
                break
            jobs.append(item)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
    print(f"{len(jobs)} images to convert with {a.workers} processes ...", flush=True)
    work = [(k, path, os.path.join(a.out, "images", split, name + ".jpg"), draws)
            for k, (split, name, path, _, _, draws) in enumerate(jobs)]
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for done, (k, sc, size) in enumerate(ex.map(_convert, work, chunksize=8), 1):
            results[k] = (sc, size)
            if done % 500 == 0:
                print(f"  {done} / {len(jobs)}", flush=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(UNIFIED)]
    cid = {n: i + 1 for i, n in enumerate(UNIFIED)}
    sets = {sp: {"images": [], "annotations": [], "categories": cats} for sp in ("train", "val", "test")}
    counts = {sp: {} for sp in sets}
    for k, (split, name, path, objs, meta, _) in enumerate(jobs):
        s, size = results[k]
        if s is None:
            print(f"skip {path}: {size}", file=sys.stderr)
            continue
        w, h = size
        iid = len(sets[split]["images"]) + 1
        sets[split]["images"].append({"id": iid, "file_name": f"images/{split}/{name}.jpg", "width": w, "height": h, **meta})
        for cls, (x0, y0, x1, y1), text in objs:
            x0, y0, x1, y1 = (max(0.0, min(v * s, lim)) for v, lim in ((x0, w), (y0, h), (x1, w), (y1, h)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            ann = {"id": len(sets[split]["annotations"]) + 1, "image_id": iid, "category_id": cid[cls],
                   "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                   "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0}
            if text:
                ann["text"] = text
            sets[split]["annotations"].append(ann)
            counts[split][cls] = counts[split].get(cls, 0) + 1
    for sp, d in sets.items():
        json.dump(d, open(os.path.join(a.out, f"{sp}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{sp}: {len(d['images'])} images, {len(d['annotations'])} boxes  {dict(sorted(counts[sp].items()))}", flush=True)
    json.dump(stamp, open(stamp_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
