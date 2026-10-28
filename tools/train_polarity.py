"""Train the polarity models (recog/polarity.py) on
  - Digitize-HCD's named port locations: DC voltage sources (Positive / Negative), DC current sources (Flowing
    From / To), diodes and Zener diodes (Anode / Cathode);
  - CGHD's polar symbols (DC sources, batteries, diodes, LEDs, Zeners) in the phone photos: their rotation tags
    say where the '+' / cathode is. CGHD's test drafters (tools/circuit_data.py) are held out: hands never seen.

    python tools/train_polarity.py                  # data/circuits_raw/{dhcd/dhcd.zip, cghd} (tools/circuit_data.py)
    python tools/train_polarity.py --eval-only      # measure the models in models/polarity

The strokes (Digitize-HCD's crops: the dark ink - the green frame baked into them is light and drops out; CGHD's
photos: recog/circuit_wiring.strokes, as the reader sees them) are turned so that the two ends lie on a line, in
both orders (the label flips), with small shifts and pen-width changes. Held out of Digitize-HCD: every crop whose
number falls in a 10th block of 50 (augmented copies of one symbol have neighbouring numbers).
-> models/polarity/<class>.safetensors, results/polarity_eval.json
"""
import argparse
import glob
import io
import json
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--zip", default=os.path.join(HERE, "data", "circuits_raw", "dhcd", "dhcd.zip"))
ap.add_argument("--cghd", default=os.path.join(HERE, "data", "circuits_raw", "cghd"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "polarity"))
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cghd-limit", type=int, default=0, help="at most this many CGHD photos (a quick try)")
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--classes", default="", help="only these, comma-separated (voltage_dc,current_dc,diode)")
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from recog.polarity import H, ROLES, W, Net, aligned  # noqa: E402

CLASSES = {"voltage_dc": ["V-DC"], "current_dc": ["I-DC"], "diode": ["Diode", "Zener Diode"]}
BASE = "Digitize-HCD Dataset/Component Port Location Data/"
# CGHD: where the '+' / cathode lies in the symbol's own frame (rotation 0), as (dx, dy) on screen; a symbol's
# rotation tag turns it counter-clockwise (checked by eye on DC sources, batteries, diodes, LEDs and Zeners).
# Mirrored symbols and those without a rotation tag are left out.
CGHD_POLAR = {"voltage.dc": ("voltage_dc", (0.0, -1.0)), "voltage.battery": ("voltage_dc", (-1.0, 0.0)),
              "diode": ("diode", (1.0, 0.0)), "diode.zener": ("diode", (1.0, 0.0)),
              "diode.light_emitting": ("diode", (1.0, 0.0))}


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def crop_ink(img):
    """pen strokes of a crop: darker than its paper by a margin; the green frame baked into the crops is not ink"""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    g = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    green = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]) > 40
    paper = np.median(g)
    return (g < min(paper - 60, 170)) & ~green


def load(z, folders, role):
    """[(ink mask, first-role point, other point, held out?)]"""
    out = []
    for folder in folders:
        names = sorted(n for n in z.namelist() if n.startswith(BASE + folder + "/XY Coordinates/") and n.endswith(".txt"))
        for n in names[: a.limit or None]:
            pts = {}
            for line in z.read(n).decode("utf-8", "replace").splitlines():
                p = line.rsplit(" ", 2)
                if len(p) == 3:
                    pts[p[0].strip()] = (float(p[1]), float(p[2]))
            if role not in pts or len(pts) != 2:
                continue
            other = next(v for k, v in pts.items() if k != role)
            stem = os.path.splitext(os.path.basename(n))[0]
            ink = crop_ink(Image.open(io.BytesIO(z.read(BASE + folder + "/Input Images/" + stem + ".jpg"))))
            k = int(re.search(r"(\d+)$", stem).group(1))
            out.append((ink, pts[role], other, (k // 50) % 10 == 0))
    return out


def turn(u, deg):
    """u turned counter-clockwise on screen (y grows downwards) by deg degrees"""
    t = math.radians(deg)
    return (u[0] * math.cos(t) + u[1] * math.sin(t), -u[0] * math.sin(t) + u[1] * math.cos(t))


def box_ends(box, u):
    """where the line through the box's centre along u leaves the box: (end in direction u, the other end)"""
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    t = min(hw / abs(u[0]) if abs(u[0]) > 1e-6 else 1e9, hh / abs(u[1]) if abs(u[1]) > 1e-6 else 1e9)
    return (cx + t * u[0], cy + t * u[1]), (cx - t * u[0], cy - t * u[1])


def _cghd_jobs(root):
    """[(image path, [(unified class, box, first-role direction or None)], [text boxes], held out?)] - the photos
    with polar symbols that carry a rotation tag (mirrored ones and those without a tag are not used)"""
    from circuit_data import CGHD_MAP, CGHD_TEST
    jobs = []
    for xml in sorted(glob.glob(os.path.join(root, "drafter_*", "annotations", "*.xml"))):
        objs, texts, n_polar = [], [], 0
        for o in ET.parse(xml).getroot().findall("object"):
            b = o.find("bndbox")
            if b is None:
                continue
            name = o.findtext("name")
            uni = CGHD_MAP.get(name, "other")
            if uni is None:
                continue
            box = [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if uni == "text":
                texts.append(box)
                continue
            u = None
            if name in CGHD_POLAR and b.findtext("rotation") is not None and not (b.findtext("mirror") or o.findtext("mirror")):
                u = turn(CGHD_POLAR[name][1], float(b.findtext("rotation")))
                n_polar += 1
            objs.append((CGHD_POLAR[name][0] if name in CGHD_POLAR else uni, box, u))
        imgs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(xml)), "images", os.path.basename(xml)[:-4] + ".*"))
        if n_polar and imgs:
            jobs.append((imgs[0], objs, texts, int(re.search(r"drafter_(-?\d+)", xml).group(1)) in CGHD_TEST))
    return jobs


def _photo(path):
    """a CGHD photo as the reader sees it: turned upright, scaled to LONG_SIDE -> (image, scale) or None"""
    from recog.circuit_reader import LONG_SIDE
    from recog.strokes import scale_to
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:                                    # noqa: BLE001  (a broken file: skipped)
        return None
    return scale_to(im, LONG_SIDE), LONG_SIDE / max(im.size)


def _cghd_image(job):
    """one photo, classical strokes (recog/circuit_wiring.strokes) -> _cghd_crops"""
    from recog.circuit_wiring import strokes
    ph = _photo(job[0])
    return [] if ph is None else _cghd_crops((job, strokes(ph[0]), ph[1]))


def _cghd_crops(args):
    """(job, strokes, scale) -> [(class, strokes around the symbol, first-role end, other end, held out?)]. As the
    reader does: the wiring stage's two ends of each part (where its wires meet its box); the annotation's
    rotation says which end has the first role."""
    from recog.circuit_wiring import _end_points, netlist_from_boxes
    (path, objs, texts, held), mask, s = args
    comps = [(cls, [v * s for v in box]) for cls, box, _ in objs]
    _, dbg = netlist_from_boxes(mask, comps, [([v * s for v in b], None) for b in texts], polarity=False, return_debug=True)
    out = []
    h, w = mask.shape
    for (cls, box, u), e in zip(objs, dbg["ends"]):
        if u is None or not e or len(e) != 2 or None in e:
            continue
        box = [v * s for v in box]
        pts = _end_points(box, dbg["labels"], e, h, w)
        if pts is None:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        proj = [(p[0] - cx) * u[0] + (p[1] - cy) * u[1] for p in pts]
        if abs(proj[0] - proj[1]) < 0.3 * max(box[2] - box[0], box[3] - box[1]):
            continue                                     # the two ends are not apart along the symbol: unsure
        first, other = (pts[0], pts[1]) if proj[0] > proj[1] else (pts[1], pts[0])
        r = int(1.3 * max(box[2] - box[0], box[3] - box[1], abs(first[0] - other[0]), abs(first[1] - other[1]))) + 8
        mx, my = int((first[0] + other[0]) / 2), int((first[1] + other[1]) / 2)
        x0, y0 = max(0, mx - r), max(0, my - r)
        out.append((cls, mask[y0:my + r, x0:mx + r].copy(), (first[0] - x0, first[1] - y0),
                    (other[0] - x0, other[1] - y0), held))
    return out


def cghd_load(root):
    """{cls: [(strokes around the symbol, first-role end, other end, held out?)]} from CGHD's photos"""
    import multiprocessing as mp

    from recog.strokes import DEFAULT_DIR, StrokeModel
    jobs = _cghd_jobs(root)[: a.cghd_limit or None]
    say(f"CGHD: {len(jobs)} photos with polar symbols")
    res = []
    learned = os.path.isfile(os.path.join(DEFAULT_DIR, "model.safetensors"))
    with mp.get_context("spawn").Pool(max(1, min(8, (os.cpu_count() or 2) - 1))) as pool:
        if learned:                                      # what the reader uses: the learned strokes (on the GPU,
            sm = StrokeModel(DEFAULT_DIR)                # here), the wiring in the worker processes

            def masks():
                for j in jobs:
                    ph = _photo(j[0])
                    if ph is not None:
                        yield j, sm(ph[0]), ph[1]
            it = pool.imap(_cghd_crops, masks(), chunksize=2)
        else:
            it = pool.imap(_cghd_image, jobs, chunksize=4)
        for k, r in enumerate(it):
            res.append(r)
            if (k + 1) % 200 == 0:
                say(f"  {k + 1}/{len(jobs)} photos")
    out = {}
    for r in res:
        for cls, crop, f, o, held in r:
            out.setdefault(cls, []).append((crop, f, o, held))
    say("CGHD: " + ", ".join(f"{k} {len(v)} ({sum(x[3] for x in v)} of test drafters)" for k, v in out.items()))
    return out


def accuracy(net, items):
    """share of items whose first-role end the net picks (both orders asked, as recog/polarity.py does)"""
    if not items:
        return None
    right = 0
    with torch.no_grad():
        for ink, first, other, _ in items:
            x = torch.from_numpy(np.stack([aligned(ink, first, other), aligned(ink, other, first)]))[:, None]
            p = torch.sigmoid(net(x)).flatten().tolist()
            right += (0.5 * (p[0] + 1 - p[1])) > 0.5
    return right / len(items)


def sample(item, rnd):
    ink, first, other, _ = item
    u = rnd.random()
    if u < 0.3:                                        # a thicker pen
        ink = ndi.binary_dilation(ink, iterations=rnd.choice((1, 2)))
    elif u < 0.45:                                     # a thinner one
        thin = ndi.binary_erosion(ink)
        if thin.sum() > 0.5 * ink.sum():
            ink = thin
    if rnd.random() < 0.2:                             # specks of dirt / paper texture
        ink = ink | ndi.binary_dilation(np.random.default_rng(rnd.randrange(1 << 30)).random(ink.shape) < 0.002)
    def jitter(p):
        return (p[0] + rnd.uniform(-6, 6), p[1] + rnd.uniform(-6, 6))
    f, o = jitter(first), jitter(other)
    if rnd.random() < 0.5:
        return aligned(ink, f, o), 1.0
    return aligned(ink, o, f), 0.0


def train_one(cls, dh, cg):
    """dh, cg: training items from Digitize-HCD's crops and CGHD's photos; half of each batch comes from the
    photos when there are any (they are fewer, and they look like what the reader sees)"""
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    rnd = random.Random(0)
    steps = a.epochs * max(1, (len(dh) + len(cg)) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    net.train()

    def pick():
        src = cg if cg and (not dh or rnd.random() < 0.5) else dh
        return src[rnd.randrange(len(src))]
    for step in range(steps):
        batch = [sample(pick(), rnd) for _ in range(64)]
        x = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
        y = torch.tensor([b[1] for b in batch])[:, None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0:
            say(f"  {cls} step {step + 1}/{steps} loss {loss.item():.3f}")
    return net.eval()


def main():
    from safetensors.torch import load_file, save_file
    z = zipfile.ZipFile(a.zip) if os.path.isfile(a.zip) else None
    cghd = cghd_load(a.cghd) if os.path.isdir(a.cghd) else {}
    if z is None and not cghd:
        sys.exit(f"neither {a.zip} nor {a.cghd} is there - run tools/circuit_data.py first")
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    for cls, folders in CLASSES.items():
        if a.classes and cls not in a.classes.split(","):
            continue
        dh = load(z, folders, ROLES[cls]) if z is not None else []
        cg = cghd.get(cls, [])
        train = [x for x in dh + cg if not x[3]]
        if a.eval_only:
            net = Net()
            net.load_state_dict(load_file(os.path.join(a.out, f"{cls}.safetensors")))
            net.eval()
        else:
            if not train:
                continue
            say(f"{cls}: training on {len(train)} ({sum(1 for x in dh if not x[3])} Digitize-HCD, "
                f"{sum(1 for x in cg if not x[3])} CGHD)")
            net = train_one(cls, [x for x in dh if not x[3]], [x for x in cg if not x[3]])
            save_file({k: v.contiguous() for k, v in net.state_dict().items()}, os.path.join(a.out, f"{cls}.safetensors"))
        acc_dh = accuracy(net, [x for x in dh if x[3]])
        acc_cg = accuracy(net, [x for x in cg if x[3]])
        rep[cls] = {"dhcd_held_out_accuracy": None if acc_dh is None else round(acc_dh, 4),
                    "dhcd_held_out": sum(1 for x in dh if x[3]),
                    "cghd_test_drafters_accuracy": None if acc_cg is None else round(acc_cg, 4),
                    "cghd_test_drafters": sum(1 for x in cg if x[3]),
                    "train_dhcd": sum(1 for x in dh if not x[3]), "train_cghd": sum(1 for x in cg if not x[3])}
        say(f"{cls}: " + json.dumps(rep[cls]))
    if not a.eval_only:
        json.dump({"classes": [c for c in CLASSES if os.path.isfile(os.path.join(a.out, f"{c}.safetensors"))],
                   "roles": ROLES, "input": [H, W]}, open(os.path.join(a.out, "config.json"), "w"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "polarity_eval.json")
    if a.classes and os.path.isfile(path):                # some classes only: the others' numbers stay
        rep = {**json.load(open(path, encoding="utf-8")), **rep}
    json.dump(rep, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
