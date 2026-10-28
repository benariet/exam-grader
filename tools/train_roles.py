"""Train the terminal-role models (recog/roles.py) on Digitize-HCD's named port locations: op-amps (In+ / In- / Out),
BJTs (Base / Collector / Emitter, NPN and PNP), MOSFETs (Gate / Drain / Source, N and P channel).

    python tools/train_roles.py                     # data/circuits_raw/dhcd/dhcd.zip (tools/circuit_data.py)

Each crop's strokes (the green frame baked into Digitize-HCD's crops dropped, as in tools/train_polarity.py) with
one of its ports marked -> that port's role; turned by quarter turns and small angles, mirrored, thickened /
thinned, the points jittered. The transistor's kind from its strokes alone. Held out: every crop whose number falls
in a 10th block of 50. -> models/roles/<family>_roles.safetensors, <family>_kind.safetensors, results/roles_eval.json
"""
import argparse
import io
import json
import math
import os
import random
import re
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
ap = argparse.ArgumentParser()
ap.add_argument("--zip", default=os.path.join(HERE, "data", "circuits_raw", "dhcd", "dhcd.zip"))
ap.add_argument("--out", default=os.path.join(HERE, "models", "roles"))
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default=None)
a = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from recog.roles import KINDS, ROLES, SIZE, Net, marker, square  # noqa: E402

FOLDERS = {"opamp": [("Op-Amp", None)], "bjt": [("BJT-NPN", "npn"), ("BJT-PNP", "pnp")],
           "fet": [("MOSFET-N", "nmos"), ("MOSFET-P", "pmos")]}
BASE = "Digitize-HCD Dataset/Component Port Location Data/"
DEV = a.device or ("cuda" if torch.cuda.is_available() else "cpu")


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def crop_ink(img):
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    g = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    green = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]) > 40
    return (g < min(np.median(g) - 60, 170)) & ~green


def load(z, fam):
    """[(strokes, {role: (x, y)}, kind, held out?)]"""
    out = []
    for folder, kind in FOLDERS[fam]:
        names = sorted(n for n in z.namelist() if n.startswith(BASE + folder + "/XY Coordinates/") and n.endswith(".txt"))
        for n in names[: a.limit or None]:
            pts = {}
            for line in z.read(n).decode("utf-8", "replace").splitlines():
                p = line.rsplit(" ", 2)
                if len(p) == 3:
                    pts[p[0].strip()] = (float(p[1]), float(p[2]))
            if set(pts) != set(ROLES[fam]):
                continue
            stem = os.path.splitext(os.path.basename(n))[0]
            ink = crop_ink(Image.open(io.BytesIO(z.read(BASE + folder + "/Input Images/" + stem + ".jpg"))))
            k = int(re.search(r"(\d+)$", stem).group(1))
            out.append((ink, pts, kind, (k // 50) % 10 == 0))
    return out


def turned(ink, pts, rnd):
    """the crop turned (quarter turns + a small angle) and maybe mirrored, the points moved along"""
    h, w = ink.shape
    ang = rnd.choice((0, 90, 180, 270)) + rnd.uniform(-12, 12)
    im = Image.fromarray(ink.astype(np.uint8) * 255).rotate(ang, resample=Image.BILINEAR, expand=True)
    W, H = im.size
    t = math.radians(ang)
    cx, cy = w / 2, h / 2

    def move(p):                                       # PIL turns counter-clockwise about the centre
        dx, dy = p[0] - cx, p[1] - cy
        return (dx * math.cos(t) + dy * math.sin(t) + W / 2, -dx * math.sin(t) + dy * math.cos(t) + H / 2)
    arr = np.asarray(im) > 127
    new = {r: move(p) for r, p in pts.items()}
    if rnd.random() < 0.5:
        arr = arr[:, ::-1]
        new = {r: (W - 1 - p[0], p[1]) for r, p in new.items()}
    u = rnd.random()
    if u < 0.25:
        arr = ndi.binary_dilation(arr, iterations=rnd.choice((1, 2)))
    elif u < 0.4:
        thin = ndi.binary_erosion(arr)
        if thin.sum() > 0.5 * arr.sum():
            arr = thin
    return arr, new


def sample(item, fam, rnd):
    ink, pts, kind, _ = item
    arr, new = turned(ink, pts, rnd)
    role = rnd.randrange(3)
    p = new[ROLES[fam][role]]
    sq, (q,) = square(arr, [(p[0] + rnd.uniform(-4, 4), p[1] + rnd.uniform(-4, 4))])
    return sq, q, role, (KINDS[fam].index(kind) if kind else -1)


def train(fam, items):
    tr = [x for x in items if not x[3]]
    te = [x for x in items if x[3]]
    net, knet = Net(2, 3).to(DEV), (Net(1, 2).to(DEV) if fam in KINDS else None)
    params = list(net.parameters()) + (list(knet.parameters()) if knet else [])
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    steps = a.epochs * max(1, 3 * len(tr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    rnd = random.Random(0)
    for step in range(steps):
        batch = [sample(tr[rnd.randrange(len(tr))], fam, rnd) for _ in range(64)]
        sq = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
        mk = torch.from_numpy(np.stack([marker(b[1]) for b in batch]))[:, None]
        y = torch.tensor([b[2] for b in batch])
        loss = torch.nn.functional.cross_entropy(net(torch.cat([sq, mk], 1).to(DEV)), y.to(DEV))
        if knet is not None:
            yk = torch.tensor([b[3] for b in batch])
            loss = loss + torch.nn.functional.cross_entropy(knet(sq.to(DEV)), yk.to(DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0:
            say(f"  {fam} step {step + 1}/{steps} loss {loss.item():.3f}")
    net.eval()
    if knet is not None:
        knet.eval()
    right, right_all, kind_right = 0, 0, 0                # held out, as they come (no turning)
    with torch.no_grad():
        for ink, pts, kind, _ in te:
            names = ROLES[fam]
            sq, ps = square(ink, [pts[r] for r in names])
            x = torch.from_numpy(np.stack([np.stack([sq, marker(p)]) for p in ps])).to(DEV)
            lp = torch.log_softmax(net(x), -1).cpu().numpy()
            pred = max(__import__("itertools").permutations(range(3)), key=lambda pm: sum(lp[i, pm[i]] for i in range(3)))
            right += sum(pred[i] == i for i in range(3))
            right_all += all(pred[i] == i for i in range(3))
            if knet is not None:
                kind_right += int(knet(torch.from_numpy(sq)[None, None].to(DEV)).argmax()) == KINDS[fam].index(kind)
    rep = {"held_out_parts": len(te), "legs_right": round(right / max(1, 3 * len(te)), 4),
           "parts_all_legs_right": round(right_all / max(1, len(te)), 4), "train_parts": len(tr)}
    if knet is not None:
        rep["kind_right"] = round(kind_right / max(1, len(te)), 4)
    return net, knet, rep


def main():
    from safetensors.torch import save_file
    z = zipfile.ZipFile(a.zip)
    os.makedirs(a.out, exist_ok=True)
    rep = {}
    for fam in FOLDERS:
        items = load(z, fam)
        say(f"{fam}: {len(items)} crops")
        net, knet, r = train(fam, items)
        save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, os.path.join(a.out, f"{fam}_roles.safetensors"))
        if knet is not None:
            save_file({k: v.detach().cpu().contiguous() for k, v in knet.state_dict().items()}, os.path.join(a.out, f"{fam}_kind.safetensors"))
        rep[fam] = r
        say(f"{fam}: {r}")
    json.dump({"families": list(FOLDERS), "roles": ROLES, "kinds": KINDS, "size": SIZE},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rep, open(os.path.join(HERE, "results", "roles_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
