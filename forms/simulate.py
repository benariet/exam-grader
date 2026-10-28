"""Simulated students for the answer-box exams: they fill the printed forms by hand, make the mistakes
students make, and hand in a scan or a phone photo. Their handwriting is MNIST's TEST digits (writers
the cell reader never saw); every student keeps a few versions of each digit, a size, a slant, a pen.

    python -m forms.simulate                 # 24 students per exam -> data/course_elec101/sim/

What a student writes (truth.json keeps it): with probability = his ability the right answer, rounded
his way (2 decimals, 3 significant digits, or more); otherwise a typical mistake - the sign, a factor
1000 (mA written as A), rounding too early, a wrong formula; sometimes nothing. Now and then he
crosses out an answer box and writes the answer beside it (the form's rule: that goes to a human).
The page: a scan (slightly turned, blurred, JPEG) or a photo (perspective, a table around it,
uneven light; a few upside down).
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from forms.cells import _mnist_glyph, load_source  # noqa: E402
from forms.read_form import id_valid  # noqa: E402

COURSE_DIR = os.path.join(HERE, "data", "course_elec101")


def random_id(rng) -> str:
    while True:
        s = "".join(str(int(d)) for d in rng.integers(0, 10, 8))
        for last in range(10):
            if id_valid(s + str(last)):
                return s + str(last)


def blank_pages(course_dir, ex_id, dpi=150):
    cache = os.path.join(course_dir, "sim", "_blank")
    os.makedirs(cache, exist_ok=True)
    pre = os.path.join(cache, ex_id)
    if not os.path.isfile(pre + "-1.png"):
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", os.path.join(course_dir, f"{ex_id}.pdf"), pre], check=True)
    out, k = [], 1
    while os.path.isfile(f"{pre}-{k}.png"):
        out.append(Image.open(f"{pre}-{k}.png").convert("RGB"))
        k += 1
    return out


def fmt_student(v: float, style: str) -> str:
    if style == "2dec":
        s = f"{v:.2f}".rstrip("0").rstrip(".")
    elif style == "3sig":
        s = f"{v:.3g}"
        if "e" in s:
            s = f"{v:.2f}".rstrip("0").rstrip(".")
    else:
        s = f"{v:.3f}".rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s[:8]


def wrong_answer(v: float, rng, style) -> str:
    kinds = ["sign", "factor", "round", "formula", "formula"]
    if v == 0:
        kinds = ["formula"]
    for _ in range(10):
        k = kinds[int(rng.integers(len(kinds)))]
        if k == "sign":
            w = -v
        elif k == "factor":
            w = v * (1000 if rng.random() < 0.5 else 0.001)
        elif k == "round":
            w = float(round(v)) if abs(v) >= 1 else round(v, 1)
        else:
            w = v * float(rng.choice([0.5, 2, 0.25, 4, 1.5, 0.7, 3, 10, 0.1])) + float(rng.uniform(-1, 1)) * (abs(v) < 2)
        s = fmt_student(w, style)
        if len(s) <= 8 and abs(float(s) - v) > 0.011 * abs(v) + 1e-9:
            return s
    return fmt_student(v * 2 + 1, style)


class Writer:
    """one student's hand"""

    def __init__(self, rng, digits, labels, own=None):
        """own: indices of ONE real writer's digits (a collection with writer IDs) - the student then writes
        every digit with that person's own varying instances, as a real student does; otherwise three
        versions per digit are picked from the whole collection"""
        self.rng = rng
        self.size = rng.uniform(0.5, 0.82)                  # glyph height / cell height
        self.slant = rng.uniform(-9, 9)
        self.pen = rng.choice(["thin", "normal", "normal", "thick"])
        self.color = [(25, 25, 30), (20, 40, 150), (30, 60, 170), (15, 15, 15)][int(rng.integers(4))]
        self.versions = {}
        for d in range(10):
            mine = [i for i in (own if own is not None else []) if labels[i] == d]
            if len(mine) >= 2:
                self.versions[d] = [digits[i] for i in mine]
            else:                                   # this writer never wrote d: borrow from the collection
                self.versions[d] = [digits[i] for i in rng.choice(np.where(labels == d)[0], 3, replace=False)]
        self.minus_len = rng.uniform(0.35, 0.7)
        self.dot_r = rng.uniform(2.0, 3.6)
        self.jitter = rng.uniform(0.04, 0.12)

    def glyph(self, ch: str, cell_h: int):
        rng = self.rng
        if ch.isdigit():
            vs = self.versions[int(ch)]
            g = _mnist_glyph(vs[int(rng.integers(len(vs)))], int(cell_h * self.size * rng.uniform(0.92, 1.08)), rng)
            if self.pen == "thick":
                g = g.filter(ImageFilter.MaxFilter(3))
            if self.slant or True:
                g = g.rotate(self.slant + float(rng.uniform(-3, 3)), resample=Image.BILINEAR, expand=True)
            return g, 0.0
        if ch == "-":
            L = int(cell_h * 0.75 * self.minus_len)
            g = Image.new("L", (L + 8, 14), 0)
            ImageDraw.Draw(g).line([(4, 7), (4 + L, 7 + rng.uniform(-2, 2))], fill=255, width={"thin": 2, "normal": 3, "thick": 4}[self.pen])
            return g.rotate(float(rng.uniform(-6, 6)), resample=Image.BILINEAR, expand=True), 0.0
        if ch == ".":
            r = self.dot_r * rng.uniform(0.85, 1.15)
            g = Image.new("L", (14, 14), 0)
            ImageDraw.Draw(g).ellipse((7 - r, 7 - r, 7 + r, 7 + r), fill=255)
            return g, 0.28                                   # a decimal point sits low
        raise ValueError(ch)

    def write(self, page: np.ndarray, box, ch: str):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        g, dy = self.glyph(ch, int(h))
        cx = (x0 + x1) / 2 + self.rng.normal(0, self.jitter) * w
        cy = (y0 + y1) / 2 + (dy + self.rng.normal(0, self.jitter)) * h
        paste_ink(page, g, cx, cy, self.color)


def paste_ink(page: np.ndarray, glyph: Image.Image, cx, cy, color):
    g = np.asarray(glyph, np.float32) / 255.0
    h, w = g.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    H, W = page.shape[:2]
    a0, b0, a1, b1 = max(0, -y0), max(0, -x0), min(h, H - y0), min(w, W - x0)
    if a1 <= a0 or b1 <= b0:
        return
    cov = g[a0:a1, b0:b1, None]
    reg = page[y0 + a0:y0 + a1, x0 + b0:x0 + b1].astype(np.float32)
    page[y0 + a0:y0 + a1, x0 + b0:x0 + b1] = (reg * (1 - cov) + np.array(color, np.float32) * cov).astype(np.uint8)


def scribble_work(page: np.ndarray, area, writer: Writer, rng):
    """a few lines of 'work' in a question's work area (only to make the page look used)"""
    x0, y0, x1, y1 = area
    for line in range(int(rng.integers(1, 3))):
        y = y0 + 30 + line * 45
        if y > y1 - 20:
            break
        x = x1 - 30
        for _ in range(int(rng.integers(4, 12))):
            g, _ = writer.glyph(str(int(rng.integers(10))), 52)
            paste_ink(page, g, x, y, writer.color)
            x -= g.width + rng.integers(2, 10)
            if x < x0 + 30:
                break


def degrade(img: Image.Image, rng, mode: str) -> Image.Image:
    if mode == "scan":
        im = img.rotate(float(rng.uniform(-1.8, 1.8)), resample=Image.BICUBIC, fillcolor=(255, 255, 255),
                        translate=(float(rng.uniform(-15, 15)), float(rng.uniform(-15, 15))))
        s = rng.uniform(0.97, 1.03)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 0.8))))
        a = np.asarray(im, np.float32) + rng.normal(0, rng.uniform(2, 6), (im.height, im.width, 1))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if rng.random() < 0.5:
            im = im.convert("L")
    else:                                                    # a phone photo on a table
        import cv2
        W, H = img.size
        bw, bh = int(W * 1.35), int(H * 1.3)
        base = np.ones((bh, bw, 3), np.float32) * np.array(rng.uniform(60, 170, 3), np.float32)
        base += np.linspace(-30, 30, bw)[None, :, None] + rng.normal(0, 6, (bh, bw, 1))
        ox, oy = (bw - W) / 2, (bh - H) / 2
        src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
        d = 0.05 * W
        dst = np.float32([[ox + rng.uniform(-d, d), oy + rng.uniform(-d, d)], [ox + W + rng.uniform(-d, d), oy + rng.uniform(-d, d)],
                          [ox + rng.uniform(-d, d), oy + H + rng.uniform(-d, d)], [ox + W + rng.uniform(-d, d), oy + H + rng.uniform(-d, d)]])
        M = cv2.getPerspectiveTransform(src, dst)
        page = cv2.warpPerspective(np.asarray(img, np.float32), M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(-1, -1, -1))
        mask = cv2.warpPerspective(np.ones((H, W), np.float32), M, (bw, bh), flags=cv2.INTER_LINEAR)[..., None]
        out = page * mask + base * (1 - mask)
        yy, xx = np.mgrid[0:bh, 0:bw]
        light = 1 - rng.uniform(0.1, 0.35) * ((xx / bw - rng.uniform(0, 1)) ** 2 + (yy / bh - rng.uniform(0, 1)) ** 2)
        out = out * light[..., None]
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.3)))
        out += rng.normal(0, 4, out.shape)
        scale = rng.uniform(1.1, 1.6)                            # phone resolution
        im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((int(bw * scale), int(bh * scale)), Image.BICUBIC)
        if rng.random() < 0.12:
            im = im.rotate(180)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(rng.integers(60, 92)))
    return Image.open(io.BytesIO(buf.getvalue()))


def simulate(n_students=24, seed=7, course_dir=COURSE_DIR, photo_share=0.0, exams=None, out_name="sim",
             digit_source="mnist_test", writer_consistent=False):
    """digit_source: whose handwriting the students write with (forms.cells.SOURCES) - for a fair
    measurement one the cell reader was not trained on. writer_consistent: every student is one real
    writer of the collection (needs writer IDs: CVL), all his digits that person's own instances"""
    rng = np.random.default_rng(seed)
    digits, labels = load_source(digit_source)
    hands = None
    if writer_consistent:
        from forms.cells import source_writers
        w = source_writers(digit_source)
        if w is None:
            raise ValueError(f"{digit_source} has no writer IDs")
        ids = sorted(set(w))
        rng.shuffle(ids)
        hands = [np.where(w == x)[0] for x in ids]
    n_made = 0
    course = json.load(open(os.path.join(course_dir, "course.json"), encoding="utf-8"))
    out_root = os.path.join(course_dir, out_name)
    truth_all = []
    for ex_id in (exams or course["exams"]):
        ex = json.load(open(os.path.join(course_dir, f"{ex_id}.json"), encoding="utf-8"))
        blanks = blank_pages(course_dir, ex_id)
        tpl = ex["template"]["pages"]
        where = {}                                           # item -> page number
        for pno, p in tpl.items():
            for item in p["items"]:
                where[item] = int(pno)
        os.makedirs(os.path.join(out_root, ex_id), exist_ok=True)
        for s in range(n_students):
            ability = float(rng.beta(5, 2))
            writer = Writer(rng, digits, labels, own=hands[n_made % len(hands)] if hands else None)
            wid = str(ids[n_made % len(ids)]) if hands else None
            n_made += 1
            style = str(rng.choice(["2dec", "2dec", "3sig", "3dec"]))
            pages = [np.asarray(b).copy() for b in blanks]
            sid = random_id(rng)
            for k, ch in enumerate(sid):
                writer.write(pages[where["id"] - 1], tpl[str(where["id"])]["items"]["id"][k], ch)
            truth = {"exam": ex_id, "student": f"{ex_id}_s{s + 1:02d}", "id": sid, "ability": round(ability, 3), "items": {},
                     **({"writer": wid} if wid else {})}
            for q in ex["questions"]:
                if q["kind"] == "open":
                    truth["items"][q["id"]] = {"written": None, "kind": "open"}
                    continue
                r = rng.random()
                if r < 0.04:
                    text = ""                                # left empty
                elif q["kind"] == "choice":
                    text = str(q["answer"]) if rng.random() < ability else str(int(rng.choice([o for o in (1, 2, 3, 4) if o != q["answer"]])))
                else:
                    text = fmt_student(q["answer"], style) if rng.random() < ability else wrong_answer(q["answer"], rng, style)
                crossed = bool(text) and rng.random() < 0.03
                pno = where[q["id"]]
                boxes = tpl[str(pno)]["items"][q["id"]]
                page = pages[pno - 1]
                for k, ch in enumerate(text):
                    writer.write(page, boxes[k], ch)
                if crossed:                                  # the whole box crossed out, the answer beside it
                    img = Image.fromarray(page)
                    d = ImageDraw.Draw(img)
                    xa, ya, xb, yb = boxes[0][0], boxes[0][1], boxes[len(text) - 1][2], boxes[0][3]
                    d.line([(xa, ya), (xb, yb)], fill=writer.color, width=3)
                    d.line([(xa, yb), (xb, ya)], fill=writer.color, width=3)
                    page[:] = np.asarray(img)
                truth["items"][q["id"]] = {"written": text, "kind": q["kind"], "crossed": crossed}
            for p_i, page in enumerate(pages):               # some 'work' in the work areas
                for q in ex["questions"]:
                    if where.get(q["id"]) == p_i + 1 and rng.random() < 0.7:
                        b = tpl[str(p_i + 1)]["items"][q["id"]]
                        area = (60, b[0][1] - 150, 1180, b[0][1] - 30)
                        scribble_work(page, area, writer, rng)
            mode = "photo" if rng.random() < photo_share else "scan"
            files = []
            for p_i, page in enumerate(pages):
                im = degrade(Image.fromarray(page), rng, mode)
                f = os.path.join(out_root, ex_id, f"{truth['student']}_p{p_i + 1}.jpg")
                im.convert("RGB").save(f, quality=95)
                files.append(os.path.relpath(f, course_dir))
            truth["mode"], truth["files"] = mode, files
            truth_all.append(truth)
        print(ex_id, "done", flush=True)
    for t in truth_all:
        t["digit_source"] = digit_source
    with open(os.path.join(out_root, "truth.json"), "w", encoding="utf-8") as fh:
        json.dump(truth_all, fh, ensure_ascii=False, indent=1)
    return truth_all


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sim_dev")
    ap.add_argument("--digits", default="mnist_test", help="handwriting source: " + ", ".join(__import__("forms.cells").cells.SOURCES))
    ap.add_argument("--photos", type=float, default=0.0, help="share of students who hand in phone photos")
    ap.add_argument("--writer-consistent", action="store_true",
                    help="every student is one real writer of the collection (CVL has writer IDs)")
    a = ap.parse_args()
    simulate(a.students, a.seed, photo_share=a.photos, out_name=a.out, digit_source=a.digits,
             writer_consistent=a.writer_consistent)
