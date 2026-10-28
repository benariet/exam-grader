"""Read a scanned page (PDF or image) with the mixed Hebrew + math reader - fully local.

    python read_page.py scan.pdf            # -> results/<name>_read.png (+ .txt)
    python read_page.py photo.jpg           # a phone photo: found, straightened and flattened first

What it does: renders the page at --dpi (default 150: handwriting then has the size the models
were trained on); a phone photo (table around the page, uneven light - or --photo) is first cut
out, straightened and flattened, its notebook rules removed (recog.page.flatten_photo); then it
cleans the page and finds the handwritten lines (recog/page.py: rules, margin line, borders, punch
holes and specks removed; touching lines split), runs recog.mixed_reader.MixedReader on every line
and draws each segment with its reading.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def load_pages(path: str, dpi: int):
    if not path.lower().endswith(".pdf"):
        img = Image.open(path)
        return [img.convert("RGB")]
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    except ImportError:
        with tempfile.TemporaryDirectory() as tmp:          # poppler fallback (Linux / WSL)
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", path, os.path.join(tmp, "p")], check=True)
            return [Image.open(os.path.join(tmp, f)).convert("RGB") for f in sorted(os.listdir(tmp))]


CIRCUIT_MIN_PARTS = 2                  # a figure the circuit reader finds this many parts in is a circuit


def merge_boxes(boxes, gap):
    """boxes closer than gap to each other -> one box around them (a drawing whose wires break into pieces)"""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def find_circuits(img, reader):
    """the circuit drawings on a page, found by the circuit detector itself (a drawing's pieces would otherwise be
    taken for lines of text): [(x0, y0, x1, y1)] in img's pixels - two or more parts, the strokes that join them and
    the labels beside them"""
    import numpy as np
    from scipy import ndimage as ndi

    from recog.circuit_reader import LONG_SIDE
    from recog.circuit_wiring import strokes as classical
    from recog.strokes import scale_to
    small = scale_to(img.convert("RGB"), LONG_SIDE)
    s = LONG_SIDE / max(img.size)
    found = reader.detector(small)
    parts = [b for c, b, _ in found if c != "text"]
    if len(parts) < CIRCUIT_MIN_PARTS:
        return []
    mask = reader.strokes(small) if reader.strokes else classical(small)
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))
    g = 0.015 * LONG_SIDE
    keep = set()
    for x0, y0, x1, y1 in parts:                         # the strokes that reach a part: wires, the part itself
        sub = lab[max(0, int(y0 - g)):int(y1 + g), max(0, int(x0 - g)):int(x1 + g)]
        keep |= set(np.unique(sub[sub > 0]).tolist())
    region = np.isin(lab, sorted(keep))
    for x0, y0, x1, y1 in parts:
        region[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
    grown = ndi.distance_transform_edt(~region) <= 0.06 * LONG_SIDE     # a broken wire: still one drawing
    rl, n = ndi.label(grown, structure=np.ones((3, 3)))
    out = []
    for k, o in enumerate(ndi.find_objects(rl), 1):
        ys, xs = np.nonzero(region[o] & (rl[o] == k))         # the drawing itself, not the margin it was grouped by
        if not len(xs):
            continue
        y0, y1, x0, x1 = o[0].start + ys.min(), o[0].start + ys.max() + 1, o[1].start + xs.min(), o[1].start + xs.max() + 1
        inside = [b for b in parts if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1]
        if len(inside) < CIRCUIT_MIN_PARTS:
            continue
        for c, (a, b, cc, d), _ in found:                 # its labels: text boxes touching the region
            if c == "text" and a < x1 and cc > x0 and b < y1 and d > y0:
                x0, y0, x1, y1 = min(x0, a), min(y0, b), max(x1, cc), max(y1, d)
        out.append(tuple(int(round(v / s)) for v in (x0, y0, x1, y1)))
    return out


def read_circuits(img, figures, reader):
    """the figures that are circuits: [{"box", "netlist", "found": [(cls, box, score)] in page coordinates}];
    figures close together are read as one drawing"""
    out = []
    for x0, y0, x1, y1 in merge_boxes(figures, 0.03 * max(img.size)):
        m = max(20, int(0.08 * max(x1 - x0, y1 - y0)))
        bx = (max(0, x0 - m), max(0, y0 - m), min(img.width, x1 + m), min(img.height, y1 + m))
        try:
            net, found = reader.read(img.crop(bx))
        except Exception as e:                       # noqa: BLE001  (a reader failure leaves the figure to a human)
            print(f"circuit reader failed on the figure at {bx}: {e}")
            continue
        if sum(c.type != "other" for c in net.components) >= CIRCUIT_MIN_PARTS:
            from recog.circuit_reader import shift_notes
            shift_notes(net, 1.0, bx[0], bx[1])          # a doubtful spot the wiring noted, in page coordinates
            out.append({"box": (x0, y0, x1, y1), "netlist": net,
                        "found": [(c, (b[0] + bx[0], b[1] + bx[1], b[2] + bx[0], b[3] + bx[1]), sc) for c, b, sc in found]})
    return out


def read_pages(path: str, dpi: int = 150, reader=None, photo: bool = False, scan: bool = False,
               language: str = "hebrew", circuits: bool | None = None, progress=None):
    """-> per page {"page": cleaned page image, "info": {...}, "lines": [[Segment, ...] per line,
    boxes in page coordinates], "figures": [(x0, y0, x1, y1)], "circuits": [...] (read_circuits)}. A segment
    written inside a frame the student drew (info["answer_boxes"]) gets .boxed = True - most likely a final answer.
    language: "hebrew" (Hebrew, English and math) or "english" (an exam in English: English and math)
    circuits: read the figures that are circuits (recog/circuit_reader.py) - None: when its detector is trained
    progress: a function given a line of text as the reading goes (each page's layout, the heavy readers)"""
    from recog.mixed_reader import MixedReader
    from recog.page import clean_scan, find_lines, flatten_photo, looks_like_photo
    reader = reader or MixedReader(language=language)
    # heavy readers (Hebrew in context, Uni-MuMER, an English VLM - opt-in) read after the whole document's
    # layout, each over all its pieces, one model on the GPU at a time (reader.finish())
    defer = hasattr(reader, "finish")
    if defer:
        reader.defer_heavy = True
    if circuits is None:
        from recog.circuit_reader import DETECTOR_DIR
        circuits = os.path.isfile(os.path.join(DETECTOR_DIR, "config.json"))
    cr = None
    out = []
    say = progress or (lambda msg: None)
    imgs = load_pages(path, dpi)
    say(f"{len(imgs)} pages - their layout and first readings")
    for pno, img in enumerate(imgs, 1):
        flat = None
        if photo or (not scan and looks_like_photo(img)):
            img, flat = flatten_photo(img)
        page, info = clean_scan(img)
        if flat:
            info = {**info, "photo": {k: v for k, v in flat.items() if k != "photo"}}
        lines = []
        regions = []
        if circuits:                                    # circuit drawings first: their area is not text
            if cr is None:
                from recog.circuit_reader import CircuitReader
                cr = CircuitReader()
            regions = find_circuits(img, cr)
            if regions:
                blank = page.copy()
                for x0, y0, x1, y1 in regions:
                    blank.paste(255, (x0, y0, x1, y1))
                page_for_lines = blank
            else:
                page_for_lines = page
        else:
            page_for_lines = page
        found, figures = find_lines(page_for_lines, return_figures=True)
        for y0, _, line_img in found:
            for segs in reader.read(line_img, bands=[(0, line_img.height)])[0]:     # one line: keep it whole
                for s in segs:                              # back to page coordinates
                    s.box = (s.box[0], s.box[1] + y0, s.box[2], s.box[3] + y0)
                    s.boxed = in_box(s.box, info.get("answer_boxes", []))
                lines.append(segs)
        circs = read_circuits(img, regions + list(figures), cr) if circuits and (regions or figures) else []
        out.append({"page": page, "info": info, "lines": lines, "figures": figures, "circuits": circs})
        say(f"page {pno}/{len(imgs)}: {len(lines)} lines" + (" (blank)" if not lines else ""))
    if defer:
        try:
            reader.finish(progress=progress)
        except TypeError:                               # (a reader whose finish() takes no progress)
            reader.finish()
        reader.defer_heavy = False
    return out


def in_box(box, frames) -> bool:
    """most of the segment lies inside one of the frames"""
    x0, y0, x1, y1 = box
    area = max(1, (x1 - x0) * (y1 - y0))
    for a, b, c, d in frames:
        ov = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
        if ov >= 0.6 * area:
            return True
    return False


def font(size):
    for f in (["C:/Windows/Fonts/arial.ttf"] if os.name == "nt" else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--photo", action="store_true", help="treat as a phone photo even if it looks like a scan")
    ap.add_argument("--scan", action="store_true", help="treat as a scan even if it looks like a photo")
    ap.add_argument("--language", choices=("hebrew", "english"), default="hebrew",
                    help="the exam's language: hebrew (Hebrew + English + math, default) or english (English + math)")
    ap.add_argument("--check", action="store_true", help="also check the derivations on the page against themselves: "
                    "formula + the data written -> is the written result right? (grader/selfcheck.py, no key needed)")
    ap.add_argument("--values", default=None, help="with --check: a JSON file of the problem's data, by name "
                    "({\"R_1\": \"2k\", \"E\": \"10\"}) - a circuit drawing that only names its parts is judged with them, "
                    "and so is a derivation that uses a value the student does not write (the exam's own)")
    args = ap.parse_args()
    values = json.load(open(args.values, encoding="utf-8")) if args.values else None

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.path))[0]
    for pno, res in enumerate(read_pages(args.path, args.dpi, photo=args.photo, scan=args.scan,
                                                language=args.language), 1):
        out_png = os.path.join(args.out, f"{name}_p{pno}_read.png")
        txt = write_page(res, pno, out_png, language=args.language, check=args.check, values=values)
        print("\n".join(txt))
        print("wrote", os.path.relpath(out_png, HERE))


COLORS = {"hebrew": (40, 110, 220), "math": (230, 120, 20), "english": (30, 150, 70)}


def _draw_text(d, xy, label, f, fill, rtl):
    try:
        d.text(xy, label, font=f, fill=fill, anchor="ra", direction="rtl" if rtl else None)
    except (KeyError, ValueError, OSError):            # Pillow without libraqm: no text direction
        d.text(xy, label[::-1] if rtl else label, font=f, fill=fill, anchor="ra")


def _text_width(d, label, f, rtl):
    try:
        return d.textlength(label, font=f, direction="rtl" if rtl else None)
    except (KeyError, ValueError, OSError):
        return d.textlength(label, font=f)


def mark_checks(d, checks, lines):
    """each check's verdict drawn beside the page line it is about, with its number in the list: a green tick (right,
    sure), a red cross (wrong, sure), an orange '?' (a suggestion - a human should look)"""
    by_line = {}
    for k, c in enumerate(checks, 1):
        if 0 <= getattr(c, "line", -1) < len(lines) and lines[c.line]:
            by_line.setdefault(c.line, []).append((k, c))
    f = font(13)
    for ln, cs in by_line.items():
        boxes = [s.box for s in lines[ln]]
        x, y = max(14, min(b[0] for b in boxes) - 26), (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2
        if any(c.verdict not in ("right", "wrong", "wrong step") or not c.sure for _, c in cs):
            col, kind = (230, 130, 0), "?"
        elif any(c.verdict in ("wrong", "wrong step") for _, c in cs):     # a false step: as wrong as a false result
            col, kind = (220, 0, 0), "x"
        else:
            col, kind = (0, 160, 0), "v"
        if kind == "v":
            d.line([(x - 9, y), (x - 3, y + 7), (x + 9, y - 9)], fill=col, width=4)
        elif kind == "x":
            d.line([(x - 8, y - 8), (x + 8, y + 8)], fill=col, width=4)
            d.line([(x - 8, y + 8), (x + 8, y - 8)], fill=col, width=4)
        else:
            d.text((x - 6, y - 12), "?", font=font(22), fill=col)
        d.text((x - 10, y + 11), ",".join(str(k) for k, _ in cs[:4]), font=f, fill=col)


def write_page(res, pno: int, out_png: str, language: str = "hebrew", check: bool = False,
               values: dict | None = None) -> list[str]:
    """draw one read page (boxes + readings) to out_png and its text to out_png's .txt; -> the text lines.
    check: + the derivation checks (grader/selfcheck.py); values: the problem's data for its circuits"""
    from demo_mixed import pretty, readings
    page, info, lines, figures = res["page"], res["info"], res["lines"], res["figures"]
    txt = [f"page {pno}: {info}, {len(lines)} lines, {len(figures)} figure(s) not read as text"]
    panel_w = 900
    canvas = Image.new("RGB", (page.width + panel_w, page.height), "white")
    canvas.paste(page.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(canvas)
    f = font(16)
    circuit_boxes = [tuple(c["box"]) for c in res.get("circuits", [])]
    for x0, y0, x1, y1 in figures:                     # drawings / signatures: not text -> human
        if any(a <= x0 and b <= y0 and x1 <= c and y1 <= d for a, b, c, d in circuit_boxes):
            continue
        d.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline=(140, 140, 140), width=2)
        d.text((x0, y1 + 4), "figure - not read", font=f, fill=(120, 120, 120))
        txt.append(f"figure at x={x0}-{x1}, y={y0}-{y1}: not read as text")
    for k, c in enumerate(res.get("circuits", []), 1):  # circuits: the parts found, the netlist
        net = c["netlist"]
        x0, y0, x1, y1 = c["box"]
        d.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline=(150, 0, 170), width=2)
        for cls, (a, b, cc, dd), sc in c["found"]:
            d.rectangle((a, b, cc, dd), outline=(150, 0, 170), width=1)
            d.text((a, dd + 1), cls, font=font(11), fill=(150, 0, 170))
        d.text((x0, y1 + 4), f"circuit {k}: {len(net.components)} parts", font=f, fill=(150, 0, 170))
        from recog.circuit_reader import noted_spots
        for sx, sy in noted_spots(net):                  # what looks like a part's symbol left in the wires
            d.ellipse((sx - 14, sy - 14, sx + 14, sy + 14), outline=(230, 0, 0), width=3)
        txt.append(f"circuit {k} at x={x0}-{x1}, y={y0}-{y1}:")
        for comp in net.components:
            txt.append(f"  {comp.id} {comp.type} {comp.nodes}" + (f" + at {comp.plus}" if comp.plus else "")
                       + (f" '{comp.label}'" if comp.label else "") + (f" = {comp.value}" if comp.value else ""))
        for cur in net.currents:
            txt.append(f"  current {cur.label} through {cur.through}: {cur.frm} -> {cur.to}")
        for vol in net.voltages:
            txt.append(f"  voltage {vol.label}: + at {vol.plus}, - at {vol.minus}" + (f" (across {vol.across})" if vol.across else ""))
        if net.node_labels:
            txt.append(f"  nodes named {net.node_labels}")
        for node, eq in net.kcl().items():
            txt.append(f"  Kirchhoff at {node}: {eq}")
        for prob in net.problems():
            txt.append(f"  to check: {prob}")
    for x0, y0, x1, y1 in info.get("answer_boxes", []):  # frames the student drew (removed from the page)
        d.rectangle((x0, y0, x1, y1), outline=(200, 0, 160), width=1)
    for k, segs in enumerate(lines, 1):
        parts = []
        for s in segs:
            x0, y0, x1, y1 = s.box
            d.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=COLORS[s.kind], width=2)
            t0, p0 = readings(s.nbest)[0]
            shown = t0 if s.kind != "math" else "$" + pretty(t0) + "$"
            if s.kind == "math" and getattr(s, "math2", None):          # the second formula reader (Uni-MuMER)
                t2 = s.math2[0][0]
                if t2.replace(" ", "") != t0.replace(" ", ""):
                    shown += " / 2nd reader: $" + pretty(t2) + "$"
            if "unreadable" in getattr(s, "note", ""):
                shown = "[unreadable]"
            parts.append((shown + (" [boxed]" if getattr(s, "boxed", False) else ""), p0, s.kind, s.p_kind))
        txt.append(f"line {k}: " + "  |  ".join(f"[{kind} {pk:.0%}] {sh} ({p:.0%})" for sh, p, kind, pk in parts))
        y = segs[0].box[1]
        x = canvas.width - 10                          # right-to-left, like the page
        for sh, p, kind, _ in (parts if language == "hebrew" else parts[::-1]):
            label = f"{sh} ({p:.0%})"
            w = _text_width(d, label, f, kind == "hebrew")
            if x - w < page.width + 10:                # wrap inside the panel
                x, y = canvas.width - 10, y + 20
            _draw_text(d, (x, y), label, f, COLORS[kind], kind == "hebrew")
            x -= w + 18
    if check:
        from grader.selfcheck import check_pages
        checks = check_pages([res], values=values)
        txt.append(f"derivation checks (no key - the student's own formula and data): {len(checks)}")
        for k, c in enumerate(checks, 1):
            txt.append(f"  [{k}] {c.problem if isinstance(c.problem, str) else 'problem %d' % c.problem}: {c.verdict.upper()} - {c.text} "
                       f"(reading {c.confidence:.0%}{'' if c.sure else ', unsure: a human should look'})")
        mark_checks(d, checks, lines)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    canvas.save(out_png)
    with open(out_png.replace(".png", ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    return txt


if __name__ == "__main__":
    main()
