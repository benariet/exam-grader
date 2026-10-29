"""Find what the students crossed out on read exams (recog/crossed.py) and mark the reading pieces under it - fully
local, from the scans and the saved readings:

    python tools/find_crossed.py <exam.pdf> <its json> [--pages 9,15] [--png <folder>] [--write <out.json>]

Prints, per page, each crossed-out region (kind, place as page fractions) and the pieces under it with their reading;
--png draws the regions on the pages (to look at on this computer); --write saves a copy of the json in which each
piece under crossing-out has "crossed": {"kind", "share"} (tools/recheck_exams.py leaves those out of the checks)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FULL = 0.6              # a piece this much under crossing-out is crossed out (less: partly - noted, still read)


def page_inks(pdf: str, data, dpi: int, pages=None) -> dict:
    """{page number: (the cleaned page's ink, letter height)} as the reading saw them (photos flattened the same way)"""
    import numpy as np
    from read_page import load_pages
    from recog.page import clean_scan, flatten_photo, letter_height
    imgs = load_pages(pdf, dpi)
    out = {}
    for pno, img in enumerate(imgs, 1):
        if pages and pno not in pages:
            continue
        if pno <= len(data) and data[pno - 1].get("info", {}).get("photo"):
            img, _ = flatten_photo(img)
        page, _ = clean_scan(img)
        ink = np.asarray(page) < 128
        out[pno] = (ink, letter_height(ink))
    return out


def reading(s) -> str:
    return (s.get("math2") or s.get("nbest") or [["", 0]])[0][0]


def handwriting(ink, booklet: str, pno: int):
    """the page's ink without what is printed there (the booklet's template of the page laid on it, recog/printed.py) -
    unchanged without a template"""
    if not booklet:
        return ink
    from recog.printed import printed_on
    from tools.build_booklet import load
    t = load(booklet, pno)
    return ink if t is None else ink & ~printed_on(t, ink)


def mark(data, inks: dict, debug: bool = False, booklet: str = "") -> dict:
    """each piece under crossing-out gets "crossed": {"kind", "share"} -> {page: regions}; booklet: the printed
    templates' folder (tools/build_booklet.py) - only the handwriting looked at"""
    from recog.crossed import covered, crossed_regions
    found = {}
    from recog.page import letter_height
    for pno, (ink, lh) in inks.items():
        if booklet:
            ink = handwriting(ink, booklet, pno)
            if ink.sum() > 2000:                         # (the handwriting's letters: bigger than the print's)
                lh = letter_height(ink)
        inks[pno] = (ink, lh)
        regions = crossed_regions(ink, lh, debug=debug)
        found[pno] = regions
        for segs in data[pno - 1]["lines"]:
            for s in segs:
                s.pop("crossed", None)
                c = covered(s["box"], regions, ink)
                if c:
                    s["crossed"] = c
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("json")
    ap.add_argument("--pages", default="", help="e.g. 9,15 (default: all)")
    ap.add_argument("--dpi", type=int, default=150, help="the dpi the saved reading was made at")
    ap.add_argument("--png", default="", help="a folder: the pages drawn with the regions found")
    ap.add_argument("--write", default="", help="a copy of the json with the crossed pieces marked")
    ap.add_argument("--debug", action="store_true", help="every long line looked at too (kind 'line')")
    ap.add_argument("--booklet", default="", help="the printed pages' templates (tools/build_booklet.py): left out")
    args = ap.parse_args()
    data = json.load(open(args.json, encoding="utf-8"))
    pages = set()
    for p in args.pages.split(","):
        if "-" in p:
            a, b = (int(v) for v in p.split("-"))
            pages |= set(range(a, b + 1))
        elif p.strip():
            pages.add(int(p))
    pages = pages or None
    inks = page_inks(args.pdf, data, args.dpi, pages)
    found = mark(data, inks, args.debug, args.booklet)
    n_full = n_part = 0
    for pno in sorted(found):
        ink, lh = inks[pno]
        H, W = ink.shape
        regions = found[pno]
        if not regions:
            continue
        print(f"page {pno} (letter height {lh:.0f} px): {len(regions)} region(s)")
        for r in regions:
            x0, y0, x1, y1 = r["box"]
            extra = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k not in ("kind", "box", "line")}
            print(f"  {r['kind']:8s} x {x0 / W:.2f}-{x1 / W:.2f} y {y0 / H:.2f}-{y1 / H:.2f} {extra}")
        for segs in data[pno - 1]["lines"]:
            for s in segs:
                if args.debug:
                    bx = s["box"]
                    print(f"      piece x {bx[0] / W:.2f}-{bx[2] / W:.2f} y {bx[1] / H:.2f}-{bx[3] / H:.2f} {s['kind'][:4]} {reading(s)[:50]}")
                c = s.get("crossed")
                if c:
                    full = c["share"] >= FULL
                    n_full += full
                    n_part += not full
                    print(f"    {'CROSSED' if full else 'part   '} {c['kind']:8s} {c['share']:.2f} {s['kind']:6s} {reading(s)[:70]}")
        if args.png:
            from PIL import Image, ImageDraw
            os.makedirs(args.png, exist_ok=True)
            img = Image.fromarray(((~ink) * 255).astype("uint8")).convert("RGB")
            d = ImageDraw.Draw(img)
            color = {"scribble": (230, 0, 0), "strike": (0, 120, 255), "hatch": (0, 170, 0)}
            for r in regions:
                d.rectangle(r["box"], outline=color.get(r["kind"], (128, 128, 128)), width=3)
                if "line" in r:
                    d.line(r["line"], fill=color.get(r["kind"], (128, 128, 128)), width=2)
            img.save(os.path.join(args.png, f"p{pno}_crossed.png"))
    print(f"pieces crossed out: {n_full}, partly: {n_part}")
    if args.write:
        json.dump(data, open(args.write, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
