"""The printed part of each page of an exam booklet (recog/printed.py), from many students' scans of it - run once per
exam, fully local; the template holds only what is the same in most copies (the booklet as printed), no one's writing:

    python tools/build_booklet.py <folder of scans> <out folder> [--exams 10,11,13,...] [--n 12] [--pages 1-15]
        [--load-only]

-> <out>/p<k>.png per page (black: printed) and <out>/booklet.json (the exams used, each page's printed share). The
cleaned scans are kept in <out>/cache/ (packed bits) so that the work can be done in parts: --load-only first (some
exams per run), then the pages (some per run)."""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def cached_inks(pdf: str, cache: str, dpi: int):
    """the exam's cleaned pages' ink [bool arrays], from the cache or read and cached"""
    import numpy as np
    f = os.path.join(cache, os.path.basename(pdf)[:-4] + ".npz")
    if os.path.isfile(f):
        z = np.load(f)
        return [np.unpackbits(z[f"p{k}"], count=int(np.prod(z[f"s{k}"]))).reshape(z[f"s{k}"]).astype(bool)
                for k in range(int(z["n"]))]
    from read_page import load_pages
    from recog.page import clean_scan
    inks = [np.asarray(clean_scan(img)[0]) < 128 for img in load_pages(pdf, dpi)]
    os.makedirs(cache, exist_ok=True)
    np.savez(f + ".tmp.npz", n=len(inks), **{f"p{k}": np.packbits(a) for k, a in enumerate(inks)},
             **{f"s{k}": np.array(a.shape) for k, a in enumerate(inks)})
    os.replace(f + ".tmp.npz", f)
    return inks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    ap.add_argument("--exams", default="", help="comma-separated exam numbers (default: the first --n by number)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--pages", default="", help="a-b: only these pages now (default: all)")
    ap.add_argument("--load-only", action="store_true", help="only read and cache the exams' pages")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    import numpy as np
    from PIL import Image
    from recog.printed import build_template
    pdfs = [p for p in glob.glob(os.path.join(args.folder, "*.pdf")) if re.fullmatch(r"\d+\.pdf", os.path.basename(p))]
    pdfs.sort(key=lambda p: int(os.path.basename(p)[:-4]))
    if args.exams:
        want = {x.strip() for x in args.exams.split(",")}
        pdfs = [p for p in pdfs if os.path.basename(p)[:-4] in want]
    pdfs = pdfs[: args.n]
    cache = os.path.join(args.out, "cache")
    print(f"{len(pdfs)} exams: {', '.join(os.path.basename(p)[:-4] for p in pdfs)}", flush=True)
    inks = []
    for pdf in pdfs:
        inks.append(cached_inks(pdf, cache, args.dpi))
        print(f"  {os.path.basename(pdf)}: {len(inks[-1])} pages", flush=True)
    if args.load_only:
        return
    n = min(len(x) for x in inks)
    ks = range(1, n + 1)
    if args.pages:
        a, b = (int(v) for v in args.pages.split("-"))
        ks = range(a, min(b, n) + 1)
    os.makedirs(args.out, exist_ok=True)
    jf = os.path.join(args.out, "booklet.json")
    info = json.load(open(jf)) if os.path.isfile(jf) else {"pages": {}}
    info.update({"exams": [os.path.basename(p)[:-4] for p in pdfs], "dpi": args.dpi, "n_pages": n})
    for k in ks:
        t = build_template([x[k - 1] for x in inks])
        Image.fromarray(np.where(t, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"p{k}.png"), optimize=True)
        info["pages"][str(k)] = {"printed_share": round(float(t.mean()), 4),
                                 "ink_share_mean": round(float(np.mean([x[k - 1].mean() for x in inks])), 4)}
        print(f"page {k}: printed {t.mean() * 100:.1f}% of the page (ink {info['pages'][str(k)]['ink_share_mean'] * 100:.1f}%)",
              flush=True)
    json.dump(info, open(jf, "w"), indent=1)


def load(out: str, page: int):
    """the template of a page (bool, True: printed) or None"""
    import numpy as np
    from PIL import Image
    f = os.path.join(out, f"p{page}.png")
    return np.asarray(Image.open(f).convert("L")) < 128 if os.path.isfile(f) else None


if __name__ == "__main__":
    main()
