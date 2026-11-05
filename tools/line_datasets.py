"""Published Hebrew handwriting lines -> line crops + labels.jsonl that tools/finetune_mishkefet.py reads:

    python tools/line_datasets.py parquet "data/hf/cyttic__trocr-hebrew-matan/data/train-*.parquet" \
        --out data/lines/matan --name matan_train --source hf:cyttic/trocr-hebrew-matan
    python tools/line_datasets.py page data/pinkas/pinkas_dataset_images_and_xmls --out data/lines/pinkas \
        --name pinkas_train --only data/pinkas/train_set.txt --source "zenodo:3569694 Pinkas"
    python tools/line_datasets.py heldout data/lines/matan/matan_test.jsonl --exclude data/lines/matan/matan_train.jsonl \
        --n 300 --out evalset/heldout_matan300.jsonl

parquet: a Hugging Face line dataset (columns image, text, and writer / writer_id / source_doc when there is one).
page: PAGE-XML pages (an image beside each .xml; each TextLine's box cropped, its TextEquiv the text; lines with an
editor's [..] mark or under 3 characters left out). heldout: n test lines spread over the writers, none whose text
the training lines have (the matan writers copy the same paragraphs: 433 of its 1455 test lines are train texts),
image paths made absolute so the file can live anywhere."""
import argparse
import collections
import glob
import io
import json
import os
import random
import re
import xml.etree.ElementTree as ET


def _tag(e) -> str:
    return e.tag.rsplit("}", 1)[-1]


def from_parquet(pattern: str, out: str, name: str, source: str = "") -> int:
    import pyarrow.parquet as pq
    from PIL import Image
    os.makedirs(os.path.join(out, "lines"), exist_ok=True)
    n = 0
    with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(pattern)):
            for r in pq.read_table(p).to_pylist():
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                rel = f"lines/{name}_{n:05d}.png"
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("L").save(os.path.join(out, rel))
                who = r.get("writer") or r.get("writer_id") or r.get("source_doc") or ""
                f.write(json.dumps({"image": rel, "text": text, "kind": "hebrew", "source": f"{source} {who}".strip()},
                                   ensure_ascii=False) + "\n")
                n += 1
    return n


def page_lines(xml_path: str) -> list:
    """[(box (x0, y0, x1, y1), text)] of a PAGE-XML page's text lines, the box padded by an eighth of its height"""
    out = []
    for tl in ET.parse(xml_path).getroot().iter():
        if _tag(tl) != "TextLine":
            continue
        coords = next((c for c in tl if _tag(c) == "Coords"), None)
        uni = [u for t in tl if _tag(t) == "TextEquiv" for u in t.iter() if _tag(u) == "Unicode"]
        text = (uni[0].text or "").strip() if uni else ""
        if coords is None or len(text) < 3 or re.search(r"[\[\]]", text):
            continue
        pts = [tuple(map(int, p.split(","))) for p in coords.get("points").split()]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        pad = max(4, (max(ys) - min(ys)) // 8)
        out.append(((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), text))
    return out


def from_page_xml(folder: str, out: str, name: str, source: str = "", only=None) -> int:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(os.path.join(out, "lines"), exist_ok=True)
    n = 0
    with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for x in sorted(glob.glob(os.path.join(folder, "*.xml"))):
            if only is not None and os.path.basename(x) not in only:
                continue
            stem = x[:-4]
            img = next((stem + e for e in (".jpg", ".png", ".tif", ".jpeg") if os.path.exists(stem + e)), None)
            if not img:
                continue
            page = Image.open(img).convert("L")
            for (x0, y0, x1, y1), text in page_lines(x):
                rel = f"lines/{name}_{n:05d}.png"
                page.crop((max(0, x0), max(0, y0), min(page.width, x1), min(page.height, y1))).save(
                    os.path.join(out, rel))
                f.write(json.dumps({"image": rel, "text": text, "kind": "hebrew",
                                    "source": f"{source} {os.path.basename(stem)}".strip()}, ensure_ascii=False) + "\n")
                n += 1
    return n


def heldout(test: str, exclude: list, n: int, seed: int = 0) -> list:
    """n rows of the test file, round-robin over its writers (the last word of 'source'), none whose text is in the
    exclude files; image paths absolute"""
    seen = {json.loads(r)["text"] for p in exclude for r in open(p, encoding="utf-8")}
    base = os.path.dirname(os.path.abspath(test))
    by = collections.defaultdict(list)
    for r in open(test, encoding="utf-8"):
        d = json.loads(r)
        if d["text"] not in seen:
            d["image"] = os.path.join(base, d["image"])
            by[(d.get("source") or "").split()[-1:][0] if d.get("source") else ""].append(d)
    rng, pick = random.Random(seed), []
    while len(pick) < n and any(by.values()):
        for w in sorted(by):
            if by[w] and len(pick) < n:
                pick.append(by[w].pop(rng.randrange(len(by[w]))))
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("parquet")
    a.add_argument("pattern")
    b = sub.add_parser("page")
    b.add_argument("folder")
    b.add_argument("--only", help="a file of the .xml names to take (a published train / test split)")
    for s in (a, b):
        s.add_argument("--out", required=True)
        s.add_argument("--name", required=True)
        s.add_argument("--source", default="")
    c = sub.add_parser("heldout")
    c.add_argument("test")
    c.add_argument("--exclude", nargs="*", default=[])
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "parquet":
        print(args.name, from_parquet(args.pattern, args.out, args.name, args.source), "lines")
    elif args.cmd == "page":
        only = ({ln.strip() for ln in open(args.only, encoding="utf-8") if ln.strip()}   # one per line ('Page 1.xml')
                if args.only else None)
        print(args.name, from_page_xml(args.folder, args.out, args.name, args.source, only), "lines")
    else:
        rows = heldout(args.test, args.exclude, args.n, args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(len(rows), "lines,", len({(r.get("source") or "").split()[-1:][0] for r in rows if r.get("source")}),
              "writers ->", args.out)


if __name__ == "__main__":
    main()
