"""The circuits students drew in their exams, read alone - no text reading (tools/read_exams.py reads the text, hours
an exam): each page cleaned, the booklet's print erased (the question's own printed circuit is not the student's),
the drawings the circuit detector finds read into netlists, saved for tools/circuit_key.py to judge against the
solution (<out>/<n>.circuits.json: per page, each drawing's box and netlist). Fully local, like the other readers.

    python tools/read_exam_circuits.py "C:\\...\\Stud exam" --out "C:\\...\\grader_runs\\circuits" --booklet booklet/moedC
        [--only 1,3] [--pages 7-9]

Resumable: a student whose JSON is there is skipped."""
import argparse
import dataclasses
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def netlist_dict(net) -> dict:
    """a Netlist as plain JSON (grader.circuit.Netlist.from_dict reads it back)"""
    from tools.read_exams import _plain
    return _plain(dataclasses.asdict(net))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--booklet", default="", help="the printed booklet's templates (tools/build_booklet.py): the print erased")
    ap.add_argument("--only", default="", help="comma-separated student numbers (the file names)")
    ap.add_argument("--pages", default="", help="a-b: only these pages of each exam")
    ap.add_argument("--letters", default="", help="the letters the questions name ('ABCS'): a letter read as none of "
                    "them ('3' for a 'B') re-read as one")
    args = ap.parse_args()
    from read_page import booklet_pages, clean_scan, find_circuits, handwriting_only, has_writing, load_pages, read_circuits
    from recog.circuit_reader import CircuitReader
    os.makedirs(args.out, exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    rng = None
    if args.pages:
        a, b = (int(x) for x in args.pages.split("-"))
        rng = set(range(a, b + 1))
    pdfs = [p for p in glob.glob(os.path.join(args.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)]
    pdfs.sort(key=lambda p: (int(re.sub(r"\D", "", os.path.basename(p)) or 0), p))
    book = booklet_pages(args.booklet)
    cr = CircuitReader(letters=args.letters)
    t_all = time.time()
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        out = os.path.join(args.out, f"{name}.circuits.json")
        if (only and name not in only) or os.path.isfile(out):
            continue
        t0 = time.time()
        imgs = load_pages(pdf, args.dpi)
        use_book = book and book["n"] == len(imgs)
        pages = []
        for pno, img in enumerate(imgs, 1):
            if rng is not None and pno not in rng:
                continue
            page, info = clean_scan(img)
            rec = {"page": pno, "booklet": None, "circuits": []}
            if use_book and pno in book["pages"]:
                page, hand, binfo = handwriting_only(page, book["pages"][pno])
                rec["booklet_match"] = binfo.get("booklet_match")
                if hand is not None:
                    rec["booklet"] = pno
                    if not has_writing(hand):
                        rec["no_handwriting"] = True
                        pages.append(rec)
                        continue
            try:
                regions = find_circuits(page, cr)
                circs = read_circuits(page, regions, cr) if regions else []
            except Exception as e:                   # noqa: BLE001 - one page's failure: the next one
                rec["failed"] = f"{type(e).__name__}: {e}"
                circs = []
            for c in circs:
                rec["circuits"].append({"box": [int(v) for v in c["box"]], "netlist": netlist_dict(c["netlist"]),
                                        "found": [[k, [round(float(v), 1) for v in b], round(float(s), 3)]
                                                  for k, b, s in c["found"]]})
            pages.append(rec)
        json.dump({"exam": name, "pages_in_scan": len(imgs), "booklet_used": bool(use_book), "pages": pages},
                  open(out + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(out + ".tmp", out)
        n_c = sum(len(p["circuits"]) for p in pages)
        print(f"{name}: {len(imgs)} pages, {n_c} drawings read, {time.time() - t0:.0f} s", flush=True)
    print(f"done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    main()
