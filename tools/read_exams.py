"""Read students' scanned exams - fully local, on this computer (its GPU when it has one): every page of every
<n>.pdf in a folder (not <n>_checked.pdf, the copy the lecturer marked), drawn with its readings and derivation checks
(PNG + TXT per page, as read_page.py) and its readings saved for the checker to judge again without reading again
(<out>/<n>.json: per page, per line, each segment's kind, box, p_kind and n-best readings).

    python tools/read_exams.py "C:\\...\\Stud exam" --out "C:\\...\\grader_runs"

Resumable: a student whose JSON is there is skipped. Progress: <out>/progress.log. On several GPUs, one reader per GPU on
its share of the students: '--shard 0/2' and '--shard 1/2' (each with its own CUDA_VISIBLE_DEVICES - read_exams.cmd does it)"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _plain(v):
    """a page's info for JSON: numbers, strings, lists, dicts"""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, bool)) or v is None:
        return v
    try:
        return float(v) if not float(v).is_integer() else int(v)
    except (TypeError, ValueError):
        return str(v)


def _segment(s) -> dict:
    """a segment for JSON - with what the heavy readers added: the second formula reader's readings (math2: Uni-MuMER -
    the checks weigh them with the first reader's), the formula reader's look at a Hebrew piece (formula_nbest: a value
    the checks may miss), the reader's note, a frame around it (boxed: a final answer)"""
    d = {"kind": s.kind, "box": _plain(s.box), "p_kind": float(s.p_kind), "nbest": [[t, float(lp)] for t, lp in s.nbest]}
    for k in ("math2", "formula_nbest"):
        v = getattr(s, k, None)
        if v:
            d[k] = [[t, float(lp)] for t, lp in v]
    if getattr(s, "note", None):
        d["note"] = str(s.note)
    if getattr(s, "boxed", False):
        d["boxed"] = True
    return d


def _share(pdfs: list, shard: str) -> list:
    """'k/n': the k-th of n shares of the exams (every n-th, from the k-th, in the sorted order - the same whatever is done)"""
    if not shard:
        return pdfs
    k, n = (int(x) for x in shard.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"--shard {shard}: k must be 0..n-1")
    return [p for i, p in enumerate(pdfs) if i % n == k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", default="", help="comma-separated student numbers (the file names), e.g. 1,3,4")
    ap.add_argument("--shard", default="", help="k/n: this reader's share of the exams, e.g. 0/2 and 1/2 on two GPUs")
    args = ap.parse_args()
    from read_page import read_pages, write_page
    from recog.mixed_reader import MixedReader
    os.makedirs(os.path.join(args.out, "read"), exist_ok=True)
    log = open(os.path.join(args.out, "progress.log"), "a", encoding="utf-8")

    def say(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    pdfs = [p for p in glob.glob(os.path.join(args.folder, "*.pdf")) if not re.search(r"_checked\.pdf$", p, re.I)]
    pdfs.sort(key=lambda p: (int(re.sub(r"\D", "", os.path.basename(p)) or 0), p))
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    n_all, pdfs = len(pdfs), _share(pdfs, args.shard)
    todo = [p for p in pdfs if (not only or os.path.splitext(os.path.basename(p))[0] in only)
            and not os.path.isfile(os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".json"))]
    say(f"{n_all} exams in {args.folder}" + (f" - share {args.shard}: {len(pdfs)} of them" if args.shard else "") +
        f"; to read now: {len(todo)}" + (f" ({', '.join(os.path.splitext(os.path.basename(p))[0] for p in todo[:10])})" if todo else ""))
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none - on the CPU (slow)"
    except Exception as e:                           # noqa: BLE001
        gpu = f"unknown ({type(e).__name__})"
    say(f"GPU: {gpu}; Hebrew in context: {'on' if os.environ.get('EXAM_GRADER_HEBREW_CONTEXT') == '1' else 'off'}; "
        f"second formula reader: {os.environ.get('EXAM_GRADER_FORMULA2') or 'off'}")
    say("loading the readers...")
    t_load = time.time()
    reader = MixedReader(language="hebrew")
    say(f"readers ready in {time.time() - t_load:.0f} s")
    done_secs = []
    for pdf in pdfs:
        name = os.path.splitext(os.path.basename(pdf))[0]
        out_json = os.path.join(args.out, f"{name}.json")
        if (only and name not in only) or os.path.isfile(out_json):
            continue
        t0 = time.time()
        left = [p for p in todo if p != pdf and not os.path.isfile(os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + ".json"))]
        say(f"{name}: reading" + (f" (then {len(left)} more" + (f", about {sum(done_secs) / len(done_secs) * len(left) / 60:.0f} min"
                                                                    if done_secs else "") + ")" if left else ""))
        try:
            pages = read_pages(pdf, args.dpi, reader=reader, circuits=False, progress=lambda m: say(f"{name}: {m}"))
        except Exception as e:                       # noqa: BLE001 - one exam's failure: the next one
            say(f"{name}: FAILED {type(e).__name__}: {e}")
            continue
        data, n_checks = [], 0
        for pno, res in enumerate(pages, 1):
            try:
                txt = write_page(res, pno, os.path.join(args.out, "read", f"{name}_p{pno}.png"), check=True)
                n_checks += sum(1 for t in txt if t.startswith("  ") and ("right" in t or "wrong" in t))
            except Exception as e:                   # noqa: BLE001
                say(f"{name} page {pno}: drawing/checks failed {type(e).__name__}: {e}")
            if pno == 1 or pno % 5 == 0 or pno == len(pages):
                say(f"{name}: drawn and checked {pno}/{len(pages)} pages")
            data.append({"page": pno, "info": _plain(res["info"]), "figures": _plain(res.get("figures", [])),
                         "lines": [[_segment(s) for s in segs] for segs in res["lines"]]})
        json.dump(data, open(out_json + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(out_json + ".tmp", out_json)
        done_secs.append(time.time() - t0)
        say(f"{name}: DONE - {len(pages)} pages, {sum(len(p['lines']) for p in data)} lines, {round(time.time() - t0)} s; "
            f"saved {out_json}")
    say("done" + (f" (share {args.shard})" if args.shard else ""))


if __name__ == "__main__":
    main()
