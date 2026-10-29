"""One exam read by several processes, each its share of the pages (tools/read_exams.py --pages a-b -> <n>.p<a>-<b>.json):
join the shares into <n>.json - every page from the share that read it.

    python tools/join_pages.py <out folder> <n>"""
import glob
import json
import os
import re
import sys


def join(parts: list) -> list:
    """the shares' page lists (each as long as the exam, the pages it did not read {"skipped": true}) -> one list;
    a page no share read, or read by two, is an error - but one every share left out on purpose (read_exams
    --skip-pages: {"left_out": true}) stays an empty page {"page", "left_out": true, "lines": []}"""
    n = {len(p) for p in parts}
    if len(n) != 1:
        raise ValueError(f"shares of different lengths: {sorted(n)}")
    out = []
    for i in range(n.pop()):
        got = [p[i] for p in parts if not p[i].get("skipped")]
        if not got and all(p[i].get("left_out") for p in parts):
            out.append({"page": i + 1, "left_out": True, "info": {}, "figures": [], "lines": []})
            continue
        if len(got) != 1:
            raise ValueError(f"page {i + 1}: read by {len(got)} shares")
        out.append(got[0])
    return out


def main():
    out, name = sys.argv[1:3]
    files = sorted(glob.glob(os.path.join(out, f"{name}.p*-*.json")),
                   key=lambda f: int(re.search(r"\.p(\d+)-", f).group(1)))
    data = join([json.load(open(f, encoding="utf-8")) for f in files])
    dst = os.path.join(out, f"{name}.json")
    json.dump(data, open(dst + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(dst + ".tmp", dst)
    print(f"{name}: {len(data)} pages from {len(files)} shares -> {dst}")


if __name__ == "__main__":
    main()
