"""Dump per-chunk signals (classifiers + all three readers) for a mixed evaluation folder, so the
Hebrew / English / math decision rules can be tuned offline (on development sets only).

    python tools_extract_chunks.py mixed_dev_en out.json
    python tools_extract_chunks.py mixed_dev_en out_0.json 0/2     # every 2nd line from the 1st: run
    python tools_extract_chunks.py mixed_dev_en out_1.json 1/2     # parts in parallel, then concatenate
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from recog.english import english_score, looks_english  # noqa: E402
from recog.mixed_reader import MixedReader, _acoustic, math_plausible  # noqa: E402
from recog.segment import pad_crop, segment_chunks, segment_lines, smooth_line  # noqa: E402

data, out_path = sys.argv[1], sys.argv[2]
D = os.path.join(HERE, "data", data)
truth = json.load(open(os.path.join(D, "truth.json"), encoding="utf-8"))
if len(sys.argv) > 3:                                    # "i/n": only every n-th line, starting at i
    i, n = map(int, sys.argv[3].split("/"))
    truth = truth[i::n]
r = MixedReader()
rows = []
for it in truth:
    page = Image.open(os.path.join(D, it["file"])).convert("L")
    for y0, y1 in segment_lines(page):
        line = page.crop((0, y0, page.width, y1))
        boxes = segment_chunks(line)
        raw = [r.clf.p_math(line.crop((a, 0, b, line.height))) for a, b in boxes]
        sm = smooth_line(raw)
        ink = np.asarray(line) < 128
        ys = np.where(ink.any(1))[0]
        ink_h = int(ys.max() - ys.min() + 1) if len(ys) else line.height
        for (a, b), p, q in zip(boxes, raw, sm):
            cover = {}
            for s in it["segments"]:
                ov = max(0, min(b, s["x"][1]) - max(a, s["x"][0]))
                cover[s["kind"]] = cover.get(s["kind"], 0) + ov
            main = max(cover, key=cover.get) if cover else "none"
            kind_true = main if cover and cover[main] >= 0.6 * (b - a) else "mixed"
            crop = pad_crop(line, (a, 0, b, line.height))
            f = r.frm.recognize(crop, n_best=1)[0]
            h = r.heb.recognize(crop, n_best=1)[0]
            e = r.eng.recognize(crop, n_best=1)[0]
            rows.append({
                "file": it["file"], "x": [int(a), int(b)], "width": int(b - a), "ink_h": ink_h,
                "ink_px": int(ink[:, a:b].sum()), "true": kind_true, "p_raw": float(p), "p_smooth": float(q),
                "p_en": float(r.en_clf.p_math(line.crop((a, 0, b, line.height)))) if r.en_clf else None,
                "f_text": f.text, "f_conf": math.exp(f.logp / max(1, len(f.text.split()))), "f_min": f.min_token_p,
                "f_plausible": math_plausible(f.text),
                "h_text": h.text, "h_conf": math.exp(_acoustic(h) / max(1, len(h.text))),
                "e_text": e.text, "e_conf": math.exp(e.logp / max(1, len(e.text))), "e_min": e.min_token_p,
                "e_looks": looks_english(e.text), "e_looks_short": looks_english(e.text, allow_short=True),
                "e_score": english_score(e.text)})
json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(data, len(rows), "chunks ->", out_path)
