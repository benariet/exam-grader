"""Render MathWriting InkML (online pen strokes) to an image - used to build math training crops."""
import re

import numpy as np
from PIL import Image, ImageDraw

_trace = re.compile(r"<trace(?:\s[^>]*)?>(.*?)</trace>", re.S)


def read_traces(path):
    txt = open(path, encoding="utf-8").read()
    strokes = []
    for t in _trace.findall(txt):
        pts = []
        for p in t.strip().split(","):
            v = p.split()
            if len(v) >= 2:
                pts.append((float(v[0]), float(v[1])))
        if pts:
            strokes.append(np.array(pts))
    return strokes


def render(path, height=44, pen=3, pad=4):
    strokes = read_traces(path)
    allp = np.concatenate(strokes)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    s = (height - 2 * pad) / max(1e-6, (y1 - y0))
    W = int((x1 - x0) * s) + 2 * pad + 1
    img = Image.new("L", (W, height), 255)
    d = ImageDraw.Draw(img)
    for st in strokes:
        pts = [((x - x0) * s + pad, (y - y0) * s + pad) for x, y in st]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse((x - pen / 2, y - pen / 2, x + pen / 2, y + pen / 2), fill=0)
        else:
            d.line(pts, fill=0, width=pen, joint="curve")
            for x, y in (pts[0], pts[-1]):
                d.ellipse((x - pen / 2 + 0.5, y - pen / 2 + 0.5, x + pen / 2 - 0.5, y + pen / 2 - 0.5), fill=0)
    return img
