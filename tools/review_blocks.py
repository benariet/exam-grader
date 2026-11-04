"""A local page to look at how each answer was read as a block (tools/read_blocks.py): every page's scan beside its
transcription, line by line - opened from disk, nothing leaves the computer.

    python tools/review_blocks.py <exam.pdf> <blocks.json> --out <folder>/<n>_blocks.html [--dpi 110]
        [--pieces <the same exam read piece by piece: read_exams.py's / the merged json>]

Each strip the model read is outlined on the scan with its number, and its lines are listed under the same number;
a line the checks can use (it has math) is marked; --pieces adds a third column: what the piece-by-piece reading read
on the page (its math pieces, top to bottom), to compare."""
import argparse
import base64
import html
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PAGE = """<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Answers read as blocks</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#e3e3e8;--accent:#2a6edb;--math:#0b7a3e}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--card:#2c2c2e;--line:#3a3a3c;--accent:#5b9bff;--math:#4cc38a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 12px}
.page{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px;margin:0 0 16px}
.scan img{width:100%;background:#fff;border-radius:4px}
.meta{color:var(--muted);font-size:13px;margin-bottom:6px}
.strip{margin:8px 0;padding-top:6px;border-top:1px solid var(--line)}
.strip b{color:var(--accent)}
.ln{padding:2px 0;white-space:pre-wrap;word-break:break-word}
.ln.m{direction:ltr;text-align:left;font-family:ui-monospace,monospace;font-size:13px;color:var(--math)}
</style></head><body><h1>__TITLE__</h1>__BODY__</body></html>"""


def kind(line: str, math_of) -> str:
    """a transcribed line's look: math alone - left to right, in the math colour; with Hebrew - as text"""
    hebrew = any(0x590 <= ord(c) <= 0x5FF for c in line)
    return "ln m" if math_of(line) and not hebrew else "ln"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("blocks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--read-dpi", type=int, default=150)
    ap.add_argument("--pieces", default="", help="the same exam read piece by piece (a third column)")
    args = ap.parse_args()
    pieces = json.load(open(args.pieces, encoding="utf-8")) if args.pieces else None
    from PIL import ImageDraw, ImageFont
    from read_page import load_pages
    from tools.read_blocks import math_of
    blocks = json.load(open(args.blocks, encoding="utf-8"))
    imgs = load_pages(args.pdf, args.dpi)
    f = args.dpi / args.read_dpi
    body = []
    for b in blocks:
        if not b["strips"]:
            continue
        img = imgs[b["page"] - 1].convert("RGB")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=22)
        except TypeError:                                   # (an older Pillow)
            font = ImageFont.load_default()
        right = []
        for k, s in enumerate(b["strips"], 1):
            x0, y0, x1, y1 = (v * f for v in s["box"])
            d.rectangle((x0, y0, x1, y1), outline=(42, 110, 219), width=3)
            d.text((x0 + 4, y0 + 2), str(k), fill=(42, 110, 219), font=font)
            lines = "".join(f'<div class="{kind(ln, math_of)}">{html.escape(ln)}</div>' for ln in s["text"].splitlines())
            right.append(f'<div class="strip"><b>{k}</b>{lines}</div>')
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        third = ""
        if pieces and b["page"] <= len(pieces):
            segs = sorted((s for row in pieces[b["page"] - 1]["lines"] for s in row if s["kind"] == "math"),
                          key=lambda s: (s["box"][1], s["box"][0]))
            got = [(s.get("math2") or s["nbest"])[0][0] for s in segs]
            third = (f'<div><div class="meta">בשיטת הקטעים ({len(got)} קטעי מתמטיקה)</div>'
                     + "".join(f'<div class="ln m">{html.escape(t)}</div>' for t in got) + '</div>')
        body.append(f'<section class="page"><div class="scan"><div class="meta">עמוד {b["page"]}</div>'
                    f'<img src="{src}" alt="עמוד {b["page"]}"></div><div><div class="meta">כבלוק אחד '
                    f'({len(b["lines"])} שורות)</div>{"".join(right)}</div>{third}</section>')
    title = f"מבחן {os.path.splitext(os.path.basename(args.pdf))[0]} - כל תשובה נקראה כבלוק אחד"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(PAGE.replace("__TITLE__", html.escape(title)).replace("__BODY__", "".join(body)))
    print(f"{len(body)} pages -> {args.out}")


if __name__ == "__main__":
    main()
