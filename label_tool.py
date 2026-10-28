"""Turn your own scanned exam pages into Hebrew training data - quickly.

    python label_tool.py scans/*.png            # -> data/my_labels/label.html (open it in a browser)

1. every page is split into lines and chunks; Hebrew chunks are kept (formulas are skipped)
2. each Hebrew chunk is pre-filled with the current model's reading
3. chunks are sorted by the model's confidence, least sure first - correcting those teaches the
   model the most, and the confident ones can usually be skipped
4. "Save labels" downloads labels.json -> put it in data/my_labels/ and fine-tune:
       python train_hebrew.py --init models/hebrew-crnn-cpu2/model.pt --extra data/my_labels/labels.json
"""
import argparse
import base64
import glob
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402

from recog.hebrew import HebrewRecognizer  # noqa: E402
from recog.segment import ScriptClassifier, layout, pad_crop  # noqa: E402

PAGE = """<!doctype html><html lang="he"><head><meta charset="utf-8"><title>Label lines</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#e3e3e8;--accent:#2a6edb;--low:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--card:#2c2c2e;--line:#3a3a3c;--accent:#5b9bff;--low:#fb923c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0;flex:1}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}
.item img{max-width:100%;background:#fff;border-radius:4px;display:block;margin-inline-start:auto}
.row{display:flex;gap:10px;align-items:center;margin-top:8px}
input[type=text]{flex:1;font-size:18px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
.conf{font-size:12px;color:var(--muted);min-width:90px}.conf.low{color:var(--low);font-weight:600}
label{font-size:13px;color:var(--muted);white-space:nowrap}
</style></head><body>
<header><h1>Correct the readings (least confident first)</h1>
<span id="count" class="conf"></span><button onclick="save()">Save labels</button></header>
<main id="list"></main>
<script>
const ITEMS = __ITEMS__;
const list = document.getElementById('list');
ITEMS.forEach((it, i) => {
  const d = document.createElement('div'); d.className = 'item';
  d.innerHTML = `<img src="${it.img}" alt="line ${i+1}">
    <div class="row"><span class="conf ${it.conf < 0.8 ? 'low' : ''}">confidence ${(it.conf*100).toFixed(0)}%</span>
    <input type="text" dir="rtl" value="${it.text.replace(/"/g,'&quot;')}" data-i="${i}">
    <label><input type="checkbox" data-skip="${i}"> skip</label></div>`;
  list.appendChild(d);
});
document.getElementById('count').textContent = ITEMS.length + ' lines';
function save() {
  const out = [];
  document.querySelectorAll('input[type=text]').forEach(inp => {
    const i = +inp.dataset.i;
    if (document.querySelector(`[data-skip="${i}"]`).checked) return;
    out.push({file: ITEMS[i].file, text: inp.value.trim(), model_reading: ITEMS[i].text});
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = 'labels.json'; a.click();
}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+", help="scanned page images (png/jpg)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "my_labels"))
    args = ap.parse_args()
    paths = [p for pat in args.pages for p in sorted(glob.glob(pat))]
    os.makedirs(os.path.join(args.out, "lines"), exist_ok=True)
    clf, heb = ScriptClassifier(), HebrewRecognizer()
    items = []
    for pi, p in enumerate(paths):
        page = Image.open(p).convert("L")
        for li, line in enumerate(layout(page, clf)):
            for si, (kind, box, pk) in enumerate(line):
                if kind != "hebrew":
                    continue
                crop = pad_crop(page, box)
                name = f"p{pi:03d}_l{li:02d}_s{si}.png"
                crop.save(os.path.join(args.out, "lines", name))
                c = heb.recognize(crop, n_best=1)[0]
                ac = c.acoustic_logp if c.acoustic_logp is not None else c.logp     # image only, no word LM
                conf = math.exp(ac / max(1, len(c.text)))
                buf = io.BytesIO(); crop.save(buf, format="PNG")
                items.append({"file": os.path.join("lines", name), "text": c.text, "conf": round(conf, 3),
                              "img": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
    items.sort(key=lambda it: it["conf"])
    with open(os.path.join(args.out, "label.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
    print(f"{len(items)} Hebrew lines from {len(paths)} page(s) -> {os.path.join(args.out, 'label.html')}")


if __name__ == "__main__":
    main()
