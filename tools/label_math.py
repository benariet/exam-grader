"""Measure how well the formula readers read real exams: a sample of calculation lines (math pieces with '=' and a
digit, on the questions' pages) cut from the scans into ONE local HTML page, each beside what Qwen3-VL and Uni-MuMER
read; a person picks the right reading, corrects it, or marks the line unreadable, and saves the labels. Nothing
leaves the computer - the page holds the cut lines as pictures and is opened from disk.

    python tools/label_math.py --out <folder>/math_labels.html --pages <pages.json> --n 20 \
        --exam 3 <3.pdf> <merged 3.json> --exam 6 <6.pdf> <merged 6.json> ...

Then tools/score_math_labels.py <the saved labels.json> <merged jsons ...>: per reader, the lines read right."""
import argparse
import base64
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

CALC = re.compile(r"=")


def calculation(seg) -> bool:
    """a piece the checks would use: '=' and a digit in either reader's reading, more than a few characters"""
    v = (seg.get("math2_vlm") or seg.get("math2") or [["", 0]])[0][0]
    m = (seg.get("math2_mumer") or [["", 0]])[0][0]
    both = v + " " + m
    return bool(CALC.search(both) and re.search(r"\d", both) and len(re.sub(r"\s", "", v or m)) > 6)


def candidates(data, exam: str, qpages: dict) -> list:
    """-> [(exam, page, question, box, qwen reading, uni-mumer reading)] of the calculation pieces on question pages"""
    out = []
    for pno, page in enumerate(data, 1):
        q = (qpages.get(exam, {}).get(str(pno)) or {}).get("question")
        if q is None:
            continue
        for segs in page["lines"]:
            for s in segs:
                if s["kind"] == "math" and calculation(s):
                    out.append((exam, pno, q, s["box"], (s.get("math2_vlm") or s.get("math2") or [["", 0]])[0][0],
                                (s.get("math2_mumer") or [["", 0]])[0][0]))
    return out


def sample(cands: list, n: int, seed: int) -> list:
    """n of them at random (a fixed seed), spread over the questions in proportion, at least one from each"""
    rnd = random.Random(seed)
    by_q = {}
    for c in cands:
        by_q.setdefault(c[2], []).append(c)
    total = len(cands)
    out = []
    for q, cs in sorted(by_q.items()):
        k = min(len(cs), max(1, round(n * len(cs) / total)))
        out += rnd.sample(cs, k)
    rnd.shuffle(out)
    return out[:n]


def page_images(pdf: str, data, dpi: int, read_dpi: int) -> list:
    """the pages at dpi, the ones the reading flattened as photos flattened the same way (their boxes are in it)"""
    from read_page import load_pages
    from recog.page import flatten_photo
    imgs = load_pages(pdf, dpi)
    f, low = dpi / read_dpi, None
    for p, page in enumerate(data):
        if page["info"].get("photo"):
            low = low or load_pages(pdf, read_dpi)
            flat, _ = flatten_photo(low[p])
            imgs[p] = flat.resize((round(flat.width * f), round(flat.height * f)))
    return imgs


def cut(img, box, f: float, pad: int = 6):
    x0, y0, x1, y1 = box
    g = img.convert("L")
    return g.crop((max(0, int((min(x0, x1) - pad) * f)), max(0, int((min(y0, y1) - pad) * f)),
                   min(g.width, int((max(x0, x1) + pad) * f)), min(g.height, int((max(y0, y1) + pad) * f))))


PAGE = """<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Math lines</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#6e6e73;--card:#fff;--line:#e3e3e8;--accent:#2a6edb}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f2f2f7;--muted:#a1a1a6;--card:#2c2c2e;--line:#3a3a3c;--accent:#5b9bff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
h1{font-size:18px;margin:0;flex:1}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}
.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:12px 0}
.item img{max-width:100%;background:#fff;border-radius:4px;display:block}
.opt{display:flex;gap:8px;align-items:flex-start;margin-top:6px;direction:ltr;font-family:ui-monospace,monospace;font-size:13px}
.opt span{word-break:break-all}
input[type=text]{width:100%;box-sizing:border-box;font-size:16px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);direction:ltr;font-family:ui-monospace,monospace;margin-top:6px}
.meta{font-size:12px;color:var(--muted)}
.done{border-color:var(--accent)}
p.help{color:var(--muted);margin:4px 0 12px}
</style></head><body>
<header><h1>מה כתוב בשורה?</h1><span id="count" class="meta"></span><button onclick="save()">שמירת התיוג</button></header>
<p class="help">לכל שורה: לבחור את הקריאה הנכונה, או לתקן בתיבה (כתיב פשוט מספיק, למשל sqrt(0.6) או 1.5e-12), או לסמן "מחוק" / "לא קריא". שורה שלא נגעתם בה לא נספרת. "שמירת התיוג" מוריד קובץ math_labels.json - להעביר אותו לתיקיית grader_runs.</p>
<main id="list"></main>
<script>
const ITEMS = __ITEMS__;
const list = document.getElementById('list');
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
ITEMS.forEach((it, i) => {
  const d = document.createElement('div'); d.className = 'item'; d.id = 'it' + i;
  d.innerHTML = `<div class="meta">${i + 1} / ${ITEMS.length} · מבחן ${it.exam}, עמוד ${it.page}, שאלה ${it.q}</div>
    <img src="${it.img}" alt="שורה ${i + 1}">
    <label class="opt"><input type="radio" name="r${i}" value="a"><span>A: ${esc(it.a)}</span></label>
    <label class="opt"><input type="radio" name="r${i}" value="b"><span>B: ${esc(it.b)}</span></label>
    <label class="opt"><input type="radio" name="r${i}" value="fix"><span>אחרת - מה שבתיבה:</span></label>
    <input type="text" id="t${i}" value="${esc(it.a)}">
    <label class="opt"><input type="radio" name="r${i}" value="crossed"><span>מחוק (הסטודנט מחק את השורה)</span></label>
    <label class="opt"><input type="radio" name="r${i}" value="unreadable"><span>לא קריא</span></label>`;
  list.appendChild(d);
  d.querySelector(`#t${i}`).addEventListener('input', () => { d.querySelector(`input[value=fix]`).checked = true; mark(); });
  d.querySelectorAll(`input[type=radio]`).forEach(r => r.addEventListener('change', mark));
});
function mark() {
  let n = 0;
  ITEMS.forEach((it, i) => { const on = document.querySelector(`input[name=r${i}]:checked`); document.getElementById('it' + i).classList.toggle('done', !!on); if (on) n++; });
  document.getElementById('count').textContent = `${n} / ${ITEMS.length} תויגו`;
}
mark();
function save() {
  const out = [];
  ITEMS.forEach((it, i) => {
    const on = document.querySelector(`input[name=r${i}]:checked`);
    if (!on) return;
    const v = on.value, truth = v === 'a' ? it.a : v === 'b' ? it.b : v === 'fix' ? document.getElementById('t' + i).value : null;
    out.push({exam: it.exam, page: it.page, q: it.q, box: it.box, choice: v, truth: truth, unreadable: v === 'unreadable', crossed: v === 'crossed', a: it.a, b: it.b, a_is: it.a_is});
  });
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'math_labels.json'; link.click();
}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", required=True, help="tools/exam_questions.py's pages.json")
    ap.add_argument("--exam", nargs=3, action="append", metavar=("N", "PDF", "JSON"), required=True)
    ap.add_argument("--n", type=int, default=20, help="lines per exam")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--read-dpi", type=int, default=150)
    args = ap.parse_args()
    qpages = json.load(open(args.pages, encoding="utf-8"))
    items, rnd = [], random.Random(args.seed)
    for exam, pdf, path in args.exam:
        data = json.load(open(path, encoding="utf-8"))
        cands = candidates(data, exam, qpages)
        picked = sample(cands, args.n, args.seed + int(exam))
        imgs = page_images(pdf, data, args.dpi, args.read_dpi)
        for ex, pno, q, box, v, m in picked:
            buf = io.BytesIO()
            cut(imgs[pno - 1], box, args.dpi / args.read_dpi).save(buf, "PNG")
            swap = rnd.random() < 0.5                     # (which reader is A is random: no habit of picking the first)
            items.append({"exam": ex, "page": pno, "q": q, "box": box, "img": "data:image/png;base64," +
                          base64.b64encode(buf.getvalue()).decode(), "a": m if swap else v, "b": v if swap else m,
                          "a_is": "mumer" if swap else "qwen"})
        print(f"exam {exam}: {len(cands)} calculation pieces, {len(picked)} picked")
    rnd.shuffle(items)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")))
    print(f"{len(items)} lines -> {args.out}")


if __name__ == "__main__":
    main()
