"""A page to look the circuit reader's work over by eye, beside the exams on this computer (it never leaves it): for
each exam, each drawing tools/read_exam_circuits.py read on a question's pages - cut from the scan, the parts the
detector found boxed on it (transistors red, letters blue, other parts orange) - with what was read: the
transistors, the letters at the nodes, the function tools/circuit_key.py found and its verdict.

    python tools/circuit_review.py <circuits folder> --exams "<folder of the exams' PDFs>" --pages pages.json --q 2
        -> <circuits folder>/review.html (one file, the pictures in it)
"""
import argparse
import base64
import glob
import html
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

COLORS = {"fet": (220, 30, 30), "text": (30, 60, 220)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("circuits")
    ap.add_argument("--exams", required=True)
    ap.add_argument("--pages", default="")
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    import pypdfium2 as pdfium
    from PIL import ImageDraw
    qp = json.load(open(a.pages, encoding="utf-8")) if a.pages else {}
    key_file = os.path.join(a.circuits, "circuit_key.json")
    key = {r["exam"]: r for r in json.load(open(key_file, encoding="utf-8")).get(str(a.q), [])} if os.path.isfile(key_file) else {}
    files = sorted(glob.glob(os.path.join(a.circuits, "*.circuits.json")),
                   key=lambda f: int(re.sub(r"\D", "", os.path.basename(f).split(".")[0]) or 0))
    out = ["<!doctype html><html lang='he' dir='rtl'><meta charset='utf-8'><title>Circuit review</title>",
           "<style>body{font-family:Arial,sans-serif;margin:16px;background:#fafafa}figure{display:inline-block;"
           "margin:8px;padding:8px;background:#fff;border:1px solid #ddd;vertical-align:top;max-width:100%}"
           "img{max-width:100%;height:auto}figcaption{direction:ltr;text-align:left;font-size:13px;white-space:pre-wrap}"
           "h2{margin-top:28px}</style>",
           f"<h1>בדיקת קריאת המעגלים - שאלה {a.q}</h1>",
           "<p>אדום: טרנזיסטור שזוהה. כחול: אות שזוהתה. כתום: רכיב אחר. מתחת לכל ציור: מה נקרא ממנו.</p>"]
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        name = d["exam"]
        own = qp.get(name, {})
        qmap = {int(p): v.get("question") for p, v in own.items()}
        mine = [p for p in d["pages"] if p["circuits"] and (qmap.get(p["page"]) == a.q if qmap else True)]
        k = key.get(name, {})
        out.append(f"<h2>מבחן {html.escape(name)} - {html.escape(str(k.get('verdict', '')))}</h2>")
        if not mine:
            out.append("<p>לא נמצא ציור בדפי השאלה.</p>")
            continue
        pdf_path = os.path.join(a.exams, f"{name}.pdf")
        if not os.path.isfile(pdf_path):
            out.append("<p>(קובץ המבחן לא נמצא)</p>")
            continue
        pdf = pdfium.PdfDocument(pdf_path)
        for p in mine:
            page = pdf[p["page"] - 1].render(scale=a.dpi / 72).to_pil().convert("RGB")
            for c in p["circuits"]:
                x0, y0, x1, y1 = c["box"]
                m = 0.06 * max(x1 - x0, y1 - y0)
                bx = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(page.width, int(x1 + m)), min(page.height, int(y1 + m)))
                crop = page.crop(bx)
                dr = ImageDraw.Draw(crop)
                for cls, b, sc in c["found"]:
                    col = COLORS.get(cls, (240, 140, 0))
                    dr.rectangle((b[0] - bx[0], b[1] - bx[1], b[2] - bx[0], b[3] - bx[1]), outline=col, width=2)
                    if cls not in ("fet", "text"):
                        dr.text((b[0] - bx[0], b[1] - bx[1] - 11), cls, fill=col)
                buf = io.BytesIO()
                crop.save(buf, "PNG")
                net = c["netlist"]
                fets = [q for q in net["components"] if q["type"] in ("nmos", "pmos", "fet")]
                lines = [f"page {p['page']}: {len(fets)} transistors read"]
                for q in fets:
                    r = q.get("roles") or {}
                    lab = lambda n: f"{n} ({net['node_labels'][n]})" if n in net["node_labels"] else str(n)   # noqa: E731
                    if {"Gate", "Drain", "Source"} <= set(r):
                        lines.append(f"  {q['id']} {q['type']}: gate {lab(r['Gate'])}, ends {lab(r['Drain'])} - {lab(r['Source'])}")
                    else:
                        lines.append(f"  {q['id']} {q['type']}: legs {', '.join(lab(n) for n in q['nodes'])} (not all found)")
                lines.append("letters at nodes: " + ", ".join(sorted(set(map(str, net["node_labels"].values())))))
                if k.get("page") == p["page"] and k.get("functions"):
                    lines.append("functions: " + "; ".join(f"{n} = {v}" for n, v in k["functions"].items()))
                b64 = base64.b64encode(buf.getvalue()).decode()
                out.append(f"<figure><img src='data:image/png;base64,{b64}'><figcaption>{html.escape(chr(10).join(lines))}"
                           "</figcaption></figure>")
    path = os.path.join(a.circuits, "review.html")
    open(path, "w", encoding="utf-8").write("\n".join(out) + "</html>\n")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
