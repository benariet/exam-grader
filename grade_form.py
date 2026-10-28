"""Grade a filled answer-box exam (forms/make_exam.py) - a scanned PDF or phone photos of its pages.

    python grade_form.py student.pdf
    python grade_form.py p1.jpg p2.jpg p3.jpg --course data/course_elec101
    python grade_form.py student.pdf --roster students.csv   # match IDs against the class roster
    -> results/<name>_form.txt / .json, and results/<name>_form_p<k>.png (the page as the tool saw it,
       with what it read in every answer box: green = graded automatically, orange = to a human)

See forms/read_form.py for how a page is found, straightened and read.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def overlay(fg, res, out_prefix):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from read_page import font
    ex = fg.exams[res["exam"]]
    items = {r["id"]: r for r in res["items"]}
    files = []
    for pno in res["pages_read"]:
        page = res["_pages"][pno]
        im = Image.fromarray(page).convert("RGB")
        d = ImageDraw.Draw(im)
        f = font(26)
        for item, c in res["_cells"].items():
            if ex["template"]["pages"][str(pno)]["items"].get(item) is None:
                continue
            boxes = c["boxes"]
            x0, y0 = boxes[0][0], boxes[0][1]
            x1, y1 = boxes[-1][2], boxes[-1][3]
            if item == "id":
                sid = res["student_id"]
                col = (20, 150, 60) if sid["valid"] else (230, 120, 0)
                label = f"ID {sid['id']}" + (" ✓" if sid["valid"] else " ?")
            else:
                r = items[item]
                col = (20, 150, 60) if r["route"] == "auto" else (230, 120, 0)
                label = f"'{r['read']}'" + (f"  {r['score']:g}/{r['points']:g}" if r["route"] == "auto" else "  → human")
            d.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), outline=col, width=3)
            d.text((x0, y1 + 6), label, font=f, fill=col)
        p = f"{out_prefix}_p{pno}.png"
        im.save(p)
        files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--course", default=os.path.join(HERE, "data", "course_elec101"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--roster", default=None, help="file with the enrolled students' IDs (any text/CSV); "
                    "default: the file named by the course's course.json 'roster', if any")
    ap.add_argument("--model", default=None, help="cell reader folder (default models/cell-reader)")
    args = ap.parse_args()
    from forms.cells import CellReader
    from forms.read_form import FormGrader, load_images, load_roster, report_text
    fg = FormGrader(args.course, CellReader(args.model) if args.model else None,
                    load_roster(args.roster) if args.roster else None)
    images = load_images(args.paths)
    res = fg.grade(images, keep_pages=True)
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.paths[0]))[0]
    prefix = os.path.join(args.out, f"{name}_form")
    text = report_text(res)
    print(text)
    with open(prefix + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    if res.get("exam"):
        files = overlay(fg, res, prefix)
        print("wrote", ", ".join(os.path.relpath(p, HERE) for p in [prefix + ".txt"] + files))
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(prefix + ".json", "w", encoding="utf-8") as fh:
        json.dump(clean, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
