"""Free answer sheets - each answer written under its question's number, in any order, on plain pages (no booklet with
a place for each answer) - read by the vision model page by page, joined per student, the multiple-choice letters
checked against the key and the short answers graded against the key's points, all against a teacher's marks. Made on
the public Anna University Data Science exam (Mendeley Data, 10.17632/sf3kvjwknt.1, CC BY 4.0: 50 students, 20
multiple-choice letters and 15 short answers of 2 marks each, the teacher's marks per question). On the GPU server.

    python tools/free_sheet.py read <PDFs' folder> --out readings.json [--mcq 20] [--questions 35] [--only 1,2]
        -> {"<pdf>": {"pages": [raw text, ...], "answers": {"1": "B", ..., "21": "..."}}}
    python tools/free_sheet.py reread readings.json <PDFs' folder> --out halves.json     (a student some written
        answer of whom was not found: his pages again in halves; merged into readings.json as 'halves')
    python tools/free_sheet.py grade readings.json --key answerkey.txt --questions-text Question.txt --out grades.json
        -> {"<pdf>": {"21": [mark, raw], ...}}
    python tools/free_sheet.py score readings.json grades.json --key answerkey.txt --teacher <marks.csv>
        -> the letters read against the teacher's record, the marks against his (counts only)
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

PAGE_PROMPT = ("This image is one page of a student's handwritten exam answer sheet. Each answer starts with its "
               "question number (1 to {n}), not necessarily in order; questions 1 to {m} are multiple choice, answered "
               "with one letter (A, B, C or D). For every answer on this page write one line of JSON: "
               '{{"q": <its number>, "answer": "<what is written, exactly; for 1 to {m} only the letter>"}}. A text '
               'at the top of the page with no number continues the previous page\'s last answer: write it as '
               '{{"q": "cont", "answer": "..."}}. Where something cannot be read write ? - do not guess. Write '
               "nothing else.")
GRADE_PROMPT = ("You are grading a university Data Science exam. Question {q}: {question}\nThe answer key: {key}\n"
                "The student's handwritten answer (as transcribed): {answer}\nThe question is worth 2 marks. Give the "
                "mark a fair teacher would: 2 for a correct and complete answer, 1 or 1.5 for a partly correct one, "
                "0.5 for a weak but relevant one, 0 for a wrong one. Answer with the mark only (0, 0.5, 1, 1.5 or 2).")
LINE = re.compile(r'\{\s*"q"\s*:\s*"?(\w+)"?\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')


def page_answers(text: str):
    """the model's lines -> [(question or 'cont', answer)]"""
    out = []
    for m in LINE.finditer(text):
        q, a = m.group(1), m.group(2).replace('\\"', '"').strip()
        out.append((q if q == "cont" else str(int(q)) if q.isdigit() else q, a))
    return out


STOP = set("the and for are that this with from its can not but has have was were which what how why when where who "
           "into than then them they their there these those been being will would should could also such each other "
           "use used using give given two one data answer question".split())


def _words(text: str):
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in STOP}


def by_content(segs, refs, mcq: int, last_q: int):
    """[[question, text]] in reading order -> the same, a number read twice set right: while some question has no
    answer, each answer under a number met twice goes to whichever of that number and the ones missing its words
    share most with (the question's wording and its key, refs) - when clearly more (24 read as 21)"""
    from collections import defaultdict
    by = defaultdict(list)
    for i, (q, _) in enumerate(segs):
        by[q].append(i)
    missing = [str(q) for q in range(mcq + 1, last_q + 1) if str(q) not in by]
    for q in [q for q, ix in by.items() if len(ix) > 1]:
        for i in by[q]:
            if not missing:
                return segs
            w = _words(segs[i][1])
            score = {c: len(w & _words(refs.get(c, ""))) for c in [q] + missing}
            best = max(missing, key=lambda c: score[c])
            if score[best] >= 2 and score[best] >= 1.5 * score[q] + 1:
                segs[i][0] = best
                missing.remove(best)
    return segs


def join_pages(pages, mcq: int, refs=None, last_q: int = 0):
    """each page's answers -> {question: answer}: a continuation added to the last answer before it (a single letter
    right before a page's question 2 is question 1's - its '1' read as something else), a letter answer kept to its
    letter and to the first page that gives it, an answer met twice joined (with refs - {question: its wording and
    key} - first set right by its content, by_content); a page giving many questions one same letter (a near blank
    page - the back's writing showing through - read as a list) gives no letters"""
    from collections import Counter
    got, segs, last = {}, [], None
    for answers in pages:
        letters = Counter(a.strip().upper() for q, a in answers if q.isdigit() and int(q) <= mcq)
        blind = bool(letters) and letters.most_common(1)[0][1] >= 8 and len(letters) == 1
        for i, (q, a) in enumerate(answers):
            if q == "cont":
                nxt = answers[i + 1][0] if i + 1 < len(answers) else ""
                if re.fullmatch(r"\W*[A-Da-d]\W*", a) and nxt == "2" and "1" not in got:
                    q = "1"                            # (question 1's letter, its number misread)
                elif re.search(r"answer all|each question|carries|marks\b", a, re.I) and len(a) < 60:
                    continue                           # (the paper's instructions, copied)
                elif last and int(last) > mcq:
                    segs[-1][1] = (segs[-1][1] + " " + a).strip()
                    continue
                elif last == str(mcq) and not any(x[0] == str(mcq + 1) for x in segs) and len(a) > 15:
                    q = str(mcq + 1)                   # (text right after the last letter: the first written answer,
                else:                                  #  its number not seen)
                    continue
            if not q.isdigit():
                continue
            if int(q) <= mcq:                          # (a letter; 'True' / 'False' the options A / B)
                if blind:
                    continue
                tf = re.search(r"\b(true|false)\b", a, re.I)
                m = re.search(r"\b([A-Da-d])\b", a) or re.fullmatch(r"\W*([A-Da-d])\W*", a)
                a = m.group(1).upper() if m else ("A" if tf.group(1).lower() == "true" else "B") if tf else "?"
                if got.get(q, "?") != "?":
                    continue
                got[q] = a
            else:
                segs.append([q, a])
            last = q
    if refs:
        segs = by_content(segs, refs, mcq, last_q or max([int(q) for q, _ in segs] + [mcq]))
    for q, a in segs:
        got[q] = (got[q] + " " + a).strip() if q in got else a
    return got


def read(a):
    import pypdfium2 as pdfium
    import torch
    from recog.hebrew_vlm import HebrewRecognizer
    rd = HebrewRecognizer(model_dir=a.vlm, dtype=torch.bfloat16 if torch.cuda.is_available() and
                          torch.cuda.is_bf16_supported() else None)
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    only = {x for x in a.only.split(",") if x}
    for pdf in sorted(glob.glob(os.path.join(a.folder, "*.pdf"))):
        name = os.path.splitext(os.path.basename(pdf))[0]
        if name in out or (only and name.split("_")[-1] not in only):
            continue
        doc = pdfium.PdfDocument(pdf)
        raws, pages = [], []
        for i in range(len(doc)):
            w = doc[i].get_size()[0]
            img = doc[i].render(scale=a.width / w).to_pil().convert("RGB")
            c = rd.recognize(img, n_best=1, max_new_tokens=a.tokens,
                             prompt=PAGE_PROMPT.format(n=a.questions, m=a.mcq))[0]
            raws.append(c.text)
            pages.append(page_answers(c.text))
        out[name] = {"pages": raws, "answers": join_pages(pages, a.mcq)}
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(name, len(doc), "pages,", len(out[name]["answers"]), "answers", flush=True)


def answers_of(r, mcq, refs=None, last_q=0):
    """a student's readings -> his answers, joined anew from the pages' raw lines (when kept) - so a better joining
    needs no reading again; a written answer the whole pages missed taken from the pages read again in halves
    ('halves', reread: an answer met in both halves joined once - the longer)"""
    got = join_pages([page_answers(p) for p in r["pages"]], mcq, refs, last_q) if r.get("pages") else r["answers"]
    if r.get("halves"):
        more = {}
        for h in r["halves"]:
            for q, a in join_pages([page_answers(h)], mcq, refs, last_q).items():
                if int(q) > mcq and len(a) > len(more.get(q, "")):
                    more[q] = a
        for q, a in more.items():
            if q not in got:
                got[q] = a
    return got


def reread(a):
    """a student some written answer of whom was not found under its number: each of his pages read again in two
    halves (the top 55 %, the bottom 55 %: half as much writing for the model to keep its numbers apart) ->
    readings' 'halves' - answers_of takes from them only the questions the whole pages missed"""
    import pypdfium2 as pdfium
    import torch
    from recog.hebrew_vlm import HebrewRecognizer
    rd = HebrewRecognizer(model_dir=a.vlm, dtype=torch.bfloat16 if torch.cuda.is_available() and
                          torch.cuda.is_bf16_supported() else None)
    readings = json.load(open(a.readings, encoding="utf-8"))
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    only = {x for x in a.only.split(",") if x}
    for name, r in readings.items():
        if name in out or (only and name.split("_")[-1] not in only):
            continue
        got = answers_of(r, a.mcq)
        if all(str(q) in got for q in range(a.mcq + 1, a.questions + 1)):
            continue
        doc = pdfium.PdfDocument(os.path.join(a.folder, name + ".pdf"))
        halves = []
        for i in range(len(doc)):
            w = doc[i].get_size()[0]
            img = doc[i].render(scale=a.width / w).to_pil().convert("RGB")
            for y0, y1 in ((0.0, 0.55), (0.45, 1.0)):
                part = img.crop((0, int(y0 * img.height), img.width, int(y1 * img.height)))
                c = rd.recognize(part, n_best=1, max_new_tokens=a.tokens,
                                 prompt=PAGE_PROMPT.format(n=a.questions, m=a.mcq))[0]
                halves.append(c.text)
        out[name] = halves
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(name, len(halves), "halves", flush=True)


def key_of(path):
    """answerkey.txt (Question_Number,Type,Correct_Answer) -> {question: answer}"""
    return {r["Question_Number"]: r["Correct_Answer"] for r in csv.DictReader(open(path, encoding="utf-8"))}


def questions_of(path):
    """Question.txt -> {number: its text (first line)}"""
    qs = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if m:
            qs[m.group(1)] = m.group(2).strip()
    return qs


def grade(a):
    import torch
    from PIL import Image
    from recog.hebrew_vlm import HebrewRecognizer
    rd = HebrewRecognizer(model_dir=a.vlm, dtype=torch.bfloat16 if torch.cuda.is_available() and
                          torch.cuda.is_bf16_supported() else None)
    readings = json.load(open(a.readings, encoding="utf-8"))
    key, qs = key_of(a.key), questions_of(a.questions_text)
    refs = {q: qs.get(q, "") + " " + key.get(q, "") for q in set(qs) | set(key)}
    blank = Image.new("RGB", (32, 32), "white")      # (the model is asked about text: an empty picture with it)
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    for name, r in readings.items():
        if name in out:
            continue
        marks = {}
        for q in range(a.mcq + 1, a.questions + 1):
            ans = answers_of(r, a.mcq, refs, a.questions).get(str(q), "").strip()
            if not ans or set(ans) <= set("? "):
                marks[str(q)] = [None, ""]             # (no answer read: not graded)
                continue
            c = rd.recognize(blank, n_best=1, max_new_tokens=8, prompt=GRADE_PROMPT.format(
                q=q, question=qs.get(str(q), ""), key=key.get(str(q), ""), answer=ans))[0]
            m = re.search(r"\d+(?:\.\d+)?", c.text)
            marks[str(q)] = [min(2.0, float(m.group())) if m else None, c.text]
        out[name] = marks
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        print(name, sum(1 for v in marks.values() if v[0] is not None), "graded", flush=True)


def score(a):
    """counts only: how many letters read as the teacher recorded them; the marks against his"""
    readings = json.load(open(a.readings, encoding="utf-8"))
    grades = json.load(open(a.grades, encoding="utf-8")) if a.grades else {}
    key = key_of(a.key)
    qs = questions_of(a.questions_text) if a.questions_text else {}
    refs = {q: qs.get(q, "") + " " + key.get(q, "") for q in set(qs) | set(key)} if qs else None
    teacher = {r["Student_ID"].split("_")[-1]: r for r in csv.DictReader(open(a.teacher, encoding="utf-8"))}
    letters = [0, 0, 0]                                # (as recorded, other, not read)
    found = []                                         # (an answer found under its number, the teacher's mark)
    mcq_marks, short = [], []
    for name, r in readings.items():
        t = teacher.get(name.split("_")[-1])
        if not t:
            continue
        mine = 0
        mine_answers = answers_of(r, a.mcq, refs, a.questions)
        for q in range(1, a.mcq + 1):
            got, rec = mine_answers.get(str(q), "?"), (t.get(str(q)) or "").strip().upper()
            rec = {"TRUE": "A", "FALSE": "B"}.get(rec, rec)
            letters[0 if got == rec else 2 if got == "?" else 1] += 1
            mine += got == key.get(str(q))
        mcq_marks.append((mine, float(t["MCQ mark"])))
        for q in range(a.mcq + 1, a.questions + 1):
            tm = (t.get(str(q)) or "").strip()
            try:
                tm = float(tm)
            except ValueError:                         # ('', 'NC', 'NM': blank, not marked - nothing given)
                tm = 0.0
            g = (grades.get(name) or {}).get(str(q), [None])[0]
            short.append((g, tm))
            found.append((str(q) in mine_answers, tm))
    n = sum(letters)
    print(f"letters: {n} - as the teacher recorded {letters[0]} ({100 * letters[0] / max(1, n):.1f}%), "
          f"another {letters[1]}, not read {letters[2]}")
    print(f"written answers found under their number: {sum(1 for f, _ in found if f)} of {len(found)}; of those the "
          f"teacher gave marks to, not found: {sum(1 for f, t in found if not f and t > 0)} of "
          f"{sum(1 for _, t in found if t > 0)}")
    d = [abs(x - y) for x, y in mcq_marks]
    print(f"multiple-choice mark (of 20) against the teacher's: exact {sum(1 for x in d if x == 0)} of {len(d)}, "
          f"mean difference {sum(d) / max(1, len(d)):.2f}")
    graded = [(g, t) for g, t in short if g is not None]
    if graded:
        ex = sum(1 for g, t in graded if abs(g - t) < 1e-9)
        near = sum(1 for g, t in graded if abs(g - t) <= 0.5 + 1e-9)
        mae = sum(abs(g - t) for g, t in graded) / len(graded)
        print(f"short answers graded {len(graded)} of {len(short)}: the teacher's mark exactly {ex} "
              f"({100 * ex / len(graded):.0f}%), within half a mark {near} ({100 * near / len(graded):.0f}%), "
              f"mean difference {mae:.2f}")
        unread = [t for g, t in short if g is None]
        print(f"  not graded (nothing read): {len(unread)}, of them the teacher gave marks to "
              f"{sum(1 for t in unread if t > 0)}")
        tot = {}
        for name in readings:
            t = teacher.get(name.split("_")[-1])
            if not t or name not in grades:
                continue
            mine = sum(v[0] or 0 for v in grades[name].values())
            tot[name] = (mine, float(t["2 marks"]))
        d = [abs(x - y) for x, y in tot.values()]
        print(f"short answers' total (of 30) against the teacher's: mean difference {sum(d) / max(1, len(d)):.2f}, "
              f"within 3 marks {sum(1 for x in d if x <= 3)} of {len(d)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("folder")
    r.add_argument("--out", required=True)
    r.add_argument("--mcq", type=int, default=20)
    r.add_argument("--questions", type=int, default=35)
    r.add_argument("--width", type=int, default=1400)
    r.add_argument("--tokens", type=int, default=700)
    r.add_argument("--only", default="")
    r.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    g = sub.add_parser("grade")
    g.add_argument("readings")
    g.add_argument("--key", required=True)
    g.add_argument("--questions-text", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--mcq", type=int, default=20)
    g.add_argument("--questions", type=int, default=35)
    g.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    h = sub.add_parser("reread")
    h.add_argument("readings")
    h.add_argument("folder")
    h.add_argument("--out", required=True)
    h.add_argument("--mcq", type=int, default=20)
    h.add_argument("--questions", type=int, default=35)
    h.add_argument("--width", type=int, default=1400)
    h.add_argument("--tokens", type=int, default=500)
    h.add_argument("--only", default="")
    h.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    s = sub.add_parser("score")
    s.add_argument("readings")
    s.add_argument("grades", nargs="?", default="")
    s.add_argument("--key", required=True)
    s.add_argument("--teacher", required=True)
    s.add_argument("--questions-text", default="", help="the questions' wording: an answer under a number met twice "
                   "set right by its content")
    s.add_argument("--mcq", type=int, default=20)
    s.add_argument("--questions", type=int, default=35)
    a = ap.parse_args()
    {"read": read, "reread": reread, "grade": grade, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
