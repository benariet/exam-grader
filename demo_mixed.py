"""Mixed Hebrew + math answers, end to end, fully local.

Question (ECE): a current of 2 mA flows through a 2.5 kOhm resistor. What is the voltage across it,
and is it high or low compared with 10 V?

data/demo/answer_A.png, answer_B.png are made of REAL handwriting: the Hebrew words were cut
from test-split lines of writers the recognizer never saw (A: writer w4, B: writer w569), the
formulas are MathWriting inks that were kept out of the classifier's training.

Pipeline: page -> lines -> chunks -> Hebrew/math classifier -> segments ->
          Hebrew recognizer (CRNN-CTC) | formula recognizer (Pix2Text MFR), n-best with probabilities ->
          rubric -> score distribution -> confidence -> auto / human review.
Output: printed report + results/mixed_<student>.png
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from grader.mixed import MixedCriterion, grade_mixed, readings  # noqa: E402
from recog.mixed_reader import MixedReader  # noqa: E402

QUESTION = "זרם של 2mA עובר בנגד של 2.5kΩ. מה המתח על הנגד, והאם הוא גבוה או נמוך ביחס ל-10V?"
RUBRIC = [
    MixedCriterion("ohm", "formula_any", 3, ["V=IR", "I=\\frac{V}{R}", "R=\\frac{V}{I}"], description="uses Ohm's law"),
    MixedCriterion("value", "numeric_final", 5, ["5 V"], description="final answer V = 5 V"),
    MixedCriterion("low", "keyword", 2, ["נמוך"], description="says 5 V is low compared with 10 V"),
]


def read_answer(path, reader):
    page = Image.open(path).convert("L")
    lines, segments = reader.read(page)
    return page, lines, segments


def transcript(lines):
    out = []
    for segs in lines:
        out.append("  ".join(s.nbest[0][0] if s.kind == "hebrew" else f"${s.nbest[0][0]}$" for s in segs))
    return out


def pretty(tex: str) -> str:
    """Readable display of a LaTeX reading (display only - grading uses the raw text)."""
    import re
    t = tex.replace(" ", "")

    def frac(m):
        a, b = m.group(1), m.group(2)
        a = a if len(a) == 1 else f"({a})"
        b = b if len(b) == 1 else f"({b})"
        follows = m.string[m.end():m.end() + 1]
        return f"({a}/{b})" if follows and (follows.isalnum() or follows == "\\") else f"{a}/{b}"
    for _ in range(3):
        t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", frac, t)
    for a, b in (("\\cdot", "\u00b7"), ("\\times", "\u00d7"), ("\\Omega", "\u03a9"), ("\\pi", "\u03c0"),
                 ("\\mu", "\u00b5"), ("\\omega", "\u03c9"), ("\\tau", "\u03c4"), ("\\rightarrow", "\u2192"),
                 ("\\approx", "\u2248")):
        t = t.replace(a, b)
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])", "", t)          # (not \\rightarrow, \\leftarrow)
    t = re.sub(r"\\(?:vert|mid)(?![A-Za-z]{2})", "|", t)          # '\\verta' (spaces dropped above) -> '|a'
    t = re.sub(r"\\sqrt\{([^{}]*)\}", "\u221a\\1", t)
    return _ell(t)


def _ell(t: str) -> str:
    """show a lowercase L as the script letter: in most fonts 'l' and 'I' look the same"""
    import re
    return re.sub(r"(?<![A-Za-z\\])l(?![A-Za-z])", "\u2113", t)


def _font(size, bold=False):
    for f in (["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"] if os.name == "nt" else []) + \
             ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw(page, lines, res, path, title):
    f, fs, fb = _font(17), _font(15), _font(19, bold=True)
    colors = {"hebrew": (40, 110, 220), "math": (230, 120, 20), "english": (30, 150, 70)}
    LEFT, TOP, ROWGAP = 560, 64, 40
    line_rows = []
    for segs in lines:
        y0 = min(s.box[1] for s in segs); y1 = max(s.box[3] for s in segs)
        line_rows.append((segs, y0, y1))
    H = TOP + sum(y1 - y0 + 12 + ROWGAP for _, y0, y1 in line_rows) + 60
    H = max(H, TOP + 40 + 52 * (len(res.criteria) + 2))
    W = LEFT + page.width
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    d.text((W - 20, 16), title, font=fb, fill="black", anchor="ra", direction="rtl")
    y = TOP
    for segs, y0, y1 in line_rows:                       # page line, its boxes, then its reading
        strip = page.crop((0, y0 - 6, page.width, y1 + 6)).convert("RGB")
        img.paste(strip, (LEFT, y))
        for s in segs:
            x0, sy0, x1, sy1 = s.box
            d.rectangle((LEFT + x0 - 3, y + sy0 - y0 + 3, LEFT + x1 + 3, y + sy1 - y0 + 9), outline=colors[s.kind], width=2)
        y += y1 - y0 + 12
        parts = []
        for s in segs:
            shown = [(t if s.kind != "math" else pretty(t), p) for t, p in readings(s.nbest)]
            alt = next(((t, p) for t, p in shown[1:] if t != shown[0][0]), None)
            # an unsure reading is shown with its runner-up, so the uncertainty is visible
            parts.append((shown[0][0], shown[0][1], s.kind, alt if shown[0][1] < 0.7 else None))
        x = W - 24                                     # reading order = right to left
        for t, p, kind, alt in parts:
            label = f"{t} ({p:.0%})" + (f" / {alt[0]} ({alt[1]:.0%})" if alt else "")
            w = d.textlength(label, font=fs, direction="rtl" if kind == "hebrew" else None)
            d.text((x, y + 2), label, font=fs, fill=colors[kind], anchor="ra", direction="rtl" if kind == "hebrew" else None)
            x -= w + 22
        y += ROWGAP
    y = TOP
    d.text((20, y), f"score: {res.most_likely_score:g} / {res.max_points:g}   (expected {res.expected_score:.2f})", font=fb, fill="black"); y += 30
    col = (20, 130, 60) if res.route == "auto" else (200, 90, 20)
    d.text((20, y), f"confidence {res.confidence:.0%}  ->  {res.route.upper()}", font=fb, fill=col); y += 26
    if res.route != "auto":
        d.text((20, y), res.reason, font=fs, fill=col)
    y += 36
    kinds = {c.id: c.type for c in RUBRIC}
    for o in res.criteria:
        mark = "+" if o.p_full >= 0.5 else "-"
        d.text((20, y), f"[{mark}] {o.description}  ({o.points:g} pt)   P(full) = {o.p_full:.2f}", font=f,
               fill=(20, 130, 60) if o.p_full >= 0.5 else (190, 40, 40)); y += 22
        short = o.short
        if kinds.get(o.id) in ("formula_any", "numeric_final"):     # formula first, then the explanation
            head, sep, tail = short.partition("  ")
            short = pretty(head) + sep + _ell(tail)
        d.text((44, y), short, font=fs, fill=(110, 110, 110)); y += 30
    d.text((20, H - 30), "blue = Hebrew   green = English   orange = math   % = probability of the reading"
                         "   \u2113 = lowercase L",
           font=fs, fill=(110, 110, 110))
    img.save(path)


def main():
    reader = MixedReader()
    truth = json.load(open(os.path.join(HERE, "data", "demo", "truth.json"), encoding="utf-8"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    print("question:", QUESTION, "\n")
    for sid in sorted(truth):
        page, lines, segments = read_answer(os.path.join(HERE, "data", "demo", f"answer_{sid}.png"), reader)
        res = grade_mixed(RUBRIC, segments)
        print(f"===== student {sid} (Hebrew handwriting: writer {truth[sid]['hebrew_writer']}, never seen in training)")
        for k, (got, want) in enumerate(zip(transcript(lines), truth[sid]["lines"])):
            want_s = "  ".join(x["text"] if x["kind"] == "hebrew" else f"${x['text']}$" for x in want)
            print(f"  line {k + 1}: read   {got}\n          truth  {want_s}")
        for s in segments:
            print(f"    [{s.kind:6} p_kind={s.p_kind:.2f}] " + " | ".join(f"{t} ({p:.2f})" for t, p in readings(s.nbest))
                  + (f"   <- CROSS-CHECK: {s.note}" if getattr(s, "note", "") else ""))
        for o in res.criteria:
            print(f"  - {o.description:38} {o.points:g} pt   P(full)={o.p_full:.2f}   {o.evidence}")
        print(f"  => score {res.most_likely_score:g}/{res.max_points:g}  expected {res.expected_score}  "
              f"confidence {res.confidence:.0%}  route={res.route}  ({res.reason})")
        print(f"     score distribution: {res.distribution}\n")
        draw(page, lines, res, os.path.join(HERE, "results", f"mixed_{sid}.png"), f"תלמיד {sid}: " + QUESTION)
    imgs = [Image.open(os.path.join(HERE, "results", f"mixed_{sid}.png")) for sid in sorted(truth)]
    both = Image.new("RGB", (max(i.width for i in imgs), sum(i.height for i in imgs) + 12 * (len(imgs) - 1)), (200, 200, 200))
    y = 0
    for im in imgs:
        both.paste(im, (0, y)); y += im.height + 12
    both.save(os.path.join(HERE, "results", "mixed_demo.png"))
    print("wrote results/mixed_A.png, results/mixed_B.png, results/mixed_demo.png")


if __name__ == "__main__":
    main()
