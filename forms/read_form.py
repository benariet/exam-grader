"""Read and grade a filled answer-box exam (forms/make_exam.py) from a scan or a phone photo.

    python grade_form.py student.pdf                      # or photos: page1.jpg page2.jpg page3.jpg
    python grade_form.py student.pdf --course data/course_elec101

Per page: the four corner markers are found (solid dark squares, the ones nearest the image's
corners), the page is straightened onto the printed template by the homography they give, and the
QR code says which exam and page it is (a page photographed upside down is turned). Every answer
row is then located exactly (the printed cell borders are matched within a few pixels - paper bends,
lenses distort), every cell is read by forms.cells.CellReader, and the cells' probabilities make an
n-best list of the answer ('5.71': 0.97, '5.11': 0.02, ...).

Grading (grader.core over those readings): a numeric answer within the course's tolerance gets the
points; an item goes to a human when the reading is not sure enough (AUTO_CONF), a cell looks
crossed out or unreadable, or the answer is open. The student's ID (9 cells) is read the same way
and checked by its check digit - a misread digit almost always breaks it.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

AUTO_CONF = 0.98          # probability of the most likely score, to grade without a human (development set:
#                           0.95 -> 2 automatic errors in 744 answers, 0.97 -> 1, 0.98 -> 0)
MAX_READINGS = 24
CELL_SHRINK = 4           # px (150 dpi) cut from every side of a printed cell: its border stays out
WIDEN = 3                 # px beyond the printed cell for the second view of it
STRAY_INK = 8             # dark pixels in a cell the reader calls empty: something is written there
GAP_INK = 6               # dark pixels in the gap between two cells: writing crosses there
GAP_MIN = 2               # ... in this many gaps of one answer: a line across the box (crossed out). One gap
#                           alone is mostly a big digit reaching over its border (13x as often on the
#                           development set) - the reading's own confidence decides those
ID_SHARE = 0.98           # the ID: the most likely reading must have a valid check digit and hold this share
#                           of all valid readings' probability (a misread digit breaks the check digit, so
#                           the valid readings concentrate on the true ID)
ID_MIN_P = 0.5            # ... and be at least this likely by itself: two misread
#                           digits can cancel out in the check digit ('2'->'7' and '6'->'1' did, on CVL, at p 0.47)
# with the course's class roster (course.json "roster": a file of IDs) the ID is matched against the
# enrolled students instead: the roster ID that best explains the 9 cells is accepted when
ROSTER_SHARE = 0.999      # ... it holds this share of the probability of all roster IDs,
ROSTER_LOGRATIO = -12.0   # ... its probability is at least e^-12 of the best free reading's (a student
#                           missing from the roster is not matched to someone else: on the simulated
#                           sets the true ID scored >= -6.6 on the development data, a wrong one <= -23),
ROSTER_MISMATCH = 1       # ... and it differs from the cells' own best reading in at most one cell (with
#                           30 near-twin IDs - two digits apart - on the roster and the student's own ID
#                           missing, none was accepted on any simulated set)


# ------------------------------------------------------------------ geometry
def _gray(img: Image.Image) -> np.ndarray:
    from recog.page import ink_gray
    return ink_gray(img)


def find_markers(g: np.ndarray):
    """centres of the four corner markers (tl, tr, bl, br) in image pixels, or None"""
    import cv2
    H, W = g.shape
    k = max(15, int(min(H, W) / 25) | 1)
    bg = cv2.GaussianBlur(cv2.dilate(g.astype(np.float32), np.ones((k, k), np.uint8)), (0, 0), k / 3)
    norm = g / np.maximum(bg, 1)
    dark = (norm < 0.55).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        side = (w + h) / 2
        if not (0.006 * max(H, W) <= side <= 0.07 * max(H, W)):
            continue
        if not (0.7 <= w / max(h, 1) <= 1.43) or area < 0.78 * w * h:
            continue
        cand.append((area, cents[i][0], cents[i][1]))
    if len(cand) < 4:
        return None
    cand.sort(reverse=True)
    big = [c for c in cand if c[0] >= 0.45 * cand[min(3, len(cand) - 1)][0]]
    out = []
    for cx, cy in ((0, 0), (W, 0), (0, H), (W, H)):
        best = min(big, key=lambda c: (c[1] - cx) ** 2 + (c[2] - cy) ** 2)
        out.append((best[1], best[2]))
    if len({(round(x), round(y)) for x, y in out}) < 4:
        return None
    return out


def straighten(img: Image.Image, page_tpl: dict, page_px):
    """-> page image in template space (150 dpi), or None if the markers are not found"""
    import cv2
    g = _gray(img)
    m = find_markers(g)
    if m is None:
        return None
    dst = np.float32([page_tpl["markers"][k] for k in ("tl", "tr", "bl", "br")])
    M = cv2.getPerspectiveTransform(np.float32(m), dst)
    W, H = page_px
    warped = cv2.warpPerspective(g.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    return np.clip(warped, 0, 255).astype(np.uint8)


def read_qr(page: np.ndarray, box=None):
    """-> (text, centre of the code in page pixels) - only inside box (+ a margin) when given"""
    import cv2
    det = cv2.QRCodeDetector()
    ox, oy, r = 0, 0, page
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        p = 40
        ox, oy = max(0, x0 - p), max(0, y0 - p)
        r = page[oy:y1 + p, ox:x1 + p]
    for scale in (1.0, 2.0):
        rr = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else r
        txt, pts, _ = det.detectAndDecode(rr)
        if txt:
            c = np.asarray(pts, np.float32).reshape(-1, 2).mean(0) / scale if pts is not None else np.zeros(2)
            return txt, (float(c[0] + ox), float(c[1] + oy))
    return "", None


def refine_row(page: np.ndarray, boxes, search=10):
    """the printed row of cells as it lies on this page: match the cell borders within +-search px"""
    import cv2
    x0 = int(min(b[0] for b in boxes)) - 4
    y0 = int(min(b[1] for b in boxes)) - 4
    x1 = int(max(b[2] for b in boxes)) + 4
    y1 = int(max(b[3] for b in boxes)) + 4
    tpl = np.zeros((y1 - y0, x1 - x0), np.float32)
    for b in boxes:
        a, c, e, f = int(b[0]) - x0, int(b[1]) - y0, int(b[2]) - x0, int(b[3]) - y0
        tpl[c:f, a:a + 2] = tpl[c:f, e - 2:e] = 1
        tpl[c:c + 2, a:e] = tpl[f - 2:f, a:e] = 1
    H, W = page.shape
    sx0, sy0 = max(0, x0 - search), max(0, y0 - search)
    sx1, sy1 = min(W, x1 + search), min(H, y1 + search)
    region = 255.0 - page[sy0:sy1, sx0:sx1].astype(np.float32)
    if region.shape[0] < tpl.shape[0] or region.shape[1] < tpl.shape[1]:
        return boxes, 0.0
    res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    dx, dy = loc[0] + sx0 - x0, loc[1] + sy0 - y0
    if score < 0.2:
        return boxes, float(score)
    return [[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy] for b in boxes], float(score)


def gap_ink(page: np.ndarray, boxes) -> list:
    """dark pixels in the middle of every gap between two neighbouring cells of a row"""
    out = []
    for a, b in zip(boxes, boxes[1:]):
        xm = (a[2] + b[0]) / 2
        x0, x1 = int(round(xm - 1.5)), int(round(xm + 1.5))
        y0, y1 = int(round(max(a[1], b[1]))) + 4, int(round(min(a[3], b[3]))) - 4
        reg = page[y0:y1, x0:x1]
        if reg.size == 0:
            out.append(0)
            continue
        bg = np.percentile(page[y0:y1, int(a[0]):int(b[2])], 90)
        out.append(int((reg < bg - 70).sum()))
    return out


def cell_crops(page: np.ndarray, boxes, shrink=CELL_SHRINK):
    out = []
    for b in boxes:
        x0, y0, x1, y1 = int(round(b[0])) + shrink, int(round(b[1])) + shrink, int(round(b[2])) - shrink, int(round(b[3])) - shrink
        out.append(page[max(0, y0):y1, max(0, x0):x1])
    return out


# ------------------------------------------------------------------ cells -> answer readings
NUM_RE = re.compile(r"^-?(\d+(\.\d+)?|\.\d+)$")


def nbest_strings(P: np.ndarray, classes, k=MAX_READINGS, per_cell=3):
    """cells' probabilities (n x C) -> [(string, prob)] by beam search. Empty cells at the ends drop
    out; an empty cell BETWEEN characters stays as '_' (the form says: one character per cell, from
    the first cell on - a gap means a character the reader did not see, a lost decimal point most
    often), and '?' is a cell that looks crossed out or unreadable. Neither makes a number."""
    beams = [("", 1.0)]
    for row in P:
        top = np.argsort(row)[::-1][:per_cell]
        nb = []
        for s, p in beams:
            for c in top:
                ch = classes[c]
                nb.append((s + ("_" if ch == " " else ch), p * float(row[c])))
        merged = {}
        for s, p in nb:
            merged[s] = merged.get(s, 0.0) + p
        beams = sorted(merged.items(), key=lambda t: -t[1])[:k]
    out = {}
    for s, p in beams:
        s = s.strip("_")
        out[s] = out.get(s, 0.0) + p
    return sorted(out.items(), key=lambda t: -t[1])


def stray_ink(crop: np.ndarray, contrast: float = 70) -> int:
    """dark pixels in a cell crop (what is clearly darker than its paper)"""
    if crop.size == 0:
        return 0
    bg = np.percentile(crop, 90)
    return int((crop < bg - contrast).sum())


def infer_point(P: np.ndarray, crops, classes) -> np.ndarray:
    """The form's rule says one character per cell, from the first cell on, no gaps. So a cell that
    looks empty BETWEEN two written characters is not empty: the one character small enough to be
    missed there is a decimal point. If such a cell shows any trace of ink (or the reader gives the
    point some probability), its 'empty' probability becomes 'point'. Only when the answer has no
    other point and only one such cell; a cell with no trace at all stays a gap (-> a human)."""
    e, d = classes.index(" "), classes.index(".")
    top = P.argmax(1)
    written = [i for i, k in enumerate(top) if k != e]
    if len(written) < 2 or (top == d).any():
        return P
    inner = [i for i in range(written[0] + 1, written[-1]) if top[i] == e]
    if len(inner) != 1:
        return P
    i = inner[0]
    if P[i, d] >= 0.02 or stray_ink(crops[i], 40) >= 2:
        P = P.copy()
        P[i, d] += P[i, e]
        P[i, e] = 0.0
    return P


def guard_empty(P: np.ndarray, crops, classes) -> np.ndarray:
    """a cell the reader calls empty although ink is there: its 'empty' probability goes to 'unreadable'
    (a faint decimal point, a mark at the cell's edge) - the answer then goes to a human"""
    e, j = classes.index(" "), classes.index("?")
    P = P.copy()
    for i, c in enumerate(crops):
        if P[i].argmax() == e and stray_ink(c) >= STRAY_INK:
            P[i, j] += P[i, e] * 0.9
            P[i, e] *= 0.1
    return P


ID_WEIGHTS = (1, 2, 1, 2, 1, 2, 1, 2, 1)


def id_valid(s: str) -> bool:
    """Israeli ID check digit"""
    if not re.fullmatch(r"\d{9}", s):
        return False
    tot = 0
    for d, w in zip(s, ID_WEIGHTS):
        v = int(d) * w
        tot += v - 9 if v > 9 else v
    return tot % 10 == 0


def load_roster(path: str) -> list:
    """IDs from a text / CSV file: every 5-9 digit number in it (a spreadsheet drops leading zeros:
    padded back to 9 digits)"""
    ids = []
    for tok in re.findall(r"(?<!\d)\d{5,9}(?!\d)", open(path, encoding="utf-8-sig").read()):
        tok = tok.zfill(9)
        if id_valid(tok) and tok not in ids:
            ids.append(tok)
    return ids


def roster_match(P: np.ndarray, classes, roster):
    """-> (id, share, log ratio, cells differing from the free reading) for the roster ID that best
    explains the ID cells' probabilities P (9 x classes)"""
    dig = [classes.index(str(d)) for d in range(10)]
    logP = np.log(np.clip(P[:, dig], 1e-12, 1.0))
    best_free = float(np.log(np.clip(P.max(1), 1e-12, 1.0)).sum())
    R = np.array([[int(c) for c in r] for r in roster])
    ll = logP[np.arange(P.shape[0])[None, :], R].sum(1)
    k = int(ll.argmax())
    m = float(ll[k])
    share = float(1.0 / np.exp(ll - m).sum())
    top = [classes[i] for i in P.argmax(1)]
    mism = sum(1 for a, b in zip(roster[k], top) if a != b)
    return roster[k], share, m - best_free, mism


WRITER_SHARE = 0.5        # a cell read as c, for a student whose k the ID showed to be read as c: this share
#                           of its c probability goes to k (the answer then rarely stays sure enough)


def apply_confusions(P: np.ndarray, confusions, share: float = WRITER_SHARE) -> np.ndarray:
    if not confusions:
        return P
    P = P.copy()
    for c, k in confusions:
        moved = P[:, c] * share
        P[:, c] -= moved
        P[:, k] += moved
    return P


def decimal_shift(text: str, answer: float, rel_tol: float) -> bool:
    """'24' or '0.24' for 2.4: the value is the answer times a power of ten (not 1)"""
    if not NUM_RE.match(text or "") or answer == 0:
        return False
    v = float(text)
    if v == 0 or (v < 0) != (answer < 0):
        return False
    k = round(math.log10(abs(v / answer)))
    return k != 0 and abs(v / (answer * 10 ** k) - 1) <= rel_tol


@dataclass
class ItemResult:
    id: str
    kind: str
    points: float
    route: str                       # auto | review
    score: float | None = None
    read: str = ""
    confidence: float | None = None
    reason: str = ""
    readings: list = field(default_factory=list)
    truth_note: str = ""


def grade_number(q: dict, readings, rel_tol: float, wrong_to_human: bool, crosses: bool = False) -> ItemResult:
    from grader.core import Reading, grade_question
    from grader.rubric import Criterion, Question
    pts = float(q["points"])
    total = sum(p for _, p in readings)
    if not readings:
        return ItemResult(q["id"], q["kind"], pts, "review", reason="no reading")
    top, ptop = readings[0]
    if top == "" and ptop / max(total, 1e-9) >= AUTO_CONF and ptop >= AUTO_CONF and not crosses:
        return ItemResult(q["id"], q["kind"], pts, "auto", 0.0, "", ptop, "no answer written")
    if q["kind"] == "choice":
        ref, tol = str(q["answer"]), 0.0
        ok = lambda s: s in ("1", "2", "3", "4")  # noqa: E731
    else:
        ref, tol = repr(float(q["answer"])), rel_tol
        ok = lambda s: bool(NUM_RE.match(s))  # noqa: E731
    # three kinds of reading: a number (or option), an unreadable one ('?': a cell looks crossed out,
    # '_': a character missed between two), and one that breaks the answer's grammar ('5.71.', '-5-7') -
    # nobody writes those: they are the reader's small doubts about empty cells and are dropped
    kept = [(s, p) for s, p in readings if ok(s) or "?" in s or "_" in s]
    rest = max(0.0, 1.0 - total)
    norm = sum(p for _, p in kept) + rest
    if not kept or norm <= 0:
        return ItemResult(q["id"], q["kind"], pts, "review", reason=f"'{top}' is not a number", read=top)
    # an unreadable share ('?', '_') counts against the confidence: on the development set, ignoring
    # even a small one turned a doubtful '7' into a sure '4'
    cands = [Reading(s if ok(s) else "?", p / norm) for s, p in kept]
    if rest > 1e-6:
        cands.append(Reading("?", rest / norm))
    qq = Question(q["id"], q.get("text", ""), [Criterion("value", "final_numeric", pts, ref, rel_tol=tol,
                                                         prefix_error_credit=0.0)])
    g = grade_question(qq, cands, confidence_threshold=AUTO_CONF)
    route, reason = g.route, g.reason
    if route == "review" and "unparseable" in reason:
        reason = reason.replace("of the reading probability is unparseable", "of the reading looks unreadable or crossed out")
    if crosses:
        route, reason = "review", "writing crosses the cells' borders (crossed out? written between cells?)"
    elif "?" in top:
        route, reason = "review", "a cell looks crossed out or unreadable"
    elif "_" in top:
        route, reason = "review", "an empty cell between characters (a character not seen?)"
    elif not ok(top):
        route, reason = "review", f"'{top}' is not a number" if q["kind"] == "number" else f"'{top}' is not an option 1-4"
    if route == "auto" and g.most_likely_score < pts and q["kind"] == "number" and decimal_shift(top, float(q["answer"]), rel_tol):
        # the same digits as the right answer with the point elsewhere: a missed or misplaced point in
        # the reading costs the student the whole item, so a human looks (a real factor-10 slip stays 0)
        route, reason = "review", "the answer's digits with the decimal point elsewhere (point misread?)"
    if route == "auto" and g.most_likely_score < pts and wrong_to_human:
        route, reason = "review", "wrong final answer: a human checks the work for partial credit"
    return ItemResult(q["id"], q["kind"], pts, route, g.most_likely_score, top, g.confidence, reason,
                      [(s, round(p, 4)) for s, p in readings[:5]])


# ------------------------------------------------------------------ a whole student
class FormGrader:
    def __init__(self, course_dir: str, reader=None, roster=None):
        """roster: a list of IDs (default: the file named by course.json "roster", if any)"""
        from forms.cells import CellReader
        self.course_dir = course_dir
        self.course = json.load(open(os.path.join(course_dir, "course.json"), encoding="utf-8"))
        self.exams = {e: json.load(open(os.path.join(course_dir, f"{e}.json"), encoding="utf-8")) for e in self.course["exams"]}
        self.reader = reader or CellReader()
        if roster is None and self.course.get("roster"):
            roster = load_roster(os.path.join(course_dir, self.course["roster"]))
        self.roster = roster or []

    def identify(self, img: Image.Image):
        """-> (exam id, page no., straightened page) using any exam's template (all share the layout grid)"""
        any_ex = next(iter(self.exams.values()))
        tpl = any_ex["template"]
        page = straighten(img, tpl["pages"]["1"], tpl["page_px"])
        if page is None:
            return None, None, None, "corner markers not found"
        qr_box = tpl["pages"]["1"]["qr"]
        flipped = lambda a: np.ascontiguousarray(a[::-1, ::-1])  # noqa: E731
        # the code is where the template has it - unless the page was photographed upside down (the
        # markers then map it to the opposite corner); the QR decoder reads a code in any orientation,
        # so where it sits tells which way up the page is
        found = None
        for rot in (0, 180):
            p = page if rot == 0 else flipped(page)
            txt, _ = read_qr(p, qr_box)
            if txt:
                found = (txt, p)
                break
        if found is None:
            txt, c = read_qr(page)
            if txt:
                W, H = page.shape[1], page.shape[0]
                qx, qy = (qr_box[0] + qr_box[2]) / 2, (qr_box[1] + qr_box[3]) / 2
                upright = (c[0] - qx) ** 2 + (c[1] - qy) ** 2 <= (c[0] - (W - qx)) ** 2 + (c[1] - (H - qy)) ** 2
                found = (txt, page if upright else flipped(page))
        if found:
            m = re.fullmatch(r"([\w-]+)\|(\w+)\|(\d+)", found[0])
            if m and m.group(1) == self.course["id"] and m.group(2) in self.exams:
                return m.group(2), int(m.group(3)), found[1], ""
            return None, None, page, f"QR code of another course or exam: {found[0]}"
        return None, None, page, "QR code not read"

    def read_page(self, ex: dict, pno: int, page: np.ndarray):
        tpl = ex["template"]["pages"][str(pno)]
        out = {}
        for item, boxes in tpl["items"].items():
            boxes, score = refine_row(page, boxes)
            crops = cell_crops(page, boxes)
            # two views of every cell - inside its border, and a little beyond it: a digit that reaches
            # over the border reads differently in the two, and the average is then unsure
            P_in = self.reader.probs(crops)
            P_wide = self.reader.probs(cell_crops(page, boxes, shrink=-WIDEN)) if WIDEN else P_in
            cl = self.reader.classes
            P = (P_in + P_wide) / 2
            if item != "id":
                P = infer_point(P, crops, cl)
            P = guard_empty(P, crops, cl)
            out[item] = {"P": P, "P_in": guard_empty(P_in, crops, cl), "boxes": boxes,
                         "row_match": score, "gap_ink": gap_ink(page, boxes)}
        return out

    def writer_confusions(self, cells, classes):
        """what the reader gets wrong in THIS student's handwriting, learnt from the ID: when the roster
        says which ID the 9 cells hold (the best roster ID, by its share and ratio - its cell count does
        not matter here) and a cell was read as another digit, the student's k is read as c - so in his
        answers a c may be a k. Used only to lower confidence, never to raise it."""
        if not self.roster or "id" not in cells:
            return []
        P = cells["id"]["P"]
        r, share, lr, mism = roster_match(P, classes, self.roster)
        if not (share >= ROSTER_SHARE and lr >= ROSTER_LOGRATIO) or mism == 0:
            return []
        out = []
        for i, k in enumerate(r):
            c = classes[int(P[i].argmax())]
            if c != k and c.isdigit():
                out.append((classes.index(c), classes.index(k)))
        return out

    def grade(self, images, exam_hint=None, keep_pages=False):
        """images: the pages of ONE student's exam (any order) -> result dict"""
        return self.score(*self.read(images, exam_hint), keep_pages=keep_pages)

    def read(self, images, exam_hint=None):
        """-> (exam id, {page no.: straightened page}, {item: cells}, problems)"""
        pages, problems = {}, []
        exam_id = exam_hint
        for i, img in enumerate(images):
            ex_id, pno, page, why = self.identify(img)
            if ex_id is None:
                problems.append(f"image {i + 1}: {why}")
                continue
            if exam_id and ex_id != exam_id:
                problems.append(f"image {i + 1}: page of {ex_id}, not {exam_id}")
                continue
            exam_id = ex_id
            pages[pno] = page
        cells = {}
        if exam_id is not None:
            for pno, page in pages.items():
                cells.update(self.read_page(self.exams[exam_id], pno, page))
        return exam_id, pages, cells, problems

    def score(self, exam_id, pages, cells, problems, keep_pages=False):
        if exam_id is None:
            return {"exam": None, "problems": problems, "items": []}
        ex = self.exams[exam_id]
        rel_tol = float(self.course.get("rel_tol", 0.01))
        wrong_to_human = bool(self.course.get("wrong_to_human", False))
        classes = self.reader.classes
        confusions = self.writer_confusions(cells, classes)
        items = []
        for q in ex["questions"]:
            if q["kind"] == "open":
                items.append(ItemResult(q["id"], "open", float(q["points"]), "review", reason="open answer: graded by a human"))
                continue
            if q["id"] not in cells:
                items.append(ItemResult(q["id"], q["kind"], float(q["points"]), "review", reason="its page is missing"))
                continue
            rd = nbest_strings(apply_confusions(cells[q["id"]]["P"], confusions), classes)
            crosses = sum(1 for v in cells[q["id"]]["gap_ink"] if v >= GAP_INK) >= GAP_MIN
            items.append(grade_number(q, rd, rel_tol, wrong_to_human, crosses))
        sid = {"id": "", "confidence": 0.0, "valid": False}
        if "id" in cells and self.roster:
            r, share, lr, mism = roster_match(cells["id"]["P"], classes, self.roster)
            if share >= ROSTER_SHARE and lr >= ROSTER_LOGRATIO and mism <= ROSTER_MISMATCH:
                sid = {"id": r, "confidence": round(share, 6), "valid": True, "via": "roster",
                       "log_ratio": round(lr, 2), "cells_corrected": mism}
        if "id" in cells and not sid["valid"]:
            cands = nbest_strings(cells["id"]["P"], classes, k=200, per_cell=3)
            valid = [(s, p) for s, p in cands if id_valid(s)]
            mass = sum(p for _, p in valid)
            if valid:
                s, p = valid[0]
                share = p / mass
                # the most likely reading itself must be the valid one: a less likely reading picked
                # because its check digit works was a wrong ID once in each simulated set
                sid = {"id": s, "confidence": round(share, 4),
                       "valid": share >= ID_SHARE and s == cands[0][0] and p >= ID_MIN_P,
                       "p": round(p, 4), "top_reading": cands[0][0]}
                if self.roster and sid["valid"] and s not in self.roster:
                    # read clearly, but not an enrolled student: a stranger - or two identical misreads the
                    # check digit cannot catch (one '4' read '6' twice: once in 720 simulated students)
                    sid["valid"] = False
                    sid["note"] = "not on the class roster"
            else:
                sid = {"id": cands[0][0] if cands else "", "confidence": 0.0, "valid": False,
                       "top_reading": cands[0][0] if cands else ""}
        auto = [r for r in items if r.route == "auto"]
        review = [r for r in items if r.route != "auto"]
        return {"exam": exam_id, "title": ex["title"], "student_id": sid, "problems": problems,
                "pages_read": sorted(pages), "pages_expected": len(ex["pages"]),
                "items": [r.__dict__ for r in items],
                "auto_points": sum(r.score for r in auto), "auto_max": sum(r.points for r in auto),
                "review_points": sum(r.points for r in review), "max_points": sum(r.points for r in items),
                "complete": not review and len(pages) == len(ex["pages"]),
                "_cells": cells, **({"_pages": pages} if keep_pages else {})}


def load_images(paths):
    from read_page import load_pages
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += load_images(sorted(glob.glob(os.path.join(p, "*"))))
        else:
            out += load_pages(p, 150)
    return out


def report_text(res: dict) -> str:
    if not res.get("exam"):
        return "exam not identified: " + "; ".join(res.get("problems", []))
    sid = res["student_id"]
    if sid["valid"] and sid.get("via") == "roster":
        how = "(matched to the class roster" + (f", {sid['cells_corrected']} cell corrected" if sid.get("cells_corrected") else "") + ")"
    elif sid["valid"]:
        how = "(check digit OK" + (f", confidence {sid['confidence']:.0%}" if sid.get("confidence") else "") + ")"
    elif sid.get("note"):
        how = "(read clearly but NOT on the class roster - to a human)"
    else:
        how = "(not sure - to a human)"
    lines = [f"{res['title']} ({res['exam']})", f"ID: {sid['id'] or '?'}  {how}", ""]
    for r in res["items"]:
        if r["route"] == "auto":
            lines.append(f"  {r['id']:>3}  {r['score']:g} / {r['points']:g}   read '{r['read']}'   auto ({r['confidence']:.0%})"
                         + (f" - {r['reason']}" if r["reason"] and "confidence" not in r["reason"] else ""))
        else:
            guess = f", tool's guess {r['score']:g}" if r["score"] is not None else ""
            lines.append(f"  {r['id']:>3}    ? / {r['points']:g}   read '{r['read']}'   to a human: {r['reason']}{guess}")
    lines += ["", f"graded by the tool: {res['auto_points']:g} of {res['auto_max']:g}",
              f"for a human: {res['review_points']:g} points" if res["review_points"] else "nothing left for a human: final grade "
              f"{res['auto_points']:g} / {res['max_points']:g}"]
    if res["problems"]:
        lines.append("problems: " + "; ".join(res["problems"]))
    return "\n".join(lines)
