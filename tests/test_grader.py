import os

import pytest

from grader import Criterion, Question, Reading, grade_question
from grader.numeric import compare_numeric, parse_numeric
from grader.symbolic import compare_formula


# ---------- numeric parsing / comparison ----------

@pytest.mark.parametrize("text,value,unit", [
    ("4.7 kΩ", 4.7, "kohm"),
    ("4,7kΩ", 4.7, "kohm"),
    ("4.7 kohm", 4.7, "kohm"),
    (r"$3.3\,\text{mA}$", 3.3, "mA"),
    ("10^{-3} F", 1e-3, "F"),
    ("2.2 µF", 2.2, "uF"),
    ("2.2 μF", 2.2, "uF"),           # Greek mu, not micro sign
    ("f_c = 159 Hz", 159, "Hz"),
    ("R_{th} = 1.5e3 Ω", 1500, "ohm"),
    ("0.5", 0.5, ""),
    ("−12 V", -12, "V"),              # unicode minus
])
def test_parse_numeric(text, value, unit):
    a = parse_numeric(text)
    assert a.value == pytest.approx(value)
    assert a.unit == unit


@pytest.mark.parametrize("student,reference,status", [
    ("4.7 kΩ", "4700 Ω", "correct"),
    ("4.65 kΩ", "4700 Ω", "correct"),          # 1.1% off, within 2%
    ("4.1 kΩ", "4700 Ω", "wrong"),
    ("4.7 Ω", "4700 Ω", "prefix_error"),        # k dropped
    ("4.7 MΩ", "4700 Ω", "prefix_error"),       # k read as M
    ("0.0047", "4700 Ω", "prefix_error"),       # unitless, 1e-6 off
    ("4700", "4700 Ω", "correct"),              # unit omitted, implied
    ("4.7 mA", "4700 Ω", "unit_mismatch"),
    ("159 Hz", "159.15 Hz", "correct"),
    ("1.59 kHz", "159.15 Hz", "prefix_error"),  # decimal point slip ×10
    ("10 krad/s", "1591.5 Hz", "factor_2pi"),   # ω instead of f
    ("1000 Hz", "159.15 Hz", "factor_2pi"),     # dropped the 2π
    ("abc", "4700 Ω", "parse_error"),
])
def test_compare_numeric(student, reference, status):
    assert compare_numeric(student, reference).status == status


# ---------- symbolic ----------

@pytest.mark.parametrize("student,reference,status", [
    (r"\frac{1}{2\pi RC}", r"\frac{1}{2 \pi R C}", "equivalent"),
    (r"f_c = \frac{1}{2\pi RC}", r"\frac{1}{2\pi RC}", "equivalent"),          # with lhs
    (r"(2\pi\sqrt{LC})^{-1}", r"\frac{1}{2\pi\sqrt{LC}}", "equivalent"),
    (r"V_{in}\frac{R_2}{R_1+R_2}", r"\frac{V_{in} R_2}{R_2+R_1}", "equivalent"),
    (r"\frac{1}{1+sRC}", r"\frac{1}{sRC+1}", "equivalent"),
    (r"\frac{1}{1+s^2RC}", r"\frac{1}{1+sRC}", "not_equivalent"),
    (r"\frac{1}{2\pi R}", r"\frac{1}{2\pi RC}", "not_equivalent"),              # missing C
    (r"\frac{R_1}{R_1+R_2}", r"\frac{R_2}{R_1+R_2}", "not_equivalent"),         # swapped
    ("1/(2*pi*R*C)", r"\frac{1}{2\pi RC}", "equivalent"),                       # plain text
    (r"\frac{1}{2\pi RC", r"\frac{1}{2\pi RC}", "parse_error"),                 # broken LaTeX
])
def test_compare_formula(student, reference, status):
    assert compare_formula(student, reference).status == status


# ---------- confidence / routing ----------

Q = Question("q1", "RC low-pass, R=10k, C=100n. Find f_c.", [
    Criterion("formula", "final_formula", 4, r"\frac{1}{2\pi RC}"),
    Criterion("value", "final_numeric", 6, "159.15 Hz", rel_tol=0.02),
])


def _two_part(formula, value):
    # prototype: one text per question -> here we encode "formula ; value" and split
    return formula, value


def test_confident_correct():
    q = Question("v", "", [Criterion("value", "final_numeric", 6, "159.15 Hz")])
    r = grade_question(q, [Reading("159 Hz", 0.9), Reading("159 Hz", 0.1)])
    assert r.most_likely_score == 6 and r.confidence == 1.0 and r.route == "auto"


def test_ambiguous_reading_routes_to_review():
    q = Question("v", "", [Criterion("value", "final_numeric", 6, "4700 Ω")])
    # OCR can't decide between 4.7 and 4.1 (7/1 confusion) -> scores differ -> review
    r = grade_question(q, [Reading("4.7 kΩ", 0.6), Reading("4.1 kΩ", 0.4)])
    assert r.route == "review"
    assert r.expected_score == pytest.approx(3.6)
    assert r.score_distribution == {0.0: 0.4, 6.0: 0.6}


def test_ambiguity_that_does_not_change_score_is_fine():
    q = Question("v", "", [Criterion("value", "final_numeric", 6, "4700 Ω")])
    # 4.7 vs 4.70 vs 4.69 all land in tolerance -> confident even though OCR is unsure
    r = grade_question(q, [Reading("4.7 kΩ", 0.5), Reading("4.70 kΩ", 0.3), Reading("4.69kΩ", 0.2)])
    assert r.route == "auto" and r.confidence == 1.0


def test_prefix_error_partial_credit():
    q = Question("v", "", [Criterion("value", "final_numeric", 6, "4700 Ω", prefix_error_credit=0.5)])
    r = grade_question(q, [Reading("4.7 Ω")])
    assert r.most_likely_score == 3.0


def test_parse_error_routes_to_review():
    q = Question("v", "", [Criterion("value", "final_numeric", 6, "4700 Ω")])
    # 30% of the mass is unreadable -> review
    r = grade_question(q, [Reading("4.7 kΩ", 0.7), Reading("4.7 k?", 0.3)])
    assert r.route == "review"
    # a 1% garbage beam must not force a review
    r = grade_question(q, [Reading("4.7 kΩ", 0.99), Reading("4.7 k?", 0.01)])
    assert r.route == "auto" and r.most_likely_score == 6
    # nothing readable at all
    r = grade_question(q, [Reading("???", 1.0)])
    assert r.route == "review" and r.confidence == 0.0


def test_equation_with_non_symbol_lhs_is_not_reduced_to_rhs():
    # 'lim ... = -inf' must not be compared as just '-inf'
    v = compare_formula(r"1 + \frac{1}{1!} + \frac{1}{2!}", r"\lim_{x \to \frac{\pi}{2}+0} \tan x = -\infty")
    assert v.status == "not_equivalent"


def test_unbalanced_brackets_are_a_parse_error():
    assert compare_formula(r"x^{i+2) \times k", r"x^{i+2j} \times k").status == "parse_error"
    assert compare_formula(r"\lim_{x \rightarrow 1} x", r"\lim_{x \rightarrow 1} x").status == "equivalent"


def test_left_hand_side_must_match():
    # same right-hand side, different quantity on the left -> not the same statement
    assert compare_formula(r"V = \frac{V}{R}", r"I=\frac{V}{R}").status == "not_equivalent"
    # 'I' (current) is parsed as the imaginary unit by the LaTeX parser - must still work
    assert compare_formula(r"I = \frac{1}{R} V", r"I=\frac{V}{R}").status == "equivalent"
    assert compare_formula(r"V = R I", r"V=IR").status == "equivalent"
    assert compare_formula(r"1 = \frac{1}{R} V", r"I=\frac{V}{R}").status == "not_equivalent"


def test_current_I_is_not_the_imaginary_unit():
    # Math-Verify's verify() says all of these are equal to 'I = V/R' (I parsed as sqrt(-1)) - we must not
    assert compare_formula(r"I = V R", r"I=\frac{V}{R}").status == "not_equivalent"
    assert compare_formula(r"I = 5", r"I=\frac{V}{R}").status == "not_equivalent"
    assert compare_formula(r"I = \frac{R}{V}", r"I=\frac{V}{R}").status == "not_equivalent"
    assert compare_formula(r"I=\frac{V}{R}", r"I=\frac{V}{R}").status == "equivalent"


# ---------- mixed Hebrew + math answers ----------
from grader.mixed import MixedCriterion, grade_mixed, keyword_present  # noqa: E402
from recog.segment import Segment, smooth_line  # noqa: E402


def test_keyword_with_prefix_and_one_typo():
    assert keyword_present("זה ערך נמוך", "נמוך")
    assert keyword_present("ולכן הוא והנמוך", "נמוך")          # prefixes ו+ה
    assert keyword_present("זה ערך נמוח", "נמוך")              # one edit
    assert not keyword_present("זה ערך גבוה", "נמוך")


RUBRIC_T = [MixedCriterion("ohm", "formula_any", 3, ["V=IR", "I=\\frac{V}{R}"]),
            MixedCriterion("value", "numeric_final", 5, ["5 V"]),
            MixedCriterion("low", "keyword", 2, ["נמוך"])]


def _seg(kind, nbest, p=0.99):
    return Segment(kind, (0, 0, 1, 1), p, nbest)


def test_mixed_confident_full_marks():
    segs = [_seg("hebrew", [("כאשר", -0.1)]), _seg("math", [("V = I \\cdot R", -0.01)]),
            _seg("hebrew", [("ולכן", -0.1)]), _seg("math", [("V = 5 V", -0.01)]),
            _seg("hebrew", [("זה ערך נמוך", -0.2)])]
    r = grade_mixed(RUBRIC_T, segs)
    assert r.most_likely_score == 10 and r.route == "auto"


def test_mixed_uncertain_reading_goes_to_review():
    segs = [_seg("math", [("1 = \\frac{1}{R} V", -0.7), ("I = \\frac{1}{R} V", -1.1)]),   # I or 1 ?
            _seg("math", [("V = 5 V", -0.01)]), _seg("hebrew", [("זה גבוה", -0.1)])]
    r = grade_mixed(RUBRIC_T, segs)
    assert r.route == "review" and 0 < r.criteria[0].p_full < 1


def test_final_answer_uses_last_formula_with_a_number():
    segs = [_seg("math", [("V = 5 V", -0.01)]), _seg("math", [("\\cap ^ { \\sigma }", -0.2)])]
    r = grade_mixed(RUBRIC_T[1:2], segs)
    assert r.most_likely_score == 5


def test_uncertain_layout_goes_to_review():
    segs = [_seg("math", [("V = 5 V", -0.01)], p=0.6)]
    assert grade_mixed(RUBRIC_T[1:2], segs).route == "review"


def test_line_smoothing_fixes_isolated_uncertain_chunk_only():
    assert smooth_line([0.41, 0.76, 0.02])[1] < 0.6           # unsure 'math' between Hebrew words
    assert smooth_line([0.02, 0.98, 0.03])[1] > 0.85          # a confident formula stays math


def test_final_answer_forms():
    assert parse_numeric(r"V \approx 5 V").value == 5
    assert parse_numeric("V = IR = 5 V").value == 5          # chain: the last part is the result
    with pytest.raises(Exception):
        parse_numeric(r"\pi ^ { \prime } > 2 = 5")           # an inequality is not a final answer


def test_confusable_I_gives_partial_credit_probability():
    segs = [_seg("math", [("l = \\frac { 1 } { R } V", -0.3), ("1 = \\frac { 1 } { R } V", -1.6)])]
    r = grade_mixed(RUBRIC_T[:1], segs)
    assert 0.3 < r.criteria[0].p_full < 0.7 and r.route == "review"    # unsure, not confidently wrong


def test_canvas_is_fixed_size():
    import numpy as np
    from recog.segment import CANVAS_W, NORM_H, to_canvas
    for w in (5, 120, 900):
        assert to_canvas(np.ones((NORM_H, w), np.float32)).shape == (NORM_H, CANVAS_W)


def test_math_plausible_for_cross_check():
    from recog.mixed_reader import math_plausible
    assert math_plausible("V = 5 V")
    assert math_plausible("V = I \\cdot R")
    assert math_plausible("I = \\frac { 1 } { R } V")
    assert not math_plausible("\\cap ^ { \\sigma }")            # typical formula-reader output on a Hebrew word
    assert not math_plausible("\\gamma \\varrho _ { \\it 3 }")
    assert not math_plausible("V R")                           # no '=' and no digit: too weak to overrule


def test_word_lm_fixes_lookalike_letters():
    """Image slightly prefers 'באשר', the word LM knows 'כאשר' is far more common."""
    import numpy as np
    from recog.hebrew import WordLM, ctc_prefix_beam_search
    from recog.hebrew_text import CHAR2ID, ID2CHAR, to_logical, to_visual

    def TinyLM():
        return WordLM.from_table({"כאשר": -7.6, "באשר": -11.4}, -19.8)

    K = max(ID2CHAR) + 1
    vis = to_visual("כאשר")                                   # CTC order = visual
    T = 2 * len(vis)
    lp = np.full((T, K), -30.0)
    for i, ch in enumerate(vis):
        lp[2 * i, CHAR2ID[ch]] = 0.0
        lp[2 * i + 1, 0] = 0.0                                # blank between letters
    k = vis.index("כ")                                        # the ambiguous letter: ב 0.6 / כ 0.4
    lp[2 * k, CHAR2ID["כ"]] = np.log(0.4); lp[2 * k, CHAR2ID["ב"]] = np.log(0.6)

    def best(**kw):
        ids = ctc_prefix_beam_search(lp, n_best=1, **kw)[0][0]
        return to_logical("".join(ID2CHAR[i] for i in ids))
    assert best() == "באשר"                                   # image only
    assert best(lm=TinyLM(), lm_weight=0.4) == "כאשר"         # with the word LM


def test_word_lm_course_words_and_prefixes(tmp_path):
    """Technical terms from course_words.txt count as frequent words; prefixed forms are found."""
    import json
    from recog.hebrew import WordLM
    (tmp_path / "words.tsv").write_text("כאשר\t-7.6\nנגד\t-8.4\nבעכבה\t-18.0\n" + "".join(f"מילה{i}\t-12.0\n" for i in range(5)),
                                         encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({"oov_logp": -19.8, "prefix_logp": -3.0, "course_logp": -9.0}),
                                        encoding="utf-8")
    (tmp_path / "course_words.txt").write_text("# comment line: טרנזיסטור is NOT read from here\n"
                                               "עכבה, קיבול (capacitance) נגד מגבר-שרת\n", encoding="utf-8")
    lm = WordLM(str(tmp_path))
    assert lm.logp("עכבה") == -9.0 and lm.logp("קיבול") == -9.0
    assert lm.logp("נגד") == -8.4                      # already more frequent: unchanged
    assert lm.logp("טרנזיסטור") == -19.8               # comments are ignored
    assert lm.logp("בנגד") == -8.4 - 3.0               # one prefix letter
    assert lm.logp("ובעכבה") == -9.0 - 6.0             # two prefix letters, course word stem
    assert lm.logp("בעכבה") == -9.0 - 3.0              # rare in the list, but its stem is a course word
    assert lm.logp("גנד") == -19.8                     # 'ג' is not a prefix
    assert lm.logp("וכאשר") == -19.8                   # prefixes are stripped only from course words
    assert lm.logp("V=5") is None                      # no Hebrew letters: no LM term
    assert lm.logp("מגבר-שרת") == -9.0                 # a hyphenated term is one word, as the reader sees it
    assert "שרת" not in lm.course_words


def test_split_formula_is_joined():
    """'V  =  5V' cut into three chunks, the outer ones classified Hebrew: the lone '=' pulls them back."""
    from types import SimpleNamespace
    from PIL import Image
    from recog.mixed_reader import CONFLICT_CAP, MixedReader

    class FakeFormula:
        def recognize(self, crop, n_best=1):
            return [SimpleNamespace(text="V = 5 V", logp=-0.1, min_token_p=0.9)]

    r = MixedReader.__new__(MixedReader)
    r.frm = FakeFormula()
    line = Image.new("L", (300, 40), 255)
    hebrew_word = [200, 260, "hebrew", 0.95, "", 0.02, "\\cap"]
    chunks = [[10, 40, "hebrew", 0.6, "", 0.50, "\\vee"], [60, 70, "math", 0.5, "x", 0.43, "="],
              [90, 140, "hebrew", 0.8, "", 0.35, "\\varsigma \\lor"], hebrew_word]
    out = r._join_split_formulas(line, chunks)
    assert [c[:3] for c in out] == [[10, 140, "math"], [200, 260, "hebrew"]]
    assert out[0][3] == CONFLICT_CAP and out[0][6] == "V = 5 V"
    # a clearly Hebrew neighbour (raw p_math < 0.15) is never pulled in
    chunks = [[10, 40, "hebrew", 0.9, "", 0.05, "x"], [60, 70, "math", 0.5, "", 0.43, "="], hebrew_word]
    assert r._join_split_formulas(line, chunks) == chunks


def test_lowercase_volt_unit():
    from grader.numeric import compare_numeric, parse_numeric
    assert parse_numeric("V = 5 v").unit == "V"
    assert parse_numeric("3 mv").unit == "mV"
    assert compare_numeric("V = 5 v", "5 V").status == "correct"
    assert compare_numeric("5000 mv", "5 V").status == "correct"


def test_value_on_either_side_of_equals():
    """In a right-to-left line ('המתח = 5V') the formula reader may see '5 V =' or '5 V = V'."""
    from grader.numeric import compare_numeric, parse_numeric
    assert parse_numeric("5 V =").value == 5
    assert compare_numeric("5 V = V", "5 V").status == "correct"
    assert compare_numeric("V = 5 V", "5 V").status == "correct"
    assert compare_numeric("R = 4.7 k\\Omega = 4700 \\Omega", "4.7 kohm").status == "correct"
    assert compare_numeric("V = I R = 5", "5 V").status == "correct"


def _ruled_page():
    """demo answer A pasted on a 'notebook page': light-blue rules, specks, a dark scanner border"""
    import numpy as np
    from PIL import Image, ImageDraw
    ans = Image.open(os.path.join(os.path.dirname(__file__), "..", "data", "demo", "answer_A.png")).convert("L")
    page = Image.new("RGB", (ans.width + 120, ans.height + 160), "white")
    d = ImageDraw.Draw(page)
    for y in range(30, page.height, 37):
        d.line((0, y, page.width, y), fill=(214, 226, 242), width=2)        # rules: light blue
    page.paste(Image.merge("RGB", [ans] * 3), (60, 80), mask=ans.point(lambda v: 255 if v < 160 else 0))
    rng = np.random.default_rng(0)
    for x, y in rng.integers(0, [page.width, page.height], size=(40, 2)):
        d.point((int(x), int(y)), fill=(40, 40, 40))                       # specks
    d.rectangle((page.width - 6, 0, page.width, page.height), fill=(20, 20, 20))   # scanner border
    return page


def test_page_cleaning_and_lines():
    import numpy as np
    from recog.page import clean_scan, find_lines
    page, info = clean_scan(_ruled_page())
    ink = np.asarray(page) < 128
    assert ink[:, -10:].sum() == 0                                    # border gone
    lines = find_lines(page)
    assert len(lines) == 3                                            # the three answer lines, rules ignored


def test_fraction_stays_one_line():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from recog.page import find_lines
    img = Image.new("L", (420, 260), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30) if os.path.isfile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else ImageFont.load_default()
    d.text((150, 40), "x+1", font=f, fill=0)                          # numerator
    d.line((130, 92, 260, 90), fill=0, width=3)                       # slightly slanted bar
    d.text((180, 104), "2", font=f, fill=0)                           # denominator
    d.text((60, 190), "a b c d e f g", font=f, fill=0)                # a separate line below
    lines = find_lines(img)
    assert len(lines) == 2
    y0, y1, _ = lines[0]
    assert y0 < 50 and y1 > 120                                       # numerator, bar and denominator together


def test_short_line_squeezed_under_a_long_one():
    """the end of a sentence written right under it (a notebook): the long line's descenders fill the
    gap, so only the short line's own columns show the valley (Tal's page, 26.10)"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from recog.page import find_lines
    font = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf")
                 if os.path.isfile(p)), None)
    if font is None:
        pytest.skip("no font with Hebrew letters")
    f = ImageFont.truetype(font, 34)
    long, short = "אני בודק אותך איך אתה קורא עברית", "וגם אנגלית."
    img = Image.new("L", (980, 300), 255)
    d = ImageDraw.Draw(img)
    d.text((40, 60), long, font=f, fill=0)
    W, w = d.textlength(long, font=f), d.textlength(short, font=f)
    for k in range(10):                                               # long handwritten descenders (ק, ך)
        x = 60 + k * int((W - w - 60) / 10)                           # under the rest of the line
        d.line((x, 88, x - 4, 126), fill=0, width=3)
    d.text((40 + W - w, 98), short, font=f, fill=0)
    d.text((40, 225), "a separate line", font=f, fill=0)
    lines = find_lines(img)
    assert len(lines) == 3
    (a0, a1, la), (b0, b1, lb) = lines[0], lines[1]
    ink_a, ink_b = np.asarray(la) < 128, np.asarray(lb) < 128
    assert np.where(ink_b.any(0))[0].min() > 40 + W - w - 10          # the short line: only its own letters
    assert a1 >= 120                                                  # the descenders stay with the long line
    assert ink_a[:, 50:70].any()


def test_looped_formula_output_is_detected():
    from recog.mixed_reader import looks_looped
    assert looks_looped("\\log _ { 2 } " + "\\log _ { 4 } " * 7)
    assert not looks_looped("V = I \\cdot R")
    assert not looks_looped("R = 1 0 0 0 0 0 0 \\Omega")      # many zeros are a number, not a loop
    unit = "\\put ( 1 . 5 , 1 ) { \\line ( 1 , 0 ) { 1 0 } } "
    assert looks_looped("\\unitlength . 5 c m " + unit * 4)          # a long unit, 4 times
    assert not looks_looped("\\frac { 1 } { 2 } + \\frac { 1 } { 2 } + \\frac { 1 } { 2 } = \\frac { 3 } { 2 }")


def test_looks_english():
    from recog.english import looks_english
    assert looks_english("Action log") and looks_english("it is incorrect")
    assert not looks_english("V = IR") and not looks_english("x2 + 1") and not looks_english("sin x")
    assert not looks_english("RC") and not looks_english("Vout")
    assert not looks_english("is") and looks_english("is", allow_short=True)
    assert not looks_english("xy", allow_short=True)


def test_english_keyword():
    from grader.mixed import keyword_present
    assert keyword_present("The voltage is LOW here", "low")
    assert keyword_present("it is lower", "lower") and keyword_present("it is lowr", "lower")   # 1 edit for long words
    assert not keyword_present("it is slow", "low")                                          # whole words only
    assert keyword_present("זה ערך נמוך", "נמוך")                                             # Hebrew unchanged


def _english_decision(p_en, eng, widths=(100, 100, 100), doubt=None):
    """layout_checked on a line of blocks (left to right, 30 px tall, 60 px apart; narrower than 18 px = a dot)
    with faked classifiers and readers: p_math 0.02 everywhere (Hebrew script); p_en for the blocks
    that are not dots; per block whether the English reader agrees / has doubts."""
    from types import SimpleNamespace
    from PIL import Image, ImageDraw
    from recog.mixed_reader import MixedReader

    line = Image.new("L", (40 + sum(widths) + 60 * len(widths), 60), 255)
    d = ImageDraw.Draw(line)
    x = 20
    for w in widths:                                  # word gaps of 2 letter heights
        d.rectangle((x, 15, x + w - 1, 45), fill=0)
        x += w + 60
    calls = {"clf": 0, "en": 0, "op": 0}

    def counter(key, values):
        def f(*_a, **_k):
            calls[key] += 1
            return values[calls[key] - 1]
        return f

    doubt = doubt or [False] * len(widths)
    r = MixedReader.__new__(MixedReader)
    r.clf = SimpleNamespace(p_math=counter("clf", [0.02] * len(widths)))
    r.en_clf = SimpleNamespace(p_math=counter("en", p_en))
    r.eng = object()
    r.join_split_formulas = False
    r._opinions = counter("op", [{"math": False, "heb": False, "eng": e, "eng_sure": e, "eng_short": e,
                                  "eng_doubt": e or dt, "f": "", "h": "", "e": "word"} for e, dt in zip(eng, doubt)])
    (segs,) = r.layout_checked(line)
    return [(k, round(pk, 2), note) for k, _, pk, note in reversed(segs)]         # left to right


def test_english_decision_rules():
    from recog.mixed_reader import CONFLICT_CAP
    # classifier and English reader agree -> English, confident
    out = _english_decision([0.95, 0.05, 0.05], [True, False, False])
    assert out[0][0] == "english" and out[0][1] >= 0.8 and out[1][0] == "hebrew" and out[1][1] >= 0.8 and len(out) == 2
    # the classifier half-thinks English, the reader does not confirm -> stays Hebrew, but to review
    out = _english_decision([0.05, 0.6, 0.05], [False, False, False])
    assert [k for k, _, _ in out] == ["hebrew"] and out[0][1] <= CONFLICT_CAP and "Hebrew or English?" in out[0][2]
    # ... unless it stands next to a confirmed English word: then English (still to review)
    out = _english_decision([0.95, 0.75, 0.05], [True, False, False])
    assert [k for k, _, _ in out] == ["english", "hebrew"] and out[0][1] <= CONFLICT_CAP
    assert "English next to English" in out[0][2]
    # the reader reads English words where the classifier sees none -> review
    out = _english_decision([0.05, 0.05, 0.05], [False, False, False], doubt=[False, True, False])
    assert [k for k, _, _ in out] == ["hebrew"] and out[0][1] <= CONFLICT_CAP and "Hebrew or English?" in out[0][2]
    # a Hebrew chunk right next to an English word that might be English too (0.35) -> review
    out = _english_decision([0.95, 0.35, 0.05], [True, False, False])
    assert [k for k, _, _ in out] == ["english", "hebrew"] and out[1][1] <= CONFLICT_CAP
    assert "next to an English word" in out[1][2]


def test_dots_next_to_english():
    from recog.mixed_reader import CONFLICT_CAP
    # a dot between two English words is part of the phrase: one English segment, no review
    out = _english_decision([0.95, 0.95], [True, False, True], widths=(100, 10, 100))
    assert [k for k, _, _ in out] == ["english"] and out[0][1] >= 0.8
    # a dot between an English and a Hebrew word could belong to either -> review
    out = _english_decision([0.95, 0.05], [True, False, False], widths=(100, 10, 100))
    assert [k for k, _, _ in out] == ["english", "hebrew"] and out[1][1] <= CONFLICT_CAP


def test_english_rules_match_tuning_script():
    """recog/mixed_reader.py and its mirror in tune_english_rules.py must not drift apart"""
    import tune_english_rules as t
    from recog import mixed_reader as m
    assert t.SHIPPED == {"tiny": m.TINY, "econf": m.EN_MIN_CONF, "econf_doubt": m.EN_DOUBT_CONF, "short_ok": True,
                         "agree": m.EN_AGREE, "alone": m.EN_ALONE, "pen_min": None, "doubt": m.EN_DOUBT,
                         "context": m.EN_CONTEXT, "boundary": m.EN_BOUNDARY, "pk": "combined", "tiny_en": "absorb",
                         "reader_min": 0.0}


def test_columns_stay_apart():
    """two answers side by side (a gap of many letter heights) are two segments, not one formula"""
    from types import SimpleNamespace
    from PIL import Image, ImageDraw
    from recog.mixed_reader import MixedReader
    line = Image.new("L", (700, 60), 255)
    d = ImageDraw.Draw(line)
    for x0, x1 in ((20, 200), (240, 280), (500, 680)):        # formula, '= 6' close by, formula far away
        d.rectangle((x0, 15, x1, 45), fill=0)
    r = MixedReader.__new__(MixedReader)
    r.clf = SimpleNamespace(p_math=lambda *_: 0.95)
    r.en_clf = None
    r.eng = None
    r.join_split_formulas = False
    r._opinions = lambda *_a, **_k: {"math": True, "heb": False, "eng": False, "eng_sure": False, "eng_short": False,
                                     "eng_doubt": False, "f": "1 + 5 = 6", "h": "", "e": ""}
    (segs,) = r.layout_checked(line)
    assert [(k, b[0], b[2]) for k, b, _, _ in reversed(segs)] == [("math", 20, 281), ("math", 500, 681)]


def test_grade_page_arithmetic_items():
    from types import SimpleNamespace as S
    from grader.page import grade_page, safe_eval
    import math
    seg = lambda nbest, kind="math", pk=0.95: S(kind=kind, nbest=[(t, math.log(p)) for t, p in nbest],   # noqa: E731
                                                p_kind=pk, box=(0, 0, 1, 1))                     # n-best: (text, log p)
    segments = [seg([("1 + 5 = 6", 0.97), ("1 + 5 = 0", 0.03)]),          # right, confidently
                seg([("1 9 - 4 = 1 1", 0.95), ("1 9 - 4 = 1 7", 0.05)]),  # wrong, confidently
                seg([("9 8 - 1 1 = 8 7", 0.5), ("9 8 - 1 1 = 7 8", 0.5)]),  # readings disagree
                seg([("3 6 / 6 = 6", 0.99)], pk=0.5),                       # layout unsure
                seg([("\\textcircled { 2 } 1 2 + 1 2 + 1 3 = 3 7", 0.99)]),  # item number in front
                seg([("7 \\times 7 = 4 9", 0.99)], kind="hebrew")]          # not a formula segment
    rubric = {"items": [{"id": "a", "points": 8, "arithmetic": "1+5"}, {"id": "b", "points": 8, "arithmetic": "19-4"},
                        {"id": "c", "points": 8, "arithmetic": "98-11"}, {"id": "d", "points": 8, "arithmetic": "36/6"},
                        {"id": "e", "points": 6, "arithmetic": "12+12+13"}, {"id": "f", "points": 8, "arithmetic": "7*7"},
                        {"id": "g", "points": 8, "open": "explain"}]}
    res = grade_page(rubric, segments)
    by = {r["id"]: r for r in res["items"]}
    assert (by["a"]["route"], by["a"]["score"]) == ("auto", 8.0)
    assert (by["b"]["route"], by["b"]["score"]) == ("auto", 0.0)
    assert by["c"]["route"] == "review" and "disagree" in by["c"]["reason"]
    assert by["d"]["route"] == "review" and "formula" in by["d"]["reason"]
    assert (by["e"]["route"], by["e"]["score"]) == ("auto", 6.0)
    assert by["f"]["route"] == "review" and "not found" in by["f"]["reason"]
    assert by["g"]["route"] == "review"
    assert (res["auto_points"], res["auto_max"], res["review_points"], res["max_points"]) == (14.0, 22.0, 32.0, 54.0)
    assert safe_eval("70/7") == 10 and safe_eval("__import__('os')") is None


def test_arithmetic_factor_ten_is_wrong():
    """81/9 = 90 is wrong - the half credit for an SI-prefix slip is for answers with units, on request"""
    import math
    from types import SimpleNamespace as S
    from grader.page import grade_page
    segs = [S(kind="math", nbest=[("8 1 / 9 = 9 0", math.log(0.99))], p_kind=0.95, box=(0, 0, 1, 1))]
    r = grade_page({"items": [{"id": "9", "points": 10, "arithmetic": "81/9"}]}, segs)["items"][0]
    assert (r["route"], r["score"]) == ("auto", 0.0)
    r = grade_page({"items": [{"id": "9", "points": 10, "arithmetic": "81/9", "prefix_credit": 0.5}]}, segs)["items"][0]
    assert r["score"] == 5.0


def test_grade_page_named_values():
    """circuits-exam items: 'i = -4 A' found by its name, a boxed answer preferred, units checked"""
    import math
    from types import SimpleNamespace as S
    from grader.page import grade_page, var_key
    seg = lambda nbest, boxed=False, pk=0.95: S(kind="math", nbest=[(t, math.log(p)) for t, p in nbest],  # noqa: E731
                                                p_kind=pk, box=(0, 0, 1, 1), boxed=boxed)
    segments = [seg([("i = 9 A", 0.99)]),                                      # an intermediate i, not boxed
                seg([("\\Longrightarrow \\quad i = - 4 \\, A", 0.99)], boxed=True),   # the boxed final answer
                seg([("\\textcircled { b } P _ { 3 A } = - ( 3 \\times V _ { 1 } ) = - 4 5 w", 0.99)]),
                seg([("V _ { 1 } = 1 5 m V", 0.99)]),                          # wrong prefix: 0 unless asked
                seg([("P _ { 9 V = } + ( - 4 ) ( 9 ) = - 3 6 W", 0.99)])]     # '=' read inside the braces
    rubric = {"items": [{"id": "i", "points": 4, "value": "i", "answer": "-4 A"},
                        {"id": "p3", "points": 4, "value": "P_{3A}", "answer": "-45 W"},
                        {"id": "v1", "points": 4, "value": "V_1", "answer": "15 V"},
                        {"id": "p9", "points": 4, "value": "P_{9V}", "answer": "-36 W"},
                        {"id": "p2", "points": 4, "value": "P_{2A}", "answer": "-26 W"}]}
    by = {r["id"]: r for r in grade_page(rubric, segments)["items"]}
    assert (by["i"]["route"], by["i"]["score"]) == ("auto", 4.0) and "boxed" in by["i"]["reason"]
    assert (by["p3"]["route"], by["p3"]["score"]) == ("auto", 4.0)                 # lowercase w = watts
    assert (by["v1"]["route"], by["v1"]["score"]) == ("auto", 0.0)
    assert (by["p9"]["route"], by["p9"]["score"]) == ("auto", 4.0)
    assert by["p2"]["route"] == "review" and "not found" in by["p2"]["reason"]
    assert var_key("\\mathrm{P}_{3 A}") == var_key("p3a") == "p3a" and var_key("\\dot{\\imath}_1") == "i1"


def test_answer_box_found_and_removed():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from recog.page import clean_scan, find_lines
    img = Image.new("L", (700, 900), 255)        # page-like height: a short vertical stroke is no margin line
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28) if os.path.isfile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else ImageFont.load_default()
    d.text((60, 230), "first line above", font=f, fill=0)
    d.text((120, 305), "V = 15 V", font=f, fill=0)
    d.line((105, 292, 300, 294), fill=0, width=3); d.line((300, 294, 302, 350), fill=0, width=3)   # a frame drawn
    d.line((302, 350, 104, 349), fill=0, width=3); d.line((104, 349, 108, 299), fill=0, width=3)  # in 4 strokes,
    d.text((60, 390), "a line below", font=f, fill=0)                                            # one corner open
    page, info = clean_scan(img)
    assert len(info["answer_boxes"]) == 1
    x0, y0, x1, y1 = info["answer_boxes"][0]
    assert x0 < 110 and x1 > 298 and y0 < 296 and y1 > 347
    ink = np.asarray(page) < 128
    assert not ink[293:296, 150:250].any()                       # the top of the frame is gone
    assert ink[305:345, 120:290].sum() > 200                    # the answer is still there
    assert len(find_lines(page)) == 3


def test_red_ink_and_decimal_points_kept():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from recog.page import clean_scan
    img = Image.new("RGB", (500, 160), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34) if os.path.isfile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else ImageFont.load_default()
    d.text((60, 50), "26", font=f, fill=(255, 139, 137))       # red pen, light in plain luminance
    d.rectangle((106, 80, 108, 82), fill=(255, 139, 137))       # a small decimal point (9 px: a speck by size)
    d.text((116, 50), "67 V", font=f, fill=(255, 139, 137))
    d.rectangle((400, 20, 402, 22), fill=(0, 0, 0))             # a speck far from any writing
    page, _ = clean_scan(img)
    ink = np.asarray(page) < 128
    assert ink[50:95, 60:200].sum() > 300                       # the red writing is ink
    assert ink[79:85, 105:111].any()                            # the decimal point stays
    assert not ink[18:26, 398:406].any()                        # the lone speck goes


def test_named_values_by_page_and_fractions():
    """two questions both ask for 'i': "page" decides; 'i = -5/2 A' is -2.5 A"""
    import math
    from types import SimpleNamespace as S
    from grader.page import grade_page
    seg = lambda t, page: S(kind="math", nbest=[(t, math.log(0.99))], p_kind=0.95, box=(0, 0, 1, 1),  # noqa: E731
                             boxed=False, page=page)
    segments = [seg("i = - 4 A", 1), seg("i = - \\frac { 5 } { 2 } A", 2),
                seg("P _ { 9 A } = - \\frac { 1 1 7 } { 2 } = - 5 8 . 5 W", 2)]
    rubric = {"items": [{"id": "q1", "points": 3, "page": 1, "value": "i", "answer": "-4 A"},
                        {"id": "q2", "points": 3, "page": 2, "value": "i", "answer": "-2.5 A"},
                        {"id": "q2b", "points": 3, "page": 2, "value": "P_{9A}", "answer": "-58.5 W"}]}
    by = {r["id"]: r for r in grade_page(rubric, segments)["items"]}
    assert all((by[k]["route"], by[k]["score"]) == ("auto", 3.0) for k in by), by


# ------------------------------------------------------------------ answer-box forms (forms/)
def test_form_bidi_isolates_formulas():
    from forms.make_exam import bidi
    out = bidi("נגד של 470 Ω מחובר למקור של 12 V. מהו הזרם? (במיליאמפר)")
    assert '<bdi dir="ltr">470 Ω</bdi>' in out and '<bdi dir="ltr">12 V</bdi>' in out
    assert "(במיליאמפר)" in out                                  # Hebrew in parentheses stays Hebrew
    out = bidi("מהו מתח היציאה Vout (המתח על R2)?")
    assert '<bdi dir="ltr">Vout</bdi> (' in out and '<bdi dir="ltr">R2</bdi>)?' in out


def test_form_id_check_digit_and_readings():
    import numpy as np
    from forms.read_form import id_valid, nbest_strings
    assert id_valid("000000018") and not id_valid("000000019") and not id_valid("12345678")
    classes = list("0123456789") + ["-", ".", " ", "?"]
    P = np.full((4, 14), 1e-4)
    for i, (c, p) in enumerate([("5", 0.97), (".", 0.99), ("7", 0.9), (" ", 0.99)]):
        P[i, classes.index(c)] = p
    P[2, classes.index("1")] = 0.09                              # 7 or 1?
    P /= P.sum(1, keepdims=True)
    rd = nbest_strings(P, classes)
    assert rd[0][0] == "5.7" and rd[1][0] == "5.1"
    want = P[0, classes.index("5")] * P[1, classes.index(".")] * P[2, classes.index("7")] * P[3, classes.index(" ")]
    assert abs(rd[0][1] - want) < 1e-3


def test_form_grade_number_routes():
    from forms.read_form import grade_number
    q = {"id": "q1", "points": 20, "kind": "number", "answer": 5.714286}
    r = grade_number(q, [("5.71", 0.98), ("5.11", 0.01)], 0.01, False)
    assert (r.route, r.score) == ("auto", 20.0)
    r = grade_number(q, [("-5.71", 0.98)], 0.01, False)
    assert (r.route, r.score) == ("auto", 0.0)                    # a confident wrong answer: 0
    r = grade_number(q, [("-5.71", 0.98)], 0.01, True)
    assert r.route == "review"                                    # unless wrong answers go to a human
    r = grade_number(q, [("5.71", 0.6), ("5.11", 0.35)], 0.01, False)
    assert r.route == "review" and r.score == 20.0                 # unsure: the guess, but a human decides
    r = grade_number(q, [("5.?1", 0.99)], 0.01, False)
    assert r.route == "review" and "crossed" in r.reason
    r = grade_number(q, [("", 0.99)], 0.01, False)
    assert (r.route, r.score) == ("auto", 0.0)                    # nothing written
    c = {"id": "q6", "points": 15, "kind": "choice", "answer": 3}
    assert grade_number(c, [("3", 0.99)], 0.01, False).score == 15.0
    assert grade_number(c, [("7", 0.99)], 0.01, False).route == "review"


def test_form_pages_found_and_straightened():
    """a printed exam page, turned and put in perspective like a phone photo: markers, QR, cells found"""
    import glob
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image
    pdf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "course_elec101", "E4.pdf")
    if not os.path.isfile(pdf):
        pytest.skip("course forms not built (python -m forms.make_exam)")
    import cv2
    from forms import read_form as R
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "150", "-png", "-f", "2", "-l", "2", pdf, os.path.join(tmp, "p")], check=True)
        page = np.asarray(Image.open(glob.glob(os.path.join(tmp, "p*.png"))[0]).convert("L"), np.float32)
    H, W = page.shape
    src = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    dst = np.float32([[180, 140], [W + 120, 210], [150, H + 230], [W + 210, H + 160]])
    photo = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (W + 400, H + 400), borderValue=120)
    photo = Image.fromarray(np.ascontiguousarray(photo[::-1, ::-1]).astype(np.uint8))       # upside down too

    class Dummy:
        classes = list("0123456789") + ["-", ".", " ", "?"]

        def probs(self, crops):
            return np.full((len(crops), 14), 1 / 14)
    fg = R.FormGrader(os.path.dirname(pdf), reader=Dummy())
    ex, pno, straight, why = fg.identify(photo)
    assert (ex, pno) == ("E4", 2), why
    cells = fg.read_page(fg.exams["E4"], 2, straight)
    for item, c in cells.items():
        want = fg.exams["E4"]["template"]["pages"]["2"]["items"][item]
        assert max(abs(a - b) for x, y in zip(c["boxes"], want) for a, b in zip(x, y)) <= 4


def test_form_decimal_point_inferred_and_point_shift_reviewed():
    import numpy as np
    from forms.read_form import decimal_shift, grade_number, infer_point
    classes = list("0123456789") + ["-", ".", " ", "?"]
    P = np.full((5, 14), 1e-4)
    for i, c in enumerate("2 4  "):                       # '2', a cell that looks empty, '4'
        P[i, classes.index(c)] = 0.99
    P[1, classes.index(".")] = 0.03                       # ... with a trace of a point
    P /= P.sum(1, keepdims=True)
    crops = [np.full((50, 40), 255, np.uint8) for _ in range(5)]
    Q = infer_point(P, crops, classes)
    assert classes[int(Q[1].argmax())] == "."             # between two characters: a decimal point
    P2 = P.copy()
    P2[1, classes.index(".")] = 1e-4                      # no trace at all: stays a gap (-> a human)
    assert classes[int(infer_point(P2, crops, classes)[1].argmax())] == " "
    assert decimal_shift("24", 2.4, 0.01) and decimal_shift("0.0255", 25.53, 0.01) and not decimal_shift("25", 2.4, 0.01)
    q = {"id": "q1", "points": 15, "kind": "number", "answer": 2.4}
    r = grade_number(q, [("24", 0.995)], 0.01, False)
    assert r.route == "review" and "decimal point" in r.reason   # never an automatic 0 for a shifted point


def test_beam_search_grammar_keeps_only_allowed_readings():
    """a reader asked for '<number> <letter A-D>' never returns a reading outside that grammar, even when
    its own best guess is outside it ('16 6' -> '16 C')"""
    import numpy as np
    from recog.beam import beam_search
    vocab = ["<s>", "</s>", "1", "6", " ", "C", "G"]
    seq_pref = {1: "1", 2: "6", 3: " ", 4: "6"}        # the model's favourite: '16 6' then end

    def step(ids):
        out = np.full((ids.shape[0], len(vocab)), -5.0)
        for b, row in enumerate(ids):
            want = seq_pref.get(len(row), "</s>")
            out[b, vocab.index(want)] = 3.0
            if len(row) == 4:
                out[b, vocab.index("C")] = 2.5      # second choice at the letter position
        return out

    free = beam_search(step, start_id=0, eos_id=1, num_beams=2, max_new_tokens=6)
    assert "".join(vocab[t] for t in free[0].tokens[1:-1]) == "16 6"

    import regex
    rx = regex.compile(r"\d{1,2} [A-D]")

    def allowed(tokens):
        text = "".join(vocab[t] for t in tokens)
        ok = [i for i in range(2, len(vocab)) if rx.fullmatch(text + vocab[i], partial=True)]
        if rx.fullmatch(text):
            ok.append(1)
        return np.array(ok, dtype=np.int64)

    held = beam_search(step, start_id=0, eos_id=1, num_beams=2, max_new_tokens=6, allowed=allowed)
    texts = ["".join(vocab[t] for t in h.tokens[1:] if t != 1) for h in held]
    assert texts[0] == "16 C"
    assert all(rx.fullmatch(t) for t in texts)


def test_cvl_writers_split_without_overlap():
    import os
    from forms.cells import DIGITS
    d = os.path.join(DIGITS, "cvl", "train")
    if not os.path.isdir(d):
        pytest.skip("CVL digits not downloaded")
    from forms.cells import load_source
    fx, _ = load_source("cvl_fit")
    hx, _ = load_source("cvl_hold")
    tx, _ = load_source("cvl_train")
    assert len(fx) + len(hx) == len(tx) and len(hx) > 500


def test_roster_matches_ids_and_refuses_strangers(tmp_path):
    """the class roster recovers an ID with a doubtful cell, and a student who is not on the roster is
    never matched to a near-twin ID (two digits apart) of someone who is"""
    import numpy as np
    import forms.read_form as R
    classes = list("0123456789") + ["-", ".", " ", "?"]
    true_id = "123456782"
    assert R.id_valid(true_id)
    P = np.full((9, len(classes)), 1e-4)
    for i, ch in enumerate(true_id):
        P[i, classes.index(ch)] = 0.99
    P[4, classes.index("5")], P[4, classes.index("6")] = 0.45, 0.54      # cell 5: '6' slightly ahead of the true '5'
    P /= P.sum(1, keepdims=True)
    twins = []
    for i in range(9):
        for j in range(i + 1, 9):
            for a in "0123456789":
                for b in "0123456789":
                    t = list(true_id)
                    if a == t[i] or b == t[j]:
                        continue
                    t[i], t[j] = a, b
                    t = "".join(t)
                    if R.id_valid(t):
                        twins.append(t)
    twins = twins[:40]
    r, share, lr, mism = R.roster_match(P, classes, [true_id] + twins)
    assert r == true_id and share >= R.ROSTER_SHARE and lr >= R.ROSTER_LOGRATIO and mism <= R.ROSTER_MISMATCH
    r, share, lr, mism = R.roster_match(P, classes, twins)                 # own ID missing
    assert not (share >= R.ROSTER_SHARE and lr >= R.ROSTER_LOGRATIO and mism <= R.ROSTER_MISMATCH)
    f = tmp_path / "roster.csv"
    f.write_text("name,id\nA,123456782\nB,23456783\n", encoding="utf-8")      # a spreadsheet dropped a leading 0
    assert R.id_valid("023456783")
    assert R.load_roster(str(f)) == ["123456782", "023456783"]


def test_writer_confusions_from_the_id_lower_confidence():
    """a student whose '1' the reader takes for '7' in his ID (the roster knows the true ID): in his
    answers a confident '7' is no longer confident"""
    import numpy as np
    import forms.read_form as R
    classes = list("0123456789") + ["-", ".", " ", "?"]
    true_id = "103282380"
    P = np.full((9, len(classes)), 1e-4)
    for i, ch in enumerate(true_id):
        P[i, classes.index(ch)] = 0.99
    P[0, classes.index("1")], P[0, classes.index("7")] = 0.2, 0.79      # his '1' looks like a 7
    P /= P.sum(1, keepdims=True)
    fg = R.FormGrader.__new__(R.FormGrader)
    fg.roster = [true_id, "123456782"]
    conf = fg.writer_confusions({"id": {"P": P}}, classes)
    assert conf == [(classes.index("7"), classes.index("1"))]
    ans = np.full((2, len(classes)), 1e-4)
    ans[0, classes.index("7")], ans[1, classes.index("2")] = 0.999, 0.999      # '72'
    ans /= ans.sum(1, keepdims=True)
    adj = R.apply_confusions(ans, conf)
    assert adj[0, classes.index("7")] < 0.6 and adj[0, classes.index("1")] > 0.4
    assert np.allclose(adj.sum(1), 1.0) and abs(adj[1, classes.index("2")] - ans[1, classes.index("2")]) < 1e-9
    fg.roster = []
    assert fg.writer_confusions({"id": {"P": P}}, classes) == []


def test_copy_sheets_people_and_labels(tmp_path):
    """two people's filled copy sheets in one scan: every kept cell is labelled with what was printed,
    and with who wrote it (a person = his pages in a row)"""
    import shutil
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm not installed")
    from forms.cells import DIGITS
    if not os.path.isfile(os.path.join(DIGITS, "semeion.data")):
        pytest.skip("Semeion digits not downloaded (python download_digits.py)")
    from forms.collect import load_real, read_filled, simulate_filled
    a = simulate_filled(1, seed=3, digit_source="semeion", out_dir=str(tmp_path / "s"))
    b = simulate_filled(1, seed=4, digit_source="semeion", out_dir=str(tmp_path / "s"))
    st = read_filled(a + b, "scan", out_root=str(tmp_path / "real"))
    assert st["people"] == 2 and st["pages_not_found"] == 0 and st["rows_kept"] >= 40
    crops, labels, who = load_real(str(tmp_path / "real"), with_people=True)
    assert len(crops) == st["cells"] and set(who) == {"scan#1", "scan#2"}
    only1 = load_real(str(tmp_path / "real"), people={"scan#1"})[0]
    assert 0 < len(only1) < len(crops)


def test_hebrew_three_widths_pool_readings():
    """reading a Hebrew line at three widths returns pooled, sorted readings (and no fewer than asked)"""
    from recog.hebrew import DEFAULT_DIR, HebrewRecognizer
    line = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hebrew_lines", "000.png")
    if not (os.path.isfile(os.path.join(DEFAULT_DIR, "model.onnx")) and os.path.isfile(line)):
        pytest.skip("Hebrew model or test line missing")
    rec = HebrewRecognizer()
    one = rec.recognize(line, n_best=3, tta=False)
    three = rec.recognize(line, n_best=3, tta=True)
    assert len(three) == 3 and three[0].text
    assert all(three[i].logp >= three[i + 1].logp for i in range(len(three) - 1))
    assert one[0].text in {c.text for c in rec.recognize(line, n_best=5, tta=True)} or three[0].text


def test_selfcheck_formula_data_result():
    """no answer key: the student's own formula + data -> is his result right? (Tal's page, 26.10)"""
    from grader.selfcheck import check_statements, statements
    st = []
    for tex in ("1) a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c=9", "2) a^{2}+b^{2}=c^{2} \\vert 4^{2}+2^{2}=\\sqrt{20}",
                "3) A=\\pi r^{2} \\vert r=5", "A=78.54"):
        st += statements(tex)
    assert [(c.problem, c.verdict) for c in check_statements(st)] == [
        (1, "wrong"), (2, "right value, wrong equality"), (3, "right")]
    assert "c = 5" in check_statements(st)[0].text
    # a letter the formula does not have: most likely a misread digit (the 4 of 4^2 read as y)
    assert [c.verdict for c in check_statements(statements("a^{2}+b^{2}=c^{2} \\vert y^{2}+2^{2}=\\sqrt{20}"))] == ["misread?"]
    # the formula reader's spaced tokens; a missing value -> cannot check
    assert [c.verdict for c in check_statements(statements("A = \\pi r ^ { 2 } \\vert r = 5") + statements("A = 7 8 . 5 4"))] == ["right"]
    assert [c.verdict for c in check_statements(statements("A = \\pi r ^ { 2 }") + statements("A = 7 8 . 5 4"))] == ["cannot check"]
    # an unsure reading never gives a sure verdict
    assert not check_statements(statements("a^{2}+b^{2}=c^{2} \\vert a=3, b=4", conf=0.4) + statements("c=9"))[0].sure


def test_selfcheck_finds_a_missing_value_in_a_piece_read_as_text():
    """'A = pi r^2 | r=5' where the layout called '|r=5' Hebrew: the formula reader's reading of that piece
    gives the missing r, and the check can still be made (only as a suggestion)"""
    from grader.selfcheck import check_pages

    class Seg:
        def __init__(self, kind, x, text, p_kind=0.9, formula=None):
            self.kind, self.box, self.nbest, self.p_kind = kind, (x, 0, x + 10, 10), [(text, -0.1)], p_kind
            if formula:
                self.formula_nbest = [(formula, -0.2)]
    page = {"lines": [[Seg("hebrew", 100, "ה", formula="\\frac { 3 } { }"), Seg("math", 200, "A = \\pi r ^ { 2 }"),
                       Seg("hebrew", 420, "צון", 0.5, formula="\\mid r = 5")],
                      [Seg("math", 240, "A = 7 8 . 5 4", 1.0)]]}
    (c,) = check_pages([page])
    assert c.verdict == "right" and "r=5" in c.text and not c.sure


def test_equations_checked_against_the_circuit_drawn():
    """Tal's circuit (E1 R1 | R2 to ground | R3 E2, node A, I1 I2 I3 into / out of A): Kirchhoff's law as the arrows
    give it, a sign slip, the node voltage right and wrong, a loop equation - checked against the circuit itself"""
    import os
    from grader.circuit import truth_netlist
    from grader.circuit_check import check_against, circuit_statements, parse_side
    from grader.selfcheck import check_pages
    net = truth_netlist(os.path.join(os.path.dirname(__file__), "..", "data", "my_scans", "tal_circuit_truth.json"))
    assert str(parse_side("R_{2}(E_{1}R_{3}+E_{2}R_{1})")) == "R_2*(E_1*R_3 + E_2*R_1)"

    def verdicts(tex):
        return [c.verdict for c in check_against(circuit_statements(tex), net)]
    assert verdicts("I_1 + I_3 = I_2") == ["right"]
    assert verdicts("I_{1} = I_{2} + I_{3}") == ["wrong"]
    assert verdicts("V_A = \\frac{R_2 (E_1 R_3 + E_2 R_1)}{R_1 R_2 + R_1 R_3 + R_2 R_3}") == ["right"]
    assert verdicts("V_A = \\frac{R_2 (E_1 R_3 + E_2 R_1)}{R_1 R_2 + R_1 R_3}") == ["wrong"]
    assert verdicts("E_1 - I_1 R_1 - I_2 R_2 = 0") == ["right"]
    assert verdicts("E_1 - I_1 R_1 + I_2 R_2 = 0") == ["wrong"]
    assert verdicts("x = 5") == []
    vals = {"R1": "1k", "R2": "2k", "R3": "1k", "E1": "6V", "E2": "3V"}      # the values written by the parts
    for c in net.components:
        c.value = vals[c.id]
    assert verdicts("I_2 = 1 . 8 \\mathrm { m A }") == ["right"] and verdicts("V_A = 3 V") == ["wrong"]
    for c in net.components:
        c.value = None

    class Seg:
        def __init__(self, x, text):
            self.kind, self.box, self.nbest, self.p_kind = "math", (x, 0, x + 10, 10), [(text, -0.05)], 0.95
    page = {"lines": [[Seg(100, "I _ { 1 } + I _ { 3 } = I _ { 2 }")]], "circuits": [{"box": (0, 0, 1, 1), "netlist": net}]}
    checks = [c for c in check_pages([page]) if c.problem == "circuit 1"]
    assert [c.verdict for c in checks] == ["right"] and checks[0].sure


def test_math_validation_steps_units_equations():
    """the tool checks the mathematics itself: each '=' of a computation, the result from the formula and
    data, units set aside, equations solved, plain arithmetic checked"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for t in lines:
            st += statements(t)
        return check_statements(st)
    c = verdicts("a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c^{2}=9+16=24", "c=\\sqrt{24}")
    assert c[-1].verdict == "wrong" and "9+16" in c[-1].text and "c = 5" in c[-1].text      # the slip is named
    assert verdicts("a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c^{2}=9+16=25", "c=5")[-1].verdict == "right"
    assert verdicts("a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c^{2}=25")[-1].verdict == "right"          # c from c^2 = 25
    assert verdicts("V = I R \\vert I = 2 A , R = 5 \\Omega", "V = 10 V")[-1].verdict == "right"     # units aside
    assert verdicts("V = I R \\vert I = 2 A , R = 5 \\Omega", "V = 7 V")[-1].verdict == "wrong"
    assert verdicts("2x+3=7", "x=2")[-1].verdict == "right"                                          # an equation
    assert verdicts("2x+3=7", "x=5")[-1].verdict == "wrong"
    assert [x.verdict for x in verdicts("12+12+13=34")] == ["wrong step"]                            # arithmetic
    assert [x.verdict for x in verdicts("7 \\times 8 = 56")] == ["right"]
    assert verdicts("A \\approx \\pi r^{2} \\vert r = 5", "A \\approx 78.5")[-1].verdict == "right"  # rounded, approx


def test_two_formula_readers_vote():
    """the second formula reader's reading joins the first's; where they agree the votes add up"""
    from grader.selfcheck import _mix
    mixed = _mix([("y ^ { 2 } + 4", 0.9), ("4 ^ { 2 } + 4", 0.1)], [("4^{2}+4", 1.0)])
    assert mixed[0][0].replace(" ", "") == "4^{2}+4" and abs(mixed[0][1] - 0.55) < 1e-9
    assert _mix([("x", 1.0)], None) == [("x", 1.0)]


def test_math_validation_no_false_alarm_on_the_next_part():
    """a computation in other letters under a formula is the next part of the answer, not a misread digit
    (KFUPM EE201 quiz 2 key: 'P_{3A} = -(3 x v_1) = -45 W' under 'V_1 = V_2 + 9')"""
    from grader.selfcheck import check_statements, statements
    st = statements("V_{1} = V_{2} + 9 \\Longrightarrow V_{1} = ( 3 \\times 2 ) + 9") + statements("V_{1} = 15 V")
    st += statements("P_{3A} = - ( 3 \\times v_{i} ) = - 45 w")
    assert "misread?" not in [c.verdict for c in check_statements(st)]


def test_number_format_for_the_gpu(monkeypatch):
    """RTX 2080 Ti (Turing, no fast bfloat16): a 2B reader in float32 when it fits, the language model in
    float16 (speed; NaN -> reloaded), an 8B reader in 4 bits; Ampere and newer: bfloat16; an old Xeon: float32"""
    torch = pytest.importorskip("torch")
    from recog import gpu
    monkeypatch.delenv("EXAM_GRADER_DTYPE", raising=False)
    monkeypatch.delenv("EXAM_GRADER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(10.3 * 2**30), 11 * 2**30))
    assert gpu.pick(2.13) == ("cuda", torch.float32)
    assert gpu.pick(1.72, fast=True) == ("cuda", torch.float16)
    assert gpu.needs_4bit(8.77) and not gpu.needs_4bit(2.13)
    assert gpu.fallback_dtype(2.13) == torch.float32 and gpu.fallback_dtype(8.77) == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(6.9 * 2**30), 11 * 2**30))
    assert gpu.pick(2.13) == ("cuda", torch.float16)              # the LM already took 3.4 GB
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    assert gpu.pick(2.13) == ("cuda", torch.bfloat16)
    monkeypatch.setenv("EXAM_GRADER_DTYPE", "fp32")
    assert gpu.pick(1.72, fast=True)[1] == torch.float32
    monkeypatch.delenv("EXAM_GRADER_DTYPE")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gpu, "_cpu_bf16", lambda: False)
    monkeypatch.setattr(gpu, "_ram_free_gb", lambda: 57.0)
    assert gpu.pick(2.13) == ("cpu", torch.float32)               # Xeon Gold 6130: no bf16 instructions
    monkeypatch.setattr(gpu, "_ram_free_gb", lambda: 5.0)
    assert gpu.pick(2.13) == ("cpu", torch.bfloat16)              # ... unless memory is short
    monkeypatch.setattr(gpu, "_cpu_bf16", lambda: True)
    assert gpu.pick(2.13) == ("cpu", torch.bfloat16)


def test_heavy_readers_run_after_the_layout_one_at_a_time():
    """deferred (read_page.read_pages): the pieces keep the ordinary readings until finish(), which runs
    each heavy reader over all its pieces and parks the others first - one model on the GPU at a time"""
    from types import SimpleNamespace
    from PIL import Image
    from recog.mixed_reader import MixedReader
    from recog.segment import Segment
    log = []

    class Heavy:
        def __init__(self, name, text):
            self.name, self.text = name, text

        def park(self):
            log.append(("park", self.name))

        def recognize(self, crop, n_best=1, **kw):
            log.append(("read", self.name))
            return [SimpleNamespace(text=self.text, logp=-0.5)]

    r = MixedReader.__new__(MixedReader)
    r.hebrew_tta, r.defer_heavy, r._pending, r.english_vlm = True, True, [], "x"
    lm, mumer, vlm = Heavy("lm", ""), Heavy("mumer", "c=5"), Heavy("vlm", "hi there")
    ctx = Heavy("ctx", "אני בודק")
    ctx.lm = lm
    r._ctx, r._mumer, r._en_vlm = ctx, mumer, vlm
    crop = Image.new("L", (40, 20), 255)
    heb, frm, eng = (Segment("hebrew", (0, 0, 1, 1), 0.9, [("אני בודכ", -1.0)]),
                     Segment("math", (0, 0, 1, 1), 0.9, [("c=9", -0.2)]), Segment("english", (0, 0, 1, 1), 0.9, [("hi thera", -1.0)]))
    r._heavy_job("english", eng, crop, 3)
    r._heavy_job("math2", frm, crop, 1)
    r._heavy_job("hebrew", heb, crop, 3)
    assert heb.nbest[0][0] == "אני בודכ" and not hasattr(frm, "math2") and log == []
    r.finish()
    assert heb.nbest[0][0] == "אני בודק" and frm.math2[0][0] == "c=5" and frm.nbest[0][0] == "c=9"
    assert eng.nbest[0][0] == "hi there" and r._pending == []
    reads = [x for x in log if x[0] == "read"]
    assert reads == [("read", "ctx"), ("read", "mumer"), ("read", "vlm")]      # grouped, in this order
    assert log.index(("park", "lm")) < log.index(("read", "mumer"))           # the LM leaves before Uni-MuMER
    assert log.index(("park", "mumer")) < log.index(("read", "vlm"))


def test_a_lone_second_reading_is_not_certain():
    """Uni-MuMER returns one reading: it weighs with its own probability, not 1 (Tal's page, line '2) a²+b²=c² |
    4²+2²=√20': Pix2Text fairly sure, p 0.61; Uni-MuMER unsure of a garbled '2 7 a²...', p 0.18 -> Pix2Text's stays
    on top); readings that differ only in how items are separated ('|', '\\vert', ',') are one reading"""
    from grader.selfcheck import _mix, _raw
    p2t = [("2 \\vert a ^ { 2 } + b ^ { 2 } = c ^ { 2 } \\vert y ^ { 2 } + 2 ^ { 2 } = \\sqrt { 2 0 }", -0.488),
           ("2 ) a ^ { 2 } + b ^ { 2 } = c ^ { 2 } \\vert y ^ { 2 } + 2 ^ { 2 } = \\sqrt { 2 0 }", -2.319)]
    mumer = [("2 7 a ^ { 2 } + b ^ { 2 } = c ^ { 2 } 4 ^ { 2 } + 2 ^ { 2 } = \\sqrt { 2 0 }", -1.694)]
    assert _mix(_raw(p2t), _raw(mumer))[0][0].startswith("2 \\vert")
    p2t = [("\\vert \\vert a ^ { 2 } + b ^ { 2 } = c ^ { 2 } \\vert a = 3 \\vert b = 4", -2.259),
           ("1 ) a ^ { 2 } + b ^ { 2 } = c ^ { 2 } \\vert a = 3 \\vert b = 4", -2.337)]
    mumer = [("1 ) a ^ { 2 } + b ^ { 2 } = c ^ { 2 } | a = 3 , b = 4", -0.873)]
    mixed = _mix(_raw(p2t), _raw(mumer))
    assert mixed[0][0].startswith("1 )") and len(mixed) == 2 and abs(sum(p for _, p in mixed) - 1) < 1e-9


def test_foreign_letter_is_a_digit_in_the_other_reading():
    """Tal's page, problem 2: Pix2Text reads '2 | a²+b²=c² | y²+2²=√20' (4 read as y), Uni-MuMER '2 7 a²+b²=c²
    4²+2²=√20' (no separators, but the 4): the y, foreign to a²+b²=c², is taken as the other reading's 4 ->
    'right value, wrong equality' instead of 'misread?'; not when the other reading has a longer number there"""
    from grader.selfcheck import check_statements, statements
    best = r"2 \vert a ^ { 2 } + b ^ { 2 } = c ^ { 2 } \vert y ^ { 2 } + 2 ^ { 2 } = \sqrt { 2 0 }"
    st = statements(best, piece=0)
    other = r"2 7 a ^ { 2 } + b ^ { 2 } = c ^ { 2 } 4 ^ { 2 } + 2 ^ { 2 } = \sqrt { 2 0 }"
    checks = check_statements(st, others={0: [best, other]})
    assert [c.verdict for c in checks] == ["right value, wrong equality"] and "y taken as 4" in checks[0].text
    assert checks[0].confidence <= 0.5                                     # a repair is a suggestion
    assert [c.verdict for c in check_statements(st)] == ["misread?"]
    longer = r"a ^ { 2 } + b ^ { 2 } = c ^ { 2 } 1 4 ^ { 2 } + 2 ^ { 2 } = \sqrt { 2 0 }"
    assert [c.verdict for c in check_statements(st, others={0: [longer]})] == ["misread?"]


def test_circuit_netlist_kcl_and_solution():
    """Tal's two-source circuit: KCL at A from the arrows drawn, node A's voltage by nodal analysis
    (Millman), and a reading with a flipped source and a reversed arrow caught by the comparison"""
    import sympy as sp
    from grader.circuit import Netlist, compare, parse_json, truth_netlist
    t = truth_netlist(os.path.join(os.path.dirname(__file__), "..", "data", "my_scans", "tal_circuit_truth.json"))
    assert t.problems() == [] and t.kcl() == {"A": "I1 + I3 = I2"}
    s = t.solve()
    E1, E2, R1, R2, R3 = sp.symbols("E1 E2 R1 R2 R3")
    assert sp.simplify(s["V_A"] - (E1 / R1 + E2 / R3) / (1 / R1 + 1 / R2 + 1 / R3)) == 0
    assert sp.simplify(s["I1"] + s["I3"] - s["I2"]) == 0
    reading = parse_json('```json\n{"components": [{"id": "V1", "type": "voltage source", "nodes": ["a", "g"], "plus": "a"},'
                         '{"id": "X", "type": "resistor", "nodes": ["a", "b"]}, {"id": "R2", "type": "resistor", "nodes": ["b", "g"]},'
                         '{"id": "Y", "type": "resistor", "nodes": ["b", "c"]}, {"id": "V2", "type": "voltage_source",'
                         '"nodes": ["c", "g"], "plus": "g"},], "ground": "g", "node_labels": {"b": "A"}, "currents": ['
                         '{"label": "I1", "through": "X", "from": "a", "to": "b"}, {"label": "I2", "through": "R2", "from": "b", "to": "g"},'
                         '{"label": "I3", "through": "Y", "from": "b", "to": "c"}]}\n```')
    c = compare(Netlist.from_json(reading), t)
    assert c["topology_exact"] and c["polarity"] == "1/2" and c["current_directions"] == "2/3" and c["kcl"] == "0/1"
    assert Netlist.from_json(reading).kcl() == {"A": "I1 = I2 + I3"}


def test_wiring_from_strokes_and_boxes():
    """a drawn loop: source on the left, resistor on the right, a ground below; one wire broken by a lifted pen
    and one wire crossing another without joining it - two nodes, the ground one is 0"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    m = np.zeros((400, 500), bool)
    m[50:54, 50:450] = True                         # top wire
    m[300:304, 50:450] = True                       # bottom wire
    m[50:304, 50:54] = True                         # left side (the source sits on it)
    m[50:304, 446:450] = True                       # right side (the resistor)
    m[300:304, 240:246] = False                     # a lifted pen: a 6 px break in the bottom wire
    m[302:340, 300:304] = True                      # the ground's stub
    m[20:380, 150:153] = True                       # a stray line crossing top and bottom...
    comps = [("voltage_dc", (30, 140, 75, 210)), ("resistor", (425, 130, 470, 230)), ("gnd", (285, 335, 320, 360)),
             ("crossover", (140, 40, 165, 65)), ("crossover", (140, 290, 165, 315))]   # ...but through crossovers
    for cls, (x0, y0, x1, y1) in comps:
        if cls in ("voltage_dc", "resistor"):
            m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True    # the symbol's own ink
    net = netlist_from_boxes(m, comps, [((480 - 60, 240, 480, 260), "10k")])
    kinds = sorted((c.type, tuple(sorted(c.nodes))) for c in net.components)
    assert net.ground == "0"
    assert len(kinds) == 2 and kinds[0][1] == kinds[1][1] and "0" in kinds[0][1], kinds
    r = next(c for c in net.components if c.type == "resistor")
    assert r.value == "10k"


def test_label_reader_shapes():
    """the circuit-label CRNN: crops of any width in one batch, one reading each"""
    pytest.importorskip("torch")
    from PIL import Image, ImageDraw
    from recog.labels import CHARS, CRNN, batch, decode, prep
    im = Image.new("L", (120, 40), 255)
    ImageDraw.Draw(im).text((5, 10), "R1 10k", fill=0)
    arrs = [prep(im), prep(im.resize((50, 30)))]
    out = CRNN().eval()(batch(arrs))
    assert out.shape[0] == 2 and out.shape[2] == len(CHARS) + 1 and arrs[0].shape[0] == 32
    reads = decode(out, CHARS, [-(-x.shape[1] // 4) for x in arrs])
    assert len(reads) == 2 and all(0.0 <= c <= 1.0 for _, c in reads)


def test_label_beam_search_with_what_labels_look_like():
    """an O the network slightly prefers to a 0 after a 1: greedy decoding reads '1Ok', the beam search weighed by a
    character n-gram model of labels reads '10k' (and keeps a repeated letter apart by its blank)"""
    import numpy as np
    from recog.labels import CHARS, CharLM, beam_decode
    lm = CharLM(["10k", "100k", "1k", "R1", "20k", "4.7k", "10V", "R2"] * 5, n=4)
    assert lm.logp("1", "0") > lm.logp("1", "O")

    def frames(*rows):                                   # each row: {char: probability}, the rest the blank's
        lp = np.full((len(rows), len(CHARS) + 1), 1e-6)
        for t, row in enumerate(rows):
            for ch, p in row.items():
                lp[t, 0 if ch == "-" else CHARS.index(ch) + 1] = p
            lp[t, 0] += max(0.0, 1 - sum(lp[t, 1:]))
        return np.log(lp / lp.sum(1, keepdims=True))
    lp = frames({"1": 0.9}, {"-": 0.9}, {"O": 0.5, "0": 0.45}, {"-": 0.9}, {"k": 0.9})
    assert beam_decode(lp, CHARS, None, bonus=0.0)[0] == "1Ok"
    text, sure = beam_decode(lp, CHARS, lm)
    assert text == "10k" and 0.0 < sure <= 1.0
    lp2 = frames({"1": 0.9}, {"0": 0.9}, {"-": 0.9}, {"0": 0.9}, {"k": 0.9})
    assert beam_decode(lp2, CHARS, lm)[0] == "100k"
    # knowing the part a label stands by: a first letter the network barely prefers as 'C' is an 'R' by a resistor
    from recog.labels import KIND_CODE, KindLM, kind_of
    klm = CharLM([KIND_CODE["R"] + t for t in ["R1", "R2", "R3", "10k", "2k"] * 5] +
                 [KIND_CODE["C"] + t for t in ["C1", "C2", "10µF", "1n"] * 5], n=5)
    lp3 = frames({"C": 0.46, "R": 0.44}, {"-": 0.9}, {"1": 0.9})
    assert beam_decode(lp3, CHARS, KindLM(klm, "R"))[0] == "R1" and beam_decode(lp3, CHARS, KindLM(klm, "C"))[0] == "C1"
    parts = [("resistor", (100, 100, 200, 140)), ("capacitor", (400, 100, 440, 200))]
    assert kind_of((120, 60, 160, 90), parts) == "R" and kind_of((450, 130, 480, 160), parts) == "C"
    assert kind_of((900, 900, 930, 930), parts) == "?"


def test_compare_symmetric_circuit_and_ac_source():
    """a voltage divider fits its transcription under two renamings of its nodes: the comparison takes the one under
    which the source's '+' agrees; a source drawn as AC has no polarity to count"""
    from grader.circuit import Component, Netlist, compare
    truth = Netlist([Component("V1", "voltage_source", ["N1", "0"], plus="N1"), Component("R1", "resistor", ["N1", "N2"]),
                     Component("R2", "resistor", ["N2", "0"])], ground="0")
    read = Netlist([Component("V1", "voltage_source", ["a", "b"], plus="a"), Component("R1", "resistor", ["a", "c"]),
                    Component("R2", "resistor", ["c", "b"])])
    c = compare(read, truth)
    assert c["topology_exact"] and c["polarity"] == "1/1", c
    read.components[0].ac, read.components[0].plus = True, None
    assert compare(read, truth)["polarity"] == "0/0"
    assert not any("polarity" in p for p in read.problems())


def test_circuit_label_text():
    from recog.circuit_reader import label_text
    assert [label_text(t) for t in ("R _ { 2 }", "1 k \\Omega", "I _ { 1 } \\rightarrow", "\\mathrm { 1 0 m A }")] == \
        ["R2", "1kΩ", "I1", "10mA"]


def test_wire_lost_along_a_stretch_is_bridged():
    """the bottom wire of a loop lost for 120 px (drawn on a notebook rule, erased with it): its two ends face each
    other with nothing between - one wire; ends that do not face each other stay apart"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    m = np.zeros((400, 600), bool)
    m[50:54, 50:550] = True                          # top wire
    m[300:304, 50:550] = True                        # bottom wire ...
    m[300:304, 240:360] = False                      # ... lost in the middle
    m[50:304, 50:54] = True                          # left (the source)
    m[50:304, 546:550] = True                        # right (the resistor)
    comps = [("voltage_dc", (30, 140, 75, 210)), ("resistor", (525, 130, 570, 230))]
    for cls, (x0, y0, x1, y1) in comps:
        m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    net = netlist_from_boxes(m, comps)
    assert len(net.nodes()) == 2 and all(sorted(c.nodes) == net.nodes() for c in net.components), net.components
    m[300:345, 236:240] = True                       # the left stub turns down at its end: the ends do not face each other
    net2 = netlist_from_boxes(m, comps)
    assert len(net2.nodes()) == 3, net2.components


def test_dependent_sources_and_ideal_opamp_solved():
    """a current-controlled current source (2 I_x, I_x the current R1 carries), a voltage-controlled voltage source
    (3 V_A) and an inverting amplifier with an ideal op-amp (gain -Rf/R1) - solved by nodal analysis"""
    from grader.circuit import Component, Current, Netlist
    net = Netlist([Component("I1", "current_source", ["n1", "0"], plus="n1", value="1"),
                   Component("R1", "resistor", ["n1", "0"], value="10"),
                   Component("G1", "dependent_current_source", ["n2", "0"], plus="n2", value="2 I_x"),
                   Component("R2", "resistor", ["n2", "0"], value="5")],
                  [Current("I_x", "R1", "n1", "0")], ground="0", node_labels={"n1": "A", "n2": "B"})
    assert net.solve() == {"V_A": 10, "V_B": 10, "I_x": 1} and not net.problems()
    net2 = Netlist([Component("V1", "voltage_source", ["n1", "0"], plus="n1", value="2"),
                    Component("E1", "dependent_voltage_source", ["n2", "0"], plus="n2", value="3V_A"),
                    Component("R2", "resistor", ["n2", "0"], value="4")], ground="0", node_labels={"n1": "A", "n2": "B"})
    assert net2.solve() == {"V_A": 2, "V_B": 6}
    net2.components[1].value = None                  # what it follows not read: no solution, a problem to check
    assert net2.solve() == {} and any("follows" in p for p in net2.problems())
    amp = Netlist([Component("V1", "voltage_source", ["a", "0"], plus="a", value="1"),
                   Component("R1", "resistor", ["a", "m"], value="1k"), Component("Rf", "resistor", ["m", "o"], value="10k"),
                   Component("OA1", "opamp", ["0", "m", "o"], roles={"In+": "0", "In-": "m", "Out": "o"})],
                  ground="0", node_labels={"o": "out"})
    assert amp.solve()["V_out"] == -10
    amp.components[3].roles = {"In-": "m", "Out": "o"}  # a leg not read: not solved
    assert amp.solve() == {}


def test_wiring_dependent_source_and_terminals():
    """a diamond (dependent current source) labelled '2I_x' on the left, a resistor on the right, and the top wire
    running on to two open terminals - the source's label says what it follows, the terminals are the ports"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    m = np.zeros((400, 700), bool)
    m[50:54, 50:650] = True                          # top wire, on to terminal a at the right
    m[300:304, 50:650] = True                        # bottom wire, on to terminal b
    m[50:304, 50:54] = True                          # left (the diamond)
    m[50:304, 446:450] = True                        # the resistor
    comps = [("dep_current", (25, 140, 80, 215)), ("resistor", (425, 130, 470, 230)),
             ("terminal", (645, 40, 670, 65)), ("terminal", (645, 290, 670, 315))]
    for cls, (x0, y0, x1, y1) in comps[:2]:
        m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    net = netlist_from_boxes(m, comps, [((90, 160, 140, 190), "2I_x")], polarity=False)
    dep = next(c for c in net.components if c.type == "dependent_current_source")
    assert dep.value == "2I_x" and len(net.components) == 2, net.components
    assert len(net.ports) == 2 and set(net.ports) == set(dep.nodes), (net.ports, dep.nodes)


def test_first_order_circuit_switches_and_time_constant():
    """V1 (10 V) - switch - R1 (2k) - node A - R2 (2k) and C1 (1u) to ground: steady state, Thevenin resistance and
    the time constant in each switch state; the checks try both states and do not call a moment's value wrong"""
    from grader.circuit import Component, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([Component("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"),
                   Component("S1", "switch", ["n0", "n1"]), Component("R1", "resistor", ["n1", "a"], value="2k"),
                   Component("R2", "resistor", ["a", "0"], value="2k"), Component("C1", "capacitor", ["a", "0"], value="1u")],
                  ground="0", node_labels={"a": "A"})
    assert net.solve() == {}                         # the switch's state not given
    assert net.solve(switches={"S1": "closed"})["V_A"] == 5 and net.solve(switches={"S1": "open"})["V_A"] == 0
    assert net.resistance_seen("a", "0", {"S1": "closed"}) == 1000
    assert abs(float(net.time_constant({"S1": "closed"})) - 1e-3) < 1e-12 and abs(float(net.time_constant({"S1": "open"})) - 2e-3) < 1e-12

    def verdicts(tex):
        return [c.verdict for c in check_against(circuit_statements(tex), net)]
    assert verdicts("\\tau = 1 ms") == ["right"] and verdicts("\\tau = 2 m s") == ["right"] and verdicts("\\tau = 3 ms") == ["wrong"]
    assert verdicts("V_A = 5") == ["right"] and verdicts("V_A = 7") == []
    sym = Netlist([Component("V1", "voltage_source", ["n0", "0"], plus="n0"), Component("R1", "resistor", ["n0", "a"]),
                   Component("C1", "capacitor", ["a", "0"])], ground="0", node_labels={"a": "A"})
    checks = check_against(circuit_statements("\\tau = R_1 C_1"), sym)
    assert [c.verdict for c in checks] == ["right"] and checks[0].sure
    assert [c.verdict for c in check_against(circuit_statements("\\tau = 2 R_1 C_1"), sym)] == ["wrong"]


def test_switched_first_order_responses_checked():
    """x(t) after a switching: x(inf) + (x(0+) - x(inf)) e^(-t/tau) for the node voltages, the parts' voltages and
    currents - every switching the drawing allows is tried; a start that does not fit is not judged sure (it may
    come from an earlier switching); a two-way switch (common end, throw 1, throw 2) moves between its throws"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, parse_side
    assert str(parse_side("10 e^{-t/RC} u(t)", timed=True, known={"r", "c"})) == "10*exp(-t/(C*R))"
    assert str(parse_side("5(1-e^{-1000t})\\,mA", timed=True)) == "0.005 - 0.005*exp(-1000*t)"
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"]),
                   C("R1", "resistor", ["n1", "a"], value="2k"), C("R2", "resistor", ["a", "0"], value="2k"),
                   C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})
    r = net.first_order_response({"S1": "open"}, {"S1": "closed"})
    t = __import__("sympy").Symbol("t")
    assert r["values"]["v_C1"].subs(t, 0) == 0 and r["values"]["i_C1"].subs(t, 0) == __import__("sympy").Rational(1, 200)

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("v_C(t) = 5(1 - e^{-1000t})") == [("right", True)]             # S1 closes: tau = (R1 || R2) C1
    assert checks("v_C(t) = 5 e^{-t/\\tau}") == [("right", True)]               # S1 opens: tau = R2 C1
    assert checks("i_C(t) = 5 e^{-1000t}\\,mA") == [("right", True)]
    assert checks("V_A(t) = 5 - 5e^{-t/(R_1 R_2 C_1/(R_1+R_2))}") == [("right", True)]
    assert checks("v_C(t) = 10(1 - e^{-1000t})") == [("wrong", True)]
    assert checks("v_C(t) = 5(1 - e^{-2000t})") == [("wrong", True)]
    assert checks("v_C(t) = 5 - 2e^{-1000t}") == [("wrong", False)]             # its end and tau fit, not its start
    assert checks("v_C(t) = 5") == [("right", False)]                            # the steady state
    rl = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12", label="V_s"),
                  C("S1", "switch", ["n0", "n1"], label="S"), C("R1", "resistor", ["n1", "a"], value="4", label="R"),
                  C("L1", "inductor", ["a", "0"], value="2", label="L")], ground="0", node_labels={"a": "A"})
    assert checks("i_L(t)=\\frac{V_s}{R}\\left(1-e^{-\\frac{R}{L}t}\\right)", rl) == [("right", True)]
    assert checks("v_L(t) = 12e^{-2t}", rl) == [("right", True)]
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", label="E"), C("S1", "switch", ["n0", "n1"]),
                  C("R1", "resistor", ["n1", "a"], label="R"), C("C1", "capacitor", ["a", "0"], label="C")],
                 ground="0", node_labels={"a": "A"})
    assert checks("v_C(t) = E(1-e^{-t/RC})", rc) == [("right", False)]          # cut off before: taken uncharged
    assert checks("i(t) = \\frac{E}{R} e^{-t/RC}", rc) == [("right", False)]   # one loop: its current
    two = Netlist([C("V1", "voltage_source", ["p", "0"], plus="p", value="10"), C("S1", "switch", ["c", "p", "0"]),
                   C("R1", "resistor", ["c", "a"], value="1k"), C("C1", "capacitor", ["a", "0"], value="2u")],
                  ground="0", node_labels={"a": "A"})
    assert not two.problems() and two.solve(switches={"S1": "2"})["V_A"] == 0
    assert checks("v_C(t) = 10e^{-500t}", two) == [("right", True)]               # throw 1 -> 2: it discharges
    assert checks("v_C(t) = 10e^{-1000t}", two) == [("wrong", True)]
    two.components[1].state = "2"                                                # drawn at throw 2: it charges
    assert checks("v_C(t) = 10e^{-500t}", two) == [("right", False)]


def test_second_order_and_laplace_domain_answers_checked():
    """a series RLC switched on: the Laplace-domain response (any order: what each C / L held carried over, the DC
    source a step) - x(t) answers in time, X(s) answers, and H(s) / Z(s), which the drawing never names"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rlc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"]),
                   C("R1", "resistor", ["n1", "n2"], value="10"), C("L1", "inductor", ["n2", "a"], value="0.1"),
                   C("C1", "capacitor", ["a", "0"], value="1m")], ground="0", node_labels={"a": "A"})

    def checks(tex, n=rlc):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    # alpha = 50, w_d = 86.6: underdamped; before the switch closes the capacitor is cut off - taken uncharged
    assert checks("v_C(t) = 10 - 11.547 e^{-50t}\\sin(86.603t + 60^\\circ)") == [("right", False)]
    assert checks("i(t) = 1.1547e^{-50t}\\sin(86.6t)") == [("right", False)]
    assert checks("v_C(t) = 10 - 10e^{-50t}\\cos(86.6t)") == [("wrong", False)]         # its final value fits only
    assert checks("v_C(t) = 5 - 5e^{-50t}\\cos(86.6t)") == [("wrong", True)]
    assert checks("V_C(s) = \\frac{100000}{s(s^2+100s+10000)}") == [("right", False)]
    assert checks("V_C(s) = \\frac{10}{s}") == [("right", True)]                       # the switch opens: it holds
    lp = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", label="V_i"), C("R1", "resistor", ["i", "o"], label="R"),
                  C("C1", "capacitor", ["o", "0"], label="C")], ground="0", node_labels={"o": "o", "i": "i"})
    assert checks("H(s) = \\frac{1}{1 + sRC}", lp) == [("right", True)]
    assert checks("H(s)=\\frac{1/RC}{s+1/RC}", lp) == [("right", True)]
    assert checks("H(s) = \\frac{1}{sRC}", lp) == [("wrong", True)]
    assert checks("Z_{in}(s) = R + \\frac{1}{sC}", lp) == [("right", True)]
    amp = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", value="1"), C("R1", "resistor", ["i", "m"], value="1k"),
                   C("C1", "capacitor", ["m", "y"], value="1u"), C("R2", "resistor", ["m", "y"], value="10k"),
                   C("OA1", "opamp", ["0", "m", "y"])], ground="0")
    amp.components[-1].roles = {"In+": "0", "In-": "m", "Out": "y"}
    assert checks("H(s) = -\\frac{R_2/R_1}{1 + sR_2C_1}", amp) == [("right", True)]
    assert checks("H(s) = -10", amp) == [("wrong", True)]


def test_diode_circuits_solved_in_both_models():
    """each diode on or off, the one consistent way - ideal and with a 0.7 V drop (the problem says which: an
    answer that one model gives is right, not sure; one neither gives is wrong); values with prefixes inside a
    formula ('\\frac{10 - 0.7}{2k}') read in base units"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, parse_side
    assert abs(float(parse_side("\\frac{10 - 0.7}{2k}")) - 0.00465) < 1e-12 and float(parse_side("\\frac{2k}{4}")) == 500
    assert float(parse_side("10\\,\\mathrm{k\\Omega}")) == 10000 and str(parse_side("2s + 1")) == "2*s + 1"
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                   C("D1", "diode", ["a", "b"], plus="b"), C("R2", "resistor", ["b", "0"], value="1k")],
                  ground="0", node_labels={"a": "A", "b": "B"})
    sol, state = net.solve_diodes(vd=0.7)
    assert state == {"D1": "on"} and abs(float(sol["V_B"]) - 4.65) < 1e-9
    net.components[2].plus = "a"                                                   # turned round: off
    assert net.solve_diodes(vd=0.7)[1] == {"D1": "off"}
    net.components[2].plus = "b"

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_B = 4.65") == [("right", False)] and checks("V_B = 5") == [("right", False)]
    assert checks("I_D = \\frac{10 - 0.7}{2k}") == [("right", False)] and checks("V_B = 3") == [("wrong", True)]
    # the parts' own quantities with the diode as it turned out: P_R2 = 4.65 V x 4.65 mA (0.7 V), I of the one loop
    assert checks("P_{R2} = 21.6 mW") == [("right", False)] and checks("I = 4.65 mA") == [("right", False)]
    assert checks("P_{R2} = 50 mW") == [("wrong", True)] and checks("P = 46.5 mW") == [("right", False)]
    gate = Netlist([C("V1", "voltage_source", ["p", "0"], plus="p", value="5"), C("V2", "voltage_source", ["q", "0"], plus="q", value="0"),
                    C("D1", "diode", ["p", "y"], plus="y"), C("D2", "diode", ["q", "y"], plus="y"),
                    C("R1", "resistor", ["y", "0"], value="1k")], ground="0", node_labels={"y": "Y"})
    assert gate.solve_diodes(vd=0.7)[1] == {"D1": "on", "D2": "off"}
    assert checks("I_{D2} = 0", gate) == [("right", True)]                           # both models agree


def test_parts_voltages_currents_powers_and_resonance_checked():
    """V_{R1}, I_{R1}, P_{R1}, I_s, P_s, one loop's I: by a part's name, in any choice of reference directions (none
    is drawn); the load that takes the most power (R_L = R_th); w_0 / f_0 with one L and one C; AC power from a source
    written in time ('10cos(1000t)'): its amplitudes"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10", label="E"),
                   C("R1", "resistor", ["a", "n0"], value="1k", label="R_1"), C("R2", "resistor", ["a", "0"], value="4k", label="R_2")],
                  ground="0", node_labels={"a": "A"})

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_{R1} = 2") == checks("I_{R1} = 2 mA") == checks("P_{R_2} = 16mW") == [("right", True)]
    assert checks("P_E = 20 mW") == checks("I_s = 2mA") == checks("P_{R1}=I^2 R_1") == [("right", True)]
    assert checks("V_{R2} - V_{R1} = 10") == [("right", True)]                    # V_R1 taken the other way round
    assert checks("V_{R1} + V_{R2} = 7") == [("wrong", True)] and checks("I = 3 mA") == [("wrong", True)]
    mp = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12"), C("R1", "resistor", ["n0", "a"], value="2k"),
                  C("R2", "resistor", ["a", "0"], value="2k"), C("RL", "resistor", ["a", "0"], label="R_L")],
                 ground="0", node_labels={"a": "A"})
    assert checks("R_L = 1k", mp) == [("right", False)] and checks("R_L = 2k", mp) == []
    ac = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10cos(1000t)", ac=True),
                  C("R1", "resistor", ["n0", "a"], value="10"), C("L1", "inductor", ["a", "b"], value="10m"),
                  C("C1", "capacitor", ["b", "0"], value="100u")], ground="0", node_labels={"a": "A", "b": "B"})
    assert checks("\\omega_0 = 1000", ac) == checks("I = 1", ac) == [("right", True)]
    assert checks("P_R = 5 W", ac) == [("right", True)] and checks("P_R = 2.5 W", ac) == [("wrong", True)]
    assert checks("P_R = 10 W", ac) == [("wrong", True)]        # 10cos(1000t): an amplitude - its RMS value is 7.07


def test_dc_and_ac_sources_superposed():
    """a DC and an AC source together: each group alone (C open for the DC one, phasors for the AC one), added up -
    'v_C(t) = 5 + 7.07 cos(1000t - 45°)' checked in time"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rc = Netlist([C("V1", "voltage_source", ["n1", "0"], plus="n1", value="5"),
                  C("V2", "voltage_source", ["n2", "n1"], plus="n2", value="10cos(1000t)", ac=True),
                  C("R1", "resistor", ["n2", "a"], value="1k"), C("C1", "capacitor", ["a", "0"], value="1u")],
                 ground="0", node_labels={"a": "A"})
    assert rc.solve("phasor") == {} and rc.solve() == {}
    dc, ph = rc.superposed()["V_A"]
    assert dc == 5 and abs(complex(ph[1000]) - (5 - 5j)) < 1e-9

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), rc)]
    assert checks("v_C(t) = 5 + 7.07\\cos(1000t - 45^\\circ)") == [("right", True)]
    assert checks("i_C(t) = 7.07\\cos(1000t + 45^\\circ)\\,mA") == [("right", True)]
    assert checks("v_A(t) = 5 + 7.07\\cos(1000t + 45^\\circ)") == [("wrong", True)]
    assert checks("v_A(t) = 7.07\\cos(1000t - 45^\\circ)") == [("wrong", True)]       # the DC part left out


def test_numbers_from_the_problem_text_not_judged_against_a_drawing_of_names():
    """a drawing that names its parts (E, R_1, R_2) without values: an answer in numbers took them from the problem's
    text - 'cannot check' for the human (with what the circuit gives in its names), never 'wrong'; an answer in the
    names, or a relation between the circuit's quantities, is judged"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    sym = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", label="E"), C("R1", "resistor", ["n0", "a"], label="R_1"),
                   C("R2", "resistor", ["a", "0"], label="R_2")], ground="0", node_labels={"a": "A"})

    def checks(tex, n=sym):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_A = 5") == checks("V_A = E/2") == checks("I_{R1} = 2 mA") == [("cannot check", False)]
    assert checks("V_A = \\frac{E R_1}{R_1 + R_2}") == [("wrong", True)]
    assert checks("\\frac{V_A}{R_2} + \\frac{V_A - E}{R_1} = 0") == [("right", True)]
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", label="E"), C("S1", "switch", ["n0", "n1"]),
                  C("R1", "resistor", ["n1", "a"], label="R"), C("C1", "capacitor", ["a", "0"], label="C")],
                 ground="0", node_labels={"a": "A"})
    assert checks("v_C(t) = 10(1 - e^{-1000t})", rc) == [] and checks("\\tau = 1 ms", rc) == []
    assert checks("\\tau = 2RC", rc) == [("wrong", True)]


def test_transformer_and_two_port_read_and_checked():
    """an ideal transformer (its coils from where the wires meet it, the ratio label '1:2' beside it; its far side
    measured from its own S-) and a T network between four open terminals (z / y / h / transmission parameters) -
    drawn without a ground symbol, the reference guessed"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit import turns_ratio
    from grader.circuit_check import check_against, circuit_statements
    assert turns_ratio("1:2") == 2 and str(turns_ratio("N_1:N_2")) == "N_2/N_1"
    m = np.zeros((400, 900), bool)
    m[100:104, 50:350] = True
    m[300:304, 50:350] = True
    m[100:304, 50:54] = True
    m[100:104, 450:800] = True
    m[300:304, 450:800] = True
    m[100:304, 796:800] = True
    comps = [("voltage_dc", (25, 170, 80, 230)), ("resistor", (170, 80, 230, 125)), ("transformer", (340, 90, 460, 315)),
             ("resistor", (775, 170, 820, 230))]
    for cls, (x0, y0, x1, y1) in comps:
        if cls != "transformer":
            m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    m[110:295, 360:380] = True
    m[110:295, 420:440] = True
    labels = [((380, 40, 420, 70), "1:2"), ((150, 40, 200, 70), "1Ω"), ((830, 180, 870, 210), "4Ω"), ((90, 180, 130, 210), "10V")]
    net = netlist_from_boxes(m, comps, labels, polarity=False)
    t = next(c for c in net.components if c.type == "transformer")
    v = next(c for c in net.components if c.type == "voltage_source")
    assert t.value == "1:2" and set(t.roles) == {"P+", "P-", "S+", "S-"} and t.roles["P-"] in v.nodes
    assert any("dots" in p_ for p_ in net.problems())                   # the dots are not read: suggestions only
    v.plus, t.state = v.nodes[0], "dots read"

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("I_{R2} = 2.5") == checks("V_{R2} = 10") == checks("I_{R1} = 5") == [("right", True)]
    assert checks("V_{R2} = 20") == [("wrong", True)]
    m = np.zeros((400, 900), bool)
    m[100:104, 60:840] = True
    m[300:304, 60:840] = True
    m[100:304, 448:452] = True
    comps = [("resistor", (220, 80, 300, 125)), ("resistor", (600, 80, 680, 125)), ("resistor", (425, 170, 475, 240)),
             ("terminal", (40, 90, 65, 115)), ("terminal", (40, 290, 65, 315)), ("terminal", (835, 90, 860, 115)),
             ("terminal", (835, 290, 860, 315))]
    for cls, (x0, y0, x1, y1) in comps[:3]:
        m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    tee = netlist_from_boxes(m, comps, [((240, 40, 280, 70), "2Ω"), ((620, 40, 660, 70), "4Ω"), ((490, 190, 530, 220), "6Ω")],
                             polarity=False)
    assert len(tee.terminals) == 4 and not tee.problems()
    assert checks("z_{11} = 8", tee) == checks("z_{21} = 6", tee) == checks("h_{21} = -0.6", tee) == [("right", True)]
    assert checks("t_{11} = \\frac{4}{3}", tee) == [("right", True)] and checks("z_{11} = 2", tee) == [("wrong", True)]


def test_mesh_and_nodal_equations_in_the_students_own_unknowns():
    """mesh currents I_1, I_2 (circular arrows the reader does not take) and node voltages the drawing does not label:
    the equations solved together - right when every unknown is one of the circuit's currents / voltages (either
    direction of a mesh current), a suggestion of wrong when none is"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("E1", "voltage_source", ["p", "0"], plus="p", value="10", label="E_1"),
                   C("R1", "resistor", ["p", "a"], value="2k", label="R_1"), C("R3", "resistor", ["a", "0"], value="4k", label="R_3"),
                   C("R2", "resistor", ["a", "q"], value="2k", label="R_2"),
                   C("E2", "voltage_source", ["q", "0"], plus="q", value="5", label="E_2")], ground="0", node_labels={"a": "A"})

    def checks(*eqs):
        st = [x for e in eqs for x in circuit_statements(e)]
        return [(c.verdict, c.sure) for c in check_against(st, net)]
    assert checks("I_1 (R_1 + R_3) - I_2 R_3 = E_1", "I_2 (R_2 + R_3) - I_1 R_3 = -E_2") == [("right", True)]
    assert checks("I_1 (R_1 + R_3) + I_2 R_3 = E_1", "I_2 (R_2 + R_3) + I_1 R_3 = E_2") == [("right", True)]   # I_2 the other way
    assert checks("6k I_1 - 4k I_2 = 10", "-4k I_1 + 6k I_2 = -5") == [("right", True)]
    assert checks("I_1 (R_1 + R_3) - I_2 R_3 = E_1", "I_2 (R_2 + R_3) - I_1 R_3 = E_2") == [("wrong", False)]
    assert checks("\\frac{V_1 - E_1}{R_1} + \\frac{V_1}{R_3} + \\frac{V_1 - E_2}{R_2} = 0") == [("right", True)]


def test_bjt_bias_solved_in_the_region_that_fits():
    """NPN / PNP with their legs read: V_BE 0.7 V; active (I_C = beta I_B, beta from the label 'β=100' by it) or
    saturation (V_CE 0.2 V) or cutoff, whichever fits; I_B, I_C, V_CE, the base's Thevenin V_TH / R_TH, beta as a
    name; beta not written: answers in numbers cannot be checked"""
    from grader.circuit import Component as C, Netlist, transistor_beta
    from grader.circuit_check import check_against, circuit_statements
    assert transistor_beta("β=100") == 100 and transistor_beta("h_{FE} = 150") == 150 and str(transistor_beta(None)) == "beta"
    q = C("Q1", "npn", ["b", "c", "e"], value="β=100")
    q.roles = {"Base": "b", "Collector": "c", "Emitter": "e"}
    net = Netlist([C("V1", "voltage_source", ["v", "0"], plus="v", value="12", label="V_{CC}"),
                   C("R1", "resistor", ["v", "b"], value="100k", label="R_1"), C("R2", "resistor", ["b", "0"], value="22k", label="R_2"),
                   C("RC", "resistor", ["v", "c"], value="4.7k", label="R_C"), C("RE", "resistor", ["e", "0"], value="1k", label="R_E"), q],
                  ground="0", node_labels={"b": "B", "c": "C", "e": "E"})
    sol, regions = net.solve_transistors()
    assert regions == {"Q1": "active"} and abs(float(sol["I_C"]) - 1.2299e-3) < 1e-6

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_{TH} = \\frac{R_2 V_{CC}}{R_1 + R_2}") == checks("I_C = \\beta I_B") == [("right", True)]
    assert checks("I_B = \\frac{V_{TH} - 0.7}{R_{TH} + (\\beta+1) R_E}") == checks("I_C = 1.23 mA") == [("right", True)]
    assert checks("I_C = 1.4 mA") == [("wrong", True)]
    sat = Netlist([C("V1", "voltage_source", ["v", "0"], plus="v", value="12"), C("RB", "resistor", ["v", "b"], value="100k"),
                   C("RC", "resistor", ["v", "c"], value="2k"), C("Q1", "npn", ["b", "c", "0"], value="β=100")], ground="0")
    sat.components[-1].roles = {"Base": "b", "Collector": "c", "Emitter": "0"}
    assert sat.solve_transistors()[1] == {"Q1": "saturation"} and checks("I_C = 5.9 mA", sat) == [("right", True)]
    q.value = None
    assert checks("I_C = 1.23 mA") == [("cannot check", False)]


def test_common_emitter_amplifier_small_signal():
    """a CE amplifier (divider bias, R_E bypassed, coupling capacitors, load R_L): the bias, then each transistor's
    hybrid-pi model at mid-band - g_m = I_C / V_T (25 or 26 mV: the course's; one of them right is not sure), r_pi,
    A_v, R_in, R_out; R_C \\| R_L read as in parallel"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, parse_side
    assert str(parse_side("-g_m (R_C \\| R_L)")) == "-R_C*R_L*g_m/(R_C + R_L)" and str(parse_side("r_\\pi", True)) == "r_pi"
    q = C("Q1", "npn", ["b", "c", "e"], value="β=100")
    q.roles = {"Base": "b", "Collector": "c", "Emitter": "e"}
    net = Netlist([C("V1", "voltage_source", ["v", "0"], plus="v", value="12", label="V_{CC}"),
                   C("R1", "resistor", ["v", "b"], value="100k", label="R_1"), C("R2", "resistor", ["b", "0"], value="22k", label="R_2"),
                   C("RC", "resistor", ["v", "c"], value="4.7k", label="R_C"), C("RE", "resistor", ["e", "0"], value="1k", label="R_E"),
                   C("CE", "capacitor", ["e", "0"], value="100u", label="C_E"), q,
                   C("VS", "voltage_source", ["s", "0"], plus="s", value="10m cos(1000t)", ac=True, label="v_s"),
                   C("C1", "capacitor", ["s", "b"], value="10u", label="C_1"), C("C2", "capacitor", ["c", "o"], value="10u", label="C_2"),
                   C("RL", "resistor", ["o", "0"], value="10k", label="R_L")], ground="0", node_labels={"b": "B", "c": "C", "o": "o"})
    ss = net.small_signal(0.025)
    assert abs(float(ss["av"]) + 157.29) < 0.05 and abs(float(ss["rin"]) - 1826.8) < 0.5 and float(ss["rout"]) == 4700
    sts = [x for t in ("g_m = \\frac{I_C}{V_T}", "A_v = -g_m (R_C \\| R_L)", "R_{in} = R_1 \\| R_2 \\| r_\\pi", "R_{out} = R_C",
                       "g_m = 49.2 mS", "A_v = -g_m R_C") for x in circuit_statements(t)]
    got = [(c.verdict, c.sure) for c in check_against(sts, net)]
    assert got == [("right", True)] * 4 + [("right", False), ("wrong", True)], got


def test_mosfet_bias_and_common_source_amplifier():
    """an N-channel MOSFET with V_t and K written by it ('K = 0.5mA/V^2'; k_n means twice K): the region that fits -
    saturation I_D = K (V_GS - V_t)^2 (the root with V_ov >= 0 and V_DS >= V_ov) - then g_m = 2 K V_ov at mid-band; K as
    a name ('2K V_{ov}' is not 2 kilo), \\sqrt read"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit import Component as C, Netlist, fet_params
    from grader.circuit_check import check_against, circuit_statements
    m = np.zeros((400, 600), bool)
    m[200:204, 50:250] = True
    m[160:240, 290:300] = True
    got = netlist_from_boxes(m, [("fet", (250, 150, 350, 250))],
                             [((370, 170, 450, 200), "V_t = 1V"), ((370, 210, 470, 240), "K = 0.5mA/V^2")], polarity=False)
    assert got.components[0].params == {"V_t": "1V", "K": "0.5mA/V^2"}
    q = C("M1", "nmos", ["g", "d", "s"])
    q.roles, q.params = {"Gate": "g", "Drain": "d", "Source": "s"}, {"V_t": "1V", "K": "0.5mA/V^2"}
    assert fet_params(q) == (__import__("sympy").Rational(1, 2000), 1)
    amp = Netlist([C("V1", "voltage_source", ["v", "0"], plus="v", value="12", label="V_{DD}"),
                   C("R1", "resistor", ["v", "g"], value="2M", label="R_1"), C("R2", "resistor", ["g", "0"], value="1M", label="R_2"),
                   C("RD", "resistor", ["v", "d"], value="6k", label="R_D"), C("RS", "resistor", ["s", "0"], value="2k", label="R_S"),
                   C("CS", "capacitor", ["s", "0"], value="100u"), q,
                   C("VS", "voltage_source", ["in", "0"], plus="in", value="1m cos(1000t)", ac=True),
                   C("C1", "capacitor", ["in", "g"], value="1u"), C("C2", "capacitor", ["d", "o"], value="1u"),
                   C("RL", "resistor", ["o", "0"], value="12k", label="R_L")], ground="0", node_labels={"g": "G", "d": "D", "s": "S", "o": "o"})
    sol, regions = amp.solve_transistors()
    assert regions == {"M1": "saturation"} and abs(float(sol["I_D"]) - 0.8486e-3) < 1e-7
    sts = [x for t in ("I_D = K (V_{GS} - V_t)^2", "V_{DS} = 5.21", "g_m = 2K V_{ov}", "g_m = 2\\sqrt{K I_D}",
                       "A_v = -g_m (R_D \\| R_L)", "I_D = 2.65 mA") for x in circuit_statements(t)]
    assert [(c.verdict, c.sure) for c in check_against(sts, amp)] == [("right", True)] * 5 + [("wrong", True)]


def test_zener_regulator():
    """a Zener regulator: in breakdown at V_Z (its label) while the load is light, off once the load pulls the node
    below V_Z - I_Z, V_Z checked"""
    from grader.circuit import Component as C, Netlist, zener_voltage
    from grader.circuit_check import check_against, circuit_statements
    assert zener_voltage("5V1") == zener_voltage("V_Z = 5.1 V") == __import__("sympy").Rational(51, 10)
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12"), C("R1", "resistor", ["n0", "a"], value="500"),
                   C("DZ", "zener", ["a", "0"], plus="a", value="5.1V"), C("RL", "resistor", ["a", "0"], value="1k", label="R_L")],
                  ground="0", node_labels={"a": "A"})
    assert net.solve_diodes(vd=0.7)[1] == {"DZ": "breakdown"}

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("I_Z = 8.7 mA") == checks("V_A = 5.1") == [("right", True)]
    assert checks("I_Z = 13.8 mA") == [("wrong", True)]
    net.components[3].value = "300"
    assert net.solve_diodes(vd=0.7)[1] == {"DZ": "off"} and checks("V_A = 4.5") == [("right", True)]


def test_phasor_circuit_labelled_with_impedances():
    """a phasor-domain drawing - the source '10∠0°', parts labelled with their impedances ('3+j4Ω', '-j2Ω'; the wiring
    gives such labels to the nearest part / source) - solved without any w; Z_eq / Z_in what the source sees"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit import Component as C, Netlist, complex_value
    from grader.circuit_check import check_against, circuit_statements
    assert complex_value("3 - j4") == 3 - 4 * __import__("sympy").I and complex_value("10Ω") is None
    assert abs(complex(complex_value("5∠53.13°")) - (3 + 4j)) < 1e-3
    m = np.zeros((400, 700), bool)
    m[50:54, 50:650] = True
    m[300:304, 50:650] = True
    m[50:304, 50:54] = True
    m[50:304, 446:450] = True
    comps = [("voltage_dc", (25, 140, 80, 215)), ("inductor", (425, 130, 470, 230))]
    for cls, (x0, y0, x1, y1) in comps:
        m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    got = netlist_from_boxes(m, comps, [((90, 160, 160, 190), "10∠0° V"), ((480, 160, 540, 190), "j4Ω")], polarity=False)
    v = next(c for c in got.components if c.type == "voltage_source")
    assert v.value == "10∠0° V" and v.ac and next(c for c in got.components if c.type == "inductor").value == "j4Ω"
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10∠0°"), C("Z1", "resistor", ["n0", "a"], value="3+j4Ω", label="Z_1"),
                   C("C1", "capacitor", ["a", "0"], value="-j2Ω", label="Z_2")], ground="0", node_labels={"a": "A"})
    assert net.components[0].ac

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("V_A = 5.55\\angle -123.7^\\circ") == checks("I = \\frac{10}{3 + j2}") == [("right", True)]
    assert checks("Z_{eq} = Z_1 + Z_2") == checks("Z_{in} = 3.61\\angle 33.7^\\circ") == [("right", True)]
    assert checks("V_A = 5\\angle -90^\\circ") == checks("Z_{eq} = 3 + j6") == [("wrong", True)]


def test_coupled_coils_and_line_voltages():
    """coupled coils from their labels ('L_1 = 2H', 'L_2 = 8H', 'M = 2H' beside the coupled-coil symbol): v_P = jw (L1 i_P
    + M i_S), v_S = jw (M i_P + L2 i_S); a three-phase Y drawn with its sources '120∠0°', '120∠-120°', '120∠120°':
    line voltages V_{ab} between named nodes"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit import Component as C, Netlist, coupling
    from grader.circuit_check import check_against, circuit_statements
    m = np.zeros((400, 900), bool)
    m[100:104, 50:350] = True
    m[300:304, 50:350] = True
    m[100:304, 50:54] = True
    m[100:104, 450:800] = True
    m[300:304, 450:800] = True
    m[100:304, 796:800] = True
    comps = [("voltage_ac", (25, 170, 80, 230)), ("transformer", (340, 90, 460, 315)), ("resistor", (775, 170, 820, 230))]
    for cls, (x0, y0, x1, y1) in comps:
        if cls != "transformer":
            m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    m[110:295, 360:380] = True
    m[110:295, 420:440] = True
    got = netlist_from_boxes(m, comps, [((280, 180, 330, 210), "2H"), ((470, 180, 520, 210), "8H"), ((375, 40, 430, 70), "M = 2H")],
                             polarity=False)
    t = next(c for c in got.components if c.type == "transformer")
    assert coupling(t) == (2, 8, 2), t.params
    T = C("T1", "transformer", ["p", "0", "s", "0"], state="dots read")
    T.roles, T.params = {"P+": "p", "P-": "0", "S+": "s", "S-": "0"}, {"L1": "2H", "L2": "8H", "M": "2H"}
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10cos(10t)", ac=True),
                   C("R1", "resistor", ["n0", "p"], value="10"), T, C("R2", "resistor", ["s", "0"], value="40")],
                  ground="0", node_labels={"p": "P", "s": "S"})
    r = net.solve("phasor")
    assert abs(complex(r["V_P"]) - (7 + 4j)) < 1e-9 and abs(complex(r["V_S"]) - (4 - 2j)) < 1e-9
    y = Netlist([C("Va", "voltage_source", ["a", "0"], plus="a", value="120∠0°"), C("Vb", "voltage_source", ["b", "0"], plus="b", value="120∠-120°"),
                 C("Vc", "voltage_source", ["c", "0"], plus="c", value="120∠120°"), C("Za", "resistor", ["a", "m"], value="10+j5Ω"),
                 C("Zb", "resistor", ["b", "m"], value="10+j5Ω"), C("Zc", "resistor", ["c", "m"], value="10+j5Ω")],
                ground="0", node_labels={"a": "a", "b": "b", "c": "c", "m": "N"})

    def checks(tex, n=y):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_{ab} = 207.8\\angle 30^\\circ") == checks("I_{Za} = 10.73\\angle -26.57^\\circ") == [("right", True)]
    assert checks("V_{ab} = 120\\angle 30^\\circ") == [("wrong", True)]


def test_values_the_student_writes_for_a_drawing_of_names():
    """the drawing names its parts only; the student copies the problem's data first ('E = 12 V', 'R_1 = 2k\\Omega'):
    the rest of the work is checked with those values - suggestions, not verdicts"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    sym = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", label="E"), C("R1", "resistor", ["n0", "a"], label="R_1"),
                   C("R2", "resistor", ["a", "0"], label="R_2")], ground="0", node_labels={"a": "A"})
    sts = [x for t in ("E = 12 V", "R_1 = 2k\\Omega", "R_2 = 4 k\\Omega", "V_A = 8", "I = 2 mA", "V_A = 6") for x in circuit_statements(t)]
    got = [(c.verdict, c.sure) for c in check_against(sts, sym)]
    assert got == [("right", False), ("right", False), ("wrong", False)], got


def test_corner_frequency_energy_and_no_schmitt_trigger():
    """a first-order filter's corner (omega_c = 1/tau, f_c), the energy a capacitor holds in the steady state, units
    Hz / J / rad/s; an op-amp fed back to In+ only (a Schmitt trigger) is not solved as a linear one"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    lp = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", value="1cos(1000t)", ac=True),
                  C("R1", "resistor", ["i", "o"], value="1k"), C("C1", "capacitor", ["o", "0"], value="1u")], ground="0",
                 node_labels={"o": "o"})

    def checks(tex, n=lp):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("f_c = 159.2 Hz") == checks("\\omega_c = 1000 rad/s") == [("right", True)]
    assert checks("f_c = 200 Hz") == [("wrong", True)]
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                  C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})
    assert checks("W_C = 50 \\mu J", rc) == [("right", False)]
    sch = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", value="1"), C("R1", "resistor", ["0", "p"], value="1k"),
                   C("R2", "resistor", ["p", "y"], value="10k"), C("OA", "opamp", ["p", "i", "y"]),
                   C("RL", "resistor", ["y", "0"], value="1k")], ground="0")
    sch.components[3].roles = {"In+": "p", "In-": "i", "Out": "y"}
    assert sch.solve() == {}


def test_formula_reader_output_checked_as_written():
    """the formula reader's own LaTeX (as it wrote typical circuit answers rendered as print): spaced tokens and digits,
    '\\!' / '\\,' / '~' spaces, '\\mathrm { c o s }', a unit after a phasor - read like the clean form"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, parse_side, tidy
    assert tidy("7 . 0 7 \\mathrm { c o s } ( 1 0 0 0 t )") == "7.07 \\cos ( 1000 t )"
    assert [x.side_texts for x in circuit_statements("I \\! = 2 . 5 ~ m A")] == [["I", "0.0025"]]
    assert [x.side_texts for x in circuit_statements("f _ { c } = 1 5 9 \\, H z")] == [["f_{c}", "159.0"]]
    assert abs(complex(parse_side("3 . 6 1 \\angle 3 3 . 7 ^ { \\circ } \\, \\, \\Omega", ac=True)) - (3.004 + 2.002j)) < 0.01
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"]),
                   C("R1", "resistor", ["n1", "a"], value="2k"), C("R2", "resistor", ["a", "0"], value="2k"),
                   C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})

    def checks(tex, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("v _ { C } ( t ) \\! = 5 ( 1 - e ^ { - 1 0 0 0 t } )") == [("right", True)]
    assert checks("V _ { C } ( s ) = \\frac { 5 0 0 0 } { s ( s + 1 0 0 0 ) }") == [("right", True)]
    g = Netlist([C("U1", "and", ["y", "a", "b"]), C("U2", "or", ["f", "y", "c"])], node_labels={"a": "A", "b": "B", "c": "C", "f": "F"})
    g.components[0].roles = {"out": "y", "in1": "a", "in2": "b"}
    g.components[1].roles = {"out": "f", "in1": "y", "in2": "c"}
    assert checks("F { = } A B { } + C", g)[0][0] == "right" and checks("F = \\overline { { A B } } + C", g)[0][0] == "wrong"
    # as it wrote this night's answer types: a name spaced into letters ('B W', 'p f'), a spaced unit ('V A', 'd B'), a
    # prime as a superscript, '\\mathrm { = }', '\\mid' bars, '\\mathrm { l o g }', a spaced subscript inside bars
    rlc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"], state="open"),
                   C("R1", "resistor", ["n1", "n2"], value="10"), C("L1", "inductor", ["n2", "a"], value="0.1"),
                   C("C1", "capacitor", ["a", "0"], value="1m")], ground="0", node_labels={"a": "A"})
    ac = Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10cos(1000t)", ac=True),
                  C("R1", "resistor", ["a", "b"], value="3"), C("L1", "inductor", ["b", "0"], value="4m")], ground="0")
    lp = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", value="1"), C("R1", "resistor", ["i", "o"], value="1k"),
                  C("C1", "capacitor", ["o", "0"], value="1u")], ground="0", node_labels={"o": "o", "i": "i"})
    for tex, n, want in [("B W \\! = \\! 1 0 0 \\, r a d / s", rlc, "right"),
                         ("i _ { L } ^ { \\, \\prime } ( 0 \\, ^ { + } \\, ) = 1 0 0 \\, A / s", rlc, "right"),
                         ("i _ { L } ^ { ' } ( 0 ^ { + } ) = 1 0 0 \\, A / s", rlc, "right"),
                         ("P \\mathrm { = } 6 \\mathrm { ~ } W", ac, "right"), ("S = 1 0 ~ V A", ac, "right"),
                         ("p f \\! = 0 . 6", ac, "right"), ("| H ( j 1 0 0 0 ) | _ { d B } = - 6", lp, "wrong"),
                         ("2 0 \\mathrm { l o g } \\! \\mid \\! H ( j 1 0 0 0 ) \\! \\mid \\! = \\! - 3 ~ d B", lp, "right"),
                         ("| Z _ { i n } ( j 1 0 0 0 ) | = 1 4 1 4", lp, "right"),
                         ("p f \\mathrm { = } 0 . 6 \\mathrm { ~ ( l e a d i n g ) }", ac, "wrong"),
                         ("p f = 0 . 6 \\, \\operatorname { l a g g i n g }", ac, "right")]:
        assert checks(tex, n) == [(want, True)], tex
    assert [x.side_texts for x in circuit_statements("P \\sb = 1 6 ~ m W")] == [["P", "0.016"]]   # a stray \\sb
    # a decimal comma ('7,5') is no separator of two statements; thousands ('1 , 000') stay one number - joined
    assert [x.side_texts for x in circuit_statements("V _ { A } = 7 , 5 ~ V , I _ { 1 } = 2 , 5 m A")] == [["V_{A}", "7.5"], ["I_{1}", "0.0025"]]
    assert [x.side_texts for x in circuit_statements("R = 1 , 0 0 0")] == [["R", "1000"]]


def test_switch_drawn_state_and_two_way_switch_read():
    """the wiring stage reads a switch as drawn - open (its lever short of the other contact) or closed - and a
    two-way switch's three ends (common, throw 1, throw 2), drawn at the throw its lever reaches"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes

    def loop(closed):
        m = np.zeros((320, 600), bool)
        m[150:154, 50:250] = True
        m[150:154, 350:550] = True
        for k in range(100 if closed else 90):
            x, y = 250 + k, (152 if closed else 152 - k // 2)
            m[y - 2:y + 2, x - 2:x + 2] = True
        m[150:300, 50:54] = True
        m[150:300, 546:550] = True
        m[296:300, 50:270] = True
        m[296:300, 330:550] = True
        m[288:308, 270:330] = True
        return m
    comps = [("switch", (240, 100, 360, 165)), ("resistor", (265, 280, 335, 316))]
    for closed in (False, True):
        sw = next(c for c in netlist_from_boxes(loop(closed), comps, polarity=False).components if c.type == "switch")
        assert sw.state == ("closed" if closed else "open") and len(set(sw.nodes)) == 2, sw
    m = np.zeros((400, 600), bool)
    m[200:204, 50:250] = True                        # common end in from the left
    m[120:124, 350:550] = True                       # throw 1 (upper)
    m[280:284, 350:550] = True                       # throw 2 (lower)
    for k in range(100):                             # the lever: common to throw 1
        x, y = 250 + k, 202 - int(0.8 * k)
        m[y - 2:y + 2, x - 2:x + 2] = True
    sw = netlist_from_boxes(m, [("switch", (240, 110, 360, 295))], polarity=False).components[0]
    assert len(set(sw.nodes)) == 3 and sw.state == "1", sw


def test_thevenin_at_terminals_and_opamp_gain_checked():
    """between the open terminals a, b: V_th, R_th, I_N; an inverting amplifier: V_o and A_v - names the drawing does
    not carry, as exam answers use them"""
    from grader.circuit import Component, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([Component("V1", "voltage_source", ["n0", "0"], plus="n0"),
                   Component("R1", "resistor", ["n0", "na"]), Component("R2", "resistor", ["na", "0"])],
                  ground="0", node_labels={"na": "a", "0": "b"}, ports=["na", "0"])

    def verdicts(tex, n=net):
        return [c.verdict for c in check_against(circuit_statements(tex), n)]
    assert verdicts("V_{th} = \\frac{R_2 V_1}{R_1 + R_2}") == ["right"]
    assert verdicts("R_{th} = \\frac{R_1 R_2}{R_1 + R_2}") == ["right"] and verdicts("R_{th} = R_1 + R_2") == ["wrong"]
    assert verdicts("I_N = \\frac{V_1}{R_1}") == ["right"]
    amp = Netlist([Component("V1", "voltage_source", ["a", "0"], plus="a"), Component("R1", "resistor", ["a", "m"]),
                   Component("Rf", "resistor", ["m", "o"]),
                   Component("OA1", "opamp", ["0", "m", "o"], roles={"In+": "0", "In-": "m", "Out": "o"})], ground="0")
    assert verdicts("V_o = -\\frac{R_f}{R_1} V_i", amp) == ["right"]
    assert verdicts("A_v = -\\frac{R_f}{R_1}", amp) == ["right"] and verdicts("A_v = 1 + \\frac{R_f}{R_1}", amp) == ["wrong"]


def test_logic_gates_read_and_boolean_answers_checked():
    """a NAND (A, B in on the left, its output on the right) into an OR with C, and a NOT drawn as a triangle: the
    outputs found by where the wires come in; F's function from the gates; Boolean answers checked (De Morgan's
    forms are the same answer)"""
    import numpy as np
    from grader.circuit import Component, Netlist
    from grader.circuit_check import check_against, circuit_statements, parse_bool
    from recog.circuit_wiring import _gate_output
    m = np.zeros((200, 300), bool)
    assert _gate_output(m, (100, 50, 160, 110), [(100, 65), (100, 95), (160, 80)]) == 2
    tri = np.zeros((200, 300), bool)                 # a NOT: tall at its back (left), a tip on the right
    for x in range(100, 160):
        half = int(30 * (160 - x) / 60)
        tri[80 - half:80 + half + 1, x] = (x < 102) | (abs(np.arange(80 - half, 80 + half + 1) - 80) >= half - 1)
    assert _gate_output(tri, (100, 50, 160, 110), [(160, 80), (100, 80)]) == 0
    assert str(parse_bool("A \\oplus B")) == "A ^ B" and str(parse_bool("(A + B)'")) == "~(A | B)"
    for tex in ("\\neg A \\cdot B + C", "\\lnot A \\land B \\lor C", "\\sim A B + C", "!A B + C", "\\neg a b + c"):
        assert str(parse_bool(tex)).upper() == "C | (B & ~A)", tex          # other ways of writing 'not', 'and', 'or'
    assert str(parse_bool("\\neg (A + B)")) == "~(A | B)" and str(parse_bool("A * B")) == "A & B"
    net = Netlist([Component("U1", "nand", ["x", "a", "b"], roles={"out": "x", "in1": "a", "in2": "b"}),
                   Component("U2", "or", ["f", "x", "c"], roles={"out": "f", "in1": "x", "in2": "c"})],
                  node_labels={"a": "A", "b": "B", "c": "C", "f": "F"})

    def verdicts(tex):
        return [c.verdict for c in check_against(circuit_statements(tex), net)]
    assert verdicts("F = \\overline{A B} + C") == ["right"] and verdicts("F = A' + B' + C") == ["right"]
    assert verdicts("F = A B + C") == ["wrong"]


def test_sinusoidal_circuit_phasors_checked():
    """10cos(1000t) into R = 1k and C = 1u: V_A = 10 / (1 + j) = 5 - j5 = 7.07∠-45°; answers in rectangular, polar
    and time form are checked, a wrong angle is not"""
    from grader.circuit import Component, Netlist, ac_phasor
    from grader.circuit_check import check_against, circuit_statements
    assert ac_phasor("5sin(100t)")[1] == 100 and abs(complex(ac_phasor("5sin(100t)")[0]) - (-5j)) < 1e-9
    net = Netlist([Component("V1", "voltage_source", ["n1", "0"], plus="n1", value="10cos(1000t)", ac=True),
                   Component("R1", "resistor", ["n1", "a"], value="1k"), Component("C1", "capacitor", ["a", "0"], value="1u")],
                  ground="0", node_labels={"a": "A"})
    assert abs(complex(net.solve("phasor")["V_A"]) - (5 - 5j)) < 1e-9 and net.solve("dc") == {}

    def verdicts(tex):
        return [c.verdict for c in check_against(circuit_statements(tex), net)]
    assert verdicts("V_A = 5 - j5") == ["right"] and verdicts("V_A = 7.07 \\angle -45^{\\circ}") == ["right"]
    assert verdicts("V_A = 7.07 \\angle 45^{\\circ}") == ["wrong"]
    assert verdicts("v_A(t) = 7.07 \\cos(1000t - 45^{\\circ})") == ["right"]
    assert verdicts("v_A(t) = 7.07 \\cos(1000t + 45^{\\circ})") == ["wrong"]


def test_current_arrow_and_node_letter():
    """a source, R1 on the top wire and R2 on the right one; an arrow drawn above R1 (close to it, along it)
    pointing right, 'I1' at its tail, 'A' written by the top-right corner: I1 runs through R1 from the source's
    node to R2's, and that node is A - the arrow is not taken for one of R1's wires"""
    import numpy as np
    from recog.circuit_wiring import netlist_from_boxes

    def line(m, a, b, r=1):
        for t in np.linspace(0, 1, 200):
            x, y = int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1])))
            m[y - r:y + r + 2, x - r:x + r + 2] = True
    m = np.zeros((400, 500), bool)
    m[50:54, 50:450] = True                          # top wire
    m[300:304, 50:450] = True                        # bottom wire
    m[50:304, 50:54] = True                          # left (the source)
    m[50:304, 446:450] = True                        # right (R2)
    m[302:340, 300:304] = True                       # the ground's stub
    comps = [("voltage_dc", (30, 140, 75, 210)), ("resistor", (200, 35, 300, 70)), ("resistor", (425, 130, 470, 230)),
             ("gnd", (285, 335, 320, 360))]
    for cls, (x0, y0, x1, y1) in comps[:3]:
        m[y0:y1, x0:x1] = False
    m[140:210, 50:54] = True; m[160:190, 40:65] = True               # the source's ink, leads included
    for k in range(5):                                                # R1's zigzag, leads included
        line(m, (200 + 20 * k, 52), (210 + 20 * k, 38 if k % 2 == 0 else 66))
        line(m, (210 + 20 * k, 38 if k % 2 == 0 else 66), (220 + 20 * k, 52))
    m[130:230, 446:450] = True; m[150:210, 436:460] = True            # R2's ink
    line(m, (205, 18), (292, 18))                                     # the arrow: a line...
    line(m, (292, 18), (279, 8)); line(m, (292, 18), (279, 28))       # ...with its head on the right
    texts = [((165, 2, 200, 30), "I1"), ((456, 18, 478, 40), "A")]
    net = netlist_from_boxes(m, comps, texts)
    v = next(c for c in net.components if c.type == "voltage_source")
    r1, r2 = [c for c in net.components if c.type == "resistor"]
    top_left = next(n for n in v.nodes if n != "0")
    top_right = next(n for n in r2.nodes if n != "0")
    assert sorted(r1.nodes) == sorted([top_left, top_right]), net.components
    assert [(c.label, c.through, c.frm, c.to) for c in net.currents] == [("I1", r1.id, top_left, top_right)], net.currents
    assert net.node_labels == {top_right: "A"}, net.node_labels


def test_strokes_drop_tilted_rules_keep_wires():
    """notebook rules photographed at a slant go, in any pen colour - the drawing stays whole: a wire crossing a rule,
    a wire drawn right along a rule, and a bus of long parallel wires reaching the paper's side"""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    from recog.circuit_wiring import strokes

    def page(pen, rule, slope, along=False, bus=False, W=1200, H=900):
        img = Image.new("RGB", (W, H), (244, 239, 228))
        d = ImageDraw.Draw(img)
        for y0 in range(-60, H + 60, 55):
            d.line((0, y0, W, y0 + slope * W), fill=rule, width=2)
        y = lambda x: 545 + slope * x + (0 if along else 9 - 18 * (x - 300) / 600)   # noqa: E731  on / across a rule
        pts = [(300, 330), (900, 330), (900, y(900)), (300, y(300)), (300, 330)]
        d.line(pts, fill=pen, width=5)
        for k in range(6 if bus else 0):
            d.line((100, 640 + 40 * k, W - 5, 640 + 40 * k), fill=pen, width=5)
        a = np.asarray(img, dtype=np.float32) + np.random.default_rng(0).normal(0, 3, (H, W, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), pts

    for pen, rule, slope, along, bus in [((60, 60, 66), (165, 190, 228), 0.03, False, False),    # grey pen, blue rules
                                         ((200, 70, 60), (165, 190, 228), 0.03, False, False),   # red pen
                                         ((60, 60, 66), (165, 190, 228), 0.03, False, True),     # a bus below
                                         ((60, 60, 66), (150, 150, 150), 0.02, True, False)]:    # grey, on a rule
        img, pts = page(pen, rule, slope, along, bus)
        m = strokes(img)
        lab, _ = ndi.label(m, structure=np.ones((3, 3)))
        on = {int(lab[int(y0 + t * (y1 - y0)) - 3:int(y0 + t * (y1 - y0)) + 4,
                      int(x0 + t * (x1 - x0)) - 3:int(x0 + t * (x1 - x0)) + 4].max())
              for (x0, y0), (x1, y1) in zip(pts, pts[1:]) for t in (0.2, 0.5, 0.8)}
        assert len(on) == 1 and 0 not in on, (pen, rule, on)              # the loop: whole, one piece
        assert m[:250].sum() == 0 and m[:, 1000:1150][:600].sum() == 0, (pen, rule)   # no rule left
        for k in range(6 if bus else 0):
            assert m[640 + 40 * k - 3:640 + 40 * k + 4, 150:1150].any(0).mean() > 0.95, k


def test_tal_drawing_wired_from_hand_set_boxes():
    """Tal's scanned drawing (a pale blue pen on blue rules, the bottom wire and two current arrows drawn along rules)
    with its parts' boxes set by hand: the classical strokes and the wiring give its netlist entirely right - the
    nodes, both sources' polarity, node A, each current through its resistor in its direction, Kirchhoff at A"""
    pytest.importorskip("pypdfium2")
    from grader.circuit import compare, truth_netlist
    from read_page import load_pages
    from recog.circuit_wiring import netlist_from_boxes, strokes
    from recog.strokes import scale_to
    here = os.path.join(os.path.dirname(__file__), "..", "data", "my_scans")
    if not os.path.isfile(os.path.join(here, "tal_circuit.pdf")):
        pytest.skip("Tal's scan not here")
    small = scale_to(load_pages(os.path.join(here, "tal_circuit.pdf"), 150)[0], 1600)
    s = small.height / 1210                                       # the boxes were set at 110 dpi
    comps = [("resistor", (250, 485, 380, 600)), ("resistor", (530, 495, 615, 585)), ("resistor", (412, 620, 475, 750)),
             ("voltage_dc", (78, 670, 192, 800)), ("voltage_dc", (660, 665, 770, 800)), ("gnd", (410, 930, 470, 1010))]
    texts = [((432, 490, 470, 522), "A"), ((175, 390, 235, 460), "I1"), ((650, 380, 720, 455), "I3"),
             ((330, 655, 385, 730), "I2"), ((505, 670, 585, 720), "R2"), ((25, 715, 75, 760), "E1"), ((780, 710, 846, 755), "E2")]
    net = netlist_from_boxes(strokes(small), [(c, tuple(v * s for v in b)) for c, b in comps],
                             [(tuple(v * s for v in b), t) for b, t in texts])
    c = compare(net, truth_netlist(os.path.join(here, "tal_circuit_truth.json")))
    assert c["topology_exact"] and c["polarity"] == "2/2" and c["node_letters"] == "1/1", c
    assert c["currents_on_the_right_part"] == "3/3" and c["current_directions"] == "3/3" and c["kcl"] == "1/1", c


def test_second_order_parameters_checked():
    """alpha, w_0 / w_n, zeta, w_d and the bandwidth of a second-order circuit - from its natural response's
    s^2 + 2 alpha s + w_n^2, whatever the topology - judged like the time constant: they belong to no moment"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    series = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"]),
                      C("R1", "resistor", ["n1", "n2"], value="10"), C("L1", "inductor", ["n2", "a"], value="0.1"),
                      C("C1", "capacitor", ["a", "0"], value="1m")], ground="0", node_labels={"a": "A"})
    assert series.natural_polynomial({"S1": "closed"}) == [1, 100, 10000]
    parallel = Netlist([C("I1", "current_source", ["0", "a"], plus="a", value="2"), C("R1", "resistor", ["a", "0"], value="100"),
                        C("L1", "inductor", ["a", "0"], value="0.1"), C("C1", "capacitor", ["a", "0"], value="10u")], ground="0")

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    for tex, verdict in [("\\alpha = 50", "right"), ("\\alpha = \\frac{R_1}{2L_1}", "right"), ("\\alpha = 100", "wrong"),
                         ("\\omega_0 = 100 \\frac{rad}{s}", "right"), ("\\zeta = 0.5", "right"), ("\\zeta = 0.7", "wrong"),
                         ("\\omega_d = 86.6", "right"), ("\\omega_d = 100", "wrong"), ("BW = 100 rad/s", "right")]:
        assert checks(tex, series) == [(verdict, True)], tex
    for tex, verdict in [("\\alpha = 500", "right"), ("\\alpha = \\frac{1}{2R_1C_1}", "right"), ("\\omega_n = 1000", "right"),
                         ("\\zeta = 0.5", "right"), ("\\zeta = 2", "wrong")]:
        assert checks(tex, parallel) == [(verdict, True)], tex
    assert checks("V_A = 3", series) == []                  # a moment's value: still not judged wrong


def test_values_just_before_just_after_and_long_after_a_switching():
    """x(0^-), x(0^+), x(\\infty) and dx/dt at 0^+ against the switchings the drawing allows: what C and L held carries
    over (the initial value theorem on X(s)), the steady states before and after - in any order of circuit"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"], state="open"),
                  C("R1", "resistor", ["n1", "a"], value="2k"), C("R2", "resistor", ["a", "0"], value="2k"),
                  C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})
    rl = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12"), C("S1", "switch", ["n0", "n1"], state="closed"),
                  C("R1", "resistor", ["n1", "a"], value="4"), C("L1", "inductor", ["a", "0"], value="2"),
                  C("R2", "resistor", ["a", "0"], value="12")], ground="0", node_labels={"a": "A"})
    rlc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"], state="open"),
                   C("R1", "resistor", ["n1", "n2"], value="10"), C("L1", "inductor", ["n2", "a"], value="0.1"),
                   C("C1", "capacitor", ["a", "0"], value="1m")], ground="0", node_labels={"a": "A"})

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    for tex, n, verdict in [("v_C(0^-) = 0", rc, "right"), ("v_C(0^+) = 5", rc, "wrong"), ("i_C(0^+) = 5 mA", rc, "right"),
                            ("v_C(\\infty) = 5 V", rc, "right"), ("v_C(\\infty) = 10", rc, "wrong"),
                            ("i_L(0^+) = 3", rl, "right"), ("V_A(0^+) = -36", rl, "right"), ("V_A(0^+) = 36", rl, "wrong"),
                            ("i_L(\\infty) = 0", rl, "right"), ("v_L(0^+) = 10", rlc, "right"),
                            ("\\frac{di_L}{dt}(0^+) = 100", rlc, "right"), ("\\left.\\frac{di_L}{dt}\\right|_{t=0^+} = 10", rlc, "wrong"),
                            ("i_L'(0^+) = 100 A/s", rlc, "right"), ("\\frac{dv_C}{dt}(0^+) = 0", rlc, "right")]:
        assert checks(tex, n) == [(verdict, True)], tex
    assert checks("i_C(0) = 5 mA", rc) == [("right", False)]           # 0 alone: 0^- or 0^+ - they differ here


def test_ac_power_totals_power_factor_and_rms():
    """what the one source gives: P, Q, S = P + jQ, |S| ('S = 10 VA'), the power factor (pf, cos phi), the load's
    angle phi, I_rms / V_rms - amplitudes for a source written in time, either convention for one given as a phasor"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements

    def net(v):
        return Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value=v, ac=True),
                        C("R1", "resistor", ["a", "b"], value="3"), C("L1", "inductor", ["b", "0"], value="4m")], ground="0")
    timed = net("10cos(1000t)")

    def checks(tex, n=timed):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    for tex, verdict in [("P = 6 W", "right"), ("P = 12 W", "wrong"), ("Q = 8 VAR", "right"), ("S = 6 + j8", "right"),
                         ("S = 6 - j8", "wrong"), ("S = 10 VA", "right"), ("|S| = 20", "wrong"), ("pf = 0.6", "right"),
                         ("\\cos\\phi = 0.8", "wrong"), ("\\phi = 53.13^\\circ", "right"), ("\\phi = 36.87^\\circ", "wrong"),
                         ("I_{rms} = 1.414", "right"), ("I_{rms} = 2", "wrong"), ("V_{rms} = 7.07", "right")]:
        assert checks(tex) == [(verdict, True)], tex
    phasor = Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10∠0°"), C("R1", "resistor", ["a", "b"], value="3"),
                      C("L1", "inductor", ["b", "0"], value="j4Ω")], ground="0")      # amplitude or RMS value: the course's
    assert checks("P = 6 W", phasor) == [("right", False)] and checks("P = 12 W", phasor) == [("right", False)]
    assert checks("P = 9 W", phasor) == [("wrong", True)] and checks("pf = 0.6", phasor) == [("right", True)]
    y3 = Netlist([C("Va", "voltage_source", ["a", "n"], plus="a", value="10cos(100t)", ac=True),     # three phases
                  C("Vb", "voltage_source", ["b", "n"], plus="b", value="10cos(100t - 120°)", ac=True),
                  C("Vc", "voltage_source", ["c", "n"], plus="c", value="10cos(100t + 120°)", ac=True),
                  C("R1", "resistor", ["a", "m"], value="5"), C("R2", "resistor", ["b", "m"], value="5"),
                  C("R3", "resistor", ["c", "m"], value="5")], ground="n")
    assert checks("P = 30 W", y3) == [("right", True)] and checks("P = 20 W", y3) == [("wrong", True)]
    assert checks("P = 10 W", y3) == [("right", False)]                 # one phase's: the question may ask for it


def test_frequency_response_answers_checked():
    """H(j\\omega) as a function of w, its value at one w (j1000), at 0 and at infinity, its magnitude (also in dB) and
    angle - against the circuit's H(s) at s = jw; Z_in(jw) the same way"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    lp = Netlist([C("V1", "voltage_source", ["i", "0"], plus="i", value="1"), C("R1", "resistor", ["i", "o"], value="1k"),
                  C("C1", "capacitor", ["o", "0"], value="1u")], ground="0", node_labels={"o": "o", "i": "i"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), lp)]
    for tex, verdict in [("H(j\\omega) = \\frac{1}{1 + j0.001\\omega}", "right"), ("H(j\\omega) = \\frac{1}{1 + j\\omega}", "wrong"),
                         ("|H(j\\omega)| = \\frac{1}{\\sqrt{1 + (0.001\\omega)^2}}", "right"), ("|H(j1000)| = 0.707", "right"),
                         ("|H(j1000)| = 0.5", "wrong"), ("H(j10^3) = 0.5 - 0.5j", "right"), ("\\angle H(j1000) = -45^\\circ", "right"),
                         ("\\angle H(j1000) = 45^\\circ", "wrong"), ("20\\log|H(j1000)| = -3 dB", "right"),
                         ("|H(j1000)|_{dB} = -6", "wrong"), ("H(0) = 1", "right"), ("H(\\infty) = 0", "right"),
                         ("|Z_{in}(j1000)| = 1414", "right")]:
        assert checks(tex) == [(verdict, True)], tex
    assert check_against(circuit_statements("|H(j1000)| = 0.707"), lp)[0].text.startswith("'|H(j1000)| = 0.707'")


def test_problem_values_given_by_the_lecturer():
    """a drawing that only names its parts (R_1, R_2, E): numbers are no verdict against it, or with the student's own
    copied data only suggestions - with the problem's data given once for the question they are judged, sure"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", label="E"), C("R1", "resistor", ["n0", "a"], label="R_1"),
                   C("R2", "resistor", ["a", "0"], label="R_2")], ground="0", node_labels={"a": "A"})

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("V_A = 8", net) in ([], [("cannot check", False)])
    given = with_values(net, {"E": "10", "R_1": "1k", "R_{2}": "4k"})
    assert checks("V_A = 8", given) == [("right", True)] and checks("V_A = 2", given) == [("wrong", True)]
    assert checks("V_A = \\frac{E R_2}{R_1 + R_2}", given) == [("right", True)]
    ac = with_values(net, {"E": "10cos(1000t)", "R_1": "1k", "R_2": "4k"})
    assert ac.components[0].ac and checks("V_A = 8", ac) == [("right", True)]
    assert with_values(net, {}) is net


def test_strokes_on_squared_paper():
    """squared paper's columns go like its rows - a grey or a blue grid under a grey or blue pen: the drawing stays
    one piece and next to nothing else is left"""
    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    from recog.circuit_wiring import strokes
    for pen, grid in [((60, 60, 66), (185, 185, 185)), ((40, 60, 150), (170, 195, 225)), ((60, 60, 66), (170, 195, 225))]:
        W, H = 1200, 900
        img = Image.new("RGB", (W, H), (245, 243, 238))
        d = ImageDraw.Draw(img)
        for k in range(-40, max(W, H) + 40, 40):
            d.line((0, k, W, k + 12), fill=grid, width=2)
            d.line((k, 0, k - 9, H), fill=grid, width=2)
        d.line([(300, 330), (900, 330), (900, 600), (300, 600), (300, 330)], fill=pen, width=5)
        d.line((600, 330, 600, 600), fill=pen, width=5)
        a = np.asarray(img, dtype=np.float32) + np.random.default_rng(0).normal(0, 3, (H, W, 3))
        m = strokes(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
        lab, _ = ndi.label(m, structure=np.ones((3, 3)))
        loop = lab[330, 450]
        outside = m.copy()
        outside[310:620, 280:920] = False
        assert loop and lab[600, 750] == loop and lab[450, 600] == loop, (pen, grid)
        assert (lab == loop)[:300].sum() == 0 and outside.sum() < 1000, (pen, grid, int(outside.sum()))


def test_equivalent_resistance_the_source_sees():
    """R_eq / R_in / R_T: the resistance the one source sees; with terminals drawn as well R_eq may mean theirs -
    either is right then, not sure, and a value that is neither is wrong"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements

    def net(**kw):
        return Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                        C("R2", "resistor", ["a", "0"], value="4k")], ground="0", **kw)
    plain, port = net(node_labels={"a": "A"}), net(node_labels={"a": "a", "0": "b"}, ports=["a", "0"])

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("R_{eq} = 5k", plain) == [("right", True)] and checks("R_T = 5 k\\Omega", plain) == [("right", True)]
    assert checks("R_{eq} = 3k", plain) == [("wrong", True)]
    assert checks("R_{eq} = 5k", port) == [("right", False)] and checks("R_{eq} = 800", port) == [("right", False)]
    assert checks("R_{eq} = 3k", port) == [("wrong", True)] and checks("R_{th} = 800", port) == [("right", True)]
    assert check_against(circuit_statements("V_A = 8"), port)[0].text == "'V_A = 8': as the circuit gives"


def test_resonance_unsure_where_definitions_differ():
    """w_0 = 1/sqrt(LC) is judged sure only where it is the resonance by every definition (a plain series or parallel
    RLC); a coil's resistance beside the capacitor moves the unity-power-factor frequency (866 here): not sure then"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    series = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10cos(1000t)", ac=True),
                      C("R1", "resistor", ["n0", "a"], value="10"), C("L1", "inductor", ["a", "b"], value="10m"),
                      C("C1", "capacitor", ["b", "0"], value="100u")], ground="0")
    coil = Netlist([C("I1", "current_source", ["0", "a"], plus="a", value="1cos(1000t)", ac=True), C("R1", "resistor", ["a", "b"], value="5"),
                    C("L1", "inductor", ["b", "0"], value="10m"), C("C1", "capacitor", ["a", "0"], value="100u")], ground="0")

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("\\omega_0 = 900", series) == [("wrong", True)] and checks("\\omega_0 = 1000", coil) == [("right", True)]
    assert checks("\\omega_0 = 866", coil) == [("wrong", False)]


def test_reader_retries_darker_strokes_on_a_grainy_photo():
    """a photo on a grainy table: the strokes take in the grain round the drawing and the reading has problems (too
    dense, part-like ink); the classical strokes again with a darker threshold give the circuit right, without them"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.circuit_reader import retry_darker
    from recog.circuit_wiring import netlist_from_boxes, strokes
    rng = np.random.default_rng(0)
    base = np.full((700, 1000), 235.0) - 70 * (rng.random((700, 1000)) < 0.25)      # grey specks
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(img)
    pen = (40, 40, 40)
    d.line([(200, 300), (200, 150), (400, 150)], fill=pen, width=5, joint="curve")
    d.line([(400 + 25 * k, 150 + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)], fill=pen, width=5)
    d.line([(600, 150), (800, 150), (800, 330)], fill=pen, width=5, joint="curve")
    d.line([(760, 330), (840, 330)], fill=pen, width=5)
    d.line([(760, 370), (840, 370)], fill=pen, width=5)
    d.line([(800, 370), (800, 550), (200, 550), (200, 400)], fill=pen, width=5, joint="curve")
    d.ellipse((150, 300, 250, 400), outline=pen, width=5)
    comps = [("voltage_dc", (140, 290, 260, 410)), ("resistor", (390, 110, 610, 190)), ("capacitor", (750, 320, 850, 380))]
    wire = lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True, polarity=False)     # noqa: E731
    mask = strokes(img)
    net, dbg = wire(mask, comps, [])
    assert len(net.problems()) > 3
    net, _, _ = retry_darker(img, net, dbg, mask, comps, [], wire, strokes)
    assert len(net.notes) == 1 and "darker strokes" in net.notes[0]          # still a doubt for the verdicts
    assert sorted(len(set(c.nodes)) for c in net.components) == [2, 2, 2] and len({n for c in net.components for n in c.nodes}) == 3


def test_q_of_an_rlc_quality_factor_or_reactive_power():
    """'Q' in an AC circuit with an L and a C: the quality factor (w_n / bandwidth) or the reactive power - either is
    right, not sure; neither is wrong; with its unit ('VAR') it is the reactive power"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    series = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10cos(1000t)", ac=True),
                      C("R1", "resistor", ["n0", "a"], value="10"), C("L1", "inductor", ["a", "b"], value="10m"),
                      C("C1", "capacitor", ["b", "0"], value="100u")], ground="0")

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), series)]
    assert checks("Q = 1") == [("right", False)] and checks("Q = 0") == [("right", False)]     # w_0 L / R; at resonance
    assert checks("Q = 0 VAR") == [("right", True)] and checks("Q = 1 VAR") == [("wrong", True)]
    assert checks("Q = 5") == [("wrong", True)] and checks("Q_L = 5") == [("right", True)]
    # in DC, Q of the one capacitor is the charge it holds in the steady state, C v ('80 \\mu C')
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                  C("R2", "resistor", ["a", "0"], value="4k"), C("C1", "capacitor", ["a", "0"], value="10u")], ground="0")
    for tex in ("Q = 80 \\mu C", "Q_C = 80 \\mu C", "Q = 8 \\cdot 10^{-5}"):
        assert [c.verdict for c in check_against(circuit_statements(tex), rc)] == ["right"], tex
    assert check_against(circuit_statements("Q = 100 \\mu C"), rc) == []       # another moment, maybe: not judged


def test_power_factor_lagging_or_leading():
    """'pf = 0.6 lagging': the power factor and the load's character together - lagging for an inductive load,
    leading for a capacitive one; the right number with the wrong word is wrong"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements

    def net(part, value):
        return Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10cos(1000t)", ac=True),
                        C("R1", "resistor", ["a", "b"], value="3"), C("X1", part, ["b", "0"], value=value)], ground="0")
    ind, cap = net("inductor", "4m"), net("capacitor", "250u")

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("pf = 0.6 lagging", ind) == [("right", True)] and checks("pf = 0.6 (leading)", ind) == [("wrong", True)]
    assert checks("pf = 0.6 \\text{ leading}", cap) == [("right", True)] and checks("pf = 0.6 lagging", cap) == [("wrong", True)]
    assert checks("\\cos\\phi = 0.8 lagging", ind) == [("wrong", True)] and checks("pf = 0.6", cap) == [("right", True)]


def test_reactance_part_impedance_and_the_sources_frequency():
    """X_L / X_C (either sign, as courses define X_C), Z_L = jwL / Z_C = 1/(jwC) by the part's name, the sources'
    w and f - against the circuit's phasors"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements

    def net(part, value):
        return Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10cos(1000t)", ac=True),
                        C("R1", "resistor", ["a", "b"], value="3"), C("X1", part, ["b", "0"], value=value)], ground="0")
    ind, cap = net("inductor", "4m"), net("capacitor", "250u")

    def checks(tex, n):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), n)]
    assert checks("X_L = 4 \\Omega", ind) == [("right", True)] and checks("X_L = 5", ind) == [("wrong", True)]
    assert checks("Z_L = j4", ind) == [("right", True)] and checks("Z_L = -j4", ind) == [("wrong", True)]
    assert checks("X_C = 4", cap) == checks("X_C = -4", cap) == [("right", True)] and checks("Z_C = j4", cap) == [("wrong", True)]
    assert checks("\\omega = 1000", ind) == [("right", True)] and checks("f = 159.2 Hz", ind) == [("right", True)]


def test_powers_of_ten_with_units_and_one_parts_name():
    """'10^{-3} s', '2 \\cdot 10^{-3} s', '5 \\times 10^{3} \\Omega' read in base units; 'C' alone for the one capacitor
    ('tau = (R_1 \\parallel R_2) C')"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import _value_side, check_against, circuit_statements
    assert float(_value_side("2 \\cdot 10^{-3} s")) == 0.002 and float(_value_side("5 \\times 10^{3} \\Omega")) == 5000.0
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="2k"),
                  C("R2", "resistor", ["a", "0"], value="2k"), C("C1", "capacitor", ["a", "0"], value="1u")], ground="0")
    for tex in ("\\tau = 10^{-3} s", "\\tau = (R_1 \\parallel R_2) C"):
        assert [c.verdict for c in check_against(circuit_statements(tex), rc)] == ["right"], tex


def test_superposition_shares_with_primes():
    """V_A', I_2'' or V_A^{(1)} (not V_A to the power 1): what one source gives alone - right when it is one source's
    share (not sure: which one is first is the student's numbering); one that is no source's share is wrong, unsure"""
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values
    net = Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                   C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                   C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")],
                  [Current("I1", "R1", "n1", "a"), Current("I2", "R2", "a", "0"), Current("I3", "R3", "n3", "a")],
                  ground="0", node_labels={"a": "A"})
    given = with_values(net, {"E1": "12", "E2": "6", "R1": "2", "R2": "4", "R3": "4"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), given)]
    assert checks("V_A' = 6") == checks("V_A^{(2)} = 1.5") == checks("I_2'' = 0.375") == [("right", False)]
    assert checks("V_A' = 5") == [("wrong", False)] and checks("V_A = 7.5") == [("right", True)]


def test_resistance_a_capacitor_sees():
    """one capacitor and no terminals drawn: R_th is the resistance it sees (its time constant's R); R_eq may be that
    or what the source sees - either right, not sure; tau listed once though variants are tried beside it"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="2k"),
                  C("R2", "resistor", ["a", "0"], value="2k"), C("C1", "capacitor", ["a", "0"], value="1u")], ground="0")

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), rc)]
    assert checks("R_{th} = 1 k\\Omega") == checks("R_{eq} = 1k") == checks("R_{eq} = 4k") == [("right", False)]
    assert checks("\\tau = 1 ms") == [("right", True)] and checks("\\tau = 2 ms") == [("wrong", True)]


def test_a_nodes_own_name_wins_over_a_conventional_one():
    """a node labelled T: V_T is its voltage, not Thevenin's V_T between A and B (the conventional names fill in only
    what the drawing does not name)"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["t", "0"], plus="t", value="10"), C("R1", "resistor", ["t", "a"], value="1k"),
                   C("R2", "resistor", ["t", "b"], value="2k"), C("R3", "resistor", ["a", "0"], value="3k"),
                   C("R4", "resistor", ["b", "0"], value="1k"), C("R5", "resistor", ["a", "b"], value="5k")], ground="0",
                  node_labels={"a": "A", "b": "B", "t": "T"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("V_T = 10") == checks("V_{th} = 3.247") == [("right", True)]


def test_time_functions_with_seconds_inside():
    """'v_C(t) = 5 - 5e^{-t/1 ms}', 'i_C(t) = 5 mA \\cdot e^{-t/1ms}': a time inside the exponent in seconds"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    rc = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("S1", "switch", ["n0", "n1"], state="open"),
                  C("R1", "resistor", ["n1", "a"], value="2k"), C("R2", "resistor", ["a", "0"], value="2k"),
                  C("C1", "capacitor", ["a", "0"], value="1u")], ground="0", node_labels={"a": "A"})
    for tex in ("v_C(t) = 5 - 5 e^{-t / 1 ms}", "i_C(t) = 5 mA \\cdot e^{-t/1ms}"):
        assert [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), rc)] == [("right", True)], tex
    assert [c.verdict for c in check_against(circuit_statements("v_C(t) = 5 - 5 e^{-t / 2 ms}"), rc)] == ["wrong"]


def test_polar_angle_without_degrees_may_be_radians():
    """'2\\angle -0.9273' (no degree mark): read in degrees first, then in radians - right then, not sure"""
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["a", "0"], plus="a", value="10cos(1000t)", ac=True),
                   C("R1", "resistor", ["a", "b"], value="3"), C("L1", "inductor", ["b", "0"], value="4m")],
                  [Current("I1", "R1", "a", "b")], ground="0")

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("I_1 = 2\\angle -53.13") == [("right", True)] and checks("I_1 = 2\\angle -0.9273") == [("right", False)]
    assert checks("I_1 = 2\\angle 0.9273") == [("wrong", True)]


def test_mesh_currents_named_j():
    """mesh equations in J_1, J_2 (as some courses name mesh currents) solved together like I_a, I_b: right when they
    come out as the circuit's own currents"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values
    net = with_values(Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                               C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                               C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")], ground="0"),
                      {"E1": "12", "E2": "6", "R1": "2", "R2": "4", "R3": "4"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("12 = 6 J_1 - 4 J_2 \\quad -6 = 8 J_2 - 4 J_1") == [("right", True)]
    assert checks("6 J_1 - 4 J_2 = 13 \\quad -4 J_1 + 8 J_2 = -6") == [("wrong", False)]


def test_bare_p_may_be_one_parts_power():
    """'P = ...' not what the sources give in all, but one part's power (the question may ask for R2's): right, not
    sure; a value that is no part's power and not the total is wrong"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                   C("R2", "resistor", ["a", "0"], value="4k")], ground="0", node_labels={"a": "A"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("P = 20 mW") == [("right", True)] and checks("P = 16 mW") == [("right", False)]
    assert checks("P = 30 mW") == [("wrong", True)] and checks("P_{R2} = 4 mW") == [("wrong", True)]


def test_equations_against_two_circuits_on_a_page():
    """the problem's circuit and the student's own equivalent of it, or two questions' drawings: an equation right
    against one of them is right (not wrong against the other); a 'wrong' is sure only when both could judge it"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    from grader.selfcheck import _across_circuits
    one = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="10"), C("R1", "resistor", ["n0", "a"], value="1k"),
                   C("R2", "resistor", ["a", "0"], value="4k")], ground="0", node_labels={"a": "A"})
    two = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12"), C("R1", "resistor", ["n0", "a"], value="2k"),
                   C("R2", "resistor", ["a", "0"], value="1k")], ground="0", node_labels={"a": "A"})
    stmts = circuit_statements("V_A = 8") + circuit_statements("V_A = 3") + circuit_statements("V_A = 4")
    per = [check_against(stmts, n, problem=f"circuit {k}") for k, n in ((1, one), (2, two))]
    got = {(c.pieces[0], c.problem): (c.verdict, c.sure) for c in _across_circuits(per)}
    assert got[("V_A = 8", "circuit 1")] == ("right", True) and ("V_A = 8", "circuit 2") not in got
    assert got[("V_A = 4", "circuit 2")] == ("right", True) and ("V_A = 4", "circuit 1") not in got
    assert got[("V_A = 3", "circuit 1")] == ("wrong", True) and got[("V_A = 3", "circuit 2")] == ("wrong", True)


def test_wiring_notes_a_part_the_detector_missed():
    """a part whose box the detector missed leaves its symbol in the wires - a zigzag, a source's circle, a
    capacitor's plates: the reading notes it (Netlist.notes, among its problems, so no verdict on it is sure); with
    every box there nothing is noted"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.circuit_wiring import netlist_from_boxes
    img = Image.new("L", (1000, 700), 0)
    d = ImageDraw.Draw(img)
    line = lambda pts: d.line(pts, fill=255, width=5, joint="curve")      # noqa: E731
    line([(200, 300), (200, 150), (400, 150)])
    line([(400 + 25 * k, 150 + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)])   # a zigzag
    line([(600, 150), (800, 150), (800, 330)])
    d.line([(760, 330), (840, 330)], fill=255, width=5)                  # a capacitor's plates
    d.line([(760, 370), (840, 370)], fill=255, width=5)
    line([(800, 370), (800, 550), (200, 550), (200, 400)])
    d.ellipse((150, 300, 250, 400), outline=255, width=5)                # a source's circle
    mask = np.asarray(img) > 0
    src, res, cap = ("voltage_dc", (140, 290, 260, 410)), ("resistor", (390, 110, 610, 190)), ("capacitor", (750, 320, 850, 380))
    whole = netlist_from_boxes(mask, [src, res, cap], polarity=False)
    assert whole.notes == [] and len(whole.components) == 3
    for comps, what, (x, y) in [([src, cap], "turns over and over", (500, 150)), ([res, cap], "closed loop", (200, 350)),
                                ([src, res], "parallel strokes ending facing nothing", (800, 350))]:
        net = netlist_from_boxes(mask, comps, polarity=False)
        assert net.notes and all(what in n for n in net.notes), net.notes
        import re
        px, py = map(float, re.search(r"at \((\d+), (\d+)\)", net.notes[0]).groups())
        assert abs(px - x) < 60 and abs(py - y) < 60 and net.notes[0] in net.problems()
    # the reader then looks at what the detector saw there below its threshold: the best guess covering the spot is
    # taken when the reading has fewer problems with it (a box elsewhere, or a label, is not)
    from recog.circuit_reader import recover_missed
    wire = lambda m, c, t: netlist_from_boxes(m, c, t, return_debug=True, polarity=False)   # noqa: E731
    net, dbg = wire(mask, [src, cap], [])
    spares = [("inductor", (380, 100, 620, 200), 0.2), ("resistor", res[1], 0.3), ("text", (450, 120, 550, 180), 0.35),
              ("capacitor", (100, 600, 150, 650), 0.25)]
    net, _, comps, taken = recover_missed(net, dbg, mask, [src, cap], [], spares, wire)
    assert [t[0] for t in taken] == ["resistor"] and net.notes == [] and len(net.components) == 3 and len(comps) == 3
    from recog.circuit_reader import noted_spots, shift_notes     # the spots moved to the page's frame
    moved = shift_notes(netlist_from_boxes(mask, [src, cap], polarity=False), 0.5, 100, 10)
    assert all(abs(x - 350) < 40 and abs(y - 85) < 40 for x, y in noted_spots(moved)) and noted_spots(moved)
    # a table's grain taken for ink round the circuit: one note that the strokes are too dense to tell its wires
    rough = mask | (np.random.default_rng(0).random(mask.shape) < 0.3)
    notes = netlist_from_boxes(rough, [src, res, cap], polarity=False).notes
    assert len(notes) == 1 and "too dense" in notes[0]


def test_current_arrows_drawn_on_the_wires():
    """a current's arrowhead drawn on the wire itself (no arrow beside the part): the current of the part that wire
    leads to, the way the head points - or, when the wire branches ahead of it (a node), of the part it comes from"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.circuit_wiring import netlist_from_boxes
    img = Image.new("L", (1000, 800), 0)
    d = ImageDraw.Draw(img)
    line = lambda pts: d.line(pts, fill=255, width=5, joint="curve")      # noqa: E731
    zig_h = lambda x0, y: [(x0 + 25 * k, y + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)]   # noqa: E731
    zig_v = lambda x, y0: [(x + (0 if k in (0, 6) else (-25 if k % 2 else 25)), y0 + 25 * k) for k in range(7)]   # noqa: E731
    line([(150, 350), (150, 150), (200, 150)]); line(zig_h(200, 150)); line([(400, 150), (500, 150), (600, 150)])
    line(zig_h(600, 150)); line([(800, 150), (850, 150), (850, 350)])
    line([(500, 150), (500, 300)]); line(zig_v(500, 300)); line([(500, 450), (500, 650)])
    line([(150, 450), (150, 650), (850, 650), (850, 450)])
    d.ellipse((100, 350, 200, 450), outline=255, width=5); d.ellipse((800, 350, 900, 450), outline=255, width=5)
    d.ellipse((492, 142, 508, 158), fill=255)                           # node A's dot
    line([(438, 138), (455, 150), (438, 162)])                          # between R1 and A, pointing at A
    line([(488, 215), (500, 232), (512, 215)])                          # between A and the middle resistor, down
    line([(712, 638), (695, 650), (712, 662)])                          # on the bottom wire, pointing left
    mask = np.asarray(img) > 0
    comps = [("voltage_dc", (90, 340, 210, 460)), ("resistor", (190, 110, 410, 190)), ("resistor", (590, 110, 810, 190)),
             ("voltage_dc", (790, 340, 910, 460)), ("resistor", (460, 290, 540, 460))]
    texts = [((430, 95, 470, 125), "I1"), ((520, 205, 560, 235), "I2"), ((480, 100, 520, 130), "A"), ((680, 665, 720, 695), "I3")]
    net = netlist_from_boxes(mask, comps, texts, polarity=False)
    a = next(n for n, t in net.node_labels.items() if t == "A")
    cur = {c.label: (c.through, c.frm, c.to) for c in net.currents}
    r1 = next(c for c in net.components if c.id == "R1")
    mid = next(c for c in net.components if a in c.nodes and c.id not in ("R1", "R2"))
    assert cur["I1"] == ("R1", next(n for n in r1.nodes if n != a), a)          # into A from R1
    assert cur["I2"] == (mid.id, a, next(n for n in mid.nodes if n != a))       # out of A down through the middle one
    v2 = next(c for c in net.components if c.id == "V2")                       # left along the bottom wire: it
    bottom = next(n for n in v2.nodes if n in mid.nodes)                        # branches ahead (the middle one's
    assert cur["I3"] == ("V2", next(n for n in v2.nodes if n != bottom), bottom)  # wire), so V2's, out at the bottom
    assert not [p for p in net.problems() if "direction" in p and p.startswith(("I1", "I2"))]


def test_circuit_reader_end_to_end_with_a_missed_part_and_a_wire_arrow():
    """the reader as a whole (a stand-in detector): the resistor the detector saw only below its threshold is taken
    from its spares once the wiring notes its zigzag in the wires, and I1's arrowhead on the wire then gives R1's
    current; each label is read knowing the part beside it ('?' - none near)"""
    from PIL import Image, ImageDraw
    from recog.circuit_reader import CircuitReader
    img = Image.new("RGB", (1000, 700), (245, 243, 238))
    d = ImageDraw.Draw(img)
    pen = (40, 40, 40)
    line = lambda pts: d.line(pts, fill=pen, width=5, joint="curve")      # noqa: E731
    line([(200, 300), (200, 150), (400, 150)])
    line([(400 + 25 * k, 150 + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)])
    line([(600, 150), (800, 150), (800, 330)])
    d.line([(760, 330), (840, 330)], fill=pen, width=5)
    d.line([(760, 370), (840, 370)], fill=pen, width=5)
    line([(800, 370), (800, 550), (200, 550), (200, 400)])
    d.ellipse((150, 300, 250, 400), outline=pen, width=5)
    line([(285, 138), (302, 150), (285, 162)])

    class Detector:
        last_spares = []

        def __call__(self, small):
            s = small.width / 1000
            sc = lambda b: tuple(v * s for v in b)                       # noqa: E731
            self.last_spares = [("resistor", sc((390, 110, 610, 190)), 0.3)]
            return [("voltage_dc", sc((140, 290, 260, 410)), 0.9), ("capacitor", sc((750, 320, 850, 380)), 0.8),
                    ("text", sc((280, 95, 320, 125)), 0.9)]
    asked = []

    class Labels:
        def read(self, crops, kinds=None):
            asked.append(kinds)
            return [("I1", 0.95) for _ in crops]
    net, found = CircuitReader(detector=Detector(), strokes=False, labels=Labels()).read(img)
    assert sorted(c.type for c in net.components) == ["capacitor", "resistor", "voltage_source"] and net.notes == []
    r1 = next(c for c in net.components if c.type == "resistor")
    v1 = next(c for c in net.components if c.type == "voltage_source")
    top = next(n for n in r1.nodes if n in v1.nodes)
    assert [(c.label, c.through, c.frm) for c in net.currents] == [("I1", r1.id, top)]
    assert ("resistor", 0.3) in [(c, sc) for c, _, sc in found] and asked == [["?"]]


def test_decimal_comma_in_derivations():
    """'c = 2,5' (the reader writes '2 , 5') is two and a half, not 'c = 2' and a stray 5; '1 , 0 0 0' is a thousand;
    'a = 3 , b = 4' stays two statements"""
    from grader.selfcheck import statements
    assert [(s.kind, s.number) for s in statements("c = 2 , 5")] == [("value", 2.5)]
    assert [s.number for s in statements("x = 1 , 0 0 0")] == [1000.0]
    assert [s.number for s in statements("a = 3 , b = 4")] == [3.0, 4.0]


def _marked_drawing(texts_extra=(), ac=False):
    """a source at the left, R1 along the top, R2 (or C1, AC) down the right side - drawn in a mask"""
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.new("L", (1000, 800), 0)
    d = ImageDraw.Draw(img)
    line = lambda pts: d.line(pts, fill=255, width=5, joint="curve")      # noqa: E731
    zig_h = lambda x0, y: [(x0 + 25 * k, y + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)]   # noqa: E731
    zig_v = lambda x, y0: [(x + (0 if k in (0, 6) else (-25 if k % 2 else 25)), y0 + 25 * k) for k in range(7)]   # noqa: E731
    line([(150, 350), (150, 150), (300, 150)]); line(zig_h(300, 150))
    if ac:
        line([(500, 150), (700, 150), (700, 330)])
        d.line([(660, 330), (740, 330)], fill=255, width=5); d.line([(660, 370), (740, 370)], fill=255, width=5)
        line([(700, 370), (700, 650), (150, 650), (150, 450)])
    else:
        line([(500, 150), (700, 150), (700, 300)]); line(zig_v(700, 300)); line([(700, 450), (700, 650), (150, 650), (150, 450)])
    d.ellipse((100, 350, 200, 450), outline=255, width=5)
    comps = [("voltage_ac" if ac else "voltage_dc", (90, 340, 210, 460)), ("resistor", (290, 110, 510, 190)),
             ("capacitor", (650, 320, 750, 380)) if ac else ("resistor", (660, 290, 740, 460)), ("gnd", (680, 640, 720, 700))]
    return np.asarray(img) > 0, comps


def test_voltage_marks_beside_parts():
    """'+' and '-' drawn by a part's two ends with a voltage's name between them ('V_x', 'v_o'): that voltage, '+'
    node minus '-' node, its sign checked; a source's '+' / '-' drawn beside it: its polarity; the name is the
    voltage's, not a part's"""
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit_check import check_against, circuit_statements
    mask, comps = _marked_drawing()
    for flip in (False, True):
        a, b = ("-", "+") if flip else ("+", "-")
        texts = [((280, 70, 310, 100), a), ((490, 70, 520, 100), b), ((370, 60, 430, 100), "V_x"),
                 ((760, 290, 790, 320), "+"), ((760, 430, 790, 460), "-"), ((760, 355, 810, 395), "v_o"),
                 ((40, 380, 80, 420), "10V"), ((360, 200, 440, 230), "2k"), ((600, 360, 650, 390), "3k"),
                 ((215, 320, 245, 350), "+"), ((215, 455, 245, 485), "-")]
        net = netlist_from_boxes(mask, comps, texts, polarity=False)
        src = next(c for c in net.components if c.type == "voltage_source")
        top = next(n for n in src.nodes if n != "0")
        assert src.plus == top and not net.problems() and all(c.label is None for c in net.components)
        vx = next(v for v in net.voltages if v.label == "V_x")
        vo = next(v for v in net.voltages if v.label == "v_o")
        assert vx.across == "R1" and vo.across == "R2" and vo.minus == "0"
        assert (vx.plus == top) != flip

        def verdicts(tex):
            return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
        assert verdicts("V_x = 4") == [("wrong" if flip else "right", True)]
        assert verdicts("V_x = -4") == [("right" if flip else "wrong", True)]
        assert verdicts("v_o = 6") == [("right", True)] and verdicts("v_o = -6") == [("wrong", True)]


def test_ac_source_polarity_open_or_marked():
    """an AC source drawn as a circle with a sine has no polarity: answers are checked both ways round it - right
    either way, but not sure; its '+' / '-' drawn beside it fix the phasor's reference (sure verdicts)"""
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit_check import check_against, circuit_statements
    mask, comps = _marked_drawing(ac=True)
    base = [((10, 380, 85, 420), "10cos(1000t)"), ((360, 200, 440, 230), "1k"), ((760, 335, 820, 365), "1u"),
            ((560, 100, 600, 130), "A")]
    net = netlist_from_boxes(mask, comps, base, polarity=False)

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert verdicts("V_A = 5 - j5") == [("right", False)] and verdicts("V_A = -5 + j5") == [("right", False)]
    assert verdicts("V_A = 5 + j5") == [("wrong", True)]
    net = netlist_from_boxes(mask, comps, base + [((215, 320, 245, 350), "-"), ((215, 455, 245, 485), "+")], polarity=False)
    assert verdicts("V_A = -5 + j5") == [("right", True)] and verdicts("V_A = 5 - j5") == [("wrong", True)]


def test_marked_voltage_after_a_switching():
    """a voltage marked across the resistor of a switched RC: v_x(t), v_x(0^+) with the sign the marks give"""
    from grader.circuit import Component, Netlist, Voltage
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([Component("E", "voltage_source", ["a", "0"], value="10V", plus="a"),
                   Component("S1", "switch", ["a", "b"], value="t=0", state="open"),
                   Component("R1", "resistor", ["b", "c"], value="1k"),
                   Component("C1", "capacitor", ["c", "0"], value="1u")], ground="0",
                  voltages=[Voltage("v_x", "c", "b", "R1")])

    def verdicts(tex):
        return [c.verdict for c in check_against(circuit_statements(tex), net)]
    assert verdicts("v_x(t) = -10 e^{-1000t}") == ["right"] and verdicts("v_x(t) = 10 e^{-1000t}") == ["wrong"]
    assert verdicts("v_x(0^+) = -10") == ["right"] and verdicts("v_x(0^+) = 10") == ["wrong"]


def test_coupled_coils_dots_read():
    """a dot drawn by one end of each coil (a small filled blob touching nothing): that end is the coil's '+' - the
    primary's at its bottom here, the secondary's at its top - and the netlist no longer flags the dots"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.circuit_wiring import netlist_from_boxes
    m = np.zeros((400, 900), bool)
    m[100:104, 50:350] = True
    m[300:304, 50:350] = True
    m[100:304, 50:54] = True
    m[100:104, 450:800] = True
    m[300:304, 450:800] = True
    m[100:304, 796:800] = True
    comps = [("voltage_dc", (25, 170, 80, 230)), ("resistor", (170, 80, 230, 125)), ("transformer", (340, 90, 460, 315)),
             ("resistor", (775, 170, 820, 230))]
    for cls, (x0, y0, x1, y1) in comps:
        if cls != "transformer":
            m[y0 + 5:y1 - 5, x0 + 10:x1 - 10] = True
    m[110:295, 360:380] = True
    m[110:295, 420:440] = True
    img = Image.fromarray(m.astype(np.uint8) * 255)
    d = ImageDraw.Draw(img)
    d.ellipse((343, 279, 353, 289), fill=255)                            # by the primary's bottom end
    d.ellipse((447, 110, 457, 120), fill=255)                            # by the secondary's top end
    m = np.asarray(img) > 0
    labels = [((380, 40, 420, 70), "1:2"), ((150, 40, 200, 70), "1Ω"), ((830, 180, 870, 210), "4Ω"), ((90, 180, 130, 210), "10V")]
    net = netlist_from_boxes(m, comps, labels, polarity=False)
    t = next(c for c in net.components if c.type == "transformer")
    v = next(c for c in net.components if c.type == "voltage_source")
    r_far = next(c for c in net.components if c.type == "resistor" and c.value == "4Ω")
    assert t.state == "dots read" and not any("dots" in p_ for p_ in net.problems())
    assert t.roles["P+"] in v.nodes and t.roles["P-"] not in v.nodes      # the primary's dot at the source's side
    top = next(n for n in r_far.nodes if n != t.roles["S-"])
    assert t.roles["S+"] == top


def test_port_marks_and_an_unmarked_ports_sign():
    """V_th between two open terminals: the drawing's '+' / '-' by them (or names a / b) say which is '+'; with
    neither, either sign is right (which terminal is '+' is the student's choice)"""
    import numpy as np
    from PIL import Image, ImageDraw
    from recog.circuit_wiring import netlist_from_boxes
    from grader.circuit_check import check_against, circuit_statements
    img = Image.new("L", (1000, 800), 0)
    d = ImageDraw.Draw(img)
    line = lambda pts: d.line(pts, fill=255, width=5, joint="curve")      # noqa: E731
    zig_h = lambda x0, y: [(x0 + 25 * k, y + (0 if k in (0, 8) else (-30 if k % 2 else 30))) for k in range(9)]   # noqa: E731
    zig_v = lambda x, y0: [(x + (0 if k in (0, 6) else (-25 if k % 2 else 25)), y0 + 25 * k) for k in range(7)]   # noqa: E731
    line([(150, 350), (150, 150), (300, 150)]); line(zig_h(300, 150)); line([(500, 150), (600, 150), (600, 300)])
    line(zig_v(600, 300)); line([(600, 450), (600, 650), (150, 650), (150, 450)])
    line([(600, 150), (850, 150)]); line([(600, 650), (850, 650)])
    d.ellipse((100, 350, 200, 450), outline=255, width=5)
    d.ellipse((850, 140, 870, 160), outline=255, width=4); d.ellipse((850, 640, 870, 660), outline=255, width=4)
    mask = np.asarray(img) > 0
    comps = [("voltage_dc", (90, 340, 210, 460)), ("resistor", (290, 110, 510, 190)), ("resistor", (560, 290, 640, 460)),
             ("terminal", (845, 135, 875, 165)), ("terminal", (845, 635, 875, 665))]
    base = [((40, 380, 80, 420), "12V"), ((360, 200, 440, 230), "4k"), ((660, 360, 710, 390), "12k"),
            ((215, 320, 245, 350), "+"), ((215, 455, 245, 485), "-")]
    for marks, plus_top in (([], None), ([((890, 170, 920, 200), "+"), ((890, 600, 920, 630), "-")], True),
                            ([((890, 170, 920, 200), "-"), ((890, 600, 920, 630), "+")], False)):
        net = netlist_from_boxes(mask, comps, base + marks, polarity=False)

        def verdicts(tex):
            return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
        assert verdicts("R_{th} = 3k") == [("right", True)]
        if plus_top is None:
            assert not net.ports_marked and verdicts("V_{th} = 9") == verdicts("V_{th} = -9") == [("right", True)]
        else:
            assert net.ports_marked
            assert verdicts("V_{th} = 9") == [("right" if plus_top else "wrong", True)]
            assert verdicts("V_{th} = -9") == [("wrong" if plus_top else "right", True)]


def test_units_in_brackets_and_chains_of_equalities():
    """'3.33 [V]' (a unit in square brackets, as Israeli exams write it - a prefix counts: '[mA]'; after an answer in
    the parts' names it is dropped), '\\simeq' / '\\cong' / '\\div', a repeating decimal '3.\\overline{3}', a numeric
    fraction with its unit after it; a chain 'V_A = ... = ...' checked one equality at a time - each later side
    against the quantity named first: an arithmetic slip at its end is wrong"""
    from grader.circuit import Component, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([Component("E", "voltage_source", ["a", "0"], value="10V", plus="a", label="E"),
                   Component("R1", "resistor", ["a", "m"], value="2k", label="R_1"),
                   Component("R2", "resistor", ["m", "0"], value="1k", label="R_2")], ground="0", node_labels={"m": "A"})

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    for tex in ("V_A = 3.33 [V]", "V_A = 3.33 \\left[ V \\right]", "I = 3.33 [mA]", "I = \\frac{10}{3} [mA]", "R_{eq} = 3 [k\\Omega]",
                "V_A \\simeq 3.33", "V_A \\cong 3.33", "V_A = 3.\\overline{3}", "V_A = 10 \\div 3", "V_A = \\frac{10}{3}V",
                "V_A = \\frac{E R_2}{R_1 + R_2} [V]", "R_{eq} = R_1 + R_2 [\\Omega]",
                "V_A = 3.33 = \\frac{10}{3}", "V_A = \\frac{E R_2}{R_1+R_2} = 3.33", "R_{eq} = R_1 + R_2 = 2k + 1k = 3k"):
        assert verdicts(tex) == [("right", True)], tex
    assert verdicts("I = 3 [mA]") == [("wrong", True)]
    assert verdicts("V_A = \\frac{10}{3} = 3.66") == [("wrong", True)]                    # the slip at the end
    assert verdicts("V_A = \\frac{E R_1}{R_1+R_2} = 3.33") == [("wrong", True)]          # the slip in the middle


def test_magnitudes_and_an_ac_fed_filters_functions():
    """|V_o|, |Z|, |I| (the bars are not statement separators): the size of the circuit's value; a filter fed by an AC
    source: H(jw), H(s), Z_in(s), the source a unit input; a phasor's or a wave's unit in brackets with its prefix
    ('[mA]', '[k\\Omega]'); an angle as a fraction of pi"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["a", "0"], value="10cos(1000t)", plus="a", ac=True),
                   C("R1", "resistor", ["a", "m"], value="1k"), C("C1", "capacitor", ["m", "0"], value="1u")],
                  ground="0", node_labels={"m": "o"})

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    for tex in ("|V_o| = 7.07", "|Z| = 1.41 [k\\Omega]", "|I| = 7.07 mA", "\\left| V_o \\right| = 7.07 V", "|H(j1000)| = 0.707",
                "H(s) = \\frac{1}{1 + 0.001 s}", "Z_{in}(s) = 1000 + \\frac{10^6}{s}", "V_o = 5 - j5 [V]",
                "I = 7.07 \\angle 45^\\circ [mA]", "Z = 1 - j1 [k\\Omega]", "i(t) = 7.07 \\cos(1000t + 45^\\circ) [mA]",
                "V_o = 7.07 \\angle -\\frac{\\pi}{4} [V]"):
        assert verdicts(tex) == [("right", True)], tex
    for tex in ("|V_o| = 5", "H(s) = \\frac{1}{1 + s}", "I = 4 + j5 [mA]"):
        assert verdicts(tex) == [("wrong", True)], tex


def test_maximum_power_at_a_port():
    """two open terminals and no load drawn: P_max = V_th^2 / (4 R_th) is checked; 'R_L = 3k' is right (not sure) when
    it is R_th - another R_L may be the problem's own data, so it is not called wrong"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("E", "voltage_source", ["t", "0"], value="12V", plus="t"), C("R1", "resistor", ["t", "a"], value="4k"),
                   C("R2", "resistor", ["a", "0"], value="12k")], ground="0", node_labels={"a": "a", "0": "b"}, ports=["a", "0"])

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert verdicts("P_{max} = 6.75 mW") == verdicts("P_{max} = \\frac{V_{th}^2}{4R_{th}}") == [("right", True)]
    assert verdicts("P_{max} = 13.5 mW") == [("wrong", True)]
    assert verdicts("R_L = 3k") == [("right", False)] and verdicts("R_L = 6k") == []


def test_gain_in_decibels_and_an_op_amps_input_voltages():
    """'A_v = 20 dB', '20\\log|A_v| = 20', '|A_v|_{dB} = 20' (a power gain: 10 log10); an op-amp's V_+ / V_- (its
    inputs - a wrong one not sure: V_+ / V_- may name the supply rails)"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    inv = Netlist([C("V1", "voltage_source", ["i", "0"], value="1V", plus="i"), C("R1", "resistor", ["i", "m"], value="1k", label="R_1"),
                   C("R2", "resistor", ["m", "o"], value="10k", label="R_f"),
                   C("OA1", "opamp", ["0", "m", "o"], roles={"In+": "0", "In-": "m", "Out": "o"})], ground="0", node_labels={"o": "o"})
    non = Netlist([C("V1", "voltage_source", ["i", "0"], value="1V", plus="i"), C("R1", "resistor", ["m", "0"], value="1k", label="R_1"),
                   C("R2", "resistor", ["m", "o"], value="9k", label="R_f"),
                   C("OA1", "opamp", ["i", "m", "o"], roles={"In+": "i", "In-": "m", "Out": "o"})], ground="0", node_labels={"o": "o"})

    def verdicts(tex, net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    for tex in ("A_v = 20 dB", "A_v = 20 [dB]", "20\\log|A_v| = 20", "|A_v|_{dB} = 20", "A_v = -10"):
        assert verdicts(tex, inv) == [("right", True)], tex
    assert verdicts("A_v = 26 dB", inv) == [("wrong", True)]
    assert verdicts("V_- = 1", non) == verdicts("V_{-} = V_{+} = 1", non) == [("right", True)]
    assert verdicts("V^- = -15", non) == [("wrong", False)]


def test_operating_point_names_and_pair():
    """a BJT's operating point: I_CQ, V_CEQ, I_BQ by name, and 'Q = (4.98 V, 1.23 mA)' as a pair (either order, by the
    units; '[V]' too)"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    q = C("Q1", "npn", ["b", "c", "e"], value="β=100")
    q.roles = {"Base": "b", "Collector": "c", "Emitter": "e"}
    net = Netlist([C("V1", "voltage_source", ["v", "0"], plus="v", value="12", label="V_{CC}"),
                   C("R1", "resistor", ["v", "b"], value="100k", label="R_1"), C("R2", "resistor", ["b", "0"], value="22k", label="R_2"),
                   C("RC", "resistor", ["v", "c"], value="4.7k", label="R_C"), C("RE", "resistor", ["e", "0"], value="1k", label="R_E"), q],
                  ground="0", node_labels={"b": "B", "c": "C", "e": "E"})

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    for tex in ("I_{CQ} = 1.23 mA", "V_{CEQ} = 4.98 [V]", "I_{BQ} = 12.3 \\mu A", "Q = (4.98 V, 1.23 mA)", "Q(1.23mA, 4.98V)",
                "Q_Q = (4.98 [V], 1.23 [mA])"):
        assert verdicts(tex) == [("right", True)], tex
    assert verdicts("Q = (5.5 V, 1.23 mA)") == [("wrong", True)] and verdicts("I_{CQ} = 2 mA") == [("wrong", True)]


def test_dependent_source_following_a_marked_voltage():
    """a dependent source labelled '0.001 V_x' where V_x is a voltage marked '+' / '-' across R1 (not a node's): it
    follows that voltage - and the copies of the netlist the checks solve keep the marks"""
    from grader.circuit import Component as C, Netlist, Voltage
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("E", "voltage_source", ["a", "0"], value="10V", plus="a"), C("R1", "resistor", ["a", "b"], value="1k"),
                   C("R2", "resistor", ["b", "0"], value="1k"),
                   C("G1", "dependent_current_source", ["0", "b"], value="0.001 V_x", plus="b")],
                  ground="0", node_labels={"b": "B"}, voltages=[Voltage("V_x", "a", "b", "R1")])
    assert net.solve()["V_B"] == __import__("sympy").Rational(20, 3)

    def verdicts(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert verdicts("V_B = 6.67") == verdicts("V_x = 3.33") == verdicts("I_{R2} = 6.67 mA") == [("right", True)]
    assert verdicts("V_x = -3.33") == [("wrong", True)]


def test_checks_marked_beside_their_lines():
    """each check knows the page line it is about (a chain's, an equation system's last line), and read_page draws its
    verdict there: a tick, a cross or a '?'"""
    from types import SimpleNamespace as NS
    from PIL import Image, ImageDraw
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    from read_page import mark_checks
    net = Netlist([C("E", "voltage_source", ["a", "0"], value="10V", plus="a"), C("R1", "resistor", ["a", "m"], value="2k"),
                   C("R2", "resistor", ["m", "0"], value="1k")], ground="0", node_labels={"m": "A"})
    checks = check_against(circuit_statements("V_A = 3.33", line=2) + circuit_statements("V_A = 4", line=0), net)
    assert sorted((c.line, c.verdict) for c in checks) == [(0, "wrong"), (2, "right")]
    img = Image.new("RGB", (400, 200), "white")
    lines = [[NS(box=(100, 10, 300, 40))], [NS(box=(100, 70, 300, 100))], [NS(box=(5, 130, 300, 160))]]
    mark_checks(ImageDraw.Draw(img), checks, lines)
    px = img.load()
    assert any(px[x, y][0] > 180 and px[x, y][1] < 60 for x in range(60, 95) for y in range(10, 45))     # red cross, line 1
    assert any(px[x, y][1] > 120 and px[x, y][0] < 60 for x in range(2, 30) for y in range(125, 165))     # green tick, line 3


def test_derivation_data_above_and_units_worked_out_in_si():
    """the data written above a computation count for it ('a = 3', 'b = 4', then 'c = \\sqrt{a^2+b^2} = 6' - a false
    step), and the givens just above a formula for that formula; every value written with its unit: the formula is
    worked out in SI units and compared in the answer's own ('R = 5000 \\Omega' for 10 V / 2 mA is as right as
    '5 k\\Omega', and '5 \\Omega' is wrong - the numbers as written would take 2 mA for 2 A)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [c.verdict for c in check_statements(st)]
    assert verdicts("a = 3", "b = 4", "c = \\sqrt{a^2 + b^2} = 6") == ["wrong step"]
    assert verdicts("a = 3", "b = 4", "c = \\sqrt{a^2 + b^2} = 5") == ["right"]
    assert verdicts("V = 10 V", "I = 2 mA", "R = \\frac{V}{I}", "R = 5 k\\Omega") == ["right"]
    assert verdicts("V = 10 V", "I = 2 mA", "R = \\frac{V}{I}", "R = 5000 \\Omega") == ["right"]
    assert verdicts("V = 10 V", "I = 2 mA", "R = \\frac{V}{I}", "R = 5 \\Omega") == ["wrong"]
    assert verdicts("P = \\frac{V^2}{R} \\vert V = 10 V, R = 2 k\\Omega", "P = 0.05 W") == ["right"]
    assert verdicts("P = \\frac{V^2}{R} \\vert V = 10 V, R = 2 k\\Omega", "P = 50 mW") == ["right"]
    assert verdicts("V = I R \\vert I = 2, R = 5", "V = 10") == ["right"]                 # no units: as written
    assert verdicts("V = 10 [V]", "I = 2 [mA]", "R = \\frac{V}{I}", "R = 5 [k\\Omega]") == ["right"]


def test_current_squared_keeps_its_power():
    """a lone I is the current: its powers are kept ('I^2 R' was read as I R by the LaTeX parser, which takes I for
    sqrt(-1)) - in the formula comparison and in the derivation checks"""
    from grader.selfcheck import _parse, check_statements, statements
    from grader.symbolic import compare_formula
    assert str(_parse("I^2 R")) == "I**2*R" and str(_parse("\\frac{V}{I}")) == "V/I"       # (letters keep their case)
    assert compare_formula("P = I R", "P = I^2 R").status == "not_equivalent"
    assert compare_formula("P = R I^{2}", "P = I^2 R").status == "equivalent"
    st = statements("P = I^2 R \\vert I = 2 A, R = 5 \\Omega") + statements("P = 10 W")
    assert [c.verdict for c in check_statements(st)] == ["wrong"]


def test_trailing_letters_are_not_taken_for_units():
    """Math-Verify's unit stripping dropped a trailing c d g h l m o s t ('E = m g h' read as E = m): off - in the
    formula comparison and in the derivation checks"""
    from grader.selfcheck import _parse, check_statements, statements
    from grader.symbolic import compare_formula
    assert str(_parse("m g h")) == "g*h*m" and str(_parse("x_0 + v_0 t")) == "t*v_0 + x_0" and str(_parse("\\omega t")) == "omega*t"
    assert compare_formula("E = m g", "E = m g h").status == "not_equivalent"
    assert compare_formula("E = mgh", "E = m g h").status == "equivalent"
    st = statements("E = m g h \\vert m = 2 kg, g = 10, h = 5 m") + statements("E = 20 J")
    assert [c.verdict for c in check_statements(st)] == ["wrong"]


def test_percentages_in_derivations():
    """'\\eta = 80\\%' is 0.8 - against a formula that gives a ratio, or one with '\\cdot 100\\%' in it"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for t in lines:
            st += statements(t)
        return [c.verdict for c in check_statements(st)]
    eta = "\\eta = \\frac{P_{out}}{P_{in}} \\vert P_{out} = 80 W, P_{in} = 100 W"
    assert verdicts(eta, "\\eta = 80\\%") == verdicts(eta, "\\eta = 0.8") == ["right"]
    assert verdicts(eta, "\\eta = 70\\%") == ["wrong"]
    assert verdicts("\\eta = \\frac{P_{out}}{P_{in}} \\cdot 100\\% \\vert P_{out} = 80 W, P_{in} = 100 W", "\\eta = 80\\%") == ["right"]


def test_scientific_e_and_delta_names():
    """'1.5e-3' is 1.5 times 10^-3 (not Euler's e minus 3); '\\Delta V' one name (not delta times V) - a derivation
    with them is checked"""
    from grader.selfcheck import _parse, check_statements, statements
    assert str(_parse("1.5e-3")) == str(_parse("1.5 \\cdot 10^{-3}")) and str(_parse("2e^{-t}")) == "2*exp(-t)"
    assert len(_parse("\\Delta V").free_symbols) == 1 and len(_parse("\\frac{\\Delta V}{\\Delta I}").free_symbols) == 2
    st = statements("R = \\frac{\\Delta V}{\\Delta I} \\vert \\Delta V = 2 V, \\Delta I = 4 mA") + statements("R = 500 \\Omega")
    assert [c.verdict for c in check_statements(st)] == ["right"]


def test_degrees_and_products_in_derivations():
    """sin(30°) is 0.5 (degrees read as such, not radians), an angle given in degrees fits a formula in radians;
    '2(3)', '2\\pi(50)(0.1)' are products"""
    from grader.selfcheck import _parse, check_statements, statements

    def verdicts(*lines):
        st = []
        for t in lines:
            st += statements(t)
        return [c.verdict for c in check_statements(st)]
    assert verdicts("\\sin(30^\\circ) = 0.5") == verdicts("\\cos(60^{\\circ}) = 0.5") == ["right"]
    assert verdicts("P = V I \\cos(\\phi) \\vert V = 10, I = 2, \\phi = 60^\\circ", "P = 10") == ["right"]
    assert verdicts("\\theta = \\arctan(\\frac{X}{R}) \\vert X = 1, R = 1", "\\theta = 45^\\circ") == ["right"]
    assert str(_parse("2(3)")) == "2*3" and verdicts("X_L = 2\\pi(50)(0.1) = 31.4") == ["right"]


def test_substitution_with_units_inline():
    """'V = I R = 2 mA \\cdot 5 k\\Omega = 10 V': a substitution written with its units is worked out in SI units and
    its steps checked ('[mA]' too) - the units' letters were read as letters of the problem, and nothing was checked"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [c.verdict for c in check_statements(statements(t))]
    for t in ("V = I R = 2 mA \\cdot 5 k\\Omega = 10 V", "V = I R = 2 [mA] \\cdot 5 [k\\Omega] = 10 [V]",
              "I = \\frac{V}{R} = \\frac{10 V}{5 k\\Omega} = 2 mA", "P = V I = 10 V \\cdot 2 mA = 20 mW"):
        assert verdicts(t) == ["right"], t
    assert verdicts("V = I R = 2 mA \\cdot 5 k\\Omega = 12 V") == verdicts("I = \\frac{V}{R} = \\frac{10 V}{5 k\\Omega} = 2 A") == ["wrong step"]
    assert verdicts("R = \\frac{10}{2} = 5 k\\Omega") == ["right"]          # numbers only: engineering units, as written


def test_values_with_a_bare_prefix():
    """'R_1 = 1k', 'C = 10\\mu' (how EE students write values) are values, in the units as written - they used to be
    taken for formulas and break the problem; an answer in a lone 'm' ('22 m' for 22 ms) is taken as written"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for t in lines:
            st += statements(t)
        return [c.verdict for c in check_statements(st)]
    div = "V_{out} = V_{in} \\frac{R_2}{R_1 + R_2} \\vert V_{in} = 12, R_1 = 1k, R_2 = 2k"
    assert verdicts(div, "V_{out} = 8") == ["right"] and verdicts(div, "V_{out} = 6") == ["wrong"]
    rc = "\\tau = R C \\vert R = 2.2 k\\Omega, C = 10 \\mu F"
    assert verdicts(rc, "\\tau = 22 m") == verdicts(rc, "\\tau = 22 ms") == verdicts(rc, "\\tau = 0.022 s") == ["right"]
    assert verdicts(rc, "\\tau = 22 s") == ["wrong"]


def test_parallel_resistances_in_derivations():
    """'R_1 \\| R_2' (\\parallel, ||) is R_1 R_2 / (R_1 + R_2) - the LaTeX parser read '2 \\| 3' as 3, so a right
    '2 \\| 3 = 1.2' was a false step; chains of them left to right"""
    from grader.selfcheck import _parse, check_statements, statements
    from grader.symbolic import compare_formula

    def verdicts(*lines):
        st = []
        for t in lines:
            st += statements(t)
        return [c.verdict for c in check_statements(st)]
    assert verdicts("R_{eq} = 2 \\| 3 = 1.2") == verdicts("R_{eq} = 6 \\| 3 \\| 2 = 1") == ["right"]
    assert verdicts("R_{eq} = 2 \\| 3 = 1.5") == ["wrong step"]
    assert verdicts("R_{eq} = R_1 \\| R_2 \\vert R_1 = 2, R_2 = 3", "R_{eq} = 1.2") == ["right"]
    assert compare_formula("R = R_1 \\parallel R_2", "R = \\frac{R_1 R_2}{R_1 + R_2}").status == "equivalent"


def test_substitution_step_in_si_or_as_written():
    """'R = 5 k\\Omega' above, then 'I = V/R = 10/5000 = 0.002 A' (substituted in SI units) or 'I = V/R = 10/5 = 2'
    (as written): either way the step holds"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [c.verdict for c in check_statements(st)]
    assert "wrong step" not in verdicts("V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R} = \\frac{10}{5000} = 0.002 A")
    assert "wrong step" not in verdicts("V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R} = \\frac{10}{5} = 2 mA")
    assert "wrong step" in verdicts("V = 10 V", "R = 5 k\\Omega", "I = \\frac{V}{R} = \\frac{10}{50} = 0.2 A")


def test_derivations_as_the_reader_writes_them():
    """the formula reader's habits in derivations - spaced letters in a subscript ('V _ { o u t }'), a unit in
    \\mathrm with its letters spaced ('\\mathrm { m A }'), spaced digits, \\left( \\right) products - are evened out
    as in the circuit checks (they gave 'cannot check' or nothing)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [c.verdict for c in check_statements(st)]
    assert verdicts("V _ { o u t } = V _ { i n } \\frac{R_2}{R_1 + R_2} \\vert V _ { i n } = 1 2, R_1 = 1k, R_2 = 2k",
                    "V _ { o u t } = 8") == ["right"]
    assert verdicts("V = I R \\vert I = 2 \\mathrm { m A }, R = 5 k\\Omega", "V = 1 0 V") == ["right"]
    assert verdicts("\\tau = R C \\vert R = 1 k\\Omega, C = 1 \\mu F", "\\tau = 1 \\mathrm { m s }") == ["right"]
    assert verdicts("X_L = 2\\pi \\left( 5 0 \\right) \\left( 0 . 1 \\right) = 3 1 . 4") == ["right"]


def test_sums_limits_integrals_in_derivations():
    """a sum, a product, a limit or an integral is worked out ('\\sum_{k=1}^{4} k = 10' - and the '=' inside its
    braces does not cut the statement)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [c.verdict for c in check_statements(statements(t))]
    for t in ("\\sum_{k=1}^{4} k = 10", "\\prod_{k=1}^{4} k = 24", "\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1", "\\int_0^1 x dx = 0.5"):
        assert verdicts(t) == ["right"], t
    assert verdicts("\\sum_{k=1}^{4} k = 11") == verdicts("\\int_0^1 x dx = 1") == ["wrong step"]


def test_numeric_answers_written_the_israeli_way():
    """the rubric's numeric answers: a unit in square brackets ('3.33 [V]', '2 [mA]'), '80\\%', '45^\\circ', a number
    worked out ('\\frac{10}{3} V', '10\\sqrt{2} V', '3.\\overline{3}'), a bare prefix ('5k' for 5 kΩ - the unit then
    counts as missing; '5 m' for 5 mA) - all used to be parse errors or a unit mismatch"""
    from grader.numeric import compare_numeric, unit_given
    for st, ref in (("3.33 [V]", "3.33 V"), ("2 [mA]", "0.002 A"), ("3.33 \\left[ V \\right]", "3.33 V"), ("80\\%", "0.8"),
                    ("45^\\circ", "45 deg"), ("\\frac{10}{3} V", "3.333 V"), ("10\\sqrt{2} V", "14.14 V"), ("3.\\overline{3} V", "3.333 V"),
                    ("5k", "5000 ohm"), ("5 k", "5 kohm"), ("5 m", "0.005 A"), ("5 m", "5 m")):
        assert compare_numeric(st, ref).status == "correct", (st, ref)
    assert compare_numeric("4 [V]", "3.33 V").status == "wrong" and compare_numeric("5k", "4000 ohm").status == "wrong"
    assert not unit_given("5k", "5000 ohm") and unit_given("3.33 [V]", "3.33 V") and unit_given("5 m", "5 m")


def test_equations_with_infinity_compared_side_by_side():
    """'\\lim_{x \\to \\pi/2 + 0} \\tan x = -\\infty' against itself: the sides compared one by one (their difference is
    oo - oo, which is no number: the same equation was judged different, sure)"""
    from grader.symbolic import compare_formula
    t = "\\lim _ { x \\rightarrow \\frac { \\pi } { 2 } + 0 } \\tan x = - \\infty"
    assert compare_formula(t, t).status == "equivalent"
    assert compare_formula(t, t.replace("- \\infty", "\\infty")).status == "not_equivalent"
    assert compare_formula("x^2 + 2x = 3", "x^2 + 2x - 3 = 0").status == "equivalent"


def test_equation_roots_named_apart():
    """'s^2 + 3s + 2 = 0', then 's_1 = -1', 's_2 = -2' (a characteristic equation's roots): each checked against the
    equation's solutions - they used to get no verdict"""
    from grader.selfcheck import check_statements, statements

    def verdicts(*lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [c.verdict for c in check_statements(st)]
    assert verdicts("s^2 + 3s + 2 = 0", "s_1 = -1", "s_2 = -2") == ["right", "right"]
    assert verdicts("x^2 - 5x + 6 = 0", "x_1 = 2, x_2 = 4") == ["right", "wrong"]


def test_complex_numbers_and_magnitudes_in_derivations():
    """j is the imaginary unit ('\\frac{10}{3 + j4} = 1.2 - j1.6'), a phasor in polar form ('2 \\angle -53.13^\\circ',
    degrees), a magnitude '|3 + j4| = 5' (the bars are not statement separators)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [c.verdict for c in check_statements(statements(t))]
    for t in ("(3 + j4)(1 - j) = 7 + j", "I = \\frac{10}{3 + j4} = 1.2 - j1.6", "I = \\frac{10}{3 + j4} = 2 \\angle -53.13^\\circ",
              "Z = 3 + j4 = 5 \\angle 53.13^\\circ", "|3 + j4| = 5", "|Z| = \\sqrt{3^2 + 4^2} = 5", "V = 10\\angle{30^{\\circ}} = 8.66 + j5"):
        assert verdicts(t) == ["right"], t
    assert verdicts("I = \\frac{10}{3 + j4} = 1.2 + j1.6") == verdicts("|3 + j4| = 7") == ["wrong step"]


def _deriv_verdicts(case):
    from grader.selfcheck import check_statements, statements
    st = []
    for k, t in enumerate(case if isinstance(case, (list, tuple)) else [case]):
        st += statements(t, line=k, piece=k)
    return [c.verdict for c in check_statements(st)]


def test_calculus_worked_out_in_letters():
    """a derivative or an integral worked out in letters is compared as an expression ('\\frac{d}{dx} x^3 = 3x^2'; an
    indefinite integral up to its constant, '-\\cos^2 x / 2' as right as '\\sin^2 x / 2'); an integral's evaluation
    '\\left[ \\frac{x^3}{3} \\right]_0^2'; dy/dx of a letter nobody defined is not taken for 0 (a false 'wrong step')"""
    for case in ("\\frac{d}{dx} x^3 = 3x^2", "\\frac{d}{dx}(x \\ln x) = \\ln x + 1", "\\int x dx = \\frac{x^2}{2} + C",
                 "\\int \\sin x \\cos x dx = -\\frac{\\cos^2 x}{2} + C", "\\int \\frac{1}{x} dx = \\ln|x| + C", "\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}",
                 "\\int_{0}^{\\pi} \\sin x \\, dx = \\left[ -\\cos x \\right]_0^{\\pi} = 2"):
        assert set(_deriv_verdicts(case)) == {"right"}, case
    for case in ("\\frac{d}{dx} x^3 = 3x", "\\int x dx = x^2 + C", "\\int_{0}^{2} x^{2} \\, dx = \\left[ \\frac{x^3}{3} \\right]_0^2 = 3"):
        assert "wrong step" in _deriv_verdicts(case), case
    assert _deriv_verdicts(["x = 2", "\\frac{dy}{dx} = 3x^2 = 12"]) == ["right"]          # dy/dx: no verdict on it
    assert _deriv_verdicts(["\\int_0^a x dx = 2a", "a = 4"]) == ["right"]                  # an equation, not an identity


def test_functions_defined_and_their_derivatives():
    """'f(x) = x^2 + 1', then 'f(2) = 5'; f'(x), y' and dy/dx under 'y = x^3' checked against the derivative itself;
    'f(x) = 7' of a known f is an equation; a prime is no longer dropped (V_A = V_A' + V_A'' was V_A = 2 V_A)"""
    assert _deriv_verdicts(["f(x) = x^2 + 1", "f(2) = 5"]) == ["right"]
    assert _deriv_verdicts(["f(x) = x^2 + 1", "f(2) = 4"]) == ["wrong step"]
    assert _deriv_verdicts(["f(x) = x^2", "f'(x) = 2x", "f'(3) = 6"]) == ["right", "right"]
    assert _deriv_verdicts(["f(x) = x^2", "f'(x) = x"]) == ["wrong step"]
    assert _deriv_verdicts(["y = x^3", "y' = 3x^2"]) == _deriv_verdicts(["y = x^3", "\\frac{dy}{dx} = 3x^2"]) == ["right"]
    assert _deriv_verdicts(["y = x^3", "\\frac{dy}{dx} = x^2"]) == ["wrong step"]
    assert _deriv_verdicts(["f(x) = 2x + 1", "f(x) = 7", "x = 3"]) == ["right"]
    assert _deriv_verdicts(["f(x) = x^3 - 3x", "f'(x) = 3x^2 - 3 = 0", "x = 1"]) == ["right", "right"]
    assert _deriv_verdicts(["V_A' = 3", "V_A'' = 2", "V_A = V_A' + V_A'' = 5"]) == ["right"]
    assert _deriv_verdicts(["V_A' = 3", "V_A'' = 2", "V_A = V_A' + V_A'' = 6"]) == ["wrong step"]


def test_units_as_fractions_and_values_written_bare():
    """'v = 10 \\frac{m}{s}', 'a = 3 m/s^2', '\\rho = 2 \\frac{g}{cm^3}' (= 2000 kg/m^3), '72 \\frac{km}{h} = 20 \\frac{m}{s}';
    data written bare are in SI units or the student's own: T = 2\\pi/\\omega with omega = 1000 is 6.28 ms (a false 'wrong'
    before); with 'C = 1 \\mu F' beside a bare omega the answer in SI units is right, the one past the prefix a suggestion"""
    from grader.selfcheck import _si
    assert abs(_si("km/h") - 1 / 3.6) < 1e-12 and abs(_si("\\frac{g}{cm^3}") - 1000) < 1e-9
    assert _deriv_verdicts(["v = \\frac{d}{t} \\vert d = 100 m, t = 10 s", "v = 10 \\frac{m}{s}"]) == ["right"]
    assert _deriv_verdicts(["F = m a \\vert m = 2 kg, a = 3 \\frac{m}{s^2}", "F = 5 N"]) == ["wrong"]
    assert _deriv_verdicts(["\\rho = \\frac{m}{V} \\vert m = 2 g, V = 1 cm^3", "\\rho = 2000 \\frac{kg}{m^3}"]) == ["right"]
    assert _deriv_verdicts(["\\rho = \\frac{m}{V} \\vert m = 2 g, V = 1 cm^3", "\\rho = 2 \\frac{kg}{m^3}"]) == ["wrong"]
    assert _deriv_verdicts("v = 72 \\frac{km}{h} = 20 \\frac{m}{s}") == ["right"]
    assert _deriv_verdicts("72 \\frac{km}{h} = 25 \\frac{m}{s}") == ["wrong step"]
    assert _deriv_verdicts(["T = \\frac{2 \\pi}{\\omega} \\vert \\omega = 1000", "T = 6.28 ms"]) == ["right"]
    assert _deriv_verdicts(["T = \\frac{2 \\pi}{\\omega} \\vert \\omega = 1000", "T = 6.28 s"]) == ["wrong"]
    xc = ["X_C = \\frac{1}{\\omega C} \\vert \\omega = 1000, C = 1 \\mu F"]
    assert _deriv_verdicts(xc + ["X_C = 1000 \\Omega"]) == _deriv_verdicts(xc + ["X_C = 1 k\\Omega"]) == ["right"]
    from grader.selfcheck import check_statements, statements
    st = statements(xc[0], line=0, piece=0) + statements("X_C = 0.001", line=1, piece=1)
    (c,) = check_statements(st)
    assert c.verdict == "right" and not c.sure                    # 1/(1000 * 1): the micro left out?
    assert _deriv_verdicts(["V = I R \\vert I = 2, R = 5", "V = 10 mV"]) == ["right"]      # his own units
    assert _deriv_verdicts("I = \\frac{V}{R} = \\frac{5}{1k} = 5 mA") == ["right"]          # a bare prefix in a substitution
    assert _deriv_verdicts("t = 2 min = 120 s") == ["right"] and _deriv_verdicts("t = 2 min = 100 s") == ["wrong step"]
    assert _deriv_verdicts(["A = 2 h", "h = 3", "A = 6"]) == ["right"]                       # h a letter there
    assert _deriv_verdicts("G = 20 \\log(100) = 40 dB") == ["right"]


def test_roots_plus_minus_and_words_between_statements():
    """'x = \\pm 3', 'x_{1,2} = \\frac{5 \\pm 1}{2}', the quadratic formula with its roots named either way round; 'x = 1
    \\text{ or } x = -2', a Hebrew word between two statements ('2x = 4 \\text{ ולכן } x = 2' was one false step),
    '3x = 12 \\rightarrow x = 4', unicode signs ('2 · 3 = 6' was read 2.3), a computation a line at a time"""
    assert _deriv_verdicts("x^2 = 9 \\Rightarrow x = \\pm 3") == ["right"]
    assert _deriv_verdicts(["x^2 - 5x + 6 = 0", "x_{1,2} = \\frac{5 \\pm 1}{2}"]) == ["right", "right"]
    assert _deriv_verdicts(["x^2 - 5x + 6 = 0", "x_{1,2} = \\frac{5 \\pm 2}{2}"]) == ["wrong", "wrong"]
    q = ["x_{1,2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\vert a = 1, b = -5, c = 6"]
    assert _deriv_verdicts(q + ["x_1 = 3", "x_2 = 2"]) == _deriv_verdicts(q + ["x_1 = 2", "x_2 = 3"]) == ["right", "right"]
    assert _deriv_verdicts(q + ["x_1 = 4", "x_2 = 2"]) == ["wrong", "right"]
    assert _deriv_verdicts(["(x - 1)(x + 2) = 0", "x = 1 \\text{ or } x = -2"]) == ["right", "right"]     # (each root)
    assert sorted(_deriv_verdicts(["(x - 1)(x + 2) = 0", "x = 1 \\text{ or } x = 2"])) == ["right", "wrong"]
    assert _deriv_verdicts("2x = 4 \\text{ ולכן } x = 2") == ["right"]
    assert _deriv_verdicts("3x = 12 \\rightarrow x = 4") == ["right"]
    for t in ("2 · 3 = 6", "6 ÷ 2 = 3", "4 × 5 = 20", "5 − 3 = 2", "\\sin^2(30^\\circ) + \\cos^2(30^\\circ) = 1"):
        assert _deriv_verdicts(t) == ["right"], t
    assert _deriv_verdicts(["I = \\frac{V}{R}", "I = \\frac{12}{4}", "I = 4 A"])[0] == "wrong step"
    assert _deriv_verdicts("R = 100 \\pm 5") == []                 # a tolerance: no roots


def test_rubric_answers_with_compound_units():
    """the rubric's numeric answers with units as fractions or powers ('10 \\frac{m}{s}', '9.8 m/s^2', '2000 \\frac{kg}{m^3}'
    against '2 g/cm^3', '40 \\frac{mA}{V}'): parse errors before - and a reference written 'm/s^2' was refused"""
    from grader.numeric import compare_numeric
    for st, ref in (("10 \\frac{m}{s}", "10 m/s"), ("36 \\frac{km}{h}", "10 m/s"), ("9.8 \\frac{m}{s^2}", "9.8 m/s^2"),
                    ("2000 \\frac{kg}{m^3}", "2 g/cm^3"), ("40 \\frac{mA}{V}", "0.04 A/V"), ("50 cm^2", "0.005 m^2"),
                    ("10 \\left[ \\frac{m}{s} \\right]", "10 m/s"), ("5 N \\cdot m", "5 N*m"), ("314 \\frac{rad}{s}", "314 rad/s")):
        assert compare_numeric(st, ref).status == "correct", (st, ref)
    assert compare_numeric("10 \\frac{m}{s}", "10 km/h").status == "wrong"
    assert compare_numeric("\\frac{10}{3} V", "3.333 V").status == compare_numeric("10\\sqrt{2} V", "14.14 V").status == "correct"


def test_laplace_transforms_in_derivations():
    """'\\mathcal{L}\\{e^{-2t}\\} = \\frac{1}{s+2}' and the inverse ('\\mathcal{L}^{-1}\\{\\frac{1}{s(s+1)}\\} = 1 - e^{-t}', u(t) the
    step) worked out by sympy and compared as expressions - nothing was checked before"""
    for case in ("\\mathcal{L}\\{e^{-2t}\\} = \\frac{1}{s+2}", "\\mathcal{L}^{-1}\\left\\{\\frac{1}{s(s+1)}\\right\\} = (1 - e^{-t}) u(t)",
                 "X(s) = \\mathcal{L}\\{t e^{-t}\\} = \\frac{1}{(s+1)^2}", "\\mathcal{L}\\{\\sin(\\omega t)\\} = \\frac{\\omega}{s^2 + \\omega^2}",
                 "\\mathcal{L}^{-1}\\{\\frac{s+3}{(s+1)(s+2)}\\} = 2e^{-t} - e^{-2t}"):
        assert _deriv_verdicts(case) == ["right"], case
    assert _deriv_verdicts("\\mathcal{L}\\{e^{-2t}\\} = \\frac{1}{s-2}") == ["wrong step"]
    assert _deriv_verdicts("\\mathcal{L}^{-1}\\left\\{\\frac{1}{s(s+1)}\\right\\} = 1 + e^{-t}") == ["wrong step"]


def test_identities_and_partial_fractions():
    """an identity in letters is right when it holds ('(a+b)^2 = a^2 + 2ab + b^2', 'sin^2 x + cos^2 x = 1'); a partial
    fraction expansion that does not hold is wrong, and its coefficients are checked once found (A = 2, B = -1); other
    relations in letters stay equations to solve ('x^2 + 1 = 2x', then x = 1)"""
    for case in ("(a+b)^2 = a^2 + 2ab + b^2", "\\sin^2 x + \\cos^2 x = 1", "\\frac{s+3}{(s+1)(s+2)} = \\frac{2}{s+1} - \\frac{1}{s+2}"):
        assert _deriv_verdicts(case) == ["right"], case
    assert _deriv_verdicts("(a+b)^2 = a^2 + b^2") == []             # may be an equation: no verdict
    assert _deriv_verdicts("\\frac{s+3}{(s+1)(s+2)} = \\frac{2}{s+1} + \\frac{1}{s+2}") == ["wrong step"]
    setup = ["\\frac{s+3}{(s+1)(s+2)} = \\frac{A}{s+1} + \\frac{B}{s+2}", "A = 2"]
    assert _deriv_verdicts(setup + ["B = -1"]) == ["right"] and _deriv_verdicts(setup + ["B = 1"]) == ["wrong"]
    assert _deriv_verdicts(["x^2 + 1 = 2x", "x = 1"]) == ["right"]


def test_second_and_partial_derivatives():
    """'\\frac{d^2}{dx^2} x^3 = 6x' (read as d^2 times x^3 over dx^2 before), y'' and d^2y/dx^2 under 'y = x^3' (the order
    kept: taken for the first derivative it was a false 'wrong step'); a partial derivative takes the other letters for
    constants; a lone letter's stays unworked"""
    assert _deriv_verdicts("\\frac{d^2}{dx^2} x^3 = 6x") == ["right"]
    assert _deriv_verdicts("\\frac{d^2}{dx^2} x^3 = 3x") == ["wrong step"]
    assert _deriv_verdicts(["y = x^3", "\\frac{d^2 y}{dx^2} = 6x"]) == _deriv_verdicts(["y = x^3", "y'' = 6x"]) == ["right"]
    assert _deriv_verdicts(["y = x^3", "\\frac{d^2 y}{dx^2} = 3x^2"]) == ["wrong step"]
    assert _deriv_verdicts("\\frac{\\partial}{\\partial x}(x^2 y) = 2xy") == ["right"]
    assert _deriv_verdicts("\\frac{\\partial}{\\partial x}(x^2 y) = 2x") == ["wrong step"]
    assert _deriv_verdicts("\\frac{\\partial f}{\\partial x} = 2xy") == []
    assert _deriv_verdicts(["x = 2", "\\frac{d^2 y}{dx^2} = 6x = 12"]) == ["right"]


def test_right_number_with_the_wrong_unit():
    """the unit the formula gives its letter, from the data's units: 'R = 5 kV' for 10 V / 2 mA has the right number
    and the wrong unit (it passed as right) - a suggestion for a human; '5 kΩ', 'V/A' are right; with data written
    bare nothing is said about units"""
    assert _deriv_verdicts(["R = \\frac{V}{I} \\vert V = 10 V, I = 2 mA", "R = 5 k\\Omega"]) == ["right"]
    assert _deriv_verdicts(["R = \\frac{V}{I} \\vert V = 10 V, I = 2 mA", "R = 5 kV"]) == ["right value, wrong unit"]
    assert _deriv_verdicts(["P = V I \\vert V = 10 V, I = 2 A", "P = 20 J"]) == ["right value, wrong unit"]
    assert _deriv_verdicts(["P = V I \\vert V = 10 V, I = 2 A", "P = 30 J"]) == ["wrong"]
    assert _deriv_verdicts(["F = m a \\vert m = 2 kg, a = 3 \\frac{m}{s^2}", "F = 6 N"]) == ["right"]
    assert _deriv_verdicts(["\\omega = 2 \\pi f \\vert f = 50 Hz", "\\omega = 314 \\frac{rad}{s}"]) == ["right"]
    assert _deriv_verdicts(["V = I R \\vert I = 2, R = 5", "V = 10 J"]) == ["right"]       # bare data: no unit known


def test_substitutions_carrying_more_units():
    """'\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 ms' (the last side was left as written: a false 'wrong step', 0.001 is
    not 1), '= 22 m' for 22 ms when the chain says so, '2 kg \\cdot 3 \\frac{m}{s^2} = 6 N', '72 km/h'; a one-letter unit that
    is the problem's own letter is no unit ('F = \\mu N = 0.3 N = 6 N': N the normal force - a false 'wrong step' before)"""
    assert _deriv_verdicts("\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 ms") == ["right"]
    assert _deriv_verdicts("\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 2 ms") == ["wrong step"]
    assert _deriv_verdicts("\\tau = R C = 2.2 k\\Omega \\cdot 10 \\mu F = 22 m") == ["right"]
    assert _deriv_verdicts("F = m a = 2 kg \\cdot 3 \\frac{m}{s^2} = 6 N") == ["right"]
    assert _deriv_verdicts("F = m a = 2 kg \\cdot 3 \\frac{m}{s^2} = 7 N") == ["wrong step"]
    assert _deriv_verdicts("F = \\mu N = 0.3 N = 6 N") == []
    assert _deriv_verdicts("Q = C V = 10 \\mu F \\cdot 5 V = 50 \\mu C") == ["right"]


def test_time_functions_against_their_laplace_transforms():
    """V(s) then v(t) (I_L(s) then i_L(t), H(s) then h(t)): the inverse transform against the time function, for t > 0;
    the t of 'i(t)', which the parser makes complex, is a real letter like the others (i(0) was not worked out)"""
    assert _deriv_verdicts(["V(s) = \\frac{10}{s(s+5)}", "v(t) = 2 - 2e^{-5t}"]) == ["right"]
    assert _deriv_verdicts(["V(s) = \\frac{10}{s(s+5)}", "v(t) = 2 + 2e^{-5t}"]) == ["wrong step"]
    assert _deriv_verdicts(["I(s) = \\frac{2}{s+3}", "i(t) = 2e^{-3t} u(t)"]) == ["right"]
    assert _deriv_verdicts(["I_L(s) = \\frac{2}{s+3}", "i_L(t) = 3e^{-2t}"]) == ["wrong step"]
    assert _deriv_verdicts(["X(s) = \\frac{s}{s^2 + 4}", "x(t) = \\cos(2t)"]) == ["right"]
    assert _deriv_verdicts(["i(t) = 2e^{-3t}", "i(0) = 2"]) == ["right"]


def test_systems_of_equations_solved_together():
    """mesh / node equations written one per line (or in a brace): the values written after them are checked against
    the system's solution - before, only against the last equation (a wrong I_1 consistent with it passed); a system
    with no single solution falls back to the last equation"""
    eqs = ["10 = 5I_1 - 2I_2", "0 = -2I_1 + 6I_2"]
    assert _deriv_verdicts(eqs + ["I_1 = 2.31 A", "I_2 = 0.77 A"]) == ["right", "right"]
    assert _deriv_verdicts(eqs + ["I_1 = 3 A", "I_2 = 1 A"]) == ["wrong", "wrong"]
    assert _deriv_verdicts(["\\frac{V_A - 10}{2} + \\frac{V_A - V_B}{4} = 0", "\\frac{V_B - V_A}{4} + \\frac{V_B}{4} = 0", "V_A = 8", "V_B = 4"]) == ["right", "right"]
    assert _deriv_verdicts(["\\begin{cases} 2x + y = 5 \\\\ x - y = 1 \\end{cases}", "x = 2", "y = 2"]) == ["right", "wrong"]
    assert _deriv_verdicts(["x + y + z = 6", "x - y = 0", "y + 2z = 7", "x = 1.67", "y = 1.67", "z = 2.67"]) == ["right"] * 3
    assert _deriv_verdicts(["10 = R I_1 - 2 I_2", "0 = -2I_1 + 6I_2", "I_1 = 2.31", "I_2 = 0.77"]) == ["right"]   # R unknown
    # as the formula reader writes a brace: an array whose rows it wraps in braces (they were lost)
    assert _deriv_verdicts(["\\left\\{ \\begin{array} { l } { 1 0 = 5 I _ { 1 } - 2 I _ { 2 } } \\\\ { 0 = - 2 I _ { 1 } + 6 I _ { 2 } } "
                            "\\end{array} \\right.", "I _ { 1 } = 2 . 3 1", "I _ { 2 } = 0 . 7 7"]) == ["right", "right"]


def test_determinants_written_with_matrices():
    """'\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix} = -2': the rows of a matrix kept through the formula reader's
    evening-out (its '\\\\' was taken for a space: a false 'wrong step', 4 is not -2)"""
    assert _deriv_verdicts("\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix} = -2") == ["right"]
    assert _deriv_verdicts("\\det\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix} = -2") == ["right"]
    assert _deriv_verdicts("\\begin{vmatrix} 1 & 2 \\\\ 3 & 4 \\end{vmatrix} = 2") == ["wrong step"]
    assert _deriv_verdicts("\\det \\begin{bmatrix} 2 & 0 & 1 \\\\ 1 & 3 & 2 \\\\ 1 & 1 & 1 \\end{bmatrix} = 0") == ["right"]
    # the formula reader writes a determinant as an array between bars - no system of equations (a false 'wrong step')
    assert _deriv_verdicts("\\left| \\begin{array} { l l } { 1 } & { 2 } \\\\ { 3 } & { 4 } \\end{array} \\right| = - 2") == ["right"]
    assert _deriv_verdicts("\\left| \\begin{array}{cc} 1 & 2 \\\\ 3 & 4 \\end{array} \\right| = 2") == ["wrong step"]


def test_loop_equations_in_the_drawn_currents():
    """'-12 + 2k I_1 + 4k I_2 = 0': an equation in the drawn currents alone that is no node's KCL - it got no verdict at
    all; now checked against the circuit's solution (right, sure); not holding it is a suggestion only - the student's
    I_1, I_2 may be mesh currents of his own named like the drawing's arrows"""
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values
    net = with_values(Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                               C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                               C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")],
                              [Current("I1", "R1", "n1", "a"), Current("I2", "R2", "a", "0"), Current("I3", "R3", "n3", "a")],
                              ground="0", node_labels={"a": "A", "n1": "B", "n3": "D"}),
                      {"E1": "12", "E2": "6", "R1": "2k", "R2": "4k", "R3": "4k"})

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), net)]
    for t in ("-12 + 2k I_1 + 4k I_2 = 0", "2000 I_1 + 4000 I_2 = 12", "-12 + 2k I_1 - 4k I_3 + 6 = 0"):
        assert verdicts(t) == [("right", True)], t
    assert verdicts("-12 + 2k I_1 + 4k (I_1 - I_2) = 0") == [("wrong", False)]
    assert verdicts("I_1 + I_3 = I_2") == [("right", True)] and verdicts("I_1 + I_2 = I_3") == [("wrong", True)]


def test_phasor_node_equations_with_prefixed_impedances():
    """'\\frac{V_o}{-j1k}', 'Z_C = -j1k': a prefixed value right after j was taken for letters (no verdict)"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    lp = Netlist([C("V1", "voltage_source", ["a", "0"], value="10cos(1000t)", plus="a", ac=True),
                  C("R1", "resistor", ["a", "m"], value="1k"), C("C1", "capacitor", ["m", "0"], value="1u")], ground="0", node_labels={"m": "o"})

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), lp)]
    assert verdicts("\\frac{V_o - 10}{1k} + \\frac{V_o}{-j1k} = 0") == verdicts("Z_C = -j1k") == verdicts("Z = 1k - j1k") == [("right", True)]
    assert verdicts("\\frac{V_o - 10}{1k} + \\frac{V_o}{j1k} = 0") == [("wrong", True)]


def test_node_names_that_read_like_other_names():
    """a node labelled 1 beside an unlabelled source whose own id is V1: V_1 is the node's voltage (the id is no name on
    the drawing - the source's value was taken: a false 'wrong'); two nodes labelled A and a (names compared without
    case): an answer about either is never a sure 'wrong'"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["x", "0"], plus="x", value="9"), C("R1", "resistor", ["x", "q"], value="1k"),
                   C("R2", "resistor", ["q", "c"], value="2k"), C("R3", "resistor", ["c", "0"], value="3k")], ground="0",
                  node_labels={"x": "b", "c": "1"})

    def verdicts(t, n=net):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), n)]
    assert verdicts("V_1 = 4.5") == [("right", True)] and verdicts("V_b = 9") == [("right", True)]
    two = Netlist([C("V1", "voltage_source", ["t", "0"], plus="t", value="12"), C("R1", "resistor", ["t", "o"], value="1k"),
                   C("R2", "resistor", ["o", "0"], value="1k"), C("R3", "resistor", ["o", "e"], value="1k"),
                   C("R4", "resistor", ["e", "0"], value="1k")], ground="0", node_labels={"t": "A", "e": "a"})
    assert all(not sure for v, sure in verdicts("V_A = 12", two) + verdicts("V_a = 2.4", two) if v == "wrong")


def test_numbered_exercises_and_functions_on_a_page():
    """on a page (check_pages, each line its own problem grouping): '3) \\tau = R C = ...' after exercise 2 is a new
    exercise (it was put under P = V^2/R and its step never checked), and "f'(x) = 2x" stays with 'f(x) = x^2 + 1' (a
    problem of its own, f unknown there: no verdict)"""
    from grader.selfcheck import check_pages

    class Seg:
        def __init__(self, text):
            self.kind, self.box, self.nbest, self.p_kind = "math", (100, 0, 110, 10), [(text, -0.01)], 0.99
    lines = ["1) P = \\frac{V^2}{R} \\vert V = 10 V, R = 2 k\\Omega", "P = 50 mW", "2) \\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 ms",
             "3) f(x) = x^2 + 1", "f(2) = 5", "f'(x) = 2x"]
    checks = check_pages([{"lines": [[Seg(t)] for t in lines]}])
    assert sorted((c.problem, c.verdict) for c in checks) == [(1, "right"), (2, "right"), (3, "right"), (3, "right")]


def test_circuit_answers_with_a_unit_of_another_kind_and_guessed_ports():
    """'V_A = 7.5 A': the value the circuit gives, with a current's unit - 'right value, wrong unit' (never sure right;
    inside a chain too), a unit inside a numeric expression ('\\frac{12 - 7.5}{2 k\\Omega}') is no such case; nodes
    labelled A and B with no terminals drawn: R_eq taken between them is a guess - a mismatch is no sure 'wrong'"""
    from grader.circuit import Component as C, Current, Netlist
    from grader.circuit_check import check_against, circuit_statements, with_values
    net = with_values(Netlist([C("E1", "voltage_source", ["n1", "0"], plus="n1", label="E1"), C("R1", "resistor", ["n1", "a"], label="R1"),
                               C("R2", "resistor", ["a", "0"], label="R2"), C("R3", "resistor", ["n3", "a"], label="R3"),
                               C("E2", "voltage_source", ["n3", "0"], plus="n3", label="E2")],
                              [Current("I1", "R1", "n1", "a"), Current("I2", "R2", "a", "0"), Current("I3", "R3", "n3", "a")],
                              ground="0", node_labels={"a": "A", "n1": "B", "n3": "D"}),
                      {"E1": "12", "E2": "6", "R1": "2k", "R2": "4k", "R3": "4k"})

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), net)]
    unit = ("right value, wrong unit", False)
    assert verdicts("V_A = 7.5 V") == [("right", True)] and verdicts("V_A = 7.5") == [("right", True)]
    assert verdicts("V_A = 7.5 A") == [unit] and verdicts("V_A = 7.5 [A]") == [unit] and verdicts("I_1 = 2.25 mV") == [unit]
    assert verdicts("P_{R1} = 10.125 mA") == [unit] and verdicts("R_{eq} = 1 kV") == [unit]
    assert verdicts("V_A = \\frac{E_1/R_1 + E_2/R_3}{1/R_1+1/R_2+1/R_3} = 7.5 A") == [unit]
    assert verdicts("I_1 = \\frac{12 - 7.5}{2 k\\Omega}") == [("right", True)]
    assert verdicts("V_A = 8 A") == [("wrong", True)]
    assert verdicts("R_{eq} = 1 k\\Omega") == [("right", True)] and verdicts("R_{eq} = 2 k\\Omega") == [("wrong", False)]
    from grader.circuit import Component as C2
    rl = Netlist([C2("V1", "voltage_source", ["n0", "0"], plus="n0", value="12"), C2("S1", "switch", ["n0", "n1"], state="closed"),
                  C2("R1", "resistor", ["n1", "a"], value="4"), C2("L1", "inductor", ["a", "0"], value="2"),
                  C2("R2", "resistor", ["a", "0"], value="12")], ground="0", node_labels={"a": "A"})
    assert [(c.verdict, c.sure) for c in check_against(circuit_statements("i_L(0^+) = 3 V"), rl)] == [unit]
    y = Netlist([C2("Va", "voltage_source", ["a", "0"], plus="a", value="120∠0°"), C2("Vb", "voltage_source", ["b", "0"], plus="b", value="120∠-120°"),
                 C2("Vc", "voltage_source", ["c", "0"], plus="c", value="120∠120°"), C2("Za", "resistor", ["a", "m"], value="10+j5Ω"),
                 C2("Zb", "resistor", ["b", "m"], value="10+j5Ω"), C2("Zc", "resistor", ["c", "m"], value="10+j5Ω")],
                ground="0", node_labels={"a": "a", "b": "b", "c": "c", "m": "N"})
    for t, want in (("I_{Za} = 10.73\\angle -26.57^\\circ A", ("right", True)), ("I_{Za} = 10.73\\angle -26.57^\\circ V", unit),
                    ("I_{Za} = (9.6 - j4.8) V", unit)):                                   # phasors with their units
        assert [(c.verdict, c.sure) for c in check_against(circuit_statements(t), y)] == [want], t


def test_fourier_series_indices_taken_as_whole_numbers():
    """a Fourier series' index n, k: '\\cos(n\\pi) = (-1)^n', '\\int_0^\\pi \\sin(nx) dx = \\frac{1-(-1)^n}{n}' hold at whole n
    only - compared there too (no verdict before); one that holds at some whole n only ('= \\frac{2}{n}': odd n) gets none"""
    for case in ("\\int_0^{\\pi} \\sin(n x) dx = \\frac{1-\\cos(n\\pi)}{n} = \\frac{1-(-1)^n}{n}",
                 "a_n = \\frac{2}{\\pi}\\int_0^{\\pi} x \\cos(n x) dx = \\frac{2}{\\pi} \\cdot \\frac{(-1)^n - 1}{n^2}",
                 "b_n = \\frac{1}{\\pi}\\int_{-\\pi}^{\\pi} x \\sin(n x) dx = \\frac{2(-1)^{n+1}}{n}",
                 "\\int_0^{2\\pi} \\sin^2(n x) dx = \\pi", "\\cos(n\\pi) = (-1)^n", "e^{-j k \\pi} = (-1)^k"):
        v = _deriv_verdicts(case)
        assert v and set(v) == {"right"}, (case, v)
    assert _deriv_verdicts("\\int_0^{\\pi} \\sin(n x) dx = \\frac{2}{n}") == []
    assert _deriv_verdicts("\\cos(\\frac{n \\pi}{2}) = 0") == []


def test_derivatives_with_constants_named_by_habit():
    """d/dt of an expression with other letters was never worked out (a letter may be a function of t: v, i, \\Phi):
    with them taken for constants, an agreeing side is right - sure for the usual names of constants (A, \\omega, \\tau,
    k, n), not for others (V_m: v); a disagreeing one is a wrong step only for those names and an expression in the
    variable (not d/dt (J \\omega^2 / 2), where \\omega may change); one with the function's derivative says nothing"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("\\frac{d}{dt}(A\\cos(\\omega t)) = -A\\omega\\sin(\\omega t)") == [("right", True)]
    assert verdicts("\\frac{d}{dt}(A\\cos(\\omega t)) = A\\omega\\sin(\\omega t)") == [("wrong step", True)]
    assert verdicts("\\frac{d}{dx} x^n = n x^{n-1}") == [("right", True)]
    assert verdicts("\\frac{d}{dt}(V_m \\sin(\\omega t + \\phi)) = V_m \\omega \\cos(\\omega t + \\phi)") == [("right", False)]
    assert verdicts("\\frac{d}{dt}(V_m \\sin(\\omega t + \\phi)) = -V_m \\omega \\cos(\\omega t + \\phi)") == []
    for t in ("\\frac{d}{dt}(C v) = C \\frac{dv}{dt}", "\\frac{d}{dt}(\\frac{1}{2} J \\omega^2) = J \\omega \\alpha",
              "\\frac{d}{dx}(x y) = y + x \\frac{dy}{dx}", "\\frac{dy}{dx} = 3x^2"):
        assert verdicts(t) == [], t


def test_one_character_subscripts():
    """a subscript without braces is one character: '\\omega_d t' is omega_d times t (it was read as one name
    omega_{dt}: a correct derivative judged a wrong step), 'r_e' and '\\omega_e t' are read at all (e: Euler's number)"""
    from grader.selfcheck import _parse
    assert str(_parse("\\omega_d t")) == "omega_d*t" and _parse("r_e") is not None and _parse("\\omega_e t") is not None
    assert _deriv_verdicts("\\frac{d}{dt}(e^{-\\alpha t}\\cos(\\omega_d t)) = -\\alpha e^{-\\alpha t}\\cos(\\omega_d t)"
                           " - \\omega_d e^{-\\alpha t}\\sin(\\omega_d t)") == ["right"]
    assert _deriv_verdicts(["r_e = \\frac{V_T}{I_E} \\vert V_T = 25 mV, I_E = 1 mA", "r_e = 25 \\Omega"]) == ["right"]


def test_phasors_and_complex_values_in_derivations():
    """complex values are data now ('Z = 3 + j4' was dropped): steps and answers with them are checked, in polar form
    too; phasor mesh equations solved together; complex roots of a characteristic equation; '|Z| = 5' against the size
    of what the formula gives; with complex data a real answer may use their sizes ('P = I^2 R = 20' for I = 2∠30°);
    steps in SI units ('L = 10 mH': j\\omega L = j10 - a false 'wrong step' before); a capital J is a letter (a
    current density: 'J = 6' was judged against sqrt(-1)); Re{Z}, \\angle Z, I^* are left alone (read as products)"""
    ok = ["right"]
    assert _deriv_verdicts(["Z = 3 + j4", "I = \\frac{10}{Z} = 2 \\angle -53.13^\\circ"]) == ok
    assert _deriv_verdicts(["Z = 3 + j4", "I = \\frac{10}{Z} = 2 \\angle 53.13^\\circ"]) == ["wrong step"]
    mesh = ["(2+j3) I_1 - j3 I_2 = 10", "-j3 I_1 + (4+j1) I_2 = 0"]
    assert _deriv_verdicts(mesh + ["I_1 = 1.786 - j1.071", "I_2 = 1.515 \\angle 45^\\circ"]) == ["right", "right"]
    assert _deriv_verdicts(mesh + ["I_1 = 2.29 \\angle -30.96^\\circ", "I_2 = 1.515 \\angle 45^\\circ"]) == ["wrong", "right"]
    assert _deriv_verdicts(["s^2 + 2s + 5 = 0", "s_1 = -1 + j2", "s_2 = -1 - j2"]) == ["right", "right"]
    assert _deriv_verdicts(["x^2 + 4 = 0", "x = 2j"]) == ok
    rl = "Z = R + j\\omega L \\vert R = 3, \\omega = 1000, L = 4 mH"
    assert _deriv_verdicts([rl, "Z = 3 + j4"]) == ok and _deriv_verdicts([rl, "|Z| = 5"]) == ok
    assert _deriv_verdicts([rl, "|Z| = 6"]) == ["wrong"]
    assert _deriv_verdicts(["I = 2 \\angle 30^\\circ", "R = 5", "P = I^2 R = 20"]) == ok
    assert _deriv_verdicts(["\\omega = 1000", "L = 10 mH", "Z_L = j\\omega L = j10"]) == ok
    assert _deriv_verdicts(["J = \\sigma E \\vert \\sigma = 2, E = 3", "J = 6"]) == ok and _deriv_verdicts(["j = 3"]) == []
    for case in (["Z = 3 + j4", "\\Re\\{Z\\} = 3"], ["Z = 3 + j4", "\\angle Z = 53.13^\\circ"],
                 ["V = 10", "I = 2 \\angle -30^\\circ", "S = V I^* = 20 \\angle 30^\\circ"]):
        assert "wrong" not in " ".join(_deriv_verdicts(case)), case


def test_time_functions_against_their_phasors():
    """'I = 2\\angle -53.13^\\circ', then 'i(t) = 2\\cos(1000t - 53.13^\\circ)': the time function against the phasor by
    the same letter (a peak or an RMS phasor, its cosine for reference; a sine written for it is shifted) - right
    sure; a mismatch is wrong but not sure (the course may use another convention); a value without an angle is no
    phasor (V = 10, then v(t): nothing)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    i = "I = 2 \\angle -53.13^\\circ"
    assert verdicts([i, "i(t) = 2\\cos(1000t - 53.13^\\circ)"]) == [("right", True)]
    assert verdicts([i, "i(t) = 2.83\\cos(1000t - 53.13^\\circ)"]) == [("right", True)]          # an RMS phasor
    assert verdicts([i, "i(t) = 2\\sin(1000t + 36.87^\\circ)"]) == [("right", True)]
    assert verdicts(["I = 1.2 - j1.6", "i(t) = 2\\cos(\\omega t - 53.13^\\circ)"]) == [("right", True)]
    assert verdicts([i, "i(t) = 2\\cos(1000t + 53.13^\\circ)"]) == [("wrong", False)]
    assert verdicts(["V = 10", "v(t) = 10\\cos(1000t)"]) == []


def test_frequency_response_from_the_transfer_function():
    """'H(j\\omega) = ...' or 'H(j2) = ...' after 'H(s) = ...': s = j\\omega put into H(s) and compared (H(j\\omega) was read
    as H times j times omega); a number within its decimals (polar too) - a wrong one is wrong"""
    h = "H(s) = \\frac{1}{1 + sRC}"
    assert _deriv_verdicts([h, "H(j\\omega) = \\frac{1}{1 + j\\omega RC}"]) == ["right"]
    assert _deriv_verdicts([h, "H(j\\omega) = \\frac{1}{1 - j\\omega RC}"]) == ["wrong step"]
    h = "H(s) = \\frac{1}{s+1}"
    assert _deriv_verdicts([h, "H(j2) = 0.2 - j0.4"]) == ["right"] and _deriv_verdicts([h, "H(j2) = 0.447 \\angle -63.43^\\circ"]) == ["right"]
    assert _deriv_verdicts([h, "H(j2) = 0.447 \\angle 63.43^\\circ"]) == ["wrong"]
    assert _deriv_verdicts([h, "|H(j\\omega)| = \\frac{1}{\\sqrt{1 + \\omega^2}}"]) == ["right"]          # its size
    assert _deriv_verdicts([h, "|H(j\\omega)| = \\frac{1}{1 + \\omega^2}"]) == ["wrong step"]
    assert _deriv_verdicts([h, "|H(j2)| = 0.447"]) == ["right"] and _deriv_verdicts([h, "|H(j\\omega)|^2 = \\frac{1}{1 + \\omega^2}"]) == []
    assert _deriv_verdicts([h, "20\\log|H(j1)| = -3"]) == ["right"] and _deriv_verdicts([h, "20\\log|H(j1)| = 3"]) == ["wrong"]  # in dB
    assert _deriv_verdicts([h, "20\\log_{10}|H(j\\omega)| = -10\\log(1+\\omega^2)"]) == ["right"]


def test_numbers_written_in_a_base():
    """'(1011)_2 = 11', '1011_{(2)} = 11', '(FF)_{16} = 255', '(101.1)_2 = 5.5': numbers in their base (they were read as
    1011 or a falling factorial: false wrong steps in digital-systems work); a digit outside its base: no verdict"""
    for case in ("(1011)_2 = 11", "(1011)_2 = (11)_{10}", "1011_{(2)} = 11", "(FF)_{16} = 255", "(101.1)_2 = 5.5",
                 "(1011)_{2} = 1 \\cdot 2^3 + 0 \\cdot 2^2 + 1 \\cdot 2 + 1 = 11"):
        v = _deriv_verdicts(case)
        assert v and set(v) == {"right"}, (case, v)
    assert _deriv_verdicts("(1011)_2 = 12") == ["wrong step"] and _deriv_verdicts("(129)_2 = 5") == []
    assert _deriv_verdicts(["x_2 = 3", "R_{10} = 5"]) == []


def test_time_functions_at_a_time_with_its_unit():
    """'v_C(1 ms) = 6.32' under 'v_C(t) = 10(1 - e^{-t/\\tau})', '\\tau = 1 ms': the time in seconds (it was read as v_C
    times 1 m s), and the value a step to check (it was taken for an equation to solve for tau); 'H(2s)' stays H at 2s"""
    from grader.selfcheck import _parse
    v = "v_C(t) = 10(1 - e^{-t/\\tau})"
    assert _deriv_verdicts([v, "\\tau = 1 ms", "v_C(1 ms) = 6.32"]) == ["right"]
    assert _deriv_verdicts([v, "\\tau = 1 ms", "v_C(1 ms) = 3.68"]) == ["wrong step"]
    assert _deriv_verdicts(["i(t) = 2e^{-t/\\tau}", "\\tau = 0.5 s", "i(1 s) = 0.27"]) == ["right"]
    assert _deriv_verdicts(["v_C(t) = 10(1 - e^{-1000t})", "v_C(2 ms) = 8.65 V"]) == ["right"]
    assert "s" in str(_parse("H(2s)"))


def test_units_inside_a_fraction():
    """'I = \\frac{10}{2 k\\Omega} = 5 mA': the unit belongs to the denominator - the closing brace after it was taken
    for a \\mathrm's and the whole fraction scaled (a false wrong step: '10 is not 0.005'); millisiemens ('40 mS') are
    a unit now ('g_m = \\frac{1 mA}{25 mV} = 40 mS', 'r_\\pi = \\frac{100}{40 mS} = 2.5 k\\Omega')"""
    assert _deriv_verdicts("I = \\frac{V}{R} = \\frac{10}{2 k\\Omega} = 5 mA") == ["right"]
    assert _deriv_verdicts("I = \\frac{10}{2 k\\Omega} = 4 mA") == ["wrong step"]
    assert _deriv_verdicts("g_m = \\frac{I_C}{V_T} = \\frac{1 mA}{25 mV} = 40 mS") == ["right"]
    assert _deriv_verdicts("r_\\pi = \\frac{\\beta}{g_m} = \\frac{100}{40 mS} = 2.5 k\\Omega") == ["right"]
    assert _deriv_verdicts("f = \\frac{1}{T} = \\frac{1}{2 ms} = 500 Hz") == ["right"]


def test_a_step_in_units_of_different_kinds():
    """'P = V I = 10 V \\cdot 2 mA = 20 mA': the step's left side comes out in W, its right side is in A - a 'wrong unit'
    note for a human (the number is right; it passed as right); right units, no note"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return sorted(c.verdict for c in check_statements(statements(t)))
    assert verdicts("P = V I = 10 V \\cdot 2 mA = 20 mA") == ["right", "wrong unit"]
    assert verdicts("R = \\frac{10 V}{2 mA} = 5 kV") == ["right", "wrong unit"]
    for t in ("P = V I = 10 V \\cdot 2 mA = 20 mW", "\\tau = R C = 1 k\\Omega \\cdot 1 \\mu F = 1 ms",
              "E = \\frac{1}{2} C V^2 = \\frac{1}{2} \\cdot 10 \\mu F \\cdot (5 V)^2 = 125 \\mu J", "f = \\frac{1}{T} = \\frac{1}{2 ms} = 500 Hz"):
        assert verdicts(t) == ["right"], t


def test_rubric_answers_as_phasors():
    """a rubric's key may be a phasor or an impedance ('2\\angle -53.13^\\circ A', '30 + j40 VA' - it was refused as a
    reference, and every phasor answer a parse error): polar against rectangular, units with prefixes; the size alone
    for a phasor key, its conjugate, the angle wrong - partly right, for a human; a plain key against '10\\angle 0^\\circ'"""
    from grader import Criterion, Question, Reading, grade_question
    from grader.numeric import compare_numeric
    ref = "2\\angle -53.13^\\circ A"
    for st in ("1.2 - j1.6 A", "2 \\angle -53.1^\\circ A", "2000\\angle -53.13^\\circ mA", "I = 2∠-53.13°"):
        assert compare_numeric(st, ref).status == "correct", st
    assert [compare_numeric(st, ref).status for st in ("2 A", "2\\angle 53.13^\\circ A", "2\\angle 30^\\circ A", "3 A",
                                                       "2\\angle -53.13^\\circ V", "0.002\\angle -53.13^\\circ A")] == \
        ["magnitude_only", "conjugate", "angle_wrong", "wrong", "unit_mismatch", "prefix_error"]
    assert compare_numeric("0.05\\angle 53.13^\\circ kVA", "30 + j40 VA").status == "correct"
    assert compare_numeric("10\\angle 0^\\circ V", "10 V").status == "correct"
    q = Question("q1", "", [Criterion("c", "final_numeric", 4, ref)])
    assert grade_question(q, [Reading("1.2 - j1.6 A", 1.0)]).route == "auto"
    assert grade_question(q, [Reading("1.2 - j1.6 A", 1.0)]).most_likely_score == 4
    assert grade_question(q, [Reading("2 A", 1.0)]).route == "review"            # the size only: the lecturer's call


def test_identities_with_the_values_written_above():
    """'I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}' under 'R = 2', 'L = 1': compared with the values put in (a false
    wrong step: the letters were compared as letters); with no values written, sides that differ in letters may use the
    problem's own (printed on the exam) - no verdict, not a wrong step; a dropped variable (y) is still one"""
    v = ["V(s) = \\frac{10}{s}", "R = 2", "L = 1"]
    assert _deriv_verdicts(v + ["I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}"]) == ["right"]
    assert _deriv_verdicts(v + ["I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+3)}"]) == ["wrong step"]
    assert _deriv_verdicts(["V(s) = \\frac{10}{s}", "I(s) = \\frac{V(s)}{R + sL} = \\frac{10}{s(s+2)}"]) == []
    assert _deriv_verdicts("\\frac{d}{dt}(V_m \\sin(\\omega t)) = 100\\cos(100 t)") == []
    assert _deriv_verdicts(["a = 3", "\\frac{d}{dx}(a x^2) = 3x"]) == ["wrong step"]
    assert _deriv_verdicts("\\frac{\\partial}{\\partial x}(x^2 y) = 2x") == ["wrong step"]


def test_values_put_into_an_expression_in_letters():
    """'v_C(t) = 10 + (0 - 10)e^{-t/\\tau} = 10(1 - e^{-1000t})' under '\\tau = 1 ms': two sides in the same letter once the
    data are in, compared on the scales they change over (around t = 1 ms) - right; a wrong rate (e^{-100t}) is not
    passed for it (every decay is 0 at t = 1 s), and gets no verdict either (a rounded value, a convention unknown)"""
    tau = "\\tau = 1 ms"
    assert _deriv_verdicts([tau, "v_C(t) = 10 + (0 - 10)e^{-t/\\tau} = 10(1 - e^{-1000t})"]) == ["right"]
    assert _deriv_verdicts([tau, "v_C(t) = 10 + (0 - 10)e^{-t/\\tau} = 10(1 - e^{-100t})"]) == []
    assert _deriv_verdicts(["\\tau = 2.2 ms", "v(t) = 5e^{-t/\\tau} = 5e^{-455t}"]) == ["right"]
    assert _deriv_verdicts(["\\omega = 1000", "V_m = 10", "v(t) = V_m \\cos(\\omega t) = 10\\cos(1000t)"]) == ["right"]
    assert _deriv_verdicts(["\\omega = 1000", "V_m = 10", "v(t) = V_m \\cos(\\omega t) = 10\\cos(100t)"]) == []


def test_page_rubric_with_phasor_answers():
    """grade_page with a rubric whose answers are phasors: the student's value found by its name and compared as a
    complex number (polar against rectangular); the conjugate goes to a human, and the reason says so"""
    from types import SimpleNamespace as NS
    from grader.page import grade_page
    segs = [NS(kind="math", box=(0, 0, 10, 10), p_kind=0.95, nbest=[("I = 2 \\angle -53.13^\\circ A", -0.01)], page=1),
            NS(kind="math", box=(0, 20, 10, 30), p_kind=0.95, nbest=[("Z = 3 + j4 \\Omega", -0.01)], page=1),
            NS(kind="math", box=(0, 40, 10, 50), p_kind=0.95, nbest=[("V_o = 7.07 \\angle 45^\\circ V", -0.01)], page=1)]
    rubric = {"name": "t", "items": [{"id": "1", "points": 4, "value": "I", "answer": "1.2 - j1.6 A"},
                                     {"id": "2", "points": 3, "value": "Z", "answer": "5 \\angle 53.13^\\circ \\Omega"},
                                     {"id": "3", "points": 3, "value": "V_o", "answer": "7.07 \\angle -45^\\circ V"}]}
    got = {r["id"]: r for r in grade_page(rubric, segs)["items"]}
    assert (got["1"]["route"], got["1"]["score"]) == ("auto", 4.0) and (got["2"]["route"], got["2"]["score"]) == ("auto", 3.0)
    assert got["3"]["route"] == "review" and "conjugate" in got["3"]["reason"]


def test_plain_text_math_and_emf_names():
    """model solutions typed as plain text: '10⁻⁸', '√2', '½', 'R2 ∥ R3', 'KCL: ...', '20 ms' after '1 / 50', 'kWh' (four
    false wrong steps before: 1.72·10⁻⁸ read without its power, √2 as nothing, '50 / 1.5·10⁻⁶' as (50 / 1.5)·10⁻⁶, 20 ms
    against 0.02); and E with a subscript - an EMF, as Israeli courses name sources - was dropped by the parser
    ('E_2 + 3' read as 3; 'E_1 - I R_1' not read at all)"""
    from grader.selfcheck import _parse
    for case in ("R = ρL / A = 1.72·10⁻⁸ · 50 / 1.5·10⁻⁶ = 0.573 Ω", "Vp = √2 · 230 = 325.27 V", "T = 1 / f = 1 / 50 = 20 ms",
                 "R = V² / P = 230² / 2000 = 26.45 Ω", "E = ½ C V² = 0.5 · 100 µF · 144 = 7.2 mJ", "E = P · t = 1.5 · 6 · 30 = 270 kWh",
                 "R2 ∥ R3 = 1 kΩ;  R = 1 + 1 = 2 kΩ;  I = 12 / 2k = 6 mA", "KCL: 3 + 2.5 = 1.2 + I  =>  I = 4.3 A"):
        v = _deriv_verdicts(case)
        assert v and not {"wrong", "wrong step"} & set(v), (case, v)
    assert _deriv_verdicts("KCL: 3 + 2.5 = 1.2 + I  =>  I = 6.45 A") == ["wrong"]
    assert str(_parse("E_2 + 3")) == "E_2 + 3" and _parse("E_1 - I R_1") is not None and str(_parse("\\frac{E_1}{R_1}")) == "E_1/R_1"
    assert _deriv_verdicts(["E_2 = 6", "V_A = E_2 + 3 = 9"]) == ["right"] and _deriv_verdicts(["E_2 = 6", "V_A = E_2 + 3 = 3"]) == ["wrong step"]
    assert _deriv_verdicts(["V_A = E_1 - I_1 R_1 \\vert E_1 = 12, I_1 = 2.25 mA, R_1 = 2 k\\Omega", "V_A = 7.5"]) == ["right"]


def test_derivations_with_the_lecturers_values():
    """the problem's data as the lecturer gives it once (--values) serve the derivation checks too: 'I = \\frac{V}{R}',
    'I = 25.53 mA' with V = 12 V, R = 470 Ω given was 'cannot check'; a value the student writes overrides it; a chain's
    last value in its unit ('= 25.53 mA') compares in SI with them"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines, values=None):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st, values=values)]
    vals = {"V": "12 V", "R": "470 \\Omega"}
    assert verdicts(["I = \\frac{V}{R}", "I = 25.53 mA"]) == [("cannot check", False)]
    assert verdicts(["I = \\frac{V}{R}", "I = 25.53 mA"], vals) == [("right", True)]
    assert verdicts(["I = \\frac{V}{R} = 25.53 mA"], vals) == [("right", True)]
    assert verdicts(["I = \\frac{V}{R} = 30 mA"], vals) == [("wrong step", True)]
    assert verdicts(["V = 10", "I = \\frac{V}{R}", "I = 21.28 mA"], vals) == [("right", True)]     # the student's V
    assert verdicts(["R = 2 k\\Omega", "V = 10", "I = \\frac{V}{R} = 5 mA"]) == [("right", True)]


def test_boxed_underlined_and_labelled_answers():
    """a frame, an underline, a colour around an answer ('\\boxed{V = 10 V}') hid it from the derivation checks and
    made it unparseable for the rubric; a label over the equals sign ('\\overset{KVL}{=}') and a definition's sign
    (':=', '\\triangleq', '\\equiv') split it wrongly - each is now read as its content, an equals sign"""
    from grader.normalize import undecorate
    from grader.numeric import compare_numeric
    from grader.selfcheck import check_statements, statements
    from grader.symbolic import compare_formula

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    given = "V = I R \\vert I = 2 A, R = 5 \\Omega"
    assert verdicts([given, "\\boxed{V = 10 V}"]) == [("right", True)]
    assert verdicts([given, "\\boxed{V = 12 V}"]) == [("wrong", True)]
    assert verdicts(["\\fbox{2 \\cdot 3 = 6}"]) == [("right", True)]
    assert verdicts(["{\\color{red} 3 + 4 = 8}"]) == [("wrong step", True)]
    assert verdicts(["I = 2 A, R = 5 \\Omega", "V \\overset{Ohm}{=} I R = 12 V"]) == [("wrong step", True)]
    assert verdicts(["\\tau := R C", "R = 1 k\\Omega, C = 1 \\mu F", "\\tau = 1 ms"]) == [("right", True)]
    assert verdicts(["x \\xrightarrow{KVL} 2x = 10 \\Rightarrow x = 5"]) == [("right", True)]
    assert verdicts(["7 \\equiv 1 \\pmod{3}"]) == []                  # a congruence: not an equation
    for t in ("\\boxed{10 V}", "\\underline{\\underline{10 V}}", "V = \\boxed{10 V}", "\\textcolor{red}{10 V}"):
        assert compare_numeric(t, "10 V").status == "correct", t
    assert compare_numeric("\\boxed{12 V}", "10 V").status == "wrong"
    for t in ("\\underline{I = \\frac{V}{R}}", "I \\overset{def}{=} \\frac{V}{R}", "I := \\frac{V}{R}"):
        assert compare_formula(t, "I = \\frac{V}{R}").status == "equivalent", t
    assert compare_formula("\\boxed{I = V R}", "I = \\frac{V}{R}").status == "not_equivalent"
    assert undecorate("\\boxed{V = 10") == " V = 10"                  # a frame never closed: the command dropped


def test_angles_in_degrees_without_the_mark_and_decibels():
    """'\\cos(60) = 0.5', '\\arctan(\\frac{4}{3}) = 53.13' (degrees, the mark left out), 'A_v = 100 = 40 dB', a phasor as
    e^{j45} or \\angle 0.927 (radians) were false wrong steps; '5 \\cos(10) = 4.92' holds only with 10 in degrees - 10
    may be radians from omega t and a calculator in degrees: right, not sure; a wrong one stays wrong"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in (["\\cos(60) = 0.5"], ["\\sin(30) + \\cos(60) = 1"], ["\\cos(36.87) = 0.8"], ["\\tan^{-1}(\\frac{4}{3}) = 53.13"],
                  ["\\varphi = \\arctan(\\frac{4}{3}) = 53.13"], ["A_v = 100 = 40 dB"], ["|H| = 0.707 = -3 dB"], ["A = 7 = 17 dB"],
                  ["Z = 1 + j = 1.414 e^{j 45}"], ["Z = 3 + j4 = 5 \\angle 0.927"], ["Z = 1 + j1 = \\sqrt{2} \\angle \\frac{\\pi}{4}"],
                  ["\\cos(2) = -0.416"], ["I = \\frac{12}{470} = 2.553 \\cdot 10^{-2} A"], ["I = \\frac{12}{470} = 2.553e-2 A"]):
        assert verdicts(right) == [("right", True)], right
    assert verdicts(["v = 5 \\cos(10) = 4.92"]) == [("right", False)]          # radians, a calculator in degrees?
    assert verdicts(["A_p = 100 = 20 dB"]) == [("right", False)]               # a power's 10 log10
    for wrong in (["\\cos(60) = 0.6"], ["\\arctan(1) = 40"], ["A_v = 100 = 30 dB"], ["Z = 1 + j = 1.414 e^{j 30}"],
                  ["I = \\frac{12}{470} = 3.553E-2 A"]):
        assert verdicts(wrong) == [("wrong step", True)], wrong
    # with a formula: the answer in degrees, the data's angle bare, a gain in dB
    assert verdicts(["\\varphi = \\arctan(\\frac{X}{R})", "X = 4, R = 3", "\\varphi = 53.13"]) == [("right", True)]
    assert verdicts(["\\varphi = \\arctan(\\frac{X}{R})", "X = 4, R = 3", "\\varphi = 50"]) == [("wrong", True)]
    assert verdicts(["P = V I \\cos(\\varphi)", "V = 10, I = 2, \\varphi = 60", "P = 10"]) == [("right", True)]
    assert verdicts(["P = V I \\cos(\\varphi)", "V = 10, I = 2, \\varphi = 60", "P = 12"]) == [("wrong", True)]
    assert verdicts(["A_v = g_m R_D", "g_m = 10 mS, R_D = 10 k\\Omega", "A_v = 40 dB"]) == [("right", True)]
    assert verdicts(["A_v = g_m R_D", "g_m = 10 mS, R_D = 10 k\\Omega", "A_v = 30 dB"]) == [("wrong", True)]


def test_a_rounded_value_is_no_wrong_step():
    """'\\sqrt{2} = 1.4', '\\frac{10}{3} = 3.3 A', '\\frac{100}{7} = 14' were wrong steps (1 % only, the decimals written not
    looked at in a chain): a value rounded to the decimals written is right - up to 15 % ('0.0001' for 0.00005 is no
    rounding), an integer from two digits on ('5' for 4.6 is not)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("\\sqrt{2} = 1.4", "\\pi = 3.1", "\\frac{1}{3} = 0.3", "\\frac{2}{3} = 0.7", "I = \\frac{10}{3} = 3.3 A",
                  "\\frac{100}{7} = 14", "\\frac{1}{7} = 0.14", "V = 50 \\cdot 10^{-6} \\cdot 1 = 0.00005"):
        assert verdicts([right]) == [("right", True)], right
    for wrong in ("\\frac{2}{3} = 0.6", "\\frac{100}{7} = 15", "\\frac{1}{3} = 0.4", "\\frac{23}{5} = 5",
                  "V = 50 \\cdot 10^{-6} \\cdot 1 = 0.0001"):
        assert verdicts([wrong]) == [("wrong step", True)], wrong
    assert verdicts(["I = \\frac{V}{R}", "V = 100, R = 7", "I = 14"]) == [("right", True)]


def test_steps_in_letters_along_a_chain_and_names_after_numbers():
    """'y = (x+1)^2 = x^2 + 2x + 1': the step between the sides in letters was never checked (only a two-sided identity
    was); a wrong partial fraction expansion along a chain is a wrong step, another relation in letters may be an
    equation to solve ('V_A = 10 - 2I = 4I + 2': nothing). '\\frac{2 V_m}{\\pi}' was read as 2 volts and a stray '_m' (a
    false wrong step)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["y = (x+1)^2 = x^2 + 2x + 1"]) == [("right", True)]
    assert verdicts(["(x+1)^2 = x^2 + 2x + 1 = (x+1)(x+1)"]) == [("right", True), ("right", True)]
    assert verdicts(["X(s) = \\frac{1}{s(s+1)} = \\frac{1}{s} - \\frac{1}{s+1}"]) == [("right", True)]
    assert verdicts(["X(s) = \\frac{1}{s(s+1)} = \\frac{1}{s} + \\frac{1}{s+1}"]) == [("wrong step", True)]
    assert verdicts(["y = (x+1)^2 = x^2 + x + 1"]) == [("wrong step", True)]     # (an expansion written: see below)
    assert verdicts(["V_A = 10 - 2I = 4I + 2"]) == []
    assert verdicts(["V_{avg} = \\frac{2 V_m}{\\pi} = \\frac{2 \\cdot 10}{\\pi} = 6.37"]) == [("right", True)]
    assert verdicts(["P = \\frac{1}{2} V_m I_m = \\frac{1}{2} \\cdot 10 \\cdot 2 = 10 W"]) == [("right", True)]


def test_powers_of_ten_parallel_percent_and_kilo_ohm_habits():
    """'\\frac{1}{2\\pi \\cdot 10^3 \\cdot 10^{-6}} = 159.15' was read 10^(3 \\cdot 10^{-6}) (a false wrong step, '0.159 is not
    159'); '30k \\| 10k = 7.5k' lost its parallel ('10 is not 7500'); '\\frac{5}{10} \\cdot 100 = 50\\%' took 50 % for 0.5; a
    formula in percent ('\\eta = \\frac{P_o}{P_i} \\cdot 100') against '\\eta = 50\\%'; '2.6 \\cdot 10^{-2}' and '26 mA' for
    25.53 mA are roundings; '\\frac{3 - 0.7}{7.5 + 101} = 21.2 \\mu A' - bare numbers in kΩ, so mA - right, not sure;
    '= 0.6 \\text{lagging}' a word, not letters"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("f_c = \\frac{1}{2\\pi \\cdot 10^3 \\cdot 10^{-6}} = 159.15 Hz", "\\omega_c = \\frac{1}{10^3 \\cdot 10^{-6}} = 1000",
                  "R_{th} = R_1 \\| R_2 = 30k \\| 10k = 7.5k", "\\eta = \\frac{5}{10} \\cdot 100 = 50\\%",
                  "I = \\frac{12}{470} = 2.6 \\cdot 10^{-2} A", "I = \\frac{12}{470} = 26 mA",
                  "pf = \\frac{P}{|S|} = \\frac{300}{500} = 0.6 \\text{ lagging}"):
        assert verdicts([right]) == [("right", True)], right
    for wrong in ("\\eta = \\frac{5}{10} \\cdot 100 = 60\\%", "I = \\frac{12}{470} = 2.7 \\cdot 10^{-2} A", "I = \\frac{12}{470} = 27 mA",
                  "I = \\frac{2.3 V}{108.5 k\\Omega} = 21.2 mA", "I_B = \\frac{3 - 0.7}{7.5 + 101 \\cdot 1} = 30 \\mu A"):
        assert verdicts([wrong]) == [("wrong step", True)], wrong
    assert verdicts(["I_B = \\frac{3 - 0.7}{7.5 + 101 \\cdot 1} = 21.2 \\mu A"]) == [("right", False)]
    assert verdicts(["\\eta = \\frac{P_o}{P_i} \\cdot 100", "P_o = 5, P_i = 10", "\\eta = 50\\%"]) == [("right", True)]
    assert verdicts(["\\eta = \\frac{P_o}{P_i} \\cdot 100", "P_o = 5, P_i = 10", "\\eta = 60\\%"]) == [("wrong", True)]
    assert verdicts(["\\eta = \\frac{P_o}{P_i}", "P_o = 5, P_i = 10", "\\eta = 0.5\\%"]) == [("wrong", True)]
    assert verdicts(["I = \\frac{V}{R}", "V = 12, R = 470", "I = 2.6 \\cdot 10^{-2}"]) == [("right", True)]


def test_binary_arithmetic_boolean_ellipsis_and_loose_roots():
    """a digital-systems exam writes binary without the base: '1 + 1 = 10', '101 + 011 = 1000', '1111 = 15', '-5 = 1011'
    (two's complement) were sure wrong steps - right now; '1 + 1 = 1' (Boolean) right, not sure; a wrong one says its
    binary value. '1 + 2 + ... + 10 = 55' and '5 \\bmod 3 = 2' were read as 13 and 3 (false wrong steps): no verdict.
    '\\sqrt{16} = \\pm 4' checks the root itself (its '-4' was a wrong step); 'x = \\pm 4' no false step '+4 = -4'"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("1 + 1 = 10", "101 + 011 = 1000", "1011 - 0110 = 0101", "11 \\cdot 11 = 1001", "1111 = 15", "-5 = 1011",
                  "-5 = 11111011", "S = 101 + 011 = 1000", "\\sqrt{16} = \\pm 4"):
        assert verdicts([right]) == [("right", True)], right
    assert verdicts(["1 + 1 = 1"]) == [("right", False)]
    for wrong in ("1 + 1 = 11", "101 + 011 = 1001", "1111 = 14", "-5 = 1010", "10 + 10 = 30", "1 \\cdot 1 = 0", "\\sqrt{16} = \\pm 5"):
        assert verdicts([wrong]) == [("wrong step", True)], wrong
    for worked in ("1 + 2 + ... + 10 = 55", "1 + 2 + \\cdots + 10 = 55", "5 \\bmod 3 = 2"):     # (worked out now)
        assert verdicts([worked]) == [("right", True)], worked
    assert verdicts(["x^2 = 9", "x = \\pm 4"]) == [("wrong", True)]
    assert verdicts(["\\frac{1}{3} = 0.333..."]) == [("right", True)]          # a repeating decimal's dots
    st = statements("101 + 011 = 1001")                                        # the message names the binary value
    assert "in binary 1000" in check_statements(st)[0].text


def test_rubric_bare_number_in_si_and_angle_in_deg():
    """a bare number for a quantity with a unit was read in the key's unit only: '0.02553' for 25.53 mA was a prefix error
    - in SI units it is right; '10 \\angle 30 deg' was a parse error"""
    from grader.numeric import compare_numeric
    assert compare_numeric("0.02553", "25.53 mA").status == "correct"
    assert compare_numeric("25.53", "25.53 mA").status == "correct"
    assert compare_numeric("25530", "25.53 mA").status == "prefix_error"
    assert compare_numeric("10 \\angle 30 deg", "8.66 + j5").status == "correct"


def test_signed_powers_without_braces_and_x_as_times():
    """'2.55 \\cdot 10^-3' (no braces, as typed or read) was read as 10 - a false wrong step on every such value; 'e^-1000t'
    as e^{-} times 1000t; '2.55 x 10^{-3}' left x a letter (no verdict) - now 10^{-3}, e^{-1000t} and a times sign, in the
    derivation checks and the rubric alike"""
    from grader.numeric import compare_numeric
    from grader.selfcheck import check_statements, statements
    from grader.symbolic import compare_formula

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("I = 12/4700 = 2.55 \\cdot 10^-3 A", "\\tau = 10^3 \\cdot 10^-6 = 10^-3", "I = \\frac{12}{4700} = 2.55 x 10^{-3} A"):
        assert verdicts([right]) == [("right", True)], right
    for wrong in ("I = 12/4700 = 3.55 \\cdot 10^-3 A", "I = \\frac{12}{4700} = 3.55 x 10^{-3} A"):
        assert verdicts([wrong]) == [("wrong step", True)], wrong
    assert verdicts(["v(t) = 10 e^-1000t", "v(0.001) = 3.68"]) == [("right", True)]
    assert verdicts(["y = 2x + 3", "x = 2", "y = 7"]) == [("right", True)]          # a letter x stays a letter
    assert compare_formula("I = 2.55 \\cdot 10^-3", "I = 0.00255").status == "equivalent"
    assert compare_numeric("2.55 x 10^{-3} A", "2.553 mA").status == "correct"


def test_circuit_values_rounded_to_their_digits():
    """'V_A = 3.3 V' for 3.333 V was 'close (rounded?)', not sure - a value rounded to the digits written is the circuit's
    (sure); a wrong rounding ('6.6' for 6.667) or a coarse one ('3 mA', '0.003 A' for 3.333 mA) stays as it was;
    '3.333... V' was not read"""
    from grader.circuit import Component as C, Netlist, Voltage
    from grader.circuit_check import check_against, circuit_statements
    div = Netlist([C("E", "voltage_source", ["a", "0"], value="10V", plus="a", label="E"),
                   C("R1", "resistor", ["a", "m"], value="2k", label="R_1"), C("R2", "resistor", ["m", "0"], value="1k", label="R_2")],
                  ground="0", node_labels={"m": "A"}, voltages=[Voltage("V_x", "a", "m", "R1")])

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), div)]
    for right in ("I = 3.3 mA", "V_A = 3.3 V", "V_A = 3.3 [V]", "I = 0.0033 A", "V_x = 6.7 V", "V_A = 3.333... V"):
        assert verdicts(right) == [("right", True)], right
    for near in ("I = 3.4 mA", "V_x = 6.6 V"):
        assert verdicts(near) == [("right", False)], near
    for wrong in ("I = 3 mA", "I = 0.003 A"):
        assert verdicts(wrong) == [("wrong", True)], wrong


def test_messages_show_fractions_angles_and_percent_plainly():
    """a step's message showed '\\frac102π' for 1/(2π), '\\angle', '^\\circ' as typed, and any '\\cdot 0.01' as a percent
    sign ('8.85·10^-12%0.001'): fractions now read a/b, angles ∠ and °, and only a percent sign the student wrote is '%'"""
    from grader.selfcheck import _pretty, check_statements, statements
    assert _pretty("\\frac{10}{3}") == "10/3"
    assert _pretty("\\frac{1}{2\\pi}") == "1/(2π)"
    assert _pretty("\\frac{V_{in}}{R_1}") == "V_in/R_1"
    assert _pretty("\\frac{8.85 \\cdot 10^{-12} \\cdot 0.01}{0.001}") == "(8.85·10⁻¹²·0.01)/0.001"
    assert _pretty("2^10") == "2¹⁰" and _pretty("x^2 3") == "x²3"                # (it showed '2¹0')
    assert _pretty("2\\angle -53.13^\\circ") == "2∠-53.13°"
    st = statements("C = \\frac{8.85 \\cdot 10^{-12} \\cdot 0.01}{0.001} = 88.5 pF")
    (c,) = check_statements(st)
    assert c.verdict == "right" and "%" not in c.text
    (c,) = check_statements(statements("\\eta = \\frac{80}{100} = 80\\%"))
    assert c.verdict == "right" and "80%" in c.text
    assert [x.verdict for x in check_statements(statements("\\eta = \\frac{5}{6} = 83\\%"))] == ["right"]


def test_a_bare_micro_prefix_in_a_substitution():
    """'\\frac{1}{2\\pi \\cdot 1 k \\cdot 1 \\mu} = 159 Hz', '60 \\cdot 1k \\cdot 100 \\mu': a value with a bare micro, as EE students
    write capacitances, was a letter (no verdict) - now micro, like a bare k; a friction's or a mobility's \\mu stays a letter"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("f_c = \\frac{1}{2 \\pi R C} = \\frac{1}{2 \\pi \\cdot 1 k \\cdot 1 \\mu} = 159 Hz",
                  "V_{ripple} = \\frac{V_p}{f R C} = \\frac{10}{60 \\cdot 1k \\cdot 100 \\mu} = 1.67 V",
                  "\\tau = R C = 2k \\cdot 10\\mu = 20 ms", "F = \\mu N = 0.3 \\cdot 10 = 3"):
        assert verdicts([right]) == [("right", True)], right
    assert verdicts(["V_{ripple} = \\frac{10}{60 \\cdot 1k \\cdot 100 \\mu} = 2.67 V"]) == [("wrong step", True)]


def test_unnamed_plus_minus_roots_and_percent_after_a_hundred():
    """'s = -1 \\pm j3' (no 's_{1,2}') after its equation got no verdict - its two roots are checked now; 'PO = 100
    e^{...} = 35\\%' took 35 % for 0.35 after the 100 had made it percent (a false wrong step) - '\\frac{5}{10} = 0.5\\%'
    with no 100 in it stays wrong"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["s^2 + 2s + 10 = 0 \\Rightarrow s = -1 \\pm j3"]) == [("right", True), ("right", True)]
    assert verdicts(["s^2 + 2s + 10 = 0", "s = -1 \\pm j2"]) == [("wrong", True), ("wrong", True)]
    assert verdicts(["x^2 - 5x + 6 = 0", "x = \\frac{5 \\pm 1}{2}"]) == [("right", True), ("right", True)]
    assert verdicts(["R = 100 \\pm 5"]) == []                                  # a tolerance: nothing
    assert verdicts(["PO = 100 e^{-\\frac{0.316 \\pi}{\\sqrt{1-0.316^2}}} = 35\\%"]) == [("right", True)]
    assert verdicts(["PO = 100 e^{-\\frac{0.316 \\pi}{\\sqrt{1-0.316^2}}} = 45\\%"]) == [("wrong step", True)]
    assert verdicts(["\\frac{5}{10} = 0.5\\%"]) == [("wrong step", True)]


def test_integral_limits_and_parameters():
    """'\\int_0^1 4t^2 dt' had its limit joined to the next number - an upper limit of 14 ('E = \\int_0^1 4t^2 dt = \\frac{4}{3}'
    and 'A = \\int_0^2 3 dt = 6' were false wrong steps); '\\int_0^1 (2t)^2 dt' was read as 1 times (2t)^2 (no verdict);
    '\\int_0^\\infty e^{-at} dt = \\frac{1}{a}' was left with its convergence condition - the letters are taken positive"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("E = \\int_0^1 4t^2 dt = \\frac{4}{3}", "A = \\int_0^2 3 dt = 6", "E = \\int_0^1 (2t)^2 dt = \\frac{4}{3}",
                  "\\int_0^1 \\left(2t\\right)^2 dt = \\frac{4}{3}", "\\int_0^\\infty e^{-at} dt = \\frac{1}{a}",
                  "\\int_0^\\infty t e^{-at} dt = \\frac{1}{a^2}", "\\sin^2 30^\\circ = 0.25", "2(3) = 6", "V = 10^3 (2) = 2000"):
        assert verdicts([right]) == [("right", True)], right
    assert verdicts(["A = \\int_0^2 3 dt = 7"]) == [("wrong step", True)]
    assert verdicts(["\\int_0^\\infty e^{-at} dt = a"]) == []                  # an equation in a, maybe: never wrong


def test_letters_written_in_both_cases_are_two_quantities():
    """the LaTeX parser lowercased every letter: 'A_v = -\\frac{R_C}{r_e + R_E}' merged r_e and R_E (a right gain judged
    wrong, for sure), 'T = 2t' and 'x = \\frac{t}{T}' merged T and t. A name written in both cases in one statement now
    names two quantities; a letter written in one case only is folded as before - a v read for V still matches"""
    from grader.selfcheck import check_statements, statements
    from grader.symbolic import compare_formula

    def verdicts(lines, values=None):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st, values=values)]
    assert verdicts(["A_v = -\\frac{R_C}{r_e + R_E}", "R_C = 2, r_e = 0.025, R_E = 1", "A_v = -1.95"]) == [("right", True)]
    assert verdicts(["A_v = -\\frac{R_C}{r_e + R_E}", "R_C = 2, r_e = 0.025, R_E = 1", "A_v = -1"]) == [("wrong", True)]
    assert verdicts(["T = 2 t", "t = 3", "T = 6"]) == [("right", True)]
    assert verdicts(["x = \\frac{t}{T}", "t = 1, T = 4", "x = 0.25"]) == [("right", True)]
    assert verdicts(["V = I R", "I = 2, R = 5", "v = 10"]) == [("right", True)]          # one case only: folded
    assert verdicts(["A_v = -\\frac{R_C}{r_e + R_E}", "A_v = -1.95"],
                    {"R_C": "2", "r_e": "0.025", "R_E": "1"}) == [("right", True)]     # the lecturer's names too
    assert compare_formula("x = \\frac{t}{T}", "x = \\frac{t}{T}").status == "equivalent"
    assert compare_formula("x = \\frac{t}{T}", "x = 1").status == "not_equivalent"
    assert compare_formula("v = I r", "V = I R").status == "equivalent"                 # a case slip, as before


def test_small_signal_resistances_are_not_the_resistors_drawn():
    """the circuit checks matched names without their case: 'r_e = 25 Ω' (the emitter's small-signal resistance) was
    judged against the resistor R_E = 1 kΩ drawn beside it - wrong, for sure. A lowercase r_e, r_\\pi, r_o names its own
    quantity; R_E is still the resistor"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("E", "voltage_source", ["a", "0"], value="10V", plus="a", label="V_{CC}"),
                   C("R1", "resistor", ["a", "m"], value="2k", label="R_C"), C("R2", "resistor", ["m", "0"], value="1k", label="R_E")],
                  ground="0", node_labels={"m": "A"})

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(t), net)]
    assert verdicts("r_e = 25 \\Omega") == []
    assert verdicts("r_e = \\frac{V_T}{I_E} = \\frac{25 mV}{1 mA} = 25 \\Omega") == []
    assert verdicts("R_E = 1 k\\Omega") == [("right", True)]
    assert verdicts("R_E = 2 k\\Omega") == [("wrong", True)]


def test_a_number_before_a_fraction_is_a_product_or_a_mixed_number():
    """the LaTeX parser reads '2 \\frac{1}{2}' as the mixed number 2.5: 'P = 2 \\frac{1}{2} = 1' and 'x = 3 \\frac{1}{4} =
    0.75' (a product, as engineers write) were false wrong steps - either reading holds now; neither, a wrong step"""
    from grader.selfcheck import _pretty, check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for right in ("P = 2 \\frac{1}{2} = 1", "x = 2 \\frac{1}{2} = 2.5", "x = 3 \\frac{1}{4} = 0.75", "V = 12 \\frac{2}{6} = 4"):
        assert verdicts([right]) == [("right", True)], right
    assert verdicts(["x = 3 \\frac{1}{4} = 2"]) == [("wrong step", True)]
    assert verdicts(["y = 2x", "x = 1 \\frac{1}{2}", "y = 3"]) == [("right", True)]
    assert _pretty("2 \\frac{1}{2}") == "2 1/2"


def test_a_root_of_a_number_without_braces():
    """'\\sqrt 20' is √2 · 0 to TeX (one token): '|Z| = \\sqrt 20 = 4.47' was a false wrong step - the whole number is under
    the root now ('\\sqrt 2', '\\sqrt{x}' as before)"""
    from grader.selfcheck import _parse, check_statements, statements
    assert str(_parse("\\sqrt 20")) == str(_parse("\\sqrt{20}")) and str(_parse("\\sqrt 2")) == "sqrt(2)"
    assert [(c.verdict, c.sure) for c in check_statements(statements("|Z| = \\sqrt 20 = 4.47"))] == [("right", True)]
    assert [(c.verdict, c.sure) for c in check_statements(statements("|Z| = \\sqrt 20 = 5"))] == [("wrong step", True)]


def test_unit_conversions_with_a_prefix_or_a_power_of_ten():
    """'1 \\mu F = 10^{-6} F', '1 ms = 10^{-3} s' got no verdict: the unit after a prefix or a power of ten was taken for a
    letter of the problem (F a force) - a unit right after a number, its prefix between, is a unit ('F = \\mu N = 0.3 N =
    6 N', N a force there, still nothing)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    for right in ("1 \\mu F = 10^{-6} F", "1 ms = 10^{-3} s", "1 nF = 10^{-9} F", "1 kWh = 3.6 MJ", "30 min = 0.5 h"):
        assert verdicts(right) == [("right", True)], right
    assert verdicts("1 \\mu F = 10^{-3} F") == [("wrong step", True)]
    assert verdicts("2 A = 200 mA") == [("wrong step", True)]
    assert verdicts("F = \\mu N = 0.3 N = 6 N") == []


def test_a_function_with_letters_evaluated_with_their_values():
    """'v_C(t) = V_0 e^{-t/\\tau}' with 'V_0 = 10, \\tau = 1' (before or after it), then 'v_C(1) = 3.68': with two letters
    in it the value was taken for a formula, never checked - it is a chain with the values put in now"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["V_0 = 10, \\tau = 1", "v_C(t) = V_0 e^{-t/\\tau}", "v_C(1) = 3.68"]) == [("right", True)]
    assert verdicts(["v_C(t) = V_0 e^{-t/\\tau}", "V_0 = 10, \\tau = 1", "v_C(1) = 3.68"]) == [("right", True)]
    assert verdicts(["v_C(t) = V_0 e^{-t/\\tau}", "V_0 = 10, \\tau = 1", "v_C(1) = 5"]) == [("wrong step", True)]
    assert verdicts(["f(x) = a x^2", "a = 2", "f(3) = 18"]) == [("right", True)]
    assert verdicts(["f(x) = 2x + 1", "f(x) = 7", "x = 3"]) == [("right", True)]          # still an equation for x


def test_typed_arrows_and_delta():
    """plain-text work types its arrows: '2x = 6 ⇒ x = 3', '→', '⟹', '∴', '⇔' were no separators - one false step
    ('6⇒x = 3': 6 is not 3); a typed 'ΔV' was Delta times V - one name now, as '\\Delta V'"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    for arrow in ("⇒", "→", "⟹", "∴", "⇔"):
        assert verdicts([f"2x = 6 {arrow} x = 3"]) == [("right", True)], arrow
    assert verdicts(["2x = 6 ⇒ x = 4"]) == [("wrong", True)]
    assert verdicts(["\\lim_{x → 0} \\frac{\\sin x}{x} = 1"]) == [("right", True)]
    assert verdicts(["R = ΔV/ΔI", "ΔV = 2 V, ΔI = 4 mA", "R = 500 Ω"]) == [("right", True)]


def test_probability_notation_and_chains_with_an_unreadable_side():
    """'P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.3': P of a relation was read as a number ('2 is not 0.3'), 'E[X^2] - E[X]^2'
    as e times X^2 (a false wrong step) - left unread now; and a chain with a side the parser cannot read keeps the
    steps of numbers around it ('1 - 0.7 = 0.3' - the whole was dropped before)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.3") == [("right", True)]
    assert verdicts("P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.4") == [("wrong step", True)]
    assert verdicts("Var(X) = E[X^2] - E[X]^2 = 4.9 - 4.41 = 0.49") == [("right", True)]
    assert verdicts("E[X] = 2.1") == [] and verdicts("1 + 2 + ... + 10 = 55") == [("right", True)]
    assert verdicts("I = \\Re\\{Z\\} = 3 + 1 = 4") == [("right", True)]


def test_matrices_written_out():
    """matrices written out: a chain of them in numbers is compared entry by entry ('M^{-1} = \\frac{1}{-2} [...] = [...]',
    a product, a transpose - '^T', or '^t' as Israeli courses write it), each entry to its digits; 'A x = b' with the
    unknowns in letters is a system of equations - the values after it are checked against its solution. A matrix power
    used to crash the checks (sympy recursed without end), and the parser read 'M + 2' as 2, 'M^{\\top}' as 4 - no
    verdict on what cannot be worked out. Messages show a matrix as '[1, 2; 3, 4]', a determinant as 'det[...]'"""
    from grader.selfcheck import _pretty, check_statements, statements

    def checks(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]

    def verdicts(lines):
        return [(v, s) for v, s, _ in checks(lines)]
    inv = "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}^{-1} = \\frac{1}{-2} \\begin{pmatrix} 4 & -2 \\\\ -3 & 1 \\end{pmatrix} = "
    assert verdicts([inv + "\\begin{pmatrix} -2 & 1 \\\\ 1.5 & -0.5 \\end{pmatrix}"]) == [("right", True)] * 2
    got = checks([inv + "\\begin{pmatrix} -2 & 1 \\\\ 3 & -0.5 \\end{pmatrix}"])
    assert [g[:2] for g in got] == [("right", True), ("wrong step", True)]
    assert got[1][2].endswith("[-2, 1; 1.5, -0.5] is not [-2, 1; 3, -0.5]") and "1/-2·[4, -2; -3, 1]" in got[1][2]
    prod = "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix} \\begin{pmatrix} 1 \\\\ 1 \\end{pmatrix} = "
    assert verdicts([prod + "\\begin{pmatrix} 3 \\\\ 7 \\end{pmatrix}"]) == [("right", True)]
    assert verdicts([prod + "\\begin{pmatrix} 3 \\\\ 8 \\end{pmatrix}"]) == [("wrong step", True)]
    for t in ("^{T}", "^t", "^{\\top}"):
        assert verdicts(["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}" + t + " = \\begin{pmatrix} 1 & 3 \\\\ 2 & 4 \\end{pmatrix}"]) \
            == [("right", True)], t
    rounded = "\\begin{pmatrix} 3 & 1 \\\\ 1 & 4 \\end{pmatrix}^{-1} = \\begin{pmatrix} 0.36 & -0.09 \\\\ -0.09 & "
    assert verdicts([rounded + "0.27 \\end{pmatrix}"]) == [("right", True)]
    assert verdicts([rounded + "0.37 \\end{pmatrix}"]) == [("wrong step", True)]
    assert verdicts(["\\begin{bmatrix} 1+j & -j \\\\ -j & 2 \\end{bmatrix} \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix} = "
                     "\\begin{bmatrix} 1 \\\\ 2-j \\end{bmatrix}"]) == [("right", True)]
    assert verdicts(["\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1} = \\frac{1}{ad - bc} "
                     "\\begin{pmatrix} d & -b \\\\ -c & a \\end{pmatrix}"]) == [("right", True)]
    system = "\\begin{bmatrix} 2 & -1 \\\\ -1 & 3 \\end{bmatrix} \\begin{bmatrix} V_1 \\\\ V_2 \\end{bmatrix} = \\begin{bmatrix} 1 \\\\ 2 \\end{bmatrix}"
    assert verdicts([system, "V_1 = 1", "V_2 = 1"]) == [("right", True)] * 2
    got = checks([system, "V_1 = 1", "V_2 = 2"])
    assert [g[:2] for g in got] == [("right", True), ("wrong", True)] and "V_1 = 1, V_2 = 1" in got[1][2]
    for bad in (["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix} + 2 = 5"], ["A = \\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}"],
                ["\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}^{\\top} = 4"], ["\\begin{pmatrix} 1 & 2 & 3 \\\\ 4 & 5 \\end{pmatrix} = 2"],
                ["\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1} = \\frac{1}{ad - bc} \\begin{pmatrix} a & -b \\\\ -c & d \\end{pmatrix}"]):
        assert verdicts(bad) == [], bad
    assert verdicts(["\\det \\begin{pmatrix} 2 & 1 \\\\ 1 & 3 \\end{pmatrix} = 5"]) == [("right", True)]
    assert _pretty("\\begin{vmatrix} 2 & 1 \\\\ 1 & 3 \\end{vmatrix} = 5") == "det[2, 1; 1, 3]=5"


def test_the_degrees_note_only_where_it_changes_the_value():
    """'\\cos(60^\\circ) = -0.952' is wrong either way: the message said '0.5 (0.5 with the angle in degrees)' - the note
    only when the angle in degrees gives another value ('\\cos(60) = 0.6')"""
    from grader.selfcheck import check_statements, statements
    (c,) = check_statements(statements("\\cos(60^\\circ) = -0.952"))
    assert (c.verdict, c.sure) == ("wrong step", True) and "with the angle in degrees" not in c.text and "0.5 is not -0.952" in c.text
    (c,) = check_statements(statements("\\cos(60) = 0.6"))
    assert c.verdict == "wrong step" and "(0.5 with the angle in degrees)" in c.text


def test_bare_prefixes_in_values_and_results():
    """EE students write values with a bare prefix: 'C = 100n', 'C = 10\\mu', 'R = 1k' were taken as 100, 10, 1 - and a
    formula over them judged a false sure 'wrong' ('\\tau = RC = 0.1 ms', 'X_C = 318 \\Omega'); a result '= 100n' as 100.
    A bare m is metres or milli: milli where the formula or the step holds only so ('L = 10m', '\\frac{100}{80 m} = 1.25
    k\\Omega') - never a sure wrong from that reading ('\\rho \\frac{100 m}{1 mm^2}': metres). A single bare prefix in a
    product counts when the result has one ('2 \\cdot 4k = 8 kHz'); an inequality separates statements ('f_s \\ge ...');
    'bit', 'bps' are units"""
    from grader.selfcheck import check_statements, statements

    def checks(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]

    def verdicts(lines):
        return [(v, s) for v, s, _ in checks(lines)]
    assert verdicts(["R = 1k", "C = 100n", "\\tau = R C", "\\tau = 0.1 ms"]) == [("right", True)]
    assert verdicts(["R = 1k", "C = 100n", "\\tau = R C", "\\tau = 1 ms"]) == [("wrong", True)]
    got = checks(["L = 10m", "C = 1\\mu", "f_0 = \\frac{1}{2\\pi\\sqrt{LC}}", "f_0 = 1.59 kHz"])
    assert [g[:2] for g in got] == [("right", True)] and "L in milli" in got[0][2]
    assert verdicts(["L = 10m", "C = 1\\mu", "f_0 = \\frac{1}{2\\pi\\sqrt{LC}} = 1.59 kHz"]) == [("right", True)]
    assert verdicts(["C = 10\\mu", "X_C = \\frac{1}{2 \\pi 50 C} = 318 \\Omega"]) == [("right", True)]
    for last, want in (("100n", "right"), ("100 n", "right"), ("0.1 \\mu", "right"), ("200n", "wrong step")):
        assert verdicts(["C = \\frac{1}{2 \\pi 1000 \\cdot 1590} = " + last]) == [(want, True)], last
    # (metres or milli: a step right only with one of them, or wrong with both, is not sure - it was no verdict, a slip
    # passed: '2m \\cdot 4k = 9 V')
    assert verdicts(["R = \\rho \\frac{L}{A} = 1.7 \\cdot 10^{-8} \\frac{100 m}{1 mm^2} = 1.7 \\Omega"]) == [("right", False)]
    assert verdicts(["r_\\pi = \\frac{\\beta}{g_m} = \\frac{100}{80 m} = 1.25 k\\Omega"]) == [("right", True)]
    assert verdicts(["r_\\pi = \\frac{\\beta}{g_m} = \\frac{100}{80 m} = 2.5 k\\Omega"]) == [("wrong step", False)]
    assert verdicts(["V = I R = 2m \\cdot 4k = 9 V"]) == [("wrong step", False)]
    assert verdicts(["A_v = -g_m R_C = -80m \\cdot 2k = -160"]) == [("right", True)]
    assert verdicts(["\\tau = R C = 10k \\cdot 100n = 1m"]) == [("right", True)]
    assert verdicts(["f_s \\ge 2 f_{max} = 2 \\cdot 4k = 8 kHz"]) == [("right", True)]
    assert verdicts(["f_s \\ge 2 f_{max} = 2 \\cdot 4k = 9 kHz"]) == [("wrong step", True)]
    assert verdicts(["N = \\log_2 M = \\log_2 16 = 4 bit"]) == [("right", True)]
    assert verdicts(["P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.3"]) == [("right", True)]


def test_a_letter_worked_out_is_data_for_the_formulas_after_it():
    """'\\omega_n = \\sqrt{10} = 3.16', then '2\\zeta\\omega_n = 2', '\\zeta = 0.316': the formula found no value for omega_n
    ('cannot check') - a letter worked out in a chain is data for the formulas after it, like one written alone.
    Messages name the letters as written ('R=5000', not 'r=5000'; 'ω_n')"""
    from grader.selfcheck import check_statements, statements

    def checks(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]
    got = checks(["\\omega_n = \\sqrt{10} = 3.16", "2\\zeta\\omega_n = 2", "\\zeta = 0.316"])
    assert [g[:2] for g in got] == [("right", True), ("right", True)] and "ω_n=3.16" in got[1][2]
    got = checks(["\\omega_n = \\sqrt{10} = 3.16", "2\\zeta\\omega_n = 2", "\\zeta = 0.5"])
    assert [g[:2] for g in got] == [("right", True), ("wrong", True)]
    got = checks(["R = R_1 + R_2 = 2k + 3k = 5 k\\Omega", "I = \\frac{V}{R}", "V = 10 V", "I = 2 mA"])
    assert [g[:2] for g in got] == [("right", True), ("right", True)] and "V=10, R=5000" in got[1][2]
    got = checks(["T = \\frac{1}{f} = \\frac{1}{50} = 20 ms", "\\omega = \\frac{2\\pi}{T}", "\\omega = 314"])
    assert [g[:2] for g in got] == [("right", True), ("right", True)]


def test_tiny_values_are_compared_and_typed_scripts():
    """every value under 1e-9 was equal to every other (a floor of tolerance for rounding noise): '10 pF + 20 pF = 50 pF',
    '1.6 \\cdot 10^{-19} \\cdot 5 = 8 \\cdot 10^{-18}' were right - decimals are worked out exactly now, the floor kept
    beside a zero only ('0.5 + 0.3 - 0.8 = 0'); messages show '3e-11', not '0'. Typed superscripts and subscripts
    ('8×10⁻¹⁹ J', 'log₂(8)') and 'arctan' as typed are read; '12 / 4.7k' is 12 over 4.7 kilo"""
    from grader.selfcheck import check_statements, statements

    def checks(t):
        return [(c.verdict, c.sure, c.text) for c in check_statements(statements(t))]

    def verdicts(t):
        return [(v, s) for v, s, _ in checks(t)]
    got = checks("C = 10 pF + 20 pF = 50 pF")
    assert [g[:2] for g in got] == [("wrong step", True)] and got[0][2].endswith("3e-11 is not 5e-11")
    assert verdicts("C = 10 pF + 20 pF = 30 pF") == [("right", True)]
    assert verdicts("q = 1.6 \\cdot 10^{-19} \\cdot 5 = 8 \\cdot 10^{-18}") == [("wrong step", True)]
    assert verdicts("q = 1.6 \\cdot 10^{-19} \\cdot 5 = 8 \\cdot 10^{-19}") == [("right", True)]
    assert verdicts("I = I_1 - I_2 - I_3 = 0.5 - 0.3 - 0.2 = 0") == [("right", True)]
    assert verdicts("x = 0.1 + 0.2 - 0.3 = 3.16^2 - 9.9856") == [("right", True)]
    assert verdicts("E = 1.6×10⁻¹⁹ · 5 = 8×10⁻¹⁹ J") == [("right", True)]
    assert verdicts("E = 1.6×10⁻¹⁹ · 5 = 8×10⁻¹⁸ J") == [("wrong step", True)]
    assert verdicts("n = log₂(8) = 3") == [("right", True)] and verdicts("n = log₂(8) = 4") == [("wrong step", True)]
    assert verdicts("θ = arctan(4/3) = 53.13°") == [("right", True)]
    assert verdicts("y = 2·sin(30°) = 1") == [("right", True)]
    assert verdicts("I = 12 / 4.7k = 2.55 mA") == [("right", True)]


def test_an_exponential_rounded_on_the_way():
    """'I_D = 10^{-14} e^{0.7/0.0259} = 5.32 mA': the exponent kept to 3 digits (27.0 for 27.03) moves the value 3 % - a
    false wrong step at 1 %; after an exponential of a large argument a step may be 5 % off (a wrong factor, a sign
    still caught)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["I_D = 10^{-14} e^{0.7/0.0259} = 5.32 mA"]) == [("right", True)]
    assert verdicts(["I_D = 10^{-14} e^{0.7/0.0259} = 6.32 mA"]) == [("wrong step", True)]
    assert verdicts(["I_S = 10^{-14}", "V_D = 0.7", "V_T = 0.0259", "I_D = I_S e^{V_D/V_T}", "I_D = 5.32 mA"]) == [("right", True)]
    assert verdicts(["I_S = 10^{-14}", "V_D = 0.7", "V_T = 0.0259", "I_D = I_S e^{V_D/V_T}", "I_D = 6.32 mA"]) == [("wrong", True)]
    assert verdicts(["n_i = \\sqrt{2.8 \\cdot 10^{19} \\cdot 1.04 \\cdot 10^{19}} e^{-1.12/0.0518} = 7.1 \\cdot 10^{9}"]) == [("right", True)]
    assert verdicts(["v = 10 e^{-2} = 1.39"]) == [("wrong step", True)]          # (a small exponent: as before)


def test_an_equals_sign_inside_brackets_is_no_side():
    """'P(X = 2) = \\binom{5}{2} 0.3^2 0.7^3 = 0.3087' was cut at the '=' inside P(...): '2) = \\binom...' a false sure wrong
    step; and the numbers after a side in letters are steps still ('E[X] = np = 10 \\cdot 0.3 = 3')"""
    from grader.selfcheck import _pretty, check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("P(X = 2) = \\binom{5}{2} 0.3^2 0.7^3 = 0.3087") == [("right", True)]
    assert verdicts("P(X = 2) = \\binom{5}{2} 0.3^2 0.7^3 = 0.4087") == [("wrong step", True)]
    assert verdicts("P(X = 3) = \\frac{\\lambda^3 e^{-\\lambda}}{3!} = \\frac{2^3 e^{-2}}{6} = 0.18") == [("right", True)]
    assert verdicts("E[X] = np = 10 \\cdot 0.3 = 3") == [("right", True)]
    assert verdicts("E[X] = np = 10 \\cdot 0.3 = 4") == [("wrong step", True)]
    assert _pretty("\\binom{5}{2}") == "C(5,2)"


def test_approximately_and_a_value_worked_out_a_line_at_a_time():
    """'\\sim' before a number is 'approximately' ('I = \\frac{4}{3.67} \\sim 1.09 A'; 'X \\sim N(0, 1)' a distribution - not a
    value); 'V = 5 \\cdot 2 \\Rightarrow V = 10' without a formula above: a step, as it is under one (it got no verdict)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["I = \\frac{4}{2.67 + 1} \\sim 1.09 A"]) == [("right", True)]
    assert verdicts(["I = \\frac{4}{2.67 + 1} \\thickapprox 1.09 A"]) == [("right", True)]
    assert verdicts(["X \\sim N(0, 1)", "P(X > 1) = 0.159"]) == []
    assert verdicts(["V = 5 \\cdot 2 \\Rightarrow V = 10"]) == [("right", True)]
    assert verdicts(["V = 5 \\cdot 2 \\Rightarrow V = 11"]) == [("wrong step", True)]
    assert verdicts(["I = \\frac{12}{4}", "I = 3 A"]) == [("right", True)]
    assert verdicts(["I = \\frac{12}{4}", "R = 2", "I = 4 A"]) == []      # (not next to each other: another I?)


def test_a_domain_condition_after_a_function():
    """'i(t) = 2e^{-t} (t > 0)': the condition in brackets made the side unreadable and the function was lost ('i(1) =
    0.736' unchecked); a probability's relation stays ('P(X > 2)')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["i(t) = 2e^{-t} (t > 0)", "i(1) = 0.736"]) == [("right", True)]
    assert verdicts(["i(t) = 2e^{-t} (t > 0)", "i(1) = 0.836"]) == [("wrong step", True)]
    assert verdicts(["v(t) = 5 \\left( 1 - e^{-t} \\right) \\left( t \\geq 0 \\right)", "v(1) = 3.16"]) == [("right", True)]
    assert verdicts(["P(X > 2) = 1 - P(X \\le 2) = 1 - 0.7 = 0.3"]) == [("right", True)]


def test_a_letter_k_before_a_name_is_no_kilo():
    """'g_m = \\sqrt{2 k I_D} = \\sqrt{2 \\cdot 2 \\cdot 4} = 4 mS': the k of '2 k I_D' (a MOSFET's) was taken for kilo, every
    side put in SI units and '4 mS' against 4 a false wrong step - a bare prefix is no prefix before a name"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("g_m = \\sqrt{2 k I_D} = \\sqrt{2 \\cdot 2 \\cdot 4} = 4 mS") == [("right", True)]
    assert verdicts("g_m = \\sqrt{2 k I_D} = \\sqrt{2 \\cdot 2 \\cdot 4} = 5 mS") == [("wrong step", True)]
    assert verdicts("f_s \\ge 2 f_{max} = 2 \\cdot 4k = 8 kHz") == [("right", True)]


def test_a_power_of_1000_apart_with_bare_numbers_is_for_a_human():
    """'P = I^2 R = 3^2 \\cdot 4k = 36 mW': 3 for 3 mA, as EE students keep mA and kΩ - its sides put in SI units, it was a
    false sure wrong step (36000 is not 0.036): right, not sure; '\\frac{1}{2\\pi\\sqrt{10 \\cdot 1}} = 50.3 Hz' (C in µF):
    a power of 1000 apart with bare numbers is wrong, not sure - a slip of 1000 looks the same. With every unit written
    ('1 \\mu F = 10^{-3} F', '\\frac{12 V}{4.7 k\\Omega} = 2.55 A') it stays a sure wrong step"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("P = I^2 R = 3^2 \\cdot 4k = 36 mW") == [("right", False)]
    assert verdicts("f = \\frac{1}{2\\pi \\sqrt{L C}} = \\frac{1}{2 \\pi \\sqrt{10 \\cdot 1}} = 50.3 Hz") == [("wrong step", False)]
    assert verdicts("1 \\mu F = 10^{-3} F") == [("wrong step", True)]
    assert verdicts("I = \\frac{12 V}{4.7 k\\Omega} = 2.55 A") == [("wrong step", True)]
    assert verdicts("P = I^2 R = 3^2 \\cdot 4k = 46 mW") == [("wrong step", True)]


def test_the_phase_of_a_response_and_dbm():
    """'\\angle H(j1000) = -45^\\circ' after 'H(s) = \\frac{1}{1 + s/1000}': its phase at s = j1000 (in degrees, with or
    without the mark, or radians; 225° for -135°) - it was left unread; 'dBm', 'dBW' are units ('= -50 dBm' got no verdict)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    h = "H(s) = \\frac{1}{1 + s/1000}"
    for written in ("-45^\\circ", "-45", "-0.785"):
        assert verdicts([h, "\\angle H(j1000) = " + written]) == [("right", True)], written
    assert verdicts([h, "\\angle H(j1000) = -60^\\circ"]) == [("wrong", True)]
    assert verdicts(["H(s) = \\frac{10}{s(s+10)}", "\\angle H(j10) = 225^\\circ"]) == [("right", True)]
    assert verdicts(["P_r = P_t + G_t + G_r - L = 30 + 10 + 10 - 100 = -50 dBm"]) == [("right", True)]
    assert verdicts(["P_r = P_t + G_t + G_r - L = 30 + 10 + 10 - 100 = -40 dBm"]) == [("wrong step", True)]


def test_a_parallel_of_a_sum_with_prefixes():
    """'R_{eq} = (2k + 3k) \\| 5k = 2.5k': each value in brackets of its own (2 \\cdot 10^{3}) got another pair, too deep for
    the parallel's operand - '\\|' was read as a norm: a false sure wrong step ('10 is not 2500')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("R_{eq} = (2k + 3k) \\| 5k = 2.5k") == [("right", True)]
    assert verdicts("R_{eq} = (2k + 3k) \\| 5k = 3.5k") == [("wrong step", True)]
    assert verdicts("R = (50 / 1.5 \\cdot 10^{-6}) = 3.33 \\cdot 10^{7}") == [("right", True)]
    assert verdicts("R _ { e q } = \\left( 2k + 3k \\right) \\| 5k = 2 . 5k") == [("right", True)]    # (as the reader spaces it)


def test_a_chain_going_on_at_the_next_line():
    """'I = \\frac{V}{R}', then '= \\frac{10}{5}' and '= 2 A' on lines of their own, as a chain is often written: the lines
    that begin with '=' were dropped - one chain now; not after a datum ('R = 5', then '= 2 A': a false '5 = 2')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["I = \\frac{V}{R}", "= \\frac{10}{5}", "= 2 A"]) == [("right", True)]
    assert verdicts(["I = \\frac{V}{R}", "= \\frac{10}{5}", "= 3 A"]) == [("wrong step", True)]
    assert verdicts(["P = I^2 R", "= 2^2 \\cdot 5", "= 20 W"]) == [("right", True)]
    assert verdicts(["x = 2 + 3", "= 6"]) == [("wrong step", True)]
    assert not any(v == "wrong step" for v, _ in verdicts(["V = 10 V", "I = \\frac{V}{R}", "R = 5", "= 2 A"]))
    assert verdicts(["= 5 + 3 = 8"]) == []


def test_a_unit_the_reader_spaced_on_the_last_side():
    """'= 2 m A', '= 1 k H z', '= 1 0 m s' (the reader's spacing, no \\mathrm): its letters were letters of the problem - no
    verdict; a unit now, when they join into one ('= 6 a b' stays letters)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = \\frac{10}{5000} = 2 m A") == [("right", True)]
    assert verdicts("I = \\frac{10}{5000} = 3 m A") == [("wrong step", True)]
    assert verdicts("f = \\frac{1}{10^{-3}} = 1 k H z") == [("right", True)]
    assert verdicts("\\tau = R C = 1 k \\cdot 1 0 \\mu = 1 0 m s") == [("right", True)]
    assert verdicts("x = 2 \\cdot 3 = 6 a b") == []


def test_octal_numbers_tesla_and_a_milli_before_brackets():
    """'(17)_8 = 15': '(17)' was taken for an exercise's number and cut off - no verdict; tesla and weber are units ('= 40
    \\mu T', '= 5 mWb' after a B or a flux), a bare T only for a B ('t_1 = 6T' is periods); '-4m (1k \\| 25k)': milli before
    a bracket of numbers"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("(17)_8 = 15") == [("right", True)] and verdicts("(17)_8 = 17") == [("wrong step", True)]
    assert verdicts("2) x = 3 + 4 = 7") == [("right", True)]
    assert verdicts("B = \\frac{4\\pi \\cdot 10^{-7} \\cdot 10}{2\\pi \\cdot 0.05} = 40 \\mu T") == [("right", True)]
    assert verdicts("B = \\frac{4\\pi \\cdot 10^{-7} \\cdot 10}{2\\pi \\cdot 0.05} = 50 \\mu T") == [("wrong step", True)]
    assert verdicts("\\Phi = B A = 0.5 \\cdot 0.01 = 5 mWb") == [("right", True)]
    assert verdicts("t_1 = 2 \\cdot 3 = 6T") == []
    assert verdicts("A_v = -g_m (R_D \\| r_o) = -4m (1k \\| 25k) = -3.85") == [("right", True)]


def test_final_value_theorem_and_a_delayed_step():
    """'f(\\infty) = \\lim_{s \\to 0} s F(s) = 1' under 'F(s) = \\frac{5}{s(s+5)}': the limit was left unworked once F was put
    in - no verdict; '\\mathcal{L}\\{u(t-2)\\} = \\frac{e^{-2s}}{s}': the parser's u times (t - 2) is a delayed step"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    f = "F(s) = \\frac{5}{s(s+5)}"
    assert verdicts([f, "f(\\infty) = \\lim_{s \\to 0} s F(s) = 1"]) == [("right", True)]
    assert verdicts([f, "f(\\infty) = \\lim_{s \\to 0} s F(s) = 2"]) == [("wrong step", True)]
    assert verdicts([f, "f(0^+) = \\lim_{s \\to \\infty} s F(s) = 0"]) == [("right", True)]
    assert verdicts(["\\mathcal{L}\\{u(t-2)\\} = \\frac{e^{-2s}}{s}"]) == [("right", True)]
    assert verdicts(["\\mathcal{L}\\{u(t-2)\\} = \\frac{e^{-3s}}{s}"]) == [("wrong step", True)]


def test_twos_complement_with_its_base_written():
    """'-5 = (11111011)_2': with the base written the number was read unsigned (251) - a false sure wrong step; two's
    complement in as many bits as written, as without the base"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("-5 = (11111011)_2") == [("right", True)]
    assert verdicts("-5 = (11111010)_2") == [("wrong step", True)]
    assert verdicts("(11111011)_2 = 251") == [("right", True)]


def test_a_system_in_a_matrix_environment():
    """'\\begin{matrix} 2x - y = 1 \\\\ -x + 3y = 2 \\end{matrix}' (a system the reader writes in a plain matrix): it was cut at
    its '=' signs into garbage sides - two false sure wrong steps; its rows are equations now, as in a cases or an array"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    system = "\\begin{matrix} 2x - y = 1 \\\\ -x + 3y = 2 \\end{matrix}"
    assert verdicts([system, "x = 1", "y = 1"]) == [("right", True)] * 2
    assert verdicts(["\\left\\{ " + system + " \\right.", "x = 1", "y = 2"]) == [("right", True), ("wrong", True)]


def test_angles_written_with_their_mark_are_not_read_in_degrees_again():
    """'\\theta_v = 30^\\circ', '\\theta_i = -30^\\circ', 'P = \\frac{V_m I_m}{2}\\cos(\\theta_v - \\theta_i)': the angles are in radians
    already - read in degrees again (cos of 1.05 degrees, nearly 1) a wrong 'P = 10 W' was judged right, for sure"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    f = "P = \\frac{V_m I_m}{2}\\cos(\\theta_v - \\theta_i)"
    assert verdicts([f, "V_m = 10, I_m = 2, \\theta_v = 30^\\circ, \\theta_i = -30^\\circ", "P = 10 W"]) == [("wrong", True)]
    assert verdicts([f, "V_m = 10, I_m = 2, \\theta_v = 30^\\circ, \\theta_i = -30^\\circ", "P = 5 W"]) == [("right", True)]
    assert verdicts([f, "V_m = 10, I_m = 2, \\theta_v = 30, \\theta_i = -30", "P = 5 W"]) == [("right", True)]


def test_a_false_step_beside_a_right_answer_and_a_value_against_the_chain_above():
    """'I = 2' right under 'I = \\frac{V}{R}', then 'P = 2 \\cdot 10 = 30': a false step beside a right answer was dropped -
    the problem was just 'right'; '\\frac{1}{R} = ... = \\frac{1}{2}', then 'R = 3': the value against what the chain gives;
    '20\\log(100) = 60 dB': a side in dB already is no ratio ('40 is not 60')"""
    from grader.selfcheck import check_statements, statements

    def checks(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]

    def verdicts(lines):
        return [(v, s) for v, s, _ in checks(lines)]
    base = ["I = \\frac{V}{R}", "V = 10, R = 5", "I = 2"]
    assert verdicts(base + ["P = 2 \\cdot 10 = 30"]) == [("right", True), ("wrong step", True)]
    assert verdicts(base + ["P = 2 \\cdot 10 = 20"]) == [("right", True)]
    got = checks(["G = \\frac{V_o}{V_i}", "V_o = 10, V_i = 0.1", "G = 100", "G_{dB} = 20\\log(100) = 60 dB"])
    assert [g[:2] for g in got] == [("right", True), ("wrong step", True)] and got[1][2].endswith("40 is not 60")
    assert verdicts(["\\frac{1}{R} = \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2}", "R = 3"]) == [("right", True), ("wrong step", True)]
    assert verdicts(["\\frac{1}{R} = \\frac{1}{6} + \\frac{1}{3} = \\frac{1}{2}", "R = 2"]) == [("right", True)]
    assert verdicts(["c^2 = 3^2 + 4^2 = 25", "c = 25"]) == [("right", True), ("wrong step", True)]
    assert verdicts(["R = R_1 + R_2 = 5", "R = 5 k\\Omega"]) == []


def test_the_negative_root_of_a_size_is_not_sure():
    """'c = -5' for 'a^2 + b^2 = c^2' with a = 3, b = 4: a root of the equation, but the formula leaves c's sign free and
    a length (a current's size) is positive - right, not sure (it was sure); an equation's own negative root stays sure"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c = -5"]) == [("right", False)]
    assert verdicts(["a^{2}+b^{2}=c^{2} \\vert a=3, b=4", "c = 5"]) == [("right", True)]
    assert verdicts(["x^2 + x - 6 = 0", "x = -3"]) == [("right", True)]


def test_the_angle_and_the_parts_of_a_complex_letter():
    """'\\angle Z = 53.13^\\circ', '\\Re\\{Z\\} = 3', 'Im(I) = -1' after 'Z = 3 + j4', 'I = 2\\angle -30^\\circ': no verdict before
    (the parser read R times e times Z) - worked out with the letter's value; an angle in degrees with or without the mark
    (or radians), and the message in degrees"""
    from grader.selfcheck import check_statements, statements

    def checks(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]

    def verdicts(lines):
        return [(v, s) for v, s, _ in checks(lines)]
    for written in ("53.13^\\circ", "53.13", "0.927"):
        assert verdicts(["Z = 3 + j4", "\\angle Z = " + written]) == [("right", True)], written
    got = checks(["Z = 3 + j4", "\\angle Z = 60^\\circ"])
    assert [g[:2] for g in got] == [("wrong step", True)] and got[0][2].endswith("53.13° is not 60°")
    assert verdicts(["Z = -3 - j4", "\\angle Z = 233.13"]) == [("right", True)]
    assert verdicts(["Z = 3 + j4", "\\Re\\{Z\\} = 3"]) == [("right", True)]
    assert verdicts(["Z = 3 + j4", "\\Re\\{Z\\} = 4"]) == [("wrong step", True)]
    assert verdicts(["I = 2 \\angle -30^\\circ", "Im(I) = -1"]) == [("right", True)]


def test_a_bare_answer_a_power_of_1000_off_is_for_a_human():
    """'I = \\frac{V}{R}', 'I = 5' with V = 10, R = 2 kΩ: 5 for 5 mA, the unit left out - or a slip of 1000: wrong, not sure
    (it was a sure wrong); with its unit written ('I = 5 A') or another number ('I = 7') a sure wrong"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines, values=None):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st, values=values)]
    vals = {"R": "2k", "V": "10"}
    assert verdicts(["I = \\frac{V}{R}", "I = 5"], vals) == [("wrong", False)]
    assert verdicts(["I = \\frac{V}{R}", "I = 5 A"], vals) == [("wrong", True)]
    assert verdicts(["I = \\frac{V}{R}", "I = 7"], vals) == [("wrong", True)]
    assert verdicts(["I = \\frac{V}{R}", "I = 5 mA"], vals) == [("right", True)]


def test_a_slash_before_a_product():
    """'X_C = 1/2\\pi \\cdot 50 \\cdot 10^{-4} = 31.8': students mean 1 over the whole product; the parser, by the rules, (1/2)
    \\pi \\cdot 50 \\cdot 10^{-4} - a false sure wrong step. Either reading holds now; a slash before a single factor reads one way
    only, before a sum too - but right only with the brackets the student left out ('10 / 2 + 3 = 2': 10/(2+3)) is a wrong
    step not sure, for a human"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("X_C = 1/2\\pi f C = 1/2\\pi \\cdot 50 \\cdot 10^{-4} = 31.8") == [("right", True)]
    assert verdicts("X_C = 1/2\\pi \\cdot 50 \\cdot 10^{-4} = 41.8") == [("wrong step", True)]
    assert verdicts("T = 1/2 \\cdot 4 = 2") == [("right", True)]
    assert verdicts("x = 10 / 2 + 3 = 8") == [("right", True)]
    assert verdicts("x = 10 / 2 + 3 = 2") == [("wrong step", False)] and verdicts("x = 10 / 2 + 3 = 3") == [("wrong step", True)]


def test_a_greek_mu_as_micro():
    """'10 μF' with the Greek letter (as typed or read) - only the micro sign was a prefix: no verdict; a friction's 'μ N'
    stays a letter"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("\\tau = R C = 1 kΩ \\cdot 10 μF = 10 ms") == [("right", True)]
    assert verdicts("\\tau = R C = 1 kΩ \\cdot 10 μF = 20 ms") == [("wrong step", True)]
    assert verdicts("F = μ N = 0.3 \\cdot 10 = 3") == [("right", True)]


def test_typed_infinity():
    """'v(∞) = 10', '\\lim_{t → ∞}', '\\int_0^∞' as typed: no verdict - read as \\infty"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["v(t) = 10(1 − e^{−t})", "v(∞) = 10"]) == [("right", True)]
    assert verdicts(["v(t) = 10(1 − e^{−t})", "v(∞) = 5"]) == [("wrong step", True)]
    assert verdicts(["\\lim_{t → ∞} e^{-t} = 0"]) == [("right", True)]
    assert verdicts(["\\int_0^∞ e^{-2t} dt = 0.5"]) == [("right", True)]


def test_a_unit_after_a_divisors_number():
    """'I = \\frac{V}{R} = 12 / 4 k\\Omega = 3 mA', '\\lambda = c/f = 3 \\cdot 10^8 / 100 MHz = 3 m': the unit is the divisor's
    number's - it was taken for the whole side's ((12/4) kΩ): a false sure wrong step. '= 10 / 2 mV = 5 mV' reads the
    whole side's still, and a result written bare 1000 apart ('= 3' for 3 mA) is not a sure wrong"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = \\frac{V}{R} = 12 / 4 k\\Omega = 3 mA") == [("right", True)]
    assert verdicts("I = \\frac{V}{R} = 12 / 4 k\\Omega = 4 mA") == [("wrong step", True)]
    assert verdicts("\\lambda = c/f = 3 \\cdot 10^8 / 100 MHz = 3 m") == [("right", True)]
    assert verdicts("f = 1/T = 1 / 2 ms = 500 Hz") == [("right", True)]
    assert verdicts("R = V/I = 12 / 4 mA = 3 k\\Omega") == [("right", True)]
    assert verdicts("V = 10/2 mV = 5 mV") == [("right", True)]
    assert verdicts("I = 12 / 4 k\\Omega = 3") == [("wrong step", False)]
    assert verdicts("V = I R = 2 \\cdot 5 k\\Omega = 10 kV") == [("right", True)]


def test_gamma_is_a_letter():
    """'\\Gamma' (a reflection coefficient), '\\gamma' (a propagation constant) were read as Euler's constant 0.577 - 'VSWR =
    \\frac{1 + |\\Gamma|}{1 - |\\Gamma|} = \\frac{1.333}{0.667}' a false sure wrong step - and '\\Gamma_L' was not read at all;
    the gamma function stays ('\\Gamma(5) = 4!')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["VSWR = \\frac{1 + |\\Gamma|}{1 - |\\Gamma|} = \\frac{1.333}{0.667} = 2"]) == [("right", True)]
    assert verdicts(["\\Gamma = 0.333", "VSWR = \\frac{1 + |\\Gamma|}{1 - |\\Gamma|} = 3"]) == [("wrong step", True)]
    assert verdicts(["Z_L = 100", "Z_0 = 50", "\\Gamma_L = \\frac{Z_L - Z_0}{Z_L + Z_0} = 0.333"]) == [("right", True)]
    assert verdicts(["Z_L = 100", "Z_0 = 50", "\\Gamma_L = \\frac{Z_L - Z_0}{Z_L + Z_0} = 0.5"]) == [("wrong step", True)]
    assert verdicts(["\\alpha = 0.1", "\\beta = 2", "\\gamma = \\alpha + j\\beta = 0.1 + j2"]) == [("right", True)]
    assert verdicts(["\\Gamma(5) = 4! = 24"]) == [("right", True), ("right", True)]


def test_a_root_written_twice():
    """'r_1 = -1', 'r_2 = -1' for r^2 + 3r + 2 = 0: each a root - both were right, -2 missing; the second is wrong now (a
    double root may be written twice); roots named apart with the data written before the equation are checked too, a
    reading right only past the prefixes written for a human"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["r^2 + 3r + 2 = 0", "r_1 = -1", "r_2 = -1"]) == [("right", True), ("wrong", True)]
    assert verdicts(["r^2 + 2r + 1 = 0", "r_1 = -1", "r_2 = -1"]) == [("right", True), ("right", True)]
    assert verdicts(["s^2 + 4 = 0", "s_1 = j2", "s_2 = j2"]) == [("right", True), ("wrong", True)]
    assert verdicts(["x^3 - 6x^2 + 11x - 6 = 0", "x_1 = 1", "x_2 = 2", "x_3 = 3"]) == [("right", True)] * 3
    rlc = ["R = 2 \\Omega", "L = 1 mH", "C = 0.5 mF", "s^2 + \\frac{R}{L}s + \\frac{1}{LC} = 0"]
    assert verdicts(rlc + ["s_1 = -1000 + j1000", "s_2 = -1000 - j1000"]) == [("right", True)] * 2
    assert verdicts(rlc + ["s_1 = -1000 + j1000", "s_2 = -1000 + j1000"]) == [("right", True), ("wrong", True)]
    assert verdicts(rlc + ["s_1 = -1 + j", "s_2 = -1 - j"]) == [("right", False)] * 2


def test_a_computation_in_degrees_with_its_result_marked():
    """'PM = 180 - 135 = 45^\\circ', '\\theta_a = \\frac{180}{3} = 60^\\circ': numbers in degrees, the result marked - the mark
    made it radians (0.785): a false sure wrong step; a trig function's value or pi stays in radians ('\\arctan(1) =
    0.785^\\circ' is wrong), and a wrong one says it in degrees"""
    from grader.selfcheck import check_statements, statements

    def checks(t):
        return [(c.verdict, c.sure, c.text) for c in check_statements(statements(t))]
    assert [c[:2] for c in checks("PM = 180 + \\angle G(j\\omega_c) = 180 - 135 = 45^\\circ")] == [("right", True)]
    assert [c[:2] for c in checks("\\theta_a = \\frac{180}{3} = 60^\\circ")] == [("right", True)]
    got = checks("PM = 180 - 135 = 50^\\circ")
    assert [c[:2] for c in got] == [("wrong step", True)] and "45° is not 50°" in got[0][2]
    assert [c[:2] for c in checks("\\theta = \\arctan(1) = 45^\\circ")] == [("right", True)]
    assert [c[:2] for c in checks("\\theta = \\arctan(1) = 0.785^\\circ")] == [("wrong step", True)]
    assert [c[:2] for c in checks("\\theta = \\frac{\\pi}{4} = 0.785^\\circ")] == [("wrong step", True)]


def test_units_as_students_spell_them():
    """'4 K\\Omega' (a capital K), '2 kohm', '0.02 sec', '50 hz', '40 db' after a number: units - they were letters of the
    problem (no verdict); an integral's 'db' stays its differential, a kelvin stays"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = \\frac{12 V}{4 K\\Omega} = 3 mA") == [("right", True)]
    assert verdicts("I = \\frac{12 V}{4 K\\Omega} = 4 mA") == [("wrong step", True)]
    assert verdicts("T = \\frac{1}{50} = 20 msec") == [("right", True)]
    assert verdicts("A = 20\\log(100) = 40 db") == [("right", True)]
    assert verdicts("f = \\frac{1}{2\\pi \\sqrt{10^{-3} \\cdot 10^{-6}}} = 5.03 khz") == [("right", True)]
    assert verdicts("R = 2 kohm + 3 kohm = 6 kohm") == [("wrong step", True)]
    assert verdicts("\\int_0^1 2db = 2") == [("right", True)]


def test_a_unit_after_a_bracket_and_a_complex_value_with_a_prefix():
    """'12 / (2 + 2) k\\Omega', '2 mA \\cdot (1 + 2) k\\Omega': a unit after a bracket of numbers is the bracket's (it was letters:
    no verdict); a complex value with a prefixed unit last ('= 2 \\angle -53.13^\\circ mA') is in SI units too - it was
    compared as written; a lone '1.6 mA' inside 'j1.6 mA' is no number of its own"""
    from grader.selfcheck import check_statements, statements

    def checks(t):
        return [(c.verdict, c.sure, c.text) for c in check_statements(statements(t))]
    assert [c[:2] for c in checks("I = 12 / (2 + 2) k\\Omega = 3 mA")] == [("right", True)]
    assert [c[:2] for c in checks("V = 2 mA \\cdot (1 + 2) k\\Omega = 6 V")] == [("right", True)]
    assert [c[:2] for c in checks("V = 2 mA \\cdot (1 + 2) k\\Omega = 9 V")] == [("wrong step", True)]
    assert [c[:2] for c in checks("I = 10 / (3 + j4) k\\Omega = 2 \\angle -53.13^\\circ mA")] == [("right", True)]
    assert [c[:2] for c in checks("I = 10 / (3 + j4) k\\Omega = 1.2 - j1.6 mA")] == [("right", True)]
    got = checks("I = 10 / (3 + j4) k\\Omega = 2 - j1 mA")
    assert [c[:2] for c in got] == [("wrong step", True)] and "1.2 - j1.6 mA is not 2 - j1" in got[0][2]
    assert [c[:2] for c in checks("I = \\frac{10}{3 + j4} = 2 \\angle -53.13^\\circ mA")] == [("right", True)]
    assert [c[:2] for c in checks("V = (V_1 - V_2) A")] == []


def test_units_in_an_exponent_divide_out():
    """'i = 2(1 - e^{-2 ms / 1 ms}) = 1.73 mA': the units in the exponent cancel - the 2 is a bare number (2 mA): right, not
    sure, as for bare numbers 1000 apart (it was a sure wrong step); '10 e^{-1 ms/1 ms}' is read (no verdict before)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("i = 2(1 - e^{-2 ms / 1 ms}) = 1.73 mA") == [("right", False)]
    assert verdicts("v(t) = 10 e^{-t/\\tau} = 10 e^{-1 ms/1 ms} = 3.68 V") == [("right", True)]
    assert verdicts("v(t) = 10 e^{-t/\\tau} = 10 e^{-1 ms/1 ms} = 4.68 V") == [("wrong step", True)]


def test_right_only_as_written_past_a_prefix_written_above():
    """'R = 5 k\\Omega', then 'I = \\frac{10}{R} = 2 A': right with the numbers as written (10/5), but with its unit written
    the result is 2 mA - the prefix left out? right, not sure (it was sure); '= 2 mA', '= 0.002 A' and a bare '= 2' stay
    sure; a complex result likewise, and a complex result with a prefix in its unit compared in that unit ('= 1.2 - j1.6 \\mu A'
    for 1.2 - j1.6 mA was right: the tolerance of its decimals was taken in amperes)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["R = 5 k\\Omega", "I = \\frac{10}{R} = 2 mA"]) == [("right", True)]
    assert verdicts(["R = 5 k\\Omega", "I = \\frac{10}{R} = 2 A"]) == [("right", False)]
    assert verdicts(["R = 5 k\\Omega", "I = \\frac{10}{R} = 0.002 A"]) == [("right", True)]
    assert verdicts(["R = 5 k\\Omega", "I = \\frac{10}{R} = 2"]) == [("right", True)]
    assert verdicts(["Z = 3 + j4 k\\Omega", "I = \\frac{10}{Z} = 1.2 - j1.6 mA"]) == [("right", True)]
    assert verdicts(["Z = 3 + j4 k\\Omega", "I = \\frac{10}{Z} = 1.2 - j1.6 A"]) == [("right", False)]
    assert verdicts(["I = 10 / (3 + j4) k\\Omega = 1.2 - j1.6 \\mu A"]) == [("wrong step", True)]


def test_an_item_number_with_a_dot_and_an_equations_label():
    """'1. I = \\frac{10}{5} = 2 A': the item's number was read into the side ('1.I = 10/5': a false sure wrong step); an
    equation's label after a unit ('= 2 A (1)') left the side unread; a decimal ('1.5 = \\frac{3}{2}') and a product ('2 (3)')
    stay"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("1. I = \\frac{10}{5} = 2 A") == [("right", True)]
    assert verdicts("2. \\frac{10}{5} = 2") == [("right", True)]
    assert verdicts("1.5 = \\frac{3}{2}") == [("right", True)]
    assert verdicts("I = \\frac{10}{5} = 2 A (1)") == [("right", True)]
    assert verdicts("I = \\frac{10}{5} = 3 A (1)") == [("wrong step", True)]
    assert verdicts("I = 2 (3) = 6") == [("right", True)]


def test_a_value_written_as_a_quotient_with_its_units():
    """'I = 12 / 4 k\\Omega' alone was I = 3 kΩ (the unit taken for the whole side's): every formula after it a false sure
    wrong ('P = I^2 R ... = 36 mW'); a current is no kΩ - the unit is the divisor's, the value 3 mA. 'I = \\frac{12}{4 k\\Omega}',
    'V = 2 mA \\cdot 5 k\\Omega' (numbers with their units) were formulas in the units' letters: worked out in SI units now"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["I = 12 / 4 k\\Omega", "P = I^2 R", "R = 4 k\\Omega", "P = 36 mW"]) == [("right", True)]
    assert verdicts(["I = 12 / 4 k\\Omega", "V = I \\cdot 4 k\\Omega = 12 V"]) == [("right", True)]
    assert verdicts(["I = 12 / 4 k\\Omega", "V = I \\cdot 4 k\\Omega = 16 V"]) == [("wrong step", True)]
    assert verdicts(["f = 1 / 2 ms", "T = 1/f = 2 ms"]) == [("right", True)]
    assert verdicts(["V = 10/2 mV", "I = \\frac{V}{5 \\Omega} = 1 mA"]) == [("right", True)]
    assert verdicts(["I = \\frac{12}{4 k\\Omega}", "P = I^2 \\cdot 4 k\\Omega = 36 mW"]) == [("right", True)]
    assert verdicts(["V = 2 mA \\cdot 5 k\\Omega", "P = \\frac{V^2}{5 k\\Omega} = 20 mW"]) == [("right", True)]


def test_units_miscased_in_a_final_answer():
    """'3 ma' for 3 mA was 'unit_mismatch' - 0 points, graded automatically (pint reads milli-years); '40 db' likewise, '50 hz'
    and '4.7 K\\Omega' were unparsable: the unit as meant now, when it fits the key's"""
    from grader.numeric import compare_numeric
    assert compare_numeric("3 ma", "3 mA").status == "correct"
    assert compare_numeric("3 ma", "4 mA").status == "wrong"
    assert compare_numeric("40 db", "40 dB").status == "correct"
    assert compare_numeric("50 hz", "50 Hz").status == "correct"
    assert compare_numeric("4.7 K\\Omega", "4.7 k\\Omega").status == "correct"
    assert compare_numeric("20 msec", "20 ms").status == "correct"
    assert compare_numeric("2 ms", "2 mA").status == "unit_mismatch"


def test_a_range_is_no_value():
    """'f = 20 Hz - 20 kHz', '20 Hz \\sim 20 kHz': a range - it was f = -19980 Hz (or '20 Hz = 20 kHz'), every formula after
    it a false sure wrong; a difference stays one ('12 V - 0.7 V = 11.3 V', '0.5 mA - 2 mA = -1.5 mA')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["f = 20 Hz - 20 kHz", "T = \\frac{1}{f} = 50 ms"]) == []
    assert verdicts(["f = 20 Hz \\sim 20 kHz", "T = \\frac{1}{f} = 50 ms"]) == []
    assert verdicts(["V = 12 V - 0.7 V = 11.3 V"]) == [("right", True)]
    assert verdicts(["I = 0.5 mA - 2 mA = -1.5 mA"]) == [("right", True)]
    assert verdicts(["I = 2 mA - 0.5 mA = 2.5 mA"]) == [("wrong step", True)]
    assert verdicts(["I = 1 mA - 2 mA + 3 mA = 2 mA"]) == [("right", True)]


def test_electron_volts():
    """'1 eV = 1.6 \\cdot 10^{-19} J', 'E = 2 eV = 3.2 \\cdot 10^{-19} J', 'keV': a unit (semiconductor courses) - no verdict before
    (e V: Euler's number times volts)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("1 eV = 1.6 \\cdot 10^{-19} J") == [("right", True)]
    assert verdicts("E = 2 eV = 3.2 \\cdot 10^{-19} J") == [("right", True)]
    assert verdicts("E = 2 eV = 4.2 \\cdot 10^{-19} J") == [("wrong step", True)]
    assert verdicts("E = 1 keV = 1.6 \\cdot 10^{-16} J") == [("right", True)]


def test_a_unit_after_a_divisors_number_against_the_circuit():
    """the circuit checks read 'I = 12 / 6 k\\Omega = 2 mA' as (12/6) kΩ - a false sure wrong against the circuit's 2 mA; the
    quantity's name decides now (a current is no kΩ), and 'I = 12 / 6 mA' (a current in mA) stays the whole side's"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12", label="E"),
                   C("R1", "resistor", ["n0", "a"], value="4k", label="R_1"), C("R2", "resistor", ["a", "0"], value="2k", label="R_2")],
                  ground="0", node_labels={"a": "A"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("I = 12 / 6 k\\Omega = 2 mA") == [("right", True)]
    assert checks("I = 12 / 6 k\\Omega") == [("right", True)]
    assert checks("I = 12 / 6 k\\Omega = 3 mA") == [("wrong", True)]
    assert checks("I = (12 - 0) / 6 k\\Omega = 2 mA") == [("right", True)]
    assert checks("I = 12 / 6 mA") == [("right", True)]
    assert checks("R_{eq} = 12 / 2 mA = 6 k\\Omega") == [("right", True)]


def test_a_unit_after_a_phasor():
    """'10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega = 2 \\angle -53.13^\\circ mA', '2 \\angle 30^\\circ mA \\cdot 5 \\angle
    20^\\circ k\\Omega': the unit after a phasor is the phasor's (it was letters: no verdict); wrong ones caught"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = 10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega = 2 \\angle -53.13^\\circ mA") == [("right", True)]
    assert verdicts("I = 10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega = 2 \\angle 53.13^\\circ mA") == [("wrong step", True)]
    assert verdicts("I = 10 \\angle 0^\\circ / 5 \\angle 53.13^\\circ k\\Omega = 2 \\angle -53.13^\\circ A") == [("wrong step", True)]
    assert verdicts("V = 2 \\angle 30^\\circ mA \\cdot 5 \\angle 20^\\circ k\\Omega = 10 \\angle 50^\\circ V") == [("right", True)]
    assert verdicts("V = 2 \\angle 30^\\circ mA \\cdot 5 \\angle 20^\\circ k\\Omega = 10 \\angle 10^\\circ V") == [("wrong step", True)]


def test_a_hebrew_and_between_statements():
    """'R_1 = 2k \\text{ ו- } R_2 = 3k': a one-letter Hebrew 'and' was no words between statements (two letters at least), and
    its hyphen became a minus ('- R_2 = 3k'); 'ה-V_A = 5' (the Hebrew article) likewise"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert [(s.kind, s.side_texts) for s in statements("R_1 = 2k \\text{ ו- } R_2 = 3k")] == [("value", ["R_1", "2"]), ("value", ["R_2", "3"])]
    assert [(s.kind, s.side_texts) for s in statements("ה-V_A = 5")] == [("value", ["V_A", "5"])]
    assert verdicts(["R_1 = 2k \\text{ ו- } R_2 = 3k", "R = R_1 + R_2 = 5k"]) == [("right", True)]
    assert verdicts(["R_1 = 2k ו- R_2 = 3k", "R = R_1 + R_2 = 6k"]) == [("wrong step", True)]


def test_resistor_codes_and_a_values_kind():
    """'R = 4k7' (4.7 kΩ, a resistor's code), '10 V DC', '10 V_{rms}', '10 V (rms)', '10 V_{pp}' after a number: the value as
    written - they were letters of the problem (no verdict); a name stays ('V_p = \\sqrt{2} V_{rms}')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["R = 4k7", "I = \\frac{12}{R} = 2.55 mA"]) == [("right", True)]
    assert verdicts(["R = 4k7", "I = \\frac{12}{R} = 3.55 mA"]) == [("wrong step", True)]
    assert verdicts(["V = 10 V DC", "I = \\frac{V}{5} = 2"]) == [("right", True)]
    assert verdicts(["V_s = 10 V_{rms}", "I = \\frac{V_s}{5} = 2"]) == [("right", True)]
    assert verdicts(["V_{rms} = 10 V", "V_p = \\sqrt{2} V_{rms} = 14.1 V"]) == [("right", True)]


def test_radians_as_a_unit():
    """'\\theta = \\tan^{-1}(1) = 0.785 rad': rad is a unit (it was letters: no verdict) - and a result written in radians is
    not read in degrees ('= 45 rad' is wrong; '= 45' bare is the arctan in degrees)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["\\theta = \\tan^{-1}(1) = 0.785 rad"]) == [("right", True)]
    assert verdicts(["\\theta = \\tan^{-1}(1) = 45 rad"]) == [("wrong step", True)]
    assert verdicts(["\\theta = \\tan^{-1}(1) = 45"]) == [("right", True)]
    assert verdicts(["\\phi = 30^\\circ = 0.524 rad"]) == [("right", True)]
    assert verdicts(["\\phi = \\tan^{-1}(\\frac{X}{R})", "X = 10", "R = 10", "\\phi = 45 rad"]) == [("wrong", True)]
    assert verdicts(["\\omega = 2\\pi \\cdot 50 = 314 rad/s"]) == [("right", True)]


def test_a_lone_x_between_numbers_is_a_times_sign():
    """'P = 2 x 5 = 10', '2 mA x 5 k\\Omega': a lone x with spaces between two numbers (or after a number's unit) is a times
    sign - it was the letter x (no verdict); '3x = 12' keeps its x"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("P = 2 x 5 = 10") == [("right", True)]
    assert verdicts("P = 2 x 5 = 11") == [("wrong step", True)]
    assert verdicts("V = 2 mA x 5 k\\Omega = 10 V") == [("right", True)]
    assert verdicts("3x = 12 \\Rightarrow x = 4") == [("right", True)]


def test_a_tolerance_is_the_nominal_value():
    """'R = 100 \\Omega \\pm 5\\%', '5 V \\pm 2\\%', '10 V \\pm 0.1 V': a tolerance - it was two values named apart as roots
    (100.05 and 99.95), R itself none; roots stay ('s = -1 \\pm j3', 'x_{1,2} = \\frac{-3 \\pm 1}{2}')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert [(s.symbol, s.number) for s in statements("R = 100 \\Omega \\pm 5\\%")] == [(statements("R = 1")[0].symbol, 100.0)]
    assert verdicts(["R = 100 \\Omega \\pm 5\\%", "I = \\frac{10}{R} = 0.1 A"]) == [("right", True)]
    assert verdicts(["V = 10 V \\pm 0.1 V", "I = \\frac{V}{5} = 3 A"]) == [("wrong step", True)]
    assert len(statements("s = -1 \\pm j3")) == 2 and len(statements("x_{1,2} = \\frac{-3 \\pm 1}{2}")) == 2


def test_a_systems_answers_in_other_units():
    """mesh equations in SI ('12 = 2000 I_1 + ...') answered 'I_1 = 4.36 mA' were a false sure wrong (4.36 against 0.00436);
    in kΩ ('12 = 2 I_1 + ...', so mA) answered in amperes ('0.00436 A') likewise - right, not sure, now; a bare answer a power
    of 1000 off is wrong, not sure"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    si = ["12 = 2000 I_1 + 1000(I_1 - I_2)", "0 = 1000(I_2 - I_1) + 3000 I_2"]
    k = ["12 = 2 I_1 + 1(I_1 - I_2)", "0 = 1(I_2 - I_1) + 3 I_2"]
    assert verdicts(si + ["I_1 = 4.36 mA", "I_2 = 1.09 mA"]) == [("right", True)] * 2
    assert verdicts(si + ["I_1 = 0.00436", "I_2 = 0.00109"]) == [("right", True)] * 2
    assert verdicts(si + ["I_1 = 4.36", "I_2 = 1.09"]) == [("wrong", False)] * 2
    assert verdicts(k + ["I_1 = 4.36 mA", "I_2 = 1.09 mA"]) == [("right", True)] * 2
    assert verdicts(k + ["I_1 = 0.00436 A", "I_2 = 0.00109 A"]) == [("right", False)] * 2
    assert verdicts(k + ["I_1 = 5.36 mA", "I_2 = 1.09 mA"]) == [("wrong", True), ("right", True)]


def test_an_equation_with_units_in_it():
    """'12 V = 2 k\\Omega \\cdot I_1 + 1 k\\Omega (I_1 - I_2)', '10 V = I \\cdot 2 k\\Omega + 4 V', 'V = I \\cdot 5 k\\Omega': the
    numbers' units in SI (the units' letters were unknowns: 'cannot check') when one unit at least is no letter's spelling
    (kΩ, mA); the answers after it compared in SI ('I = 3 mA' for 0.003), and no unit note from coefficients whose units
    are gone; '3 V I' (3 times V times I) stays a formula in letters"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, x in enumerate(lines):
            st += statements(x, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    mesh = ["12 V = 2 k\\Omega \\cdot I_1 + 1 k\\Omega (I_1 - I_2)", "0 = 1 k\\Omega (I_2 - I_1) + 3 k\\Omega \\cdot I_2"]
    assert verdicts(mesh + ["I_1 = 4.36 mA", "I_2 = 1.09 mA"]) == [("right", True)] * 2
    assert verdicts(mesh + ["I_1 = 4.36 mA", "I_2 = 2.09 mA"]) == [("right", True), ("wrong", True)]
    assert verdicts(["10 V = I \\cdot 2 k\\Omega + 4 V", "I = 3 mA"]) == [("right", True)]
    assert verdicts(["10 V = I \\cdot 2 k\\Omega + 4 V", "I = 4 mA"]) == [("wrong", True)]
    assert verdicts(["V = I \\cdot 5 k\\Omega", "I = 2 mA", "V = 10 V"]) == [("right", True)]
    assert verdicts(["V_A = 5 mA \\cdot R_2", "R_2 = 2 k\\Omega", "V_A = 10 V"]) == [("right", True)]
    assert [s.kind for s in statements("P = 3 V I")] == ["formula"]


def test_a_bare_step_against_the_circuit():
    """'P_{R1} = 2^2 \\cdot 4 = 16 mW': the step in bare numbers is in the units kept (mA, kΩ - so mW) - it was 16 against
    the circuit's 0.016 W, a false sure wrong; a step in SI units stays, a false step is still caught"""
    from grader.circuit import Component as C, Netlist
    from grader.circuit_check import check_against, circuit_statements
    net = Netlist([C("V1", "voltage_source", ["n0", "0"], plus="n0", value="12", label="E"),
                   C("R1", "resistor", ["n0", "a"], value="4k", label="R_1"), C("R2", "resistor", ["a", "0"], value="2k", label="R_2")],
                  ground="0", node_labels={"a": "A"})

    def checks(tex):
        return [(c.verdict, c.sure) for c in check_against(circuit_statements(tex), net)]
    assert checks("P_{R1} = 2^2 \\cdot 4 = 16 mW") == [("right", True)]
    assert checks("P_{R1} = 0.002^2 \\cdot 4000 = 16 mW") == [("right", True)]
    assert checks("P_{R1} = 2^2 \\cdot 5 = 16 mW") == [("wrong", True)]
    assert checks("P_{R1} = 2^2 \\cdot 5 = 20 mW") == [("wrong", True)]


def test_a_bare_nano_in_a_wrong_step():
    """'\\tau = 10k \\cdot 100n = 2 ms': a bare n (nano) was read only where the step held - a wrong one got no verdict (n a
    letter); n and p are nano and pico either way now (a bare m stays milli only where the step holds: metres?)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("\\tau = 10k \\cdot 100n = 1 ms") == [("right", True)]
    assert verdicts("\\tau = 10k \\cdot 100n = 2 ms") == [("wrong step", True)]
    assert verdicts("X_C = \\frac{1}{2\\pi \\cdot 1 kHz \\cdot 100n} = 2.069 k\\Omega") == [("wrong step", True)]


def test_bare_prefixes_beside_written_units():
    """found by a random generator of EE computations: '(12 \\mu A)(2.2M)' - a bare M beside a written unit is mega; '4.7M x
    100n', '100 [\\mu A] x 12 M\\Omega', '10 \\mathrm{\\mu A} x 5 \\Omega' - x a times sign; a unit's '\\mathrm{k\\Omega}' is no
    letter k ('\\frac{1k}{2.2 \\mathrm{k\\Omega} + 1k}' read its bare k's again) - they had no verdict"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("V = (12 \\mu A)(2.2M) = 26.4 V") == [("right", True)]
    assert verdicts("V = (12 \\mu A)(2.2M) = 36.4 V") == [("wrong step", True)]
    assert verdicts("\\tau = 4.7M x 100n = 470 ms") == [("right", True)]
    assert verdicts("V = 100 [\\mu A] x 12M\\Omega = 1.2 kV") == [("right", True)]
    assert verdicts("V_{out} = 12mV \\cdot \\frac{1k}{2.2 \\mathrm{k\\Omega} + 1k} = 3.75 mV") == [("right", True)]
    assert verdicts("V_{out} = 12mV \\cdot \\frac{1k}{2.2 \\mathrm{k\\Omega} + 1k} = 4.75 mV") == [("wrong step", True)]
    assert verdicts("F = \\mu N = 0.3 \\cdot 10 = 3") == [("right", True)]


def test_a_complex_value_with_a_prefixed_unit():
    """found by a random generator of phasor computations: '= (-0.74 - j2.21) mA' for -0.53 - j1.6 mA was right - the
    tolerance of its two decimals was taken in amperes, not milliamperes (0.005 A: every milliampere value matched); the
    tolerance scales with the unit's power of ten now"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = \\frac{(8 - j8) V}{(3 + j6) k\\Omega} = (-0.53 - j1.6) mA") == [("right", True)]
    assert verdicts("I = \\frac{(8 - j8) V}{(3 + j6) k\\Omega} = (-0.74 - j2.21) mA") == [("wrong step", True)]
    assert verdicts("I = \\frac{2 \\angle 0° V}{(6 - j4) k\\Omega} = 0.28 \\angle 33.69° mA") == [("right", True)]
    assert verdicts("I = \\frac{2 \\angle 0° V}{(6 - j4) k\\Omega} = 0.36 \\angle 33.69° mA") == [("wrong step", True)]


def test_a_complex_values_unit_unbracketed():
    """found by a random generator of phasor computations: 'Z = 6 + j 8 k\\Omega = 10 \\angle 53.13° k\\Omega' was a sure wrong
    step - the kilo the j-part's alone (6 + j8000); 'Z = 3 - j4 \\Omega', '\\frac{12 + j5 V}{4 + j4 k\\Omega}', '3 \\Omega - j4
    \\Omega', '\\frac{10 V}{-j4 \\Omega}', '120 + j90 VA' had no verdict; a complex value's unit is the whole value's now -
    left unread where it is not clear whose ('2 \\cdot 3 + j 8 k\\Omega')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("Z = 6 + j 8 k\\Omega = 10 \\angle 53.13° k\\Omega") == [("right", True)]
    assert verdicts("Z = 6 + j 8 k\\Omega = 10 \\angle 63.13° k\\Omega") == [("wrong step", True)]
    assert verdicts("Z = 3 - j4 \\Omega = 5 \\angle -53.13° \\Omega") == [("right", True)]
    assert verdicts("Z = 3 \\Omega - j4 \\Omega = 5 \\angle -53.13° \\Omega") == [("right", True)]
    assert verdicts("I = \\frac{12 + j5 V}{4 + j4 k\\Omega} = 2.298 \\angle -22.38° mA") == [("right", True)]
    assert verdicts("I = \\frac{12 + j5 V}{4 + j4 k\\Omega} = 2.298 \\angle -42.38° mA") == [("wrong step", True)]
    assert verdicts("I = \\frac{10 V}{-j4 \\Omega} = j2.5 A") == [("right", True)]
    assert verdicts("I = \\frac{10 V}{-j4 k\\Omega} = j2.5 A") == [("wrong step", True)]
    assert verdicts("S = 120 + j90 VA = 150 \\angle 36.87° VA") == [("right", True)]
    assert verdicts("S = 1.2 + j0.9 kVA = 150 \\angle 36.87° VA") == [("wrong step", True)]
    assert verdicts("Z_C = \\frac{1}{j \\cdot 1000 \\cdot 250 \\cdot 10^{-9}} = -j4 k\\Omega") == [("right", True)]
    assert verdicts("Z = 2 \\cdot 3 + j 8 k\\Omega = 10 \\angle 53.13° k\\Omega") == []


def test_volt_amperes_inside_a_computation():
    """'\\frac{800 W}{1 kVA} = 0.8', '1 kVA \\cdot 0.6 = 600 VAR', '\\sqrt{(800 W)^2 + (600 VAR)^2} = 1 kVA': volt-amperes
    beside the numbers - they were letters of the problem (no verdict)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("\\cos\\phi = \\frac{P}{S} = \\frac{800 W}{1 kVA} = 0.8") == [("right", True)]
    assert verdicts("\\cos\\phi = \\frac{P}{S} = \\frac{800 W}{1 kVA} = 0.9") == [("wrong step", True)]
    assert verdicts("Q = S \\sin\\phi = 1 kVA \\cdot 0.6 = 600 VAR") == [("right", True)]
    assert verdicts("S = \\sqrt{P^2 + Q^2} = \\sqrt{(800 W)^2 + (600 VAR)^2} = 1.2 kVA") == [("wrong step", True)]
    assert verdicts("C = \\frac{Q_C}{\\omega V^2} = \\frac{600 VAR}{2\\pi \\cdot 50 Hz \\cdot (230 V)^2} = 36.1 \\mu F") == [("right", True)]


def test_semiconductor_units():
    """'= 2.25 \\cdot 10^4 \\mathrm{cm^{-3}}' was a sure wrong step - the power inside \\mathrm made the parser read the side
    as 10; densities, resistivities as semiconductor physics writes them ('cm^{-3}', '\\frac{1}{cm^3}', '\\Omega \\cdot cm')
    had no verdict; a length worked out in centimetres and written in micrometres ('= 59.1 \\mu m' for 0.00591 cm) was a
    sure wrong step - right now, not sure"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    n = "n = \\frac{n_i^2}{N_A} = \\frac{(1.5 \\cdot 10^{10})^2}{10^{16}} = "
    assert verdicts(n + "2.25 \\cdot 10^4 \\mathrm{cm^{-3}}") == [("right", True)]
    assert verdicts(n + "2.25 \\cdot 10^4 cm^{-3}") == [("right", True)]
    assert verdicts(n + "2.25 \\cdot 10^4 \\frac{1}{cm^3}") == [("right", True)]
    assert verdicts(n + "3.25 \\cdot 10^4 \\mathrm{cm^{-3}}") == [("wrong step", True)]
    rho = "\\rho = \\frac{1}{q n \\mu_n} = \\frac{1}{1.6 \\cdot 10^{-19} \\cdot 10^{16} \\cdot 1350} = "
    assert verdicts(rho + "0.463 \\Omega \\cdot cm") == [("right", True)]
    assert verdicts(rho + "0.563 \\Omega cm") == [("wrong step", True)]
    L = "L_n = \\sqrt{D_n \\tau_n} = \\sqrt{34.97 \\cdot 10^{-6}} = "
    assert verdicts(L + "5.91 \\cdot 10^{-3} cm") == [("right", True)]
    assert verdicts(L + "59.1 \\mu m") == [("right", False)]
    assert verdicts(L + "69.1 \\mu m") == [("wrong step", True)]
    assert verdicts(L + "591 \\mu m") == [("wrong step", True)]
    # (a mobility bare, in cm^2/Vs, a voltage in mV: the mV put the chain in SI - the result in cm^2/s is right, sure)
    assert verdicts("D_n = \\mu_n V_T = 1000 \\cdot 26 mV = 26 cm^{2}/s") == [("right", True)]
    assert verdicts("D_n = \\mu_n V_T = 1000 \\cdot 26 mV = 36 cm^{2}/s") == [("wrong step", True)]
    assert verdicts("N_D = 2 \\cdot 10^{15} cm^{-3} = 2 \\times 10^{21} m^{-3}") == [("right", True)]
    got = [(c.verdict, c.sure, c.text) for c in check_statements(statements("N_D = 2 \\cdot 10^{15} cm^{-3} = 2 \\times 10^{20} m^{-3}"))]
    assert [g[:2] for g in got] == [("wrong step", True)] and "2e+21 is not 2e+20" in got[0][2]


def test_reactive_volt_amperes_and_per_volume_units_in_a_key():
    """a rubric's key 'Q = 600 var' was a bad reference - pint knows VA only (the question could not be graded); var, VAR,
    kvar are units now; an answer per cubic centimetre ('5 \\frac{1}{cm^3}') was 'not a number'"""
    from grader.numeric import compare_numeric
    assert compare_numeric("600 VAR", "600 var").status == "correct"
    assert compare_numeric("0.6 kvar", "600 var").status == "correct"
    assert compare_numeric("0.6 kVAR", "600 VAR").status == "correct"
    assert compare_numeric("6 kvar", "600 var").status == "prefix_error"
    assert compare_numeric("700 var", "600 var").status == "wrong"
    assert compare_numeric("5 \\frac{1}{cm^3}", "5 cm^-3").status == "correct"
    assert compare_numeric("5 \\cdot 10^{6} m^{-3}", "5 cm^-3").status == "correct"


def test_a_reactance_with_its_unit_in_a_product():
    """'V = 2 mA \\cdot j4 k\\Omega = j8 V', '10 V / j5 \\Omega': a j-term after a factor with its unit has its own unit - it
    was letters (no verdict); after a bare factor it stays unread ('3 + 2 \\cdot j4 k\\Omega': whose kilo?)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("V = I Z = 2 A \\cdot j4 \\Omega = j8 V") == [("right", True)]
    assert verdicts("V = I Z = 2 mA \\cdot j4 k\\Omega = j8 V") == [("right", True)]
    assert verdicts("V = I Z = 2 mA \\cdot j4 k\\Omega = j9 V") == [("wrong step", True)]
    assert verdicts("I = 10 V / j5 \\Omega = -j2 A") == [("right", True)]
    assert verdicts("Z = 3 + 2 \\cdot j4 k\\Omega = 8.54 \\angle 69.4° k\\Omega") == []



def test_thousands_inside_a_computation():
    """'\\frac{12}{10,000} = 1.2 mA': a thousands comma kept the side unread - no verdict (a value alone, 'R = 10,000
    \\Omega', was read); the digits are joined now"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("I = \\frac{12}{10,000} = 1.2 mA") == [("right", True)]
    assert verdicts("I = \\frac{12 V}{10,000 \\Omega} = 1.2 mA") == [("right", True)]
    assert verdicts("I = \\frac{12}{10,000} = 12 mA") == [("wrong step", True)]
    assert verdicts("x = 1,000,000 \\cdot 2 = 2,000,000") == [("right", True)]


def test_a_formula_with_an_angle_in_degrees_and_prefixed_data():
    """'S = 1 kVA', '\\phi = 36.87°', then 'Q = S \\sin\\phi', 'Q = 600 VAR' was a sure wrong - 1 kVA was taken for 1 VA with
    the angle in degrees (0.6), or 1000 VA with 36.87 radians; in SI units with the angle in degrees now"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["Q = S \\sin\\phi", "S = 1 kVA", "\\phi = 36.87°", "Q = 600 VAR"]) == [("right", True)]
    assert verdicts(["Q = S \\sin\\phi", "S = 1 kVA", "\\phi = 36.87°", "Q = 0.6 kVAR"]) == [("right", True)]
    assert verdicts(["Q = S \\sin\\phi", "S = 1 kVA", "\\phi = 36.87°", "Q = 700 VAR"]) == [("wrong", True)]
    assert verdicts(["P = S \\cos\\phi", "S = 2 kVA", "\\phi = 30", "P = 1.73 kW"]) == [("right", True)]
    assert verdicts(["P = S \\cos\\phi", "S = 2 kVA", "\\phi = 30", "P = 2.73 kW"]) == [("wrong", True)]


def test_a_mobility_in_square_centimetres_per_volt_second():
    """'\\mu = 1350 cm^2/Vs': volt-seconds written together were letters - the mobility had no value, every formula with it
    'cannot check'"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    mu = "\\mu = 1350 cm^2/Vs"
    assert verdicts(["D = \\mu V_T", mu, "V_T = 25.9 mV", "D = 34.97 cm^2/s"]) == [("right", True)]
    assert verdicts(["D = \\mu V_T", mu, "V_T = 25.9 mV", "D = 44.97 cm^2/s"]) == [("wrong", True)]
    rho = ["\\rho = \\frac{1}{q N \\mu}", "q = 1.6 \\cdot 10^{-19} C", "N = 10^{16} cm^{-3}", mu]
    assert verdicts(rho + ["\\rho = 0.463 \\Omega cm"]) == [("right", True)]
    assert verdicts(rho + ["\\rho = 0.563 \\Omega cm"]) == [("wrong", True)]


def test_a_chain_with_an_angle_in_degrees_and_a_prefixed_result():
    """'P = S \\cos\\phi = 2000 \\cos(45) = 1.414 kW' was a sure wrong step - in degrees the result was compared as written
    (1414 against 1.414), in SI units in radians (1051 W)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("P = S \\cos\\phi = 2000 \\cos(45) = 1.414 kW") == [("right", True)]
    assert verdicts("P = S \\cos\\phi = 2000 \\cos(45) = 1.6 kW") == [("wrong step", True)]
    assert verdicts("X = 2200 \\sin(30) = 1.1 k\\Omega") == [("right", True)]
    assert verdicts("X = 2200 \\sin(30) = 1.3 k\\Omega") == [("wrong step", True)]


def test_a_formula_in_sizes():
    """'X = |Z| \\sin\\phi' with '|Z| = 2200 \\Omega': the size is the value the formula needs - it was 'no value for Z';
    '|I| = \\frac{|V|}{|Z|}', then '|I| = 5 mA' 'cannot be solved for |I|'; a formula with Z itself ('I = V / Z') gets no
    value from |Z|"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["X = |Z| \\sin\\phi", "|Z| = 2200 \\Omega", "\\phi = 30^\\circ", "X = 1100 \\Omega"]) == [("right", True)]
    assert verdicts(["X = |Z| \\sin\\phi", "|Z| = 2200 \\Omega", "\\phi = 30^\\circ", "X = 1430 \\Omega"]) == [("wrong", True)]
    assert verdicts(["P = |S| \\cos\\phi", "|S| = 1 kVA", "\\phi = 36.87°", "P = 800 W"]) == [("right", True)]
    assert verdicts(["|I| = \\frac{|V|}{|Z|}", "|V| = 10 V", "|Z| = 2 k\\Omega", "|I| = 5 mA"]) == [("right", True)]
    assert verdicts(["|I| = \\frac{|V|}{|Z|}", "|V| = 10 V", "|Z| = 2 k\\Omega", "|I| = 6 mA"]) == [("wrong", True)]
    assert verdicts(["I = \\frac{V}{Z}", "V = 10 V", "|Z| = 5 \\Omega", "I = 2 A"]) == [("cannot check", False)]


def test_names_of_several_letters():
    """'pf', 'BW' and a subscript written in line ('Req', 'Rth', 'Vth', 'Vrms', 'gm', 'Av') are one name each - they were
    products of letters (p times f, R times e times q): a formula with them 'cannot check' or no verdict; 'RC' stays a
    product, '100 pf' picofarads, '\\text{Vth}' words"""
    from grader.selfcheck import _ee_words, check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["\\theta = \\arccos(pf)", "pf = 0.8", "\\theta = 36.87°"]) == [("right", True)]
    assert verdicts(["pf = \\cos(\\tan^{-1}\\frac{X}{R})", "X = 4 \\Omega", "R = 3 \\Omega", "pf = 0.8"]) == [("wrong", True)]
    assert verdicts(["BW = \\frac{f_0}{Q}", "f_0 = 10 kHz", "Q = 20", "BW = 500 Hz"]) == [("right", True)]
    assert verdicts(["Req = R_1 + R_2", "R_1 = 2k", "R_2 = 3k", "Req = 5k"]) == [("right", True)]
    assert verdicts(["Req = R_1 + R_2", "R_1 = 2k", "R_2 = 3k", "Req = 6k"]) == [("wrong", True)]
    assert verdicts(["I = \\frac{Vth}{Rth + R_L}", "Vth = 7.2 V", "Rth = 1.2k", "R_L = 2.4k", "I = 2 mA"]) == [("right", True)]
    assert verdicts(["Av = -gm R_C", "gm = 40 mS", "R_C = 5k", "Av = -200"]) == [("right", True)]
    assert verdicts(["\\tau = RC", "R = 1k", "C = 1 \\mu F", "\\tau = 1 ms"]) == [("right", True)]
    assert _ee_words("C = 100 pf") == "C = 100 pf" and _ee_words("\\text{Vth}") == "\\text{Vth}"
    assert _ee_words("\\tau = RC") == "\\tau = RC"


def test_values_flowing_on_by_name_rounded_on_the_way():
    """found by a random generator of work over several lines: 'I = \\frac{V}{R_{eq}} = 208 \\mu A', then 'P = I^2 R_{eq} =
    (208 \\mu A)^2 \\cdot 24 k\\Omega = 1.04 mW' was a sure wrong step - I was 208 (as written) beside R_eq = 24000 (in SI
    units); 'P = I^2 R_{eq} = 305 mW' under 'I = 15.7 mA', 'R_{eq} = 1.25 k\\Omega' (worked out with I unrounded, 15.66 mA)
    was a sure wrong step - 1 % off, within what the rounding of the values written allows"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["R_{eq} = R_1 + R_2 = 2 k\\Omega + 22k = 24 k\\Omega", "I = \\frac{V}{R_{eq}} = 208 \\mu A",
                     "P = I^2 R_{eq} = (208 \\mu A)^2 \\cdot 24 k\\Omega = 1.04 mW"]) == [("right", True), ("right", True), ("right", True)]
    lines = ["V = 15 V", "R_{eq} = R_1 \\| R_2 = 2 k\\Omega \\| 3.3k = 1.25 k\\Omega", "I = \\frac{V}{R_{eq}} = 12 mA"]
    assert [v for v in verdicts(lines + ["P = I^2 R_{eq} = 180 mW"]) if v[0] != "right"] == []
    assert ("wrong step", True) in verdicts(lines + ["P = I^2 R_{eq} = 190 mW"])
    assert verdicts(["P = I^2 R = (13.2 mA)^2 \\cdot 1.13 k\\Omega = 199 mW"]) == [("right", True)]
    assert verdicts(["P = I^2 R = (13.2 mA)^2 \\cdot 1.13 k\\Omega = 205 mW"]) == [("wrong step", True)]
    assert verdicts(["V = 3.3 \\cdot 2.2 = 7.36"]) == [("wrong step", True)]      # (components' values are exact)


def test_a_letter_worked_out_by_another_formula_under_a_formula():
    """'P = I^2 R', then 'I = \\frac{V}{R} = \\frac{12 V}{4.7 k\\Omega} = 2.55 mA': V is no letter of P = I^2 R - the line was
    skipped, the formula 'no value for I'; its steps are checked now and its value is the formula's"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    i_line = "I = \\frac{V}{R} = \\frac{12 V}{4.7 k\\Omega} = "
    assert verdicts(["P = I^2 R", i_line + "2.55 mA", "R = 4.7 k\\Omega", "P = 30.6 mW"]) == [("right", True)]
    assert verdicts(["P = I^2 R", i_line + "2.55 mA", "R = 4.7 k\\Omega", "P = 40.6 mW"]) == [("wrong", True)]
    got = verdicts(["P = I^2 R", i_line + "3.55 mA", "R = 4.7 k\\Omega", "P = 59.2 mW"])
    assert ("wrong step", True) in got and ("right", True) in got      # (the slip; P right with the student's I)


def test_a_second_formula_defining_a_letter_of_the_first():
    """'P = I^2 R', 'I = \\frac{V}{R}', then the data and 'P = 30.6 mW': the data and the answer went to the second formula
    - 'cannot check'; the second gives the first its letter now (both checked: its own answer too)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    head = ["P = I^2 R", "I = \\frac{V}{R}", "V = 12 V", "R = 4.7 k\\Omega"]
    assert verdicts(head + ["P = 30.6 mW"]) == [("right", True)]
    assert verdicts(head + ["P = 40.6 mW"]) == [("wrong", True)]
    assert verdicts(head + ["I = 2.55 mA", "P = 30.6 mW"]) == [("right", True), ("right", True)]
    fc = ["f_c = \\frac{1}{2\\pi R C}", "R = \\frac{R_1 R_2}{R_1 + R_2}", "R_1 = 2.2k", "R_2 = 6.8k", "C = 100 nF"]
    assert verdicts(fc + ["f_c = 959 Hz"]) == [("right", True)]
    assert verdicts(fc + ["f_c = 1.2 kHz"]) == [("wrong", True)]


def test_an_angle_written_with_a_degree_sign_as_a_value():
    """'\\phi = 36.87°' (or '36.87 deg') was 36.87 radians as a value - '36.87^\\circ' was in radians already: 'Q = 1000
    \\sin\\phi = 600' under it a sure wrong step (-737.3); a degree sign is a ^\\circ now (not °C); the lecturer's '36.87°'
    was no value at all"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines, values=None):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st, values=values)]
    assert verdicts(["\\phi = 36.87°", "Q = 1000 \\sin\\phi = 600"]) == [("right", True)]
    assert verdicts(["\\phi = 36.87°", "Q = 1000 \\sin\\phi = 700"]) == [("wrong step", True)]
    assert verdicts(["\\phi = 36.87 deg", "Q = 1000 \\sin\\phi = 600"]) == [("right", True)]
    assert verdicts(["Q = S \\sin\\phi", "Q = 600 VAR"], {"S": "1 kVA", "\\phi": "36.87°"}) == [("right", True)]
    assert verdicts(["Q = S \\sin\\phi", "Q = 700 VAR"], {"S": "1 kVA", "\\phi": "36.87°"}) == [("wrong", True)]
    assert verdicts(["T = 25°C"]) == []


def test_a_gain_in_db_beside_its_ratio():
    """'A_v = 40 dB = 100' was a sure wrong step (40 against 100 - the middle side's dB dropped); 'A_v = 40 dB', then 'V_o =
    A_v V_i', 'V_o = 10 V' was a sure wrong (40 times V_i): the gain is its ratio there; 40 times V_i (4 V) is not sure -
    dB multiplied as a ratio; dB added stay right ('G = G_1 + G_2')"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["A_v = 40 dB = 100"]) == [("right", True)]
    assert verdicts(["A_v = 40 dB = 1000"]) == [("wrong step", True)]
    assert verdicts(["G = 100 = 40 dB"]) == [("right", True)]
    assert verdicts(["G_p = 30 dB = 1000"]) == [("right", False)]          # (10 log10: a power's)
    form = ["V_o = A_v V_i", "A_v = 40 dB", "V_i = 0.1 V"]
    assert verdicts(form + ["V_o = 10 V"]) == [("right", True)]
    assert verdicts(form + ["V_o = 4 V"]) == [("right", False)]
    assert verdicts(form + ["V_o = 12 V"]) == [("wrong", True)]
    assert verdicts(["G = G_1 + G_2", "G_1 = 20 dB", "G_2 = 10 dB", "G = 30 dB"]) == [("right", True)]


def test_a_gain_in_db_used_in_a_chain():
    """'A_v = 40 dB', 'V_i = 0.1 V', then 'V_o = A_v V_i = 100 \\cdot 0.1 = 10 V': 'A_v V_i = 100 \\cdot 0.1' was a sure wrong
    step (40 times 0.1 is 4) - the gain is its ratio; '= 40 \\cdot 0.1 = 4 V' (dB multiplied as a ratio) is no longer sure"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    data = ["A_v = 40 dB", "V_i = 0.1 V"]
    assert verdicts(data + ["V_o = A_v V_i = 100 \\cdot 0.1 = 10 V"]) == [("right", True), ("right", True)]
    assert verdicts(data + ["V_o = A_v V_i = 10 V"]) == [("right", True)]
    assert verdicts(data + ["V_o = A_v V_i = 40 \\cdot 0.1 = 4 V"]) == [("right", False), ("right", True)]
    assert verdicts(data + ["V_o = A_v V_i = 12 V"]) == [("wrong step", True)]
    assert verdicts(["A_v = 40 dB", "V_i = 100 mV", "V_o = A_v V_i = 10 V"]) == [("right", True)]
    assert verdicts(["G_1 = 20 dB", "G_2 = 10 dB", "G = G_1 + G_2 = 20 + 10 = 30 dB"]) == [("right", True), ("right", True)]


def test_typed_names_with_a_digit():
    """typed as plain text: 'BW = f0/Q = 10000/20 = 500 Hz' was a sure wrong step - f0 was f times 0; 'f0', 't1', 'T2',
    'N1', 'x1', 'w0' are names with their subscript now, as 'R2', 'V1' were ('j4' stays 4j)"""
    from grader.selfcheck import check_statements, statements

    def verdicts(t):
        return [(c.verdict, c.sure) for c in check_statements(statements(t))]
    assert verdicts("BW = f0/Q = 10000/20 = 500 Hz") == [("right", True)]
    assert verdicts("BW = f0/Q = 10000/20 = 600 Hz") == [("wrong step", True)]
    assert verdicts("T = t2 - t1 = 5 - 2 = 3") == [("right", True)]
    assert verdicts("Z = 3 + j4 = 5 \\angle 53.13°") == [("right", True)]


def test_typed_logs_powers_of_ten_and_ohms():
    """typed: 'log10(100) = 2' was a sure wrong step (log of 10 times 100, 3); '3**2 * 4 = 36' too ('**' read as 4);
    '12/4.7e3' was (12/4.7) times 1000 - its message; '159 Ohm' (a capital O) was letters - no verdict"""
    from grader.selfcheck import check_statements, statements

    def checks(t):
        return [(c.verdict, c.sure, c.text) for c in check_statements(statements(t))]
    assert [c[:2] for c in checks("x = log10(100) = 2")] == [("right", True)]
    assert [c[:2] for c in checks("x = log2(8) = 3")] == [("right", True)]
    got = checks("I = 12/4.7e3 = 3.55e-3 A")
    assert [c[:2] for c in got] == [("wrong step", True)] and "0.002553 is not 0.00355" in got[0][2]
    assert [c[:2] for c in checks("Xc = 1/(2*pi*f*C) = 1/(2*pi*1000*1e-6) = 159 Ohm")] == [("right", True)]
    assert [c[:2] for c in checks("Xc = 1/(2*pi*f*C) = 1/(2*pi*1000*1e-6) = 259 Ohm")] == [("wrong step", True)]
    assert [c[:2] for c in checks("P = 3**2 * 4 = 36")] == [("right", True)]          # ('**' a power: it was 4)
    assert [c[:2] for c in checks("P = 3**2 * 4 = 46")] == [("wrong step", True)]
    assert [c[:2] for c in checks("x = sqrt(10e-3*100e-9) = 3.16e-5")] == [("right", True)]


def test_a_page_of_unnumbered_parts_reusing_letters():
    """found by a random generator of exam pages: sub-problems one after another, unnumbered, reusing R, I, V, P with
    values of their own - a part's value was taken for the next one's ('R = 4.7 k' the formula's answer, 'I = V/R' judged
    with the I worked out above: false sure wrongs); a part done again ('I = V/R = 6.5 mA', then '= 5 mA') is judged on
    its own and its value is the one the lines below use; one slip, one wrong step"""
    from grader.selfcheck import check_statements, statements

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure, c.text) for c in check_statements(st)]

    def sure_wrongs(lines):
        return [v for v in verdicts(lines) if v[0] in ("wrong", "wrong step") and v[1]]
    # the next part starts with a letter foreign to a formula all of whose letters have their values
    assert sure_wrongs(["P = I^2 R", "I = 2 mA", "R = 2.2 k\\Omega", "P = 8.8 mW", "V = 10 V", "R = 2.2 k\\Omega",
                        "I = \\frac{V}{R} = 4.55 mA", "P = V I = 45.5 mW"]) == []
    # a value given again ('R = 1 k\Omega' twice) goes with the part that starts below it
    assert sure_wrongs(["P = I^2 R", "I = 5 mA", "R = 1 k\\Omega", "P = 25 mW", "R = 1 k\\Omega", "V = 5 V",
                        "I = \\frac{V}{R} = \\frac{5}{1 k\\Omega} = 5 mA", "R = 4.7 k\\Omega", "V = 5 V",
                        "I = \\frac{V}{R} = \\frac{5}{4.7 k\\Omega} = 1.06 mA"]) == []
    # worked out again: the slip once, not sure (crossed out?) - nothing else wrong
    lines = ["V = 5 V", "R = 1 k\\Omega", "I = \\frac{V}{R} = \\frac{5}{1 k\\Omega} = 6.5 mA", "V = 5 V", "R = 1 k\\Omega",
             "I = \\frac{V}{R} = 5 mA", "P = V I = 25 mW"]
    got = [v for v in verdicts(lines) if v[0] in ("wrong", "wrong step")]
    assert sure_wrongs(lines) == [] and len(got) == 1 and "6.5" in got[0][2]
    # a letter solved from a side with other letters, their values a part's above: no value of anything
    assert sure_wrongs(["V = 12 V", "R = 4.7 k\\Omega", "I = \\frac{V}{R} = 2.55 mA", "P = V I = 30.6 mW",
                        "V_{out} = V \\frac{R_2}{R + R_2} = 5 \\cdot \\frac{1 k\\Omega}{4.7 k\\Omega + 1 k\\Omega} = 0.877 V",
                        "V_{out} = V \\frac{R_2}{R + R_2} = 12 \\cdot \\frac{1 k\\Omega}{1 k\\Omega + 1 k\\Omega} = 6 V"]) == []
    # under two formulas, the second giving the first its letter: a slip below both reported once
    got = sure_wrongs(["P = I^2 R", "I = \\frac{V}{R}", "V = 10 V", "R = 1 k\\Omega", "I = 10 mA", "P = 100 mW",
                       "V_{out} = V \\frac{R_2}{R + R_2} = 5 \\cdot \\frac{3.3 k\\Omega}{2.2 k\\Omega + 3.3 k\\Omega} = 3.9 V"])
    assert len(got) == 1 and "3.9" in got[0][2]
    # (the letters of the formula the second one feeds are the problem's too: 'f_c = 107 kHz' is no next part)
    v = verdicts(["f_c = \\frac{1}{2\\pi R C}", "R = \\frac{R_1 R_2}{R_1 + R_2}", "C = 1 nF", "R_2 = 3.3 k\\Omega",
                  "R_1 = 4.7 k\\Omega", "R = 1.94 k\\Omega", "f_c = 107 kHz"])
    assert any(x[0] == "wrong" for x in v), v


def test_question_and_sub_question_marks():
    """'b) P = V I = 30 mW', '(a) R = 1 k\\Omega', 'Q1: ...', '1a) ...', 'א) ...', 'Part c: ...': a line starting with a
    question's or sub-question's mark was lost (no side 'b) P'), or read as a formula ('(a) R = 1 k\\Omega': a times R; 'Q1:
    R = 5 k\\Omega': Q1 times R) - the page got no verdict; a question's number starts a new exercise, its later parts not"""
    from grader.selfcheck import check_statements, statements

    def read(t):
        return [(x.kind, x.text, bool(x.calc.get("item"))) for x in statements(t)]
    for t in ("a) R = 5 k\\Omega", "(a) R = 5 k\\Omega", "a. R = 5 k\\Omega", "\\text{a)} R = 5 k\\Omega", "\\textbf{(a)} R = 5 k\\Omega",
              "(ii) R = 5 k\\Omega", "a: R = 5 k\\Omega", "Part a: R = 5 k\\Omega", "\\text{Part (b)} R = 5 k\\Omega",
              "2b. R = 5 k\\Omega", "1(b) R = 5 k\\Omega", "3.2) R = 5 k\\Omega"):
        assert read(t) == [("value", "R = 5 k\\Omega", False)], (t, read(t))
    for t in ("א) R = 5 k\\Omega", "ב. R = 5 k\\Omega", "(א) R = 5 k\\Omega", "סעיף א: R = 5 k\\Omega"):
        assert read(t) == [("value", "R = 5 k\\Omega", False)], (t, read(t))
    for t in ("Q1: R = 5 k\\Omega", "Q1) R = 5 k\\Omega", "1a) R = 5 k\\Omega", "Question 3 R = 5 k\\Omega", "שאלה 1: R = 5 k\\Omega",
              "\\textbf{Q2:} R = 5 k\\Omega", "1) R = 5 k\\Omega", "3.1) R = 5 k\\Omega"):
        assert read(t) == [("value", "R = 5 k\\Omega", True)], (t, read(t))
    for t, kind in (("Q_1 = 5 mC", "value"), ("Q1 = 5", "value"), ("(2a)^2 = 4a^2", "identity")):    # no marks
        assert read(t)[0][:2] == (kind, t)

    def verdicts(lines):
        st = []
        for k, t in enumerate(lines):
            st += statements(t, line=k, piece=k)
        return [(c.verdict, c.sure) for c in check_statements(st)]
    assert verdicts(["a) R = 1 k\\Omega", "V = 5 V", "I = \\frac{V}{R} = 5 mA", "b) P = V I = 25 mW"]) == [("right", True)] * 2
    assert ("wrong step", True) in verdicts(["(a) R = 1 k\\Omega", "V = 5 V", "I = \\frac{V}{R} = 5 mA", "(b) P = V I = 30 mW"])


def _page_verdicts(lines):
    from grader.selfcheck import check_statements, statements
    st = []
    for k, t in enumerate(lines):
        st += statements(t, line=k, piece=k)
    return [(c.verdict, c.sure, c.text) for c in check_statements(st)]


def test_a_computation_foreign_to_the_formula_checked_with_the_values_on_the_page():
    """found by a random generator of exam answers in sub-questions: 'f_c = \\frac{1}{2\\pi \\tau} = 207 Hz' under '\\tau = R C'
    and its work, 'I_0 = \\frac{V}{R} = 6.5 mA' after them, 'I = \\frac{V}{|Z|} = 1.59 A' after '|Z| = ... = 188 \\Omega': checked
    with no values (or never, when it started a problem of its own) - the slips were missed; now with the values known
    on the page (a step wrong only with an earlier problem's values not sure)"""
    v = _page_verdicts(["R = 10 k\\Omega", "C = 100 nF", "V = 5 V", "\\text{a)} \\tau = R C", "\\tau = 10 k\\Omega \\cdot 100 nF = 1 ms",
                        "\\text{b)} f_c = \\frac{1}{2\\pi \\tau} = 207 Hz", "\\text{c)} I_0 = \\frac{V}{R} = 500 \\mu A"])
    assert [x[:2] for x in v if x[0] == "wrong step"] == [("wrong step", True)] and "207" in [x for x in v if x[0] == "wrong step"][0][2]
    v = _page_verdicts(["R = 1 k\\Omega", "C = 1 \\mu F", "V = 5 V", "a. \\tau = R C = 1 ms", "b. f_c = \\frac{1}{2\\pi \\tau}",
                        "f_c = \\frac{1}{2\\pi \\cdot 1 ms} = 159 Hz", "c. I_0 = \\frac{V}{R} = 6.5 mA"])
    assert any(x[0] == "wrong step" and "6.5" in x[2] for x in v) and not any(x[0].startswith("wrong") and x[1] for x in v)
    v = _page_verdicts(["R = 100 \\Omega", "C = 1 \\mu F", "f = 1 kHz", "V = 230 V", "(a) X_C = \\frac{1}{2\\pi f C}",
                        "X_C = \\frac{1}{2\\pi \\cdot 1 kHz \\cdot 1 \\mu F} = 159 \\Omega",
                        "(b) |Z| = \\sqrt{R^2 + X_C^2} = \\sqrt{100^2 + 159^2} = 188 \\Omega", "(c) I = \\frac{V}{|Z|} = 1.59 A"])
    assert any(x[0] == "wrong step" and "1.59" in x[2] for x in v)
    v = _page_verdicts(["R = 100 \\Omega", "C = 1 \\mu F", "f = 1 kHz", "V = 230 V", "(a) X_C = \\frac{1}{2\\pi f C}",
                        "X_C = \\frac{1}{2\\pi \\cdot 1 kHz \\cdot 1 \\mu F} = 159 \\Omega",
                        "(b) |Z| = \\sqrt{R^2 + X_C^2} = \\sqrt{100^2 + 159^2} = 188 \\Omega", "(c) I = \\frac{V}{|Z|} = 1.22 A"])
    assert not any(x[0].startswith("wrong") for x in v), v
    # (a number in e-notation before a power is the number squared: '3.18e+03^2' is 3180^2, not 3.18 times 10^6)
    assert [x[:2] for x in _page_verdicts(["Z = \\sqrt{1e+03^2 + 3.18e+03^2} = 3.33 k\\Omega"])] == [("right", True)]


def test_an_equations_roots_each_judged():
    """'x^2 - 5x + 6 = 0', then 'x = 4, x = 2' (or on lines of their own): the last root only was judged - x = 4 never; and
    roots on lines of their own were taken for the next part of the page (a letter given again)"""
    for lines in (["x^2 - 5x + 6 = 0 \\Rightarrow x = 4, x = 2"], ["x^2 - 5x + 6 = 0", "x = 2", "x = 4"],
                  ["x^2 - 5x + 6 = 0", "x = 4", "x = 2"], ["s^2 + 3s + 2 = 0", "s = -1", "s = -3"]):
        v = _page_verdicts(lines)
        assert sorted(x[0] for x in v) == ["right", "wrong"], (lines, v)
    assert sorted(x[0] for x in _page_verdicts(["x^2 - 5x + 6 = 0", "x = 2", "x = 3"])) == ["right", "right"]
    assert [x[:2] for x in _page_verdicts(["x^2 = 9 \\Rightarrow x = \\pm 4"])] == [("wrong", True)]


def test_an_answer_or_a_computation_done_again_below_itself():
    """a line done again just below itself, put right (the first try crossed out - the reader does not see strokes): the
    second is the answer, the first not judged wrong for sure; the lines using the value use the second's"""
    v = _page_verdicts(["P = I^2 R", "I = 1 mA", "R = 10 k\\Omega", "P = 13 mW", "P = 10 mW"])
    assert ("right", True) in [x[:2] for x in v] and ("wrong", False) in [x[:2] for x in v]
    v = _page_verdicts(["P = I^2 R", "I = 1 mA", "R = 10 k\\Omega", "P = 10 mW", "P = 13 mW"])     # put wrong: the answer wrong
    assert ("wrong", True) in [x[:2] for x in v]
    v = _page_verdicts(["V = 5 V", "R = 1 k\\Omega", "I = \\frac{V}{R} = \\frac{5}{1 k\\Omega} = 6.5 mA",
                        "I = \\frac{V}{R} = \\frac{5}{1 k\\Omega} = 5 mA", "P = V I = 25 mW"])
    assert [x[:2] for x in v if x[0] == "wrong step"] == [("wrong step", False)] and "crossed out" in [x for x in v if x[0] == "wrong step"][0][2]
    v = _page_verdicts(["P = I^2 R", "I = 2 mA", "R = 2.2 k\\Omega", "P = (2 mA)^2 \\cdot 2.2 k\\Omega = 10 mW",
                        "P = (2 mA)^2 \\cdot 2.2 k\\Omega = 8.8 mW"])
    assert not any(x[0].startswith("wrong") and x[1] for x in v) and ("right", True) in [x[:2] for x in v]
    # (a value worked out keeps its digits: '= 10 mW' is two of them - as '0.01' one, and 8.8 mW passed for it)
    v = _page_verdicts(["P = I^2 R", "I = 2 mA", "R = 2.2 k\\Omega", "P = (2 mA)^2 \\cdot 2.2 k\\Omega = 10 mW"])
    assert [x[:2] for x in v if x[0] in ("right", "wrong")] == [("wrong", True)] and "10mW" in v[0][2]


def test_electronics_answers_rounded_on_the_way():
    """found by a random generator of electronics answers: 'I_C = \\beta I_B = 11 mA' (10.99 unrounded), then 'V_{CE} =
    V_{CC} - I_C R_C = 1.01 V' was a sure wrong (12 - 11 = 1; 1 % by the float's last bit) - an answer within what the
    rounding of a value worked out above moves it is no sure wrong; 'V = V_s - I R', 'V = 1 V' was 'right value, wrong
    unit' (volts less volts: '1 (no unit)'); 'I_L = \\frac{V_{th}}{R_{th} + R_L} = 1.9 mA' with R_th worked out (in ohms) and
    R_L given in kΩ was right 'in units not written', 1000 apart - a slip missed"""
    head = ["V_{CC} = 12 V", "R_C = 1 k\\Omega", "\\beta = 200", "I_B = 55 \\mu A", "I_C = \\beta I_B = 11 mA", "V_{CE} = V_{CC} - I_C R_C"]
    assert [x[:2] for x in _page_verdicts(head + ["V_{CE} = 1.01 V"])] == [("right", True)] * 2
    assert ("wrong", False) in [x[:2] for x in _page_verdicts(head + ["V_{CE} = 1.3 V"])]     # (within 11 ± 0.5 mA: for a human)
    assert ("wrong", True) in [x[:2] for x in _page_verdicts(head + ["V_{CE} = 3 V"])]
    v = _page_verdicts(["V_{CC} = 12 V", "R_C = 1 k\\Omega", "I_C = 11 mA", "V_{CE} = V_{CC} - I_C R_C", "V_{CE} = 1.3 V"])
    assert [x[:2] for x in v] == [("wrong", True)] and "V_CC=12" in v[0][2]           # (11 mA given: exact)
    assert [x[:2] for x in _page_verdicts(["V_s = 12 V", "R = 1 k\\Omega", "I = 11 mA", "V = V_s - I R", "V = 1 V"])] == [("right", True)]
    v = _page_verdicts(["V_s = 12 V", "R = 1 k\\Omega", "I = 11 mA", "V = V_s - I R", "V = 1 A"])
    assert v[0][0] == "right value, wrong unit" and "in V, not A" in v[0][2]
    v = _page_verdicts(["V_s = 9 V", "R_1 = 4.7 k\\Omega", "R_2 = 10 k\\Omega", "R_L = 1 k\\Omega",
                        "(a) V_{th} = V_s \\frac{R_2}{R_1 + R_2} = 9 \\cdot \\frac{10 k\\Omega}{4.7 k\\Omega + 10 k\\Omega} = 6.12 V",
                        "(b) R_{th} = R_1 \\| R_2", "R_{th} = 4.7 k\\Omega \\| 10 k\\Omega = 3.2 k\\Omega",
                        "(c) I_L = \\frac{V_{th}}{R_{th} + R_L} = 1.9 mA"])
    assert ("wrong step", True) in [x[:2] for x in v] and any("1.9" in x[2] for x in v if x[0] == "wrong step")


def test_boolean_algebra_judged_by_truth_tables():
    """a digital-systems exam's Boolean algebra - 'F = A\\bar{B} + \\bar{A}B = A \\oplus B', '\\overline{A + B} = \\bar{A}\\bar{B}',
    "A' + B' = (AB)'", 'F = AB + A\\bar{B}' then 'F = A': the lines were dropped (\\bar no letter to the parser) or read as
    algebra - no verdict. Each side against the next by their truth tables now (a row where they differ shown); a
    function's name ('F =', 'F(A,B,C) =') is no letter of it; a mean ('\\bar{x} = 5.2'), two segments ('\\overline{AB} =
    \\overline{CD}'), a primed name ("V' = V + IR") are no Boolean algebra"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["\\overline{A + B} = \\bar{A} \\bar{B}"]) == [("right", True)]
    got = _page_verdicts(["\\overline{A + B} = \\bar{A} + \\bar{B}"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "A=0, B=1" in got[0][2]
    assert v(["F = A\\bar{B} + \\bar{A}B = A \\oplus B"]) == [("right", True)]
    assert v(["F(A,B,C) = \\bar{A}BC + ABC = BC"]) == [("right", True)]
    assert v(["A' + B' = (AB)'"]) == [("right", True)] and v(["(AC)' = A'C'"]) == [("wrong step", True)]
    assert v(["\\overline{\\overline{A}} = A"]) == [("right", True)] and v(["A \\cdot \\bar{A} = 0"]) == [("right", True)]
    assert v(["F = AB + A\\bar{B}", "F = A(B + \\bar{B})", "F = A"]) == [("right", True)] * 2
    assert v(["F = AB + A\\bar{B}", "F = B"]) == [("wrong step", True)]
    assert v(["Y = \\overline{XY}", "Y = \\bar{X}\\bar{Y}"]) == [("wrong step", True)]     # (Y the name, a letter too)
    assert v(["F = A \\oplus B", "Y = (A + B)(A + C) = A + BC"]) == [("right", True)]     # a plain line on a Boolean page
    for lines in (["Y = (A + B)(A + C) = A + BC"], ["\\bar{x} = 5.2"], ["\\bar{v} = 0"], ["\\overline{AB} = \\overline{CD}"]):
        assert v(lines) == [], lines
    assert not any(x[0].startswith("wrong") for x in v(["V' = V + IR"]))


def test_numbers_in_bases_written_every_way():
    """'0x1F', '0b1011', '1F_{16}' and 'FF_{16}' unbracketed, '(1F)_H', '1011_B', '17_O': letters to the parser (no verdict) or a
    product; names with such subscripts stay names ('V_B', 'C_B', 'R_D', 'F_{16}')"""
    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    for t in ("0x1F = 31", "0b1011 = 11", "1F_{16} = 31_{10}", "255 = FF_{16}", "(1F)_H = 31", "1011_B = 11", "17_O = 15", "(25)_D = (11001)_B"):
        assert v(t) == [("right", True)], t
    for t in ("0x1F = 30", "1011_B = 12", "1F_{16} = 30"):
        assert v(t) == [("wrong step", True)], t
    from grader.selfcheck import statements
    for t in ("V_B = 5 V", "C_B = 10 \\mu F", "R_D = 2 k\\Omega", "F_{16} = 2"):
        assert [s.kind for s in statements(t)] == ["value"], t


def test_a_frequency_response_evaluated_at_a_frequency():
    """'H(j\\omega) = \\frac{10}{10 + j\\omega}', then '|H(j10)| = 0.707' or '\\angle H(j10) = -45^\\circ': 'no value for omega' (the
    frequency written in 'H(j10)' was dropped), the angle no verdict; at several frequencies the second was judged against
    the first's value (a false wrong step) - each at its own now"""
    head = ["H(j\\omega) = \\frac{10}{10 + j\\omega}"]
    assert [x[:2] for x in _page_verdicts(head + ["|H(j10)| = \\frac{10}{\\sqrt{200}} = 0.707"])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(head + ["|H(j10)| = 0.8"])] == [("wrong", True)]
    assert [x[:2] for x in _page_verdicts(head + ["\\angle H(j10) = -45^\\circ"])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(head + ["\\angle H(j10) = -60^\\circ"])] == [("wrong", True)]
    hp = ["H(j\\omega) = \\frac{j\\omega}{100 + j\\omega}"]
    assert sorted(x[:2] for x in _page_verdicts(hp + ["|H(j100)| = 0.707", "|H(j1000)| = 0.995"])) == [("right", True)] * 2
    assert sorted(x[:2] for x in _page_verdicts(hp + ["|H(j100)| = 0.5", "|H(j1000)| = 0.995"])) == [("right", True), ("wrong", True)]


def test_more_lines_that_got_no_verdict():
    """lines of the batteries that got no verdict: '\\angle(3 + j4) = 53.13^\\circ' (angle times a number), 'a_n = 2^n' then
    'a_3 = 9' (a sequence's term), 'r_{1,2} = -1, -2' (the second root lost), '100 W \\cdot 2 h = 200 Wh' and '500 mA \\cdot 2 h =
    1 Ah' (h a letter), '\\sqrt{250n/100p} = 50 \\Omega' (a bare n, p before a slash), '|H(j10^3)| = 0.707' (the frequency)"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["\\angle (3 + j4) = 53.13^\\circ"]) == [("right", True)] and v(["\\angle (3 + j4) = 60^\\circ"]) == [("wrong step", True)]
    assert v(["\\arg(1 + j) = 45^\\circ"]) == [("right", True)]
    assert v(["a_n = 2^n", "a_3 = 8"]) == [("right", True)] and v(["a_n = 2^n | a_3 = 9"]) == [("wrong", True)]
    assert v(["V_s = 12 V", "R_1 = 2"]) == []
    assert sorted(v(["r^2 + 3r + 2 = 0 | r_{1,2} = -1, -2"])) == [("right", True)] * 2
    assert sorted(v(["x^2 + 3x + 2 = 0", "x_{1,2} = -1, -3"])) == [("right", True), ("wrong", True)]
    assert v(["E = P t = 100 W \\cdot 2 h = 200 Wh"]) == [("right", True)] and v(["E = 100 W \\cdot 2 h = 2 kWh"]) == [("wrong step", True)]
    assert v(["Q = I t = 500 mA \\cdot 2 h = 1 Ah"]) == [("right", True)] and v(["Q = 500 mA \\cdot 2 h = 2 Ah"]) == [("wrong step", True)]
    assert v(["Z_0 = \\sqrt{L/C} = \\sqrt{250n/100p} = 50 \\Omega"]) == [("right", True)]
    assert v(["Z_0 = \\sqrt{L/C} = \\sqrt{250n/100p} = 70 \\Omega"]) == [("wrong step", True)]
    head = ["H(j\\omega) = \\frac{1000}{1000 + j\\omega}"]
    assert v(head + ["|H(j10^3)| = 0.707"]) == [("right", True)] and v(head + ["|H(j1k)| = 0.5"]) == [("wrong", True)]


def test_units_spelled_angles_in_trig_lengths_and_integer_functions():
    """more battery lines that got no verdict: '20 Watt', '10 fA', '\\cos(60 rad)' and '\\sin(30 deg)' (rad, deg letters), lengths
    before a length ('5 m - 2 m = 3 m', '2m \\cdot 3m = 6 m^2'), '17 \\bmod 5', '\\gcd(12, 18)', '1 + 2 + \\ldots + 10' (no value),
    '\\Phi(1) = 0.8413', 'Q(2) = 0.0228' (the normal distribution), a formula written after its values"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["P = I^2 R = 2^2 \\cdot 5 = 20 Watt"]) == [("right", True)] and v(["P = 2^2 \\cdot 5 = 25 Watts"]) == [("wrong step", True)]
    assert v(["I_D = I_S e^{V_D / V_T} = 10 fA \\cdot e^{700 mV / 25 mV} = 14.5 mA"]) == [("right", True)]
    assert v(["\\cos(60 rad) = 0.5"]) == [("wrong step", True)] and v(["x = \\sin(30 deg) = 0.5"]) == [("right", True)]
    assert v(["\\Delta x = 5 m - 2 m = 3 m"]) == [("right", True)] and v(["A = 2m \\cdot 3m = 5 m^2"]) == [("wrong step", True)]
    assert v(["x = 5 cm + 3 m = 3.05 m"]) == [("right", True)]
    # (a speed's metres are no length: '\\frac{3 \\cdot 10^8 m/s}{2.4 GHz} = 12.5 cm' stays judged)
    assert v(["\\lambda = \\frac{c}{f} = \\frac{3 \\cdot 10^8 m/s}{2.4 GHz} = 12.5 cm"]) == [("right", True)]
    assert v(["\\lambda = \\frac{c}{f} = \\frac{3 \\cdot 10^8 m/s}{2.4 GHz} = 15.5 cm"]) == [("wrong step", True)]
    assert v(["17 \\bmod 5 = 2"]) == [("right", True)] and v(["17 \\bmod 5 = 3"]) == [("wrong step", True)]
    assert v(["\\gcd(12, 18) = 6"]) == [("right", True)] and v(["\\gcd(12, 18) = 3"]) == [("wrong step", True)]
    assert v(["1 + 2 + \\ldots + 10 = 55"]) == [("right", True)] and v(["1 + 2 + ... + 10 = 50"]) == [("wrong step", True)]
    assert v(["P(X < 5) = \\Phi(\\frac{5 - 3}{2}) = \\Phi(1) = 0.8413"]) == [("right", True)] * 2
    assert v(["Q(2) = 0.0228"]) == [("right", True)] and v(["Q(2) = 0.05"]) == [("wrong step", True)]
    assert v(["Q = C V = 2 \\cdot 5 = 10"]) == [("right", True)]                # (Q a charge: no function)
    assert v(["V_{rms} = 10 V | V_p = 14.1 V | V_p = \\sqrt{2} V_{rms}"]) == [("right", True)]
    assert v(["V_{rms} = 10 V", "V_p = 15 V", "V_p = \\sqrt{2} V_{rms}"]) == [("wrong", False)]    # (another part's? for a human)


def test_a_wrong_expansion_in_a_chain():
    """'y = (x+1)^2 = x^2 + x + 1', '(a+b)^2 = a^2 + b^2', '2(x + 3) = 2x + 3': no verdict - a step between sides in letters that
    did not hold may be an equation to solve ('V_A = 10 - 2I = 4I + 2'); one side a product or power of sums, the other a sum,
    with a leading term in common, is an expansion written - wrong (the expansion shown)"""
    def v(t):
        return [(x[0], x[1], x[2]) for x in _page_verdicts([t])]
    got = v("y = (x+1)^2 = x^2 + x + 1")
    assert [g[:2] for g in got] == [("wrong step", True)] and "x²+2x+1" in got[0][2]
    for t in ("f = (a+b)^2 = a^2 + b^2", "y = 2(x + 3) = 2x + 3", "y = x^2 + 5x + 6 = (x+1)(x+6)"):
        assert [g[:2] for g in v(t)] == [("wrong step", True)], t
    for t in ("y = (x+1)^2 = x^2 + 2x + 1", "y = 2(x + 3) = 2x + 6", "V = I(R_1 + R_2) = I R_1 + I R_2"):
        assert [g[:2] for g in v(t)] == [("right", True)], t
    for t in ("V_A = 10 - 2I = 4I + 2", "V_A = 2(5 - I) = 4I + 2"):
        assert v(t) == [], t


def test_units_of_fields_speeds_bits_and_a_phasor_unit_before_its_angle():
    """'\\frac{8.85 \\cdot 10^{-12} F/m \\cdot 1 cm^2}{1 mm} = 0.885 pF' (F, m letters), '\\frac{100 m}{10 s} = 10 m/s' (m, s letters),
    '8 kHz \\cdot 8 bits = 64 kbps' (bits letters), '2 mA \\angle 30^\\circ \\cdot 5 k\\Omega' (2 times angle times 30): no verdict"""
    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    assert v("C = \\frac{\\epsilon_0 A}{d} = \\frac{8.85 \\cdot 10^{-12} F/m \\cdot 1 cm^2}{1 mm} = 0.885 pF") == [("right", True)]
    assert v("C = \\frac{8.85 \\cdot 10^{-12} F/m \\cdot 1 cm^2}{1 mm} = 1.2 pF") == [("wrong step", True)]
    assert v("E = 2 kV/m \\cdot 3 mm = 6 V") == [("right", True)]
    assert v("v = \\frac{d}{t} = \\frac{100 m}{10 s} = 10 m/s") == [("right", True)] and v("v = \\frac{100 m}{10 s} = 20 m/s") == [("wrong step", True)]
    assert v("v = \\frac{36 km}{0.5 h} = 72 km/h") == [("right", True)]
    assert v("R_b = f_s \\cdot N = 8 kHz \\cdot 8 bits = 64 kbps") == [("right", True)] and v("R_b = 8 kHz \\cdot 8 bits = 32 kbps") == [("wrong step", True)]
    assert v("V = 2 mA \\angle 30^\\circ \\cdot 5 k\\Omega = 10 \\angle 30^\\circ V") == [("right", True)]
    assert v("V = 2 mA \\angle 30^\\circ \\cdot 5 k\\Omega = 12 \\angle 30^\\circ V") == [("wrong step", True)]
    # (a conductivity's unit: '(\\Omega \\cdot cm)^{-1}' was letters)
    assert v("\\sigma = q N \\mu_n = 1.6 \\cdot 10^{-19} \\cdot 1 \\cdot 10^{15} \\cdot 1350 = 0.216 (\\Omega \\cdot cm)^{-1}") == [("right", True)]
    assert v("\\sigma = 1.6 \\cdot 10^{-19} \\cdot 1 \\cdot 10^{15} \\cdot 1350 = 0.3 (\\Omega \\cdot cm)^{-1}") == [("wrong step", True)]


def test_siemens_max_and_metres_under_a_force():
    """'\\frac{1}{2} + \\frac{1}{3} = 0.833 S' (S letters), '\\max(3, 5) = 5' (two statements), '\\frac{k q_1 q_2}{(0.3 m)^2} = 0.2 N'
    (m a letter): no verdict - while '(2m)^2 \\cdot 1k = 4 mW' stays milli"""
    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    assert v("G_{eq} = \\frac{1}{2} + \\frac{1}{3} = 0.833 S") == [("right", True)] and v("G = \\frac{1}{2} + \\frac{1}{3} = 0.9 S") == [("wrong step", True)]
    assert v("\\max(3, 5) = 5") == [("right", True)] and v("\\min(3, 5) = 5") == [("wrong step", True)]
    assert v("F = \\frac{k q_1 q_2}{r^2} = \\frac{9 \\cdot 10^9 \\cdot 1 \\mu C \\cdot 2 \\mu C}{(0.3 m)^2} = 0.2 N") == [("right", True)]
    assert v("F = \\frac{9 \\cdot 10^9 \\cdot 1 \\mu C \\cdot 2 \\mu C}{(0.3 m)^2} = 0.3 N") == [("wrong step", True)]
    assert v("P = I^2 R = (2m)^2 \\cdot 1k = 4 mW") == [("right", True)]


def test_a_slash_read_both_ways():
    """found by rewriting right lines the ways readers write them ('\\frac{a}{b}' as '(a)/(b)'): 'E = 1/2 C V^2' (a half
    times C V^2 - the parser takes 1/(2 C V^2), as it must '1/2\\pi f C'), 'V_o = R_2/(R_1 + R_2) V_{in}' (V_{in} times the
    ratio), '20 \\log(V_o)/(V_i)' (the log of the ratio), 's^2 + (R)/(L)s + (1)/(LC) = 0' were sure wrongs; a denominator its
    bracket closes is closed now, the other readings right not sure (and wrong either way still wrong); '\\arctan(40)/(30)
    = 53.13^\\circ' crashed"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["R_1 = 1 k\\Omega", "R_2 = 2 k\\Omega", "V_{in} = 9 V", "V_o = R_2/(R_1+R_2) V_{in}", "V_o = 6 V"]) == [("right", True)]
    assert v(["R_1 = 1 k\\Omega", "R_2 = 2 k\\Omega", "V_{in} = 9 V", "V_o = R_2/(R_1+R_2) V_{in}", "V_o = 3 V"]) == [("wrong", True)]
    assert v(["V_o = R_2/(R_1+R_2) V_{in} = 2/(1+2) 9 = 6 V"]) == [("right", True)]
    assert v(["f = 1/(2\\pi) \\sqrt{k/m}", "k = 100", "m = 1", "f = 1.59 Hz"]) == [("right", True)]
    assert v(["R = 2", "L = 1", "C = 0.5", "s^2 + (R)/(L)s + (1)/(LC) = 0", "s_1 = -1 + j", "s_2 = -1 - j"]) == [("right", True)] * 2
    # 1/2 then letters: a half (units agreeing only so: sure), or 1 over the product ('1/2RC' for alpha) - either
    assert v(["C = 10 \\mu F", "V = 5 V", "E = 1/2 C V^2", "E = 125 \\mu J"]) == [("right", True)]
    assert v(["C = 10 \\mu F", "V = 5 V", "E = 1/2 C V^2", "E = 250 \\mu J"]) == [("wrong", True)]
    assert v(["m = 2 kg", "v = 3 m/s", "E_k = 1/2 m v^2 = 9 J"]) == [("right", False)]
    got = _page_verdicts(["C = 10 \\mu F", "V = 5 V", "E = 1/2 C V^2 = 250 \\mu J"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "125" in got[0][2]          # (the other reading's value too)
    assert v(["\\alpha = 1/2RC", "R = 1 k\\Omega", "C = 1 \\mu F", "\\alpha = 500"]) == [("right", True)]
    assert v(["\\alpha = 1/2RC", "R = 1 k\\Omega", "C = 1 \\mu F", "\\alpha = 5 \\cdot 10^{-4}"]) == [("right", False)]
    assert v(["A_v = 20 \\log(V_o)/(V_i)", "V_o = 5 V", "V_i = 50 mV", "A_v = 40 dB"]) == [("right", False)]
    assert v(["A_v = 20 \\log(V_o)/(V_i)", "V_o = 5 V", "V_i = 50 mV", "A_v = 20 dB"]) == [("wrong", True)]
    assert v(["G = 10 \\log_{10} (2 W)/(2 mW) = 30 dB"]) == [("right", True)]
    assert v(["\\phi = \\tan^{-1}(X_L - X_C)/R", "X_L = 8 k\\Omega", "X_C = 4 k\\Omega", "R = 3 k\\Omega", "\\phi = 53.13°"]) == [("right", False)]
    assert v(["\\phi = \\arctan\\frac{X_L - X_C}{R} = \\arctan(40)/(30) = 53.13^\\circ"]) == [("right", True)]
    assert v(["\\lambda = \\ln(2)/(5730) = 1.21 \\cdot 10^{-4}"]) == [("right", True)]          # (the parser's own reading)
    assert v(["\\lambda = \\ln(2)/(5730) = 1.5 \\cdot 10^{-4}"]) == [("wrong step", True)]
    assert v(["P = V^2/R", "V = 10 V", "R = 5 \\Omega", "P = 2 W"]) == [("wrong", True)]
    # elsewhere on a page: an integral, a function's value, a derivative, a transfer function inline - right not sure
    # (they were sure wrongs), and wrong either way still wrong
    assert v(["\\int x dx = 1/2 x^2 + C"]) == [("right", False)] and v(["\\int x dx = 1/2 x^3 + C"]) == [("wrong step", True)]
    assert v(["f(x) = 1/2 x^2", "f(2) = 2"]) == [("right", False)] and v(["f(x) = 1/2 x^2", "f(2) = 3"]) == [("wrong step", True)]
    assert v(["y = 1/2 x^2", "y' = x"]) == [("right", False)] and v(["y = 1/2 x^2", "y' = 2x"]) == [("wrong step", True)]
    assert v(["H(s) = 1/(s+1)(s+2)", "H(0) = 0.5"]) == [("right", False)] and v(["H(s) = 1/(s+1)(s+2)", "H(0) = 1"]) == [("wrong step", True)]


def test_a_sentences_end_and_words_before_a_statement():
    """'I = 3 mA.' (a sentence's period) and 'I = 3 mA :תשובה' were read as I = 10 - false sure wrongs; 'נתון: V = 12 V' (a
    Hebrew word and a colon), '\\text{given } V = 12 V', 'so I = 3 mA', 'Solve: x^2 = 4' were unread or letters of a formula"""
    from grader.selfcheck import statements

    def vals(t):
        return [(str(s.symbol), s.number, s.unit) for s in statements(t) if s.kind == "value"]
    assert vals("I = 3 mA.") == vals("I = 3 mA :תשובה") == vals("so I = 3 mA") == vals("מכאן: I = 3 mA") == [("I", 3.0, "mA")]
    assert vals("נתון: V = 12 V, R = 4 k\\Omega") == vals("\\text{given } V = 12 V \\text{ and } R = 4 k\\Omega") == \
        [("V", 12.0, "V"), ("R", 4.0, "k\\Omega")]
    assert vals("x = 0.333...") == [("x", 0.333, "")] and vals("x = 3.") == [("x", 3.0, "")]
    assert [s.kind for s in statements("Solve: x^2 = 4")] == ["chain"]
    # the invisible direction marks Hebrew text carries when typed or pasted (read as I = 10, or a false wrong step)
    assert vals("I = 3 mA\u200f") == vals("I = 3 mA\u202c") == [("I", 3.0, "mA")]
    assert [x[:2] for x in _page_verdicts(["I = 12/4k\u200e = 3 mA"])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["V = 12 V.", "R = 4 k\\Omega.", "I = V/R", "I = 3 mA."])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["V = 12 V", "R = 4 k\\Omega", "I = V/R", "I = 4 mA."])] == [("wrong", True)]


def test_degrees_as_a_small_o_and_a_radical_with_brackets_inside():
    """'53.13^o', '30^{o}' (a degree mark as a small o: the line was unread); '√(30^2 + (80 - 40)^2) = 50' and '1/√(1 mH \\cdot
    1 µF)' (brackets inside the radical, its units put in: read as 10 - a false sure wrong)"""
    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    assert v("\\theta = \\arctan(4/3) = 53.13^o") == v("\\theta = \\arctan(4/3) = 53.13^{o}") == [("right", True)]
    assert v("\\theta = \\arctan(4/3) = 63.13^o") == [("wrong step", True)]
    assert v("P = V I \\cos\\phi = 230 \\cdot 10 \\cdot \\cos 36.87^o = 1840 W") == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["V = 10 \\angle 0^\\circ", "Z = 3 + j4", "I = V/Z = 2 \\angle -53.13^o A"])] == [("right", True)]
    assert v("Z = √(R^2 + (X_L - X_C)^2) = √(30^2 + (80 - 40)^2) = 50 Ω") == [("right", True)]
    assert v("Z = √(R^2 + (X_L - X_C)^2) = √(30^2 + (80 - 40)^2) = 60 Ω") == [("wrong step", True)]
    assert v("\\omega = 1/√(LC) = 1/√(1 mH \\cdot 1 µF) = 31623 rad/s") == [("right", True)]


def test_a_number_after_a_fraction():
    """'\\frac{2}{1+2} 12 = 8' (no product sign between): the parser took a fraction and a number after it for their sum
    ('\\frac{10}{2} 3' was 8), or split the number ('\\frac{10}{1} 0.5' was 10 times 0 times .5) - sure wrongs for right
    steps; a product now ('2/3·12' in messages), and '2 \\frac{1}{2}' still a mixed number"""
    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    assert v("V_o = \\frac{R_2}{R_1+R_2} V = \\frac{2}{1+2} 12 = 8 V") == [("right", True)]
    assert v("x = \\frac{1}{2} 10 = 5") == v("I = \\frac{10}{5} 2 = 4") == v("V_o = -R_f/R_1 V_i = -10k/1k 0.5 = -5 V") == [("right", True)]
    got = _page_verdicts(["V_o = \\frac{2}{3} 12 = 9 V"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "2/3·12" in got[0][2] and "8 is not 9" in got[0][2]
    assert v("x = 2 \\frac{1}{2} = 2.5") == [("right", True)]
    # a fraction after a fraction or a bracket (read as a mixed number: '\\frac{6}{3}\\frac{1}{2}' was 2 1/2), a decimal
    # after a root after a fraction (split: 0 times .25)
    from grader.selfcheck import _parse
    for t, want in (("\\frac{6}{3}\\frac{1}{2}", 1.0), ("(3)\\frac{1}{2}", 1.5), ("\\frac{2}{2}\\sqrt{4}0.25", 0.5), ("2\\frac{1}{2}", 2.5)):
        assert abs(float(_parse(t)) - want) < 1e-12, t


def test_a_result_and_its_data_listed_below_it():
    """'I = \\frac{V}{R} = 3 mA', then 'V = 12 V', 'R = 4 k\\Omega' (the data listed after the result): no verdict before; its
    steps with them now - a wrong one not sure (a value of the next part?)"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["I = \\frac{V}{R} = 3 mA", "V = 12 V", "R = 4 k\\Omega"]) == [("right", True)]
    assert v(["I = \\frac{V}{R} = 4 mA", "V = 12 V", "R = 4 k\\Omega"]) == [("wrong step", False)]
    assert v(["\\zeta = \\frac{R}{2}\\sqrt{\\frac{C}{L}} = 0.158", "R = 10", "C = 1 \\mu F", "L = 1 mH"]) == [("right", True)]


def test_braces_no_command_takes():
    """'I = {\\frac{12}{4k}} = 3 mA', 'I = {12} = 12' (braces a formula reader puts around a side) and '{12 \\over 4}': the side
    was not read - the line lost; '4{\\rm k}\\Omega' lost its unit. Brackets, a fraction, upright now (a braced row, '{ 2 x + y
    = 5 }', stays the row reader's)"""
    from grader.selfcheck import statements

    def v(t):
        return [x[:2] for x in _page_verdicts([t])]
    assert v("I={\\frac{12}{4\\mathrm{k}}}=3\\mathrm{mA}") == v("I = {12} = 12") == v("x = {12 \\over 4} = 3") == [("right", True)]
    assert v("I = {\\frac{12}{4k}} = 4 mA") == [("wrong step", True)]
    assert [(s.number, s.unit) for s in statements("R=4{\\rm k}\\Omega")] == [(4.0, "k\\Omega")]
    assert [(s.kind, s.text) for s in statements("{ 2 x + y = 5 }")] == [("formula", "2 x + y = 5")]
    # an array's braced cells inside '\\left( ... \\right)' stay a matrix ('\\left(' sizes only elsewhere: "f' \\left( x \\right)")
    m = ("\\left( \\begin{array} { c c } { 1 } & { 2 } \\\\ { 3 } & { 4 } \\end{array} \\right) ^ { - 1 } = \\left( \\begin{array} "
         "{ c c } { - 2 } & { 1 } \\\\ { 1 . 5 } & { - 0 . 5 } \\end{array} \\right)")
    assert v(m) == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["f \\left( x \\right) = x^2", "f' \\left( x \\right) = 2x"])] == [("right", True)]


def test_bold_and_font_switches_around_a_statement():
    """'\\textbf{I = 3 mA}', '\\mathbf{I = 3 mA}', '\\bf I = 3 mA' (a statement in bold: letters of a formula, or nothing);
    '\\textbf{Answer:} I = 3 mA' (a bold word before it); 'V_{\\textbf{out}}' (another letter than V_{out})"""
    from grader.selfcheck import statements

    def vals(t):
        return [(str(s.symbol), s.number, s.unit) for s in statements(t) if s.kind == "value"]
    for t in ("\\textbf{I = 3 mA}", "\\mathbf{I = 3 mA}", "\\bf I = 3 mA", "\\textbf{Answer:} I = 3 mA", "\\Large I = 3 mA"):
        assert vals(t) == [("I", 3.0, "mA")], t
    assert vals("V_{\\textbf{out}} = 5 V") == vals("V_{out} = 5 V") == [("V_{out}", 5.0, "V")]


def test_functions_of_functions_without_end_do_not_crash():
    """'1/\\sin \\sin \\sin ... x' (a reader's garbage, two thousand deep): the slash's other readings recursed past the
    stack - a crash; no factor there now"""
    from grader.selfcheck import check_statements, statements
    t = "y = 1/" + "\\sin " * 2000 + "x = 3"
    check_statements(statements(t))


def test_a_digital_circuits_solution_line_by_line():
    """an official solution of a CMOS exam, its math checked line by line - false sure wrongs: the nodes' values ('A = B =
    V_{DD} \\Rightarrow V_x = 0S, \\bar{V_x} = V_{DD}': strengths, states - judged as identities by the truth tables; 'V_x =
    1S' then 'V_x = 0S': another time; 'Y1 = 0S': Y and 1), a node defined ('X = A + B', then 'S = X + C = A + B + C'), the
    inputs' values put in ('A = 1, B = 0', then 'Y = \\overline{AB} = \\overline{1 \\cdot 0} = 1': against every row), the
    body effect ('\\gamma(\\sqrt{2|\\Phi_F| + V_{SB}} - \\sqrt{2|\\Phi_F|})': the gamma function of the bracket; '2|\\Phi_F| =
    0.6': no value for Φ_F)"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    page = ["F = \\bar{A} + B"]
    for lines in (["A = B = V_{DD} \\Rightarrow V_{x} = 0S,\\ \\bar{V_{x}} = V_{DD}"], ["V_{x} = 1S,\\ Y1 = 0S", "V_{x} = 0S"],
                  ["V_{x} = 0,\\ \\bar{V_{x}} = 1"], ["V_{x} = 1S,\\ \\bar{V_{x}} = 0S", "V_{x} = 0W"]):
        assert not any(x[0].startswith("wrong") for x in v(page + lines)), lines
    assert v(page + ["X = A + B", "S = X + C = A + B + C"]) == [("right", True)]
    got = _page_verdicts(page + ["X = A + B", "S = X + C = A + C"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "X = A+B" in got[0][2]
    assert v(page + ["Y = \\bar{S}A0 + SA1 = A1"]) == [("wrong step", True)]          # (A0, A1: a multiplexer's inputs)
    assert v(["A = 1, B = 0", "Y = \\overline{A B} = \\overline{1 \\cdot 0} = 1"]) == [("right", True)] * 2
    assert v(["Y = \\overline{A B}", "A = 1, B = 0", "Y = 1"]) == [("right", True)]
    assert v(["Y = \\overline{A B}", "A = 1, B = 1", "Y = 1"]) == [("wrong step", False)]    # (the values of another part?)
    assert v(["F = AB + A\\bar{B}", "F = 1"]) == [("wrong step", False)]                     # (F at some inputs?)
    assert v(["F = A + \\bar{A}", "F = 1"]) == [("right", True)]
    body = ["V_{TH} = V_{T0} + \\gamma\\left( \\sqrt{2 \\mid \\Phi_{F} \\mid + V_{SB}} - \\sqrt{2 \\mid \\Phi_{F} \\mid} \\right)",
            "V_{T0} = 0.43", "\\gamma = 0.4", "2|\\Phi_F| = 0.6", "V_{SB} = 1.151"]
    assert v(body + ["V_{TH} = 0.649"]) == [("right", True)] and v(body + ["V_{TH} = 0.75"]) == [("wrong", True)]
    assert v(["\\Gamma(5) = 24"]) == [("right", True)]


def test_values_at_an_instant():
    """'v_C(0^-) = 5 V', 'i_L(0^+)', 'v(\\infty)', 'Y_{2}( 0^{-} )' (a transient's values, a CMOS node's): the value was lost,
    and inside a step it was v_C times 0 - '\\frac{10 - V_C(0^{+})}{2 k\\Omega} = 3 mA', 'Y_2(0^-) = 1.8 - 0.649 = 1.151' false
    sure wrong steps; one name now, shown as written"""
    from grader.selfcheck import statements

    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert [(s.kind, s.number) for s in statements("v_C(0^-) = 5 V")] == [("value", 5.0)]
    assert v(["V_C(0^{+}) = 4 V", "I(0^{+}) = \\frac{10 - V_C(0^{+})}{2 k\\Omega} = 3 mA"]) == [("right", True)]
    got = _page_verdicts(["v_C(0^+) = 5 V", "i(0^+) = \\frac{12 - v_C(0^+)}{1 k\\Omega} = 6 mA"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "v_C(0⁺)" in got[0][2]
    assert v(["Y_{2}\\left( 0^{-} \\right) = 1.8 - 0.649 = 1.151V"]) == [("right", True)]
    assert v(["Y_{2}( 0^{-} ) = 1.151 V", "V_{BS} = 1.8 - Y_{2}(0^{-}) = 0.649 V"]) == [("right", True)]
    assert v(["v_C(0^-) = 5 V", "v_C(0^+) = v_C(0^-) = 5 V"]) == [("right", True)]
    assert v(["v(\\infty) = 10 V", "v(0^+) = 2 V", "\\Delta v = v(\\infty) - v(0^+) = 8 V"]) == [("right", True)]


def test_cmos_numbers_and_a_multiplexer_as_the_course_writes_them():
    """a multiplexer at its select's value ('S = 1', then 'Y = \\bar{1}D_0 + 1 \\cdot D_1 = D_1': the data inputs kept - against
    every row it was a false sure wrong step); the body effect's gamma in square-root volts ('0.4 V^{1/2}': read as √0.4;
    '0.4 \\sqrt{V}': no value); a unit squared ('\\frac{144 V^2}{4 \\Omega} = 36 W': 144² over 4); a bare f, femto ('0.69 \\cdot 2f
    \\cdot 5k = 6.9 ps': no verdict)"""
    from grader.selfcheck import statements

    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    mux = ["Y = \\bar{S}D_0 + SD_1", "S = 1"]
    assert v(mux + ["Y = \\bar{1}D_0 + 1 \\cdot D_1 = D_1"]) == [("right", True)] * 2
    assert ("wrong step", True) in v(mux + ["Y = \\bar{1}D_0 + 1 \\cdot D_1 = D_0"])
    assert v(["F = \\bar{S}A0 + SA1", "S = 0", "F = \\bar{0}A0 + 0 \\cdot A1 = A0"]) == [("right", True)] * 2
    for t in ("\\gamma = 0.4 V^{1/2}", "\\gamma = 0.4 \\sqrt{V}", "\\gamma = 0.4 \\, V^{0.5}"):
        assert [(s.kind, s.number) for s in statements(t)] == [("value", 0.4)], t
    body = ["V_{TH} = V_{T0} + \\gamma(\\sqrt{2\\phi_F + V_{SB}} - \\sqrt{2\\phi_F})", "V_{T0} = 0.4 V", "\\gamma = 0.4 V^{1/2}",
            "2\\phi_F = 0.6 V", "V_{SB} = 1 V"]
    assert v(body + ["V_{TH} = 0.596 V"]) == [("right", True)] and v(body + ["V_{TH} = 0.775 V"]) == [("wrong", True)]
    assert v(["P = \\frac{V^2}{R} = \\frac{144 V^2}{4 \\Omega} = 36 W"]) == [("right", True)]
    assert v(["P = \\frac{V^2}{R} = \\frac{144 V^2}{4 \\Omega} = 30 W"]) == [("wrong step", True)]
    assert v(["P = I^2 R = 4 mA^2 \\cdot 2 k\\Omega = 8 mW"]) == [("right", True)]
    assert v(["t_{p0} = 0.69 R_{eq} C_{in} = 0.69 \\cdot 2f \\cdot 5k = 6.9ps"]) == [("right", True)]
    assert v(["t_{p0} = 0.69 R_{eq} C_{in} = 0.69 \\cdot 2f \\cdot 5k = 69ps"]) == [("wrong step", True)]
    assert v(["\\omega = 2 \\pi f = 2 \\pi \\cdot 50 = 314"]) == [("right", True)]     # (f a frequency there)


def test_line_breaks_control_spaces_and_a_value_at_zero():
    """a line break the reader writes between two lines of work ('V = 12 V \\\\ R = 4 k\\Omega': a product of their sides -
    then 'I = V/R = 3 mA' was a false sure wrong), a control space ('Y_1 = 0.5V_{DD}\\ ': V_DD dropped - the value 0.5), a
    value at t = 0 of a function the page never defines ('v_C(0) = 5 V', then 'i(0) = \\frac{12 - v_C(0)}{1 k\\Omega} = 7 mA':
    no verdict)"""
    from grader.selfcheck import statements

    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert [s.kind for s in statements("I = 3 mA \\\\ V = 12 V")] == ["value", "value"]
    assert v(["V = 12 V \\\\ R = 4 k\\Omega", "I = \\frac{V}{R} = 3 mA"]) == [("right", True)]
    assert v(["V = 12 V \\\\ R = 4 k\\Omega", "I = \\frac{V}{R} = 4 mA"]) == [("wrong step", True)]
    assert v(["I = \\frac{12}{4k} \\\\ = 3 mA"]) == [("right", True)]
    assert v(["I = \\frac{V}{R} \\\\ = 12 / 4 k\\Omega", "= 3 mA"]) == [("right", True)]
    m = "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}^{-1} = \\begin{pmatrix} -2 & 1 \\\\ 1.5 & -0.5 \\end{pmatrix}"
    assert v([m]) == [("right", True)]
    assert [s.kind for s in statements("Y_1 = 0.5V_{DD}\\ ")] == ["formula"]
    assert [(s.kind, s.number) for s in statements("V = 5 \\ V")] == [("value", 5.0)]
    assert v(["v_C(0) = 5 V", "i(0) = \\frac{12 - v_C(0)}{1 k\\Omega} = 7 mA"]) == [("right", True)]
    got = _page_verdicts(["v_C(0) = 5 V", "i(0) = \\frac{12 - v_C(0)}{1 k\\Omega} = 6 mA"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "v_C(0)" in got[0][2]
    assert v(["v(\\infty) = 10 V", "v(0) = 2 V", "\\Delta v = v(\\infty) - v(0) = 8 V"]) == [("right", True)]
    assert v(["v_C(t) = 12 - 7 e^{-t/2}", "v_C(0) = 6 V"]) == [("wrong step", True)]     # (a function defined: its value)
    assert v(["f(x) = x^2 + 1", "f(0) = 1"]) == [("right", True)]


def test_a_wrong_result_shown_in_the_units_written():
    """a result in a prefixed unit, the side before it in letters whose values carry prefixes of their own ('C_L = 1 pF',
    'I_{avg} = 6.7 \\mu A', then 't_{pHL} = C_L \\frac{\\Delta V}{I_{avg}} = 52 ps' - ps for ns, as the lecturer marked a student):
    the message was '0.05224 is not 52' (the numbers as written, in no unit); the side's value in SI with a prefix that
    suits it now - '52.24 ns is not 52 ps'"""
    data = ["C_L = 1 pF", "\\Delta V = 0.35 V", "I_{avg} = 6.7 \\mu A"]
    got = _page_verdicts(data + ["t_{pHL} = C_L \\frac{\\Delta V}{I_{avg}} = 52 ps"])
    assert [x[:2] for x in got] == [("wrong step", True)] and "52.24 ns is not 52 ps" in got[0][2]
    assert [x[:2] for x in _page_verdicts(data + ["t_{pHL} = C_L \\frac{\\Delta V}{I_{avg}} = 52 ns"])] == [("right", True)]
    from grader.selfcheck import _in_prefix
    assert (_in_prefix(5.224e-8, "ps"), _in_prefix(4700, "kΩ"), _in_prefix(1, "cm")) == ("52.24 ns", "4.7 kΩ", None)
def test_picoseconds_in_a_substitution_and_logical_effort():
    """'t_{pd} = t_{p0} d = 6.9 ps \\cdot 3 = 20.7 ps' (a CMOS course's delays): ps inside a substitution was p times s -
    letters, no verdict (ms, µs and ns were read); logical effort's path effort, its N-th root, the stages' number"""
    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    assert v(["t_{pd} = t_{p0} d = 6.9 ps \\cdot 3 = 20.7 ps"]) == [("right", True)]
    assert v(["t_{pd} = 6.9 ps \\cdot 3 = 27 ps"]) == [("wrong step", True)]
    assert v(["t = 2 fs \\cdot 5 = 10 fs"]) == [("right", True)]
    assert v(["G = 2.22", "B = 1", "H = 16", "F = GBH = 2.22 \\cdot 1 \\cdot 16 = 35.6"]) == [("right", True)] * 2
    assert v(["\\hat{f} = F^{1/N} = 35.6^{1/3} = 3.29"]) == [("right", True)]
    assert v(["N = \\log_4 F = \\log_4 256 = 4"]) == [("right", True)]


def test_a_name_worked_out_again_and_names_with_min_max():
    """a quantity worked out again under other conditions ('|V_{Tp}| = ... = 0.537 V' at t = 0, then '|V_{Tp}| = 0.4 +
    0.4(\\sqrt{0.6 + 0.9} - \\sqrt{0.6}) = 0.580 V' at t_pHL, as the official solution does): its first step against the
    value above was a sure wrong - not sure now (a new value?), the chain's own steps as they are; subscripts with \\min,
    \\max ('V_{OH,\\max}', 'C_{g,\\min}': the statement lost) and an escaped underscore ('C_{in\\_ inv}': the name's letters
    lost - '0.69R_{eq}C_{in\\_ inv}' read as 0.69, a false wrong step)"""
    from grader.selfcheck import _parse_full, statements

    def v(lines):
        return [x[:2] for x in _page_verdicts(lines)]
    first = "|V_{Tp}| = 0.4 + 0.4(\\sqrt{0.6 + 0.649} - \\sqrt{0.6}) = 0.537 V"
    assert v([first, "|V_{Tp}| = 0.4 + 0.4(\\sqrt{0.6 + 0.9} - \\sqrt{0.6}) = 0.580 V"]) == \
        [("right", True), ("wrong step", False), ("right", True)]
    assert v(["V = 5", "V = 2 + 4 = 7"]) == [("wrong step", False), ("wrong step", True)]
    assert v(["V = 5", "V = 2 + 3 = 5"]) == [("right", True)] * 2
    assert [(s.kind, str(s.symbol)) for s in statements("V_{OH,\\max} = 1.8 V")] == [("value", "V_{OH,max}")]
    assert v(["x = \\frac{4C_{g,\\min}}{6C_{g,\\min}} = \\frac{2}{3}"]) == [("right", True)]
    assert v(["t_{p0} = 0.69R_{eq}C_{in\\_ inv} = 0.69 \\cdot 2f \\cdot 5k = 6.9ps"]) == [("right", True)]
    assert _parse_full("0.69R_{eq}C_{in\\_ inv}") == (None, None)    # (a name's letters lost: no expression, not 0.69)


def test_a_real_exams_delays_names_and_units_against_the_solution():
    """a real exam read again (exam 1, Uni-MuMER's readings): 'Psec' and '1 P F' (a handwritten p read as P), 'p s c c'
    (sec letter by letter) were no units; a chain whose first side was misread ('t\\Omega_2 = t P_{HL} = C_L ... = 51.86
    ps') gave no value to compare - its first side that is a name does now; 'V_T' against the solution's 'V_{Tp}',
    'V_{Tn}' (any, shown 'V_T?'); a value 1000 times off the key's but near it otherwise gets a prefix-slip note;
    '\\lim\\limits' lost its limit and '5.10^{9}' (5 times 10^9) was 5.1^9 - sure false wrongs in the same exam"""
    from grader.selfcheck import statements
    from tools.recheck_exams import against_key, key_values

    def u(t):
        return [(s.kind, s.unit, s.number) for s in statements(t)]
    assert u("t = 51.86 Psec") == u("t = 51.86 p s c c") == u("t = 51.86 ps") == [("value", "ps", 51.86)]
    assert u("C_L = 1 P F") == [("value", "pF", 1.0)]
    assert u("P = 5 P") == [("formula", "", None)]                      # (a letter P: not pico)
    kv = key_values(statements("t \\Omega _ { 2 } \\equiv t P _ { H L } = C _ { L } \\cdot \\frac { \\Delta v } { I _ { a v } } "
                               "= 1 P F \\cdot 0 . 6 v = 5 1 . 8 6 P s c c"))
    assert list(kv) == ["t_phl"] and abs(kv["t_phl"][0][0] - 51.86e-12) < 1e-15
    key = key_values(statements("t_{pHL} = 69.7 ns") + statements("|V_{Tp}| = 0.537 V") + statements("V_{Tn} = 0.649 V"))
    got = against_key(kv, key)
    assert got[0][3] is False and "prefix slip" in got[0][1]
    vt = against_key(key_values(statements("V_T = -0.3 V") + statements("V_{T} = 0.65 V")), key)
    assert [(n, a) for n, _, _, a in vt] == [("V_T?", False), ("V_T?", True)]
    assert against_key(key_values(statements("I = 2 mA")), key_values(statements("I_{avg} = 3 mA"))) == []
    # '\lim\limits_{x \to 0}' (where the limits are typeset): the limit was lost - a sure false wrong 'lim sinx/x = 1'
    assert [x[:2] for x in _page_verdicts(["\\lim \\limits _ { x \\rightarrow 0 } \\frac { \\sin x } { x } = 1"])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["\\sum\\nolimits_{n=1}^{3} n = 7"])] == [("wrong step", True)]
    # a dot for times before a power of ten ('T = 1/(5.10^9) = 200 ps', the same exam): it was 5.1 to the 9th, a sure wrong
    assert [x[:2] for x in _page_verdicts(["T = \\frac{1}{5.10^{9}} = 200 ps"])] == [("right", True)]
    assert [x[:2] for x in _page_verdicts(["y = 0.10^{2} = 0.01"])] == [("right", True)]     # (0.10: a decimal)


def test_a_big_vlm_as_second_formula_reader_merged_with_uni_mumer():
    """exam 1 read again with Qwen3-VL-8B (tools/reread_math_vlm.py): it reads much more of the handwriting than
    Uni-MuMER, and fills what it cannot read with something plausible - the same reading for many pieces, an answer in
    words, Hebrew written out; merge() puts its reading first only where it can be trusted. A chain naming the value
    twice ('t_{PD} = t_{PHL} = ... = 51.86 ps') gives both names to compare with the solution"""
    from grader.selfcheck import statements
    from tools.recheck_exams import key_values
    from tools.reread_math_vlm import merge

    def page(*pairs):
        return [{"info": {}, "lines": [[{"kind": "math", "box": [0, 0, 9, 9], "math2": [(v, -1.0)], "math2_mumer": [(m, -2.0)]}]
                                       for v, m in pairs]}]
    made_up = "\\tau_p = 0.69 \\cdot R \\cdot C = 0.69 \\cdot 100 \\cdot 10^3 \\cdot 100 \\cdot 10^{-12}"
    data = page(("V_T = - 0.3 V", "V _ { T } = - 0 . 3 V"), (made_up, "a"), (made_up, "b"), (made_up, "c"),
                ("There is no handwritten math in the image provided, only printed text", "d"),
                ("C_{L} = 1pF: \\text{יש להתחשב באפקט המצע}", "e"), ("2025-2026", "f"), ("2025-2026", "g"), ("2025-2026", "h"))
    assert merge(data) == {"vlm": 5, "mumer": 4}
    firsts = [segs[0]["math2"][0][0] for segs in data[0]["lines"]]
    assert firsts == ["V_T = - 0.3 V", "a", "b", "c", "d", "C_{L} = 1pF:", "2025-2026", "2025-2026", "2025-2026"]
    assert data[0]["lines"][0][0]["math2"][1][0] == "V _ { T } = - 0 . 3 V"      # (Uni-MuMER's after it)
    kv = key_values(statements("t_{PD} = t_{PHL} = C_L \\cdot \\frac{\\Delta V}{I_{av}} = 1pF \\cdot \\frac{0.6V}{1.565mA} = 51.86pscc"))
    assert sorted(kv) == ["t_pd", "t_phl"] and abs(kv["t_phl"][0][0] - 51.86e-12) < 1e-15


def test_before_a_blind_exam_made_up_endings_delays_by_value_and_an_estimate():
    """made ready before the first exam judged blind: a VLM's made-up ending ('= 200 psec x = 1': a letter set to one
    digit, found nowhere else) left out; a delay the solution names otherwise ('t_{p21}' for its 't_{NAND}') held
    against all its delays - like one of them, or like none (the sizing slip of exam 1's question 3: 12.075 ps where
    the solution has 6.9, 11.5, 20.7, 27.6); one value under two names judged once; an estimate of the points off
    (PER_FINDING a finding, fixed in advance) - 'not checked' when nothing in the question could be checked"""
    from grader.selfcheck import statements
    from tools.recheck_exams import PER_FINDING, against_key, estimate, key_values
    from tools.reread_math_vlm import drop_made_up_tail

    assert drop_made_up_tail("T = \\frac{1}{5.1 \\times 10^{-9}} = 200psec x = 1", "T = \\frac { 1 } { 5 . 1 0 } = 2 0 0 p s e c") == \
        "T = \\frac{1}{5.1 \\times 10^{-9}} = 200psec"
    assert drop_made_up_tail("t_{P21} = t_{P0} (1 + \\frac{3}{4}) = 7/4 t_{P0} = 12.075", "") .endswith("12.075")
    assert drop_made_up_tail("a = 2, b = 3", "a = 2, b") == "a = 2, b = 3"          # (b in the other reading too)
    key = key_values(statements("t_{p0} = 6.9 ps") + statements("t_{pd,F1} = 11.5 ps") + statements("t_{pd,NAND} = 20.7 ps")
                     + statements("V_{BS} = 0 V") + statements("t_{pHL} = 69.7 ns"))
    got = against_key(key_values(statements("t_{P21} = 12.075") + statements("t_{M} = 6.9 ps")), key)
    assert [(n, a) for n, _, _, a in got] == [("t_P21", None), ("t_M (as 6.9ps?)", True)]
    assert against_key(key_values(statements("V_{SGT} = 0.32")), key) == []          # (only delays: not every voltage)
    two = against_key(key_values(statements("t_{PD} = t_{PHL} = 1pF \\cdot \\frac{0.6V}{1.565mA} = 51.86ps")), key)
    assert [(n, a) for n, _, _, a in two] == [("t_PHL", False)]
    c = {"right": 2, "wrong_sure": 1, "wrong_unsure": 3, "other": 2, "agrees": 3, "differs": 2, "unlike": 0}
    assert estimate(c, 50) == (3 * PER_FINDING, 3, 11)
    assert estimate(dict(c, wrong_sure=40), 25)[0] == 25                             # (at most the question's points)
    assert estimate({k: 0 for k in c}, 25) == (None, 0, 0)                           # not checked
