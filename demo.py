"""Demo: grade ECE-style final answers from simulated OCR n-best readings.

In the real pipeline the `Reading` lists come from the handwriting recognizer
(n-best beam, or N sampled VLM transcriptions turned into an empirical distribution).
Here they are hand-written to show how confidence and routing behave.
"""
from grader import Criterion, Question, Reading, grade_question

QUESTIONS = {
    "rc_fc_formula": Question(
        "rc_fc_formula", "RC low-pass: write the cutoff frequency formula.",
        [Criterion("formula", "final_formula", 4, r"\frac{1}{2\pi RC}")]),
    "rc_fc_value": Question(
        "rc_fc_value", "R = 10 kΩ, C = 100 nF. Compute f_c.",
        [Criterion("value", "final_numeric", 6, "159.15 Hz", rel_tol=0.02,
                   prefix_error_credit=0.5, missing_unit_penalty=1)]),
    "divider": Question(
        "divider", "Voltage divider: V_out as a function of V_in, R1, R2.",
        [Criterion("formula", "final_formula", 5, r"V_{in}\frac{R_2}{R_1+R_2}")]),
    "rlc_res": Question(
        "rlc_res", "Series RLC, L = 10 mH, C = 1 µF: resonance frequency.",
        [Criterion("value", "final_numeric", 5, "1591.5 Hz", rel_tol=0.02)]),
}

CASES = [
    # (question id, student description, readings)
    ("rc_fc_formula", "correct, clean handwriting",
     [Reading(r"f_c = \frac{1}{2\pi RC}", 0.95), Reading(r"f_c = \frac{1}{2\pi R C}", 0.05)]),
    ("rc_fc_formula", "correct but written as (2πRC)^-1",
     [Reading(r"(2\pi RC)^{-1}", 1.0)]),
    ("rc_fc_formula", "forgot the 2π",
     [Reading(r"\frac{1}{RC}", 1.0)]),
    ("rc_fc_formula", "OCR unsure: RC vs R·C vs R^C",
     [Reading(r"\frac{1}{2\pi RC}", 0.55), Reading(r"\frac{1}{2\pi R\cdot C}", 0.30), Reading(r"\frac{1}{2\pi R^C}", 0.15)]),

    ("rc_fc_value", "159 Hz, clear",
     [Reading("159 Hz", 1.0)]),
    ("rc_fc_value", "159 vs 154 (9/4 confusion)",
     [Reading("159 Hz", 0.7), Reading("154 Hz", 0.3)]),
    ("rc_fc_value", "1.59 kHz (decimal slip)",
     [Reading("1.59 kHz", 1.0)]),
    ("rc_fc_value", "159, no unit",
     [Reading("159", 1.0)]),
    ("rc_fc_value", "unreadable unit",
     [Reading("159 Hz", 0.6), Reading("159 H?", 0.4)]),

    ("divider", "correct, different order",
     [Reading(r"\frac{R_2 V_{in}}{R_2 + R_1}", 1.0)]),
    ("divider", "R1/R2 swapped",
     [Reading(r"V_{in}\frac{R_1}{R_1+R_2}", 1.0)]),
    ("divider", "OCR unsure about the subscript 1 vs 2 (changes the answer!)",
     [Reading(r"V_{in}\frac{R_2}{R_1+R_2}", 0.5), Reading(r"V_{in}\frac{R_1}{R_1+R_2}", 0.5)]),

    ("rlc_res", "1.59 kHz",
     [Reading("1,59 kHz", 1.0)]),
    ("rlc_res", "wrote rad/s instead of Hz (10 krad/s)",
     [Reading("10 krad/s", 1.0)]),
]


def main() -> None:
    print(f"{'question':15} {'case':52} {'score':>9} {'E[score]':>9} {'conf':>6}  route")
    print("-" * 110)
    for qid, desc, readings in CASES:
        q = QUESTIONS[qid]
        r = grade_question(q, readings)
        print(f"{qid:15} {desc:52} {r.most_likely_score:>5.1f}/{r.max_points:<3.0f} "
              f"{r.expected_score:>9.2f} {r.confidence:>6.0%}  {r.route:6} ({r.reason})")
        for pr in r.per_reading:
            for cr in pr["criteria"]:
                print(f"{'':15}   p={pr['prob']:.2f}  {pr['text']!r:40} -> {cr['status']:14} {cr['explanation']}")
        print()


if __name__ == "__main__":
    main()
