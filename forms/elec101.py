"""'חשמל 101' (Electricity 101) - an invented introductory course with five exams, for trying the
whole answer-box route end to end. Every reference answer is computed here, never typed.

A question: {"id", "points", "text", "kind": "number" | "choice" | "open", ...}
  number: "answer" (float), "unit" (printed after the cells), "cells" (default 8), "solution"
  choice: "options" [4 strings], "answer" (1-4: the right option's number)
  open:   "lines" (height of the answer area) - graded by a human
Answers are accepted within REL_TOL (students round to two decimals).
"""
from __future__ import annotations

import math

E_CHARGE = 1.602176634e-19
REL_TOL = 0.01
COURSE = {"id": "elec101", "name": "חשמל 101", "name_en": "Electricity 101",
          "semester": "סמסטר א' תשפ\"ז", "rel_tol": REL_TOL,
          "wrong_to_human": False}     # True: a wrong final answer goes to a human (partial credit for the work)


def _r(x, n=6):
    return float(f"{x:.{n}g}")


def exam1():
    return {"id": "E1", "title": "בוחן 1 - מטען, זרם, מתח והתנגדות", "minutes": 45, "date": "2026-11-10",
            "pages": [["q1", "q2"], ["q3", "q4"], ["q5", "q6"]],
            "questions": [
                {"id": "q1", "points": 15, "kind": "number", "unit": "A", "answer": _r(36 / 15),
                 "text": "דרך חתך של מוליך עובר מטען של 36 C במשך 15 שניות. מהו הזרם במוליך?",
                 "solution": "I = Q / t = 36 / 15 = 2.4 A"},
                {"id": "q2", "points": 15, "kind": "number", "unit": "mA", "answer": _r(12 / 470 * 1000),
                 "text": "נגד של 470 Ω מחובר למקור מתח של 12 V. מהו הזרם בנגד? (במיליאמפר)",
                 "solution": "I = V / R = 12 / 470 = 0.02553 A = 25.53 mA"},
                {"id": "q3", "points": 15, "kind": "number", "unit": "Ω", "answer": _r(9 / 0.25),
                 "text": "דרך נגד זורם זרם של 0.25 A, והמתח עליו 9 V. מהי התנגדות הנגד?",
                 "solution": "R = V / I = 9 / 0.25 = 36 Ω"},
                {"id": "q4", "points": 20, "kind": "number", "unit": "Ω", "answer": _r(1.72e-8 * 50 / 1.5e-6),
                 "text": "תיל נחושת באורך 50 m ושטח החתך שלו 1.5 mm². ההתנגדות הסגולית של נחושת היא "
                         "ρ = 1.72·10⁻⁸ Ω·m. מהי התנגדות התיל?",
                 "solution": "R = ρL / A = 1.72·10⁻⁸ · 50 / 1.5·10⁻⁶ = 0.573 Ω"},
                {"id": "q5", "points": 20, "kind": "number", "unit": "J", "answer": _r(1.5 * 3600),
                 "text": "סוללה של 1.5 V מעבירה מטען כולל של 3600 C עד שהיא מתרוקנת. כמה אנרגיה היא סיפקה?",
                 "solution": "W = Q · V = 3600 · 1.5 = 5400 J"},
                {"id": "q6", "points": 15, "kind": "choice", "answer": 4,
                 "text": "המתח על נגד קבוע גדל פי 2. מה קורה לזרם בנגד?",
                 "options": ["קטן פי 2", "לא משתנה", "גדל פי 4", "גדל פי 2"],
                 "solution": "לפי חוק אוהם I = V / R: הזרם גדל פי 2 (תשובה 4)"},
            ]}


def exam2():
    return {"id": "E2", "title": "בוחן 2 - הספק ואנרגיה", "minutes": 45, "date": "2026-11-24",
            "pages": [["q1", "q2"], ["q3", "q4"], ["q5", "q6"]],
            "questions": [
                {"id": "q1", "points": 15, "kind": "number", "unit": "A", "answer": _r(2000 / 230),
                 "text": "קומקום חשמלי בהספק 2000 W מחובר לרשת 230 V. מהו הזרם שהקומקום צורך?",
                 "solution": "I = P / V = 2000 / 230 = 8.70 A"},
                {"id": "q2", "points": 15, "kind": "number", "unit": "Ω", "answer": _r(230 ** 2 / 2000),
                 "text": "מהי ההתנגדות של גוף החימום בקומקום מהשאלה הקודמת (באותם תנאים)?",
                 "solution": "R = V² / P = 230² / 2000 = 26.45 Ω"},
                {"id": "q3", "points": 15, "kind": "number", "unit": "W", "answer": _r(0.3 ** 2 * 100),
                 "text": "בנגד של 100 Ω זורם זרם של 0.3 A. מהו ההספק המתפתח בנגד?",
                 "solution": "P = I²R = 0.09 · 100 = 9 W"},
                {"id": "q4", "points": 20, "kind": "number", "unit": "kWh", "answer": _r(1.5 * 6 * 30),
                 "text": "מזגן בהספק 1.5 kW פועל 6 שעות ביום במשך 30 ימים. כמה אנרגיה הוא צורך בסך הכול?",
                 "solution": "E = P · t = 1.5 · 6 · 30 = 270 kWh"},
                {"id": "q5", "points": 20, "kind": "number", "unit": "₪", "answer": _r(3 * 2.5 * 0.62),
                 "text": "תנור חשמלי בהספק 3 kW פועל במשך שעתיים וחצי. מחיר קילוואט-שעה הוא 0.62 ₪. "
                         "מה עלות ההפעלה?",
                 "solution": "E = 3 · 2.5 = 7.5 kWh;  7.5 · 0.62 = 4.65 ₪"},
                {"id": "q6", "points": 15, "kind": "choice", "answer": 1,
                 "text": "שתי נורות, 60 W ו-100 W, מחוברות במקביל לרשת 230 V. באיזו מהן זורם זרם גדול יותר?",
                 "options": ["בנורת 100 W", "בנורת 60 W", "אותו זרם בשתיהן", "אי אפשר לדעת בלי ההתנגדויות"],
                 "solution": "המתח שווה, I = P / V: בנורת 100 W (תשובה 1)"},
            ]}


def exam3():
    par = 1 / (1 / 100 + 1 / 220 + 1 / 330)
    i_src = 12 / (1e3 + 1 / (1 / 2e3 + 1 / 2e3))
    return {"id": "E3", "title": "בוחן 3 - חיבור טורי ומקבילי", "minutes": 45, "date": "2026-12-08",
            "pages": [["q1", "q2"], ["q3", "q4"], ["q5", "q6"]],
            "questions": [
                {"id": "q1", "points": 15, "kind": "number", "unit": "Ω", "answer": _r(100 + 220 + 330),
                 "text": "שלושה נגדים, 100 Ω, 220 Ω ו-330 Ω, מחוברים בטור. מהי ההתנגדות השקולה?",
                 "solution": "R = 100 + 220 + 330 = 650 Ω"},
                {"id": "q2", "points": 15, "kind": "number", "unit": "Ω", "answer": _r(par),
                 "text": "אותם שלושה נגדים מחוברים במקביל. מהי ההתנגדות השקולה?",
                 "solution": "1/R = 1/100 + 1/220 + 1/330  =>  R = 56.90 Ω"},
                {"id": "q3", "points": 20, "kind": "number", "unit": "mA", "answer": _r(i_src * 1000),
                 "diagram": "series_parallel",
                 "text": "במעגל שבאיור E = 12 V, R1 = 1 kΩ, R2 = R3 = 2 kΩ. מהו הזרם שמספק המקור? (במיליאמפר)",
                 "solution": "R2 ∥ R3 = 1 kΩ;  R = 1 + 1 = 2 kΩ;  I = 12 / 2k = 6 mA"},
                {"id": "q4", "points": 20, "kind": "number", "unit": "V", "answer": _r(12 - i_src * 1e3),
                 "text": "באותו מעגל: מהו המתח על הנגד R2?",
                 "solution": "V(R2) = E - I·R1 = 12 - 6 = 6 V"},
                {"id": "q5", "points": 15, "kind": "number", "unit": "V", "answer": _r(9 * 6.8 / (3.3 + 6.8)),
                 "diagram": "divider",
                 "text": "במחלק המתח שבאיור E = 9 V, R1 = 3.3 kΩ, R2 = 6.8 kΩ. מהו מתח היציאה Vout (המתח על R2)?",
                 "solution": "Vout = E · R2 / (R1 + R2) = 9 · 6.8 / 10.1 = 6.06 V"},
                {"id": "q6", "points": 15, "kind": "choice", "answer": 3,
                 "text": "בחיבור טורי של נגדים, איזה גודל שווה בכל הנגדים?",
                 "options": ["המתח", "ההספק", "הזרם", "ההתנגדות"],
                 "solution": "בחיבור טורי עובר אותו זרם בכל הנגדים (תשובה 3)"},
            ]}


def kirchhoff_circuit(e1=10.0, e2=5.0, r1=2.0, r2=4.0, r3=1.0):
    """node A between R1 (to E1), R2 (to ground) and R3 (to E2); I1: E1 -> A, I3: E2 -> A, I2: A -> ground"""
    va = (e1 / r1 + e2 / r3) / (1 / r1 + 1 / r2 + 1 / r3)
    return {"VA": va, "I1": (e1 - va) / r1, "I2": va / r2, "I3": (e2 - va) / r3}


def exam4():
    k = kirchhoff_circuit()
    return {"id": "E4", "title": "בוחן 4 - חוקי קירכהוף", "minutes": 45, "date": "2026-12-22",
            "pages": [["q1", "q2"], ["q3", "q4"], ["q5", "q6"]],
            "questions": [
                {"id": "q1", "points": 20, "kind": "number", "unit": "V", "answer": _r(k["VA"]), "diagram": "kirchhoff",
                 "text": "במעגל שבאיור E1 = 10 V, E2 = 5 V, R1 = 2 Ω, R2 = 4 Ω, R3 = 1 Ω. "
                         "מהו המתח בצומת A ביחס לנקודת הייחוס (האדמה)?",
                 "solution": "KCL בצומת A:  (10 - VA)/2 + (5 - VA)/1 = VA/4  =>  VA = 40/7 = 5.71 V"},
                {"id": "q2", "points": 15, "kind": "number", "unit": "A", "answer": _r(k["I2"]),
                 "text": "מהו הזרם I2 בנגד R2?",
                 "solution": "I2 = VA / R2 = 5.714 / 4 = 1.43 A"},
                {"id": "q3", "points": 15, "kind": "number", "unit": "A", "answer": _r(k["I1"]), "diagram": "kirchhoff",
                 "text": "(אותו מעגל, האיור חוזר כאן) מהו הזרם I1 בכיוון המסומן באיור?",
                 "solution": "I1 = (E1 - VA) / R1 = (10 - 5.714) / 2 = 2.14 A"},
                {"id": "q4", "points": 20, "kind": "number", "unit": "A", "answer": _r(k["I3"]),
                 "text": "מהו הזרם I3 בכיוון המסומן באיור? שים לב לסימן: אם הזרם זורם הפוך לחץ, התשובה שלילית.",
                 "solution": "I3 = (E2 - VA) / R3 = (5 - 5.714) / 1 = -0.71 A"},
                {"id": "q5", "points": 15, "kind": "number", "unit": "A", "answer": _r(3 + 2.5 - 1.2),
                 "text": "לצומת נכנסים שני זרמים, 3 A ו-2.5 A. מהצומת יוצאים זרם של 1.2 A וזרם נוסף I. מהו I?",
                 "solution": "KCL: 3 + 2.5 = 1.2 + I  =>  I = 4.3 A"},
                {"id": "q6", "points": 15, "kind": "choice", "answer": 3,
                 "text": "חוק המתחים של קירכהוף (KVL) נובע מ:",
                 "options": ["שימור המטען", "חוק אוהם", "שימור האנרגיה", "חוק קולון"],
                 "solution": "סכום המתחים בלולאה סגורה הוא אפס: שימור האנרגיה (תשובה 3)"},
            ]}


def exam5():
    return {"id": "E5", "title": "מבחן מסכם", "minutes": 120, "date": "2027-02-02",
            "pages": [["q1", "q2", "q3"], ["q4", "q5", "q6"], ["q7", "q8"]],
            "questions": [
                {"id": "q1", "points": 12, "kind": "number", "unit": "mC", "answer": _r(100e-6 * 12 * 1e3),
                 "text": "קבל של 100 µF טעון למתח 12 V. מהו המטען על הקבל? (במיליקולון)",
                 "solution": "Q = C · V = 100 µF · 12 V = 1.2 mC"},
                {"id": "q2", "points": 12, "kind": "number", "unit": "mJ", "answer": _r(0.5 * 100e-6 * 12 ** 2 * 1e3),
                 "text": "מהי האנרגיה האגורה בקבל מהשאלה הקודמת? (במיליג'אול)",
                 "solution": "E = ½ C V² = 0.5 · 100 µF · 144 = 7.2 mJ"},
                {"id": "q3", "points": 12, "kind": "number", "unit": "s", "answer": _r(10e3 * 47e-6),
                 "text": "במעגל RC הנגד הוא R = 10 kΩ והקבל C = 47 µF. מהו קבוע הזמן τ של המעגל?",
                 "solution": "τ = R · C = 10 kΩ · 47 µF = 0.47 s"},
                {"id": "q4", "points": 12, "kind": "number", "unit": "V", "answer": _r(230 * math.sqrt(2)),
                 "text": "מתח הרשת הוא 230 V (ערך אפקטיבי, RMS). מהו מתח השיא?",
                 "solution": "Vp = √2 · 230 = 325.27 V"},
                {"id": "q5", "points": 12, "kind": "number", "unit": "ms", "answer": _r(1 / 50 * 1e3),
                 "text": "תדר הרשת הוא 50 Hz. מהו זמן המחזור? (במילישניות)",
                 "solution": "T = 1 / f = 1 / 50 = 20 ms"},
                {"id": "q6", "points": 15, "kind": "number", "unit": "W", "answer": _r(230 ** 2 / 46),
                 "text": "גוף חימום של 46 Ω מחובר לרשת 230 V (RMS). מהו ההספק הממוצע שלו?",
                 "solution": "P = V² / R = 230² / 46 = 1150 W"},
                {"id": "q7", "points": 10, "kind": "choice", "answer": 2,
                 "text": "במעגל זרם ישר, לאחר זמן רב, קבל מתנהג כמו:",
                 "options": ["קצר (חוט)", "נתק", "נגד קטן", "מקור זרם"],
                 "solution": "הקבל נטען עד שלא עובר בו זרם: נתק (תשובה 2)"},
                {"id": "q8", "points": 15, "kind": "open", "lines": 6,
                 "text": "הסבר בקצרה מה תפקידו של מפסק פחת בלוח החשמל הביתי, ומתי הוא מנתק את המעגל.",
                 "solution": "מודד הפרש בין הזרם בפאזה לזרם באפס; זליגה (למשל דרך גוף אדם) -> מנתק (בדרך כלל ב-30 mA)"},
            ]}


EXAMS = [exam1, exam2, exam3, exam4, exam5]


def all_exams():
    out = [f() for f in EXAMS]
    for ex in out:
        assert sum(q["points"] for q in ex["questions"]) == 100, ex["id"]
        ids = [q["id"] for q in ex["questions"]]
        assert sorted(sum(ex["pages"], [])) == sorted(ids), ex["id"]
    return out


if __name__ == "__main__":
    for ex in all_exams():
        print(ex["id"], ex["title"])
        for q in ex["questions"]:
            print("  ", q["id"], q["points"], q["kind"], q.get("answer"), q.get("unit", ""))
