# exam-grader — local prototype: handwriting → reading with probabilities → grade + confidence

Everything runs on this machine. Models live in `models/`, data in `data/`; after `setup.cmd`
no network is needed (the evaluations run with `HF_HUB_OFFLINE=1`).

## Setup (Windows)

Double-click **`setup.cmd`**. It logs everything to `setup.log` and:

1. detects the GPU / CUDA driver and creates `.venv` (Python 3.13) with the matching torch build
2. installs `requirements.txt`
3. downloads the formula and English models → `models/pix2text-mfr`, `models/trocr-small-handwritten`
   and the public datasets → `data/`
4. runs the unit tests and the formula evaluation on 70 real handwritten formulas
5. trains the Hebrew line recognizer → `models/hebrew-crnn` (GPU: ~15–30 min) and evaluates it on
   300 lines from writers it never saw

Re-running is safe (finished steps are skipped). To retrain Hebrew, delete `models/hebrew-crnn`.

## How it fits together

```
image of an answer box
   │  recog/formula.py   Pix2Text MFR on ONNX Runtime        (numbers + formulas, LaTeX out)
   │  recog/hebrew.py    CRNN-CTC on ONNX Runtime            (Hebrew text lines)
   │                     + Hebrew word model & course vocabulary (models/hebrew-lm)
   │  recog/english.py   TrOCR-small on ONNX Runtime         (English words; TrOCR-base: EXAM_GRADER_ENGLISH=base)
   │  recog/mixed_reader.py, recog/page.py: which part of a line / page is Hebrew, English or math
   ▼
n-best readings with total log-probabilities
   ▼
grader/  — each reading graded deterministically against the rubric
           (pint units + tolerance, SI-prefix / 2π slips, SymPy formula equivalence)
   ▼
distribution over scores → expected score, confidence, route = auto | human review
```

Reading uncertainty only matters when it changes the score: "4.7" vs "4.70" → confident;
"4.7" vs "4.1" → human review.

## Results so far

**Latest (28.10, morning)** - a real course's official solution checked by the grader (a digital-circuits exam: CMOS, Boolean algebra; its formulas only, as text): every false verdict it showed, fixed - then 130 random pages of that course's kinds of work and 60 switched circuits, none judged wrong for sure:
- *The body effect*: 'V_{TH} = V_{T0} + \gamma\left( \sqrt{2|\Phi_F| + V_{SB}} - \sqrt{2|\Phi_F|} \right)' - a lowercase gamma before a bracket was the gamma function of the bracket (V_TH = 2.05 - a false sure wrong for 0.649); '2|\Phi_F| = 0.6' gave no value for Φ_F ('cannot check'): the size is worked out of it now; gamma in square-root volts ('\gamma = 0.4 V^{1/2}' was read as √0.4, '0.4 \sqrt{V}' as no value).
- *A value at an instant* ('Y_2(0^-) = 1.8 - 0.649 = 1.151 V', 'v_C(0^+) = v_C(0^-) = 5 V', 'i_L(0^+)', 'v(\infty)'): the value was lost, and inside a step the name was a product with 0 ('V_{BS} = 1.8 - Y_2(0^-) = 0.649 V', '\frac{10 - V_C(0^{+})}{2 k\Omega} = 3 mA': false sure wrong steps). One name now, shown as written ('v_C(0⁺)'): a switched RC or RL circuit's page is checked step by step.
- *Boolean algebra as a CMOS course writes it*: the nodes' states ('A = B = V_{DD} \Rightarrow V_x = 0S, \bar{V_x} = V_{DD}' - strengths and values were judged as identities by the truth tables; 'V_x = 1S', then 'V_x = 0S' at another time: '1 = 0'), indexed names ('Y1 = 0S' was Y times 1; 'A0', 'D_1'), a node defined then used ('X = A + B', then 'S = X + C = A + B + C': X a letter of its own), the inputs' values put in ('A = 1, B = 0', then 'Y = \overline{AB} = \overline{1 \cdot 0} = 1'; a multiplexer at its select's value, 'Y = \bar{1}D_0 + 1 \cdot D_1 = D_1': against every row) - false sure wrongs, all. A wrong value at the values written is not sure now (the values of another part?), and 'F = AB + A\bar{B}' then 'F = 1' is wrong, not sure (its value at some inputs?).
- *More units*: a unit squared ('\frac{144 V^2}{4 \Omega} = 36 W' was 144² over 4 - a false sure wrong; '4 mA^2 \cdot 2 k\Omega'), a bare f, femto, as the solution writes a capacitance ('0.69 \cdot 2f \cdot 5k = 6.9 ps': no verdict).
- *Two lines in one reading*: a line break the formula reader writes in a crop of two lines ('V = 12 V \\ R = 4 k\Omega') was a product of their sides - then 'I = \frac{V}{R} = 3 mA' a false sure wrong ('12 is not 3'). The lines apart now (a line that begins with '=' goes on from the one before; an aligned block's or a matrix's rows as before): in the test batteries' 249 cases of several lines, two of them in one reading give the verdicts of the lines apart (43 of 80 random such pages did not).
- *A control space* ('Y_1 = 0.5V_{DD}\ ', '\mathbf{\ }' as a document's converter writes it): V_DD before it was dropped - 'Y_1 = 0.5', a false value. A space now.
- *A value at t = 0* of a function the page never defines ('v_C(0) = 5 V', then 'i(0) = \frac{12 - v_C(0)}{1 k\Omega} = 7 mA'): no value, no verdict; one name now, as 'v_C(0^+)'. A function the page defines ('v_C(t) = 12 - 7e^{-t/\tau}', 'f(x) = ...', 'H(s) = ...') keeps its value worked out.
- *One real exam against the lecturer's sheet, remarks and the official solution* (tools): `tools/exam_questions.py` - which question each scanned page answers, by the exam's own booklet (every page matched to the booklet's printed pages, in order; the backs follow their fronts); `tools/read_exams.py` keeps the second formula reader's readings (Uni-MuMER's - they were lost from the saved readings) and reads on several GPUs at once (`--shard`; read_exams.cmd starts a reader per GPU); `tools/recheck_exams.py --pages --grades --solution`: each question's pages judged together, beside the points the lecturer took off, and each value the student names against the official solution's ('Y_2(0⁻) = 1.5V; the solution: 1.151V' - a slip in a starting value is right step by step, wrong against the key).
- *A wrong result shown in the units written*: 't_{pHL} = C_L \frac{\Delta V}{I_{avg}} = 52 ps' with 'C_L = 1 pF', 'I_{avg} = 6.7 \mu A' (ps for ns - a slip the lecturer marked in the real exam) was 'wrong: 0.05224 is not 52', the numbers as written in no unit; '52.24 ns is not 52 ps' now.
- *Picoseconds and femtoseconds inside a substitution* ('t_{pd} = t_{p0} d = 6.9 ps \cdot 3 = 20.7 ps', a CMOS course's delays): p times s - letters, no verdict (ms, µs and ns were read). Logical effort along a path (G, H, F = GBH, F^{1/N}, D = N\hat{f} + P, \log_4 F) is in the random digital-circuits pages now.
- *A quantity worked out again* under other conditions ('|V_{Tp}| = ... = 0.537 V' at t = 0, then '|V_{Tp}| = 0.4 + 0.4(\sqrt{0.6 + 0.9} - \sqrt{0.6}) = 0.580 V' at t_pHL, as the official solution does): the chain's first step was judged against the value above - a sure wrong step; not sure now ('|V_Tp| was 0.537 above: a new value of it here?'), the chain's own steps as before.
- *Names with \min, \max in a subscript* ('V_{OH,\max}', 'C_{g,\min}'): the statement was lost. *An escaped underscore inside a name* ('C_{in\_ inv}', a document converter's): the parser took a piece of the side for all of it - '0.69R_{eq}C_{in\_ inv}' read as 0.69, a false sure wrong step; a side whose names' letters the parser loses is no expression now. The official solution, read again with its table cells (243 pieces of math): no sure wrong in it.

`tools/check_fuzz.py`: 369 + 132 derivations, 60 random quotients with the unit after the divisor's number only (with a wrong version of each, and the result bare 1000 apart - never a sure wrong), and a resistor's current written 'V / R k\Omega' against the circuit, 40 random two-mesh systems written in kΩ, in ohms or with their units and answered in mA, in amperes or bare (with a wrong version of each), 400 random EE computations across templates and renderings, 200 random formula-data-answer problems, 300 random phasor computations with their units written every way, 200 random semiconductor computations in centimetres, 200 random problems with angles (power factor, phasor parts; a formula with prefixed data or a chain) (with a wrong version of each), 150 random pages of work over several lines, values flowing on by name (with one slip in most), 150 random formula problems with a value worked out on its own line and 150 with two formulas, the second defining a letter of the first, 120 random pages of two or three parts reusing letters (a part done again among them), 100 random answers in marked sub-questions carrying values on, 80 random lines done again and equations' roots, 100 random electronics answers rounded on the way, 150 random Boolean pages (identities and the classic wrong ones, complements every way), 80 random numbers in bases, 100 random frequency responses at the frequencies written, 150 random expansions and equations along a chain, 150 random formulas with slashes inline (with a wrong version of each), 400 random products written side by side, 150 random pages with Hebrew words around the math, 150 random computations typed as plain text, 130 random pages of a digital-circuits exam (the nodes' states, indexed names, nodes defined, the inputs' values put in, a multiplexer, a decoder; the body effect, the square law, delays, noise margins, power), 60 random switched circuits with values at instants, 80 random pages with two lines of work in one reading; 338 tests.

**Earlier (28.10, early morning)** - more false verdicts, found the same way (work written as students write it, across EE topics: signals, electromagnetics, semiconductors, control, power, op-amps, transmission lines):

- *A unit after a divisor's number* ('I = \frac{V}{R} = 12 / 4 k\Omega = 3 mA', 'f = 1/T = 1 / 2 ms = 500 Hz', '\lambda = c/f = 3 \cdot 10^8 / 100 MHz = 3 m' - the commonest way a quotient is written) was taken for the whole side's: (12/4) kΩ, a false sure wrong step. The unit is the divisor's now; the whole side's reading stays where the next side holds only so ('V = 10/2 mV = 5 mV'). A result written bare a power of 1000 off after a side with units ('= 12 / 4 k\Omega = 3': 3 for 3 mA) is wrong, not sure.
- *\Gamma and \gamma* (a reflection coefficient, a propagation constant) were read as Euler's constant 0.577: 'VSWR = \frac{1 + |\Gamma|}{1 - |\Gamma|} = \frac{1.333}{0.667}' and even '\Gamma = 0.333' were false sure wrong steps, '\Gamma_L' was not read at all. They are letters now (the gamma function stays: '\Gamma(5) = 4!').
- *A root written twice* ('r_1 = -1', 'r_2 = -1' for r^2 + 3r + 2 = 0: each a root, -2 missing) was right twice; the second is wrong now (a double root may be written twice). Roots named apart with the data written before the equation ('R = 2 \Omega', 'L = 1 mH', 'C = 0.5 mF', then 's^2 + \frac{R}{L}s + \frac{1}{LC} = 0', 's_1 = -1000 + j1000') are checked (no verdict before) - in SI units, and right only as written past the prefixes is for a human.
- *A computation in degrees with its result marked* ('PM = 180 - 135 = 45^\circ', '\theta_a = \frac{180}{3} = 60^\circ') - the mark made the result radians: a false sure wrong step. A trig function's value or pi stays in radians ('\arctan(1) = 0.785^\circ' is wrong), and a wrong step says it in degrees ('45° is not 50°').
- *A value written as a quotient with its units* ('I = 12 / 4 k\Omega' alone) was I = 3 kΩ - every formula after it a false sure wrong ('P = I^2 R ... = 36 mW'): a current is no kΩ, so the unit is the divisor's (the name decides: I, V, R, Z, X, G, f, t, \tau, \lambda, \omega - not E, P, L, C, whose names mean more than one thing). Numbers with their units worked out ('I = \frac{12}{4 k\Omega}', 'V = 2 mA \cdot 5 k\Omega') are values in SI units now, not formulas in the units' letters.
- *Right only as written past a prefix written above* ('R = 5 k\Omega', then 'I = \frac{10}{R} = 2 A' - 10/5 is 2, but the current is 2 mA) was a sure right: right, not sure now, for a human; '= 2 mA', '= 0.002 A' and a bare '= 2' stay sure. A complex result with a prefix in its unit was compared in amperes with the tolerance of its decimals ('= 1.2 - j1.6 \mu A' passed for 1.2 - j1.6 mA): it is compared in its own unit now.
- *Units where students put them*: after a bracket of numbers ('12 / (2 + 2) k\Omega', '2 mA \cdot (1 + 2) k\Omega', '10 / (3 + j4) k\Omega' - it was letters: no verdict); a complex value with a prefixed unit last ('= 2 \angle -53.13^\circ mA'); units inside an exponent divide out ('i = 2(1 - e^{-2 ms / 1 ms}) = 1.73 mA': the 2 is a bare number - right, not sure, as bare numbers 1000 apart are; it was a sure wrong step); a division right after a unit ('e^{-1 ms/1 ms}' was unread).
- *Units as students spell them*: '4 K\Omega' (a capital K), '2 kohm', '0.02 sec', '50 hz', '40 db' after a number - letters of the problem before (no verdict); in the rubric's final answers too, where '3 ma' for 3 mA and '40 db' were 'unit_mismatch' - 0 points, automatically (pint reads milli-years and deci-barns) - and '50 hz', '4.7 K\Omega' unparsable.
- *An item number with a dot* ('1. I = \frac{10}{5} = 2 A': read as 1.I - a false sure wrong step) and an equation's label after a unit ('= 2 A (1)': the side unread).
- *Messages*: a root of a fraction reads '√(2/50)' (it was '\sqrt2/50').
- *In the circuit checks too*: 'I = 12 / 6 k\Omega = 2 mA' was (12/6) kΩ - a false sure wrong against the circuit's 2 mA; the quantity's name decides there as well ('I = 12 / 6 mA', a current in mA, stays the whole side's).
- *A range is no value*: 'f = 20 Hz - 20 kHz' (or '20 Hz \sim 20 kHz') made f = -19980 Hz (or '20 Hz = 20 kHz'), every formula after it a false wrong; a difference stays one ('12 V - 0.7 V = 11.3 V', '0.5 mA - 2 mA = -1.5 mA').
- *Electron-volts* ('1 eV = 1.6 \cdot 10^{-19} J', 'E = 2 eV = 3.2 \cdot 10^{-19} J', keV) are a unit - e V was Euler's number times volts: no verdict. The reader's '\left( 3 + j4 \right) k\Omega' and an equation's label '\left( 1 \right)' read as the plain brackets do.
- *A unit after a phasor* ('10 \angle 0^\circ / 5 \angle 53.13^\circ k\Omega = 2 \angle -53.13^\circ mA', '2 \angle 30^\circ mA \cdot 5 \angle 20^\circ k\Omega') is the phasor's - it was letters: no verdict.
- *A one-letter Hebrew 'and'* ('R_1 = 2k \text{ ו- } R_2 = 3k') was no words between statements, and its hyphen became a minus ('- R_2 = 3k'); 'ה-V_A = 5' likewise.
- *A resistor's code* ('R = 4k7': 4.7 kΩ) and *a value's kind after its unit* ('10 V DC', '10 V_{rms}', '10 V (rms)', '10 V_{pp}') - letters of the problem before: no verdict (a name stays: 'V_p = \sqrt{2} V_{rms}').
- *Radians are a unit* ('\tan^{-1}(1) = 0.785 rad'), and a result written in radians is not read in degrees ('= 45 rad' is wrong; '= 45' bare is the arctan in degrees).
- *A system's answers in other units*: mesh equations in SI ('12 = 2000 I_1 + ...') answered 'I_1 = 4.36 mA' were a false sure wrong (4.36 against 0.00436), and so were equations in kΩ ('12 = 2 I_1 + ...', so mA) answered in amperes - right (not sure for the second) now; a bare answer a power of 1000 off is wrong, not sure.
- *An equation with units in it* ('12 V = 2 k\Omega \cdot I_1 + 1 k\Omega (I_1 - I_2)', '\frac{V_A - 12 V}{2 k\Omega} + \frac{V_A}{4 k\Omega} = 0', 'V = I \cdot 5 k\Omega'): its numbers in SI units when one unit at least is no letter's spelling (kΩ, mA - '3 V I' stays 3 times V times I) - the units' letters were unknowns: 'cannot check'; the answers after it compared in SI.
- *A lone x between numbers* ('P = 2 x 5 = 10', '2 mA x 5 k\Omega') is a times sign; *a tolerance* ('R = 100 \Omega \pm 5\%', '10 V \pm 0.1 V') is the nominal value (it was two values named apart as roots).
- *A bare step against the circuit* ('P_{R1} = 2^2 \cdot 4 = 16 mW': mA and kΩ, so mW) was 16 against the circuit's 0.016 W - a false sure wrong.
- When neither reading of a quotient with a unit holds, the quantity's name chooses the message ('V = 10/2 mV = 6 mV': 0.005 is not 0.006 - not '5000 is not 0.006').
- *Found by two random generators* (EE computations across 12 templates - Ohm's law, powers, time constants, cut-off and resonance frequencies, resistances, reactances, a divider, energies, a charge, a capacitor's voltage at a time, a gain in dB - written many ways; and a formula, its data and the answer in units every way): a bare n or p (nano, pico) was read only where the step held, so a wrong step had no verdict ('\tau = 10k \cdot 100n = 2 ms'); a bare prefix beside written units ('(12 \mu A)(2.2M)'), an x after a bare prefix or a unit in brackets or \mathrm ('4.7M x 100n', '100 [\mu A] x 12 M\Omega'), a unit's '\mathrm{k\Omega}' taken for a letter k (the bare k's beside it unread), a charge or an energy as a result ('= 220.9 nC'), 'µsec', 'nsec' - no verdict before. Not one false verdict among 2000 generated cases now.
- *A complex value with a prefixed unit* ('I = \frac{(8 - j8) V}{(3 + j6) k\Omega} = (-0.74 - j2.21) mA', for -0.53 - j1.6 mA) was right - the tolerance of its two decimals was taken in amperes (0.005 A: every milliampere value matched); it scales with the unit's power of ten now.
- *A complex value's unit, unbracketed*: 'Z = 6 + j 8 k\Omega = 10 \angle 53.13^\circ k\Omega' was a false sure wrong - the kilo the j-part's alone (6 + j8000); 'Z = 3 - j4 \Omega', '\frac{12 + j5 V}{4 + j4 k\Omega}', '3 \Omega - j4 \Omega', '\frac{10 V}{-j4 \Omega}', 'S = 120 + j90 VA' had no verdict. The unit is the whole value's now - left unread where it is not clear whose ('2 \cdot 3 + j 8 k\Omega'). Found by a random generator of phasor computations with their units (300 of them in the fuzz now: the code before had 8 false sure wrongs and 114 without a verdict among them).
- *Semiconductor physics in centimetres*: '= 2.25 \cdot 10^4 \mathrm{cm^{-3}}' was a false sure wrong - a power inside \mathrm made the formula parser read the side as 10 (any '\mathrm{X^{n}}' lost what followed; fixed in the parser, for the rubric's formulas too). Densities, resistivities and conductivities as the course writes them ('cm^{-3}', '\frac{1}{cm^3}', '\Omega \cdot cm', a unit after '10^{15}') had no verdict. A length worked out in centimetres and written in micrometres ('L_n = \sqrt{34.97 \cdot 10^{-6}} = 59.1 \mu m', 'W = ... = 0.302 \mu m') was a false sure wrong (59.1 µm is 0.00591 cm) - right now, not sure; 'D_n = 1000 \cdot 26 mV = 26 cm^2/s' (the mV put the chain in SI, the mobility bare in cm^2/Vs) was a false sure wrong - right. Found by a random generator of semiconductor computations (200 in the fuzz now: the code before failed 104 of them - 34 false sure wrongs, 70 without a verdict).
- *Volt-amperes beside the numbers* ('\frac{800 W}{1 kVA} = 0.8', '1 kVA \cdot 0.6 = 600 VAR', '\frac{600 VAR}{2\pi \cdot 50 Hz \cdot (230 V)^2}') were letters of the problem - no verdict. Very large values in messages are written short ('2e+21 is not 2e+20', not 22 digits).
- *Reactive volt-amperes in a rubric's key*: 'Q = 600 var' was a bad reference - the units library knows VA only, so the question could not be graded at all; var, VAR, kvar are units now ('0.6 kVAR' for 600 var is right, '6 kvar' a prefix error), and an answer per cubic centimetre ('5 \frac{1}{cm^3}') is read (it was 'not a number').
- *An angle in degrees with prefixed data or a prefixed answer*: 'S = 1 kVA', '\phi = 36.87°', then 'Q = S \sin\phi', 'Q = 600 VAR' was a false sure wrong - with the angle in degrees the data were taken as written (1 kVA as 1 VA: Q = 0.6), in SI units the angle was taken in radians; so was the chain 'P = S \cos\phi = 2000 \cos(45) = 1.414 kW' (1414 against 1.414 as written, 1051 W in radians). Both readings together now. Found by a random generator of power-factor and phasor-part problems (200 in the fuzz now: the code before had 32 false sure wrongs among them, and missed 16 wrong answers).
- *A formula in sizes* ('X = |Z| \sin\phi', 'P = |S| \cos\phi' with '|Z| = 2200 \Omega', '|S| = 1 kVA' written) was 'no value for Z' - the size is the value such a formula needs (not when it has Z itself: 'I = V / Z'); '|I| = \frac{|V|}{|Z|}', then '|I| = 5 mA' 'could not be solved for |I|'.
- *A mobility in cm^2/Vs* (volt-seconds written together) and *a rate in s^{-1}* were letters - 'cannot check' / no verdict. *Thousands inside a computation* ('\frac{12}{10,000} = 1.2 mA') had no verdict - the digits are joined now. *A reactance after a factor with its unit* ('V = 2 mA \cdot j4 k\Omega = j8 V', '10 V / j5 \Omega') has its own unit (it was letters).
- *Values flowing on by name over several lines*: 'I = \frac{V}{R_{eq}} = 208 \mu A', then 'P = I^2 R_{eq} = (208 \mu A)^2 \cdot 24 k\Omega = 1.04 mW' was a false sure wrong - I was 208 (as written) beside R_eq = 24000 (in SI units, from its own line); a value worked out on a line is kept in SI units too now. And 'P = I^2 R_{eq} = 305 mW' under 'I = 15.7 mA', 'R_{eq} = 1.25 k\Omega' (worked out with I unrounded, 15.66 mA) was a false sure wrong, 1 % off: a step is held to what the rounding of the values written allows - decimals of three digits and more, written in it ('(13.2 mA)^2 \cdot 1.13 k\Omega = 199 mW') or above it and used by name (components' values like '3.3', '2.2' are exact: '3.3 \cdot 2.2 = 7.36' stays wrong). Found by a random generator of three-line work (150 pages in the fuzz now: the code before judged 4 to 12 correct pages in 150 wrong for sure, and on pages with one slip judged a second, right line wrong 8 to 14 times).
- *Names of several letters*: 'pf', 'BW', and a subscript written in line ('Req', 'Rth', 'Vth', 'Vrms', 'Vcc', 'fmax', 'gm', 'Av', 'hfe') were products of letters (p times f, R times e times q) - a formula with them 'cannot check' or no verdict; one name each now ('RC' stays a product, '100 pf' picofarads).
- *A value worked out on its own line under a formula*: 'f_c = \frac{1}{2\pi R C}', then 'R = R_1 \| R_2 = 2.2k \| 6.8k = 1.66 k\Omega' (or 'I = \frac{V}{R} = \frac{12 V}{4.7 k\Omega} = 2.55 mA' under 'P = I^2 R'): R_1, R_2, V are no letters of the formula, so the line was skipped - its steps unchecked, the formula 'no value for R'. Its steps are checked now and its value is the formula's (150 such problems in the fuzz: the code before gave no verdict on any of them).
- *A second formula defining a letter of the first* ('P = I^2 R', 'I = \frac{V}{R}', then 'V = 12 V', 'R = 4.7 k\Omega', 'P = 30.6 mW'; 'f_c = \frac{1}{2\pi R C}', 'R = \frac{R_1 R_2}{R_1 + R_2}'): the data and the answers went to the second formula only - 'cannot check'. The second gives the first its letter now, in SI units too (and is checked on its own when its letter is answered). 150 such problems in the fuzz (no verdict on any before).
- *An angle written with a degree sign as a value*: '\phi = 36.87°' (or '36.87 deg') was 36.87 radians - '36.87^\circ' was in radians already - so 'Q = 1000 \sin\phi = 600' under it was a false sure wrong (-737.3); such a value is in radians now too (not °C, °F). The lecturer's '36.87°' was no value at all.
- *A gain in dB beside its ratio*: 'A_v = 40 dB = 100' was a false sure wrong step (40 against 100 - a middle side's dB was dropped); right now (10 log10, a power's, not sure). 'A_v = 40 dB', then 'V_o = A_v V_i', 'V_o = 10 V' was a false sure wrong (40 times V_i): in a formula a gain in dB is read as its ratio too; 40 times V_i ('V_o = 4 V') is no longer sure - dB multiplied as a ratio (dB added, 'G = G_1 + G_2', stay right). So in a chain: 'A_v = 40 dB', 'V_i = 0.1 V', then 'V_o = A_v V_i = 100 \cdot 0.1 = 10 V' was a false sure wrong step (40 times 0.1); '= 40 \cdot 0.1 = 4 V' is not sure now. 200 random pieces of work with gains in dB in the fuzz (the code before judged 80 of them wrong for sure).
- *Typed as plain text* (as the course's model solutions are): 'BW = f0/Q = 10000/20 = 500 Hz' was a false sure wrong step - f0 was f times 0; 'f0', 't1', 'T2', 'N1', 'x1', 'w0' are names with their subscript now, as 'R2', 'V1' were ('j4' stays 4j). 'log10(100) = 2' was a false sure wrong (log of 10 times 100), 'log2(8)' too; '3**2 \cdot 4 = 36' (a program's power) was read as 4 - a false sure wrong; '12/4.7e3' is 12 over 4700 (it was (12/4.7) times 1000 in messages); '159 Ohm' with a capital O was letters (no verdict).
- Messages: an angle answered in degrees is shown in degrees ('\arctan(4/3) = 50°': 53.13° is not 50° - it showed radians, 0.9273 against 0.8727).
- *A page of several parts, unnumbered* (as students write sub-questions one after another, reusing R, I, V, P): a part's values were taken for the next one's - its 'R = 4.7 k' judged as the formula's answer, 'I = V/R' judged with the I worked out above (false sure wrongs on 21 of 100 random pages). A letter given again with another value, or a value for a letter foreign to a formula all of whose letters have their values, starts the next part now (values written again go with it); an equation's roots stay together ('x = 2', 'x = 3'), and so does an answer written again at once.
- *Done again below itself* (the first try crossed out - the reader does not see strokes): 'I = \frac{V}{R} = 6.5 mA', then '= 5 mA' - the second was judged against the first (a false wrong step, and every later line used 6.5 mA). Now each is judged on its own, the lines below use the second, and the first try's wrong step is not sure ('crossed out?'). An answer written twice ('P = 13 mW', 'P = 10 mW'): the last is the answer, the one before judged too, not sure.
- *Every root judged*: 'x^2 - 5x + 6 = 0', then 'x = 4, x = 2' - only the last was judged (x = 4 never).
- *Question and sub-question marks* - 'a)', '(b)', 'c.', 'א)', 'סעיף ב:', 'Part c:', '\text{d)}', 'Q1:', '1a)', '3.2)', 'שאלה 1:': the line was lost ('b) P' no side) or read as a formula ('(a) R = 1 k\Omega' a times R; 'Q1: R = 5 k\Omega' Q1 times R) - a page answered that way got no verdict at all. A question's number starts a new exercise; its later parts (b, c, ...) go on with the values above.
- *A computation in letters the formula does not have* ('f_c = \frac{1}{2\pi \tau} = 207 Hz' under '\tau = R C', 'I = \frac{V}{|Z|} = 1.59 A' after '|Z| = ... = 188 \Omega'): checked with no values, or never (when it started a problem of its own) - slips missed on 9 of 150 random exam answers. Now with the values known on the page; a step wrong only with an earlier problem's values is not sure.
- A value worked out keeps its digits: '= 10 mW' was the answer '0.01' - one digit, and 8.8 mW passed for it (a false right beside the step's wrong). '3.18e3^2' is 3180 squared (it was 3.18 times 10^6).
- *Electronics answers* (BJT bias, op-amps, diodes, Thevenin - found by a random generator of them): 'I_C = \beta I_B = 11 mA' (10.99 unrounded), then 'V_{CE} = V_{CC} - I_C R_C = 1.01 V' was a sure wrong - 12 - 11 is 1, and 1.01 was 1 % off by the float's last bit; an answer within what the rounding of a value worked out above moves it is no sure wrong now (right, or wrong not sure, for a human). 'V = V_s - I R', 'V = 1 V' was 'right value, wrong unit' - volts less volts made '1 (no unit)'; the unit is told term by term. 'R = 220 + 3180 = 3.4 k\Omega' then 'I = \frac{V}{R} = 2.94 mA' was right only 'in units not written, 1000 apart' (the 3400 taken for kΩ again); 'I_L = \frac{V_{th}}{R_{th} + R_L} = 1.9 mA' (a slip) passed the same way - that guess is for numbers written bare only. Messages show units and names as written ('3.4 kΩ', 'V_CC').
- *Boolean algebra* (a digital-systems exam): 'F = A\bar{B} + \bar{A}B = A \oplus B', '\overline{A + B} = \bar{A}\bar{B}', "A' + B' = (AB)'", 'F = AB + A\bar{B}' then 'F = A' on the next line - such lines were dropped (\bar no letter to the parser) or read as algebra: no verdict (the batteries had 14 of them). Each side is judged against the next by their truth tables now, and a wrong step shows a row where they differ ("'(AB)' = A'·B'': not the same for A=0, B=1"); 'F =', 'F(A,B,C) =' is the function's name, and a later 'F = ...' is judged against what F was. A plain line ('Y = (A + B)(A + C) = A + BC') is read so on a page with Boolean algebra on it only; a mean ('\bar{x} = 5.2'), two segments ('\overline{AB} = \overline{CD}'), a primed name ("V' = V + IR") are not.
- *Numbers in bases* written as students write them: '0x1F', '0b1011', '1F_{16}' and 'FF_{16}' unbracketed, '(1F)_H', '1011_B', '17_O' were letters or a product (no verdict); and a number in a base is exact now - '(E8D)_{16} = 3726' was right (within 1 % of 3725).
- *A frequency response at a frequency*: 'H(j\omega) = \frac{10}{10 + j\omega}', then '|H(j10)| = 0.707' was 'no value for omega' (the 10 in 'H(j10)' was dropped) and '\angle H(j10) = -45^\circ' no verdict; at several frequencies ('|H(j100)| = 0.707', '|H(j1000)| = 0.995') the second was judged against the first's value - a false wrong step. Each is judged at its own frequency now ('j10^3', 'j1k' too), the angle as the formula's angle there.
- *Lines that got no verdict* (the batteries had 18 of them, now judged - every one rightly): a bare m that is not milli where the step holds ('V = I R = 2m \cdot 4k = 9 V', '\frac{100}{80 m} = 2.5 k\Omega' - slips that passed): wrong, not sure, when wrong read either way, right not sure when right read as metres ('\rho \frac{100 m}{1 mm^2} = 1.7 \Omega'); '\sqrt{250n/100p}' (a bare n, p before a slash); 'r_{1,2} = -1, -2' (the second root was lost); 'a_n = 2^n' then 'a_3 = 9' (a sequence's term); '\angle(1 + j) = 45°'; '100 W \cdot 2 h = 200 Wh', '500 mA \cdot 2 h = 1 Ah' (h a letter).
- *A wrong expansion along a chain*: 'y = (x+1)^2 = x^2 + x + 1', '(a+b)^2 = a^2 + b^2', '2(x + 3) = 2x + 3' got no verdict (a step in letters that does not hold may be an equation to solve - 'V_A = 10 - 2I = 4I + 2'). One side a product or power of sums and the other a sum, with a leading term in common, is an expansion written: wrong, the expansion shown ('expanded, x²+2x+1'). Equations along a chain still get none; a line Boolean algebra may be ('(A + B)(A + C) = A + BC') is left to it.
- *More lines judged that got no verdict*: units spelled out ('20 Watt', '5 volts'), femto ('10 fA'), '\cos(60 rad)' (a sure wrong now - radians said, no degree guess) and '\sin(30 deg)', lengths before a length ('5 m - 2 m = 3 m', '2m \cdot 3m = 6 m^2'), '17 \bmod 5', '\gcd(12, 18)', '1 + 2 + \ldots + 10' (they had no value), the normal distribution ('\Phi(1) = 0.8413', 'Q(2) = 0.0228'), and a formula written after its values ('V_{rms} = 10 V', 'V_p = 14.1 V', then 'V_p = \sqrt{2} V_{rms}': right; wrong not sure - another part's value?).
- *Units of fields, speeds, data rates* in a substitution: '\frac{8.85 \cdot 10^{-12} F/m \cdot 1 cm^2}{1 mm} = 0.885 pF', '2 kV/m \cdot 3 mm', '\frac{100 m}{10 s} = 10 m/s', '\frac{36 km}{0.5 h} = 72 km/h', '8 kHz \cdot 8 bits = 64 kbps', a conductivity '(\Omega \cdot cm)^{-1}' (letters - no verdict); and a phasor's unit before its angle ('2 mA \angle 30^\circ \cdot 5 k\Omega': 2 times angle times 30).
- *A slash read both ways*: the parser puts what follows a slash under it ('1/2\pi f C' is 1 over 2πfC - as students mean it), so 'E = 1/2 C V^2' was 1/(2CV²), 'V_o = R_2/(R_1+R_2) V_{in}' had V_in under the bar, '20 \log(V_o)/(V_i)' was log(V_o) over V_i, and 's^2 + (R)/(L)s + (1)/(LC) = 0' a cubic - sure wrongs for right answers (19 of 150 random inline formulas; also '\int x dx = 1/2 x^2 + C', 'f(x) = 1/2 x^2' then 'f(2) = 2', 'H(s) = 1/(s+1)(s+2)' then 'H(0) = 0.5'). A denominator its bracket closes is closed now ('R_2/(R_1+R_2) V_{in}', '1/(2\pi) \sqrt{k/m}' - not before a bracket: '1/(s+1)(s+2)' is still 1 over the product); a slash before a product is read the other way too - a half times C V^2, the log of the ratio - in a step, a formula's answers or anywhere on the page: right that way, not sure ('\alpha = 1/2RC' worked out as (1/2)RC is a slip the arithmetic cannot tell), sure when only that way the units agree ('E = 125 µJ'); right only with brackets the student left out ('V/R_1+R_2 = 9/1+2 = 3', 'V_{CC} - V_{BE}/R_B'): wrong, not sure; wrong both ways, wrong - with the other reading's value ('0.002 (read as (1/2)CV²: 125) is not 250').
- *A sentence's end, words before a statement*: 'I = 3 mA.' and 'I = 3 mA :תשובה' were read as I = 10 (a false sure wrong for a right answer); 'נתון: V = 12 V' (a Hebrew word and a colon), '\text{given } V = 12 V', 'so I = 3 mA', 'Solve: x^2 = 4' were unread or letters of a formula. In 150 random pages with Hebrew words around the math ('לכן', 'נקבל', 'תשובה:', '\text{ו-}'), 65 had a false sure wrong; none now.
- *A number after a fraction* ('\frac{2}{1+2} 12 = 8', '\frac{1}{2} 10 = 5', no product sign between): the parser took a fraction and a number after it for their sum ('\frac{10}{2} 3' was 8) or split the number ('\frac{10}{1} 0.5': 10 times 0 times .5) - sure wrongs for right steps; also a fraction after a fraction or a bracket ('\frac{6}{3}\frac{1}{2}', '(3)\frac{1}{2}': read as mixed numbers, 2 1/2) and a decimal after a root after a fraction (split); products now ('2/3·12' in messages; '2 \frac{1}{2}' still a mixed number). 6 of 600 random products written side by side were misread; none now.
- *A result with its data listed below it* ('I = \frac{V}{R} = 3 mA', then 'V = 12 V', 'R = 4 k\Omega'): no verdict before; its steps with them now - a wrong one not sure (a value of the next part?).
- *Typed answers*: a micro 'u' glued to a number ('1/(2*pi*1k*1u)', 'XC = 1/(314*10u)'), 'pi' once the prefixes were put in SI units (read as p times i), '10m/10 = 1ms' (a bare m before a slash and a number: milli, as before a unit only metres - 'm/s'): no verdict before (61 of 150 random typed computations got none, right or wrong; none now).
- *Transconductances, a MOSFET's k* ('g_m = 40 mA/V', 'k_n = 2 mA/V^2'): A and V were letters - no value ('no value for k_n'); as '\frac{mA}{V}' now.
- *Degrees as a small o, radicals with brackets inside*: '53.13^o', '\cos 60^o' were unread; '√(30^2 + (80 - 40)^2) = 50' and '1/√(1 mH \cdot 1 µF)' (its units put in) were read as 40 and 10 - false sure wrongs.
- *Smaller gaps*: siemens ('0.833 S'), '\max(3, 5)', metres squared under a force ('\frac{k q_1 q_2}{(0.3 m)^2} = 0.2 N'), unicode subscripts ('V₁'); and a crash: '\arctan(40)/(30) = 53.13^\circ' (the degrees' message took a number's place).
- *Invisible direction marks* (the right-to-left and left-to-right marks Hebrew text carries when typed or pasted, zero-width spaces): 'I = 3 mA' with a mark after it was read as I = 10 - a false sure wrong; '12/4k = 3 mA' with one inside it a false wrong step. They are dropped now.
- *Braces no command takes*, as formula readers write them: 'I = {\frac{12}{4k}} = 3 mA', 'I = {12}', '{12 \over 4}' - the side was not read, the line lost; '4{\rm k}\Omega' lost its unit. Brackets, a fraction, upright now (a braced row, '{ 2 x + y = 5 }', and an array's cells stay as they were).
- *Spacing from the formula reader*: "f' \left( x \right) = 2x" (sizes around the bracket - the name before it was lost) and '1 m s' (a prefix and its unit spaced apart: m times s) got no verdict.
- '\tan^{-1}(X_L - X_C)/R' (an inverse function's power before its bracket): the arctangent of the ratio is the other reading too - right not sure, as '\log(V_o)/(V_i)'.
- *Bold*: '\textbf{I = 3 mA}', '\mathbf{I = 3 mA}', '\bf I = 3 mA' (read as text - letters of a formula - or not at all), '\textbf{Answer:} I = 3 mA', and 'V_{\textbf{out}}' (another letter than V_{out}).
- *A crash*: a slash before functions of functions without end ('1/\sin \sin \sin ... x', a reader's garbage two thousand deep) - the slash's other readings (last sync's) recursed past the stack; no factor there now.

**Earlier (27.10, night)** - more false verdicts, found by writing work the ways students write it; each fixed and kept in `tests/` and `tools/check_fuzz.py`:

- *A crash, and a net under it.* A matrix power ('\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^{-1} = ...') sent sympy into endless recursion and took the whole page's checks down; a reading the checks fail on now gets 'cannot check' and the page goes on.
- *Matrices written out* are checked: a chain of them in numbers entry by entry - a product, an inverse, a transpose ('^T', or '^t' as Israeli courses write it) - each entry to the digits written; 'A x = b' with the unknowns in letters is a system of equations, the values written after it checked against its solution. The parser read 'M + 2' as 2 and 'M^{\top}' as 4: no verdict now on what it cannot work out.
- *Bare prefixes.* 'C = 100n', 'C = 10\mu', 'R = 1k' were taken as 100, 10 and 1 - a formula over them was judged a false sure 'wrong' ('\tau = RC = 0.1 ms', 'X_C = \frac{1}{2\pi 50 C} = 318 \Omega'), and a result '= 100n' was 100. A bare m is metres or milli: milli where the formula or the step holds only so ('L = 10m', '\frac{100}{80 m} = 1.25 k\Omega') - never a 'wrong' from that reading. A single bare prefix in a product counts when the result has one ('2 \cdot 4k = 8 kHz'), but not before a name ('\sqrt{2 k I_D}': a MOSFET's k).
- *Tiny values.* Every value under 1e-9 was equal to every other (a floor of tolerance against rounding noise): '10 pF + 20 pF = 50 pF' and '1.6 \cdot 10^{-19} \cdot 5 = 8 \cdot 10^{-18}' were right. Decimals are worked out as exact fractions now, and the floor is kept only beside a zero.
- *Exponentials.* After e^{x} of a large x a step may be up to 5 % off - the exponent kept to 3 digits on the way ('I_D = 10^{-14} e^{0.7/0.0259} = 5.32 mA'); a wrong factor or sign is still caught.
- *Notation.* An '=' inside brackets ended a side ('P(X = 2) = \binom{5}{2} 0.3^2 0.7^3': a false wrong step); an inequality separates statements ('f_s \ge 2 f_{max} = 2 \cdot 4k = 8 kHz' got no verdict); '\sim' before a number is 'approximately'; a domain after a function ('i(t) = 2e^{-t} (t > 0)') no longer loses the function; typed superscripts and subscripts ('8×10⁻¹⁹ J', 'log₂(8)'), 'arctan' and 'sin(30°)' as typed; 'bit', 'bps' are units; '12 / 4.7k' is 12 over 4.7 kilo.
- *Checked now* (no verdict, or 'cannot check', before): a letter worked out in a chain is data for the formulas after it ('\omega_n = \sqrt{10} = 3.16', then '2\zeta\omega_n = 2', '\zeta = 0.316'); a value worked out a line at a time without a formula above ('V = 5 \cdot 2 \Rightarrow V = 10'); the numbers after a side in letters ('E[X] = np = 10 \cdot 0.3 = 3'); sequences and steps ('h[n] = 0.5^n u[n]', then 'h[2] = 0.25'; 'x(t) = u(t) - u(t-2)', then 'x(3) = 0' - the parser reads 'u(t-2)' as u times (t-2)).
- *A chain going on at the next line.* 'I = \frac{V}{R}', then '= \frac{10}{5}' and '= 2 A' on lines of their own - as a chain is often written - were dropped from the second line on; they are one chain now (not after a datum: 'R = 5', then '= 2 A'). A unit the reader spaced on the last side ('= 2 m A', '= 1 k H z') is a unit, not letters of the problem.
- *A power of 1000 apart with bare numbers* ('P = I^2 R = 3^2 \cdot 4k = 36 mW': 3 for 3 mA, as EE students keep mA and kΩ) was a sure wrong step once its sides were put in SI units: right, not sure; with no prefix in the result ('\frac{1}{2\pi\sqrt{10 \cdot 1}} = 50.3 Hz', C in µF) wrong, not sure - a slip of 1000 looks the same, a human decides; with every unit written it stays a sure wrong step. A parallel of a bracketed sum with prefixes ('(2k + 3k) \| 5k = 2.5k') was read as a norm: a false sure wrong step.
- *Responses.* '\angle H(j1000) = -45^\circ' after 'H(s) = ...' is checked (degrees with or without the mark, or radians); 'dBm', 'dBW' are units; the final and initial value theorems ('f(\infty) = \lim_{s \to 0} s F(s) = 1' under 'F(s) = ...' - the limit was left unworked); a delayed step in a transform ('\mathcal{L}\{u(t-2)\} = \frac{e^{-2s}}{s}').
- *More false sure 'wrong's.* A system written in a plain matrix environment ('\begin{matrix} 2x - y = 1 \\ -x + 3y = 2 \end{matrix}') was cut at its '=' signs into garbage sides; two's complement with its base written ('-5 = (11111011)_2') was read unsigned. '(17)_8 = 15' lost its number (taken for an exercise's '(17)'); tesla and weber are units ('= 40 \mu T' after a B), and milli comes before a bracket of numbers ('-4m (1k \| 25k)').
- *False sure 'right's* (found by making right work wrong in many ways - x2, x0.5, x10, a sign - and looking for what passed): an angle written with its degree mark was read in degrees again ('\theta_v = 30^\circ', '\cos(\theta_v - \theta_i)': cos of 1.05 degrees) - a wrong 'P = 10 W' for 5 W was right; a false step beside a right answer ('I = 2' under 'I = \frac{V}{R}', then 'P = 2 \cdot 10 = 30') was dropped, the problem just 'right' - it is a wrong step of its own now; a value against the chain just above it ('\frac{1}{R} = ... = \frac{1}{2}', then 'R = 3') is checked; the negative root of a size ('c = -5' for 'a^2 + b^2 = c^2') is right, not sure.
- *The parts of a complex letter* ('\angle Z = 53.13^\circ' - or without the mark -, '\Re\{Z\} = 3', 'Im(I) = -1' after Z's or I's value) are checked (the parser read R times e times Z).
- *A slash before a product* ('X_C = 1/2\pi \cdot 50 \cdot 10^{-4} = 31.8': 1 over the whole product, as students mean it - the parser takes (1/2) \pi \cdot 50 \cdot 10^{-4}) was a false sure wrong step: either reading holds now. A bare answer a power of 1000 off ('I = 5' for 5 mA under 'I = \frac{V}{R}') is wrong, not sure. A Greek μ before a unit ('10 μF') is micro, and a typed ∞ ('v(∞) = 10', '\lim_{t → ∞}') is read.
- *Messages* name the letters as written ('R=5000', not 'r=5000'; 'ω_n', 'ζ'), show '3e-11' (not '0'), '[1, 2; 3, 4]', 'det[...]', 'C(5,2)', '∫', '∞', '√(316²-50²)', 'e^(-2s)', an angle in degrees ('53.13° is not 60°'), and say when a value was taken in milli; a side in dB is no ratio ('20\log(100) = 60 dB': 40 is not 60).

`tools/check_fuzz.py`: 344 + 117 derivations (clean and as the formula reader spaces them), the random garbage pages now with matrices too; 264 tests.

**Earlier (27.10, evening)** - false 'wrong' verdicts found by writing correct work many ways, each fixed and kept in `tests/` and `tools/check_fuzz.py`:

- *The LaTeX parser's own habits.* **It lowercased every letter** (latex2sympy under Math-Verify): 'A_v = -\frac{R_C}{r_e + R_E}' merged r_e and R_E, so a right gain was judged wrong for sure; 'T = 2t', 'x = \frac{t}{T}' merged T and t. Letters now keep their case, and the checks fold them back where a letter is written in one case only (a v read for V still matches V); a name written in both cases in one statement - or among the lecturer's values - names two quantities. A number that is a power itself was taken for scientific notation ('\frac{1}{2\pi \cdot 10^3 \cdot 10^{-6}}' read with 10^(3 \cdot 10^{-6}) = 1 - on the commonest way EE students write kΩ and µF). A signed power without braces ('10^-3', 'e^-1000t') took the sign alone as the power ('10^-3' came out 10). '\sqrt 20' was √2 · 0. '2 \frac{1}{2}' is a mixed number to the parser (2.5) and a product to an engineer (1): a step that holds either way is right. An integral's limit was joined to the number after it ('\int_0^1 4t^2 dt' had an upper limit of 14), '\int_0^1 (2t)^2 dt' read as 1 times (2t)^2. Probability notation ('P(X \le 2)', 'E[X^2]', 'Var(X)') was read as numbers - it is left unread now, and a chain with an unreadable side keeps the steps of numbers around it.
- *Rounding.* A value rounded to the digits written is right ('\sqrt{2} = 1.4', '\frac{10}{3} = 3.3 A', '\frac{100}{7} = 14', '2.6 \cdot 10^{-2}' for 0.02553, '= 26 mA' for 25.53 mA) - the chain checks held every step to 1 %; up to 15 % ('0.0001' for 0.00005 is still wrong), an integer only from two digits on ('5' for 4.6 is still wrong).
- *Notation as students write it.* Degrees without the mark ('\cos(60) = 0.5', '\arctan(\frac{4}{3}) = 53.13', 'e^{j45}'; right but not sure when the angle may be radians from ωt with a calculator left in degrees - '5\cos(10) = 4.92'); a phasor's angle in radians ('5\angle 0.927', '\sqrt{2}\angle\frac{\pi}{4}'); decibels ('A_v = 100 = 40 dB', '|H| = 0.707 = -3 dB'; 10 log10 right but not sure); binary without the base ('1 + 1 = 10', '1111 = 15', '-5 = 1011' in two's complement; '1 + 1 = 1' Boolean, not sure); percent after a factor of 100 ('\frac{5}{10} \cdot 100 = 50\%', 'PO = 100 e^{...} = 35\%', '\eta = \frac{P_o}{P_i} \cdot 100' then '\eta = 50\%'); typed arrows ('⇒', '→', '⟹', '∴', '⇔') and a typed 'ΔV'; a box, an underline or a colour around an answer ('\boxed{V = 10 V}' - also in the rubric); a label over an equals sign ('\overset{KVL}{=}', ':=', '\triangleq'); an ellipsis or a modulo (no verdict now); '\sqrt{16} = \pm 4'; 'x' as a times sign before a power of ten; a power factor's 'lagging'.
- *Units.* A unit after a number in scientific notation ('2.553 \cdot 10^{-2} A', '2.553E-2 A'), after a prefix or a power of ten ('1 \mu F = 10^{-6} F'); a name with a subscript after a number ('\frac{2 V_m}{\pi}' was 2 volts and a stray '_m'); a bare micro in a substitution ('\frac{1}{2\pi \cdot 1k \cdot 1 \mu}'); parallel resistances with prefixes ('30k \| 10k = 7.5k'); bare numbers worked out in kΩ, so mA ('\frac{3 - 0.7}{7.5 + 101} = 21.2 \mu A': right, not sure - with the units written a slip of 1000 stays wrong).
- *Circuits.* A value rounded to its digits is the circuit's ('V_A = 3.3 V' for 3.333 V - it was 'rounded?', not sure); a small-signal r_e, r_\pi, r_o is its own quantity, not the resistor R_E drawn ('r_e = 25 Ω' was wrong against R_E = 1 kΩ); '3.333... V' is read.
- *Checked now* (no verdict before): steps in letters along a chain ('y = (x+1)^2 = x^2 + 2x + 1'; a wrong partial-fraction expansion is a wrong step), a function with letters evaluated with their values ('v_C(t) = V_0 e^{-t/\tau}', then 'v_C(1) = 3.68'), roots written 's = -1 \pm j3', an integral with a convergence condition ('\int_0^\infty e^{-at} dt = \frac{1}{a}', the letters taken positive), the lecturer's values (`--values`) in the derivations.
- *Messages* read plainly: '1/(2π)' (it was '\frac102π'), '2∠-53.13°', '2¹⁰', '10⁻³', and '%' only where the student wrote one.

`tools/check_fuzz.py`: 299 + 91 derivations (clean and as the reader spaces them) and 80 random angles / decibels / phasor angles with a wrong version of each; 237 tests.

Latest (27.10, afternoon): the formula parser read a subscript without braces together with the next letter when it was d or e ('\omega_d t' as one name omega_{dt}; 'r_e', '\omega_e t' not at all) - a subscript is now its one character, as in LaTeX (a correct derivative of a damped cosine had been judged a wrong step). Derivatives of expressions with other letters ('\frac{d}{dt}(A\cos(\omega t))', 'L \frac{d}{dt}(I_m \sin(\omega t))') were never worked out - a letter may be a function of t (v, i, \Phi); now they are, with those letters taken for constants: a side that agrees is right (sure for the usual names of constants - A, k, n, \omega, \tau, ...), a sign slip is a wrong step for those names when the expression is in t (not in '\frac{d}{dt}(J\omega^2/2)', where \omega may be what changes), and a side with the function's own derivative (C dv/dt) says nothing. A Fourier series' index (n in (-1)^n, \sin(n\pi), e^{-jk\pi}) is taken for a whole number: '\cos(n\pi) = (-1)^n', a_n and b_n worked out with their integrals get 'right' (no verdict before). Phasors in derivations: a complex value ('Z = 3 + j4', 'I = 2\angle -53.13^\circ') was dropped - now it is data, so 'I = \frac{10}{Z} = 2\angle -53.13^\circ' is checked (the conjugate angle a wrong step), phasor mesh / node equations are solved together, a characteristic equation's complex roots ('s_{1,2} = -1 \pm j2', 'x^2 + 4 = 0') are checked, and '|Z| = 5' against the size of what the formula gives; with complex data a real answer may use their sizes, as power formulas are written ('I = 2\angle 30^\circ', then 'P = I^2 R = 20'); a complex step in SI units ('L = 10 mH', 'j\omega L = j10' - a false wrong step before). A capital J is a letter (a current density, a moment of inertia - 'J = 6' was judged against sqrt(-1)), 'j = 3' a counter; Re{Z}, \angle Z, arg Z, I^*, \overline{I} are left alone (the parser read them as products of letters). 'H(j\omega) = ...', 'H(j2) = ...', '|H(j\omega)| = ...' or '20\log|H(j1)| = -3' after 'H(s) = ...' is checked with s = j\omega put in (H(j\omega) was read as H times j times omega). The lecturer's data for a question (`read_page.py --check --values data.json`, {"V": "12 V", "R": "470 \Omega"}) now serve the derivation checks too, not only the circuits drawn: 'I = \frac{V}{R}', 'I = 25.53 mA' with the values printed on the exam was 'cannot check' - now right (a value the student writes overrides the lecturer's); a chain's last value in its unit ('= 25.53 mA') is compared in SI as well. E with a subscript - an EMF, as Israeli courses name sources (E_1, E_2, E_{th}), or a node's e_1 - was dropped by the formula parser: 'E_2 + 3' read as 3 (a step 'E_2 + 3 = 9' with E_2 = 6 could be judged false), 'E_1 - I R_1' not read at all; fixed, in the rubric's formula comparison too. Math typed as plain text, as the course's model solutions are, is read: '10⁻⁸', 'V²', '√2', '½', 'R2' for R_2, '50 / 1.5·10⁻⁶' as 50 over 1.5·10⁻⁶, 'KCL:' before an equation, '20 ms' after '1 / 50', '12 / 2k', kWh - on the 26 model solutions of the course's five exams: 4 false wrong steps before, none now. Values put into an expression in letters ('v_C(t) = 10 + (0 - 10)e^{-t/\tau} = 10(1 - e^{-1000t})' under '\tau = 1 ms') are compared on the scales the expression changes over (around t = 1 ms - at t = 1 s every decay is 0): right when they agree, no verdict otherwise. An identity whose sides differ in letters that have values written above ('I(s) = \frac{V(s)}{R + sL} = \frac{10}{s(s+2)}' under R = 2, L = 1) is compared with the values put in - it was a false wrong step; with no values written, the values may be the problem's own (printed on the exam): no verdict rather than a wrong step. A step whose two sides are in units of different kinds ('P = V I = 10 V \cdot 2 mA = 20 mA': W against A) gets a 'wrong unit' note for a human (the number is right - it passed as right). A unit inside a fraction ('I = \frac{10}{2 k\Omega} = 5 mA') belongs to its denominator - the closing brace was taken for a \mathrm's and the whole fraction scaled: a false wrong step ('10 is not 0.005'); millisiemens ('g_m = \frac{1 mA}{25 mV} = 40 mS') are a unit. A time function at a time with its unit ('v_C(1 ms) = 6.32' under 'v_C(t) = 10(1 - e^{-t/\tau})', '\tau = 1 ms') is worked out in seconds (it was v_C times 1 m s; and the value was taken for an equation to solve for tau). Numbers in a base ('(1011)_2 = 11', '1011_{(2)}', '(FF)_{16}', '(101.1)_2') are read in it - they were read as 1011 or a falling factorial: false wrong steps in digital-systems work. A time function written after its phasor by the same letter ('I = 2\angle -53.13^\circ', then 'i(t) = 2\cos(1000t - 53.13^\circ)', a sine or an RMS phasor too) is checked against it - a mismatch is not a sure 'wrong' (the course's convention may differ). A phasor or an impedance may carry its unit ('3 + j4 \Omega', '2\angle -53.13^\circ A' - the Ω was read as omega), VA / var are units, and a complex number as written is held to half a unit in the last decimal of each of its numbers ('0.20\angle 41.82^\circ' for 0.196∠41.82°). The rubric takes phasors and impedances as keys and answers ('2\angle -53.13^\circ A' against '1.2 - j1.6 A', '30 + j40 VA' against '0.05\angle 53.13^\circ kVA' - a phasor key was refused, a phasor answer a parse error); the size alone for a phasor key, its conjugate, or the right size at another angle go to a human. Circuits: a right value with a unit of another kind ('V_A = 7.5 A', 'R_eq = 1 kV') is noted for a human (it passed as right); with nodes labelled A and B but no terminals drawn, R_eq / V_th taken between them are a guess - a mismatch is no longer a sure 'wrong' (the student's R_eq may be a part of the circuit's). `tools/check_fuzz.py` now: 250 + 64 derivations, 60 random computations with units in every position, 50 random phasor computations, 400 pages of random garbage (no crash, no hang), and the 140 real-formula grades replayed from their saved readings (`tools/replay_formula_eval.py`); 213 tests.

Earlier (27.10, midday), derivations: correct work written in 99 more ways (calculus, functions, roots, units, transforms, words between the math) gets 'right' and none is judged wrong; 26 more wrong ones are all caught (`tools/check_fuzz.py`, a new regression check: 170 + 40 derivations and some 860 circuit answers, eight minutes on a CPU). New: a derivative or an integral worked out in letters is checked as an expression (an indefinite integral up to its constant, '-cos²x/2' as right as 'sin²x/2'), an integral's evaluation bracket [F]_a^b, functions the student defines (f(2), f'(x), y' and dy/dx under y = x³), '±' (x_{1,2}, roots named either way round), units as fractions (m/s, kg/m³, km/h), a computation written a line at a time, Laplace transforms and their inverses (and a time function written after its transform, v(t) after V(s)), partial fractions (and their coefficients), identities, second and partial derivatives, systems of equations (mesh / node equations) solved together. A right number with the wrong unit ('R = 5 kV' for 10 V / 2 mA, the unit worked out from the data's) is noted for a human - it passed as right; on the page a sure false step is now a red cross. The rubric's numeric answers take units as fractions and powers ('10 \frac{m}{s}', '9.8 m/s^2' - a reference written so was refused before). In the circuit checks, a node without a letter whose own name read like another node's label ('a' beside 'A') or a part's ('c' beside C) hid that node's voltage - a voltage marked across a part came out 0 and its right value was judged wrong; fixed. Five false 'wrong's fixed in the derivations: T = 2π/ω with ω = 1000 against 'T = 6.28 ms' (bare data are in SI units or the student's own), '3x = 12 → x = 4', a Hebrew 'ולכן' between two statements, '2 · 3 = 6' (read 2.3), dy/dx under a value of x (worked out as 0); and primes were dropped by the parser (V_A = V_A' + V_A'' read as V_A = 2 V_A).

Earlier (27.10, morning), circuits: voltages marked on the drawing with '+' / '-' and a name ('V_x' across R2, 'v_o' at the output terminals) are read and their sign checked; a '+' / '-' beside a source gives its polarity, by a port's terminals which one is '+' (without it V_th is right in either sign - no more false 'wrong's there); an AC source read off a drawing had no polarity, so no AC circuit drawn could be solved at all - now its answers are checked both ways round it (right but not sure) unless its '+' is drawn; the dots of coupled coils are read when drawn. The checks read units in square brackets as Israeli exams write them ('3.33 [V]', '2 [mA]' - before, no verdict), chains 'V_A = \frac{E R_2}{R_1 + R_2} = 3.33' one equality at a time (an arithmetic slip at the end is wrong; before only the last two sides were looked at), magnitudes |V_A|, |Z|, |I|, an AC-fed filter's H(jw) / H(s), angles as fractions of pi, '\simeq', '3.\overline{3}', NOT written \neg / \lnot / ! in Boolean answers. Also checked now: P_max at a port, a gain in dB, an op-amp's V_+ / V_-, a transistor's Q-point ('Q = (4.98 V, 1.23 mA)', I_CQ, V_CEQ), dependent sources following a marked voltage; `read_page.py --check` marks each verdict beside its line on the page. The derivation checks (no key, any subject) use the data written above a computation or a formula, and work a formula out in SI units when every value has its unit - '5000 Ω' for 10 V / 2 mA was judged wrong there, and '5 Ω' right. The formula parser under all of it lost powers of I ('P = I^2 R' read as I R) and a trailing c d g h l m o s t ('E = m g h' as E = m) - in the rubric's formula comparison too, where 'P = I R' passed for 'P = I^2 R': both fixed. More of the same kind, found by writing correct derivations many ways (55 kinds, and each as the formula reader spaces it: none judged wrong now; a false 'wrong step' before for sin(30°), '2 \| 3 = 1.2', a substitution in SI units under a value in kΩ): degrees, parallel resistances, '2(3)', '1.5e-3', '\Delta V', 80 %, units written inline ('= 2 mA · 5 kΩ = 10 V'), a bare prefix ('R_1 = 1k'), sums / limits / integrals; the rubric's numeric answers take '3.33 [V]', '80\%', '45^\circ', '\frac{10}{3} V', '5k'. On the 70 real handwritten formulas: 134 of 140 grades automatic, none wrong (3 wrong in this environment before: an equation with an infinite side, compared with itself). Correct answers written many ways (274) are still never judged wrong.

Earlier (27.10), circuits: the wiring stage reads all 48 netlists of the unseen CGHD drafter exactly (26 before) - notebook rules photographed at a slant, the rules' colour taken out of the photo where the pen's differs, squared paper, pale and lightly pressed pens, a wire drawn right on a rule; still right with every part's box moved by up to 10 % (144 / 144). Circuit labels read with a beam search over a character model of labels (62 % against 58 % on unseen drafters, no retraining), and knowing the kind of part beside each label (+0.9 points more). New answers checked against the drawn circuit: values just before / after / long after a switching and dx/dt at 0^+, a second-order circuit's alpha / w_0 / zeta / w_d, AC totals (P, Q, S, |S|, power factor, angle, RMS), H(jw) and its magnitude / dB / angle, and the problem's data given once by the lecturer (`--values`). A part the detector misses no longer passes unnoticed: its symbol left in the wires is noted (the reading then gives suggestions, not verdicts) - 189 of 192 left-out parts on drafter 1, none of its 48 photos with every part boxed, nothing on Tal's drawing - and the reader then tries what the detector saw there below its threshold. The formula reader's own habits in the newer answer types ('B W', a prime as a superscript, '\mid' bars) are evened out (85 -> 100 of 100 readings judged as the clean answer). Fewer false 'wrong's: a decimal comma ('V_A = 7,5', in every derivation check too), 'Q' of an RLC may be its quality factor, a bare 'P' may be one part's power, V_A^{(1)} is superposition's share, not a power; and newly checked: a capacitor's charge, 'pf = 0.8 lagging', X_L / X_C / Z_L, the R a capacitor sees, superposition shares (V_A', I_2''), a diode circuit's part powers. Current arrows drawn on the wire itself are read (a chevron or a filled head: the current of the part the wire leads to), and a photo whose strokes took in a table or a shadow is read again with darker strokes. Details under "Circuits drawn by hand".

Earlier (26.10): answer-cell readers v2 trained on many more writers, incl. European-style digits (CVL:
answers read right 80 -> 89.5 %, graded automatically 43 -> 58 %); student IDs matched against the class
roster (accepted 90 -> 98 %, CVL 55 -> 94 %, none wrong); each student's misread digits learnt from his
ID; handwriting copy sheets to collect your students' own digits; reading with a grammar on 50 real
scanned exams; heb-ocr compared (ours stays); a bigger optional English reader; one command from scanned
copy sheets to a retrained cell reader (`retrain_from_sheets.py`, installed only if better on held-out people);
an optional "how to write the digits" line on the exam form; 1,775 technical Hebrew words; Hebrew text read
at five widths (CER 9.1 -> 8.6 %) by a Hebrew reader trained 3 more epochs (-> 8.2 %); a short line squeezed
under a long one found as a line of its own; derivations checked against themselves, without a key
(`read_page.py --check`) - since grown into a math validation: every '=' of a computation, the result from the
student's formula and data, units, equations, weighed over the readings; Hebrew read in context with a Hebrew
LLM (Dicta-LM 3.0: CER -14 % on held-out writers, -17 % on test lines, opt-in); a second formula reader
(Uni-MuMER, opt-in); what else exists, researched and partly measured ("Other handwriting models"). Details in
"Answer-box exams", "Reading a whole scanned page" and under the Hebrew / English results below.

**Formulas** (70 real handwritten formulas, CPU): 65/70 read exactly; of 140 grading decisions
(right and wrong answers), 134 auto-accepted with **0 wrong**, 6 sent to review.

**Hebrew handwriting** — off-the-shelf local models tested on writer-independent lines:

| model | result |
|---|---|
| cyttic/trocr-hebrew-matan-exp7 | hallucinates unrelated text |
| oln-1/hebrew-htr-trocr | outputs the same memorised page header for every image |
| YS17121712/hebrew-trocr | outputs invalid bytes for every image |
| Qwen3-VL-2B (zero-shot VLM) | CER 0.93 — repeats words / reads Hebrew cursive as Latin |

Hence our own recognizer (`train_hebrew.py`, ~2.5M parameters, 10 MB ONNX, runs on CPU):
its CTC beam search gives the probability of each candidate text, which feeds the same
confidence mechanism.

Shipped model (`models/hebrew-crnn-cpu`, 10 epochs on 2 CPU cores), tested on 300 lines
from 86 writers it never saw — runs on this machine, offline, ~50 ms per line:

| | 5 epochs (`hebrew-crnn-cpu5ep`) | 10 epochs (`hebrew-crnn-cpu`) |
|---|---|---|
| CER, all lines | 14.8 % | **10.8 %** (median 9.1 %) |
| CER, the quarter it is most confident about | 5.4 % | **2.9 %** |
| CER, the quarter it is least confident about | 27.8 % | 21.7 % |
| lines read exactly | 6 % | 11.7 % |

So its confidence is informative — the basis for routing hard answers to a human.
Examples: `results/hebrew_examples.png`. `setup.cmd` trains a longer run into `models/hebrew-crnn`,
which is then used automatically; fine-tune on your own scans with `train_hebrew.py --init`.

**Hebrew word model** (`models/hebrew-lm`). The CRNN alone mixes up look-alike letters (כ/ב, ן/ו) - it
read "ולכן" as "ולכו" although "ולכן" was in its n-best list. A unigram word model is fused into the
CTC beam search: ~540k Hebrew word forms with their frequencies in written Hebrew (`wordfreq`:
Wikipedia, news, web, subtitles; it also knows rare technical terms - a subtitles-only list did worse
on held-out writers, 9.9 % vs 9.3 % CER), plus a **course vocabulary** (`course_words.txt`, ~1050
terms: circuits, electronics, digital systems, signals and communication, control, electromagnetics and
power, math, programming, machine learning, units, and the words solutions are written with -
טרנזיסטור, עכבה, קיבול, דלגלג, פעפוע לאחור, ולכן ...; since 26.10 also 722 words that are rare in general
Hebrew but recur in 13 Israeli electricity / electronics curricula, bagrut question-and-solution collections
and lab manuals (Ministry of Education, Technion Moretech) - 1,775 words in all, CER on the general test lines
unchanged at 9.1 %; paste your own lecture notes or old exams there),
also found behind Hebrew prefixes (בנגד = ב + נגד). Unknown words are penalised, never
forbidden, so it only breaks near-ties: when the image clearly favours another real word it does not
override it (writer w4's "כאשר" in the demo is still read "באשר" - a real word too). The handwriting
datasets' own transcriptions are deliberately left out: many writers copied the same texts, so they
would make almost every test word "known" and inflate the result. Weights tuned on held-out training
writers (`tune_hebrew_lm.py`), then measured once on the test lines:

| 300 test lines, 86 unseen writers | CRNN only | + word model |
|---|---|---|
| CER | 10.8 % | **9.1 %** |
| CER, the 205 lines whose text never occurs in training | 11.0 % | **9.2 %** |
| WER (words) | 42.7 % | **32.2 %** |
| lines read exactly | 11.7 % | **21.7 %** |
| CER, most / least confident quarter | 2.9 % / 21.7 % | 2.3 % / 20.2 % |

**Compared with the best open Hebrew handwriting model** - heb-ocr's Mishkefet-v1 (github.com/itayinbarr/heb-ocr,
MIT code, CC-BY-NC-SA weights, 30M parameters; by its authors' measurement 2nd of 9 on the ivrit.ai
benchmark, behind Gemini Flash and ahead of the GPT and Claude models listed there) on the same 300 test lines (`tools_eval_hebocr.py`, its checkpoint loaded with
`weights_only=True` and its language-model pickles with a containers-only unpickler):

| 300 test lines, 86 unseen writers | heb-ocr | ours |
|---|---|---|
| CER, no language model | 13.4 % | **10.8 %** |
| CER, with its / our language model | 10.3 % (median 8.5 %) | **9.1 %** (median 6.7 %) |
| WER / lines exact | 39.9 % / 6 % | **32.3 % / 21.7 %** |
| time per line (CPU) | 2.5 s (3 views) | **0.075 s** |

Caveat: these lines come from the collection our model was trained on (other writers), which heb-ocr
has never seen; heb-ocr also drops final punctuation. A neutral comparison is the ivrit.ai benchmark
(`ivrit-ai/hebrew-handwriting-ocr-benchmark`, 225 lines, 10 writers) - it needs a Hugging Face login
and accepting its terms. No reason to switch so far.

**Reading every line five times** (`HebrewRecognizer.recognize(..., tta=True)`, `eval_hebrew.py --tta`): at its own
width and squeezed / stretched to 80, 90, 110 and 120 %, and the n-best lists pooled - a reading found in several
views gets their evidence summed. The widths were chosen on held-out TRAINING writers (one look 9.3 %, three widths
8.3 %, five 8.0 % CER); measured once on the 300 test lines: CER 9.1 -> **8.6 %** (median 6.7 -> 5.6 %), WER 32 ->
30 %; the most / least confident quarter 2.7 % / 18 %; ~5x the time (0.25 s per line on the development PC).
The page / mixed reader now uses it for the TEXT of every Hebrew piece, after the layout is decided with one look
(`EXAM_GRADER_HEBREW_TTA=0` turns it off): mixed_eval3 Hebrew CER 30.0 -> 29.8 % with the same layout (82 %) and
the same answers flagged. Used inside the layout decisions it changed the confidences they rely on and made
them worse (82 -> 80 %), so it stays out of them (`tools/hebrew_tta.py`).

**Trained further** (`models/hebrew-crnn-cpu2`, 26.10: hebrew-crnn-cpu + 3 epochs at a low learning rate,
`train_hebrew.py --init models/hebrew-crnn-cpu/model.pt --epochs 3 --lr 4e-4 --pct-start 0.05`, 95 minutes on 2 CPU
cores). Chosen on held-out TRAINING writers (validation CER 11.7 -> 11.2 %; with the word model 9.3 -> 9.0 %, at
five widths 8.0 -> 7.6 %), then measured once on the 300 test lines: one look 9.05 -> 8.98 %, five widths 8.55 ->
**8.17 %** (WER 30.2 -> 29.4 %, lines exact 22.3 -> 22.7 %). It is the default now;
`EXAM_GRADER_HEBREW_MODEL=models/hebrew-crnn-cpu` brings back the previous one. In the mixed reader the layout
stays about the same (dev mixed_eval 76 -> 75 %; tests 82.1 -> 83.3 % and 47.5 -> 48.3 %), never more lines wrong
and unflagged (4 -> 4 %; 0 -> 0, 5.0 -> 4.2 %), and the short Hebrew pieces both versions cut the same way are read
the same (dev 129 vs 129, test 161 vs 163 character errors) - the gain is in whole lines of text. On Tal's notebook page (7 lines, one writer) it is better with one look (23.8 -> 21.3 %)
and worse at five widths (16.4 -> 19.7 %) - too few lines to tell.

**Read in context** (26.10, `recog/hebrew_context.py`, opt-in: `EXAM_GRADER_HEBREW_CONTEXT=1`; the language
model: `python download_models.py dictalm-3.0-1.7b-base`). A person reading a smudged word uses the sentence
around it; so does this: P(text | image) ~ P(image | text) x P(text)^0.5. P(image | text) is the CRNN's CTC
probability of a whole candidate text - exact for any text, not only for what its beam found; P(text) comes
from Dicta-LM 3.0 1.7B (Dicta, Feb. 2026, Apache-2.0, a Hebrew-first LLM). Candidates: the beam's n-best, and
for every word of the best reading the vocabulary words one or two letters away from it, also behind its
prefix letters ('אור' -> 'אותך', 'מהממהרם' -> 'מה' + 'ממהרים'); a word costing the handwriting model more than
12 nats is not tried, of the rest its 8 likeliest; the line is improved word by word while the total rises.
The weight 0.5 was chosen on held-out TRAINING writers (80 lines, the run hit its time limit; 0.5 was best
from the 30th line on): CER 6.38 % (the reader in use) -> 0.25: 6.16, **0.5: 5.50**, 0.75: 5.86, 1.0: 5.86 %
(`tools/hebrew_context_eval.py --dev`, `results/hebrew_context_dev_progress.txt`). Test lines, measured once with
0.5: **9.15 -> 7.58 %** on the first 150 of the 300 (-17 %; the run was stopped there for time - the gain was
steady from the 20th line on; `results/hebrew_context_test_progress.txt`).
Caveat: the collection's sentences were copied by many writers and may be ones a big language model has
read; a student's own answer is less predictable, so expect less than on the test lines - on Tal's own
lines (7, original sentences) 19.7 -> 17.2 % ('אני בודק אור' -> 'אני בודק אותך', exact; a word that is
not a word - 'מהמהרות' - is pushed towards a real one, 'מהמקרים'). ~5-15 s per line on 2 CPU cores (a GPU: well under a second).

**English handwriting** (`recog/english.py`): Microsoft's TrOCR-small-handwritten (trained on the IAM
handwriting database; ONNX export, 0.25 GB, fetched by `download_models.py`), with the same beam search
and probabilities as the formula reader. On 150 IAM test lines (writers it was not trained on):
**CER 6.9 %, WER 17 %**, 37 % of the lines exact; the most confident quarter 1.3 % CER, the least
confident 17.7 % - 0.65 s per line on 2 cloud CPU cores, 0.26 s on the development PC, same numbers
there (`eval_english.py`).

*Bigger English reader (optional):* TrOCR-base-handwritten - `python download_models.py trocr-base-handwritten`
(0.6 GB), then `set EXAM_GRADER_ENGLISH=base`. Its full-precision encoder is needed: the published 8-bit
encoder reads IAM at 41 % CER (broken); with it and the 8-bit decoder: **CER 4.5 %, WER 11.6 %, 49 % of the
lines exact** on the same 150 lines (most / least confident quarter 1.2 % / 11.7 %), ~5x slower. Not the
default: the Hebrew/English/math decision rules were tuned on the small reader's outputs.

*Real exams, no form - reading with a grammar.* 50 scanned handwritten exams of a university Data Science
course (Mendeley Data `sf3kvjwknt`, CC BY 4.0, with the teacher's table of every student's multiple-choice
letter). The students wrote the 20 answers as a list ('1. C', '2. A', ...), in every layout - one column,
two columns, numbers in the margin. `tools/mcq_real_test.py` finds the short pieces of lines on every page and
reads them; with `--grammar` the reader may only produce '<number> <letter A-D>' (`recog/beam.py`: tokens that
cannot continue a reading of the pattern are never chosen - `FormulaRecognizer.recognize(..., grammar=(regex,
chars))`, using the `regex` module's partial matching):

| 20 exams, 400 answers | found | read right (of all) | right among the most confident quarter |
|---|---|---|---|
| TrOCR-small, plain | 35 % | 34 % | 97 % |
| TrOCR-small + grammar | 73 % | 58 % | 99 % |
| TrOCR-base + grammar | 75 % | **63 %** | 99 % |

(TrOCR-small + grammar on all 50 exams: 75 % found, 59 % right; most confident quarter 96 % right.)
The grammar doubles what is found; the misses are layouts (answers in the margin, two columns) and sentence
fragments forced into the pattern (most wrong readings are a spurious 'A'). Even for answers this simple,
free-form reading stays far from automatic grading (that needs >= 99.5 %) - which is what the printed answer
box solves.

## Mixed answers: Hebrew + equations

`demo_mixed.py` grades answers that mix Hebrew sentences and formulas (`data/demo/answer_A.png`,
`answer_B.png` - real handwriting: Hebrew words from test-split writers, MathWriting formulas):

```
page -> lines -> chunks (column gaps) -> Hebrew/math classifier (+ line-context smoothing)
     -> Hebrew segments: CRNN-CTC        math segments: Pix2Text MFR      (n-best + probabilities)
     -> rubric: "uses Ohm's law" (any equivalent form), "final answer 5 V" (units, tolerance),
        "says it is low" (Hebrew keyword, 1 edit / prefixes allowed)
     -> distribution over the total score -> confidence -> auto / human review
```

Layout = the weak link, so it is built defensively:

* two Hebrew/math classifiers (`models/script-classifier` trained on chunks of composed mixed lines,
  `models/script-classifier-tight` trained on whole words/formulas), averaged, Platt-calibrated on
  writers that are not in the evaluation set; every chunk is skeletonised and redrawn at one stroke
  width on a fixed canvas, so neither scan-vs-pen style nor batch padding can be used as a shortcut
* line-context smoothing (an unsure chunk between Hebrew words is probably Hebrew)
* a second opinion per chunk: both readers read it; if the classifier says Hebrew but the formula
  reader is confident about a plain formula (or vice versa) the label is corrected when the
  classifier was unsure, and the answer always goes to a human
* formulas cut apart by wide gaps (`V  =  5V`): an `=` cannot stand alone, so a chunk that reads as a
  bare operator is re-read together with its neighbours (unless they are clearly Hebrew) and the
  joined formula replaces the pieces - the answer then goes to a human
* English: the Hebrew/math classifier never saw English and calls most cursive English Hebrew. A
  second classifier, Hebrew vs English (`models/english-classifier`, trained by
  `train_english_classifier.py` on IAM lines vs Hebrew lines, 90 % of validation chunks right), and the
  English reader (which reads every chunk that is not a dot) decide together, safety first:
  - classifier >= 0.3 **and** the reader returns common English words at >= 0.9 per character -> English
  - classifier >= 0.9 alone -> English, but the answer goes to a human
  - an unsure chunk (classifier >= 0.7) next to an English word -> English, to a human
  - a Latin-script chunk the Hebrew/math classifier took for a formula is English when the reader
    returns common English words and the formula reader no plain formula
  - review only, labels unchanged: classifier >= 0.5 without the reader's agreement; the reader finding
    English words (>= 0.8) in a chunk called Hebrew or math; a Hebrew/math chunk right next to an English
    word with classifier >= 0.3; a dot or a single letter at the edge of an English phrase (inside one,
    it simply joins the phrase)

  The thresholds were tuned on development sets only (`data/mixed_dev_en`, `data/mixed_eval_en`,
  `data/mixed_eval`) with `tune_english_rules.py`, which mirrors the code and judges layout exactly like
  `eval_mixed.py`: a wrong layout that is *not* sent to review costs as much as 5 reviews.

`eval_mixed.py` measures this on lines composed from real handwriting of writers never seen in training
(Hebrew words from the test-split writers, English from IAM's test writers, MathWriting formulas), with
the formula / English part at the start, middle or end of the line:

| set | lines | role |
|---|---|---|
| `data/mixed_eval` | 100 Hebrew + math | development: the layout rules |
| `data/mixed_dev_en` | 100 Hebrew + English (+ math in half) | development: the English decision |
| `data/mixed_eval_en` | 120 Hebrew + English (+ math in half) | was the English test set; seen three times, then used for development |
| `data/mixed_eval2` | 100 Hebrew + math | was the Hebrew + math test set; seen again for each English rule version |
| **`data/mixed_eval3`** | 84 Hebrew + math | **test** - built after the rules were final, measured once |
| **`data/mixed_eval_en2`** | 120 Hebrew + English (+ math in half) | **test** - built after the rules were final, measured once |

The two test sets share no Hebrew line, IAM line or formula with any development set (IAM test writers
can repeat). Results with the final rules:

| | `mixed_eval3` (Hebrew + math) | `mixed_eval_en2` (Hebrew + English + math) |
|---|---|---|
| line split exactly right | **82 %** | **48 %** |
| split wrong **and not sent to review** (the dangerous cell) | **0 %** (0 of 84) | **5 %** (6 of 120) |
| sent to review | 62 % | 82 % |
| parts read as the right kind | Hebrew 99 %, math 89 % | Hebrew 96 %, English 74 %, math 74 % |
| formula read exactly / or equivalent (lines split right) | 10 % / 13 % | 27 % / 32 % |
| English CER (lines split right) | - | 14 % |

Re-measured once more with the final code after the phone-photo work below (columns, whole runs):
identical. About 7-9 s per line on 2 cloud CPU cores (one thread per process). On the old Hebrew + math test set
(`mixed_eval2`) the final rules give 70 % / 6 % dangerous / 64 % reviewed (without the English reader:
72 % / 9 % / 47 %): reading English costs a few more reviews on lines without English, and the "the
reader sees English words here" check also catches formulas hidden inside a Hebrew segment.

How the English rules got here, honestly - each version measured on the English test set of the time:

| version | `mixed_eval_en`: split right / dangerous / reviewed | what changed |
|---|---|---|
| v1 | 57 % / 5 % / 87 % | first rules |
| v2 | 45 % / 15 % / 72 % | tuned for English recall on `mixed_dev_en` |
| v3 | 45 % / 12 % / 75 % | re-tuned for safety on `mixed_dev_en` |
| v4 (final) | (development data by then) | review-only checks above; tuned on all three development sets, simulation mirroring the code exactly; measured once on the fresh sets: table above |

The remaining dangerous English lines are English words that both the classifier and the English reader
took for Hebrew or for a formula, with confidence. The grader's own reading confidence is the next safety
layer; a rubric that expects an English keyword could also force a review when no English is found.

Re-joining split formulas changes nothing on this set (its formulas are drawn compactly); it fixes
answers written with wide gaps, like student B's `V = 5V` in the demo. The Hebrew parts of these lines
were cut out of longer lines automatically and their reference texts are often off by a word, so
Hebrew accuracy is measured on the 300 test lines above instead.

A split counts as right when each part's segment holds >= 90 % of that part's ink and <= 10 % of any
other part's. An earlier version compared boxes with the pasted image widths, which include blank
margins, and so counted correct splits as wrong: it reported 63 % on `data/mixed_eval`, where the
ink rule gives 79 % (64 % by the old rule).

Formula *reading* on these varied MathWriting hands is the weak part (formulas shrunk to text
height): fine-tuning the formula model is the next step.

Note: in `grader/symbolic.py` the current `I` is mapped back to a variable - the LaTeX parser reads it
as the imaginary unit, and Math-Verify's `verify()` then says `I = V/R` equals `I = 5`. It also lost I's powers: 'I^2 R' was read as I R, so 'P = I R' passed as equal to 'P = I^2 R' and, in the derivation checks, 'P = I^2 R | I = 2 A, R = 5 Ω, P = 10 W' was right (27.10: a lone I is now parsed as a plain letter - P = 20 W). Math-Verify also strips what it takes for a unit at the end: a trailing c, d, g, h, l, m, o, s or t was dropped - 'E = m g h' read as E = m (so 'E = m g' passed as equal to it), '\omega t' as \omega, 'x_0 + v_0 t' as x_0 + v_0; that stripping is off now (our own unit handling cuts '5 s', '10 V' before parsing).

## Reading a whole scanned page - or a phone photo

```
python read_page.py scan.pdf        # PDF (Adobe Scan, phone, scanner) or an image
python read_page.py photo.jpg       # a phone photo of the page on a table works too
python read_page.py key.pdf --language english   # an exam in English: no Hebrew reader, English + math only
```

`recog/page.py` turns a real page into lines first: notebook rules and paper are removed (pen ink is
much darker than the rules), scanner borders, margin lines, punch holes, specks, underlines and lines
or waves drawn across the page are dropped, lines whose letters touch are split at deep valleys of the
row profile (never inside a fraction), every stroke goes to the line holding its centre, and large
drawings are set aside as figures. A short line written right under a long one - the end of a sentence
squeezed onto the next notebook line, where the long line's descenders (ק, ך) fill the gap - is found from
the row profile of its own columns (`_split_local`): only under a line, three or more letters one after the
other starting where the line starts, the long line's letters over it, no fraction bar, minus or wire near
the cut; the descenders stay with their line. On 27 real exam pages (KFUPM keys and quizzes, the first test
page) it changed no other line. Then every line goes through the mixed reader; the result is drawn
to `results/<name>_p1_read.png`.

Three things real answer keys needed (KFUPM EE201, below): **frames around answers** - a frame joins
the line above and below into one and is read as letters, so it is removed (one stroke or four, open
corners too) and remembered: a segment inside it is marked `boxed`, most likely a final answer;
**coloured pens** - red ink is as light as paper in plain grey levels, so strongly coloured pixels count
as dark (`ink_gray`); **decimal points** - a dot next to writing is no longer dropped as a speck (26.67
had become 2667). `--language english`: what the Hebrew/math classifier calls Hebrew is English text,
the English and formula readers cross-check each other, a chunk of digits is a number, and lines read
left to right.

A **phone photo** (the table shows around the page, or the light falls off across it) is flattened
first (`flatten_photo`): the paper is found (bright and grey, a table is more coloured), straightened
by a perspective warp, its light evened out, and the notebook rules removed - in a photo a blue pen and
the blue rules are equally dark, so rules are found by shape (long thin straight lines), and where a
pen stroke crosses a rule it is kept. Two more things a real page needed: two formulas side by side
(a column gap) stay two answers, and a run of writing is also read *as a whole* - this hand leaves
gaps between the characters, so "12+12+13=34" came apart into 9 pieces that were judged one by one,
while the formula reader reads the whole run right; the same for English sentences. A run read
differently as a whole than piece by piece always goes to review.

First real test - a notebook page in a new hand (Hebrew, English, a formula, a sketch):

| on the page | read as |
|---|---|
| כתבתי שטויות לבדיקה. | **כתבתי שטויות לבדיקה.** (exact) |
| x² + y² + 4 over 12 | **`\frac{x^{2}+y^{2}+4}{12}`** (exact) |
| the parabola sketch | set aside as a figure, not read |
| של הפרבולה היא / בשם של משתמש אחר / שאומר אם נכנסת או נכנסת | right but a letter or two per word (משתוול, וכנסת) |
| English: Action log, login | "Action og" (read as one English phrase, to review), **login** |
| English: login as, flag, it is incorrect | "logger - as", a formula `l_{ag}`, "income": all three lines sent to review |
| a line drawn over / squeezed handwriting (bla bla (x+y²)/8⁴ = bla) | garbage, or `[unreadable]` and sent to review |

Every line of this page with a wrong split or a wrong kind went to review. What would help most now:
a few dozen labelled lines of the course's own hands (`label_tool.py`, then `train_hebrew.py --init`) -
the Hebrew readings here are right in shape but wrong in a letter or two per word.

Second real test (26.10) - Tal's notebook page: four Hebrew sentences (one is "גנן גידל דגן בגן"), three
English lines, three exercises (Pythagoras twice, a circle's area), scanned with a phone. The page and its
hand transcription are in `data/my_scans/tal_notebook.pdf` / `tal_notebook_truth.json`;
`python tools/score_page.py data/my_scans/tal_notebook.pdf` scores a reading of it (words and formulas,
however the page was cut):

| Tal's page | before | with the short-line split | + the Hebrew reader trained further |
|---|---|---|---|
| Hebrew words right | 11 / 29 | **15 / 29** ("אני בודק אור איך אתה קורא" + "עברית." instead of "עערדוק אוגר איך אתה היא") | 13 / 29 |
| English words right | 4 / 13 | 4 / 13 | 4 / 13 |
| formulas right | 7 / 10 | 7 / 10 (missed: the 4 of 4² read as y at 99 %, "16+4", "r=5" after a bar called Hebrew -> review) | 7 / 10 |

The English lines stay weak even when the reader is told they are English: TrOCR-small reads "hi there cladue,
how" as "hi there said we , now", TrOCR-base as "Initi there's active , now" (a word that is not a word,
"cladue", pulls both towards real ones; an odd h). The Hebrew misreadings are a letter per word (איך -> אזר,
בגן -> באן) - this hand's own lines would teach the reader that (`label_tool.py`).

**Without a key: a derivation checked against itself** (`python read_page.py scan.pdf --check`,
`grader/selfcheck.py`). The math pieces of the page are cut into statements in reading order: a formula with
letters ('a^2+b^2=c^2', 'A = \pi r^2') opens a problem; 'x = number' is a value - the last one written for a
letter of the formula is the student's result, the ones before it the data; the formula is solved for the
result's letter and compared with what was written (half a unit in the last written digit, or 1 %); a
statement with numbers only is checked as written; a letter that is not in the problem's formula is
reported as a likely misread digit; a value the formula needs but the math pieces lack is looked for in
what the formula reader made of the pieces of the same line that the layout took for text ('|r=5' read as
Hebrew). On Tal's page, from the clean transcription, all three verdicts are right: (1) a=3, b=4 gives
c = 5, written 9 - wrong; (2) '4²+2² = √20' is false (20, not 4.47), but 20 is c², so √20 is the right c in a
wrong equality; (3) A = 25π = 78.54 - right. From the tool's own reading of the page:

```
problem 1: WRONG - a²+b²=c² with a=3, b=4: c = 5, but written c=9: wrong (reading 40%, unsure: a human should look)
problem 2: MISREAD? - 'y²+2²=√20': y is not in a²+b²=c² - a misread digit? (reading 78%, unsure: a human should look)
problem 3: RIGHT - A=πr² with r=5 (r=5 read from a piece taken for text): A = 78.54 - written A=78.54: right (reading 50%, ...)
```

(the 4 of 4² was read as y: the misreading shows itself). A verdict resting on a piece read below 80 % (or on
a value found in a text piece) is only a suggestion, and a confident misreading can still give a wrong
verdict (π read as π² would make 78.54 "wrong"), so it changes no grade: it tells the person checking where
to look.

**Math validation - what it checks** (26.10; `tests/test_grader.py::test_math_validation_*`):

| the student wrote | the tool says |
|---|---|
| `a²+b²=c² \| a=3, b=4`, then `c = 9` | WRONG: with a=3, b=4, c = 5 |
| `a²+b²=c² \| a=3, b=4`, `c² = 9+16 = 24`, `c = √24` | WRONG: c = 5; the step '9+16 = 24' is false (25) - where the slip is |
| `a²+b²=c² \| a=3, b=4`, `c² = 25` (no 'c = ...') | RIGHT: c taken from c² = 25 (the positive root) |
| `4²+2² = √20` under `a²+b²=c²` | RIGHT VALUE, WRONG EQUALITY: the left side is 20 = c², so c = √20 is right |
| `V = I R \| I = 2 A, R = 5 Ω`, `V = 10 V` | RIGHT (units set aside) |
| `2x+3 = 7`, `x = 5` | WRONG: the equation gives x = 2 |
| `12+12+13 = 34` | WRONG STEP: 37 |
| `A ≈ πr² \| r = 5`, `A ≈ 78.5` | RIGHT (≈ counts as =; 78.5 is within half a unit of 78.54) |
| `a = 3`, `b = 4`, then `c = √(a²+b²) = 6` (the data first, no formula line) | WRONG STEP: 5, not 6 (27.10: before, nothing was checked - the data above a computation were not used) |
| `V = 10 V`, `I = 2 mA`, then `R = V/I`, `R = 5000 Ω` | RIGHT - the givens just above a formula count for it, and with every value written with its unit the formula is worked out in SI units (27.10: before, 'no value for I, V'; and with the givens beside the formula, '5000 Ω' was judged wrong and '5 Ω' right - 2 mA taken for 2) |
| `P = V²/R \| V = 10 V, R = 2 kΩ`, `P = 0.05 W` / `P = 50 mW` | RIGHT both (the answer compared in its own unit) |
| `x = 3.\overline{3}`, `V = 3.33 [V]` | 3.333..., and the unit in brackets read as a unit |
| `\eta = P_{out}/P_{in} \| ...`, `\eta = 80\%` | RIGHT: 80 % is 0.8 (also against a formula with '\cdot 100\%' in it) |
| `R = \Delta V / \Delta I \| \Delta V = 2 V, \Delta I = 4 mA`, `R = 500 Ω` | RIGHT: '\Delta V' one name (it was delta times V), and '1.5e-3' is 1.5 times 10^-3 (it was e - 3) |
| `V = I R = 2 mA · 5 kΩ = 10 V` (`[mA]` too) | RIGHT: a substitution written with its units is worked out in SI units, step by step (before, the units' letters were taken for letters of the problem and nothing was checked) |
| `sin(30°) = 0.5`, `P = V I cos φ \| φ = 60°` | RIGHT: degrees read as degrees (sin(30°) was taken in radians: 'wrong step') |
| `X_L = 2π(50)(0.1) = 31.4`, `2(3) = 6` | RIGHT: products ('2(3)' was read as 5) |
| the reader's `V _ { o u t }`, `2 \mathrm { m A }`, `1 0 V`, `\left( 5 0 \right) \left( 0 . 1 \right)` | read as the clean forms (the circuit checks' evening-out applied here too: 18 of 55 correct derivations, rewritten as the reader writes, got no verdict - now 1, and none judged wrong either way) |
| `I = 10/(3 + j4) = 1.2 - j1.6 = 2∠-53.13°`, `\|3 + j4\| = 5` | RIGHT: j is the imaginary unit, a phasor in polar form is read (in degrees), a magnitude's bars are no separators - complex arithmetic used to get no verdict |
| `s^2 + 3s + 2 = 0`, `s_1 = -1`, `s_2 = -2` | RIGHT, RIGHT: an equation's roots named apart each checked (no verdict before) |
| `\sum_{k=1}^{4} k = 10`, `\lim_{x \to 0} \sin x / x = 1`, `\int_0^1 x dx = 0.5` | RIGHT: worked out (the '=' inside the sum's braces cut the statement; the bound letter was taken for a letter of the problem) |
| `R_eq = 2 \| 3 = 1.2`, `R_eq = R_1 \| R_2 \| R_1 = 2, R_2 = 3` | RIGHT: '\|' / '\parallel' / '\|\|' are resistances in parallel ('2 \| 3' was read as 3: a false 'wrong step'), in the rubric's formula comparison too |
| `V_out = V_in R_2/(R_1+R_2) \| V_in = 12, R_1 = 1k, R_2 = 2k`, `V_out = 8` | RIGHT: a value with a bare prefix ('1k', '10\mu') is a value in the units as written (it was taken for a formula, and broke the problem up) |
| `\frac{d}{dx} x^3 = 3x^2`, `\int x\,dx = \frac{x^2}{2} + C`, `\int \sin x \cos x\,dx = -\frac{\cos^2 x}{2} + C` | RIGHT: a derivative or an integral worked out in letters is compared as an expression (decided at random points, where simplify may miss an identity; an indefinite integral up to its constant) - '= 3x' is a WRONG STEP; before, none of it was checked. A limit, a sum or a bound in letters may be an equation (`\int_0^a x\,dx = 2a`, a = 4): right when it holds, never wrong |
| `\int_0^2 x^2\,dx = \left[ \frac{x^3}{3} \right]_0^2 = \frac{8}{3}` | RIGHT, RIGHT: an integral's evaluation bracket (`\left. ... \right\|_0^2`, `[...]_0^2` too) is worked out |
| `f(x) = x^2 + 1`, `f(2) = 5`; `f'(x) = 2x`, `f'(3) = 6`; `y = x^3`, `y' = 3x^2` / `\frac{dy}{dx} = 3x^2` | RIGHT: a function the student defines is used; its derivative checked against the derivative itself (`f'(x) = x` WRONG STEP); `f(x) = 7` of a known f is an equation. The parser dropped primes - y' was y, and `V_A = V_A' + V_A''` was V_A = 2 V_A (superposition shares) - now names of their own |
| `x = 2`, `\frac{dy}{dx} = 3x^2 = 12` | RIGHT for 3x² = 12; dy/dx of a letter nobody defined is left alone (it was worked out as 0: a false WRONG STEP) |
| `v = \frac{d}{t} \| d = 100 m, t = 10 s`, `v = 10 \frac{m}{s}`; `a = 3 m/s^2`; `\rho = 2 \frac{g}{cm^3}`; `72 \frac{km}{h} = 20 \frac{m}{s}` | RIGHT: units as fractions, slashes and powers of a length (before: 'cannot check', and a conversion was no statement) |
| `T = 2\pi/\omega \| \omega = 1000`, `T = 6.28 ms` | RIGHT: values written bare are taken in SI units or in the student's own (it was WRONG: 6.28 against 0.00628). With `C = 1 \mu F` beside a bare omega, `X_C = 1 kΩ` is right and `X_C = 0.001` (the micro left out?) a suggestion for a human |
| `x = \pm 3`; `x_{1,2} = \frac{5 \pm 1}{2}`; the quadratic formula, then `x_1 = 2`, `x_2 = 3` | RIGHT: '±' gives both roots, named either way round ('100 ± 5', a tolerance, is left alone) |
| `2x = 4 \text{ ולכן } x = 2`, `x = 1 \text{ or } x = -2`, `3x = 12 \rightarrow x = 4`, `2 · 3 = 6` | RIGHT: words between statements (Hebrew, 'or', 'so') and arrows cut them (each was one false WRONG STEP); '·', '×', '÷', '−' are signs ('2 · 3' was read 2.3) |
| a page of numbered exercises: `1) P = V^2/R \| ...`, `P = 50 mW`, `2) \tau = R C = 1 kΩ · 1 µF = 1 ms`, `3) f(x) = x^2 + 1`, `f(2) = 5`, `f'(x) = 2x` | RIGHT for each: a number '2)' starts a new exercise whatever its first line is (the chain under exercise 1 was never checked), and a function's later lines stay with its definition (f'(x) became a problem of its own, f unknown there) |
| `I = V/R`, `I = \frac{12}{4}`, `I = 3 A` (no values for V, R) | RIGHT for 12/4 = 3, a computation written a line at a time - the problem itself 'cannot check' |
| `\mathcal{L}\{e^{-2t}\} = \frac{1}{s+2}`, `\mathcal{L}^{-1}\{\frac{1}{s(s+1)}\} = (1 - e^{-t}) u(t)` | RIGHT: Laplace transforms and their inverses worked out (u(t) the step, δ(t) the impulse); `= \frac{1}{s-2}` a WRONG STEP |
| `\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix} = -2`, the reader's `\left\| \begin{array}{ll} 1 & 2 \\ 3 & 4 \end{array} \right\| = -2` | RIGHT: determinants (the matrix's '\\' was taken for a space by the reader's evening-out: a false WRONG STEP, '4 is not -2') |
| `10 = 5I_1 - 2I_2`, `0 = -2I_1 + 6I_2` (one per line, or in a `cases` brace, or the reader's `\left\{ \begin{array} {l} {...} \\ {...} \end{array} \right.`), then `I_1 = 2.31 A`, `I_2 = 0.77 A` | RIGHT, RIGHT: a system of equations (mesh, node) solved together and every value written after it checked - before only the last equation was used, so a wrong I_1 consistent with it passed; with no single solution (a letter without a value), the last equation alone as before |
| `V(s) = \frac{10}{s(s+5)}`, then `v(t) = 2 - 2e^{-5t}` (`I_L(s)` / `i_L(t)`, `H(s)` / `h(t)`) | RIGHT: a time function written after its transform by the same letter is checked against the inverse transform (t > 0); `2 + 2e^{-5t}` a WRONG STEP. The t of 'i(t)' came out of the parser complex, unlike every other letter - i(0) was never worked out |
| `\frac{s+3}{(s+1)(s+2)} = \frac{2}{s+1} - \frac{1}{s+2}`; `= \frac{A}{s+1} + \frac{B}{s+2}`, `A = 2`, `B = -1` | RIGHT: a partial fraction expansion checked (a wrong one is wrong), and its coefficients once found; `(a+b)^2 = a^2+2ab+b^2`, `sin²x + cos²x = 1`: right when an identity holds - any other relation in letters stays an equation to solve |
| `\frac{d^2}{dx^2} x^3 = 6x`, `y''`, `\frac{d^2y}{dx^2}` under `y = x^3`; `\frac{\partial}{\partial x}(x^2 y) = 2xy` | RIGHT: second derivatives (read as d² · x³ / dx² before), their order kept; a partial derivative takes the other letters for constants |
| `t = 2 min = 120 s`, `y = x^2` then `y(3) = 9` | RIGHT: minutes and hours beside seconds; a letter defined by a formula evaluated at a value (`A = 2 h` with h = 3 stays a letter) |
| `\tau = R C = 1 kΩ · 1 µF = 1 ms`, `= 22 m` (for 22 ms), `F = m a = 2 kg · 3 \frac{m}{s^2} = 6 N`; `F = \mu N = 0.3 N = 6 N` | RIGHT, RIGHT, RIGHT; no verdict for the last - a one-letter unit that is a letter of the problem on the line (N the normal force) is no unit. The first two were false WRONG STEPs (the last side left as written: 0.001 is not 1), the third got no verdict |
| `R = V/I \| V = 10 V, I = 2 mA`, `R = 5 kV` | RIGHT VALUE, WRONG UNIT (a suggestion): the unit the formula gives the letter from the data's units - Ω, not kV (it passed as right: 5 kV is 5000 of something); `5 kΩ`, `5000 V/A` right; `P = 30 J` for 20 W: wrong, and its unit noted |
| `\frac{d}{dt}(A\cos(\omega t)) = -A\omega\sin(\omega t)`, `L \frac{d}{dt}(I_m \sin(\omega t)) = L I_m \omega \cos(\omega t)`; `= A\omega\sin(\omega t)` | RIGHT; a sign slip a WRONG STEP: the other letters taken for constants (no verdict before - a letter may be a function of t); sure for the usual names of constants (A, k, n, \omega, \tau, ...), for others (V_m, \phi) right but not sure and never wrong; `\frac{d}{dt}(C v) = C \frac{dv}{dt}` and `\frac{d}{dt}(J\omega^2/2) = J\omega\alpha` get nothing |
| `\cos(n\pi) = (-1)^n`, `\int_0^{\pi} \sin(nx)\,dx = \frac{1-(-1)^n}{n}`, `a_n = \frac{2}{\pi}\int_0^{\pi} x\cos(nx)\,dx = \frac{2}{\pi} \cdot \frac{(-1)^n-1}{n^2}` | RIGHT: a Fourier series' index (n in (-1)^n, \sin(n\pi), e^{-jk\pi}) is a whole number (no verdict before); `= \frac{2}{n}` (odd n only) gets none |
| `\frac{d}{dt}(e^{-\alpha t}\cos(\omega_d t)) = -\alpha e^{-\alpha t}\cos(\omega_d t) - \omega_d e^{-\alpha t}\sin(\omega_d t)`; `r_e = V_T / I_E` | RIGHT: a subscript without braces is one character ('\omega_d t' was one name omega_{dt}: a false WRONG STEP; 'r_e', '\omega_e' were not read at all) |
| `Z = 3 + j4`, `I = \frac{10}{Z} = 2\angle -53.13^\circ`; the mesh equations `(2+j3) I_1 - j3 I_2 = 10`, `-j3 I_1 + (4+j1) I_2 = 0`, then `I_1 = 1.786 - j1.071`, `I_2 = 1.515\angle 45^\circ` | RIGHT: complex values are data (they were dropped: no verdict), phasor systems solved together; `= 2\angle 53.13^\circ` a WRONG STEP |
| `s^2 + 2s + 5 = 0`, `s_1 = -1 + j2`; `Z = R + j\omega L \| R = 3, \omega = 1000, L = 4 mH`, `\|Z\| = 5`; `I = 2\angle 30^\circ`, `R = 5`, `P = I^2 R = 20` | RIGHT: complex roots; a magnitude against the size of the result; complex data by their sizes for a real answer (power formulas) |
| `J = \sigma E \| \sigma = 2, E = 3`, `J = 6`; `j = 3`; `\Re\{Z\} = 3`, `\angle Z = 53.13^\circ`, `S = V I^*` | RIGHT for J (a capital J was sqrt(-1): WRONG); nothing for the rest (read as products of letters before) |
| `I = 2\angle -53.13^\circ`, then `i(t) = 2\cos(1000t - 53.13^\circ)` (or `2.83\cos(...)`, `2\sin(1000t + 36.87^\circ)`) | RIGHT: a time function against its phasor (peak or RMS, its cosine for reference); `2\cos(1000t + 53.13^\circ)` WRONG but not sure (another convention?) |
| `H(s) = \frac{1}{1 + sRC}`, then `H(j\omega) = \frac{1}{1 + j\omega RC}`; `H(s) = \frac{1}{s+1}`, then `H(j2) = 0.447\angle -63.43^\circ` | RIGHT: s = jω put into H(s) (before: H(jω) read as a product, no verdict); `\frac{1}{1 - j\omega RC}` a WRONG STEP |
| `(1011)_2 = 11`, `1011_{(2)} = (11)_{10}`, `(FF)_{16} = 255`, `(101.1)_2 = 5.5` | RIGHT: numbers in their base (each was a false WRONG STEP: 1011, or (1011)_2 as a falling factorial); `(1011)_2 = 12` a WRONG STEP; `(129)_2` (no such digit) nothing |
| `I = \frac{10}{2 k\Omega} = 5 mA`; `P = V I = 10 V \cdot 2 mA = 20 mA` | RIGHT (the unit in the denominator was applied to the whole fraction: a false WRONG STEP); for the second, RIGHT and a WRONG UNIT note (W against A) |
| `\sqrt{2} = 1.4`, `I = \frac{10}{3} = 3.3 A`, `\frac{100}{7} = 14` | RIGHT: rounded to the decimals written (each was a false WRONG STEP); `\frac{2}{3} = 0.6`, `\frac{23}{5} = 5`, `0.0001` for 0.00005 stay WRONG STEPs |
| `\cos(60) = 0.5`, `\varphi = \arctan(\frac{4}{3}) = 53.13`, `P = V I \cos\varphi \| \varphi = 60`, `1 + j = 1.414 e^{j45}`, `3 + j4 = 5\angle 0.927` | RIGHT: degrees with the mark left out, a phasor's angle in radians (false WRONG STEPs before); `5\cos(10) = 4.92` right but not sure (radians, a calculator in degrees?); `\cos(60) = 0.6` a WRONG STEP that says 0.5 in degrees |
| `A_v = 100 = 40 dB`, `\|H\| = 0.707 = -3 dB`, `A_v = g_m R_D`, then `A_v = 40 dB` | RIGHT: 20 log10 (false WRONG STEPs: '100 is not 40'); `A_p = 100 = 20 dB` right but not sure (a power's 10 log10); `= 30 dB` a WRONG STEP: 100 is 40 dB |
| `\boxed{V = 10 V}`, `\underline{10 V}`, `V \overset{KVL}{=} I R`, `\tau := R C` | read as their content, an equals sign (a box hid the answer from the checks, and was a parse error for the rubric) |
| `y = (x+1)^2 = x^2 + 2x + 1`, `X(s) = \frac{1}{s(s+1)} = \frac{1}{s} - \frac{1}{s+1}` | RIGHT: the steps in letters along a chain (nothing before); a wrong partial fraction expansion a WRONG STEP, other relations in letters may be equations to solve (`V_A = 10 - 2I = 4I + 2`) |

The verdict is weighed over the readings: each piece's n-best readings (and, with a second formula reader,
its reading too - `EXAM_GRADER_FORMULA2=mumer`, Uni-MuMER; two readers that agree add up their votes) are
combined, and the verdict's confidence is the probability of the readings that lead to it, times the
layout's confidence. With two readers each reading counts with its reader's own probability of it (not
normalised over that reader's list: Uni-MuMER returns one reading, and alone it would count as certain), and
readings that differ only in how items are separated ('|', '\vert', ',') are one. On Tal's page this keeps
Pix2Text's '2 | a²+b²=c² | y²+2²=√20' (p 0.61) over Uni-MuMER's garbled '2 7 a²+b²=c² 4²+2²=√20' (p 0.18),
and takes Uni-MuMER's '1) a²+b²=c² | a=3, b=4' (p 0.42) over Pix2Text's '| | a²+b²=c² | a=3 | b=4' (p 0.10) -
before, the lone second reading always won and problem 2 was swallowed into problem 1 with two false steps.

## Grading a page against a rubric

```
python grade_page.py photo.jpg rubric.json            # -> results/<name>_grade.txt / .json
python grade_page.py photo.jpg rubric2.json --reuse   # a changed rubric, without reading the page again
```

A rubric lists the items and their points (`data/my_scans/silly_test_rubric.json`):
`{"id": "5", "points": 8, "arithmetic": "19-4"}` - the exercise's line is found by its left side
(over the n-best readings; an item number like `2)` in front is allowed), its value is the answer;
`{"id": "a-i", "points": 4, "value": "i", "answer": "-4 A"}` - a named result (circuits): its line is
found by the name on its left side (case, braces, accents, `(b)` in front do not matter; a boxed answer
wins), the last numeric side is the value, units are checked (`45w`, `24 Volt` accepted; since 27.10 also a unit in square brackets - `3.33 [V]`, `2 [mA]` -, `80\%`, `45^\circ`, a number written as a fraction or a root - `\frac{10}{3} V`, `10\sqrt{2} V` -, a repeating decimal, and a bare prefix: `5k` for 5 kΩ (the unit then counts as missing), `5 m` for 5 mA - all of them were parse errors or a unit mismatch before); a formula answer is compared symbolically (`P = I^2 R` and `P = I R` are no longer the same, nor `E = m g` and `E = m g h`, nor `R_1 \| R_2` taken for R_2; an equation with an infinite side, '\lim ... = -\infty', is compared side by side - against itself it came out 'not equivalent', sure, in this environment's library versions: 3 of the 70 real formulas below were graded wrong automatically, now none);
`{"id": "Q1", "points": 8, "open": "..."}` - open answers go to a human. An item is graded
automatically only if its line is found with probability >= 50 %, the readings agree on the score
(>= 85 %) and the layout was sure it is a formula; otherwise it goes to a human, with the reason and
the tool's guess.

First real test - a "silly test" page photographed on a table (arithmetic in two columns, English and
Hebrew answers, a formula, scoring notes in the margin; 100 points as written on the page):

| item | written | read as | tool | correct |
|---|---|---|---|---|
| 1+5 | = 6 | `1+5` (the 6 looks like a G) | human: not found | 8 |
| 70/7 | = 10 | **70/7=10** | human (readings 72 % sure), guess 8 | 8 |
| 7x7 | = 49 | `7×7=L10` | human: answer unreadable, guess 0 | 8 |
| 14x2 | = 28 | `H×Z=28` | human: not found | 8 |
| 19-4 | = 11 | `1^{0})-4=11` | human: not found | 0 |
| 8x8 | = 64 | **8×8=64** | **8 / 8 automatically** | 8 |
| 98-11 | = 78 | `-11=78` ("98" taken for Hebrew) | human: not found | 0 |
| 36/6 | = 6 | "36/6" taken for a Hebrew word | human: not found | 8 |
| 1) dogs' legs | "normal dog have 3 legs" | **Normal dog have 3 legs** | human: open | 0 |
| 2) world wars | 1917 / 1946 | the Hebrew words partly, the years not | human: open | 0 |
| 3) | x²+y²=c² = ... | **x²+y²=c²=(x+y)+(y+y)=** ... | human: open | 0 |
| 17+17+24 | = 48 | `17+17+24=4z` | human: answer unreadable, guess 0 | 0 |
| 12+12+13 | = 34 | **12+12+13=34** | human (read as a whole), guess 0 | 0 |

The tool graded 8 points on its own (correctly) and sent 92 to a human; the right total is 48 / 100.
Of its guesses for the items it sent on, 3 of 4 are right (7x7 = 49 is misread). No automatic grade was
wrong. Read as whole runs, "98-11=78" and "36/6=6" come out right too - but the same check, unguarded,
also read short Hebrew words into formulas in the demo answers ("ולכן V=5V" -> `V=5V_1S_1`), so a run
becomes one formula only if no piece of it looks clearly Hebrew (to the Hebrew/English classifier, or a
word the Hebrew reader is fairly sure of), and these two runs do not pass. What limits this page most is
the formula reader on this hand (4, 9 and 6 drawn unusually): fine-tuning it on handwritten arithmetic
is the next step for pages like this.

Second test - three answer keys handwritten by a lecturer (KFUPM EE201 Quiz 2, Quiz 4, Major Exam 1:
`data/online_keys/`, `--language english`), 20 final answers: none graded automatically; the tool's
guess for a human was right for V1 = 15 V, P3A = -45 W, Rab = 15 Ω and wrong for vx = -26.67 V (the
minus is a short bar written above the number); the other 16 were not found or not readable. Two more keys - KFUPM
EE204 Major Exam 1 (same lecturer) and EE201 Major Exam I 2001 (another lecturer's hand, scanned at
300 dpi, cut at the right edge) - 15 answers: none automatic, right guesses for i = -5/2 A, V2, V3,
Req, a wrong one for V = -3 V (two boxed answers on one line read as one), 10 not found. All five:
35 answers, 0 graded automatically, 7 right and 2 wrong guesses for a human, 26 not found or not
readable. So on free-form solutions the reader finds about a quarter of the final answers.
Value items take `"page": n` (two questions may both ask for `i`), a fraction is evaluated
(`-5/2 A`), and a line of several statements (`... => 6i = -15 => i = -5/2 A`, `V3 = 235, V4 = ...`) is
searched statement by statement. Answer cells on a printed
form (one digit per cell, sign and decimal point in their own cells) are the way to near-complete
reading - see the roadmap.

## Answer-box exams: "חשמל 101" end to end

The roadmap's route to near-complete reading, built and measured on an invented course: the final
answer of every question goes into a printed box, one character per cell, and only the boxes are read.

```
python -m forms.make_exam                 # print the course's exams -> data/course_elec101/E1..E5.pdf (+ _key.pdf, .json)
python grade_form.py student_scan.pdf     # -> results/<name>_form.txt/.json + the pages with what was read in every box
python -m forms.simulate                  # simulated students (sim_dev, sim_test) ...
python -m forms.evaluate sim_test         # ... graded and compared with what they wrote
python -m forms.cells train --sources ... # (re)train a cell reader (see forms/cells.py); in use: models/cell-reader-v2/
```

**The course** (`forms/elec101.py`): five exams of 100 points - charge, current and Ohm's law; power and
energy; series and parallel; Kirchhoff's laws (one answer is negative); a final with capacitors, AC and
one open question. 26 numeric answers and 5 multiple-choice questions; every reference answer is
computed in code, and all 26 were checked by an independent circuit solver (nodal analysis).

**The form** (`forms/make_exam.py`, Hebrew, HTML -> PDF): per page four corner markers and a QR code
(course | exam | page); the student's ID in 9 cells; per question a work area and the answer box (8 cells:
digits, a decimal point, a minus; the unit printed after it; one cell for a multiple choice). The
template (every cell's box in 150-dpi page pixels) is saved with the exam. Optional: `--digit-guide` (or
`"digit_guide": true` in the course) adds a line "write the digits like this - 1 as a straight line, 7 with
a crossbar" with model digits, against the readers' most common confident mistake (all five exams still fit;
`--out <folder>` builds a trial copy - a course's simulated classes are tied to its templates).

**Reading** (`forms/read_form.py`): markers -> homography onto the template; the QR code says which exam
and page (and which way up); every row of cells is matched to the printed borders within a few pixels;
every cell is read by a small CNN (`forms/cells.py`: digit / minus / point / empty / unreadable, trained
on synthesized cells - MNIST training digits and MathWriting's isolated digits, borders, specks, blur;
98.95 % on cells made from MNIST's test writers - the first reader; the current ones are below), twice -
inside its border and slightly beyond it - and
the cells' probabilities make an n-best list of the answer. Rules that came out of the development set:

| rule | why |
|---|---|
| an empty-looking cell between two characters with a trace of ink is a decimal point | the form says no gaps; a faint point was the most common miss |
| the answer's digits with the point elsewhere (24 for 2.4) never gets an automatic 0 | a missed point must not cost the whole item; a real factor-10 slip is confirmed by a human |
| ink in two or more gaps between cells -> a human | a box crossed out; one gap alone is mostly a big digit (13x as often) |
| a share of the reading that looks unreadable counts against the confidence | ignoring even 5 % turned a doubtful 7 into a sure 4 |
| automatic only when the most likely score has >= 98 % | development set: 0.95 -> 2 automatic errors in 744 answers, 0.98 -> 0 |
| the ID: its most likely reading must have a valid check digit and >= 98 % of the valid readings | picking a less likely valid reading gave a wrong ID once in each set |

**Measured** on simulated students (`forms/simulate.py`): 24 per exam, handwriting from MNIST's TEST digits
(writers the reader never saw), each student with his own versions of the digits, size, slant and pen;
right answers rounded his way or typical mistakes (sign, factor 1000, early rounding, wrong formula),
empty answers, now and then a crossed-out box; pages scanned (turned, blurred, JPEG). Settings chosen
on `sim_dev` (120 students), measured once on `sim_test` (120 others):

| sim_test, 120 students, 744 answers (first reader) | |
|---|---|
| cells read right | 99.45 % |
| answers read exactly right | 94.1 % |
| answers graded automatically | 68.8 % (512) |
| automatic grades that were wrong | **0** |
| student IDs accepted automatically / wrong | 106 / 0 |
| points graded automatically, per student | 66.5 % on average; 16 students graded fully |

What goes to a human (232 answers): a share of the reading that looks unreadable (101, mostly a few %),
readings that disagree on the score (53), the answer's digits with the point elsewhere (26 - most are the
students' own factor-10/1000 slips), the simulated crossed-out boxes (25, all caught), writing across the
cells (23), other (4).
Phone photos work too (markers + QR found on every simulated photo), but the course will be scanned to
PDF, so the final numbers are scans. The handwriting here is MNIST's - a real class will differ; the next
measurement is real filled forms.

### Handwriting the reader never saw, and the second reader (v2)

MNIST-style digits are the easy case. The same simulation with digits from four other collections - each
120 exams, never used for training or tuning (`python -m forms.simulate --digits usps_test --out sim_usps`
...; `python tools/compare_readers.py sims --model <reader> --tag <name> sim_usps ...`): USPS (scanned
envelopes, 16x16), Semeion (Italian writers, 16x16 black-and-white), digits drawn on screens, and CVL
(TU Wien: Austrian writers, scanned - the closest to real exam scans, and European-style digits).

The first reader (MNIST only) read CVL's European '1' - with its long flag - as '7', confidently: 33
cell errors and 2 wrong automatic grades. So the second reader was trained on 678k synthesized cells from
many more writers: MNIST, EMNIST (NIST, 3,600 writers), HWD-V1 and 57 of CVL's 67 training writers (the
other 10 calibrate it; CVL's 60 test writers stay out). Checkpointed and resumable (`--cache` keeps the
synthesized cells), 2 epochs, ~90 min on 2 CPU cores. Two such readers with different seeds
(`models/cell-reader-v2/main`, `.../b`) are averaged (`models/cell-reader-v2/ensemble.json`) - the readers
now in use; reading takes ~1.5x as long. A third with random turns / shears of the training cells
(`--augment`) was worse (Semeion 80 % of digits vs 90 %) and is not used.

| 120 exams per set | answers read right v1 -> v2 | graded automatically v1 -> v2 | wrong automatic grades v1 -> v2 |
|---|---|---|---|
| sim_test (MNIST writers)* | 94.1 -> 95.8 % | 68.8 -> 69.0 % | 0 -> 0 |
| screen-drawn | 94.2 -> 95.0 % | 70.6 -> 74.6 % | 0 -> 0 |
| USPS | 87.8 -> 89.8 % | 60.5 -> 64.4 % | 2 -> 2 |
| CVL (European style) | 80.4 -> 89.5 % | 43.0 -> 58.2 % | 2 -> 1 |
| Semeion (16x16 black-and-white) | 70.7 -> 78.1 % | 25.0 -> 20.4 % | 0 -> 0 |

\* EMNIST and MNIST both come from NIST's forms, so v2 may have seen sim_test's writers.
The wrong automatic grades are digits that look like the other digit even to a person: USPS's '4' drawn as
one straight stroke (read '1') and a '5' shaped like a '3'; on CVL a student whose '1' is a long-flagged hook
(read '7' at 99 %; one reader alone made 4 such errors, the pair 1). The remedies are real handwriting of the
students (copy sheets, below) and a line on the form asking for one digit style (a plain '1', a crossed '7').
On Semeion's blocky 16x16 digits the pair is more often unsure (more answers right, fewer graded
automatically).

The most realistic of these tests (`--writer-consistent`): every simulated student is ONE real CVL writer
and writes every digit with that person's own varying instances (60 writers, each twice), instead of three
versions per digit picked from the whole collection. v1 -> v2: answers read right 79.7 -> 87.0 %, graded
automatically 39.9 -> 54.8 %, 2 wrong automatic grades for both (2 of ~408 automatic grades, 0.5 %: a '9'
whose loop is a filled blob, read '8'; a '3' with a closed top loop, read '8'); at a 99 % threshold 37.5 %
automatic and none wrong. That is the honest estimate for a class whose handwriting the readers have never
seen - which the copy sheets fix.

Give the course its roster - any text or CSV file with the enrolled students' IDs (a Moodle or registrar
export; IDs whose leading zero a spreadsheet dropped are padded back) - as `"roster": "students.csv"` in
`course.json`, or `python grade_form.py scan.pdf --roster students.csv`. The 9 ID cells are then matched
against the enrolled students instead of being read free: the roster ID that best explains the cells'
probabilities is accepted when it holds >= 99.9 % of the probability of all roster IDs, its probability is
at least e^-12 of the cells' own best reading, and it differs from that reading in at most one cell
(`forms/read_form.py`, `roster_match`). An ID read clearly, with a valid check digit, that is not on the
roster goes to a human, marked "NOT on the class roster": a stranger - or two identical misreads, which the
check digit cannot catch (it happened once in 720 simulated students: one writer's '4', read as '6' in two
cells of the ID).

| IDs accepted automatically (readers v2), wrong ones in brackets | without roster | with roster |
|---|---|---|
| sim_dev (MNIST writers, development set) | 92 % (0) | 100 % (0) |
| sim_test (MNIST writers) | 90 % (0) | 98 % (0) |
| USPS / screen-drawn / CVL handwriting | 76 / 97 / 55 % (0) | 98 / 100 / 94 % (0) |
| Semeion (16x16) | 17 % (**1**) | 73 % (0) |

Without a roster an ID needs a valid check digit, >= 98 % of the valid readings' probability and >= 50 % by
itself (two misread digits can cancel out in the check digit: '2'->'7' with '6'->'1' did once); the one
wrong ID left is a writer's '4' read '6' in two cells, confidently. With the roster: none.
The same readers also learn from the ID what they get wrong in each student's hand: when the roster says
an ID cell holds '1' but it was read '7', half of every '7' probability in that student's answers goes to '1'
(`writer_confusions`; it only lowers confidence). With roster, automatic share / wrong: CVL 55.4 % / 1,
sim_test 68.3 % / 0, USPS 63.7 % / 2 (one reader alone on CVL: 4 errors -> 2).

Stress test on all six sets, with the first and the current readers: the student's own ID removed from the
roster and 30 "near twins" added (valid IDs two digits apart - the check digit makes two the minimum):
nobody was ever matched to a twin.

### Collecting real handwriting: copy sheets

The cell reader has only seen public digit collections (American, Austrian, Italian writers). The
fastest way to teach it how *your* students write: copy sheets (`forms/collect.py`). Two pages with the
exam forms' markers and QR code: 24 printed numbers to copy into 8-cell boxes (every digit, the point and
the minus many times), and Hebrew and English sentences to copy onto lines. Nobody types a label - the
printed text is the label.

```
python -m forms.collect make                  # -> data/collect/sheet_v1..v4.pdf (four versions, other numbers)
                                              #    and sheets_all_versions.pdf (all four, for printing)
python retrain_from_sheets.py scans/*.pdf     # all of the above and the training, in one command (below)
```

`retrain_from_sheets.py` reads the scans (`forms.collect read`: labelled cells -> `data/collect/real/<scan>/`,
each person = his pages in a row), holds out ~20 % of the people, fine-tunes the reader in use on the others
(mixed with synthesized cells so it keeps what it knew; `python download_digits.py` first on a new machine),
measures old and new readers on the held-out people's cells, and only if the new one is better (more cells
right, no more sure-but-wrong) fine-tunes on everyone and installs it as `models/cell-reader-v2/mine`
(`ensemble.json` -> `[mine, b]`; the previous one is kept as `ensemble_before_<time>.json`). `--no-install`
only measures. Report: `results/retrain_report.json`.

Reading checks the copying, not the handwriting: a row is kept only if exactly the first len(number)
cells carry ink (a skipped or shifted cell would mislabel the whole row). Many people's sheets can be
scanned into one PDF. On simulated filled sheets, 93 of 96 rows were kept (744 labelled cells per 4
people). 20-30 people x one sheet = ~4,000-6,000 cells of real handwriting.

How many sheets are worth it? A proxy: the first, MNIST-only reader fine-tuned (1 epoch) with the digits
of N CVL writers - ~105 digits each, about one copy sheet - and tested on CVL's 60 other writers:

| people | digits read right | sure (p >= 0.98) | wrong among the sure |
|---|---|---|---|
| 0 | 89.7 % | 63 % | 1.85 % |
| 5 | 91.6 % | 69 % | 1.26 % |
| 10 | 92.6 % | 72 % | 1.25 % |
| 20 | 93.6 % | 75 % | 0.85 % |
| (v2: 57 CVL writers + 3,600 EMNIST writers, from scratch) | 95.1 % | 77 % | 0.52 % |

Every doubling of the people adds about a point; 20 sheets close most of the gap to a full training on
that style.

The copied Hebrew sentences come back too, in the Hebrew trainer's format
(`data/collect/real/<scan>/hebrew_lines.json`): `python train_hebrew.py --init models/hebrew-crnn-cpu/model.pt
--extra data/collect/real/<scan>/hebrew_lines.json --epochs 2 --lr 3e-4 --out models/hebrew-crnn` teaches the
Hebrew reader your students' hands (`models/hebrew-crnn` is used automatically when it exists).

## Training more

| what | how | expected gain |
|---|---|---|
| more epochs on a GPU | `setup.cmd` (continues from the shipped model), or `python train_hebrew.py --init models/hebrew-crnn-cpu/model.pt --epochs 40` | each 5 CPU epochs so far: 14.8 % -> 10.8 % CER |
| your own exams | `python label_tool.py scans/*.png` -> open `data/my_labels/label.html`, correct the pre-filled readings (least confident first), Save -> `labels.json`; then `python train_hebrew.py --init models/hebrew-crnn-cpu/model.pt --extra data/my_labels/labels.json` | biggest: your students' hands and course vocabulary |
| course vocabulary (no training) | paste lecture notes, old exams or a word list into `models/hebrew-lm/course_words.txt` (shipped with ~1,775 ECE, programming and ML terms) | technical terms are not "corrected" into common look-alike words |
| more Hebrew data | add parquet files (image + text columns) to `data/train/` | |
| the Hebrew/math classifier | `python prepare_mathwriting.py` (3 GB stream, keeps 0.7 GB) then `python train_segment_classifier.py` (ends with calibration) | |
| the formula reader | fine-tune Pix2Text MFR on MathWriting's 230k handwritten formulas (GPU) | varied handwriting: small fractions, D/b, u/y confusions |
| the answer-cell reader | `python download_digits.py` (~450 MB of digit collections), then `python -m forms.cells train ...` (see the copy sheets above for fine-tuning on your students' own cells) | European-style digits: CVL 80 -> 89.5 % of answers read right |
| the Hebrew/English classifier | `python prepare_data.py` (fetches IAM) then `python train_english_classifier.py`; re-tune the decision with `tools_extract_chunks.py` + `tune_english_rules.py` on dev sets | English words cut into short chunks are the hard case |

On a shared server, `EXAM_GRADER_THREADS=4` limits every model to 4 CPU threads per process.

## Other handwriting models - what exists (researched 26.10)

| | best open, local | numbers | what it needs |
|---|---|---|---|
| Hebrew lines | heb-ocr / Mishkefet-v1 (30M CTC + character 6-gram LM, weights CC-BY-NC-SA) - the same kind of model as ours | ivrit.ai benchmark (225 lines, 10 writers, Sept. 2026), median CER: Gemini Flash (cloud) 11.9 %, **Mishkefet-v1 17.5 %**, GPT / Claude 44-90 %; no open VLM listed (Qwen3-VL-2B read our Hebrew lines at 93 % CER); on our test lines heb-ocr 10.3 % vs ours 9.1 % | CPU |
| Hebrew context | Dicta-LM 3.0 (Feb. 2026; 1.7B / 12B / 24B, Apache-2.0; Hebrew-first LLMs) | used to choose among readings, `recog/hebrew_context.py` (below) | 1.7B: CPU possible (slow), GPU better |
| English lines | general VLMs: MiniCPM-V, Qwen-VL (Qwen3-VL-8B, MiniCPM-V-4.5, Apache-2.0) | zero-shot on IAM: MiniCPM-V-2.6 **2.0 % CER**, Qwen2-VL-7B close (arXiv 2503.15195); TrOCR-base here 4.5 %, -small 6.9 % | GPU (8B) |
| handwritten formulas | Uni-MuMER (Qwen-VL fine-tuned on CROHME / HME100K / MathWriting, Apache-2.0; 2B / 3B / 4B) | CROHME 2014 / 16 / 19 exact: **82-84 / 78-81 / 76-80 %**, vs 61 % for the best earlier special model and 56 % for Qwen2.5-VL-72B zero-shot | 2B: CPU possible (slow), GPU |

Measured here (CPU):

* **Uni-MuMER 2B** (`recog/mumer.py`; `tools/mumer_test.py`): on the 70 real handwritten formulas the same as
  Pix2Text MFR in meaning (66 / 70 equivalent each; exact text 47 vs 65 - it writes '( x )' where the labels
  have '\left( x \right)', the style Pix2Text was trained on). On Tal's six math lines it reads what Pix2Text
  misread: '4^{2}+2^{2}=\sqrt{20}' (Pix2Text: y^{2}), '3) A = \pi r^{2} | r = 5' in one go (Pix2Text: \pi^{2},
  and the layout had called '|r=5' Hebrew), 'A = 78.54 \approx'; it fails on the faint small '16+4'. ~9 s per
  formula on 2 CPU cores, well under a second on a GPU. The model was saved by transformers 5; the loader runs
  it on 4.57 (settings rewritten into models/uni-mumer-v4, weights linked).

* **TrOCR-large-handwritten** (`tools/trocr_large_test.py`): 60 IAM test lines CER **3.7 %** (base 4.5 %, small
  6.5 % on the same lines), 10 s per line on 2 CPU cores - but Tal's print hand stays hard for every TrOCR:
  "hi there cladue, how" -> "In there clade , now".

**On a GPU - Tal's RTX 2080 Ti (11 GB)**: double-click **`gpu_eval.cmd`**. It makes `.venv` with CUDA torch if it is
not there, downloads the three GPU readers once (Uni-MuMER 4.3 GB, Dicta-LM 3.0 1.7B 3.4 GB, Qwen3-VL-8B 17 GB - the
last in the background while the first parts run) and runs `tools/gpu_eval.py`: the strong readers against the
ones in use on our test sets, Tal's lines and his whole page; everything it prints goes to `results/gpu_eval.log`,
the numbers to `results/gpu_eval.json`, the page to `results/tal_notebook_gpu_p1_read.png`. By hand:

```
python download_models.py uni-mumer-qwen3vl-2b dictalm-3.0-1.7b-base qwen3-vl-8b
python tools/gpu_eval.py                      # env math hebrew english page (tools/server_eval.py: the old name)
python tools/gpu_eval.py --parts hebrew --hebrew-lm <DictaLM-3.0-12B-Base folder>      # a bigger Hebrew LM
```

Measured on Tal's PC (26.10: RTX 2080 Ti that also drives the screen, Ryzen 9 9900X, 94 GB; the whole run with
installs and 25 GB of downloads: 24 minutes):

| reader | test | in use | strong reader on the GPU | time |
|---|---|---|---|---|
| Hebrew in context (Dicta-LM 3.0 1.7B, float16) | 300 test lines, writers never seen | CER 8.50 % | **7.02 %** | 1.3 s / line |
| | Tal's 7 Hebrew lines | 19.7 % | **17.2 %** ('אני בודק אותך איך אתה קורא' now exact) | |
| formulas: Uni-MuMER 2B (float32) | 70 real handwritten formulas | Pix2Text 66 / 70 equivalent | 65 / 70 (3 of each reader's misses are the same limit, written right by both - '\to' vs '\rightarrow' - that the equivalence check cannot compare) | 2.0 s (Pix2Text 0.14 s) |
| | Tal's 6 math lines | Pix2Text 2 right ('\pi^{2} r^{2}', 'y^{2}', '78.54 \pi') | **3 right** + 'A = 78.54 ≈' (the ≈ is on the page), '1 6 + 1' for the faint '16+4' (Pix2Text: garbage), but '2)' read as '2 7' | |
| English: Qwen3-VL-8B (4-bit) | 150 IAM test lines | TrOCR-small 6.94 %, -base 4.52 % | **3.94 %** | 0.56 s / line |
| | Tal's 3 English lines | "there's sizable , how" ... | "hi there claudia, row" / "are you want to test" / "the text recognition" (it fixes his spelling slip) | |
| **Tal's page end to end** (`tools/gpu_eval.py --parts page`, 70 s) | words / formulas / checks | Hebrew 13 / 29, English 4 / 13, formulas 7 / 10; checks: 2 of 3 (problem 2 "misread?") | Hebrew **16 / 29**, English **7 / 13**, formulas 7 / 10 (Uni-MuMER alone 8 / 10); checks: **3 of 3** as in the transcription - wrong (c = 5, not 9) / right value, wrong equality ('4²+2² = √20': c = √20) / right - each at ~50 %, so to a human | |

The page's remaining errors are layout, not reading: 'hi' was taken for a formula (h^i), 'are you' for Hebrew; and the
Hebrew tongue-twister ('גנן גידל דגן בגן') gets little help from a language model. Problem 2 needed both formula
readers: Pix2Text kept the structure ('2 | a²+b²=c² | y²+2²=√20') but read the 4 as y, Uni-MuMER read the 4 but
glued '2)' into '2 7' - the check now takes a letter that does not belong to the formula as the digit another
reading of the same piece has in its place (a suggestion: at most 50 % sure). On Windows Math-Verify's parse
timeout (a new process per parse) failed under the runner - every formula 'did not parse'; it now runs without
that timeout there (`grader/symbolic.py`).

The page reader takes them with `EXAM_GRADER_HEBREW_CONTEXT=1` (Hebrew in context), `EXAM_GRADER_FORMULA2=mumer`
(a second formula reader whose reading votes in the math validation) and `EXAM_GRADER_ENGLISH_VLM=1` (English
pieces re-read by Qwen3-VL-8B; the layout is still decided with TrOCR). `read_page.py` runs them after the whole
document's layout, each over all its pieces, one model on the GPU at a time (an 11 GB card holds any one of them,
not all three) - `MixedReader.finish()`.

Number formats (`recog/gpu.py`): the RTX 20xx (Turing) has no fast bfloat16, so a 2B reader runs in float32 when it
fits (~8 GB), the language model in float16 for speed, the 8B English reader in 4 bits (bitsandbytes); a reader
that overflows in float16 (NaN) reloads itself in float32. Ampere and newer GPUs: bfloat16. CPU: bfloat16 where the
processor has it (AVX512-BF16 / AMX), else float32 - the university server (Xeon Gold 6130, 8 cores, 62 GB, no
GPU) gets float32 and can run all of this, slowly. `EXAM_GRADER_DTYPE=fp32|fp16|bf16` overrides.

So for Hebrew there is no ready library that reads better than a dedicated recognizer like ours; the gains
are in context (a Hebrew language model) and data. For English and formulas the strong open readers are
VLMs of 2-8 billion parameters - for a GPU (Tal's RTX 2080 Ti). A neutral Hebrew comparison against
heb-ocr needs the gated ivrit.ai benchmark (Hugging Face login and accepting its terms).

## Circuits drawn by hand (started 26.10)

The goal: from a drawn circuit, its netlist (components, the nodes they join, the currents the student named), so
the grader can check his equations against the circuit he drew - `grader/circuit.py` already writes Kirchhoff's
current law at every node with named currents and solves the node voltages symbolically (modified nodal analysis):
on Tal's drawing (`data/my_scans/tal_circuit.pdf`, two sources E1 E2, R1 R2 R3, node A) it gives I1 + I3 = I2 and
V_A = R2 (E1 R3 + E2 R1) / (R1 R2 + R1 R3 + R2 R3).

A general vision-language model is not the way: Qwen3-VL-8B asked for the netlist of Tal's drawing (`circuit_test.cmd`,
`recog/circuit_vlm.py`) describes it about right in words, but 4-bit on the GPU it got 4 of 5 components (R3 missed,
I3 through R2) and in full precision on the CPU 3 of 5 (E2 between A and ground). The reader is built from parts trained on two open collections of hand-drawn circuits - CGHD
(3,293 phone photos, 34 drafters, CC BY-SA 3.0) and Digitize-HCD (1,277 scans, 150+ volunteers, CC BY 4.0) - and
every part is measured on drafters it never saw (CGHD's test drafters 1 5 10 15 20 25 30):

| part | file | trained | measured |
|---|---|---|---|
| detector: the parts, value labels, grounds, crossovers, terminals (32 classes, below) | `recog/circuit_reader.py`, `tools/train_circuit_detector.py` | RT-DETRv2-R18 (Apache-2.0) fine-tuned, `circuit_train.cmd` on the GPU (3 h) | not yet run |
| strokes: pen strokes on any paper (ruled, squared, shadows) | `recog/strokes.py`, `tools/train_strokes.py` | small U-Net on CGHD's 356 hand-made stroke maps, `circuit_train.cmd` (45 min) | not yet run; the classical strokes beside it (below): against the 90 stroke maps of the test drafters, 97 % of the stroke pixels found (within 2 px; 93 % before the rules work below), precision 0.57 (0.55) - the table around a photographed page is still taken for ink |
| wiring: wires -> nodes, each part's two ends, labels, current arrows | `recog/circuit_wiring.py` | rules, no training | with the true boxes and the classical strokes: 48 of 48 netlists exactly right (CGHD drafter 1 against the LTspice schematics of his 12 circuits, `tools/circuit_eval.py --true-boxes`; 26 before the rules work below, 15 before the end scoring and the bridging); the polarity right in all 48; every box moved and resized at random by up to 10 % of its size, as a detector would place it: 144 of 144 (three draws), up to 20 %: 143 of 144; Tal's drawing entirely right; a part's box left out (a part the detector missed): its symbol in the wires noted in 189 of 192, none noted with every box (with the boxes moved by up to 10 / 20 %: none of the 144 readings noted - a part's own zigzag sticking out of a box set a little off, 4 - 7 pen widths from it, is not taken for a missed part) |
| polarity: a DC source's '+', a diode's cathode, a current source's arrow | `recog/polarity.py`, `tools/train_polarity.py` | small CNNs on Digitize-HCD's port locations + CGHD's rotation tags, `circuit_train.cmd` | trained on Digitize-HCD only: diodes 94 %, DC sources 85 % right on CGHD's test drafters (392 / 73 symbols); + one CGHD drafter's 39 sources: 89 % |
| labels: names ('R1', 'I_2', 'A') and values ('10kΩ', '5V') | `recog/labels.py`, `tools/train_labels.py` | small CRNN (CTC) on the labels both collections transcribe (17,700), `circuit_train.cmd` (15 min) | a 40-minute CPU run in the cloud: on drafters never seen, 58 % of all labels and 74 % of names / values exactly right (CGHD's photos 51 %, Digitize-HCD 93 %); read with a beam search weighed by a character n-gram model of the training labels (`lm.json` beside the model, made by the training): 62 % and 76 % (CGHD 56 %, Digitize-HCD 95 %); and each label read knowing the kind of part beside it (the character model of labels by resistors, by capacitors ... - '10µF' by a capacitor, 'R1' by a resistor; `lm.json` carries both): +0.9 points more on the 7,177 unseen labels (57.9 -> 58.8 % counted strictly, spaces aside); on 200 of CGHD's, greedy 58 % against the formula reader's 26 % (names / values 73 % against 36 %) |
| terminal roles: an op-amp's In+ / In- / Out, a BJT's base / collector / emitter and NPN / PNP, a MOSFET's gate / drain / source and N / P | `recog/roles.py`, `tools/train_roles.py` | small CNNs on Digitize-HCD's named ports (16,000 crops), `circuit_train.cmd` (10 min) | a CPU run in the cloud, op-amps only (the transistors not finished): 96.6 % of 500 held-out op-amps (a tenth of the collection, in blocks of 50 drawings) with all three legs right, 97.7 % of the legs |

What a BSc in electrical engineering draws, and where the reader stands:

| course | drawn | the detector's classes | what the student's answers are checked against |
|---|---|---|---|
| circuits 1 | R, sources (DC, dependent), switches, op-amps, terminals | resistor (zigzag and box), voltage_dc, voltage_rail, current_dc, dep_voltage, dep_current, switch, opamp, terminal, meter | node voltages, branch currents, Kirchhoff at a node, nodal / mesh equations (the student's own unknowns too), Thevenin / Norton at a port, maximum power (P_max = V_th^2 / 4 R_th across two open terminals; 'R_L = R_th' right, not sure - another R_L may be the problem's own data), superposition (V_A', I_2'', V_A^{(1)}: one source's share), a part's V / I / P by name, the total P the sources give (a bare 'P' that is one part's power instead: right, not sure - the question may ask for that part's), R_eq / R_in what the source sees (with terminals drawn, theirs too: either right, not sure), ideal op-amp circuits (V_+ / V_- its inputs - a wrong one not sure: they may name the supply rails) |
| circuits 1-2: storage | C, L, switches at t = 0 (two-way too) | capacitor, inductor, switch | tau, R_th the capacitor / inductor sees (R_eq: that or what the source sees), the steady state, x(t) of first-order switchings, any order (RLC) in the Laplace domain, X(s), H(s), Z(s), energy, a filter's corner; x(0^-), x(0^+), x(inf) and dx/dt at 0^+ (any order: the initial and final value theorems on X(s)); H(jw) - as a function of w, at one w, at 0 and infinity - its magnitude, in dB and its angle, Z_in(jw); a second-order circuit's alpha, w_0 / w_n, zeta, w_d and bandwidth (from its natural response, any topology) |
| circuits 2: AC | AC sources (cos / sin labels, phasors '10∠0°'), impedances 'j4Ω', transformers / coupled coils, three-phase drawn with its sources | voltage_ac, current_ac, transformer | phasors (rectangular, polar, time form), Z_in / Z_eq / Z_th, P, Q, S and |S|, the power factor (with 'lagging' / 'leading' after it: the load's character checked too) and the load's angle, a part's reactance X_L / X_C (either sign) and impedance Z_L / Z_C, the sources' w and f, I_rms / V_rms (a source written in time gives amplitudes; one given as a phasor, amplitudes or RMS values), resonance, DC + AC superposed, ideal transformers (1:n), coupled coils (L1, L2, M), line voltages V_ab, two-port z / y / h / g / transmission parameters |
| electronics | diodes (Zener, LED), BJT NPN / PNP, MOSFET N / P, op-amps, small-signal models (dependent sources again) | diode, zener, bjt, fet, opamp + the terminal roles above | diodes ideal / 0.7 V, Zener regulators, BJT bias (active / saturation / cutoff, beta from its label), MOSFET bias (V_t, K from its labels), mid-band amplifiers: g_m, r_pi, A_v (also in dB: 'A_v = 20 dB', '20\log|A_v|'), R_in, R_out; the operating point I_CQ / V_CEQ / I_DQ by name and 'Q = (4.98 V, 1.23 mA)' as a pair |
| digital logic | AND OR NOT NAND NOR XOR, ICs | and, or, not, nand, nor, xor, ic | each named output's Boolean function, by truth table (De Morgan's forms the same) |
| power electronics, machines | switches, thyristors / triacs, transformers, lamps | switch, thyristor, transformer, lamp | the linear parts (thyristors not solved) |
| everywhere | grounds, crossings without a junction, labels | gnd, crossover, text | a drawing without a ground (reference guessed), with names only (the student's own data used, as suggestions) |

Dependent sources are in no open collection: 35 % of the round DC sources of both collections are redrawn as diamonds
in place (the circle erased into the paper around it, a diamond in the drafter's pen colour and width, '+ -' or an
arrow inside; `tools/circuit_data.py`) so the detector learns the shape - real ones (exams, homework) are needed to
measure it. Also absent: a switch's t = 0 label as such, two-port boxes, flip-flops / MUX blocks,
current arrows drawn on the wire itself and voltage marks across a part (both read, but measured only on drawings made for the test). The solver (`grader/circuit.py`) takes resistors,
independent and dependent sources ('2 I_x' by a diamond: twice the current named I_x; '3 V_A': three times node A's
voltage; '0.5 V_x': half the voltage marked V_x with '+' / '-'), ideal op-amps (once their legs are read), capacitors and inductors in the steady state (C open, L short) or
the Laplace domain, switches in a given state, ammeters / voltmeters; it gives the resistance a part sees (Thevenin)
and a first-order circuit's time constant (R C or L / R). The checks know the names exam answers use though the drawing
does not carry them: between a port's two terminals V_th / V_oc, R_th / R_eq / R_ab, I_N / I_sc; with one op-amp V_o,
V_i and A_v. They try every state of the switches: 'tau = 1 ms'
right if it is the time constant in one of them; a steady-state value that fits is right, one that does not is not
called wrong (it may be about another moment, t = 0+). A function of time after a switching ('v_C(t) = 5(1 -
e^{-t/\tau})', 'i_L(t) = \frac{V_s}{R}(1 - e^{-Rt/L})', 'i(t) = 10e^{-1000t} mA', u(t) and 'for t > 0' allowed) is
checked against the first-order response x(inf) + (x(0+) - x(inf)) e^(-t/tau) of every switching the drawing allows
(each switch flipped, two at most together): the capacitor's voltage / the inductor's current carried over from the
steady state before (a capacitor cut off from everything before then is taken uncharged - such a verdict is a
suggestion); x(0+) from the circuit with the capacitor held as a source of that voltage (the inductor as a source of
that current), tau and x(inf) from the circuit after. v_C, i_L, v_R, i_C with one part of that kind, v_R1 / i_R1 by
name, node voltages (V_A), the currents drawn, i(t) of a single loop; a part's own voltage / current in either sign
(its direction is not drawn). Right when it is that function for some switching; wrong when neither its final value
nor its decay fits any; when only those fit, not its start, it may come from an earlier switching or a starting
charge the problem gives - a suggestion, not a verdict. A switch is read as drawn (the state before t = 0): closed
when ink runs unbroken through its box from one end to the other, else open; a two-way switch (three wires: the
common end alone on its side, two throws) at the throw its lever reaches. On 48 CGHD switches the drawn state looked
right by eye in most (`open` 27, `closed` 5, unknown 8, two-way 5). A right answer is sure when exactly one switching
gives it and agrees with the drawn state (or none was read). Circuits of second order and up (an RLC, two
capacitors): the same switchings in the Laplace domain - each capacitor sC beside a source of C v(0-), each inductor
1/(sL) beside one of i(0-)/s, the DC sources steps V/s - and the answer x(t) compared with the inverse transform
(poles and residues: '10 - 11.55 e^(-50t) sin(86.6t + 60°)' for a series RLC), its transform X(s) ('V_C(s) =
\frac{10^5}{s(s^2+100s+10^4)}') with the circuit's at real s over three decades. H(s) / T(s) / G(s) - the output (a
node named o / out / Vo, else the port's terminals, else the op-amp's output) over the one source - and Z(s) / Z_in(s)
(what the source sees), Z_th(s) between the port's terminals: checked though the drawing never names them. Diodes:
each on (a source of 0 V ideal, or 0.7 V) or off (open), the one consistent way (an on diode carries current anode to
cathode, an off one has less than the drop across it) - both models tried, since the problem's text says which: an
answer one of them gives is right but not sure, one that neither gives is wrong; I_D / V_D with one diode, and the parts' own V / I / P, I of a single loop and the total P with each diode as it turned out. A Zener
(its own class: CGHD's and Digitize-HCD's Zeners) also in breakdown - held at V_Z from its label '5.1V' / 'V_Z = 5.1V',
its current cathode to anode - so a regulator's I_Z, its load's current and when the Zener drops out are checked. Values
with prefixes inside a formula ('\frac{10 - 0.7}{2k}', '5 mA + 2 mA') are read in base units. A part's own voltage,
current and power by its name - V_{R1}, I_{R1}, P_{R1}, I_s / P_s of the one source, V_R / I_R / P_R with one
resistor, I of a single loop - in any choice of reference directions (none is drawn; 'V_{R2} - V_{R1} = 10' fits when
some choice makes it hold); AC power as half the real part of V I* (amplitudes) or all of it (RMS values) - the course
decides, so one of them right is not sure - and the reactive Q. One L and one C: w_0 = 1/sqrt(LC), f_0; 'Q' there may be the quality factor (w_n over the bandwidth) as well as the reactive power - either is right, not sure, neither wrong; with its unit ('VAR') it is the reactive power. One C or L:
a first-order filter's corner omega_c = 1/tau, f_c. The energy a capacitor / an inductor holds in the steady state
(W_C = C v^2/2, W_L = L i^2/2), a capacitor's charge (Q_C = C v; Q with one capacitor in DC; units mC, µC, nC). Units Hz, J, rad/s, VA, VAR, S besides V A W Ω F H s. An op-amp fed back to In+ only
(a Schmitt trigger) or not at all (a comparator) is not solved as a linear one. A load whose
value the drawing leaves open ('R_L'): 'R_L = 1 k' right when it is the resistance R_L sees (the most power).
DC and AC sources together (or AC ones at several frequencies): each group alone - the DC ones with C open and L
short, each frequency's by phasors - added up, and 'v_C(t) = 5 + 7.07 cos(1000t - 45°)' checked in time. A drawing
that names its parts without values (E, R_1, R_2) cannot judge an answer in numbers - they came from the problem's
text ('V_A = 5', or 'E/2' with R_1 = R_2 given): 'cannot check', with what the circuit gives in its names for the
human, never 'wrong'; an answer in the names, or a relation between the circuit's quantities with no numbers in it
('V_A = E - R_1 I_1', a nodal equation), is judged. The equations come as the formula reader writes them - spaced
tokens and digits ('V _ { A } = 2 . 5'), '\!' / '\,' / '~' spaces, '\mathrm { c o s }', a unit after a phasor
('3.61 \angle 33.7 ^ { \circ } \Omega'), names spaced inside braces ('v _ { C } ( t )'); circuit_check.tidy evens them
out first (measured on 30 typical answers rendered in two fonts and read by the formula reader: all 60 readings
parse to the same thing as the answer written cleanly, apart from what the circuit itself decides - 'RC' as R times C; and 50 answers of the newer kinds - x(0^+), dx/dt at 0^+, H(jw) and its magnitude / dB / angle, alpha / zeta / w_d / BW, P / Q / S / pf, R_eq - checked against their circuits: all 100 readings get the clean answer's verdict, 85 before the reader's own habits were evened out too: a name spaced into letters ('B W', 'p f'), a spaced unit ('V A', 'd B'), a prime written as a superscript, '\mathrm { = }', '\mid' for a bar, a subscript spaced inside |Z_{in}(j1000)|, a stray '\sb'); 20 more of the latest kinds (Q, a charge, a bare P, 'pf = 0.6 (leading)') all 20. A decimal comma ('V_A = 7,5') is read as a decimal point, not as two statements ('V_A = 7' would be wrong); '1,000' stays one number - in the derivation checks (`grader/selfcheck.py`) too, where the reader's spaced '1 , 0 0 0' used to be cut in two. A unit in square brackets, as Israeli exams write units ('3.33 [V]', '2 [mA]', '[k\Omega]', '\left[ V \right]') is read as that unit - a number's prefix counts, after an answer in the parts' names it is dropped (before, such an answer got no verdict, here and in the derivation checks); '\simeq' / '\cong' count as '=', '\div' as '/', a repeating decimal '3.\overline{3}' as 3.333... (the derivation checks used to read it as 3.0), a numeric fraction with its unit after it ('\frac{10}{3}V') as that number (it was judged wrong, as ten thirds times V). A chain of equalities ('V_A = \frac{E R_2}{R_1 + R_2} = 3.33') is checked one equality at a time - each later side against the quantity named first, or each side against the next ('I_1 + I_2 = I_3 = 5 mA') - and is wrong when one of them is (which one said): before, only its last two sides were checked, so 'V_A = 3.33 = \frac{10}{3}' got no verdict and 'V_A = \frac{10}{3} = 3.66' none either. When the student copies the problem's data first ('E = 12 V',
'R_1 = 2k\Omega', '\beta = 100' - a part the drawing only names, not a load R_L that may be what is asked), the rest of
the work is checked with those values: suggestions then, not verdicts. Ideal
transformers: the coils from where the four wires meet the box (side by side: the primary the left one; a coil's dot
read when drawn - a small filled blob, touching nothing, clearly by one end of each coil: that end is the coil's '+' (`circuit_wiring._coil_dots`; on CGHD's photos at hand most transformers' four wires were not found, so it is measured on one photo and a drawing made for the test); else taken at its top, the netlist carrying that problem and its verdicts staying suggestions), the
turns ratio from a label '1:2' / 'N_1:N_2' beside it; v_S = n v_P, i_P = -n i_S, the far side measured from its own
S-. Coupled coils (the same symbol, labelled 'L_1 = 2H' / 'L_2 = 8H' / 'M = 2H' or 'k = 0.5', or a bare '2H' on a
coil's side): v_P = jw (L1 i_P + M i_S), v_S = jw (M i_P + L2 i_S) (s in place of jw in the Laplace domain; two
shorts in DC). V_{AB} between any two named nodes (a line voltage of a three-phase circuit drawn with its sources
'120∠0°', '120∠-120°', '120∠120°'). Between a port's two open terminals the '+' is the one the drawing marks '+' (or names a, of a / b); with neither, V_th / V_oc and I_N are right in either sign (the student's choice of which is '+'). Four open terminals are a two-port (port 1 the left pair, port 2 the right, each's upper terminal its '+'):
z11 ... z22 (1 A pushed into one port, the other open, the sources set to zero) and from them y, h, g and the
transmission parameters t11 = A ... t22 = D. A drawing without a ground symbol: the reference is guessed (where most
sources' '-' ends meet) - answers about a node's voltage then stay suggestions; currents, a part's voltage, powers,
tau and the rest do not depend on it. Mesh and nodal analysis in the student's own unknowns (mesh currents I_1, I_2 or J_1, J_2
drawn as circular arrows the reader does not take; V_1 of a node the drawing does not label): the equations that share
them are solved together - right when every unknown comes out as one of the circuit's currents / voltages (a mesh
current is an outer branch's, either way round), a suggestion of wrong when none does (an inner mesh's current is no
branch's, so a partial match is not judged). Bipolar transistors (NPN / PNP, legs read by the roles model): V_BE
0.7 V and the region that fits - active (I_C = beta I_B, beta from a label 'β=100' / 'h_FE = 150' by it), saturation
(V_CE 0.2 V, I_C <= beta I_B) or cutoff; I_B, I_C, I_E, V_BE, V_CE, V_CB by name, beta as a name, and what the base sees
with the transistor out (V_TH / V_BB, R_TH / R_BB) - the first step of a bias problem. beta not written: the active
region in the symbol beta, so answers in numbers are 'cannot check'. An amplifier (an AC signal source): after the
bias, each transistor its hybrid-pi model at mid-band - g_m = I_C / V_T, r_pi = beta / g_m - the capacitors shorts, the
DC sources set to zero; A_v (output: a node named o / out, else R_L's, else the rightmost terminal; over the input after
a source resistance, A_vs over the source), R_in, R_out (the load off), g_m, r_pi, V_T as names. V_T is 25 or 26 mV as
the course takes it: an answer one of them gives is right, not sure. 'R_C \| R_L' / '\parallel' / '||' read as in
parallel. A numeric answer within 5 % of the circuit's value is taken as rounded on the way - right, not sure.
MOSFETs (N / P channel, legs read) with V_t and K written by them ('V_t = 1V', 'K = 0.5mA/V^2'; k_n or k' means twice
K): the region that fits - saturation (I_D = K (V_GS - V_t)^2, the root with V_ov >= 0 and V_DS >= V_ov), triode or
cutoff; I_D, V_GS, V_DS, V_ov by name, K and V_t as names ('2K V_{ov}' is that K, not 2 kilo); at mid-band g_m = 2 K
V_ov (the gate open) for A_v, R_in, R_out as above. Without K and V_t written by it a MOSFET is not solved. Sinusoidal circuits (an AC source labelled '10cos(1000t+30°)'): phasors at the sources'
angular frequency (jwC, 1/(jwL)); answers in rectangular ('5 - j5'), polar ('7.07∠-45°'; an angle with no degree mark tried in radians too, then not sure) or time form ('v_A(t) =
7.07 cos(1000t - 45°)') are checked against them. A drawing in the phasor domain - the source labelled '10∠0°' /
'120∠-120° V' / '5+j5', parts labelled with their impedances ('3+j4Ω', 'j4Ω', '-j2Ω'; the wiring gives such a label to
the nearest part, a 'V' one to the nearest source) - is solved without any w; Z_in / Z_eq (what the source sees) and
Z_th between a port's terminals are checked too. Logic circuits: each gate's output is the end alone on its side, opposite its inputs (a
NOT's, the narrow end); the gates give each named output's Boolean function, and 'F = A B + C\'' is checked by truth
table against it - De Morgan's forms of the same function are the same answer.

How the wiring decides: the strokes left once the boxes are cut out are wire; pieces that nearly touch (a pen lifted)
are one node; a part's ends are the wire pieces that leave its box squarely and run away from it (a current arrow
or a label drawn beside a part runs along it, and is not taken for a wire); a node only one part reaches is joined
to the nearest node within 8 pen widths; two wire ends that stop short of everything and face each other along one
line, with no part or other wire between, are one wire (a stretch lost on a notebook rule, a long pen lift - up to
60 pen widths). A lone letter nearer a wire than any part names that node ('A'); 'I1' goes
with the current arrow nearest it; an arrow is a long thin piece no part's end reaches, beside a two-terminal part
and along it, with a head at one end - the current runs through that part towards the head. An arrowhead drawn on the wire itself (a chevron: two short strokes slanting back from a point of the wire, one each side; a filled head too, when it is over three pen widths long) gives the current of the part that wire leads to, the way the head points; when the wire branches ahead of it (a node), of the part it comes from, out of it (none found on drafter 1's 48 photos or Tal's drawing, where none is drawn).

Voltages marked on the drawing (`circuit_wiring._voltage_marks`): a '+' and a '-' beside a part's two ends - or by two
open terminals, or by the ends of two different parts (across a chain) - with a voltage's name near them ('V_x', 'v_o',
'V_{R1}'; '+ V_x -' read as one label too) are that voltage, the '+' node's minus the '-' node's (`Netlist.voltages`,
listed by `read_page.py`): 'V_x = 4' is checked with the sign the marks give - in the steady state, after a switching
(v_x(t), v_x(0^+)), in the Laplace domain - where a part's own V_{R1} without marks is right in either sign. Marks
inside a part's box are its own (a source's '+', an op-amp's inputs) and are left alone; a '+' / '-' drawn beside a
voltage source, outside its circle, gives its polarity (an AC one's too: its phasor's reference). An AC source drawn
as a circle with a sine in it has no polarity: its answers are checked both ways round it (several meeting at one
node, a wye: all with '-' there, or all with '+') - right either way, but then not sure; a power, a magnitude or an
impedance, the same both ways, stays sure. (Before this, an AC source read off a drawing had no '+' and no AC
circuit drawn could be solved at all.)

A part the detector misses leaves its symbol in the wires, and the wiring would join its two nodes (a zigzag, a
source's circle) or lose the part (a capacitor's plates) - with the netlist looking sound: on drafter 1, with one
part's box left out at a time, 160 of the 192 netlists had no problem at all. The wiring now looks for a part's
symbol in the circuit's wires (`circuit_wiring._symbol_ink`): a closed loop of paper the size of a part's, wide
inside, with two wires running far from it - its leads (a diode's bar or a lead ending at the next part may add a
short third; a junction dot scribbled round is smaller, a junction drawn as a ring or a cell of squared paper has
three or four wires running far, a wire drawn twice leaves only a sliver between its strokes), a
stroke that turns over and over (more than 220 degrees within 10 pen widths, or three sharp turns: a zigzag, a coil -
not a small ring), two parallel strokes side by side ending facing nothing (a capacitor's plates; a pen lifted at a
corner faces the other wire). What it finds is a note on the netlist, one of its problems - so no verdict resting on
that reading is sure: on drafter 1 none on the 48 photos with every box, 189 of the 192 left-out parts found (every
resistor, coil, diode and source - a label's box over a source's circle hides an arc of it, the ink there is put back
for this; capacitors 37 of 40); on Tal's drawing none. On 50 photos of other
drafters notes come on 20 with every part boxed: on 7 the strokes round the circuit took in a table, squared paper or
a shadow (10 - 44 % of the circuit's box is 'wire', at most 6.5 % on drafter 1; above 12 % one note says the strokes
are too dense to tell the wires), on the others a stain, a junction drawn as a ring, an arrow drawn on a wire, a
label touching a wire: readings that deserve the doubt - and with a part left out, 408 of 734 are found (measured with at most two wires leaving a loop; the guard does not look for transistors' or ICs' shapes).
When the wiring notes such a spot, the reader (`circuit_reader.recover_missed`) looks at what the detector saw there
below its threshold (down to 0.15) and takes the best guess over the spot when the reading has fewer problems with it.
`read_page.py` lists the notes under the circuit ('to check: ...', in the page's pixels) and rings the spots in red on
its picture of the page. A reading with problems is also tried with the classical strokes at a darker threshold (170,
then 140; `circuit_reader.retry_darker`) and that one kept when it is sounder - a table's grain, a shadow or a stain
taken for ink drops out: of 45 other drafters' photos with problems (true boxes) 13 came out sounder (27 problems to
6, 40 to 1, 7 to 0 ...), none worse; a note says so, since a wire drawn as pale as the grain would go with it.

The classical strokes (`recog/circuit_wiring.strokes`): the photo's light is evened out in each colour channel.
Where the pen's colour differs from the notebook rules' (a grey pen on blue rules, a red or a teal pen), the rules'
colour is taken out: each pixel's darkness, a vector over the three channels, is measured along the pen's colour
less its part in the rules' colour - the rules all but vanish and the pen stays where it crosses them (the rules: the
thin lines reaching the picture's side; the pen: the thin strokes off them; a pen greyed by a shadow is kept by its
shade; when the pen would pale - blue on blue - this is not done). The rules left are found at any slant up to 8°
(`recog/page.rule_lines`), and told from long wires by shade - paler than the pen, along a rule reaching the side -
or, when the pen is as pale, by place: a ruled page's evenly spaced lines reaching its side and covering it (a bus of
long wires reaching the side covers only part of it); squared paper's columns the same way, turned. On a rule only the ink clearly darker than that rule's own
shade stays, so a wire drawn right along a rule stays; then faint thin stretches joined to the ink (a pale pen
pressed lightly). On drafter 1 this took the exact netlists from 26 to 48 of 48: rules photographed at a slant
crossed wires and joined their nodes, and pale coloured pens were half lost.

On Tal's drawing (boxes set by hand, classical strokes) the netlist is entirely right: the five parts and their
nodes, both sources' polarity, A, I1 I2 I3 each through its resistor in its direction (I3's arrow, a large head on a
short line, is read along R3), Kirchhoff at A: I1 + I3 = I2 - though the pen is a pale blue on blue rules, and the
bottom wire and both arrows' shafts are drawn along rules (`tests/test_grader.py`, `test_tal_drawing_wired_from_hand_set_boxes`).

On an exam page (`read_page.py`, `read_page.cmd`, once the detector is trained) the circuit detector looks at the
whole page first: two or more parts with the strokes that join them and the labels beside them are a circuit drawing
- read as one (pieces of a broken drawing grouped), its netlist listed under the page, its area kept out of the
text lines. The student's equations against the circuit he drew (`grader/circuit_check.py`, `read_page.py --check`): Kirchhoff's current law at every node
where all currents are named, as the arrows give it; and any equation in the parts' names or values, the node
voltages and the named currents - a node voltage, Ohm's law on a branch, a loop, a nodal equation - against the
circuit's own solution. On Tal's circuit: 'I1 + I3 = I2' right, 'I1 = I2 + I3' wrong (the arrows give I1 + I3 =
I2), his V_A right, a V_A with a term missing wrong (the circuit's V_A shown), a loop equation with a sign slip
wrong. Subscripted names are read as names ('I_1', 'E_1' - the general LaTeX parser makes I the imaginary unit and
E Euler's number). A circuit whose reading has problems (an unread polarity, a loose wire) gives suggestions only.
A drawing that only names its parts (the values in the problem's text): `read_page.py --check --values data.json`, the problem's data given once by the parts' names (`{"R_1": "2k", "E": "10", "V_s": "10cos(1000t)", "beta": "100"}`) - the numeric answers are then judged, sure; without it, the data the student copies are used, as suggestions. Two or more circuits on the pages (the problem's and an equivalent the student drew, or two questions' drawings): an equation right against one of them is right, and a 'wrong' is sure only when every circuit could judge it. A prefixed value right after j ('\frac{V_o}{-j1k}', 'Z_C = -j1k') is j times kilo-ohms - it was taken for letters (no verdict). Names that read like other names (found by random networks with tricky names): a node labelled 1 beside an unlabelled source whose own id is V1 - V_1 was taken for the source's value (a false 'wrong'); the id is no name on the drawing now; two nodes labelled A and a (compared without case, one name) give no sure verdict. An equation in the drawn currents alone that is no node's KCL ('-12 + 2k I_1 + 4k I_2 = 0', a loop) is checked against the circuit's solution - it got no verdict at all before; when it does not hold it is a suggestion, not a sure 'wrong' (the student's I_1, I_2 may be mesh currents of his own, named like the drawing's arrows). On the page's picture `read_page.py --check` draws each verdict beside the line it is about - a green tick (right, sure), a red cross (wrong, sure - since 27.10 a false step too, shown as '?' before), an orange '?' (a suggestion: a human should look) - with its number in the text's list of checks.

Still missing: dependent sources - not in the open collections; they need examples of our own (old exams / exercises), as do
the current arrows drawn on the wire itself and the voltage marks (read since 27.10, measured on drawings made for the test only:
CGHD's 26 photos at hand with '+' / '-' labels mark voltages the European way, with an arrow and U, or mark a
microphone's or a battery's terminals - no voltage is taken from any of them).

The published system (Sensors 2026, "Vision-Based Topology-Consistent Structural Parsing of Hand-Drawn Circuit
Diagrams": 95 % of 1,317 photos fully right) is not used: its weights are Python pickles, which run code when
loaded, and its code is "for research purposes".

## Tried and not adopted (26-27.10)

| idea | result |
|---|---|
| a third cell reader trained with random turns / shears (`--augment`) | worse: Semeion 80 % of digits vs 90 % |
| two zoomed views (90 / 110 %) added to the cell reader's look (`EXAM_GRADER_CELL_ZOOM=1`) | no gain where it matters: CVL sure-but-wrong 0.39 -> 0.46 % |
| the review rules re-tuned for the new readers (e.g. the border-crossing threshold 6 -> 15 px of ink) | +0.1-0.4 % graded automatically, no fewer errors - kept as they were |
| English lines read at three widths and pooled (`tools/english_tta.py`) | CER 6.4 -> 6.1 % on 60 IAM lines at 3x the time - small |
| Hebrew multi-width reading inside the mixed reader's layout decisions | worse layout (82 -> 80 %); used only for the text, after the layout is decided |
| heb-ocr as a second opinion for the Hebrew lines our reader is least sure of | no: chosen on held-out training writers, every threshold made it worse (8.0 % alone, >= 8.2 % with heb-ocr for the least sure 5-50 %), so the test lines stay at 9.0 %; the 8.3 % one could read off the test lines themselves was chance (`tools/hebrew_second_opinion.py`) |
| circuits: the strokes outside the photographed page blanked (the paper found by its brightness and grey) | stroke precision 0.55 -> 0.68 on the test drafters, but the paper finder cuts shaded paper away: recall 0.97 -> 0.90, three of drafter 1's photos lost half their drawing |
| circuits: rules removed whenever they are paler than the pen and span most of the page (not only when reaching the side) | a little more precision (0.57) but a pale pen drawn right on such a rule went with it: Tal's I3 arrow read backwards |
| circuits: the stroke threshold chosen per photo in one colour channel (the one where rules and pen differ most) | 41 of 48; taking the rules' colour out of the three channels together does better (48) |
| the missed-part guard counting any stroke that ends facing nothing (not only a capacitor's parallel plates) | also quiet on drafter 1's 48 photos, 185 of 192 found - but on other drafters' photos a long wire's pen lifts and gaps set it off far more often; a junction dot's pores as closed loops: Tal's drawing set it off |
| circuits: the missed-part guard looking only within the drawing's extent (the parts' boxes + one part's size) - the table, a paper's edge out of it | other drafters' photos with every part boxed: notes on 20 -> 18 of 50; but on drafter 1 only 124 of 192 left-out parts found (189 before): a left-out source at the edge of a small circuit lies outside the others |
| circuits: the guard ignoring what it finds inside a label's box (the loops of '0', 'o', '4' in a label touching a wire) | notes on 20 -> 19 of 50 photos, far fewer per photo - but 390 instead of 404 of 753 left-out parts found on other drafters (a resistor's zigzag beside its label): a part missed silently is worse than a doubt |
| cyttic's DiffusionPen synthetic Hebrew lines (33k, CC BY 4.0) as training data | not used yet: the generator is "diffusionpen-matan" and nothing says which writers it learnt from - probably the collection our test lines come from, which would inflate the test result; usable once tested on other writers (e.g. the ivrit.ai benchmark) |

## Files

| path | what |
|---|---|
| `grader/` | rubric, numeric (units/tolerance), symbolic (SymPy), aggregation + routing |
| `recog/formula.py`, `recog/beam.py` | formula recognizer, ONNX-only beam search with exact log-probs |
| `recog/hebrew.py`, `recog/hebrew_text.py` | Hebrew recognizer, CTC prefix beam search + word-model fusion, RTL visual↔logical order |
| `models/hebrew-lm/` | word frequencies (`words.tsv`), settings (`meta.json`), course vocabulary (`course_words.txt` - edit freely) |
| `build_hebrew_lm.py`, `tune_hebrew_lm.py` | rebuild the word list from `wordfreq` (`pip install wordfreq`); tune the word-model weights on held-out training writers |
| `recog/hebrew_vlm.py` | experimental: VLM recognizer (bigger models / LoRA on a GPU) |
| `train_hebrew.py` | trains + exports the Hebrew recognizer |
| `eval_real.py`, `eval_hebrew.py` | end-to-end evaluations → `results/` |
| `download_models.py`, `prepare_data.py` | one-time downloads into `models/`, `data/` |
| `recog/segment.py`, `grader/mixed.py` | mixed answers: layout, Hebrew/math classifier, grading |
| `read_page.py`, `recog/page.py` | whole scanned pages and phone photos: flattening, cleaning, line finding, figures |
| `forms/`, `grade_form.py`, `models/cell-reader/` | answer-box exams: the course, printing, simulated students, cell reader, reading + grading a scanned form |
| `forms/collect.py`, `data/collect/` | handwriting copy sheets: print, read the filled scans into labelled cells and lines |
| `tools/compare_readers.py` | cell readers compared end to end on the simulated classes, and cell by cell on digit collections |
| `download_digits.py` | the digit collections for training / testing the cell reader -> `data/digits` |
| `retrain_from_sheets.py` | scanned copy sheets -> fine-tuned cell reader, checked on held-out people, installed only if better |
| `tools/mcq_real_test.py` | real scanned exams (Mendeley set): multiple-choice lists read with / without the answer-format grammar |
| `tools_eval_hebocr.py` | heb-ocr (Mishkefet-v1) on our Hebrew test lines, loaded safely |
| `grade_page.py`, `grader/page.py` | grade a page against a rubric (arithmetic items found by their left side; open items to a human) |
| `grader/selfcheck.py` | math validation, no key: formula + data -> result, every '=' step, units, equations, weighed over the readings (`read_page.py --check`) |
| `grader/boolean.py` | Boolean algebra for the math validation: a side read into a logic expression ('A\bar{B} + \bar{A}B', "A' + B'", 'A \oplus B'), two sides compared by their truth tables |
| `recog/hebrew_context.py`, `tools/hebrew_context_eval.py` | Hebrew read in context (CTC score of any candidate x Dicta-LM 3.0); its evaluation |
| `recog/mumer.py`, `tools/mumer_test.py` | Uni-MuMER, a second handwritten-formula reader (Qwen3-VL-2B fine-tuned); its test |
| `gpu_eval.cmd`, `tools/gpu_eval.py` | the strong readers (Qwen3-VL-8B, Uni-MuMER, Hebrew in context) against the ones in use, and Tal's page, on a GPU (`results/gpu_eval.log`, `.json`) |
| `recog/gpu.py` | device and number format per machine (Turing: float32 / float16, newer GPUs: bfloat16), parking models between readers |
| `recog/prompts.py` | the prompts of the vision-language readers |
| `grader/circuit.py`, `recog/circuit_vlm.py`, `tools/circuit_test.py` | circuits: netlist, KCL, nodal solution, comparison with a transcription; the VLM reader and its test |
| `tools/circuit_data.py`, `tools/train_circuit_detector.py`, `circuit_train.cmd` | the hand-drawn circuit collections -> one detection set; the detector's training and measurement (+ strokes, polarity) |
| `recog/circuit_reader.py`, `recog/circuit_wiring.py`, `recog/strokes.py`, `recog/polarity.py` | a drawn circuit read end to end: detector, strokes, wiring (nodes, labels, current arrows, a missed part's symbol noted), polarity; a missed part recovered from the detector's guesses below its threshold |
| `tools/train_strokes.py`, `tools/train_polarity.py`, `tools/ltspice.py`, `tools/circuit_eval.py` | their training; LTspice schematics -> netlists; whole netlists measured against drafter 1's schematics |
| `grader/circuit_check.py` | the student's equations (Kirchhoff, node voltages, loops) checked against the circuit he drew |
| `tools/check_fuzz.py` | regression check, no models (8 min on a CPU): 298 correct circuit answers written many ways (DC formats - a resistor's current as 'V / R k\Omega' and its power in bare numbers, '2^2 \cdot 4 = 16 mW', too, and 16 wrong ones that must be caught -, AC phasors, a switching's responses), ~190 on random ladder / bridge networks with tricky names (internal node 'a' beside label 'A', an unlabelled source V1 beside node 1), ~130 more on random networks with a current source or a labelled source, resistors labelled out of order and currents drawn (and ~50 wrong answers there that must be caught), ~60 on random AC ladders (R, L, C, nodes named c / l beside C1 / L1; rectangular, polar, magnitudes, and ~25 wrong ones), 123 about magnitudes / H(jw) / dB / a marked voltage / loop and node equations (DC and phasor) (right and wrong ones, bars and units written several ways), and 369 correct derivations (clean and as the reader spaces them; calculus, Fourier coefficients, derivatives with constants, phasors, boxed answers, degrees and dB, rounded values, powers of ten, binary, letters in both cases, mixed numbers, matrices, bare prefixes, tiny values, exponentials, sequences and steps) - none may be judged wrong - and 132 wrong derivations that must be caught, 4 with a step's sides in units of different kinds that must be noted, 50 random phasor computations (Ohm's law, series / parallel impedances, a divider, |I|^2 R, a mesh system; polar or rectangular, with or without units) and a wrong version of each, the 26 model solutions of the course's exams (typed as plain text) with a wrong version of each, 400 pages of random LaTeX-ish garbage (matrices too) that must neither crash the checks nor take 20 s, 60 random computations with their units in every position (a fraction's numerator or denominator, products, \mathrm, brackets) and a wrong version of each, 60 random quotients with the unit after the divisor's number only ('12 / 4 k\Omega', '1 / 2 ms', '3 \cdot 10^{8} / 100 MHz') and a wrong version of each (the result written bare 1000 apart never a sure wrong), 40 random two-mesh systems (in kΩ, in ohms or with their units; answered in mA, in amperes or bare) and a wrong version of each, 400 random EE computations (12 templates, many renderings) and 200 random problems of a formula, its data and the answer (units every way), each with a wrong version, 300 random phasor computations with their units every way (unbracketed '3 + j4 \Omega', on each part, a reactance alone, kΩ and mA, VA and kVA) and a wrong version of each, 200 random semiconductor computations in centimetres (densities in cm^{-3}, D in cm^2/s, lengths in cm or µm, resistivities in \Omega \cdot cm, units spelt every way) and a wrong version of each, 200 random problems with angles ('P = S \cos\phi', 'Q = S \sin\phi', 'P = V I \cos\theta', 'X = |Z| \sin\phi', '\phi = \arctan\frac{X}{R}' - as a formula with prefixed data, the angle in degrees marked, bare or in radians, or as a chain, the answer in a prefixed unit) and a wrong version of each, 150 random pages of work over several lines whose values flow on by name, prefixed and rounded on the way (none may be judged wrong; with one slip, only that line), 150 random formula problems with a value worked out on its own line (a series / parallel combination, Ohm's law) and 150 with two formulas, the second defining a letter of the first, each with a wrong answer, 80 random angles in degrees without the mark, decibels and phasor angles (\cos(60), \arctan values, A_v = 100 = 40 dB, e^{j45}, \angle 0.927, a power factor angle given bare) with a wrong version of each, and the 140 grades on the 70 real handwritten formulas from their saved readings; exit status 1 on any failure |
| `tools/replay_formula_eval.py` | the real-formula evaluation (`eval_real.py`) again from the readings it saved (`data/handwritten_formulas/pix2text_readings.json`) - no recognizer, seconds: a parser or grader change may move none of the 140 grading decisions the wrong way (134 automatic, 0 wrong); part of `tools/check_fuzz.py` too |
| `recog/labels.py`, `tools/train_labels.py` | the labels of a circuit drawing (names, values) read by a CRNN, its training |
| `recog/roles.py`, `tools/train_roles.py` | which leg of an op-amp / transistor is which (and NPN / PNP, N / P), its training |
| `read_page.cmd` | drag a scan onto it: read with every reader installed (+ the derivation checks), the result opens |
| `tools/trocr_large_test.py` | TrOCR-large on IAM lines and Tal's English lines |
| `data/my_scans/tal_notebook_lines/` | Tal's page cut into its 16 lines (Hebrew, English, math) with their transcriptions |
| `tools/score_page.py`, `data/my_scans/` | a real page scored against its hand transcription (words per language, formulas) |
| `demo_mixed.py`, `eval_mixed.py` | mixed-answer demo and evaluation |
| `recog/english.py`, `eval_english.py` | English handwriting reader (TrOCR-small, ONNX), its evaluation on IAM |
| `models/english-classifier/`, `train_english_classifier.py` | Hebrew-vs-English chunk classifier and its training |
| `tools_extract_chunks.py`, `tune_english_rules.py` | dump per-chunk signals of a dev set; tune the Hebrew / English / math decision on them |
| `train_segment_classifier.py`, `prepare_mathwriting.py` | (re)train the Hebrew/math classifier |
| `demo.py`, `tests/` | simulated-OCR demo, unit tests |

## Data sources

- `linxy/LaTeX_OCR` (human_handwrite) — handwritten formulas + LaTeX
- `cyttic/trocr-hebrew-matan` — Hebrew lines with writer IDs (train 437 writers / test 109 other writers)
- `cyttic/trocr-hebrew-human` — Hebrew lines on scanned forms
- MathWriting (Google) — handwritten formulas + symbols, CC BY-NC-SA 4.0 (non-commercial)
- Hebrew word frequencies: `wordfreq` (rspeer/wordfreq) Hebrew "large" list, data CC BY-SA 4.0; its
  English list decides whether a Latin-script reading is made of English words
- English handwriting: IAM lines via `Teklia/IAM-line` (tagged MIT on Hugging Face, but the IAM database
  itself is distributed by the University of Bern for non-commercial research); the English reader is
  `microsoft/trocr-small-handwritten` (trained on IAM) in the ONNX export `Xenova/trocr-small-handwritten` -
  its model card states no license

- Digits for the cell reader: MNIST; EMNIST digits (NIST SD19, 3,600 writers) via `ernestchu/emnist-digits`;
  HWD-V1 via `leobottaro/handwritten-digits-pack`; CVL single digits (TU Wien, ICDAR 2013; zenodo 1492173,
  CC BY-NC 4.0 - non-commercial research; its validation writers are used only for testing, as the
  database asks). Test-only collections: USPS (`flwrlabs/usps`), Semeion (UCI), screen-drawn digits
  (`nguyenminh4099/handwritten-digits`)
- Real scanned exams for testing: "A Dataset of Digitized Student Examination Papers, Answer Keys, and
  Manual Evaluations" (Mendeley Data `sf3kvjwknt`, CC BY 4.0) - 50 handwritten Data Science exams with the
  teacher's item-level marks (only the anonymized table is used)

The two Hebrew datasets and the TrOCR model state no license, and IAM and CVL are for non-commercial
research: internal evaluation/research only.
