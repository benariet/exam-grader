"""Pieces of handwriting cut out (a folder of PNGs - tools/expr_box.py --crops-only) read by the GPU formula readers:
Uni-MuMER (recog/mumer.py) and the vision-language model (Qwen3-VL, recog/hebrew_vlm.py) told what the piece is - for
tools/expr_box.py --second to judge with, on the computer the exams stay on.

    python tools/read_crops.py <folder> --out readings.json [--readers mumer,vlm] [--prompt boolean]
        -> {"<the PNG's name>": {"mumer": [[latex, logp], ...], "vlm": [[latex, logp], ...]}}
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BOX_PROMPTS = {
    "boolean": ("This image is a box from a digital logic exam with printed labels, {labels}, each followed by a "
                "student's handwritten Boolean expression (the writing may run out of the box). Write the expressions, "
                "each on its own line as {lines}, in LaTeX: variables A, B, C, V_x; OR written +; AND as letters side "
                "by side or \\cdot; XOR (a circled plus) \\oplus; NOT as \\overline{{...}} or a prime '. Write "
                "exactly what is written; where something cannot be read write ? - do not guess."),
    "words": ("This image is a box from an exam with printed labels, {labels}, each followed by a student's handwritten "
              "answer in Hebrew or English (the writing may run out of the box). Write each answer exactly as written, "
              "in its own language, each on its own line as {lines}. Where something cannot be read write ? - do not "
              "guess."),
}
FINAL_PROMPT = ("This image is part of an exam page: a student's handwritten answer for {label}, maybe with its "
                "working. Write only the final value the student gave for it, with its unit if one is written (for "
                "example 12.5%). If no final value is written, write ?.")
PROMPTS = {
    "math": ("This image is handwriting from an engineering exam: a short calculation or a final value. Write it in "
             "LaTeX exactly as written, on one line, and nothing else. If something cannot be read, write ? in its "
             "place - do not guess."),
    "words": ("This image is one handwritten line from an exam answer, in Hebrew or English (a word or a few - maybe a "
              "name of a circuit component, or a short phrase). Write exactly what is written, in its own language, "
              "and nothing else. If something cannot be read, write ? in its place - do not guess."),
    "boolean": ("This image is one handwritten Boolean expression from a digital logic exam: variables A, B, C and V_x; "
                "OR written +, AND written as letters side by side or with a dot, XOR written as a circled plus, NOT as a "
                "bar over a letter or a group or a prime after it, and brackets. Write exactly what is written, in LaTeX "
                "(\\oplus, \\overline{...}, ', \\cdot), on one line, and nothing else. If something cannot be read, "
                "write ? in its place - do not guess."),
}


def label_pattern(name: str) -> str:
    """a printed label as a reader writes it back: 'Y1' -> Y1, Y_1, Y_{1}; 'Vx' -> V_x, V_{x}"""
    import re
    m = re.fullmatch(r"([A-Za-z]+?)([0-9]+|[a-z])", name) if len(name) <= 3 else None
    if m:
        return re.escape(m.group(1)) + r"\s*_?\s*\{?\s*" + re.escape(m.group(2)) + r"\s*\}?"
    return re.escape(name)


def box_rows(text: str, labels=("Vx", "Y1", "Y2")):
    """the vision model's reading of a whole box -> [(row label, its answer)]: each label ('Vx =', 'Y_{1} =', 'Y2:',
    'Function -') to the next, on one line or on several (in $..$ or \\[..\\], bold, a list's dash - as a chat model
    writes)"""
    import re
    t = re.sub(r"\*\*|`|\$|\\\[|\\\]|\\\(|\\\)", " ", text)
    alts = "|".join(f"(?P<l{i}>{label_pattern(n)})" for i, n in enumerate(labels))
    lab = re.compile(r"(?:\\mathrm\s*\{\s*)?(?:" + alts + r")\s*\}?\s*[=:\-\u2013\u2014\u2212]")
    ms = list(lab.finditer(t))
    out = []
    for i, m in enumerate(ms):
        expr = t[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(t)]
        expr = re.sub(r"^[\s,;]+|(?:[\s,;.]|\\\\|\\q?quad|\\[,;]|\n\s*[-\u2022])+$", "", expr).strip()
        name = next(labels[k] for k in range(len(labels)) if m.group(f"l{k}"))
        if expr:
            out.append((name, expr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readers", default="mumer,vlm")
    ap.add_argument("--prompt", default="boolean")
    ap.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--n-best", type=int, default=3)
    ap.add_argument("--views", default="", help="each piece read at these scales too: '1|0.75|1.33' (readers "
                    "'mumer@0.75' ...; '|' - Slurm's --export splits at commas)")
    ap.add_argument("--labels", default="Vx,Y1,Y2", help="a whole box's printed labels, top to bottom")
    ap.add_argument("--label-text", default="", help="what a value's place asks for, for the final-value question "
                    "('_' for spaces: Slurm's --export): \"the_DUTY_CICLE_printed_at_the_bottom\"")
    ap.add_argument("--sep", default="", help="what follows each label in print ('=' for expressions, '-' for words)")
    a = ap.parse_args()
    from PIL import Image
    files = sorted(glob.glob(os.path.join(a.folder, "*.png")))
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    for reader in [r for r in re.split(r"[,|]", a.readers) if r]:      # ('|' for Slurm's --export)
        t0 = time.time()
        if reader == "mumer":
            from recog.mumer import MumerReader
            rd = MumerReader()
            read = lambda img: rd.recognize(img, n_best=a.n_best, max_new_tokens=160)          # noqa: E731
        elif reader == "vlm":
            import torch
            from recog.hebrew_vlm import HebrewRecognizer
            rd = HebrewRecognizer(model_dir=a.vlm, dtype=torch.bfloat16 if torch.cuda.is_available() and
                                  torch.cuda.is_bf16_supported() else None)
            read = lambda img: rd.recognize(img, n_best=a.n_best, max_new_tokens=160, prompt=PROMPTS[a.prompt])  # noqa: E731
        else:
            raise SystemExit(f"no reader {reader}")
        print(f"{reader} ready in {time.time() - t0:.0f} s", flush=True)
        t1 = time.time()
        sep = a.sep or ("=" if a.prompt == "boolean" else "-")
        views = [float(x) for x in re.split(r"[,;|:]", a.views) if x] if a.views else [1.0]   # (a '|' for Slurm)
        for f in files:
            key = os.path.basename(f)[:-4]
            try:
                if key.endswith("_valuebox"):          # (a value's place with its label: the final value asked for)
                    if reader == "vlm":
                        c = rd.recognize(Image.open(f).convert("RGB"), n_best=1, max_new_tokens=40,
                                         prompt=FINAL_PROMPT.format(label=a.label_text.replace("_", " ")))[0]
                        out.setdefault(key[:-len("box")], {})["vlmfinal"] = [[c.text, float(c.logp)]]
                    continue
                if key.endswith("_box"):               # (a whole box: its rows read together, by the vision model)
                    if reader != "vlm":
                        continue
                    exam = key[:-4]
                    labels = [x.strip() for x in re.split(r"[,;|]", a.labels) if x.strip()]   # (a '|' for Slurm's
                    #                                                                          --export, split at commas)
                    prompt = BOX_PROMPTS[a.prompt].format(labels=", ".join(f"'{x} {sep}'" for x in labels),
                                                          lines=", ".join(f"'{x} {sep} ...'" for x in labels))
                    for text, logp in ((c.text, c.logp) for c in rd.recognize(Image.open(f).convert("RGB"), n_best=1,
                                                                             max_new_tokens=300, prompt=prompt)):
                        out.setdefault(key, {})["raw"] = text           # (as written - to see what the parse missed)
                        for name, expr in box_rows(text, labels):
                            out.setdefault(f"{exam}_{name}", {})["vlmbox"] = [[expr, float(logp)]]
                    continue
                img = Image.open(f).convert("RGB")
                for v in views:                        # (the piece at other scales too: a reading most views give
                    im = img if v == 1 else img.resize((max(8, round(img.width * v)), max(8, round(img.height * v))),
                                                       Image.BICUBIC)               # is likelier read right)
                    cands = read(im)
                    out.setdefault(key, {})[reader if v == 1 else f"{reader}@{v:g}"] = \
                        [[c.text, float(c.logp)] for c in cands]
            except Exception as e:                     # noqa: BLE001  (one piece: the rest read)
                print(f"{key}: {type(e).__name__}: {e}", flush=True)
        print(f"{reader}: {len(files)} pieces in {time.time() - t1:.0f} s", flush=True)
        del rd
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
