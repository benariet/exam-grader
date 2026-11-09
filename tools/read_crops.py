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
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BOX_PROMPT = ("This image is a box from a digital logic exam with three printed labels, 'Vx =', 'Y1 =' and 'Y2 =', each "
              "followed by a student's handwritten Boolean expression (the writing may run out of the box). Write the "
              "three expressions, each on its own line as 'Vx = ...', 'Y1 = ...', 'Y2 = ...', in LaTeX: variables A, B, "
              "C, V_x; OR written +; AND as letters side by side or \\cdot; XOR (a circled plus) \\oplus; NOT as "
              "\\overline{...} or a prime '. Write exactly what is written; where something cannot be read write ? - "
              "do not guess.")
PROMPTS = {
    "boolean": ("This image is one handwritten Boolean expression from a digital logic exam: variables A, B, C and V_x; "
                "OR written +, AND written as letters side by side or with a dot, XOR written as a circled plus, NOT as a "
                "bar over a letter or a group or a prime after it, and brackets. Write exactly what is written, in LaTeX "
                "(\\oplus, \\overline{...}, ', \\cdot), on one line, and nothing else. If something cannot be read, "
                "write ? in its place - do not guess."),
}


def box_rows(text: str):
    """the vision model's reading of a whole box -> [(row name, its expression)]: each label ('Vx =', 'Y_{1} =',
    'Y2:') to the next, on one line or on several (in $..$ or \\[..\\], bold, a list's dash - as a chat model writes)"""
    import re
    t = re.sub(r"\*\*|`|\$|\\\[|\\\]|\\\(|\\\)", " ", text)
    lab = re.compile(r"(?:\\mathrm\s*\{\s*)?(V\s*_?\s*\{?\s*x\s*\}?|Y\s*_?\s*\{?\s*([12])\s*\}?)\s*\}?\s*[=:]")
    ms = list(lab.finditer(t))
    out = []
    for i, m in enumerate(ms):
        expr = t[m.end():ms[i + 1].start() if i + 1 < len(ms) else len(t)]
        expr = re.sub(r"^[\s,;]+|(?:[\s,;.]|\\\\|\\q?quad|\\[,;]|\n\s*[-•])+$", "", expr).strip()
        if expr:
            out.append(("Vx" if m.group(1).strip().startswith("V") else f"Y{m.group(2)}", expr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readers", default="mumer,vlm")
    ap.add_argument("--prompt", default="boolean")
    ap.add_argument("--vlm", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--n-best", type=int, default=3)
    a = ap.parse_args()
    from PIL import Image
    files = sorted(glob.glob(os.path.join(a.folder, "*.png")))
    out = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {}
    for reader in a.readers.split(","):
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
        for f in files:
            key = os.path.basename(f)[:-4]
            try:
                if key.endswith("_box"):               # (a whole box: its rows read together, by the vision model)
                    if reader != "vlm":
                        continue
                    exam = key[:-4]
                    for text, logp in ((c.text, c.logp) for c in rd.recognize(Image.open(f).convert("RGB"), n_best=1,
                                                                             max_new_tokens=300, prompt=BOX_PROMPT)):
                        out.setdefault(key, {})["raw"] = text           # (as written - to see what the parse missed)
                        for name, expr in box_rows(text):
                            out.setdefault(f"{exam}_{name}", {})["vlmbox"] = [[expr, float(logp)]]
                    continue
                cands = read(Image.open(f).convert("RGB"))
                out.setdefault(key, {})[reader] = [[c.text, float(c.logp)] for c in cands]
            except Exception as e:                     # noqa: BLE001  (one piece: the rest read)
                print(f"{key}: {type(e).__name__}: {e}", flush=True)
        print(f"{reader}: {len(files)} pieces in {time.time() - t1:.0f} s", flush=True)
        del rd
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
