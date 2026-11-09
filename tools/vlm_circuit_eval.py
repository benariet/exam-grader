"""Transistor drawings read by the local vision-language model (recog/circuit_vlm.py's transistor prompt, Qwen3-VL)
instead of the circuit reader: tools/ptl_synth.py's / tools/cmos_synth.py's drawings, each cut out round its truth
boxes, the Boolean function read against the drawing's - the second reader's measure beside tools/circuit_eval.py.

    python tools/vlm_circuit_eval.py --synth ~/eg-data/ptl_hatm5 --inputs ABC [--model models/qwen3-vl-8b]
        -> how many functions right; <synth>/vlm_answers.json (each drawing's answer)
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def cut_out(img, boxes, long_side=1100):
    """the drawing round its boxes, a margin round it, scaled to long_side"""
    from PIL import Image
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    m = 0.08 * max(x1 - x0, y1 - y0)
    crop = img.crop((max(0, int(x0 - m)), max(0, int(y0 - m)), min(img.width, int(x1 + m)), min(img.height, int(y1 + m))))
    k = long_side / max(crop.size)
    return crop.resize((max(1, round(crop.width * k)), max(1, round(crop.height * k))), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--inputs", default="")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import sympy as sp
    from PIL import Image
    from grader.circuit import truth_netlist
    from recog.circuit_vlm import CircuitReader
    rd = CircuitReader(a.model)
    up = lambda e: e.xreplace({x: sp.Symbol(str(x).upper()) for x in e.free_symbols})   # noqa: E731
    files = sorted(glob.glob(os.path.join(a.synth, "*.json")))
    files = [f for f in files if not f.endswith("vlm_answers.json")][: a.limit or None]
    right, n, answers = 0, 0, {}
    for f in files:
        t = json.load(open(f, encoding="utf-8"))
        img = Image.open(f[:-5] + ".png").convert("RGB")
        k = img.width / t["page"][0]
        boxes = [[v * k for v in b] for _, b in t["boxes"]]
        truth = truth_netlist(f)
        tf = truth.switch_logic(inputs=a.inputs or None)
        if not tf:
            continue
        net, raw = rd.read_transistors(cut_out(img, boxes))
        rf = [up(g) for g in net.switch_logic(inputs=a.inputs or None).values()]
        ok = all(any(not sp.satisfiable(sp.Xor(up(g), h)) for h in rf) for g in tf.values())
        right += ok
        n += 1
        answers[os.path.basename(f)[:-5]] = {"right": ok, "answer": raw}
        print(os.path.basename(f)[:-5], ok, flush=True)
    json.dump(answers, open(os.path.join(a.synth, "vlm_answers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"the vision model's functions right: {right}/{n}")


if __name__ == "__main__":
    main()
