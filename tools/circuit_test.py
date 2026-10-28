"""Can a local vision-language model read a hand-drawn circuit? Tal's drawing (data/my_scans/tal_circuit.pdf)
read by Qwen3-VL-8B into a netlist, compared with its hand transcription (data/my_scans/tal_circuit_truth.json),
plus what the model says about it in words and the circuit solved from its reading (grader/circuit.py).

    python tools/circuit_test.py                         # 4-bit on the GPU, then full precision on the CPU
    python tools/circuit_test.py --modes gpu4            # only the 4-bit run
    python tools/circuit_test.py --image other.png --truth other_truth.json

Writes results/circuit_test.json and results/circuit_test.log (everything printed).
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
ap.add_argument("--image", default=os.path.join(HERE, "data", "my_scans", "tal_circuit.pdf"))
ap.add_argument("--truth", default=os.path.join(HERE, "data", "my_scans", "tal_circuit_truth.json"))
ap.add_argument("--modes", nargs="*", default=["gpu4", "cpubf16"], help="gpu4: 4-bit on the GPU; cpubf16: bfloat16 on the CPU")
ap.add_argument("--report", default=os.path.join(HERE, "results", "circuit_test.json"))
a = ap.parse_args()
os.makedirs(os.path.dirname(a.report), exist_ok=True)
LOG = open(os.path.splitext(a.report)[0] + ".log", "w", encoding="utf-8")


def say(*xs):
    s = " ".join(str(x) for x in xs)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def main():
    import torch

    from grader.circuit import compare, truth_netlist
    from read_page import load_pages
    from recog import gpu
    from recog.circuit_vlm import PROMPT_DESCRIBE, PROMPT_KCL, CircuitReader, crop_to_drawing
    from PIL import Image
    img = load_pages(a.image, 150)[0] if a.image.lower().endswith(".pdf") else Image.open(a.image).convert("RGB")
    crop = crop_to_drawing(img)
    crop.save(os.path.join(HERE, "results", "circuit_test_crop.png"))
    truth = truth_netlist(a.truth) if os.path.isfile(a.truth) else None
    say(f"machine: {gpu.describe()}")
    say(f"image: {os.path.basename(a.image)} {img.size} -> drawing {crop.size}")
    if truth:
        say(f"transcription: {len(truth.components)} components, KCL {truth.kcl()}")
    report = json.load(open(a.report, encoding="utf-8")) if os.path.isfile(a.report) else {}
    for mode in a.modes:
        say(f"\n===== {mode}")
        t0 = time.time()
        try:
            if mode == "gpu4":
                rd = CircuitReader(a.model, load_in_4bit=True)
            elif mode == "cpubf16":
                rd = CircuitReader(a.model, load_in_4bit=False, device="cpu", dtype=torch.bfloat16)
            else:
                raise ValueError(f"unknown mode {mode}")
            say(f"loaded in {time.time() - t0:.0f} s ({'4-bit' if rd.quantized else gpu.SHORT.get(rd.vlm.dtype)} on {rd.vlm.device})")
            out = {}
            t = time.time()
            net, raw = rd.read(crop, crop=False)
            out["netlist_secs"] = round(time.time() - t, 1)
            out["netlist_raw"] = raw
            say(f"--- netlist ({out['netlist_secs']} s):\n{raw}")
            out["problems"] = net.problems()
            out["kcl_from_reading"] = net.kcl()
            out["solved_from_reading"] = {k: str(v) for k, v in net.solve().items()}
            say(f"problems in the reading: {out['problems'] or 'none'}")
            say(f"KCL from its netlist: {out['kcl_from_reading']}")
            say(f"solved from its netlist: {out['solved_from_reading']}")
            if truth:
                out["compare"] = compare(net, truth)
                say(f"against the transcription: {json.dumps(out['compare'], ensure_ascii=False)}")
            t = time.time()
            out["description"] = rd.ask(crop, PROMPT_DESCRIBE, max_new_tokens=600)
            say(f"--- description ({time.time() - t:.0f} s):\n{out['description']}")
            t = time.time()
            out["kcl_answer"] = rd.ask(crop, PROMPT_KCL, max_new_tokens=200)
            say(f"--- KCL at A, asked directly ({time.time() - t:.0f} s):\n{out['kcl_answer']}")
            out["secs"] = round(time.time() - t0, 1)
            report[mode] = out
            del rd
            gpu.release()
        except Exception as e:                   # noqa: BLE001  (record it, go on with the next mode)
            say(traceback.format_exc())
            report[mode] = {"error": f"{type(e).__name__}: {e}"[:2000]}
        json.dump(report, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    say(f"\nreport: {os.path.relpath(a.report, HERE)}")


try:
    main()
except Exception:                                # noqa: BLE001
    say(traceback.format_exc())
