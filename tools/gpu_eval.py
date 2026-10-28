"""The strong readers on a GPU (Tal's RTX 2080 Ti, or any CUDA machine) against the readers in use - on
the test sets, Tal's notebook lines and his whole page. One report: results/gpu_eval.json (the Windows
runner gpu_eval.cmd also writes everything it prints to results/gpu_eval.log).

    python download_models.py uni-mumer-qwen3vl-2b dictalm-3.0-1.7b-base qwen3-vl-8b
    python tools/gpu_eval.py                          # env math hebrew english page
    python tools/gpu_eval.py --parts math hebrew      # some of them (the report keeps the other parts)

Parts
  env     : the machine, torch, a CUDA check, the number format each reader gets (recog/gpu.py)
  math    : Uni-MuMER (Qwen3-VL-2B fine-tuned on handwritten math) against Pix2Text MFR (in use) on the 70
            real handwritten formulas (data/handwritten_formulas) and Tal's six math lines
  hebrew  : the Hebrew line reader in context (recog/hebrew_context.py + Dicta-LM 3.0 1.7B) against the
            reader in use (five widths + word model) on the 300 test lines (unseen writers) and Tal's seven
  english : a vision-language model (default Qwen3-VL-8B; 4-bit on an 11 GB card) on 150 IAM test lines
            (the lines of eval_english.py) and Tal's three, against TrOCR-small / -base on the same lines
  page    : Tal's notebook page end to end (read_page.py) with the heavy readers on (Hebrew in context,
            Uni-MuMER as the second formula reader, the English VLM if downloaded), scored against its
            transcription (tools/score_page.py), plus the derivation checks (grader/selfcheck.py);
            the picture: results/tal_notebook_gpu_p1_read.png
A part that fails is recorded with its error and the run goes on with the next.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from PIL import Image  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", nargs="*", default=["env", "math", "hebrew", "english", "page"])
ap.add_argument("--math-model", default=os.path.join(HERE, "models", "uni-mumer-qwen3vl-2b"))
ap.add_argument("--english-model", default=os.path.join(HERE, "models", "qwen3-vl-8b"))
ap.add_argument("--hebrew-lm", default=None, help="another Hebrew language model folder (e.g. DictaLM-3.0-12B / 24B Base)")
ap.add_argument("--n-math", type=int, default=70)
ap.add_argument("--n-hebrew", type=int, default=300)
ap.add_argument("--n-english", type=int, default=150)
ap.add_argument("--load-in-4bit", choices=("auto", "yes", "no"), default="auto",
                help="the English VLM in 4 bits (bitsandbytes): auto = when it does not fit the GPU in 16 bits")
ap.add_argument("--report", default=os.path.join(HERE, "results", "gpu_eval.json"))
ap.add_argument("--no-retry", action="store_true", help="do not run again the parts that failed in the report")
a = ap.parse_args()

TAL = os.path.join(HERE, "data", "my_scans", "tal_notebook_lines")
tal = json.load(open(os.path.join(TAL, "labels.json"), encoding="utf-8")) if os.path.isdir(TAL) else []
report = json.load(open(a.report, encoding="utf-8")) if os.path.isfile(a.report) else {}
mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731


def say(*xs):
    print(f"[{time.strftime('%H:%M:%S')}]", *xs, flush=True)


def plain(o):
    """numpy numbers -> Python numbers (for json)"""
    return o.item() if hasattr(o, "item") else str(o)


def save():
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(report, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=plain)


def cer(ref, hyp):
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def complete(model_dir):
    """the model is fully downloaded: every weight file its index names is there"""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        shards = set(json.load(open(idx, encoding="utf-8"))["weight_map"].values())
        return all(os.path.isfile(os.path.join(model_dir, f)) for f in shards)
    return os.path.isfile(os.path.join(model_dir, "model.safetensors"))


def gpu_memory():
    import torch
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


# ------------------------------------------------------------------------------------------------ env
def part_env():
    import torch
    from recog import gpu
    out = {"machine": gpu.describe(), "python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.cuda.is_available()}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:
        out["transformers"] = None
    try:
        import bitsandbytes
        out["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:                       # noqa: BLE001  (not installed / no CUDA build)
        out["bitsandbytes"] = f"not usable: {e}"[:200]
    if torch.cuda.is_available():
        x = torch.randn(2048, 2048, device="cuda")
        ok = bool(torch.isfinite(x @ x).all())
        out["cuda_matmul_ok"] = ok
        for dt in (torch.float32, torch.float16):            # rough speed of the card in each format
            y = x.to(dt)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                y @ y
            torch.cuda.synchronize()
            out[f"tflops_{gpu.SHORT[dt]}"] = round(20 * 2 * 2048 ** 3 / (time.time() - t) / 1e12, 1)
    for name, d, fast in (("math (Uni-MuMER)", a.math_model, False), ("hebrew LM (Dicta-LM)", os.path.join(HERE, "models", "dictalm-3.0-1.7b-base"), True)):
        n_b = gpu.params_b(d)
        out[f"format {name}"] = f"{gpu.SHORT[gpu.pick(n_b, fast=fast)[1]]} ({n_b:.2f}B parameters)" if n_b else "not downloaded"
    n_b = gpu.params_b(a.english_model)
    out["format english VLM"] = (f"4-bit + {gpu.SHORT[gpu.pick(None, fast=True)[1]]}" if gpu.needs_4bit(n_b) else
                                 gpu.SHORT[gpu.pick(n_b)[1]]) + f" ({n_b:.2f}B parameters)" if n_b else "not downloaded"
    return out


# ------------------------------------------------------------------------------------------------ math
def part_math():
    from grader.symbolic import compare_formula
    from recog import gpu
    from recog.formula import FormulaRecognizer
    from recog.mumer import MumerReader, SOURCE_DIR, ensure_v4_layout
    dst = os.path.join(HERE, "models", "uni-mumer-v4") if os.path.abspath(a.math_model) == os.path.abspath(SOURCE_DIR) \
        else a.math_model.rstrip("/\\") + "-v4"
    t0 = time.time()
    rd = MumerReader(ensure_v4_layout(a.math_model, dst))
    say(f"Uni-MuMER loaded in {time.time() - t0:.0f} s, {gpu.SHORT.get(rd.dtype)} on {rd.device}")
    p2t = FormulaRecognizer()
    D = os.path.join(HERE, "data", "handwritten_formulas")
    labels = {os.path.basename(k): v for k, v in json.load(open(os.path.join(D, "labels.json"), encoding="utf-8")).items()}
    norm = lambda s: s.replace(" ", "")  # noqa: E731
    rows, t_m, t_p = [], 0.0, 0.0
    for i, f in enumerate(sorted(labels)[: a.n_math], 1):
        gt, path = labels[f], os.path.join(D, f)
        t = time.time()
        m = rd.recognize(path)[0].text
        t_m += time.time() - t
        t = time.time()
        p = p2t.recognize(path, n_best=1)[0].text
        t_p += time.time() - t
        eq = lambda x: compare_formula(x, gt).status == "equivalent"  # noqa: E731
        rows.append({"file": f, "gt": gt, "mumer": m, "pix2text": p, "mumer_exact": norm(m) == norm(gt),
                     "pix2text_exact": norm(p) == norm(gt), "mumer_equiv": eq(m), "pix2text_equiv": eq(p)})
        if i % 10 == 0:
            say(f"  {i} formulas: Uni-MuMER {sum(r['mumer_equiv'] for r in rows)}/{i}, "
                f"Pix2Text {sum(r['pix2text_equiv'] for r in rows)}/{i} equivalent")
    n = max(1, len(rows))
    out = {"model": os.path.basename(a.math_model), "format": gpu.SHORT.get(rd.dtype), "device": rd.device,
           "formulas": len(rows)}
    for k in ("mumer_exact", "pix2text_exact", "mumer_equiv", "pix2text_equiv"):
        out[k] = sum(r[k] for r in rows)
    out.update({"secs_per_formula_mumer": round(t_m / n, 2), "secs_per_formula_pix2text": round(t_p / n, 2),
                "gpu_memory_peak_gb": gpu_memory()})
    out["tal"] = [{"truth": t["text"], "mumer": rd.recognize(os.path.join(TAL, t["file"]))[0].text,
                   "pix2text": p2t.recognize(os.path.join(TAL, t["file"]), n_best=1)[0].text}
                  for t in tal if t["kind"] == "math"]
    out["rows"] = rows
    del rd
    gpu.release()
    return out


# ------------------------------------------------------------------------------------------------ hebrew
def part_hebrew():
    from recog import gpu
    from recog.hebrew_context import HebrewContext, HebrewLM
    from recog.hebrew_text import clean
    t0 = time.time()
    lm = HebrewLM(a.hebrew_lm) if a.hebrew_lm else HebrewLM()
    ctx = HebrewContext(lm=lm)
    say(f"Hebrew LM loaded in {time.time() - t0:.0f} s, {gpu.SHORT.get(lm.dtype)} on {lm.device}")
    d = os.path.join(HERE, "data", "hebrew_lines")
    items = json.load(open(os.path.join(d, "labels.json"), encoding="utf-8"))[: a.n_hebrew]
    out = {"lm": os.path.basename(lm.model_dir), "format": gpu.SHORT.get(lm.dtype), "device": lm.device}
    e0 = e1 = n = 0
    rows, t0 = [], time.time()
    for i, it in enumerate(items, 1):
        img, truth = Image.open(os.path.join(d, it["file"])).convert("L"), clean(it["text"])
        prepared = ctx.start(img)
        best = clean(ctx.read(img, prepared=prepared)[0].text)
        used = clean(prepared[1][0])
        e0 += Levenshtein.distance(used, truth)
        e1 += Levenshtein.distance(best, truth)
        n += len(truth)
        rows.append({"file": it["file"], "truth": truth, "in_use": used, "in_context": best})
        if i % 25 == 0 or i == len(items):
            out.update({"lines": i, "CER_in_use": round(e0 / n, 4), "CER_in_context": round(e1 / n, 4),
                        "secs_per_line": round((time.time() - t0) / i, 2), "format": gpu.SHORT.get(lm.dtype)})
            report["hebrew"] = out
            save()
            say(f"  {i} lines: CER in use {e0 / n:.2%}, in context {e1 / n:.2%}, {out['secs_per_line']} s/line")
    out["gpu_memory_peak_gb"] = gpu_memory()
    out["tal"] = []
    for t in tal:
        if t["kind"] != "hebrew":
            continue
        img = Image.open(os.path.join(TAL, t["file"])).convert("L")
        prepared = ctx.start(img)
        out["tal"].append({"truth": t["text"], "in_use": prepared[1][0], "in_context": ctx.read(img, prepared=prepared)[0].text})
    if out["tal"]:
        tr = [clean(x["truth"]) for x in out["tal"]]
        out["tal_CER_in_use"] = round(sum(Levenshtein.distance(clean(x["in_use"]), r) for x, r in zip(out["tal"], tr)) / sum(map(len, tr)), 4)
        out["tal_CER_in_context"] = round(sum(Levenshtein.distance(clean(x["in_context"]), r) for x, r in zip(out["tal"], tr)) / sum(map(len, tr)), 4)
    out["rows"] = rows
    del ctx, lm
    gpu.release()
    return out


# ------------------------------------------------------------------------------------------------ english
def part_english():
    import pandas as pd
    from recog import gpu
    from recog.hebrew_vlm import PROMPT_ENGLISH, HebrewRecognizer as VLM

    def norm(t):
        return " ".join(t.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" ;", ";")
                        .replace(" :", ":").replace(" !", "!").replace(" ?", "?").split())
    if not complete(a.english_model):
        raise FileNotFoundError(f"{a.english_model} not downloaded (python download_models.py {os.path.basename(a.english_model)})")
    n_b = gpu.params_b(a.english_model)
    q4 = a.load_in_4bit == "yes" or (a.load_in_4bit == "auto" and gpu.needs_4bit(n_b))
    t0 = time.time()
    vlm = VLM(a.english_model, load_in_4bit=q4)
    say(f"English VLM loaded in {time.time() - t0:.0f} s: {'4-bit + ' if q4 else ''}{gpu.SHORT.get(vlm.dtype)} on {vlm.device}")
    placed = getattr(vlm.model, "hf_device_map", None) or {}
    on_cpu = [k for k, v in placed.items() if str(v) in ("cpu", "disk")]
    if on_cpu:
        say(f"  {len(on_cpu)} of {len(placed)} parts kept on the CPU (the GPU is short of memory): {', '.join(on_cpu[:8])}")
    df = pd.read_parquet(os.path.join(HERE, "data", "english_lines", "iam_test.parquet"))
    idx = random.Random(0).sample(range(len(df)), min(150, len(df)))[: a.n_english]
    rows, t0 = [], time.time()
    for i, j in enumerate(idx, 1):
        r = df.iloc[j]
        hyp = norm(vlm.recognize(Image.open(io.BytesIO(r["image"]["bytes"])), n_best=1, prompt=PROMPT_ENGLISH)[0].text)
        ref = norm(r["text"])
        rows.append({"ref": ref, "hyp": hyp, "cer": round(cer(ref, hyp), 4)})
        if i % 25 == 0:
            say(f"  {i} lines: CER {mean([x['cer'] for x in rows]):.2%}, {(time.time() - t0) / i:.2f} s/line")
    out = {"model": os.path.basename(a.english_model), "quantized_4bit": q4, "format": gpu.SHORT.get(vlm.dtype),
           "parts_on_cpu": len(on_cpu),
           "lines": len(rows), "CER": round(mean([x["cer"] for x in rows]), 4),
           "secs_per_line": round((time.time() - t0) / max(1, len(rows)), 2), "gpu_memory_peak_gb": gpu_memory()}
    for tag, f in (("trocr_small", "english_eval.json"), ("trocr_base", "english_eval_base_encfp32.json")):
        p = os.path.join(HERE, "results", f)
        if os.path.isfile(p):
            old = json.load(open(p, encoding="utf-8"))["rows"][: len(rows)]
            out[f"CER_{tag}_same_lines"] = round(mean([x["cer"] for x in old]), 4)
    out["tal"] = [{"truth": t["text"], "read": vlm.recognize(os.path.join(TAL, t["file"]), n_best=1, prompt=PROMPT_ENGLISH)[0].text}
                  for t in tal if t["kind"] == "english"]
    out["rows"] = rows
    del vlm
    gpu.release()
    return out


# ------------------------------------------------------------------------------------------------ page
def part_page():
    from grader.selfcheck import check_pages
    from read_page import read_pages, write_page
    from recog.mixed_reader import MixedReader
    from score_page import norm_tex, score
    pdf = os.path.join(HERE, "data", "my_scans", "tal_notebook.pdf")
    truth = json.load(open(os.path.join(HERE, "data", "my_scans", "tal_notebook_truth.json"), encoding="utf-8"))
    os.environ["EXAM_GRADER_HEBREW_CONTEXT"] = "1"
    os.environ["EXAM_GRADER_FORMULA2"] = "mumer"
    if complete(a.english_model) and "error" not in report.get("english", {"error": ""}):
        os.environ["EXAM_GRADER_ENGLISH_VLM"] = a.english_model      # (only after it worked in the english part)
    t0 = time.time()
    pages = read_pages(pdf, reader=MixedReader())
    out = {"secs": round(time.time() - t0, 1), "heavy_readers": ["hebrew in context", "Uni-MuMER"] +
           (["English VLM"] if os.environ.get("EXAM_GRADER_ENGLISH_VLM") else [])}
    out["score"] = score(pages, truth)
    # the formulas once more with each formula reader alone, and with the two weighed (what the checks use)
    from grader.selfcheck import _mix, _raw
    segs = [s for p in pages for ln in p["lines"] for s in ln if s.kind == "math"]
    for tag, pick in (("pix2text", lambda s: s.nbest[0][0]),
                      ("mumer", lambda s: (getattr(s, "math2", None) or s.nbest)[0][0]),
                      ("both_weighed", lambda s: (_mix(_raw(s.nbest)[:3], _raw(s.math2)[:3]) if getattr(s, "math2", None)
                                                  else s.nbest)[0][0])):
        pieces = [norm_tex(pick(s)) for s in segs]
        found = 0
        for f in truth.get("math", []):
            key = norm_tex(f)
            for i, p in enumerate(pieces):
                if key and key in p:
                    pieces[i] = p.replace(key, "\0", 1)
                    found += 1
                    break
        out[f"formulas_right_{tag}"] = f"{found}/{len(truth.get('math', []))}"
    out["checks"] = [{"problem": c.problem, "verdict": c.verdict, "text": c.text, "confidence": round(c.confidence, 3),
                      "sure": c.sure} for c in check_pages(pages)]
    out["text"] = []
    for pno, res in enumerate(pages, 1):
        out["text"] += write_page(res, pno, os.path.join(HERE, "results", f"tal_notebook_gpu_p{pno}_read.png"), check=True)
    out["gpu_memory_peak_gb"] = gpu_memory()
    return out


PARTS = {"env": part_env, "math": part_math, "hebrew": part_hebrew, "english": part_english, "page": part_page}
retry = [] if a.no_retry else [p for p in PARTS if p not in a.parts and "error" in report.get(p, {})]
if retry:
    say(f"running again the parts that failed before: {' '.join(retry)}")
for name in retry + a.parts:
    say(f"===== {name}")
    t0 = time.time()
    try:
        res = PARTS[name]()
        res["part_secs"] = round(time.time() - t0, 1)
        report[name] = res
        brief = {k: v for k, v in res.items() if k not in ("rows", "text")}
        say(json.dumps(brief, ensure_ascii=False, indent=1, default=plain))
        if name == "page":
            print("\n".join(res["text"]), flush=True)
    except Exception as e:                       # noqa: BLE001  (record it, go on with the next part)
        traceback.print_exc()
        report[name] = {"error": f"{type(e).__name__}: {e}"[:2000], "part_secs": round(time.time() - t0, 1)}
        say(f"part {name} FAILED: {type(e).__name__}: {e}")
    save()
say(f"report: {os.path.relpath(a.report, HERE)}")
