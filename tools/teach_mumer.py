"""Teach the formula reader (recog/mumer.py: Uni-MuMER, Qwen3-VL-2B) lines of handwriting with their text - a LoRA on
its language part, the reader's own prompt and the text as the answer (only the answer's tokens learned); the
held-out lines read before and after (character error rate on the reader's tokens, exact lines), the adapter saved
when they are read better. The reader takes it: MumerReader(adapter_dir=...), tools/read_crops.py --mumer-adapter.

    python tools/teach_mumer.py --train <dir>/train.jsonl --held <dir>/held.jsonl --root <dir> --out <adapter dir>
        [--epochs 1] [--lr 1e-4] [--rank 16] [--accum 8] [--eval 400] [--max-train 0]
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def cer(pred: list, gold: list) -> float:
    from rapidfuzz.distance import Levenshtein
    errs = sum(Levenshtein.distance(p.split(), g.split()) for p, g in zip(pred, gold))
    return errs / max(1, sum(len(g.split()) for g in gold))


def norm(text: str) -> str:
    """a reading in the labels' tokens (tools/fermat_lines.tokens), for comparing"""
    from tools.fermat_lines import tokens
    return " ".join(tokens(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--root", required=True, help="the folder the lines' image paths are under")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval", type=int, default=400, help="held-out lines read before and after")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    from recog.mumer import PROMPT, SYSTEM, MumerReader

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    train = [json.loads(x) for x in open(a.train, encoding="utf-8")]
    held = [json.loads(x) for x in open(a.held, encoding="utf-8")]
    random.shuffle(train)
    if a.max_train:
        train = train[:a.max_train]
    held = held[:a.eval]
    print(f"{len(train)} lines to learn, {len(held)} held out", flush=True)

    rd = MumerReader(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    def read_held(tag):
        t0 = time.time()
        pred = [norm(rd.recognize(Image.open(os.path.join(a.root, r["image"])), n_best=1, max_new_tokens=160)[0].text)
                for r in held]
        gold = [r["text"] for r in held]
        c = cer(pred, gold)
        exact = sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))
        print(f"{tag}: token error {100 * c:.2f} %, lines exact {100 * exact:.1f} % ({time.time() - t0:.0f} s)",
              flush=True)
        return c

    before = read_held("before")
    model = rd.model
    model.config.use_cache = False
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = int(len(train) * a.epochs)
    n_up = max(1, steps // a.accum)                    # (a short warm-up, then down to a twentieth)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / n_up))
    tok, imp = rd.tok, rd.imp
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    model.train()
    t0, run = time.time(), 0.0
    for step in range(steps):
        r = train[step % len(train)]
        img = Image.open(os.path.join(a.root, r["image"])).convert("RGB")
        if min(img.size) < 32:
            s = 32 / min(img.size)
            img = img.resize((max(32, int(img.width * s)), max(32, int(img.height * s))), Image.BICUBIC)
        vis = imp(images=[img], return_tensors="pt")
        n_img = int(vis["image_grid_thw"].prod()) // (imp.merge_size ** 2)
        prompt = head.replace("<|image_pad|>", "<|image_pad|>" * n_img, 1)
        p_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        t_ids = tok(r["text"] + "<|im_end|>", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = torch.cat([p_ids, t_ids]).unsqueeze(0).to(rd.device)
        labels = ids.clone()
        labels[0, :len(p_ids)] = -100
        extra = {"mm_token_type_ids": rd.mm_types(ids)} if rd._mm else {}
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                    pixel_values=vis["pixel_values"].to(rd.device, rd.dtype),
                    image_grid_thw=vis["image_grid_thw"].to(rd.device), **extra)
        (out.loss / a.accum).backward()
        run = 0.98 * run + 0.02 * float(out.loss) if step else float(out.loss)
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"step {step}/{steps} loss {run:.3f} ({time.time() - t0:.0f} s)", flush=True)
    model.eval()
    model.config.use_cache = True
    rd.model = model
    after = read_held("after")
    if after < before:
        model.save_pretrained(a.out)
        json.dump({"before": before, "after": after, "lines": len(train), "epochs": a.epochs, "lr": a.lr,
                   "rank": a.rank}, open(os.path.join(a.out, "teach.json"), "w"), indent=1)
        print(f"saved -> {a.out}", flush=True)
    else:
        print("not better - not saved", flush=True)


if __name__ == "__main__":
    main()
