"""Download the public evaluation sets into ./data (once).

  data/handwritten_formulas/  70 human-written formulas + LaTeX  (linxy/LaTeX_OCR, human_handwrite/test)
  data/hebrew_lines/          300 Hebrew handwriting lines + text (cyttic/trocr-hebrew-matan, test split;
                              109 writers, disjoint from its train split). No license is stated for this
                              dataset: use it for internal evaluation only.
  data/english_lines/         IAM handwritten English lines (Teklia/IAM-line): test for eval_english.py,
                              train / validation for train_english_classifier.py. The IAM database is
                              for non-commercial research use.
"""
import io
import json
import os
import random

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def formulas():
    out = os.path.join(HERE, "data", "handwritten_formulas")
    if os.path.isfile(os.path.join(out, "labels.json")) and len(os.listdir(out)) >= 71:
        print("formulas: already there"); return
    os.makedirs(out, exist_ok=True)
    p = hf_hub_download("linxy/LaTeX_OCR", "human_handwrite/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    labels = {}
    for i, r in df.iterrows():
        f = f"{i:02d}.png"
        Image.open(io.BytesIO(r["image"]["bytes"])).save(os.path.join(out, f))
        labels[f] = r["text"]
    json.dump(labels, open(os.path.join(out, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"formulas: {len(labels)} images")


def hebrew(n=300):
    out = os.path.join(HERE, "data", "hebrew_lines")
    if os.path.isfile(os.path.join(out, "labels.json")):
        print("hebrew: already there"); return
    os.makedirs(out, exist_ok=True)
    p = hf_hub_download("cyttic/trocr-hebrew-matan", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    idx = random.Random(0).sample(range(len(df)), n)
    meta = []
    for k, i in enumerate(idx):
        r = df.iloc[i]
        f = f"{k:03d}.png"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("L").save(os.path.join(out, f))
        meta.append({"file": f, "text": r["text"], "writer": r["writer"], "source_index": int(i)})
    json.dump(meta, open(os.path.join(out, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"hebrew: {len(meta)} lines from {len({m['writer'] for m in meta})} writers")


def hebrew_val():
    """Full test split (109 writers) for validating / calibrating the Hebrew-vs-math classifier."""
    target = os.path.join(HERE, "data", "val", "matan_test.parquet")
    if os.path.isfile(target):
        print("val: already there"); return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    p = hf_hub_download("cyttic/trocr-hebrew-matan", "data/test-00000-of-00001.parquet", repo_type="dataset")
    with open(p, "rb") as fi, open(target, "wb") as fo:
        fo.write(fi.read())
    print(f"val: matan_test.parquet ({os.path.getsize(target) / 1e6:.0f} MB)")


def hebrew_train():
    """Training sets for train_hebrew.py (kept as parquet). Test writers are never included."""
    out = os.path.join(HERE, "data", "train")
    os.makedirs(out, exist_ok=True)
    for repo, src, dst in (("cyttic/trocr-hebrew-matan", "data/train-00000-of-00001.parquet", "matan_train.parquet"),
                           ("cyttic/trocr-hebrew-human", "train-00000-of-00001.parquet", "human_train.parquet")):
        target = os.path.join(out, dst)
        if os.path.isfile(target):
            print(f"train: {dst} already there"); continue
        p = hf_hub_download(repo, src, repo_type="dataset")
        with open(p, "rb") as fi, open(target, "wb") as fo:
            fo.write(fi.read())
        print(f"train: {dst} ({os.path.getsize(target) / 1e6:.0f} MB)")


def english():
    out = os.path.join(HERE, "data", "english_lines")
    os.makedirs(out, exist_ok=True)
    for split in ("test", "train", "validation"):
        target = os.path.join(out, f"iam_{split}.parquet")
        if os.path.isfile(target):
            print(f"english: iam_{split} already there"); continue
        p = hf_hub_download("Teklia/IAM-line", f"data/{split}.parquet", repo_type="dataset")
        with open(p, "rb") as fi, open(target, "wb") as fo:
            fo.write(fi.read())
        print(f"english: iam_{split}.parquet ({os.path.getsize(target) / 1e6:.0f} MB)")


if __name__ == "__main__":
    formulas()
    hebrew()
    hebrew_val()
    hebrew_train()
    english()
