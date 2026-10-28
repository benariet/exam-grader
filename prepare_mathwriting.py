"""(Optional) MathWriting subset for retraining the Hebrew-vs-math classifier.

Streams the 3.1 GB MathWriting tarball once and keeps only the 6.4k single-symbol inks and the
short equations (~65k, ~700 MB):  python prepare_mathwriting.py
Source: google-research MathWriting (CC BY-NC-SA 4.0).
"""
import urllib.request
import re, sys, os, tarfile, html
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mathwriting")
lab_re = re.compile(rb'<annotation type="normalizedLabel">(.*?)</annotation>', re.S)
sym_re = re.compile(rb'<annotation type="label">(.*?)</annotation>', re.S)
simple = re.compile(r'^[A-Za-z0-9=+\-\.\\{}_^ ,()]*$')
keep_words = ("Omega", "mA", "Hz", "mV", "\\mu", "\\tau", "\\omega", "\\pi", "\\frac{V}", "\\frac{1}{2\\pi")
n = kept = 0
URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz"
os.makedirs(out, exist_ok=True)
labels = open(os.path.join(out, "labels.tsv"), "w", encoding="utf-8")
with urllib.request.urlopen(URL) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
    for m in tf:
        if not m.isfile() or not m.name.endswith(".inkml"):
            continue
        n += 1
        parts = m.name.split("/")
        split = parts[-2]
        data = tf.extractfile(m).read()
        if split == "symbols":
            lab = sym_re.search(data)
            lab = html.unescape(lab.group(1).decode()) if lab else ""
            path = os.path.join(out, "symbols", parts[-1]); os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data); kept += 1
            labels.write(f"symbols\t{parts[-1]}\t{lab}\n")
            continue
        if split not in ("train", "valid", "test"):
            continue
        lab = lab_re.search(data)
        lab = html.unescape(lab.group(1).decode()) if lab else ""
        labels.write(f"{split}\t{parts[-1]}\t{lab}\n")
        if "=" in lab and len(lab) <= 40 and (simple.match(lab) or any(w in lab for w in keep_words)):
            path = os.path.join(out, "expr", parts[-1]); os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data); kept += 1
        if n % 50000 == 0:
            print(f"{n} inks scanned, {kept} kept", flush=True)
labels.close()
print(f"done: {n} inks scanned, {kept} kept", flush=True)
