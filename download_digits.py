"""Download the handwritten-digit collections the cell reader is trained and tested on (~450 MB), so a
cell reader can be (re)trained or fine-tuned on this computer:

    python download_digits.py              # -> data/digits/...
    python download_digits.py --test-only  # only the test collections (USPS, Semeion, screen, CVL)
    python download_digits.py --minimal    # only what retrain_from_sheets.py needs: MNIST and CVL (~55 MB)

| collection | used for | license |
|---|---|---|
| MNIST (ylecun/mnist) | training | CC BY-SA 3.0 |
| EMNIST digits (ernestchu/emnist-digits; NIST SD19, 3,600 writers) | training | NIST: public domain |
| HWD-V1 (leobottaro/handwritten-digits-pack) | training | see the dataset card |
| CVL single digits (zenodo 1492173; TU Wien) | training (train writers) / testing (validation writers) | CC BY-NC 4.0 |
| USPS (flwrlabs/usps), Semeion (UCI), screen-drawn (nguyenminh4099/handwritten-digits) | testing | see the cards |

forms/cells.py finds them in data/digits (or DIGITS_DIR / MNIST_DIR). Re-running skips what is there.
"""
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("DIGITS_DIR") or os.path.join(HERE, "data", "digits")

HF = {  # folder in data/digits: (dataset repo, files, test-only?)
    "mnist": ("ylecun/mnist", ["mnist/train-00000-of-00001.parquet", "mnist/test-00000-of-00001.parquet"], False),
    "ernestchu__emnist-digits": ("ernestchu/emnist-digits", ["data/train-00000-of-00001.parquet",
                                                             "data/test-00000-of-00001.parquet"], False),
    "leobottaro__handwritten-digits-pack": ("leobottaro/handwritten-digits-pack", ["train.parquet"], False),
    "flwrlabs__usps": ("flwrlabs/usps", ["data/test-00000-of-00001.parquet"], True),
    "nguyenminh4099__handwritten-digits": ("nguyenminh4099/handwritten-digits", ["data/test-00000-of-00001.parquet"], True),
}
SEMEION = "https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
CVL = "https://zenodo.org/records/1492173/files/cvl-single-digits-train-validation.zip?download=1"


def fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print("   already there:", os.path.relpath(path, HERE))
        return
    print("   downloading", url)
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    os.replace(tmp, path)


def main():
    test_only = "--test-only" in sys.argv
    minimal = "--minimal" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import hf_hub_download
    for folder, (repo, files, is_test) in HF.items():
        if (test_only and not is_test) or (minimal and folder != "mnist"):
            continue
        print(f"-> {repo}")
        for f in files:
            if os.path.isfile(os.path.join(OUT, folder, f)):
                print("   already there:", f)
                continue
            hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(OUT, folder))
    if not minimal:
        print("-> Semeion (UCI)")
        fetch(SEMEION, os.path.join(OUT, "semeion.data"))
    print("-> CVL single digits (zenodo 1492173, CC BY-NC 4.0)")
    d = os.path.join(OUT, "cvl")
    os.makedirs(d, exist_ok=True)
    z = os.path.join(d, "cvl-single-digits-train-validation.zip")
    def n_png(sub):
        p = os.path.join(d, sub)
        return len([f for f in os.listdir(p) if f.endswith(".png")]) if os.path.isdir(p) else 0
    if n_png("train") < 7000 or n_png("valid") < 7000:              # missing or cut short
        fetch(CVL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        print(f"   cvl/train {n_png('train')}, cvl/valid {n_png('valid')} digits")
    else:
        print("   already there: cvl/train, cvl/valid")
    print("done:", OUT)


if __name__ == "__main__":
    main()
