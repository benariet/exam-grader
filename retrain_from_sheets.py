"""Teach the answer-cell reader your students' handwriting - one command, from scanned copy sheets
(forms/collect.py: print data/collect/sheets_all_versions.pdf, let people fill it, scan to PDF).

    python retrain_from_sheets.py scans/*.pdf
    python retrain_from_sheets.py scans/*.pdf --epochs 1 --no-install     # just measure

1. reads every scan: the labelled cells of every copied number (a row is kept only if it was copied
   cell by cell from the first cell) -> data/collect/real/<scan>/ (each person = his pages in a row)
2. holds out ~20 % of the PEOPLE, fine-tunes the reader in use (models/cell-reader-v2/main) on the others'
   cells (mixed with synthesized cells, so it keeps what it knew) and measures both readers on the
   held-out people's cells - handwriting the new reader has not seen
3. only if the new reader is better there (more cells right, no more sure-but-wrong cells), fine-tunes once
   more on everyone and installs it: models/cell-reader-v2/mine, ensemble.json -> [mine, b]; the previous
   ensemble.json is kept as ensemble_before_<time>.json (copy it back to undo)

Needs torch and the digit collections for the synthesized part (python download_digits.py).
Report: results/retrain_report.json.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
V2 = os.path.join(HERE, "models", "cell-reader-v2")
REAL = os.path.join(HERE, "data", "collect", "real")


def cell_scores(reader, crops, labels):
    P = np.concatenate([reader.probs(crops[i:i + 512]) for i in range(0, len(crops), 512)])
    y = np.array(labels)
    pred, conf = P.argmax(1), P.max(1)
    sure = conf >= 0.98
    return {"cells": len(y), "right": float((pred == y).mean()), "sure": float(sure.mean()),
            "sure_wrong": int(((pred != y) & sure).sum()),
            "digits_right": float((pred[y < 10] == y[y < 10]).mean()) if (y < 10).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+", help="scanned copy sheets (PDF or images); many people per PDF is fine")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sources", default="mnist_train,cvl_fit", help="digit collections for the synthesized part")
    ap.add_argument("--cache", default=os.path.join(HERE, "data", "collect", "base_cells"),
                    help="where the synthesized part is kept between runs")
    ap.add_argument("--test-share", type=float, default=0.2)
    ap.add_argument("--no-install", action="store_true", help="measure only")
    ap.add_argument("--real-dir", default=REAL)
    ap.add_argument("--models", default=V2, help="the readers in use (folder with ensemble.json, main/, b/)")
    a = ap.parse_args()
    V = a.models

    from forms.cells import CellReader, train
    from forms.collect import load_real, read_filled
    from forms.read_form import load_images

    t0 = time.time()
    for p in a.scans:                                           # 1. scans -> labelled cells
        name = os.path.splitext(os.path.basename(p))[0]
        st = read_filled(load_images([p]), name, out_root=a.real_dir)
        print(f"{name}: {st['people']} people, {st['rows_kept']}/{st['rows']} rows kept, {st['cells']} cells, "
              f"{st['pages_not_found']} pages not recognized", flush=True)
    crops, labels, who = load_real(a.real_dir, with_people=True)
    people = sorted(set(who))
    print(f"collected: {len(crops)} cells from {len(people)} people", flush=True)
    if len(people) < 3:
        sys.exit("need at least 3 people's sheets (some are held out to check the new reader)")

    rng = np.random.default_rng(0)                               # 2. held-out people
    order = list(rng.permutation(people))
    n_test = max(1, int(round(len(people) * a.test_share)))
    test_people, train_people = set(order[:n_test]), set(order[n_test:])
    te = [i for i, w in enumerate(who) if w in test_people]
    te_crops, te_labels = [crops[i] for i in te], [labels[i] for i in te]
    work = os.path.join(HERE, "models", "cell-reader-trained")
    trial = os.path.join(work, "sheets_trial")
    shutil.rmtree(trial, ignore_errors=True)
    common = dict(epochs=a.epochs, sources=tuple(a.sources.split(",")), val="mnist_test", n_val=5000, seed=5,
                  real=a.real_dir, real_repeat=20, init=os.path.join(V, "main"), lr=1e-3, cache=a.cache)
    train(out=trial, real_people=train_people, **common)
    readers = {"in use (main + b)": CellReader(V),
               "fine-tuned": CellReader(trial),
               "fine-tuned + b": CellReader([trial, os.path.join(V, "b")])}
    report = {"people": len(people), "held_out_people": sorted(test_people), "held_out_cells": len(te),
              "readers": {k: cell_scores(r, te_crops, te_labels) for k, r in readers.items()}}
    print(f"\nheld-out people ({n_test}, {len(te)} cells):", flush=True)
    for k, v in report["readers"].items():
        print(f"  {k:20s} right {v['right']:.2%}  digits {v['digits_right']:.2%}  sure {v['sure']:.1%}  "
              f"sure-but-wrong {v['sure_wrong']}", flush=True)

    base = report["readers"]["in use (main + b)"]               # 3. better? -> train on all, install
    best = max(("fine-tuned", "fine-tuned + b"), key=lambda k: (report["readers"][k]["right"], -report["readers"][k]["sure_wrong"]))
    b = report["readers"][best]
    better = b["right"] > base["right"] and b["sure_wrong"] <= base["sure_wrong"]
    report["better"], report["best"] = bool(better), best
    if better and not a.no_install:
        final = os.path.join(work, "sheets_all")
        shutil.rmtree(final, ignore_errors=True)
        train(out=final, real_people=None, **common)
        mine = os.path.join(V, "mine")
        shutil.rmtree(mine, ignore_errors=True)
        os.makedirs(mine)
        for f in ("model.onnx", "model.pt", "meta.json"):
            shutil.copy(os.path.join(final, f), os.path.join(mine, f))
        ens = os.path.join(V, "ensemble.json")
        backup = os.path.join(V, time.strftime("ensemble_before_%Y%m%d_%H%M.json"))
        shutil.copy(ens, backup)
        members = ["mine", "b"] if best == "fine-tuned + b" else ["mine"]
        json.dump({"members": members, "note": f"fine-tuned on {len(people)} people's copy sheets "
                   f"({time.strftime('%Y-%m-%d')}); previous: {os.path.basename(backup)}"}, open(ens, "w"), indent=1)
        report["installed"] = members
        print(f"\ninstalled: {members} (previous ensemble kept as {os.path.basename(backup)})", flush=True)
    elif not better:
        print("\nnot better on the held-out people - nothing installed", flush=True)
    report["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(report, open(os.path.join(HERE, "results", "retrain_report.json"), "w"), indent=1, ensure_ascii=False)
    print(f"report: results/retrain_report.json ({report['minutes']} min)")


if __name__ == "__main__":
    main()
