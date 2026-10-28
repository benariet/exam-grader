"""Compare cell readers end to end on the simulated classes, and cell by cell on digit sources.

    python tools/compare_readers.py sims  --model models/cell-reader-v2/a --tag v2a sim_dev sim_test sim_usps ...
    python tools/compare_readers.py cells --models models/cell-reader,models/cell-reader-v2/a --sources usps_test,semeion

sims : reads every simulated student with the given reader, saves <sim>/evaluation_<tag>.json
       (forms/evaluate.py metrics) and prints one line per set.
cells: one clean synthesized cell per digit of each source; accuracy, and for the readings the reader
       is sure of (p >= 0.98) their share and error - the cell-level picture behind the grades.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
COURSE = os.path.join(HERE, "data", "course_elec101")


def run_sims(model, tag, sims, cache_dir):
    from forms.cells import CellReader
    from forms.evaluate import metrics, read_all
    from forms.read_form import FormGrader
    fg = FormGrader(COURSE, CellReader(model))
    for sim in sims:
        pk = os.path.join(cache_dir, f"{sim}_{tag}.pkl") if cache_dir else None
        if pk and os.path.isfile(pk):
            data, sec = pickle.load(open(pk, "rb"))
        else:
            data, sec = read_all(fg, sim)
            if pk:
                pickle.dump((data, sec), open(pk, "wb"))
        o = metrics(fg, data)
        o["seconds_per_student"] = sec
        o["reader"] = model
        o["sweep"] = {str(c): metrics(fg, data, c)["rates"] for c in (0.95, 0.97, 0.98, 0.99, 0.995)}
        from forms.evaluate import sim_roster
        rr = metrics(fg, data, roster=sim_roster(data))["rates"]
        o["with_roster"] = {k: rr[k] for k in ("id_right", "id_wrong_but_accepted", "auto_share", "auto_wrong")}
        json.dump(o, open(os.path.join(COURSE, sim, f"evaluation_{tag}.json"), "w"), ensure_ascii=False, indent=1)
        r = o["rates"]
        print(f"{tag} {sim}: cells {r['cell_accuracy']:.4f} answers {r['answer_read_right']:.4f} auto {r['auto_share']:.4f} "
              f"auto_wrong {r['auto_wrong']} id_right {r['id_right']:.3f} id_wrong {r['id_wrong_but_accepted']} "
              f"| roster: id_right {o['with_roster']['id_right']:.3f} wrong {o['with_roster']['id_wrong_but_accepted']} "
              f"auto {o['with_roster']['auto_share']:.4f} auto_wrong {o['with_roster']['auto_wrong']} "
              f"({sec:.1f}s/student)", flush=True)


def run_cells(models, sources, n, seed=5):
    from forms.cells import CLASSES, CellReader, load_source, synth_cell
    rows = []
    readers = {m: CellReader(m.split("+") if "+" in m else m) for m in models}
    for src in sources:
        x, y = load_source(src)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))[:n]
        cells = [synth_cell(str(y[i]), rng, x[i]) for i in idx]
        want = np.array([CLASSES.index(str(y[i])) for i in idx])
        for m, r in readers.items():
            P = np.concatenate([r.probs(cells[i:i + 512]) for i in range(0, len(cells), 512)])
            pred, conf = P.argmax(1), P.max(1)
            sure = conf >= 0.98
            row = {"source": src, "model": m, "n": len(cells), "acc": float((pred == want).mean()),
                   "sure_share": float(sure.mean()), "sure_error": float((pred[sure] != want[sure]).mean()) if sure.any() else 0.0}
            rows.append(row)
            print(f"{src:12s} {m:40s} acc {row['acc']:.4f}  sure {row['sure_share']:.3f}  err among sure {row['sure_error']:.4f}",
                  flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sims", "cells"])
    ap.add_argument("sims", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--models")
    ap.add_argument("--sources", default="mnist_test,emnist_test,semeion,usps_test,screen_test,cvl_test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--cache", default="/home/claude/cache/reads")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    if a.what == "sims":
        os.makedirs(a.cache, exist_ok=True)
        run_sims(a.model, a.tag, a.sims, a.cache)
    else:
        rows = run_cells(a.models.split(","), a.sources.split(","), a.n)
        if a.out:
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
