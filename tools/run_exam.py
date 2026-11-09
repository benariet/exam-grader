"""One exam's structured checks run in order from a config file - the course tools (truth table, expressions, words,
multiple choice, final values, the drawn circuit's judging) and the report gathering them - with the GPU readers'
part done on the server when a part needs it (only the cut pieces go there, deleted after; the exams stay here).

    python tools/run_exam.py exams/moedC.json [--parts table_q1a,choices_q2] [--server-only] [--dry-run]

The config (JSON):
    {"exams": "<the scans' folder>", "booklet": "booklet/moedC", "grades": "<the lecturer's sheet>" (optional),
     "out": "<the results' folder>",
     "server": {"host": "bgu", "runs": "eg-runs/expr", "sbatch": "eg-runs/expr/exprread.sbatch"} (optional),
     "parts": [{"name": "table_q1a", "tool": "truth_table", "args": {"page": 5, "cols": "2,4,5", ...}},
               {"name": "exprs_q1a", "tool": "expr_box", "args": {...},
                "server": {"readers": "mumer,vlm", "prompt": "boolean", "labels": "Vx|Y1|Y2", "views": "1|0.75|1.33"}},
               ...],
     "report": {"sure": "2.2=5,2.3=5"},
     "decisions": {"file": "<decisions.json>", "sheet": "<decisions.xlsx>", "values": ["value_q1f"]} (optional)}
A part's args are the tool's own options ('table_cols' -> --table-cols; true -> the flag; a list -> the option
again for each); "{out}" in a value is the results' folder. A part with "server": its pieces cut first
(--crops-only), read on the server (tools/read_crops.py by Slurm), the readings fetched and the server's copies
deleted, then the part judged with them (--second; words_box, truth_table: --readings).
"decisions": what a human marked on tools/decide_sheet.py's sheet (kept with the exams) given to words_box and
value_area; after the parts, the sheet made anew of what is still undecided (the words parts, the values named).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = {"truth_table", "expr_box", "words_box", "choice_marks", "value_area", "circuit_key", "read_exam_circuits"}
POSITIONAL = {"circuit_key": "circuits"}               # (a tool whose first argument is a folder of results)


def cli(tool, args, out):
    """a part's args -> the tool's command line"""
    cmd = [sys.executable, os.path.join(HERE, "tools", f"{tool}.py")]
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        vals = v if isinstance(v, list) else [v]
        for x in vals:
            if x is True:
                cmd.append(flag)
            elif x is False or x is None:
                continue
            else:
                cmd += [flag, str(x).replace("{out}", out)]
    return cmd


def run(cmd, dry):
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True)


def server_read(part, crops, cfg, dry):
    """the part's pieces read on the server: copied there, read by Slurm, the readings fetched, the server's copies
    deleted (and that checked) -> the readings' local path"""
    srv = cfg["server"]
    host, runs = srv["host"], srv["runs"]
    name = part["name"]
    local = os.path.join(crops, "readings.json")
    if os.path.isfile(local):
        return local
    sp = part["server"]
    files = sorted(glob.glob(os.path.join(crops, sp.get("files", "*.png"))))
    if not files and not dry:
        raise SystemExit(f"{name}: no pieces in {crops}")
    remote = f"{runs}/{name}"
    extra = f"--prompt {sp.get('prompt', 'math')}"
    if sp.get("labels"):
        extra += f" --labels {sp['labels']}"
    if sp.get("label_text"):
        extra += f" --label-text {sp['label_text'].replace(' ', '_')}"
    if sp.get("views"):
        extra += f" --views {sp['views'].replace(',', '|')}"
    run(["ssh", host, f"mkdir -p {remote}"], dry)
    run(["scp", "-q"] + files + [f"{host}:{remote}/"], dry)
    out = f"{runs}/{name}_readings.json"
    readers = sp.get("readers", "mumer,vlm").replace(",", "|")    # (no commas in a value: --export splits at them)
    export = f"ALL,IN=$HOME/{remote},OUT=$HOME/{out},READERS={readers},EXTRA={extra}"
    job = ["ssh", host, "export SLURM_CONF=/etc/slurm/slurm.conf PATH=/opt/slurm/bin:$PATH; cd " + runs + " && sbatch "
           f"--parsable --export=\"{export}\" $HOME/{srv['sbatch']}"]
    print("  $ " + " ".join(job), flush=True)
    if not dry:
        jid = subprocess.run(job, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
        while True:                                    # (the job running: a look a minute)
            time.sleep(60)
            q = subprocess.run(["ssh", host, "export SLURM_CONF=/etc/slurm/slurm.conf PATH=/opt/slurm/bin:$PATH; "
                                f"squeue -h -j {jid}"], capture_output=True, text=True).stdout.strip()
            if not q:
                break
    run(["scp", "-q", f"{host}:{out}", local], dry)
    run(["ssh", host, f"rm -rf {remote} {out}"], dry)
    if not dry:                                        # (deleted: checked apart - the exams' pieces stay here only)
        left = subprocess.run(["ssh", host, f"ls {remote} {out} 2>/dev/null"], capture_output=True, text=True).stdout
        if left.strip():
            raise SystemExit(f"{name}: the server's copies were not deleted ({remote}, {out}) - delete them by hand")
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--parts", default="", help="only these parts (their names, comma-separated)")
    ap.add_argument("--dry-run", action="store_true", help="the commands printed, none run")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    want = {x for x in a.parts.split(",") if x}
    folders = []
    for part in cfg["parts"]:
        name, tool = part["name"], part["tool"]
        if tool not in TOOLS:
            raise SystemExit(f"{name}: no tool {tool}")
        pout = os.path.join(out, name)
        folders.append(pout)
        if want and name not in want:
            continue
        print(f"== {name} ({tool})", flush=True)
        args = dict(part.get("args", {}))
        head = [cfg["exams"]] if tool not in POSITIONAL else []
        common = {"booklet": cfg.get("booklet")} if tool not in POSITIONAL else {}
        if cfg.get("grades") and tool not in ("read_exam_circuits",):
            common["grades"] = cfg["grades"]
        dec = (cfg.get("decisions") or {}).get("file")
        if dec and tool in ("words_box", "value_area"):    # (a human's marks on what the checks left: kept with
            common["decisions"] = dec                      #  the exams, tools/decide_sheet.py)
        if part.get("server") and cfg.get("server"):
            crops = os.path.join(out, name + "_crops")
            cmd = cli(tool, {**common, **args, "crops_only": crops}, out)
            run(cmd[:2] + head + cmd[2:], a.dry_run)
            readings = server_read(part, crops, cfg, a.dry_run)
            args["readings" if tool in ("words_box", "truth_table") else "second"] = readings
        if tool in POSITIONAL:                         # (its folder of results another part wrote - and its own)
            where = str(args.pop(POSITIONAL[tool])).replace("{out}", out)
            folders[-1] = where
            cmd = cli(tool, {**common, **args}, out)
            cmd.insert(2, where)
        else:
            cmd = cli(tool, {**common, **args, "out": pout}, out)
            cmd[2:2] = head
        run(cmd, a.dry_run)
    dec = cfg.get("decisions") or {}
    if dec.get("sheet") and not a.parts:                # (the answers still undecided, for a human to mark once)
        print("== decisions sheet", flush=True)
        cmd = [sys.executable, os.path.join(HERE, "tools", "decide_sheet.py"), "make"]
        for part, f in zip(cfg["parts"], folders):
            crops = os.path.join(out, part["name"] + "_crops")
            if part["tool"] == "words_box":
                cmd += ["--words", f, "--words-crops", crops]
            elif part["tool"] == "value_area" and part["name"] in dec.get("values", []):
                cmd += ["--value", f, "--value-crops", crops, "--value-readings", os.path.join(crops, "readings.json")]
        run(cmd + ["--out", dec["sheet"]], a.dry_run)
    rep = cfg.get("report")
    if rep is not None:
        print("== report", flush=True)
        cmd = [sys.executable, os.path.join(HERE, "tools", "exam_report.py"), "--runs"] + \
            [f for f in folders if os.path.isdir(f) or a.dry_run] + ["--out", os.path.join(out, "report")]
        if cfg.get("grades"):
            cmd += ["--grades", cfg["grades"]]
        if rep.get("sure"):
            cmd += ["--sure", rep["sure"]]
        run(cmd, a.dry_run)


if __name__ == "__main__":
    main()
