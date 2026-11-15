"""One exam's structured checks run in order from a config file - the course tools (truth table, expressions, words,
multiple choice, final values, the drawn circuit's judging) and the report gathering them - with the GPU readers'
part done on the server when a part needs it (only the cut pieces go there, deleted after; the exams stay here).

    python tools/run_exam.py exams/moedC.json [--parts table_q1a,choices_q2] [--server-only] [--dry-run]
    python tools/run_exam.py exams/moedC.json --judge-only [--out <results>] [--apply <the marked sheet>]
        (everything judged again from what was read before - nothing cut, nothing sent to the server, the circuits not
         read again, the expressions' readings taken as they were; with --apply, a human's marks on the decisions
         sheet taken first: a whole Moed brought up to date after the sheet is marked, on this computer alone)

The config (JSON):
    {"exams": "<the scans' folder>", "booklet": "booklet/moedC", "grades": "<the lecturer's sheet>" (optional),
     "out": "<the results' folder>",
     "server": {"host": "bgu", "runs": "eg-runs/expr", "sbatch": "eg-runs/expr/exprread.sbatch"} (optional;
                "vlm": "<another vision model's folder there>", "bits": 8 or 4 - the same for every part),
     "parts": [{"name": "table_q1a", "tool": "truth_table", "args": {"page": 5, "cols": "2,4,5", ...}},
               {"name": "exprs_q1a", "tool": "expr_box", "args": {...},
                "server": {"readers": "mumer,vlm", "prompt": "boolean", "labels": "Vx|Y1|Y2", "views": "1|0.75|1.33"}},
               ...],
     "report": {"sure": "2.2=5,2.3=5"},
     "decisions": {"file": "<decisions.json>", "sheet": "<decisions.xlsx>", "values": ["value_q1f"]} (optional)}
A part's args are the tool's own options ('table_cols' -> --table-cols; true -> the flag; a list -> the option
again for each); "{out}" in a value is the results' folder. A part with "server": its pieces cut first
(--crops-only), read on the server (tools/read_crops.py by Slurm), the readings fetched and the server's copies
deleted, then the part judged with them (--second; words_box, truth_table: --readings). Several things asked for on
one place (value_area: "label_text": "f1=the_function_of_F1;tf1=..."): read once, by the first part; the others say
"server": {"share": "<that part>"} and their own "reading_id".
"decisions": what a human marked on tools/decide_sheet.py's sheet (kept with the exams) given to words_box,
value_area, waveform, choice_marks, truth_table and expr_box; after the parts, the sheet made anew of what is still
undecided (the words parts, the values named, every drawn waveform or curve, multiple-choice mark, truth table,
expression and drawn circuit).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = {"truth_table", "expr_box", "words_box", "choice_marks", "value_area", "circuit_key", "read_exam_circuits",
         "waveform"}
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
    if srv.get("vlm"):                                 # (another vision model for every part: its folder on the
        extra += f" --vlm {srv['vlm']}"                #  server, and its weights' bits - 8 / 4 for a 32B model)
    if srv.get("bits"):
        extra += f" --bits {srv['bits']}"
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
    ap.add_argument("--judge-only", action="store_true", help="judge again from what was read before: no crops cut, "
                    "nothing sent to the server (a part's readings: <out>/<part>_crops/readings.json; a part without "
                    "them skipped), the circuits not read again, the expressions' readings reused")
    ap.add_argument("--out", default="", help="the results' folder instead of the config's")
    ap.add_argument("--apply", default="", help="the decisions sheet a human marked: its marks taken into the "
                    "config's decisions file first (tools/decide_sheet.py apply)")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    out = a.out or cfg["out"]
    os.makedirs(out, exist_ok=True)
    if a.apply:                                        # (the human's marks first: every part then takes them)
        print("== the marks on the sheet", flush=True)
        dec_file = (cfg.get("decisions") or {}).get("file")
        if not dec_file:
            raise SystemExit("--apply: the config names no decisions file")
        run([sys.executable, os.path.join(HERE, "tools", "decide_sheet.py"), "apply", a.apply, "--out", dec_file],
            a.dry_run)
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
        if tool == "circuit_key":                      # (what it leaves undecided cut out of the exams: the sheet's)
            common["exams"] = cfg["exams"]
        if dec and tool in ("words_box", "value_area", "waveform", "choice_marks", "truth_table", "expr_box",
                            "circuit_key"):    # (a human's marks on what the checks left: kept with
            common["decisions"] = dec                      #  the exams, tools/decide_sheet.py)
        if a.judge_only and tool == "read_exam_circuits" and os.path.isdir(pout):
            print("  (read before: kept)", flush=True)     # (hours of reading; its judging is circuit_key's)
            continue
        if a.judge_only and tool == "expr_box":       # (Pix2Text's readings as they were: an exam judged before
            args["reuse"] = pout                       #  takes the marks given since)
        if tool == "value_area" and part.get("server"):   # (several things read on one place: a value read as
            root = part["server"].get("share") or name     #  another's key is not called wrong)
            sib = [p_["args"]["key"] for p_ in cfg["parts"] if p_ is not part and p_["tool"] == "value_area" and
                   "key" in p_.get("args", {}) and ((p_.get("server") or {}).get("share") or p_["name"]) == root and
                   p_.get("server")]
            if sib:
                args["sibling_keys"] = "|".join(sib)
        if (part.get("server") or {}).get("share"):     # (read with another part's place: its readings, no reading
            args["second"] = os.path.join(out, part["server"]["share"] + "_crops", "readings.json")   # again)
        elif part.get("server") and a.judge_only:     # (what was read before)
            readings = os.path.join(out, name + "_crops", "readings.json")
            if not os.path.isfile(readings) and not a.dry_run:
                print(f"  (no readings in {os.path.dirname(readings)}: skipped)", flush=True)
                continue
            args["readings" if tool in ("words_box", "truth_table", "choice_marks") else "second"] = readings
        elif part.get("server") and cfg.get("server"):
            crops = os.path.join(out, name + "_crops")
            cmd = cli(tool, {**common, **args, "crops_only": crops}, out)
            run(cmd[:2] + head + cmd[2:], a.dry_run)
            readings = server_read(part, crops, cfg, a.dry_run)
            args["readings" if tool in ("words_box", "truth_table", "choice_marks") else "second"] = readings
        if tool in POSITIONAL:                         # (its folder of results another part wrote - and its own)
            where = str(args.pop(POSITIONAL[tool])).replace("{out}", out)
            folders[-1] = where
            cmd = cli(tool, {**common, **args}, out)
            cmd.insert(2, where)
        else:
            cmd = cli(tool, {**common, **args, "out": pout}, out)
            cmd[2:2] = head
        run(cmd, a.dry_run)
    rep = cfg.get("report")
    if rep is not None:
        print("== report", flush=True)
        cmd = [sys.executable, os.path.join(HERE, "tools", "exam_report.py"), "--runs"] + \
            [f for f in folders if os.path.isdir(f) or a.dry_run] + ["--out", os.path.join(out, "report")]
        if cfg.get("grades"):
            cmd += ["--grades", cfg["grades"]]
        if rep.get("sure"):
            cmd += ["--sure", rep["sure"]]
        if rep.get("rubric"):                          # (the exam's points per item: what is decided of them)
            cmd += ["--xlsx", "--rubric", rep["rubric"]] + (["--compare", "all", "--points", rep["points"]]
                                                           if rep.get("points") else [])
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
                share = (part.get("server") or {}).get("share")     # (read on another part's place: its crops)
                crops = os.path.join(out, share + "_crops") if share else crops
                cmd += ["--value", f, "--value-crops", crops, "--value-readings", os.path.join(crops, "readings.json"),
                        "--value-id", str(part.get("args", {}).get("reading_id", ""))]
            elif part["tool"] == "waveform":
                cmd += ["--wave", f]
            elif part["tool"] == "choice_marks":
                cmd += ["--choice", f]
            elif part["tool"] == "truth_table":
                cmd += ["--table", f]
            elif part["tool"] == "expr_box":
                cmd += ["--expr", f, "--expr-crops", crops]
            elif part["tool"] == "circuit_key":
                cmd += ["--circuit", f]
        if (cfg.get("report") or {}).get("rubric"):    # (the items worth most first; an answer deciding its item
            cmd += ["--rubric", cfg["report"]["rubric"],     #  alone before them - the report's, made just before)
                    "--report", os.path.join(out, "report", "by_exam.xlsx")]
        run(cmd + ["--out", dec["sheet"]], a.dry_run)


if __name__ == "__main__":
    main()
