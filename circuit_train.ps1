# exam-grader: train the hand-drawn circuit detector on this computer's GPU. Downloads the two open collections
# of hand-drawn circuits (CGHD ~5 GB, Digitize-HCD 2.1 GB) and the detector's base (RT-DETRv2, 0.08 GB) once,
# converts them (tools/circuit_data.py), trains the detector up to 3 hours (tools/train_circuit_detector.py),
# the stroke model 45 minutes (tools/train_strokes.py), the polarity models (tools/train_polarity.py: which end
# is '+', an arrow's head, a cathode), the label reader (tools/train_labels.py) and the terminal roles
# (tools/train_roles.py: op-amp In+ / In- / Out, transistor legs), measures them on drafters they never saw, and
# reads whole circuits with them (tools/circuit_eval.py). Safe to re-run: downloads and conversion are skipped when
# present.
# Everything is logged to results\circuit_train.log; the numbers go to results\circuit_detector_eval.json.
# (Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files as ANSI.)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
$log = Join-Path $root "results\circuit_train.log"
Set-Content -Path $log -Value ("circuit_train started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Run([string]$title, [scriptblock]$block) {
    Log "=== $title"
    $global:LASTEXITCODE = 0
    & $block 2>&1 | ForEach-Object {
        $s = "$_"
        Write-Host $s
        Add-Content -Path $log -Value $s -Encoding UTF8
    }
    $code = $LASTEXITCODE
    Log "=== $title -> exit $code"
    return $code
}

$vpy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Log "ERROR: .venv is missing - run gpu_eval.cmd first (it sets up Python with CUDA)"; exit 1 }
$drive = Get-PSDrive -Name $root.Substring(0, 1)
Log ("disk: {0:N0} GB free" -f ($drive.Free / 1GB))
if ($drive.Free -lt 12GB) { Log "ERROR: needs about 12 GB free (downloads 7 GB + converted images 2 GB)"; exit 1 }

Run "base model (RT-DETRv2-R18)" { & $vpy download_models.py rtdetr-v2-r18 } | Out-Null
& $vpy tools/circuit_data.py --check | Out-Null
if ($LASTEXITCODE -ne 0) {
    $c = Run "download + convert the circuit collections (32 kinds of parts, dependent sources drawn in)" { & $vpy tools/circuit_data.py }
    if ($c -ne 0) { Log "ERROR: data preparation failed - see above"; exit 1 }
} else { Log "data\circuits already prepared with these classes - skipping" }

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$c = Run "train the detector (up to 3 hours; the best on validation is kept)" { & $vpy tools/train_circuit_detector.py --minutes 180 --size 1024 --batch 4 --workers 4 }
if ($c -ne 0) {
    Log "training stopped with an error - trying again with a smaller batch (2) and input (896 px)"
    $c = Run "train the detector, smaller batch" { & $vpy tools/train_circuit_detector.py --minutes 180 --size 896 --batch 2 --workers 2 }
}
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"
$c = Run "train the stroke model (45 minutes; CGHD's stroke maps, ~0.1 GB, are downloaded first)" { & $vpy tools/train_strokes.py --minutes 45 }
$c = Run "train the polarity models ('+' / arrow head / cathode; about 20 minutes)" { & $vpy tools/train_polarity.py }
$c = Run "train the label reader (names and values: R1, 10k; 15 minutes)" { & $vpy tools/train_labels.py --minutes 15 }
$c = Run "train the terminal roles (op-amp In+ In- Out, transistor legs and kind; about 10 minutes)" { & $vpy tools/train_roles.py }
$c = Run "whole circuits: drafter 1's photos against his LTspice schematics, and Tal's drawing (results\tal_circuit_read.png)" { & $vpy tools/circuit_eval.py }
$c = Run "the same with the true component boxes (the wiring alone)" { & $vpy tools/circuit_eval.py --true-boxes --report results/circuit_eval_true_boxes.json }
Run "summary" { & $vpy -c "import json; r = json.load(open('results/circuit_detector_eval.json', encoding='utf-8')); print('test AP50 (writers never seen):', r['test_AP50_mean'], r['test_AP50_by_source']); [print('  %-14s AP50 %.3f  recall %.3f  (%d true)' % (k, v['AP50'], v['recall@0.5'], v['true'])) for k, v in sorted(r['per_class'].items(), key=lambda kv: -kv[1]['true'])]" } | Out-Null
Run "stroke model summary" { & $vpy -c "import json; r = json.load(open('results/stroke_eval.json', encoding='utf-8')); print('stroke F1 on drafters never seen - learned:', r['test']['F1_learned'], ' classical:', r['test']['F1_classical'])" } | Out-Null
Run "label reader summary" { & $vpy -c "import json; r = json.load(open('results/label_eval.json', encoding='utf-8')); print('labels read exactly on drafters never seen:', r['test'], ' formula reader on the same:', r.get('formula_reader_same_labels'))" } | Out-Null
Run "terminal roles summary" { & $vpy -c "import json; r = json.load(open('results/roles_eval.json', encoding='utf-8')); [print('  %-6s legs right %s, every leg of the part right %s, kind right %s' % (k, v['legs_right'], v['parts_all_legs_right'], v.get('kind_right', '-'))) for k, v in r.items()]" } | Out-Null
Run "polarity summary" { & $vpy -c "import json; r = json.load(open('results/polarity_eval.json', encoding='utf-8')); [print('  %-11s right on drafters never seen: %s of %s' % (k, v['cghd_test_drafters_accuracy'], v['cghd_test_drafters'])) for k, v in r.items()]" } | Out-Null
Log "ALL DONE - models\circuit-detector, stroke-unet, polarity, label-reader, roles; results\circuit_detector_eval.json, stroke_eval.json, polarity_eval.json, label_eval.json, roles_eval.json, circuit_eval.json, tal_circuit_read.png"
