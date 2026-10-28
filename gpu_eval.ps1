# exam-grader on this computer's NVIDIA GPU: makes .venv with CUDA torch (if it is not there yet),
# downloads the three GPU readers (~25 GB, once) and runs tools/gpu_eval.py - the strong readers
# against the ones in use, and Tal's page end to end. Safe to re-run: what is there is skipped.
# Everything is logged to results\gpu_eval.log; the numbers go to results\gpu_eval.json.
# (Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files as ANSI.)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
$log = Join-Path $root "results\gpu_eval.log"
Set-Content -Path $log -Value ("gpu_eval started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8

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

# ---------------------------------------------------------------- 1. machine
Run "system info" {
    $cs = Get-CimInstance Win32_ComputerSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    "machine : $($cs.Manufacturer) $($cs.Model)"
    "RAM     : {0:N0} GB" -f ($cs.TotalPhysicalMemory / 1GB)
    "CPU     : $($cpu.Name.Trim()) ($($cpu.NumberOfCores) cores / $($cpu.NumberOfLogicalProcessors) threads)"
    Get-CimInstance Win32_VideoController | ForEach-Object { "GPU     : $($_.Name)  (driver $($_.DriverVersion))" }
    $drive = Get-PSDrive -Name $root.Substring(0, 1)
    "disk    : {0:N0} GB free on {1}:" -f ($drive.Free / 1GB), $drive.Name
    "shell   : PowerShell $($PSVersionTable.PSVersion)"
} | Out-Null

$cudaVer = 0
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $smi = (& nvidia-smi 2>&1 | Out-String)
    Add-Content -Path $log -Value $smi -Encoding UTF8
    Write-Host $smi
    if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)") { $cudaVer = [int]$Matches[1] * 10 + [int]$Matches[2] }
}
if ($cudaVer -ge 132) { $flavor = "cu132" }
elseif ($cudaVer -ge 130) { $flavor = "cu130" }
elseif ($cudaVer -ge 126) { $flavor = "cu126" }
else { $flavor = "cpu" }
Log "driver CUDA level: $cudaVer  ->  torch build: $flavor"
if ($flavor -eq "cpu") { Log "WARNING: no NVIDIA driver with CUDA 12.6+ found - the readers will run on the CPU (slow). Update the NVIDIA driver for the GPU." }

# ---------------------------------------------------------------- 2. python + venv
$vpy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    $base = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.13", "3.12", "3.14")) {
            $global:LASTEXITCODE = 1
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $base = @("py", "-$v"); break }
        }
    }
    if (-not $base) {
        foreach ($p in @("C:\Program Files\Python313\python.exe", "C:\Program Files\Python312\python.exe", "C:\Program Files\Python314\python.exe")) {
            if (Test-Path $p) { $base = @($p); break }
        }
    }
    if (-not $base) { Log "ERROR: no Python 3.12-3.14 found"; exit 1 }
    Log ("base python: " + ($base -join " "))
    $exe = $base[0]; $rest = @($base | Select-Object -Skip 1)
    Run "create .venv" { & $exe @rest -m venv .venv } | Out-Null
}
Run "venv python" { & $vpy -c "import sys, platform; print(sys.version); print(platform.platform())" } | Out-Null

# ---------------------------------------------------------------- 3. packages
Run "pip upgrade" { & $vpy -m pip install --upgrade pip --progress-bar off -q } | Out-Null
$torchOk = (& $vpy -c "import torch, torchvision; print(torch.__version__)" 2>$null)
if (-not $torchOk -or $torchOk -notmatch $flavor) {
    $c = Run "torch + torchvision ($flavor)" { & $vpy -m pip install "torch==2.14.1+$flavor" "torchvision==0.29.1+$flavor" --index-url "https://download.pytorch.org/whl/$flavor" --progress-bar off }
    if ($c -ne 0) { Log "ERROR: torch install failed - see above"; exit 1 }
} else { Log "torch already installed: $torchOk" }
$c = Run "requirements" { & $vpy -m pip install -r requirements.txt --progress-bar off -q }
if ($c -ne 0) { Log "ERROR: pip install -r requirements.txt failed - see above"; exit 1 }
if ($flavor -ne "cpu") { Run "bitsandbytes (4-bit loading of the 8B English reader)" { & $vpy -m pip install "bitsandbytes>=0.46" --progress-bar off -q } | Out-Null }
Run "torch / CUDA check" { & $vpy -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); x = torch.randn(512, 512, device='cuda') if torch.cuda.is_available() else None; print(torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0), '| matmul ok:', bool(torch.isfinite(x @ x).all())) if x is not None else print('CPU only')" } | Out-Null

# ---------------------------------------------------------------- 4. models (skipped when present)
# the big English reader (Qwen3-VL-8B, 17 GB) downloads in the background while math and Hebrew run
$dlOut = Join-Path $root "results\gpu_download_vlm.log"
$dlErr = Join-Path $root "results\gpu_download_vlm.err"
Log "background download: qwen3-vl-8b (17 GB) -> results\gpu_download_vlm.log"
$dl = Start-Process -FilePath $vpy -ArgumentList @("download_models.py", "qwen3-vl-8b") -WorkingDirectory $root `
    -RedirectStandardOutput $dlOut -RedirectStandardError $dlErr -NoNewWindow -PassThru
$null = $dl.Handle      # keep the handle, or ExitCode stays empty after the process ends
Run "download the math reader and the Hebrew language model (7.7 GB)" { & $vpy download_models.py uni-mumer-qwen3vl-2b dictalm-3.0-1.7b-base } | Out-Null

# ---------------------------------------------------------------- 5. evaluation - OFFLINE from here
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Log "network access for the models is now OFF (HF_HUB_OFFLINE=1); the background download keeps its own"
Run "GPU eval: machine, math (Uni-MuMER vs Pix2Text), Hebrew in context" { & $vpy tools/gpu_eval.py --parts env math hebrew } | Out-Null

Log "waiting for the background download of qwen3-vl-8b ..."
$dl.WaitForExit()
Get-Content $dlOut -ErrorAction SilentlyContinue | Select-Object -Last 5 | ForEach-Object { Add-Content -Path $log -Value $_ -Encoding UTF8; Write-Host $_ }
Get-Content $dlErr -ErrorAction SilentlyContinue | Select-String -Pattern "Error|error|Traceback" | Select-Object -Last 10 | ForEach-Object { Add-Content -Path $log -Value "$_" -Encoding UTF8; Write-Host "$_" }
Log "background download finished (exit $($dl.ExitCode))"

Run "GPU eval: English (Qwen3-VL-8B vs TrOCR), Tal's page end to end" { & $vpy tools/gpu_eval.py --parts english page } | Out-Null
Run "summary" { & $vpy -c "import json; r = json.load(open('results/gpu_eval.json', encoding='utf-8')); m, h, e, p = (r.get(k, {}) for k in ('math', 'hebrew', 'english', 'page')); print('math    : Uni-MuMER', m.get('mumer_equiv'), '/', m.get('formulas'), '  Pix2Text', m.get('pix2text_equiv'), '/', m.get('formulas'), m.get('error', '')); print('hebrew  : CER in use', h.get('CER_in_use'), '-> in context', h.get('CER_in_context'), h.get('error', '')); print('english : CER', e.get('CER'), '  TrOCR-small same lines', e.get('CER_trocr_small_same_lines'), e.get('error', '')); print('page    :', json.dumps({k: p.get('score', {}).get(k, {}).get('words_right') for k in ('hebrew', 'english')}), p.get('formulas_right_both_weighed'), p.get('error', ''))" } | Out-Null
Log "ALL DONE - results\gpu_eval.json, results\tal_notebook_gpu_p1_read.png"
