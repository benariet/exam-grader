# exam-grader setup for Windows - creates .venv, installs everything, downloads models + data
# into the project, then runs tests, evaluations and Hebrew training OFFLINE.  Safe to re-run.
# Everything is logged to setup.log next to this script.
# (Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files as ANSI.)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::DeprecationWarning"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$log = Join-Path $root "setup.log"
Set-Content -Path $log -Value ("exam-grader setup started " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -Encoding UTF8

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
        foreach ($p in @("C:\Program Files\Python313\python.exe", "C:\Program Files\Python314\python.exe")) {
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
$c = Run "requirements" { & $vpy -m pip install -r requirements.txt --progress-bar off }
if ($c -ne 0) { Log "ERROR: pip install -r requirements.txt failed - see above"; exit 1 }
Run "torch check" { & $vpy -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" } | Out-Null

# ---------------------------------------------------------------- 4. models + data (skipped if present)
Run "download models" { & $vpy download_models.py } | Out-Null
Run "prepare data" { & $vpy prepare_data.py } | Out-Null

# ---------------------------------------------------------------- 5. everything below runs OFFLINE
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Log "network access for models is now OFF (HF_HUB_OFFLINE=1)"
Run "unit tests" { & $vpy -m pytest -q -p no:cacheprovider tests } | Out-Null
Run "formula eval - real handwriting" { & $vpy eval_real.py } | Out-Null

$hebModel = Join-Path $root "models\hebrew-crnn\model.onnx"
if (-not (Test-Path $hebModel)) {
    # continue from the best shipped model (saves only if it gets better on held-out writers)
    $init = @()
    foreach ($d in @("hebrew-crnn-cpu", "hebrew-crnn-cpu5ep")) {
        $pt = Join-Path $root "models\$d\model.pt"
        if (Test-Path $pt) { $init = @("--init", $pt, "--lr", "3e-4", "--pct-start", "0.05"); break }
    }
    if ($flavor -eq "cpu") { $targs = @("--epochs", "8", "--minutes", "50") + $init }
    else { $targs = @("--epochs", "40", "--minutes", "50") + $init }
    Run "train hebrew recognizer ($($targs -join ' '))" { & $vpy train_hebrew.py @targs } | Out-Null
} else { Log "models\hebrew-crnn already exists - skipping training (delete that folder to retrain)" }
Run "hebrew eval - 300 test lines, unseen writers" { & $vpy eval_hebrew.py --n 300 } | Out-Null
Run "english eval - 150 IAM test lines, unseen writers" { & $vpy eval_english.py --n 150 } | Out-Null
Run "mixed Hebrew + math demo" { & $vpy demo_mixed.py } | Out-Null
Run "mixed Hebrew + math evaluation - 84 fresh lines" { & $vpy eval_mixed.py --data mixed_eval3 } | Out-Null
Run "mixed Hebrew + English + math evaluation - 120 fresh lines" { & $vpy eval_mixed.py --data mixed_eval_en2 } | Out-Null
Log "ALL DONE - results are in the results\ folder"
