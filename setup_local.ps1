# setup_local.ps1 - One-time laptop setup (Windows). CPU only: scoring, ensembles, CSVs, tests.
# Training and prediction run on Colab (notebooks/), so Detectron2 is NOT installed here.
#
# Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File setup_local.ps1
# Re-running is safe: it reuses .venv and only installs what is missing.
# Add -SkipTests to skip the ~1.5 min test run at the end.

param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    # Python 3.10-3.13 (the versions this project is tested with); the py launcher first, then python on PATH
    foreach ($cmd in @("py -3.13", "py -3.12", "py -3.11", "py -3.10", "python")) {
        $exe, $arg = $cmd.Split(" ")
        try {
            $v = & $exe $arg -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(1[0-3])$') { return @($exe, $arg) }
        } catch {}
    }
    return $null
}

Write-Host "== 1/5 Python" -ForegroundColor Cyan
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10-3.13 not found. Installing Python 3.12 with winget..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    Write-Host "Python installed. CLOSE this terminal, open a new one and run this script again." -ForegroundColor Yellow
    exit 0
}
Write-Host "Using: $($py -join ' ')"

Write-Host "== 2/5 Virtual environment (.venv)" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $py[0] $py[1] -m venv .venv
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet

Write-Host "== 3/5 PyTorch (CPU build, ~200 MB)" -ForegroundColor Cyan
& $venvPy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

Write-Host "== 4/5 Project packages + Jupyter kernel" -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt ipykernel --quiet
& $venvPy -m ipykernel install --user --name solar-filament --display-name "Python (solar-filament)"

if ($SkipTests) {
    Write-Host "== 5/5 Tests skipped" -ForegroundColor Cyan
} else {
    Write-Host "== 5/5 Tests (~2-3 min)" -ForegroundColor Cyan
    & $venvPy -m pytest -q -p no:warnings tests
    if ($LASTEXITCODE -ne 0) { Write-Host "Some tests failed: show the output to Claude." -ForegroundColor Red; exit 1 }
}

Write-Host ""
Write-Host "Done. Activate the environment in a new terminal with:  .venv\Scripts\activate" -ForegroundColor Green
Write-Host "In VS Code / Jupyter pick the interpreter '.venv' or the kernel 'Python (solar-filament)'."
