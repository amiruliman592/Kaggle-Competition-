#!/usr/bin/env bash
# setup_local.sh - One-time laptop setup (Mac / Linux). CPU only: scoring, ensembles, CSVs, tests.
# Training and prediction run on Colab (notebooks/), so Detectron2 is NOT installed here.
#
# Run from the repo folder:   bash setup_local.sh            (add --skip-tests to skip the ~1.5 min test run)
# Re-running is safe: it reuses .venv and only installs what is missing.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 Python"
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))'; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.10-3.13 not found. Install Python 3.12 (Mac: 'brew install python@3.12', or python.org), then run again."
    exit 1
fi
echo "Using: $PY"

echo "== 2/5 Virtual environment (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip --quiet

echo "== 3/5 PyTorch (CPU build)"
if [ "$(uname)" = "Darwin" ]; then
    "$VPY" -m pip install torch torchvision --quiet          # Mac wheels are CPU/MPS already
else
    "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

echo "== 4/5 Project packages + Jupyter kernel"
"$VPY" -m pip install -r requirements.txt ipykernel --quiet
"$VPY" -m ipykernel install --user --name solar-filament --display-name "Python (solar-filament)"

if [ "${1:-}" = "--skip-tests" ]; then
    echo "== 5/5 Tests skipped"
else
    echo "== 5/5 Tests (~2-3 min)"
    "$VPY" -m pytest -q -p no:warnings tests || { echo "Some tests failed: show the output to Claude."; exit 1; }
fi

echo
echo "Done. Activate the environment in a new terminal with:  source .venv/bin/activate"
echo "In VS Code / Jupyter pick the interpreter '.venv' or the kernel 'Python (solar-filament)'."
