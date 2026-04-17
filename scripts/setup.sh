#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up sky segmentation project..."

REQUIRED_CONDA_ENV="sky-seg"
if [ "${CONDA_DEFAULT_ENV:-}" != "$REQUIRED_CONDA_ENV" ]; then
  echo "This project expects the conda env: $REQUIRED_CONDA_ENV" >&2
  echo "Run: conda activate $REQUIRED_CONDA_ENV" >&2
  exit 1
fi

echo "Using env: $CONDA_DEFAULT_ENV"

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p models/pretrained
mkdir -p logs

echo "Optional (Apple Silicon): pip install -r requirements-macos-metal.txt"
echo "Setup complete"
