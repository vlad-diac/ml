#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-models/best_model.keras}"
IMAGE="${1:-}"

if [ -z "$IMAGE" ]; then
  echo "Usage: $0 path/to/image.jpg" >&2
  exit 1
fi

python src/inference.py \
  --model "$MODEL" \
  --image "$IMAGE" \
  --config config.yaml
