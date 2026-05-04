#!/usr/bin/env bash

set -e

source "$(pwd)/conda/etc/profile.d/conda.sh"
conda activate comfy

cd ComfyUI

python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header "*" --disable-xsrf