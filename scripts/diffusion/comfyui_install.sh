#!/usr/bin/env bash

set -e

# ---------------------------------------------------------------------------
# Usage: ./comfyui_install.sh [--continue <step>]
#   --continue <step>   Skip all steps before <step> (0-8). Conda env must
#                       already exist when skipping steps 1 or 2.
# ---------------------------------------------------------------------------

FROM_STEP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --continue)
            if [[ -z "$2" || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --continue requires a numeric step argument (0-8)." >&2
                exit 1
            fi
            FROM_STEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--continue <step>]" >&2
            exit 1
            ;;
    esac
done

CONDA_INSTALL_DIR="$(pwd)/conda"

run_from() {
    local step=$1
    [ "$step" -ge "$FROM_STEP" ]
}

# ---------------------------------------------------------------------------

if run_from 0; then
    echo "=== STEP 0: System deps (required for CUDA builds) ==="

    apt update && apt install -y \
        build-essential \
        git \
        wget \
        ninja-build \
        cmake \
        libgl1
fi

# ---------------------------------------------------------------------------

if run_from 1; then
    echo "=== STEP 1: Install Miniconda locally ==="

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

    echo ""
    echo ">>> Miniconda installer downloaded."
    echo ">>> Please run the following command to complete the installation:"
    echo ">>>   bash miniconda.sh -p \"${CONDA_INSTALL_DIR}\""
    echo ">>> Accept the license, install to the path above, and do NOT let the"
    echo ">>> installer initialize conda (answer 'no' to the init prompt)."
    echo ""

    while true; do
        echo ">>> Once finished, press ENTER here to continue (or type 'quit' to abort)..."
        read -r _input
        if [ "${_input}" = "quit" ]; then
            echo "Aborted by user."
            exit 1
        fi
        if [ -f "${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh" ]; then
            echo ">>> Conda installation detected. Continuing..."
            break
        fi
        echo ">>> Conda installation not found at '${CONDA_INSTALL_DIR}'."
        echo ">>> Make sure you installed to that exact path and try again."
    done
fi

source "${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# ---------------------------------------------------------------------------

if run_from 2; then
    echo "=== STEP 2: Create environment ==="

    conda create -y -n comfy python=3.10
fi

conda activate comfy

# ---------------------------------------------------------------------------

if run_from 3; then
    echo "=== STEP 3: Install PyTorch (CUDA 12.1 safer for builds) ==="

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# ---------------------------------------------------------------------------

if run_from 4; then
    echo "=== STEP 4: Clone ComfyUI ==="

    if [ ! -d "ComfyUI" ]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git
    else
        echo ">>> ComfyUI already exists, skipping clone."
    fi
    cd ComfyUI
    pip install -r requirements.txt
else
    cd ComfyUI
fi

# ---------------------------------------------------------------------------

if run_from 5; then
    echo "=== STEP 5: Install ComfyUI Manager ==="

    if [ ! -d "custom_nodes/ComfyUI-Manager" ]; then
        cd custom_nodes
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git
        cd ..
    else
        echo ">>> ComfyUI-Manager already exists, skipping clone."
    fi
else
    true
fi

# ---------------------------------------------------------------------------

if run_from 6; then
    echo "=== STEP 6: Install Hunyuan3D nodes ==="

    if [ ! -d "custom_nodes/ComfyUI-Hunyuan3d-2-1" ]; then
        cd custom_nodes
        git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git
        cd ..
    else
        echo ">>> ComfyUI-Hunyuan3d-2-1 already exists, skipping clone."
    fi
fi

# ---------------------------------------------------------------------------

if run_from 7; then
    echo "=== STEP 7: Install Python deps ==="

    pip install -r custom_nodes/ComfyUI-Hunyuan3d-2-1/requirements.txt
fi

# ---------------------------------------------------------------------------

if run_from 8; then
    echo "=== STEP 8: Build custom rasterizer ==="

    cd custom_nodes/ComfyUI-Hunyuan3d-2-1

    cd hy3dpaint/custom_rasterizer
    pip install --no-build-isolation -e .
    cd ../..

    cd hy3dpaint/DifferentiableRenderer
    bash compile_mesh_painter.sh
    cd ../..

    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

    cd ../..
fi

# ---------------------------------------------------------------------------

echo "=== SETUP COMPLETE ==="
