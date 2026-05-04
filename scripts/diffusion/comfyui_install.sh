#!/usr/bin/env bash

set -e

echo "=== STEP 0: System deps (required for CUDA builds) ==="

apt update && apt install -y \
    build-essential \
    git \
    wget \
    ninja-build \
    cmake \
    libgl1

echo "=== STEP 1: Install Miniconda locally ==="

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$(pwd)/conda"

source "$(pwd)/conda/etc/profile.d/conda.sh"

echo "=== STEP 2: Create environment ==="

conda create -y -n comfy python=3.10
conda activate comfy

echo "=== STEP 3: Install PyTorch (CUDA 12.1 safer for builds) ==="

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== STEP 4: Clone ComfyUI ==="

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

echo "=== STEP 5: Install ComfyUI Manager ==="

cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

echo "=== STEP 6: Install Hunyuan3D nodes ==="

git clone https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git

echo "=== STEP 7: Install Python deps ==="

pip install -r ComfyUI-Hunyuan3d-2-1/requirements.txt

echo "=== STEP 8: Install rendering stack ==="

# nvdiffrast (NVIDIA differentiable rasterizer)
pip install git+https://github.com/NVlabs/nvdiffrast

# tiny-cuda-nn (sometimes required by pipelines)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "=== STEP 9: Build custom rasterizer ==="

cd ComfyUI-Hunyuan3d-2-1

# try standard build if provided
if [ -f "setup.py" ]; then
    pip install -e .
fi

# fallback: compile manually if folder exists
if [ -d "custom_rasterizer" ]; then
    cd custom_rasterizer
    python setup.py install || echo "Custom rasterizer build failed (may be optional depending on workflow)"
    cd ..
fi

echo "=== STEP 10: Optional: install Kaolin (can be tricky) ==="

pip install kaolin || echo "Kaolin install skipped (optional)"

echo "=== SETUP COMPLETE ==="