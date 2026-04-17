#!/usr/bin/env bash
# install.sh — Sky-segmentation project setup wizard
# Supports: macOS (arm64, x86_64) and Linux (x86_64, aarch64)
# Creates the "sky-seg" conda environment with Python 3.11 and installs all
# dependencies using the correct TensorFlow variant for your platform.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="sky-seg"
PYTHON_VERSION="3.11"
MINIFORGE_DIR="${HOME}/miniforge3"
MINIFORGE_BASE="https://github.com/conda-forge/miniforge/releases/latest/download"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_info()    { echo "[install] $*"; }
_warn()    { echo "[install] WARNING: $*" >&2; }
_error()   { echo "[install] ERROR: $*" >&2; exit 1; }
_ask_yn()  {
    # $1 = prompt, $2 = default (y/n)
    local prompt="$1" default="${2:-y}" reply
    local hint="[Y/n]"; [[ "$default" == "n" ]] && hint="[y/N]"
    read -r -p "$prompt $hint: " reply
    reply="${reply:-$default}"
    [[ "${reply,,}" == "y" ]]
}

# ---------------------------------------------------------------------------
# 1. Detect OS and architecture
# ---------------------------------------------------------------------------
_detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin)
            case "$arch" in
                arm64)  PLATFORM="macos-arm64"  ;;
                x86_64) PLATFORM="macos-x86_64" ;;
                *)      _error "Unsupported macOS architecture: $arch" ;;
            esac
            ;;
        Linux)
            case "$arch" in
                x86_64)         PLATFORM="linux-x86_64"  ;;
                aarch64|arm64)  PLATFORM="linux-aarch64" ;;
                *)              _error "Unsupported Linux architecture: $arch" ;;
            esac
            ;;
        *)
            _error "Unsupported OS: $os (only macOS and Linux are supported)"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# 2. Print banner + confirm platform
# ---------------------------------------------------------------------------
_confirm_platform() {
    echo ""
    echo "========================================"
    echo "  Sky Segmentation — Setup Wizard"
    echo "========================================"
    echo ""
    _info "Detected platform: $PLATFORM"
    echo ""

    if ! _ask_yn "Is this correct?"; then
        echo ""
        echo "Select your platform:"
        echo "  1) macOS arm64  (Apple Silicon — M1/M2/M3/M4)"
        echo "  2) macOS x86_64 (Intel Mac)"
        echo "  3) Linux x86_64 (Intel/AMD)"
        echo "  4) Linux aarch64 (ARM64)"
        read -r -p "Enter choice [1-4]: " choice
        case "$choice" in
            1) PLATFORM="macos-arm64"  ;;
            2) PLATFORM="macos-x86_64" ;;
            3) PLATFORM="linux-x86_64" ;;
            4) PLATFORM="linux-aarch64" ;;
            *) _error "Invalid choice: $choice" ;;
        esac
        _info "Platform set to: $PLATFORM"
    fi

    # macOS x86 end-of-life warning
    if [[ "$PLATFORM" == "macos-x86_64" ]]; then
        echo ""
        echo "  *** WARNING ***"
        echo "  TensorFlow 2.16 was the last release that supports macOS x86_64."
        echo "  This wizard will install tensorflow==2.16.2 (the latest supported version)."
        echo "  For newer TF versions, use an Apple Silicon Mac or a Linux machine."
        echo ""
        _ask_yn "Continue anyway?" || exit 0
    fi
}

# ---------------------------------------------------------------------------
# 3. Ask about GPU on Linux
# ---------------------------------------------------------------------------
_ask_gpu() {
    INSTALL_CUDA_TF="false"
    if [[ "$PLATFORM" == linux-* ]]; then
        echo ""
        if _ask_yn "Do you have an NVIDIA GPU (with drivers >= 525.60.13)?"; then
            INSTALL_CUDA_TF="true"
            _info "Will install tensorflow[and-cuda] (bundles CUDA 12.3 + cuDNN 8.9.7)"
        else
            _info "Will install tensorflow (CPU only)"
        fi
    fi
}

# ---------------------------------------------------------------------------
# 4. Install Miniforge3 if conda is not available
# ---------------------------------------------------------------------------
_ensure_conda() {
    # Check for conda already on PATH or in common install locations
    if command -v conda &>/dev/null; then
        _info "conda found: $(command -v conda)"
        return
    fi

    # Check default Miniforge location
    if [[ -f "${MINIFORGE_DIR}/bin/conda" ]]; then
        _info "Miniforge found at ${MINIFORGE_DIR}; initialising shell..."
        # shellcheck source=/dev/null
        source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"
        return
    fi

    echo ""
    _info "conda is not installed. Miniforge3 will be downloaded and installed."
    _info "Install location: ${MINIFORGE_DIR}"
    _ask_yn "Proceed with Miniforge3 installation?" || exit 0

    local installer_name
    case "$PLATFORM" in
        macos-arm64)   installer_name="Miniforge3-MacOSX-arm64.sh"   ;;
        macos-x86_64)  installer_name="Miniforge3-MacOSX-x86_64.sh"  ;;
        linux-x86_64)  installer_name="Miniforge3-Linux-x86_64.sh"   ;;
        linux-aarch64) installer_name="Miniforge3-Linux-aarch64.sh"  ;;
    esac

    local installer_url="${MINIFORGE_BASE}/${installer_name}"
    local installer_path="/tmp/${installer_name}"

    _info "Downloading ${installer_name}..."
    if command -v curl &>/dev/null; then
        curl -fsSL -o "$installer_path" "$installer_url"
    elif command -v wget &>/dev/null; then
        wget -q -O "$installer_path" "$installer_url"
    else
        _error "Neither curl nor wget is available. Please install one and re-run."
    fi

    _info "Running Miniforge3 installer (silent, into ${MINIFORGE_DIR})..."
    bash "$installer_path" -b -p "$MINIFORGE_DIR"
    rm -f "$installer_path"

    # shellcheck source=/dev/null
    source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

    # Add to shell rc file so future sessions find it
    local rcfile="${HOME}/.bashrc"
    [[ "$PLATFORM" == macos-* ]] && rcfile="${HOME}/.zshrc"
    if ! grep -qF "miniforge3/etc/profile.d/conda.sh" "$rcfile" 2>/dev/null; then
        {
            echo ""
            echo "# >>> conda initialize >>>"
            echo "# !! Contents within this block are managed by 'conda init' !!"
            echo "source \"${MINIFORGE_DIR}/etc/profile.d/conda.sh\""
            echo "# <<< conda initialize <<<"
        } >> "$rcfile"
        _info "Added conda initialisation to ${rcfile}"
    fi

    _info "Miniforge3 installed successfully."
}

# ---------------------------------------------------------------------------
# 5. Create (or reuse) the conda environment
# ---------------------------------------------------------------------------
_create_env() {
    echo ""
    if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        _warn "Conda environment '${ENV_NAME}' already exists."
        if _ask_yn "Re-use existing environment (skip creation)?"; then
            _info "Using existing environment '${ENV_NAME}'."
            return
        fi
        _info "Removing existing environment '${ENV_NAME}'..."
        conda env remove -n "$ENV_NAME" -y
    fi

    _info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    # Do NOT install TF via conda — always via pip (per official TF docs)
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y
    _info "Environment created."
}

# ---------------------------------------------------------------------------
# 6. Run pip installs inside the environment
# ---------------------------------------------------------------------------
_install_packages() {
    echo ""
    _info "Upgrading pip..."
    conda run -n "$ENV_NAME" python -m pip install --upgrade pip

    # TensorFlow — platform-specific variant
    echo ""
    case "$PLATFORM" in
        macos-arm64)
            _info "Installing TensorFlow for macOS arm64..."
            conda run -n "$ENV_NAME" pip install tensorflow
            _info "Installing tensorflow-metal for Metal GPU acceleration..."
            conda run -n "$ENV_NAME" pip install tensorflow-metal
            ;;
        macos-x86_64)
            _info "Installing TensorFlow 2.16.2 for macOS x86_64 (last supported version)..."
            conda run -n "$ENV_NAME" pip install "tensorflow==2.16.2"
            ;;
        linux-x86_64)
            if [[ "$INSTALL_CUDA_TF" == "true" ]]; then
                _info "Installing tensorflow[and-cuda] (GPU + bundled CUDA 12.3/cuDNN 8.9.7)..."
                conda run -n "$ENV_NAME" pip install "tensorflow[and-cuda]"
            else
                _info "Installing TensorFlow (CPU only)..."
                conda run -n "$ENV_NAME" pip install tensorflow
            fi
            ;;
        linux-aarch64)
            _info "Installing TensorFlow for Linux aarch64 (CPU only, via AWS)..."
            conda run -n "$ENV_NAME" pip install tensorflow
            ;;
    esac

    # Remaining project dependencies (TF excluded from this file)
    echo ""
    _info "Installing project dependencies from requirements.txt..."
    conda run -n "$ENV_NAME" pip install -r "${PROJECT_ROOT}/requirements.txt"
}

# ---------------------------------------------------------------------------
# 7. Create project directories
# ---------------------------------------------------------------------------
_create_dirs() {
    mkdir -p "${PROJECT_ROOT}/models/pretrained"
    mkdir -p "${PROJECT_ROOT}/logs"
    _info "Project directories ready."
}

# ---------------------------------------------------------------------------
# 8. Verify TensorFlow installation
# ---------------------------------------------------------------------------
_verify() {
    echo ""
    _info "Verifying TensorFlow installation..."
    local result
    result=$(conda run -n "$ENV_NAME" python -c \
        "import tensorflow as tf; print('TensorFlow', tf.__version__, '— OK:', tf.reduce_sum(tf.random.normal([1000, 1000])).numpy())" \
        2>&1) && {
        _info "$result"
    } || {
        _warn "Verification produced an error:"
        echo "$result"
        echo ""
        _warn "TensorFlow may still work; check the output above."
    }
}

# ---------------------------------------------------------------------------
# 9. Print success summary
# ---------------------------------------------------------------------------
_print_summary() {
    echo ""
    echo "========================================"
    echo "  Setup complete!"
    echo "========================================"
    echo ""
    echo "  Platform  : $PLATFORM"
    echo "  Env name  : $ENV_NAME"
    echo "  Python    : $PYTHON_VERSION"
    echo ""
    echo "  To activate the environment:"
    echo "    conda activate $ENV_NAME"
    echo ""
    echo "  To start training:"
    echo "    conda activate $ENV_NAME"
    echo "    ./scripts/train.sh --config config.yaml --dataset <name>"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_detect_platform
_confirm_platform
_ask_gpu
_ensure_conda
_create_env
_install_packages
_create_dirs
_verify
_print_summary
