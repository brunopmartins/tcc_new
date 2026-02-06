#!/usr/bin/env bash
# =============================================================================
# Kinship Classification Project - Linux Setup Script
# =============================================================================
# Sets up a freshly installed Linux machine (Ubuntu/Debian) to run this project.
# Supports both Docker-based and native Python workflows, with NVIDIA and AMD GPUs.
#
# Usage:
#   chmod +x setup.sh
#   sudo ./setup.sh [OPTIONS]
#
# Options:
#   --gpu nvidia       Install NVIDIA CUDA drivers and toolkit (default: auto-detect)
#   --gpu amd          Install AMD ROCm drivers
#   --gpu none         CPU-only setup (no GPU drivers)
#   --docker-only      Only set up Docker (skip native Python environment)
#   --native-only      Only set up native Python environment (skip Docker)
#   --no-venv          Skip Python virtual environment creation
#   --data-dir PATH    Set dataset directory (default: ./data)
#   --help             Show this help message
#
# Examples:
#   sudo ./setup.sh                          # Auto-detect GPU, full setup
#   sudo ./setup.sh --gpu nvidia             # NVIDIA GPU, full setup
#   sudo ./setup.sh --gpu amd --docker-only  # AMD GPU, Docker only
#   sudo ./setup.sh --gpu none --native-only # CPU-only, native Python only
# =============================================================================

set -euo pipefail

# ---- Color output helpers ---------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---- Default configuration --------------------------------------------------
GPU_PLATFORM="auto"
SETUP_DOCKER=true
SETUP_NATIVE=true
CREATE_VENV=true
DATA_DIR=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_PLATFORM="$2"
            shift 2
            ;;
        --docker-only)
            SETUP_DOCKER=true
            SETUP_NATIVE=false
            shift
            ;;
        --native-only)
            SETUP_DOCKER=false
            SETUP_NATIVE=true
            shift
            ;;
        --no-venv)
            CREATE_VENV=false
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --help)
            head -n 30 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# ---- Root check -------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)."
    exit 1
fi

SUDO_USER_NAME="${SUDO_USER:-$USER}"

# ---- GPU auto-detection -----------------------------------------------------
detect_gpu() {
    if lspci 2>/dev/null | grep -iq 'nvidia'; then
        echo "nvidia"
    elif lspci 2>/dev/null | grep -iq 'amd.*radeon\|amd.*instinct\|advanced micro devices.*display'; then
        echo "amd"
    else
        echo "none"
    fi
}

if [[ "$GPU_PLATFORM" == "auto" ]]; then
    info "Auto-detecting GPU platform..."
    GPU_PLATFORM=$(detect_gpu)
    if [[ "$GPU_PLATFORM" == "none" ]]; then
        warn "No supported GPU detected. Setting up for CPU-only."
    else
        success "Detected GPU platform: ${GPU_PLATFORM}"
    fi
fi

if [[ "$GPU_PLATFORM" != "nvidia" && "$GPU_PLATFORM" != "amd" && "$GPU_PLATFORM" != "none" ]]; then
    error "Invalid GPU platform: '${GPU_PLATFORM}'. Use 'nvidia', 'amd', or 'none'."
    exit 1
fi

# ---- Print setup summary ----------------------------------------------------
echo ""
echo "=============================================="
echo " KINSHIP CLASSIFICATION - LINUX SETUP"
echo "=============================================="
echo " GPU platform:     ${GPU_PLATFORM}"
echo " Docker setup:     ${SETUP_DOCKER}"
echo " Native setup:     ${SETUP_NATIVE}"
echo " Virtual env:      ${CREATE_VENV}"
echo " Project dir:      ${SCRIPT_DIR}"
echo "=============================================="
echo ""

# ---- Step 1: System packages ------------------------------------------------
info "Installing system packages..."

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pciutils

# Set python3.10 as the default python3 if not already
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 2>/dev/null || true

success "System packages installed."

# ---- Step 2: GPU drivers ----------------------------------------------------
if [[ "$GPU_PLATFORM" == "nvidia" ]]; then
    info "Setting up NVIDIA CUDA drivers and toolkit..."

    # Install NVIDIA driver (if not already present)
    if ! command -v nvidia-smi &>/dev/null; then
        info "Installing NVIDIA drivers..."
        add-apt-repository -y ppa:graphics-drivers/ppa
        apt-get update
        apt-get install -y nvidia-driver-535
        warn "NVIDIA drivers installed. A REBOOT is required before GPU will be available."
    else
        success "NVIDIA drivers already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    fi

    # Install CUDA toolkit
    if ! command -v nvcc &>/dev/null; then
        info "Installing CUDA toolkit 12.1..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update
        apt-get install -y cuda-toolkit-12-1
        rm -f /tmp/cuda-keyring.deb

        # Add CUDA to PATH for the invoking user
        CUDA_PROFILE_LINE='export PATH=/usr/local/cuda/bin:$PATH'
        CUDA_LD_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}'
        PROFILE_FILE="/home/${SUDO_USER_NAME}/.bashrc"
        if [[ -f "$PROFILE_FILE" ]]; then
            grep -qxF "$CUDA_PROFILE_LINE" "$PROFILE_FILE" || echo "$CUDA_PROFILE_LINE" >> "$PROFILE_FILE"
            grep -qxF "$CUDA_LD_LINE" "$PROFILE_FILE" || echo "$CUDA_LD_LINE" >> "$PROFILE_FILE"
        fi
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
        success "CUDA toolkit 12.1 installed."
    else
        success "CUDA toolkit already installed: $(nvcc --version | grep release | awk '{print $NF}')"
    fi

elif [[ "$GPU_PLATFORM" == "amd" ]]; then
    info "Setting up AMD ROCm drivers..."

    if ! command -v rocm-smi &>/dev/null; then
        info "Installing AMD ROCm 5.7..."
        apt-get install -y linux-headers-"$(uname -r)" linux-modules-extra-"$(uname -r)" 2>/dev/null || true

        wget -q https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb -O /tmp/amdgpu-install.deb
        apt-get install -y /tmp/amdgpu-install.deb
        amdgpu-install -y --usecase=rocm,hip --no-dkms
        rm -f /tmp/amdgpu-install.deb

        # Add user to video and render groups
        usermod -aG video "${SUDO_USER_NAME}"
        usermod -aG render "${SUDO_USER_NAME}"
        warn "AMD ROCm installed. A REBOOT is required. User '${SUDO_USER_NAME}' added to 'video' and 'render' groups."
    else
        success "AMD ROCm already installed: $(rocm-smi --showdriverversion 2>/dev/null | grep -oP 'Driver version:\s*\K.*' || echo 'unknown')"
    fi

else
    info "Skipping GPU driver installation (CPU-only mode)."
fi

success "GPU driver setup complete."

# ---- Step 3: Docker ---------------------------------------------------------
if [[ "$SETUP_DOCKER" == true ]]; then
    info "Setting up Docker..."

    if ! command -v docker &>/dev/null; then
        info "Installing Docker Engine..."
        curl -fsSL https://get.docker.com | sh
        systemctl enable docker
        systemctl start docker
        usermod -aG docker "${SUDO_USER_NAME}"
        success "Docker installed. User '${SUDO_USER_NAME}' added to 'docker' group."
    else
        success "Docker already installed: $(docker --version)"
    fi

    # Install Docker Compose plugin (v2)
    if ! docker compose version &>/dev/null; then
        info "Installing Docker Compose plugin..."
        apt-get install -y docker-compose-plugin
        success "Docker Compose plugin installed."
    else
        success "Docker Compose already installed: $(docker compose version --short 2>/dev/null)"
    fi

    # NVIDIA Container Toolkit (for Docker GPU passthrough)
    if [[ "$GPU_PLATFORM" == "nvidia" ]]; then
        if ! dpkg -l | grep -q nvidia-container-toolkit; then
            info "Installing NVIDIA Container Toolkit..."
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            apt-get update
            apt-get install -y nvidia-container-toolkit
            nvidia-ctk runtime configure --runtime=docker
            systemctl restart docker
            success "NVIDIA Container Toolkit installed and configured."
        else
            success "NVIDIA Container Toolkit already installed."
        fi
    fi

    # Build Docker images
    info "Building project Docker images..."
    cd "${SCRIPT_DIR}"

    if [[ "$GPU_PLATFORM" == "nvidia" || "$GPU_PLATFORM" == "none" ]]; then
        info "Building NVIDIA Docker image (kinship-nvidia)..."
        docker build -f Dockerfile.nvidia -t kinship-nvidia . && \
            success "Docker image 'kinship-nvidia' built." || \
            warn "Failed to build NVIDIA Docker image. You can retry with: docker build -f Dockerfile.nvidia -t kinship-nvidia ."
    fi

    if [[ "$GPU_PLATFORM" == "amd" ]]; then
        info "Building AMD Docker image (kinship-amd)..."
        docker build -f Dockerfile.amd -t kinship-amd . && \
            success "Docker image 'kinship-amd' built." || \
            warn "Failed to build AMD Docker image. You can retry with: docker build -f Dockerfile.amd -t kinship-amd ."
    fi

    success "Docker setup complete."
else
    info "Skipping Docker setup (--native-only specified)."
fi

# ---- Step 4: Native Python environment --------------------------------------
if [[ "$SETUP_NATIVE" == true ]]; then
    info "Setting up native Python environment..."

    if [[ "$CREATE_VENV" == true ]]; then
        if [[ ! -d "$VENV_DIR" ]]; then
            info "Creating Python virtual environment at ${VENV_DIR}..."
            sudo -u "${SUDO_USER_NAME}" python3.10 -m venv "${VENV_DIR}"
            success "Virtual environment created."
        else
            success "Virtual environment already exists at ${VENV_DIR}."
        fi
        PIP="${VENV_DIR}/bin/pip"
        PYTHON="${VENV_DIR}/bin/python"
    else
        PIP="pip3"
        PYTHON="python3"
    fi

    info "Upgrading pip..."
    sudo -u "${SUDO_USER_NAME}" "$PIP" install --upgrade pip setuptools wheel

    # Install PyTorch with appropriate backend
    info "Installing PyTorch..."
    if [[ "$GPU_PLATFORM" == "nvidia" ]]; then
        sudo -u "${SUDO_USER_NAME}" "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$GPU_PLATFORM" == "amd" ]]; then
        sudo -u "${SUDO_USER_NAME}" "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    else
        sudo -u "${SUDO_USER_NAME}" "$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install remaining dependencies
    info "Installing project dependencies..."
    sudo -u "${SUDO_USER_NAME}" "$PIP" install -r "${SCRIPT_DIR}/models/requirements.txt"

    # Verify PyTorch installation
    info "Verifying PyTorch installation..."
    sudo -u "${SUDO_USER_NAME}" "$PYTHON" -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU device:      {torch.cuda.get_device_name(0)}')
elif hasattr(torch.version, 'hip') and torch.version.hip:
    print(f'  ROCm/HIP:        {torch.version.hip}')
else:
    print('  Running in CPU-only mode')
" || warn "PyTorch verification failed. GPU drivers may require a reboot."

    success "Native Python environment setup complete."

    if [[ "$CREATE_VENV" == true ]]; then
        info "Activate the virtual environment with: source ${VENV_DIR}/bin/activate"
    fi
else
    info "Skipping native Python setup (--docker-only specified)."
fi

# ---- Step 5: Project directory structure ------------------------------------
info "Setting up project directories..."

# Data directories
DATA_ROOT="${DATA_DIR:-${SCRIPT_DIR}/data}"
mkdir -p "${DATA_ROOT}/FIW/FIDs"
mkdir -p "${DATA_ROOT}/KinFaceW-I/images"
mkdir -p "${DATA_ROOT}/KinFaceW-II/images"

# Output directories for each model
for model_dir in 01_age_synthesis_comparison 02_vit_facor_crossattn 03_convnext_vit_hybrid 04_unified_kinship_model; do
    mkdir -p "${SCRIPT_DIR}/models/${model_dir}/output/checkpoints"
    mkdir -p "${SCRIPT_DIR}/models/${model_dir}/output/results"
    mkdir -p "${SCRIPT_DIR}/models/${model_dir}/output/logs"
    mkdir -p "${SCRIPT_DIR}/models/${model_dir}/checkpoints_nvidia"
    mkdir -p "${SCRIPT_DIR}/models/${model_dir}/checkpoints_amd"
done

# Fix ownership
chown -R "${SUDO_USER_NAME}:${SUDO_USER_NAME}" "${SCRIPT_DIR}/models" "${DATA_ROOT}" 2>/dev/null || true

success "Project directories created."

# ---- Step 6: Make scripts executable ----------------------------------------
info "Making run scripts executable..."
chmod +x "${SCRIPT_DIR}/models/run_models.sh" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/models/run_all_models_nvidia.sh" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/models/run_all_models_amd.sh" 2>/dev/null || true

success "Scripts are executable."

# ---- Done -------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e " ${GREEN}SETUP COMPLETE${NC}"
echo "=============================================="
echo ""
echo " GPU platform:   ${GPU_PLATFORM}"
echo " Project root:   ${SCRIPT_DIR}"
echo " Data directory:  ${DATA_ROOT}"
if [[ "$CREATE_VENV" == true && "$SETUP_NATIVE" == true ]]; then
echo " Virtual env:    ${VENV_DIR}"
fi
echo ""
echo " Next steps:"
echo "  1. Place datasets in: ${DATA_ROOT}/"
echo "     - FIW/          (Families in the Wild)"
echo "     - KinFaceW-I/   (KinFaceW-I)"
echo "     - KinFaceW-II/  (KinFaceW-II)"
echo ""
if [[ "$SETUP_NATIVE" == true ]]; then
    if [[ "$CREATE_VENV" == true ]]; then
echo "  2. Activate the environment:"
echo "       source ${VENV_DIR}/bin/activate"
echo ""
echo "  3. Run training (native):"
    else
echo "  2. Run training (native):"
    fi
echo "       cd ${SCRIPT_DIR}/models"
echo "       ./run_models.sh"
echo ""
fi
if [[ "$SETUP_DOCKER" == true ]]; then
echo "  Docker workflow:"
echo "       cd ${SCRIPT_DIR}/models/01_age_synthesis_comparison"
if [[ "$GPU_PLATFORM" == "nvidia" ]]; then
echo "       docker compose -f docker-compose.nvidia.yml up"
elif [[ "$GPU_PLATFORM" == "amd" ]]; then
echo "       docker compose -f docker-compose.amd.yml up"
else
echo "       docker compose -f docker-compose.nvidia.yml up"
fi
echo ""
fi

# Reboot notice
NEEDS_REBOOT=false
if [[ "$GPU_PLATFORM" == "nvidia" ]] && ! command -v nvidia-smi &>/dev/null; then
    NEEDS_REBOOT=true
fi
if [[ "$GPU_PLATFORM" == "amd" ]] && ! command -v rocm-smi &>/dev/null; then
    NEEDS_REBOOT=true
fi

if [[ "$NEEDS_REBOOT" == true ]]; then
    echo -e " ${YELLOW}NOTE: A reboot is required for GPU drivers to take effect.${NC}"
    echo "       Run: sudo reboot"
    echo ""
fi

# Group change notice
if [[ "$SETUP_DOCKER" == true ]] || [[ "$GPU_PLATFORM" == "amd" ]]; then
    echo -e " ${YELLOW}NOTE: Log out and back in for group changes to take effect.${NC}"
    echo "       Or run: newgrp docker"
    echo ""
fi

echo "=============================================="
