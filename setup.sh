#!/bin/bash

set -e

ENV_NAME="temporal"
PYTHON_VERSION="3.10"
CONDA_DIR="${HOME}/miniconda3"

if ! command -v conda &> /dev/null && [ ! -x "${CONDA_DIR}/bin/conda" ]; then
    echo "Installing Miniconda to ${CONDA_DIR}..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
    rm /tmp/miniconda.sh
    "${CONDA_DIR}/bin/conda" init bash
fi

if ! command -v conda &> /dev/null; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Newer Miniconda requires accepting the default channel ToS before creating envs
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "Creating conda environment: ${ENV_NAME}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "Activating environment..."
conda activate "${ENV_NAME}"

echo "Installing PyTorch..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

conda install -y ipykernel
conda install -y -c conda-forge ffmpeg zstd   # zstd: unpacking the LeWM datasets

echo "Installing the project's Python dependencies..."
pip install -r "$(dirname "$(readlink -f "$0")")/lewm-cus/requirements.txt"
echo "Environment '${ENV_NAME}' is ready."

sudo apt install -y nvtop
sudo apt install -y tmux
sudo apt install -y htop

echo "Done!"
