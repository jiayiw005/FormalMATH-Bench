#!/bin/bash
# ============================================================================
# OSCAR Setup for Goedel-v2-8B QLoRA Fine-tuning & Generation & Verification
#
# Run ONCE on OSCAR login node:
#   bash scripts/oscar_setup.sh
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
export SCRATCH_DIR="/users/${USER}/scratch"
export PROJECT_DIR="${SCRATCH_DIR}/grind_project"
export VENV_DIR="${PROJECT_DIR}/venv"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export MODEL_ID="Goedel-LM/Goedel-Prover-V2-8B"

echo "============================================"
echo "  OSCAR Setup for Goedel-v2 Training & Gen & Verification"
echo "============================================"
echo "  User:        ${USER}"
echo "  Project:     ${PROJECT_DIR}"
echo "  HF Cache:    ${HF_HOME}"
echo ""

# ---------------------------------------------------------------------------
# 1. Create directories
# ---------------------------------------------------------------------------
echo "[1/4] Creating directories..."
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/scripts"
mkdir -p "${PROJECT_DIR}/runs"
mkdir -p "${PROJECT_DIR}/FormalMATH-Bench"
mkdir -p "${HF_HOME}"
echo "  Done."

# ---------------------------------------------------------------------------
# 2. Create virtual environment (Python 3.11 via module)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Creating virtual environment..."

# vllm requires Python 3.10+; system Python is 3.9, so load 3.11 module
module load python/3.11.11-5e66

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created: ${VENV_DIR}"
else
    echo "  Exists: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "  Python: $(python3 --version)"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip wheel setuptools

# --- Training dependencies ---
# PyTorch with CUDA (check available CUDA with: ls /usr/local/cuda*)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.44.0
pip install datasets>=2.20.0
pip install peft>=0.12.0
pip install trl>=0.9.0
pip install bitsandbytes>=0.43.0
pip install accelerate>=0.33.0
pip install scipy
pip install deepspeed

# --- Verification dependencies ---
pip install pexpect

# --- Generation dependencies ---
pip install vllm
pip install numpy
pip install jsonlines
pip install scikit-learn
pip install sentencepiece
pip install python-dotenv

# flash-attn requires a GPU node to compile — install in the SLURM job instead
echo ""
echo "  NOTE: flash-attn will be installed at job start (needs GPU to compile)."
echo ""
echo "  Installed packages:"
pip list 2>/dev/null | grep -iE "torch|transformers|peft|trl|bitsandbytes|accelerate|vllm|jsonlines"

# ---------------------------------------------------------------------------
# 4. Pre-download model weights
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Pre-downloading model weights to ${HF_HOME}..."
HF_HOME="${HF_HOME}" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', cache_dir='${HF_HOME}')
print('Model downloaded.')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Verify your files are in place:"
echo "  ${PROJECT_DIR}/finetune_qlora.py"
echo "  ${PROJECT_DIR}/data/goedel_sft/sft_train_grind.json"
echo "  ${PROJECT_DIR}/FormalMATH-Bench/generate_answers.py"
echo "  ${PROJECT_DIR}/FormalMATH-Bench/FormalMATH_lite.json"
echo ""
echo "Then submit:"
echo "  cd ${PROJECT_DIR} && sbatch scripts/oscar_train.sh   # training"
echo "  cd ${PROJECT_DIR} && sbatch scripts/gen.job           # generation"
