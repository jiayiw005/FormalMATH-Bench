#!/bin/bash
# ============================================================================
# SBATCH Job Script: Goedel-v2-8B Full-Precision LoRA Fine-tuning on OSCAR
#
# Uses DeepSpeed ZeRO-3 to shard the full bf16 model across 8 GPUs.
# Layout: 2 GPUs per model replica × DP×4 = 8 GPUs total.
#
# Submit with:  sbatch scripts/oscar_train_lora.sh
# Monitor with: squeue -u $USER
# ============================================================================

#SBATCH --job-name=goedel-grind-lora
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:8
#SBATCH -n 8
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -o /users/%u/scratch/grind_project/logs/train_lora_%j.out
#SBATCH -e /users/%u/scratch/grind_project/logs/train_lora_%j.err
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export SCRATCH_DIR="/users/${USER}/scratch"
export PROJECT_DIR="${SCRATCH_DIR}/grind_project"
export VENV_DIR="${PROJECT_DIR}/venv"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

# Activate virtual environment
module load python/3.11.11-5e66
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null)}"
source "${VENV_DIR}/bin/activate"

# Print environment info
echo "============================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${SLURMD_NODENAME}"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM:        $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Python:      $(python3 --version)"
echo "  PyTorch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:  $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  Project:     ${PROJECT_DIR}"
echo "  Mode:        Full-precision LoRA (bf16 + ZeRO-3)"
echo "============================================"
echo ""

cd "${PROJECT_DIR}" || exit 1

# ---------------------------------------------------------------------------
# Training (multi-GPU via accelerate + DeepSpeed ZeRO-3)
# ---------------------------------------------------------------------------
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file scripts/ds_config_zero3.json \
    finetune_lora.py \
    --dataset    data/goedel_sft/sft_grind_only.json \
    --model_id   Goedel-LM/Goedel-Prover-V2-8B \
    --output_dir "${SCRATCH_DIR}/checkpoints/goedel-v2-lora" \
    --lora_r       128 \
    --lora_alpha  256 \
    --epochs        4 \
    --lr         3e-4 \
    --batch_size    1 \
    --gradient_accumulation 2 \
    --max_seq_length 6144 \
    --bf16 \
    --gradient_checkpointing

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints: ${SCRATCH_DIR}/checkpoints/goedel-v2-lora/"
echo "============================================"
