#!/bin/bash
# ============================================================================
# SBATCH Job Script: Generate proofs for top30k eval set (10k sample)
#                    using the LoRA fine-tuned Goedel model
#
# Data is pre-converted by scripts/convert_top30k.py (run locally first).
# Merges the LoRA adapter if not already merged, then runs generation.
#
# Submit with:  sbatch scripts/gen_lora_top30k.job
# Monitor with: squeue -u $USER
# ============================================================================

#SBATCH --job-name=gen-lora-top30k
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:8
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH --requeue
#SBATCH -o /users/%u/scratch/grind_project/logs/gen-lora-top30k_%j.out
#SBATCH -e /users/%u/scratch/grind_project/logs/gen-lora-top30k_%j.err
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
export SCRATCH_DIR="/users/${USER}/scratch"
export PROJECT_DIR="${SCRATCH_DIR}/grind_project"
export VENV_DIR="${PROJECT_DIR}/venv"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false

BASE_MODEL="Goedel-LM/Goedel-Prover-V2-8B"
LORA_DIR="${SCRATCH_DIR}/checkpoints/goedel-v2-lora"

# Auto-detect the latest checkpoint: prefer "final/", else newest checkpoint-*
if [ -d "${LORA_DIR}/final" ]; then
    CHECKPOINT_DIR="${LORA_DIR}/final"
else
    CHECKPOINT_DIR=$(ls -d "${LORA_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi

if [ -z "${CHECKPOINT_DIR}" ] || [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No checkpoint found in ${LORA_DIR}"
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT_DIR}"

MERGED_MODEL_DIR="${SCRATCH_DIR}/checkpoints/goedel-v2-lora-merged"
INPUT_FILE="data/goedel_sft/top30k_eval_10k.json"
OUTPUT_FILE="./runs/gen_goedel-v2-lora_top30k-eval-10k_pass1.json"

# Create directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/runs"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
module load python/3.11.11-5e66
module load cuda/12.1.1 2>/dev/null || module load cuda 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null)}"
source "${VENV_DIR}/bin/activate"

# Print environment info
echo "============================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         ${SLURMD_NODENAME}"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Python:       $(python3 --version)"
echo "  Project:      ${PROJECT_DIR}"
echo "  Checkpoint:   ${CHECKPOINT_DIR}"
echo "  Merged:       ${MERGED_MODEL_DIR}"
echo "  Input:        ${INPUT_FILE}"
echo "  Output:       ${OUTPUT_FILE}"
echo "============================================"
echo ""

cd "${PROJECT_DIR}" || exit 1

# ---------------------------------------------------------------------------
# Step 1: Merge LoRA adapter into base model (if not already merged)
# ---------------------------------------------------------------------------
if [ ! -f "${MERGED_MODEL_DIR}/config.json" ]; then
    echo '=== MERGING LORA ADAPTER INTO BASE MODEL ==='
    python3 -c "
import sys, os, traceback
try:
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    base_model_id = '${BASE_MODEL}'
    adapter_dir   = '${CHECKPOINT_DIR}'
    output_dir    = '${MERGED_MODEL_DIR}'

    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading base model: {base_model_id}')
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir='${HF_HOME}',
    )

    print(f'Loading adapter from: {adapter_dir}')
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print('Merging adapter weights...')
    model = model.merge_and_unload()

    # Move to CPU before saving to avoid GPU memory fragmentation issues
    print('Moving merged model to CPU for saving...')
    model = model.cpu()

    print(f'Saving merged model to: {output_dir}')
    model.save_pretrained(output_dir, max_shard_size='4GB', safe_serialization=True)
    print(f'  Model saved. Files: {os.listdir(output_dir)}')

    print(f'Loading tokenizer from adapter: {adapter_dir}')
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, cache_dir='${HF_HOME}')
    tokenizer.save_pretrained(output_dir)

    # Verify
    if not os.path.isfile(os.path.join(output_dir, 'config.json')):
        print('ERROR: config.json was NOT created!', file=sys.stderr)
        sys.exit(1)

    print(f'Merge complete! Files in output: {os.listdir(output_dir)}')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"
    echo '=== MERGE DONE ==='
else
    echo "=== MERGED MODEL ALREADY EXISTS, SKIPPING MERGE ==="
fi

# ---------------------------------------------------------------------------
# Step 2: Verify input data exists
# ---------------------------------------------------------------------------
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "       Run locally first:"
    echo "         python scripts/convert_top30k.py --sample 10000 \\"
    echo "           --exclude data/goedel_sft/sft_grind_only.json \\"
    echo "           --output data/goedel_sft/top30k_eval_10k.json"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Generate proofs (pass@1)
# ---------------------------------------------------------------------------
echo ""
echo "=== STARTING GENERATION ==="
python FormalMATH-Bench/generate_answers.py \
  --model "${MERGED_MODEL_DIR}" \
  --input_file "${INPUT_FILE}" \
  --generated_file "${OUTPUT_FILE}" \
  --n 1 \
  --nums_answer 1 \
  --tp 2 \
  --dp 4

echo ""
echo "============================================"
echo "  Generation complete!"
echo "  Output: ${PROJECT_DIR}/${OUTPUT_FILE}"
echo "============================================"
