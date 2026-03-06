#!/bin/bash
# ============================================================================
# SBATCH Job Script: FormalMATH-Lite Generation with LoRA Fine-tuned Goedel
#
# Uses the full-precision LoRA checkpoint, merges it with the base model,
# then runs generation on FormalMATH_lite.json via vLLM (pass@4).
#
# Options (set before sbatch, or override with --export):
#   RUN_BASELINE=1    — also run generation with the original base model
#   BASELINE_ONLY=1   — skip fine-tuned, only run the baseline model
#
# Submit with:  sbatch scripts/gen_lora.job
#   or:         sbatch --export=ALL,RUN_BASELINE=1 scripts/gen_lora.job
#   or:         sbatch --export=ALL,BASELINE_ONLY=1 scripts/gen_lora.job
# Monitor with: squeue -u $USER
# ============================================================================

#SBATCH --job-name=formalmath-gen-lora
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:8
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o /users/%u/scratch/grind_project/logs/formalmath-gen-lora_%j.out
#SBATCH -e /users/%u/scratch/grind_project/logs/formalmath-gen-lora_%j.err
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

# Default: do not run base-model baseline (set RUN_BASELINE=1 to enable)
RUN_BASELINE="${RUN_BASELINE:-0}"

# Set BASELINE_ONLY=1 to skip fine-tuned and only run the baseline
BASELINE_ONLY="${BASELINE_ONLY:-0}"
if [ "${BASELINE_ONLY}" = "1" ]; then
    RUN_BASELINE="1"
fi

INPUT_FILE="FormalMATH-Bench/FormalMATH_lite.json"

# Create log + runs directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/runs"

# ---------------------------------------------------------------------------
# Activate virtual environment (created by oscar_setup.sh)
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
echo "  VRAM:         $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Python:       $(python3 --version)"
echo "  PyTorch:      $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA avail:   $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  Project:      ${PROJECT_DIR}"
echo "  Checkpoint:   ${CHECKPOINT_DIR}"
echo "  Merged:       ${MERGED_MODEL_DIR}"
echo "  Mode:         Full-precision LoRA"
echo "  RUN_BASELINE:  ${RUN_BASELINE}"
echo "  BASELINE_ONLY: ${BASELINE_ONLY}"
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
# Step 2: Generate on FormalMATH-lite (fine-tuned model, pass@4)
# ---------------------------------------------------------------------------
if [ "${BASELINE_ONLY}" != "1" ]; then
    echo '=== STARTING GENERATION (FINE-TUNED LoRA) ==='
    python FormalMATH-Bench/generate_answers.py \
      --model "${MERGED_MODEL_DIR}" \
      --input_file "${INPUT_FILE}" \
      --generated_file ./runs/gen_goedel-v2-lora_formalmath-lite_pass4.json \
      --n 4 \
      --nums_answer 4 \
      --tp 2 \
      --dp 4

    echo ""
    echo "  Fine-tuned generation complete."
else
    echo "  Skipping fine-tuned generation (BASELINE_ONLY=1)."
fi

# ---------------------------------------------------------------------------
# Step 3 (optional): Generate on FormalMATH-lite (original base model)
# ---------------------------------------------------------------------------
if [ "${RUN_BASELINE}" = "1" ]; then
    echo ""
    echo '=== STARTING GENERATION (BASELINE: original Goedel-v2-8B) ==='
    python FormalMATH-Bench/generate_answers.py \
      --model "${BASE_MODEL}" \
      --input_file "${INPUT_FILE}" \
      --generated_file ./runs/gen_Goedel-Prover-V2-8B_formalmath-lite_pass4.json \
      --n 4 \
      --nums_answer 4 \
      --tp 2 \
      --dp 4

    echo ""
    echo "  Baseline generation complete."
else
    echo ""
    echo "  Skipping baseline (set RUN_BASELINE=1 to enable)."
fi

echo ""
echo "============================================"
echo "  All generation complete!"
echo "  Output: ${PROJECT_DIR}/runs/"
echo "============================================"
