#!/bin/bash
# ============================================================================
# SBATCH Job Script: Verify Baseline Proofs on OSCAR
#
# CPU-only, 128 parallel processes — each runs verify_gen_proofs.py
# on a separate shard of the flattened input.
#
# The input is a JSON array from generate_answers.py; this script first
# flattens it to JSONL (one line per item) then shards across workers.
#
# Submit with:  sbatch scripts/oscar_verify_gen.sh
# Monitor with: squeue -u $USER
# ============================================================================

#SBATCH --job-name=verify-baseline
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o /users/%u/scratch/grind_project/logs/verify_baseline_%j.out
#SBATCH -e /users/%u/scratch/grind_project/logs/verify_baseline_%j.err
#SBATCH --mail-type=END,FAIL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_WORKERS=64

export SCRATCH_DIR="/users/${USER}/scratch"
export PROJECT_DIR="${SCRATCH_DIR}/grind_project"
export VENV_DIR="${PROJECT_DIR}/venv"
export HF_HOME="${SCRATCH_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export ELAN_HOME="${SCRATCH_DIR}/.elan"
export PATH="${ELAN_HOME}/bin:${PATH}"

REPL_PATH="${PROJECT_DIR}/repl"
LEAN_ENV_PATH="${PROJECT_DIR}/repl/test/Mathlib"

# ── Baseline generation output ──
INPUT_FILE="${PROJECT_DIR}/runs/baseline.json"

OUTPUT_DIR="${PROJECT_DIR}/runs/verify_baseline"
MERGED_OUTPUT="${OUTPUT_DIR}/verify_results.jsonl"

# Create directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/shards"
mkdir -p "${OUTPUT_DIR}/debug"

# ---------------------------------------------------------------------------
# Activate virtual environment
# ---------------------------------------------------------------------------
module load python/3.11.11-5e66
source "${VENV_DIR}/bin/activate"

# Print environment info
echo "============================================"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${SLURMD_NODENAME}"
echo "  CPUs:        ${SLURM_CPUS_ON_NODE}"
echo "  Python:      $(python3 --version)"
echo "  Project:     ${PROJECT_DIR}"
echo "  Input:       ${INPUT_FILE}"
echo "  Workers:     ${NUM_WORKERS}"
echo "============================================"
echo ""

cd "${PROJECT_DIR}" || exit 1

# ---------------------------------------------------------------------------
# Step 1: Flatten JSON array to JSONL (one line per item)
# ---------------------------------------------------------------------------
echo "=== FLATTENING INPUT JSON TO JSONL ==="

FLAT_FILE="${OUTPUT_DIR}/input_flat.jsonl"
python3 -c "
import json, sys
with open('${INPUT_FILE}') as f:
    data = json.load(f) if '${INPUT_FILE}'.endswith('.json') else [json.loads(l) for l in f if l.strip()]
for i, item in enumerate(data):
    item['_item_idx'] = i
    print(json.dumps(item, ensure_ascii=False), file=sys.stdout)
" > "${FLAT_FILE}"

TOTAL_LINES=$(wc -l < "${FLAT_FILE}")
echo "  Flattened to ${TOTAL_LINES} items"

# ---------------------------------------------------------------------------
# Step 2: Split into shards
# ---------------------------------------------------------------------------
echo "=== SPLITTING INTO ${NUM_WORKERS} SHARDS ==="

LINES_PER_SHARD=$(( (TOTAL_LINES + NUM_WORKERS - 1) / NUM_WORKERS ))
echo "  Lines per shard: ~${LINES_PER_SHARD}"

# Clean old shards
rm -f "${OUTPUT_DIR}/shards/shard_"*.jsonl

split -l "${LINES_PER_SHARD}" -d -a 3 \
    "${FLAT_FILE}" "${OUTPUT_DIR}/shards/shard_"

# Rename to .jsonl
for f in "${OUTPUT_DIR}/shards/shard_"*; do
    if [[ ! "$f" == *.jsonl ]]; then
        mv "$f" "${f}.jsonl"
    fi
done

SHARD_FILES=("${OUTPUT_DIR}/shards/shard_"*.jsonl)
ACTUAL_WORKERS=${#SHARD_FILES[@]}
echo "  Actual shards: ${ACTUAL_WORKERS}"
echo "=== SPLIT DONE ==="
echo ""

# ---------------------------------------------------------------------------
# Step 3: Launch parallel workers
# ---------------------------------------------------------------------------
echo "=== LAUNCHING ${ACTUAL_WORKERS} PARALLEL WORKERS ==="

PIDS=()
for idx in $(seq 0 $((ACTUAL_WORKERS - 1))); do
    SHARD="${SHARD_FILES[$idx]}"
    SHARD_OUT="${OUTPUT_DIR}/shards/result_$(printf '%03d' $idx).jsonl"
    SHARD_DEBUG="${OUTPUT_DIR}/debug/debug_$(printf '%03d' $idx).log"

    python3 verify_gen_proofs.py \
        --input "${SHARD}" \
        --output "${SHARD_OUT}" \
        --repl_path "${REPL_PATH}" \
        --lean_env_path "${LEAN_ENV_PATH}" \
        --debug_log "${SHARD_DEBUG}" \
        --cmd_timeout 60 \
        &

    PIDS+=($!)
    echo "  Worker ${idx}: PID=$! shard=$(basename ${SHARD})"
done

echo ""
echo "=== ALL ${ACTUAL_WORKERS} WORKERS LAUNCHED ==="
echo "  Waiting for completion..."
echo ""

# ---------------------------------------------------------------------------
# Step 4: Wait for all workers
# ---------------------------------------------------------------------------
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "  WARNING: Worker PID=${pid} exited with status ${STATUS}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== ALL WORKERS COMPLETE (${FAILED} failed) ==="

# ---------------------------------------------------------------------------
# Step 5: Merge results
# ---------------------------------------------------------------------------
echo "=== MERGING RESULTS ==="

cat "${OUTPUT_DIR}/shards/result_"*.jsonl > "${MERGED_OUTPUT}" 2>/dev/null

RESULT_LINES=$(wc -l < "${MERGED_OUTPUT}" 2>/dev/null || echo 0)
echo "  Merged results: ${RESULT_LINES} items → ${MERGED_OUTPUT}"

# Print summary stats
echo ""
echo "=== SUMMARY ==="
python3 -c "
import json
results = []
with open('${MERGED_OUTPUT}') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))
total_items = len(results)
items_with_pass = sum(1 for r in results if r.get('n_pass', 0) > 0)
total_pass = sum(r.get('n_pass', 0) for r in results)
total_answers = sum(r.get('n_answers', 0) for r in results)
total_skip = sum(r.get('n_skip', 0) for r in results)
print(f'  Items verified:    {total_items}')
print(f'  Items with ≥1 pass: {items_with_pass}/{total_items} ({items_with_pass/total_items*100:.1f}%)' if total_items else '')
print(f'  Answers passed:    {total_pass}/{total_answers} ({total_pass/total_answers*100:.2f}%)' if total_answers else '')
print(f'  Answers skipped:   {total_skip}')
" 2>/dev/null || echo "  (Could not compute summary)"

echo ""
echo "============================================"
echo "  Verification (baseline) complete!"
echo "  Results: ${MERGED_OUTPUT}"
echo "  Debug logs: ${OUTPUT_DIR}/debug/"
echo "  Failed workers: ${FAILED}/${ACTUAL_WORKERS}"
echo "============================================"
