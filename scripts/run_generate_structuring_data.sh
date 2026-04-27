#!/usr/bin/env bash
# Run the GLiNExT structuring data generator with Qwen3-4B (non-thinking).
#
# Model: Qwen/Qwen3-4B-Instruct-2507
#   This is the non-thinking instruct variant of Qwen3-4B — it does NOT
#   emit <think>...</think> reasoning blocks, so its output drops straight
#   into the schema and JSON validators in the Python pipeline.
#   (If you swap in the hybrid "Qwen/Qwen3-4B", you must pass
#    `enable_thinking=False` to apply_chat_template, otherwise reasoning
#    text will pollute every generation.)
#
# Override any default via env var, e.g.:
#   MODEL=Qwen/Qwen3-4B-Instruct-2507 NUM_SAMPLES=10000 \
#       bash scripts/run_generate_structuring_data.sh
#
# Extra CLI args are forwarded to the Python script:
#   bash scripts/run_generate_structuring_data.sh --batch-size 16 --max-retries 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- libstdc++ ABI shim ---------------------------------------------------
# vLLM/transformers pull in native libs (e.g. libicui18n.so) that need a
# newer libstdc++ than the system's. Conda envs ship a newer one in their
# own lib dir — prepend it to LD_LIBRARY_PATH so it takes precedence over
# /lib/x86_64-linux-gnu/libstdc++.so.6.
PY_PREFIX="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [[ -n "${PY_PREFIX}" && -f "${PY_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="${PY_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    echo "[env] LD_LIBRARY_PATH prepended with ${PY_PREFIX}/lib"
fi

MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507-FP8}"
OUTPUT="${OUTPUT:-data/structuring_synthetic.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TP="${TP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SEED="${SEED:-42}"
SCORE="${SCORE:-0.4}"

# Quality knobs (override if the model needs different sampling).
TEXT_TEMP="${TEXT_TEMP:-0.9}"
SCHEMA_TEMP="${SCHEMA_TEMP:-0.3}"
EXTRACTION_TEMP="${EXTRACTION_TEMP:-0.1}"
MAX_SCHEMA_TOKENS="${MAX_SCHEMA_TOKENS:-4096}"
MAX_EXTRACTION_TOKENS="${MAX_EXTRACTION_TOKENS:-8192}"
MAX_RETRIES="${MAX_RETRIES:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-2}"

mkdir -p "$(dirname "${OUTPUT}")"

echo "[run] model            = ${MODEL}"
echo "[run] output           = ${OUTPUT}"
echo "[run] num_samples      = ${NUM_SAMPLES}"
echo "[run] batch_size       = ${BATCH_SIZE}"
echo "[run] tensor_parallel  = ${TP}"
echo "[run] max_model_len    = ${MAX_MODEL_LEN}"
echo "[run] seed             = ${SEED}"
echo

exec python -u scripts/generate_structuring_data.py \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed "${SEED}" \
    --score "${SCORE}" \
    --text-temperature "${TEXT_TEMP}" \
    --schema-temperature "${SCHEMA_TEMP}" \
    --extraction-temperature "${EXTRACTION_TEMP}" \
    --max-schema-tokens "${MAX_SCHEMA_TOKENS}" \
    --max-extraction-tokens "${MAX_EXTRACTION_TOKENS}" \
    --max-retries "${MAX_RETRIES}" \
    --min-instances "${MIN_INSTANCES}" \
    "$@"
