#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for GLiNExT data-generation scripts.
#
# Defaults match scripts/generate_finepdfs_layout_data.py:
#   BASE_URL=http://localhost:8000/v1
#   API_KEY=EMPTY
#
# Example:
#   MODEL=Qwen/Qwen2.5-14B-Instruct TP=2 bash scripts/run_vllm_server.sh
#
# Extra args are forwarded to the vLLM API server:
#   bash scripts/run_vllm_server.sh --enable-prefix-caching

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# vLLM/transformers may load native libraries that need the conda libstdc++.
PY_PREFIX="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [[ -n "${PY_PREFIX}" && -f "${PY_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="${PY_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    echo "[env] LD_LIBRARY_PATH prepended with ${PY_PREFIX}/lib"
fi

MODEL="${MODEL:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
TP="${TP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

cmd=(
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --api-key "${API_KEY}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization "${GPU_MEM}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cmd+=(--enforce-eager)
fi

echo "[vllm] model             = ${MODEL}"
echo "[vllm] served_model_name = ${SERVED_MODEL_NAME}"
echo "[vllm] base_url          = http://${HOST}:${PORT}/v1"
echo "[vllm] api_key           = ${API_KEY}"
echo "[vllm] tensor_parallel   = ${TP}"
echo "[vllm] gpu_memory        = ${GPU_MEM}"
echo "[vllm] max_model_len     = ${MAX_MODEL_LEN}"
echo "[vllm] dtype             = ${DTYPE}"
echo

exec "${cmd[@]}" "$@"
