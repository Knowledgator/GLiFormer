#!/usr/bin/env bash
set -euo pipefail

args=(
  python -m gliformer.serve
  --model "${GLIFORMER_MODEL:-knowledgator/gliformer-base-v1}"
  --device "${GLIFORMER_DEVICE:-cuda}"
  --dtype "${GLIFORMER_DTYPE:-bfloat16}"
  --port "${GLIFORMER_PORT:-8000}"
  --max-batch-size "${GLIFORMER_MAX_BATCH_SIZE:-32}"
  --batch-wait-timeout-ms "${GLIFORMER_BATCH_WAIT_MS:-5}"
  --target-memory-fraction "${GLIFORMER_MEMORY_FRACTION:-0.8}"
)

[[ -n "${GLIFORMER_ROUTE_PREFIX:-}" ]] && args+=(--route-prefix "$GLIFORMER_ROUTE_PREFIX")
[[ -n "${GLIFORMER_NUM_REPLICAS:-}" ]] && args+=(--num-replicas "$GLIFORMER_NUM_REPLICAS")
[[ -n "${GLIFORMER_QUANTIZATION:-}" ]] && args+=(--quantization "$GLIFORMER_QUANTIZATION")
[[ -n "${GLIFORMER_PRECOMPILED_BATCH_SIZES:-}" ]] && args+=(--precompiled-batch-sizes "$GLIFORMER_PRECOMPILED_BATCH_SIZES")
[[ "${GLIFORMER_DISABLE_COMPILE:-false}" == "true" ]] && args+=(--no-compile)
[[ -n "${GLIFORMER_COMPILE_BACKEND:-}" ]] && args+=(--compile-backend "$GLIFORMER_COMPILE_BACKEND")
[[ "${GLIFORMER_ENABLE_SEQUENCE_PACKING:-false}" == "true" ]] && args+=(--enable-sequence-packing)
[[ "${GLIFORMER_ENABLE_FLASHDEBERTA:-false}" == "true" ]] && args+=(--enable-flashdeberta)
[[ "${GLIFORMER_DISABLE_MEMORY_CALIBRATION:-false}" == "true" ]] && args+=(--no-memory-calibration)
[[ "${GLIFORMER_ENABLE_POLYLORA:-false}" == "true" ]] && args+=(--enable-polylora)
[[ -n "${GLIFORMER_POLYLORA_DISK_CACHE_DIR:-}" ]] && args+=(--polylora-disk-cache-dir "$GLIFORMER_POLYLORA_DISK_CACHE_DIR")

if [[ -n "${GLIFORMER_POLYLORA_ADAPTERS:-}" ]]; then
  IFS=',' read -ra adapters <<< "$GLIFORMER_POLYLORA_ADAPTERS"
  for adapter in "${adapters[@]}"; do
    [[ -n "$adapter" ]] && args+=(--polylora-adapter "$adapter")
  done
fi

exec "${args[@]}" "$@"
