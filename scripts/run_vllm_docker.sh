#!/usr/bin/env bash
set -euo pipefail

# Minimal OpenAI-compatible vLLM server.
# This assumes NVIDIA Container Toolkit is installed and `docker run --gpus all ... nvidia-smi` works.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-openai}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${PROJECT_DIR}/hf-cache}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
# Exported so `docker run -e NAME` picks the value up without it showing in `ps`.
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_SWITCH_WAIT_FOR_GPU_MEM_BELOW_MIB="${VLLM_SWITCH_WAIT_FOR_GPU_MEM_BELOW_MIB:-1500}"
VLLM_SWITCH_WAIT_TIMEOUT_S="${VLLM_SWITCH_WAIT_TIMEOUT_S:-45}"

echo "Starting vLLM OpenAI server:"
echo "  model: ${MODEL}"
echo "  port : ${VLLM_PORT}"
echo "  name : ${VLLM_CONTAINER_NAME}"
echo "  cache: ${HF_CACHE_DIR}"
echo
echo "API base will be: http://127.0.0.1:${VLLM_PORT}/v1"

mkdir -p "${HF_CACHE_DIR}"

docker rm -f "${VLLM_CONTAINER_NAME}" >/dev/null 2>&1 || true

# Best-effort wait for GPU memory to clear after container stop (avoids OOM on rapid switches)
if command -v nvidia-smi >/dev/null 2>&1; then
  start_ts=$(date +%s)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "")
    if [[ -n "${used}" ]] && [[ "${used}" -le "${VLLM_SWITCH_WAIT_FOR_GPU_MEM_BELOW_MIB}" ]]; then
      break
    fi
    now=$(date +%s)
    if [[ $((now - start_ts)) -ge "${VLLM_SWITCH_WAIT_TIMEOUT_S}" ]]; then
      break
    fi
    sleep 1
  done
fi

docker run -d --name "${VLLM_CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  -p "${VLLM_BIND_HOST:-127.0.0.1}:${VLLM_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN} \
  vllm/vllm-openai:latest \
  --model "${MODEL}" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"

echo "Container started. Tail logs with:"
echo "  docker logs -f ${VLLM_CONTAINER_NAME}"

echo
echo "Waiting for readiness (GET /v1/models)..."
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "READY: http://127.0.0.1:${VLLM_PORT}/v1"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for vLLM readiness. Check logs:"
echo "  docker logs --tail 200 ${VLLM_CONTAINER_NAME}"
exit 1
