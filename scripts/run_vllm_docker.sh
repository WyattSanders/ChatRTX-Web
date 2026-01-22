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

echo "Starting vLLM OpenAI server:"
echo "  model: ${MODEL}"
echo "  port : ${VLLM_PORT}"
echo "  name : ${VLLM_CONTAINER_NAME}"
echo "  cache: ${HF_CACHE_DIR}"
echo
echo "API base will be: http://127.0.0.1:${VLLM_PORT}/v1"

mkdir -p "${HF_CACHE_DIR}"

docker rm -f "${VLLM_CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --name "${VLLM_CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  -p "${VLLM_PORT}:8000" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model "${MODEL}" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096

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
