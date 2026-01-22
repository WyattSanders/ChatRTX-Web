#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Server defaults (override via env vars)
export VLLM_PORT="${VLLM_PORT:-8001}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export LLM_API_KEY="${LLM_API_KEY:-local}"

cd "${PROJECT_DIR}"
exec uvicorn rag_webapp.main:app --host 0.0.0.0 --port "${WEB_PORT:-8080}"