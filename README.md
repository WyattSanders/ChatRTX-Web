# ChatRTX Web

This is a small **self-hosted RAG web app** for Linux servers.

- **Backend**: FastAPI (ingest docs → embed → FAISS → retrieve → call LLM)
- **Frontend**: a minimal single-page HTML UI served by FastAPI (no Node build step)
- **LLM**: any **OpenAI-compatible** endpoint (recommended: run **vLLM** in Docker with GPU)

## Security model (important)

- **Ingestion source is server-locked**: end users cannot submit arbitrary filesystem paths.
  - Configure it via `INGEST_SOURCE_PATH` on the server.
- **Model selection is allowlisted**:
  - Configure allowed models via `MODEL_CATALOG`.
  - The web UI can “Download & Activate” a model by restarting the configured vLLM docker container.
- The web UI **does not display** server filesystem paths or LLM endpoint URLs.

## Quick start

### 1) Create a Python env (recommended: conda, Python 3.11)

Your system Python is 3.13, which breaks many ML deps. Use the provided `environment.yml`:

```bash
cd /path/to/ChatRTX/rag-webapp
conda env create -f environment.yml
conda activate rag-webapp
```

### 2) Run the web app

```bash
cd /path/to/ChatRTX/rag-webapp
uvicorn rag_webapp.main:app --host 0.0.0.0 --port 8080
```

Then open `http://<server-ip>:8080`.

### 3) Configure server settings

Set environment variables (or create a local `.env` using `config.example.env` as a template):

- **Required for ingestion**:
  - `INGEST_SOURCE_PATH`: absolute path to the server-side documents folder (end users cannot change it)
- **Recommended for model selection**:
  - `MODEL_CATALOG`: comma-separated allowlist of model IDs users can select in the UI
- **LLM connection**:
  - `LLM_MODEL`: active model name
  - `VLLM_PORT`: if using the provided vLLM script (defaults to `8001`)
  - `LLM_BASE_URL`: only needed if you are not using the default `http://127.0.0.1:${VLLM_PORT}/v1`

## Recommended: run a local GPU LLM server (vLLM) in Docker

### 0) One-time: enable NVIDIA GPU access in Docker (Ubuntu host)

Your host needs **NVIDIA Container Toolkit**. If `nvidia-container-cli` is missing, run:

```bash
cd /path/to/ChatRTX/rag-webapp
sudo bash scripts/install_nvidia_container_toolkit.sh
```

This script also runs a quick GPU validation container: `nvidia/cuda ... nvidia-smi`.

### 1) Start vLLM (OpenAI-compatible)

```bash
cd /path/to/ChatRTX/rag-webapp
bash scripts/run_vllm_docker.sh Qwen/Qwen2.5-3B-Instruct
```

This will use:

- `VLLM_CONTAINER_NAME` (default: `vllm-openai`)
- `VLLM_PORT` (default: `8001`)
- `HF_CACHE_DIR` (default: `./hf-cache`)

```bash
curl http://127.0.0.1:8001/v1/models
```

## Running for “publishing” (recommended layout)

- **vLLM**: run as a docker container (managed by the app when model selection is enabled)
- **Web app**: run under a process manager (systemd/supervisor) using your conda env

At minimum, you can run the web app with:

```bash
cd /path/to/ChatRTX/rag-webapp
bash scripts/environment.sh
```

## Ingest documents

Click **Update Library** in the UI.

Note: ingestion runs against the server-configured `INGEST_SOURCE_PATH`.

## Notes

- Index data is stored under `./data/` by default (configurable via `DATA_DIR`).
- Supported files: `.txt`, `.md`, `.pdf`.

