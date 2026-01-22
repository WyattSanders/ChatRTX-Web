from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return int(v)


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_model: str
    llm_api_key: str

    embed_model: str
    data_dir: str
    ingest_source_path: str | None
    model_catalog: list[str]
    vllm_container_name: str
    vllm_port: int
    hf_cache_dir: str

    chunk_size: int
    chunk_overlap: int
    top_k: int


def _csv(name: str) -> list[str]:
    v = os.getenv(name)
    if v is None:
        return []
    parts = [p.strip() for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def get_settings() -> Settings:
    # Reads from a local `.env` if you create one; otherwise from the process env.
    load_dotenv(override=False)

    vllm_port = _int("VLLM_PORT", 8001)

    data_dir = os.getenv("DATA_DIR") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    ingest_source_path = os.getenv("INGEST_SOURCE_PATH")
    if ingest_source_path is not None:
        ingest_source_path = ingest_source_path.strip() or None
    return Settings(
        llm_base_url=(os.getenv("LLM_BASE_URL") or f"http://127.0.0.1:{vllm_port}/v1").rstrip("/"),
        llm_model=(os.getenv("LLM_MODEL") or "mistralai/Mistral-7B-Instruct-v0.3"),
        llm_api_key=(os.getenv("LLM_API_KEY") or "local"),
        embed_model=(os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"),
        data_dir=data_dir,
        ingest_source_path=ingest_source_path,
        model_catalog=_csv("MODEL_CATALOG"),
        vllm_container_name=(os.getenv("VLLM_CONTAINER_NAME") or "vllm-openai"),
        vllm_port=vllm_port,
        hf_cache_dir=(os.getenv("HF_CACHE_DIR") or "/home/wyatt/hf-cache"),
        chunk_size=_int("CHUNK_SIZE", 900),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        top_k=_int("TOP_K", 5),
    )

