from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

from rag_webapp.doc_loaders import load_path
from rag_webapp.index_store import IndexStore
from rag_webapp.llm_openai_compat import LLMMessage, OpenAICompatLLM
from rag_webapp.prompts import SYSTEM_PROMPT, RetrievedChunk, build_user_prompt
from rag_webapp.rag import ingest as ingest_docs
from rag_webapp.rag import retrieve
from rag_webapp.settings import get_settings
from rag_webapp.state import get_active_model, set_active_model
from rag_webapp.model_progress import classify_failure, parse_download_progress, tail_container_logs
from rag_webapp.vllm_docker import StartConfig, container_running, docker_available, start_vllm_openai


app = FastAPI(title="ChatRTX Web", version="0.2.0")
settings = get_settings()
store = IndexStore(settings.data_dir)
llm = OpenAICompatLLM(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "web" / "templates"))


class IngestResponse(BaseModel):
    files_seen: int
    chunks_added: int


class IndexStatsResponse(BaseModel):
    chunks: int
    documents: int


class IndexClearResponse(BaseModel):
    ok: bool


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    page: int | None = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]

class LLMHealthResponse(BaseModel):
    ok: bool
    model: str
    available_models: int = 0
    error: str | None = None


class ModelCatalogResponse(BaseModel):
    models: list[str]


class ModelStatusResponse(BaseModel):
    ok: bool
    active_model: str
    ready: bool


class ModelSelectRequest(BaseModel):
    model: str


class ModelSelectResponse(BaseModel):
    ok: bool
    active_model: str
    ready: bool
    message: str | None = None


class ModelProgressResponse(BaseModel):
    ok: bool
    active_model: str
    stage: str  # "downloading" | "starting" | "ready" | "unknown"
    pct: int | None = None
    done_bytes: int | None = None
    total_bytes: int | None = None
    eta: str | None = None
    error: str | None = None


def _active_model() -> str:
    return get_active_model(settings.data_dir) or settings.llm_model


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": settings.app_title,
            "llm_model": _active_model(),
            "data_label": "Managed by server",
            "ingest_enabled": bool(settings.ingest_source_path),
            "model_mgmt_enabled": bool(settings.model_catalog) and docker_available(),
        },
    )


@app.get("/favicon.ico")
async def favicon() -> Response:
    # Avoid noisy 404s in logs; browsers auto-request this.
    return Response(status_code=204)


@app.post("/api/ingest", response_model=IngestResponse)
async def api_ingest() -> IngestResponse:
    # SECURITY: ingestion source is configured on the server only.
    if not settings.ingest_source_path:
        raise HTTPException(status_code=503, detail="Ingestion is not configured on this server.")

    try:
        docs = load_path(settings.ingest_source_path)
    except Exception:  # noqa: BLE001
        # Don't leak filesystem paths back to the browser.
        raise HTTPException(status_code=500, detail="Ingestion source is not available on the server.") from None

    res = ingest_docs(
        store=store,
        embed_model=settings.embed_model,
        docs=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return IngestResponse(files_seen=res.files_seen, chunks_added=res.chunks_added)


@app.get("/api/index/stats", response_model=IndexStatsResponse)
async def api_index_stats() -> IndexStatsResponse:
    stats = store.stats()
    return IndexStatsResponse(chunks=stats["chunks"], documents=stats["documents"])


@app.post("/api/index/clear", response_model=IndexClearResponse)
async def api_index_clear() -> IndexClearResponse:
    store.clear()
    return IndexClearResponse(ok=True)


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest) -> ChatResponse:
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    hits = retrieve(store=store, embed_model=settings.embed_model, query=q, top_k=settings.top_k)
    chunks = [
        RetrievedChunk(source_path=h.meta.source_path, page=h.meta.page, text=h.meta.text)
        for h in hits
    ]
    user_prompt = build_user_prompt(q, chunks)

    try:
        # Fail fast with a clearer message if the local LLM server isn't running.
        _ = await llm.list_models()
        answer = await llm.chat(
            [
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ],
            model=_active_model(),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI model is temporarily unavailable. Please try again later."
            ),
        ) from e

    citations = [
        Citation(source=Path(h.meta.source_path).name, page=h.meta.page, score=h.score)
        for h in hits
    ]
    return ChatResponse(answer=answer, citations=citations)


@app.get("/api/llm/health", response_model=LLMHealthResponse)
async def api_llm_health() -> LLMHealthResponse:
    try:
        models = await llm.list_models()
        return LLMHealthResponse(
            ok=True,
            model=_active_model(),
            available_models=len(models),
            error=None,
        )
    except Exception as e:  # noqa: BLE001
        return LLMHealthResponse(
            ok=False,
            model=_active_model(),
            available_models=0,
            error="unavailable",
        )


@app.get("/api/models/catalog", response_model=ModelCatalogResponse)
async def api_models_catalog() -> ModelCatalogResponse:
    if not settings.model_catalog:
        raise HTTPException(status_code=503, detail="Model catalog is not configured.")
    return ModelCatalogResponse(models=settings.model_catalog)


@app.get("/api/models/status", response_model=ModelStatusResponse)
async def api_models_status() -> ModelStatusResponse:
    active = _active_model()
    try:
        models = await llm.list_models()
        served = models[0] if models else ""
        ready = bool(served) and (served == active)
        return ModelStatusResponse(ok=True, active_model=active, ready=ready)
    except Exception:
        return ModelStatusResponse(ok=False, active_model=active, ready=False)


@app.get("/api/models/progress", response_model=ModelProgressResponse)
async def api_models_progress() -> ModelProgressResponse:
    """
    Best-effort progress/ETA for model download/startup.
    We intentionally avoid returning server paths or raw docker logs.
    """
    active = _active_model()
    if not docker_available():
        return ModelProgressResponse(ok=False, active_model=active, stage="unknown")

    logs = tail_container_logs(settings.vllm_container_name, tail=800)
    p = parse_download_progress(logs)
    failure = classify_failure(logs)

    # If the model is served and matches active, we're ready.
    try:
        models = await llm.list_models()
        served = models[0] if models else ""
        if served and served == active:
            return ModelProgressResponse(ok=True, active_model=active, stage="ready")
    except Exception:
        pass

    # If we have a download line, call it downloading.
    if p.pct is not None and p.total_bytes and p.total_bytes > 0:
        return ModelProgressResponse(
            ok=True,
            active_model=active,
            stage="downloading",
            pct=p.pct,
            done_bytes=p.done_bytes,
            total_bytes=p.total_bytes,
            eta=p.eta,
        )

    # If the container is not running and we can classify a failure, surface it.
    if not container_running(settings.vllm_container_name) and failure:
        return ModelProgressResponse(ok=False, active_model=active, stage="unknown", error=failure)

    # Otherwise, if the container is running but not serving yet, it's starting.
    if container_running(settings.vllm_container_name):
        return ModelProgressResponse(ok=True, active_model=active, stage="starting")

    return ModelProgressResponse(ok=False, active_model=active, stage="unknown")


@app.post("/api/models/select", response_model=ModelSelectResponse)
async def api_models_select(req: ModelSelectRequest) -> ModelSelectResponse:
    if not settings.model_catalog:
        raise HTTPException(status_code=503, detail="Model catalog is not configured.")
    if not docker_available():
        raise HTTPException(status_code=503, detail="Model management is not available on this server.")

    model = (req.model or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if model not in settings.model_catalog:
        raise HTTPException(status_code=400, detail="model is not allowed")

    # Persist user choice server-side.
    set_active_model(settings.data_dir, model)

    # Restart vLLM container; this will download the model into the shared HF cache if needed.
    try:
        start_vllm_openai(
            StartConfig(
                name=settings.vllm_container_name,
                port=settings.vllm_port,
                hf_cache_dir=settings.hf_cache_dir,
                model=model,
                gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                max_model_len=settings.vllm_max_model_len,
                max_num_seqs=settings.vllm_max_num_seqs,
                hf_token=settings.hf_token,
                switch_wait_for_gpu_mem_below_mib=settings.vllm_switch_wait_for_gpu_mem_below_mib,
                switch_wait_timeout_s=settings.vllm_switch_wait_timeout_s,
            )
        )
    except Exception:
        # Don't leak host details.
        return ModelSelectResponse(
            ok=False,
            active_model=model,
            ready=False,
            message="Failed to start the model. Check server logs.",
        )

    return ModelSelectResponse(ok=True, active_model=model, ready=False, message="Model is starting…")
