"""Tests that pin down the README's "Security model" promises and RAG edge cases."""
import base64
import importlib
import re
import subprocess

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel
from starlette.routing import Mount

from rag_webapp import vllm_docker
from rag_webapp.chunking import chunk_text
from rag_webapp.doc_loaders import LoadedDoc, load_path
from rag_webapp.index_store import IndexStore
from rag_webapp.rag import ingest, retrieve

SECRET_ENV = {
    "LLM_BASE_URL": "http://llm-internal.example:9999/v1",
    "HF_TOKEN": "hf_TOPSECRETTOKEN",
    "LLM_API_KEY": "sk-NOTFORBROWSERS",
}


def _client(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INGEST_SOURCE_PATH", str(tmp_path / "src"))
    for k in ("APP_PASSWORD", "APP_USERNAME", "MODEL_CATALOG", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP",
              "LLM_BASE_URL", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LLM_API_KEY", "VLLM_CONTAINER_NAME"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import rag_webapp.main as main
    main = importlib.reload(main)
    return TestClient(main.app, raise_server_exceptions=False), main


def _basic(u, p):
    return {"Authorization": "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()}


def _concrete(path):
    return re.sub(r"\{[^}]+\}", "x", path)


def _all_requests(app):
    """Every (method, path) the app serves, discovered from the router itself."""
    out = []
    for r in app.routes:
        if isinstance(r, Mount):
            out.append(("GET", r.path.rstrip("/") + "/x"))
            continue
        for m in sorted(getattr(r, "methods", None) or {"GET"}):
            out.append((m, _concrete(r.path)))
    return out


def _no_secrets(text, tmp_path):
    for v in SECRET_ENV.values():
        assert v not in text
    assert "llm-internal.example" not in text
    assert str(tmp_path) not in text


# ---------------------------------------------------------------- auth coverage


def test_every_route_requires_auth_when_password_set(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path, APP_PASSWORD="s3cret", MODEL_CATALOG="a/b")
    reqs = _all_requests(main.app)
    assert ("GET", "/") in reqs and ("POST", "/api/models/select") in reqs  # sanity: discovery works
    # Auto-docs, unknown paths and odd methods are covered too.
    reqs += [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
             ("GET", "/static/x"), ("GET", "/does-not-exist"), ("OPTIONS", "/api/chat"), ("HEAD", "/")]
    for method, path in reqs:
        for headers in ({}, _basic("admin", "wrong"), _basic("root", "s3cret"),
                        {"Authorization": "Basic !!!notbase64"}, {"Authorization": "Bearer s3cret"}):
            r = c.request(method, path, headers=headers)
            assert r.status_code == 401, (method, path, headers, r.status_code)
            assert r.headers["www-authenticate"].startswith("Basic")


def test_correct_credentials_reach_routes(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path, APP_PASSWORD="pa:ss", APP_USERNAME="alice")
    ok = _basic("alice", "pa:ss")  # a colon in the password must still work
    assert c.get("/", headers=ok).status_code == 200
    assert c.get("/api/index/stats", headers=ok).status_code == 200


def test_auto_docs_disabled_when_auth_set(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path, APP_PASSWORD="s3cret")
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path, headers=_basic("admin", "s3cret")).status_code == 404
    c, _ = _client(monkeypatch, tmp_path)
    assert c.get("/openapi.json").status_code == 200  # unchanged when auth is off


@pytest.mark.parametrize("env", [
    {"APP_PASSWORD": "   "},
    {"APP_PASSWORD": "x", "APP_USERNAME": "bad:name"},
])
def test_misconfigured_auth_fails_loudly(monkeypatch, tmp_path, env):
    with pytest.raises(ValueError):
        _client(monkeypatch, tmp_path, **env)


@pytest.mark.parametrize("env", [
    {"CHUNK_OVERLAP": "900", "CHUNK_SIZE": "900"},
    {"CHUNK_OVERLAP": "-1"},
    {"CHUNK_SIZE": "0"},
    {"TOP_K": "0"},
    {"TOP_K": "100000"},
])
def test_bad_rag_settings_fail_at_startup(monkeypatch, tmp_path, env):
    with pytest.raises(ValueError):
        _client(monkeypatch, tmp_path, **env)


# ------------------------------------------------------ ingestion / path safety

PATHY = re.compile(r"path|file|dir|folder|source|root|doc|index", re.I)


def test_no_route_accepts_a_path_like_parameter(monkeypatch, tmp_path):
    """Guard: a new endpoint taking a path/filename/doc id must get its own traversal tests."""
    _, main = _client(monkeypatch, tmp_path)
    for r in main.app.routes:
        if not isinstance(r, APIRoute):
            continue
        d = r.dependant
        names = {p.name for p in d.path_params + d.query_params + d.header_params + d.cookie_params}
        for p in d.body_params:
            ann = p.field_info.annotation
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                names |= set(ann.model_fields)
            else:
                names.add(p.name)
        assert not [n for n in names if PATHY.search(n)], (r.path, names)


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "inside.md").write_text("inside document")
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("TOP SECRET OUTSIDE ROOT")
    return src, outside


@pytest.mark.parametrize("attempt", [
    {"params": {"path": "/etc"}},
    {"params": {"path": "../../../../etc/passwd"}},
    {"json": {"path": "/etc", "source": "..", "INGEST_SOURCE_PATH": "/"}},
    {"headers": {"X-Ingest-Source-Path": "/etc"}},
])
def test_ingest_ignores_request_supplied_paths(monkeypatch, tmp_path, attempt):
    _src(tmp_path)
    c, main = _client(monkeypatch, tmp_path)
    seen = []
    real = main.load_path
    monkeypatch.setattr(main, "load_path", lambda p: seen.append(p) or real(p))
    r = c.post("/api/ingest", **attempt)
    assert r.status_code == 200
    assert seen == [str(tmp_path / "src")]  # always the server setting
    assert r.json() == {"files_seen": 1, "chunks_added": 1}


def test_symlinks_escaping_the_root_are_not_ingested(tmp_path):
    src, outside = _src(tmp_path)
    (src / "link.txt").symlink_to(outside / "secret.txt")
    (src / "linkdir").symlink_to(outside, target_is_directory=True)
    (src / "dangling.txt").symlink_to(tmp_path / "missing.txt")
    (src / "sub").mkdir()
    (src / "sub" / "ok.txt").write_text("nested inside")
    (src / "inner_link.txt").symlink_to(src / "sub" / "ok.txt")  # stays inside: allowed
    docs = load_path(str(src))
    assert all("TOP SECRET" not in d.text for d in docs)
    assert sorted(d.text for d in docs) == ["inside document", "nested inside", "nested inside"]


def test_symlink_escape_via_endpoint(monkeypatch, tmp_path):
    src, outside = _src(tmp_path)
    (src / "link.txt").symlink_to(outside / "secret.txt")
    c, main = _client(monkeypatch, tmp_path)
    assert c.post("/api/ingest").json()["files_seen"] == 1
    _, metas = main.store.load()
    assert all("TOP SECRET" not in m.text for m in metas)


# ------------------------------------------------------- failure / leakage


def test_missing_ingest_source_is_clean_error(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path, **SECRET_ENV)  # src dir never created
    r = c.post("/api/ingest")
    assert r.status_code == 500
    assert r.json() == {"detail": "Ingestion source is not available on the server."}
    _no_secrets(r.text, tmp_path)


def test_indexing_failure_is_clean_error(monkeypatch, tmp_path):
    _src(tmp_path)
    c, main = _client(monkeypatch, tmp_path, **SECRET_ENV)

    def boom(**kw):
        raise RuntimeError(f"CUDA error loading {tmp_path}/src/inside.md")

    monkeypatch.setattr(main, "ingest_docs", boom)
    r = c.post("/api/ingest")
    assert r.status_code == 500 and "Traceback" not in r.text
    _no_secrets(r.text, tmp_path)
    assert c.post("/api/ingest").status_code == 500  # lock was released


def test_corrupt_file_is_skipped_not_fatal(monkeypatch, tmp_path):
    src, _ = _src(tmp_path)
    (src / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    c, _ = _client(monkeypatch, tmp_path)
    r = c.post("/api/ingest")
    assert r.status_code == 200 and r.json()["files_seen"] == 1


def test_responses_do_not_leak_paths_urls_or_tokens(monkeypatch, tmp_path):
    _src(tmp_path)
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b", **SECRET_ENV)
    prompts = []

    async def no_models():
        raise RuntimeError(f"connect failed: {SECRET_ENV['LLM_BASE_URL']}")

    monkeypatch.setattr(main.llm, "list_models", no_models)
    bodies = [c.get("/").text, c.get("/api/llm/health").text, c.get("/api/models/status").text,
              c.get("/api/models/catalog").text, c.get("/api/index/stats").text, c.post("/api/ingest").text]
    r = c.post("/api/chat", json={"question": "inside?"})
    assert r.status_code == 503
    bodies.append(r.text)

    async def models():
        return ["a/b"]

    async def chat(messages, model=None):
        prompts.append(messages[-1].content)
        return "answer"

    monkeypatch.setattr(main.llm, "list_models", models)
    monkeypatch.setattr(main.llm, "chat", chat)
    r = c.post("/api/chat", json={"question": "inside?"})
    assert r.status_code == 200
    assert r.json()["citations"][0]["source"] == "inside.md"
    bodies.append(r.text)
    for b in bodies + prompts:
        _no_secrets(b, tmp_path)


def test_chat_on_empty_index_is_friendly(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path)

    async def must_not_call(*a, **k):
        raise AssertionError("LLM should not be called with an empty index")

    monkeypatch.setattr(main.llm, "list_models", must_not_call)
    monkeypatch.setattr(main.llm, "chat", must_not_call)
    r = c.post("/api/chat", json={"question": "anything"})
    assert r.status_code == 200
    assert r.json()["citations"] == [] and "empty" in r.json()["answer"]


# ---------------------------------------------------------- model switching


def _no_docker_calls(monkeypatch, main):
    calls = []
    monkeypatch.setattr(main, "docker_available", lambda: True)
    monkeypatch.setattr(vllm_docker, "_has_docker", lambda: True)
    monkeypatch.setattr(vllm_docker.subprocess, "run", lambda *a, **k: calls.append(a) or pytest.fail("docker ran"))
    monkeypatch.setattr(main, "start_vllm_openai", lambda *a, **k: calls.append(a) or pytest.fail("start ran"))
    return calls


@pytest.mark.parametrize("model", [
    "evil; rm -rf /", "a/b; id", "--entrypoint=/bin/sh", "a/b --trust-remote-code", "A/B", "a/b\nx",
    "../a/b", "c/d", "a", "$(id)", "a/b\x00",
])
def test_model_outside_allowlist_never_reaches_docker(monkeypatch, tmp_path, model):
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b")
    calls = _no_docker_calls(monkeypatch, main)
    r = c.post("/api/models/select", json={"model": model})
    assert r.status_code == 400 and calls == []


def test_model_select_rejects_non_string(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b")
    calls = _no_docker_calls(monkeypatch, main)
    assert c.post("/api/models/select", json={"model": ["a/b"]}).status_code == 422
    assert calls == []


@pytest.mark.parametrize("model", ["--entrypoint=sh", "a/b c", "a/../../etc", "-v", "a/b/c", ""])
def test_docker_layer_rejects_bad_model_before_touching_container(monkeypatch, tmp_path, model):
    """Defence in depth: even if the allowlist were misconfigured, argv stays safe."""
    calls = []
    monkeypatch.setattr(vllm_docker, "_has_docker", lambda: True)
    monkeypatch.setattr(vllm_docker.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError):
        vllm_docker.start_vllm_openai(vllm_docker.StartConfig(
            name="vllm-openai", port=8001, hf_cache_dir=str(tmp_path / "hf"), model=model))
    assert calls == []  # the running container was not removed


@pytest.mark.parametrize("name", ["-f", "--rm", "a b", "a;b", ""])
def test_bad_container_name_rejected(monkeypatch, name):
    calls = []
    monkeypatch.setattr(vllm_docker, "_has_docker", lambda: True)
    monkeypatch.setattr(vllm_docker.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError):
        vllm_docker.remove_container(name)
    with pytest.raises(ValueError):
        vllm_docker.container_running(name)
    assert calls == []


def test_docker_argv_shape(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(vllm_docker, "_has_docker", lambda: True)
    monkeypatch.setattr(vllm_docker.subprocess, "run",
                        lambda args, **kw: calls.append((args, kw)) or subprocess.CompletedProcess(args, 0))
    vllm_docker.start_vllm_openai(vllm_docker.StartConfig(
        name="vllm-openai", port=8001, hf_cache_dir=str(tmp_path / "hf"), model="org/model-1.5B"))
    for args, kw in calls:
        assert isinstance(args, list) and not kw.get("shell")
        assert "timeout" in kw  # nothing can hang the switch forever
    run_args = calls[-1][0]
    assert run_args[run_args.index(vllm_docker.VLLM_IMAGE) + 1:][:2] == ["--model", "org/model-1.5B"]


def test_failed_switch_restores_previous_model(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b,c/d", LLM_MODEL="a/b")
    monkeypatch.setattr(main, "docker_available", lambda: True)
    monkeypatch.setattr(main, "container_running", lambda name: False)
    started = []

    def start(cfg):
        started.append(cfg.model)
        if cfg.model == "c/d":
            raise subprocess.CalledProcessError(125, ["docker", "run"], stderr="/srv/secret/path")

    monkeypatch.setattr(main, "start_vllm_openai", start)
    r = c.post("/api/models/select", json={"model": "c/d"})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is False
    assert body["active_model"] == "a/b" and "/srv" not in r.text
    assert started == ["c/d", "a/b"]  # previous model relaunched
    assert main._active_model() == "a/b"  # state not switched
    assert not main._model_lock.locked()


def test_successful_switch_persists_choice(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b,c/d", LLM_MODEL="a/b")
    monkeypatch.setattr(main, "docker_available", lambda: True)
    monkeypatch.setattr(main, "start_vllm_openai", lambda cfg: None)
    r = c.post("/api/models/select", json={"model": "c/d"})
    assert r.json()["ok"] is True and main._active_model() == "c/d"


# ------------------------------------------------------------- RAG correctness


@pytest.mark.parametrize("size,overlap", [(10, 0), (10, 3), (10, 9), (7, 2), (1, 0), (50, 10)])
@pytest.mark.parametrize("text", [
    "abcdefghijklmnopqrstuvwxyz0123456789",
    "  leading and trailing whitespace   ",
    "word " * 40,
    "line one\n\nline two\n\n\n   line three   \nend",
    "x",
])
def test_chunk_boundaries_cover_text_exactly(text, size, overlap):
    chunks = chunk_text(text, size, overlap)
    covered = set()
    for c in chunks:
        assert text[c.start_char:c.end_char] == c.text  # offsets match stored text
        assert 0 < len(c.text) <= size
        covered.update(range(c.start_char, c.end_char))
    # No non-whitespace character is dropped.
    assert {i for i, ch in enumerate(text) if not ch.isspace()} <= covered
    # Windows advance monotonically (trimming whitespace can make two windows
    # share a start, but spans never go backwards).
    assert [c.start_char for c in chunks] == sorted(c.start_char for c in chunks)
    assert [c.end_char for c in chunks] == sorted(c.end_char for c in chunks)


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 11), (10, -1), (0, 0), (-5, -10)])
def test_chunk_overlap_must_be_less_than_size(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)
    with pytest.raises(ValueError):
        chunk_text("", size, overlap)  # validated even for empty input


def test_retrieve_bounds_top_k_and_handles_empty(tmp_path):
    store = IndexStore(str(tmp_path))
    assert retrieve(store, "fake", "q", top_k=5) == []  # no index yet
    docs = [LoadedDoc(source_path=f"/d/{i}.txt", page=None, text=t) for i, t in enumerate(["aa", "bb", "cc"])]
    ingest(store, "fake", docs, chunk_size=100, chunk_overlap=10)
    assert len(retrieve(store, "fake", "aa", top_k=10_000)) == 3
    assert len(retrieve(store, "fake", "aa", top_k=0)) == 1


def test_hits_attributed_to_their_source(tmp_path):
    store = IndexStore(str(tmp_path))
    texts = {"/d/a.txt": "apple " * 50, "/d/b.txt": "banana " * 50, "/d/c.pdf": "cherry " * 50}
    docs = [LoadedDoc(source_path=p, page=1, text=t) for p, t in texts.items()]
    ingest(store, "fake", docs, chunk_size=60, chunk_overlap=10)
    for h in retrieve(store, "fake", "banana", top_k=20):
        src = texts[h.meta.source_path]
        assert src[h.meta.start_char:h.meta.end_char] == h.meta.text


def test_index_metadata_mismatch_returns_nothing(tmp_path):
    store = IndexStore(str(tmp_path))
    docs = [LoadedDoc(source_path=f"/d/{i}.txt", page=None, text=t) for i, t in enumerate(["aa", "bb"])]
    ingest(store, "fake", docs, chunk_size=100, chunk_overlap=10)
    lines = store.meta_path.read_text().splitlines()
    store.meta_path.write_text(lines[0] + "\n")  # simulate a torn save
    assert retrieve(store, "fake", "bb", top_k=5) == []
