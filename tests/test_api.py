import base64
import importlib
import subprocess

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("INGEST_SOURCE_PATH", str(tmp_path / "src"))
    for k in ("APP_PASSWORD", "APP_USERNAME", "MODEL_CATALOG"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import rag_webapp.main as main
    main = importlib.reload(main)
    return TestClient(main.app), main


def _basic(u, p):
    return {"Authorization": "Basic " + base64.b64encode(f"{u}:{p}".encode()).decode()}


def test_auth_required_when_password_set(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path, APP_PASSWORD="s3cret")
    assert c.get("/api/models/status").status_code == 401
    assert c.get("/api/models/status", headers=_basic("admin", "wrong")).status_code == 401
    assert c.get("/api/models/status", headers=_basic("admin", "s3cret")).status_code == 200


def test_open_when_password_unset(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path)
    assert c.get("/api/models/status").status_code == 200


def test_ingest_endpoint(monkeypatch, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("hello world")
    c, _ = _client(monkeypatch, tmp_path)
    for _ in range(2):
        r = c.post("/api/ingest")
        assert r.status_code == 200 and r.json() == {"files_seen": 1, "chunks_added": 1}


def test_model_select_rejects_unlisted(monkeypatch, tmp_path):
    c, main = _client(monkeypatch, tmp_path, MODEL_CATALOG="a/b")
    monkeypatch.setattr(main, "docker_available", lambda: True)
    assert c.post("/api/models/select", json={"model": "evil; rm -rf /"}).status_code == 400


def test_docker_args_keep_token_out_of_argv(monkeypatch, tmp_path):
    from rag_webapp import vllm_docker

    calls = []
    monkeypatch.setattr(vllm_docker, "_has_docker", lambda: True)
    monkeypatch.setattr(vllm_docker, "remove_container", lambda *a, **k: None)
    monkeypatch.setattr(vllm_docker.subprocess, "run",
                        lambda args, **kw: calls.append((args, kw)) or subprocess.CompletedProcess(args, 0))
    vllm_docker.start_vllm_openai(vllm_docker.StartConfig(
        name="v", port=8001, hf_cache_dir=str(tmp_path / "hf"), model="a/b", hf_token="hf_SECRET"))
    args, kw = calls[-1]
    assert not any("hf_SECRET" in a for a in args)
    assert kw["env"]["HF_TOKEN"] == "hf_SECRET"
    assert "127.0.0.1:8001:8000" in args
    assert f"{tmp_path / 'hf'}:/root/.cache/huggingface" in args
