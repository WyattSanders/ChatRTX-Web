from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def docker_available() -> bool:
    return _has_docker()


def _run(args: list[str], timeout_s: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def remove_container(name: str) -> None:
    if not _has_docker():
        return
    # ignore errors (container may not exist)
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, text=True)


@dataclass(frozen=True)
class StartConfig:
    name: str
    port: int
    hf_cache_dir: str
    model: str


def start_vllm_openai(cfg: StartConfig) -> None:
    if not _has_docker():
        raise RuntimeError("docker is not available")

    # fixed command; model is validated upstream against allowlist
    remove_container(cfg.name)

    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            cfg.name,
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"{cfg.port}:8000",
            "-v",
            f"{cfg.hf_cache_dir}:/root/.cache/huggingface",
            "vllm/vllm-openai:latest",
            "--model",
            cfg.model,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--gpu-memory-utilization",
            "0.90",
            "--max-model-len",
            "4096",
        ],
        timeout_s=120,
    )

