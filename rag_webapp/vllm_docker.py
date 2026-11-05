from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
import time


def _has_docker() -> bool:
    return shutil.which("docker") is not None


def docker_available() -> bool:
    return _has_docker()

def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _gpu_mem_used_mib() -> int | None:
    """
    Returns GPU0 memory used in MiB (int) or None if unavailable.
    """
    if not _has_nvidia_smi():
        return None
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
        timeout=3,
    )
    if r.returncode != 0:
        return None
    try:
        # first line = GPU 0
        return int((r.stdout or "").strip().splitlines()[0].strip())
    except Exception:
        return None


def _wait_for_gpu_mem_below(target_mib: int, timeout_s: int) -> bool:
    """
    Best-effort wait for GPU memory to drop after stopping a container.
    This avoids starting a new model while the old one is still being torn down.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        used = _gpu_mem_used_mib()
        if used is None:
            return True
        if used <= target_mib:
            return True
        time.sleep(1.0)
    return False


def _run(args: list[str], timeout_s: int = 60, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=env,
    )


def remove_container(name: str, wait_for_gpu_mem_below_mib: int | None = None, timeout_s: int = 45) -> None:
    if not _has_docker():
        return
    # ignore errors (container may not exist)
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, text=True)
    if wait_for_gpu_mem_below_mib is not None:
        _wait_for_gpu_mem_below(wait_for_gpu_mem_below_mib, timeout_s=timeout_s)


@dataclass(frozen=True)
class StartConfig:
    name: str
    port: int
    hf_cache_dir: str
    model: str
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 2048
    max_num_seqs: int = 1
    hf_token: str | None = None
    # On model switch, wait for GPU memory to drop below this threshold before starting.
    # Keeps us from OOMing due to old container teardown still in progress.
    switch_wait_for_gpu_mem_below_mib: int = 1500
    switch_wait_timeout_s: int = 45
    # Host interface for the published vLLM port. vLLM has no auth, so keep it
    # on loopback unless another host really needs direct access.
    bind_host: str = "127.0.0.1"


def start_vllm_openai(cfg: StartConfig) -> None:
    if not _has_docker():
        raise RuntimeError("docker is not available")

    # fixed command; model is validated upstream against allowlist
    remove_container(
        cfg.name,
        wait_for_gpu_mem_below_mib=cfg.switch_wait_for_gpu_mem_below_mib,
        timeout_s=cfg.switch_wait_timeout_s,
    )

    args = [
            "docker",
            "run",
            "-d",
            "--name",
            cfg.name,
            "--gpus",
            "all",
            "--ipc=host",
            "-p",
            f"{cfg.bind_host}:{cfg.port}:8000",
            "-v",
            f"{os.path.abspath(cfg.hf_cache_dir)}:/root/.cache/huggingface",
    ]

    # For gated/private models. Either env var works; we set both for compatibility.
    # Pass names only (`-e NAME`) so docker reads the values from our env: the
    # token never appears in the process list / argv.
    env = None
    if cfg.hf_token:
        env = {**os.environ, "HF_TOKEN": cfg.hf_token, "HUGGING_FACE_HUB_TOKEN": cfg.hf_token}
        args += ["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"]

    args += [
            "vllm/vllm-openai:latest",
            "--model",
            cfg.model,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--gpu-memory-utilization",
            str(cfg.gpu_memory_utilization),
            "--max-model-len",
            str(cfg.max_model_len),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
    ]

    os.makedirs(cfg.hf_cache_dir, exist_ok=True)
    _run(args, timeout_s=120, env=env)


def container_running(name: str) -> bool:
    if not _has_docker():
        return False
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        text=True,
        capture_output=True,
        timeout=5,
    )
    return (r.stdout or "").strip().lower() == "true"

