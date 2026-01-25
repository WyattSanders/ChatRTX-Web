from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


_TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|[^\|]*\|\s*(?P<done>\d+(?:\.\d+)?)"
    r"(?P<done_unit>[KMG]?)B?/(?P<total>\d+(?:\.\d+)?)"
    r"(?P<total_unit>[KMG]?)B?\s*\[[^\]<]*<(?P<eta>[^\]]+)\]"
)

_VLLM_KV_OOM_RE = re.compile(r"(No available memory for the cache blocks|CUDA out of memory|torch\\.OutOfMemoryError)", re.IGNORECASE)
_HF_401_RE = re.compile(r"(401|Unauthorized|Invalid token)", re.IGNORECASE)
_HF_GATED_RE = re.compile(r"(gated repo|requires authentication|You are trying to access a gated repo)", re.IGNORECASE)


def _unit_to_bytes(n: float, unit: str) -> int:
    # huggingface tqdm usually uses decimal units (kB/MB/GB). We'll treat as base-10.
    mult = {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}.get(unit or "", 1)
    return int(n * mult)


@dataclass(frozen=True)
class DownloadProgress:
    pct: int | None
    done_bytes: int | None
    total_bytes: int | None
    eta: str | None
    raw_line: str | None


def tail_container_logs(container_name: str, tail: int = 400) -> str:
    """
    Returns recent logs for a container. Keep tail small to avoid heavy calls.
    """
    try:
        r = subprocess.run(
            ["docker", "logs", "--tail", str(tail), container_name],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return ""

    out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
    return out


def parse_download_progress(log_text: str) -> DownloadProgress:
    """
    Parse the last tqdm-ish progress line from HuggingFace downloads.
    Example lines often look like:
      model.safetensors:  42%|█████▏ | 1.23G/2.95G [00:31<00:42, 41.2MB/s]
    """
    best: DownloadProgress | None = None
    for line in log_text.splitlines():
        m = _TQDM_RE.search(line)
        if not m:
            continue
        pct = int(m.group("pct"))
        done = float(m.group("done"))
        total = float(m.group("total"))
        done_b = _unit_to_bytes(done, m.group("done_unit"))
        total_b = _unit_to_bytes(total, m.group("total_unit"))
        eta = m.group("eta").strip() if m.group("eta") else None
        best = DownloadProgress(pct=pct, done_bytes=done_b, total_bytes=total_b, eta=eta, raw_line=line.strip())
    return best or DownloadProgress(pct=None, done_bytes=None, total_bytes=None, eta=None, raw_line=None)


def classify_failure(log_text: str) -> str | None:
    """
    Best-effort classification for common failures, without leaking host details.
    """
    if _VLLM_KV_OOM_RE.search(log_text):
        return "insufficient_vram"
    if _HF_401_RE.search(log_text) or _HF_GATED_RE.search(log_text):
        return "auth_required"
    return None

