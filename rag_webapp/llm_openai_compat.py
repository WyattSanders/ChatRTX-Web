from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class OpenAICompatLLM:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Works with:
    - vLLM openai server (http://.../v1)
    - many local gateways that implement /v1/chat/completions
    - NIM gateways that expose OpenAI-compatible APIs (if configured similarly)
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
        items = data.get("data") or []
        out: list[str] = []
        for it in items:
            mid = it.get("id")
            if isinstance(mid, str) and mid:
                out.append(mid)
        return out

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.2,
        max_tokens: int = 512,
        model: str | None = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

