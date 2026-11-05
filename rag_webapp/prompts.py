from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievedChunk:
    source_path: str
    page: int | None
    text: str


SYSTEM_PROMPT = """You are a helpful company knowledge base assistant that answers questions using the provided context documents.
Answer accurately and concisely using only the information from the provided context.
If the context is insufficient or the answer is not present, say so clearly and suggest what additional information might help.
Always cite sources by filename (and page if provided) using the [N] notation matching the context numbers."""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        # Never leak server filesystem paths into the prompt; use filename only.
        src = Path(c.source_path).name
        if c.page is not None:
            src = f"{src} (p.{c.page})"
        context_blocks.append(f"[{i}] SOURCE: {src}\n{c.text}".strip())

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    return f"""Context:
{context}

Question:
{question}

Answer (include citations like [1], [2] that map to the sources above):"""

