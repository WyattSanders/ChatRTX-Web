from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start_char: int
    end_char: int
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: list[Chunk] = []
    i = 0
    n = len(t)
    step = chunk_size - chunk_overlap
    while i < n:
        j = min(i + chunk_size, n)
        piece = t[i:j].strip()
        if piece:
            chunks.append(Chunk(start_char=i, end_char=j, text=piece))
        if j == n:
            break
        i += step
    return chunks

