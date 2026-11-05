from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start_char: int
    end_char: int
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split ``text`` into overlapping windows.

    Windows of ``chunk_size`` chars advance by ``chunk_size - chunk_overlap``, so
    every character is covered and the last window ends at the end of the text.
    ``start_char``/``end_char`` index into ``text`` itself and
    ``text[start_char:end_char] == chunk.text`` (surrounding whitespace trimmed).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    # A negative overlap would make the step larger than the window and skip text.
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    text = text or ""
    n = len(text.rstrip())
    i = len(text) - len(text.lstrip())
    if i >= n:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    while i < n:
        j = min(i + chunk_size, n)
        window = text[i:j]
        start = i + (len(window) - len(window.lstrip()))
        end = i + len(window.rstrip())
        if start < end:
            chunks.append(Chunk(start_char=start, end_char=end, text=text[start:end]))
        if j == n:
            break
        i += step
    return chunks

