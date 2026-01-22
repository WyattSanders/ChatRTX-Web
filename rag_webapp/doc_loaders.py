from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class LoadedDoc:
    source_path: str
    page: int | None
    text: str


def load_path(path: str) -> list[LoadedDoc]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.is_dir():
        docs: list[LoadedDoc] = []
        for child in sorted(p.rglob("*")):
            if child.is_file():
                docs.extend(load_file(child))
        return docs

    return load_file(p)


def load_file(p: Path) -> list[LoadedDoc]:
    ext = p.suffix.lower()
    if ext in {".txt", ".md"}:
        return [LoadedDoc(source_path=str(p), page=None, text=p.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".pdf":
        reader = PdfReader(str(p))
        out: list[LoadedDoc] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                out.append(LoadedDoc(source_path=str(p), page=i + 1, text=text))
        return out

    return []

