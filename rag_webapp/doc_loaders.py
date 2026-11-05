from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

try:
    from docx import Document as DocxDocument  # python-docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedDoc:
    source_path: str
    page: int | None
    text: str


def load_path(path: str) -> list[LoadedDoc]:
    """Load documents from ``path`` (the server-configured ingestion root).

    Only files that really live under the root are read: a symlink inside the
    root that points elsewhere (e.g. ``notes.txt -> /etc/shadow``) is skipped.
    A file that fails to parse is logged and skipped instead of aborting the run.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    root = p.resolve(strict=True)

    if root.is_dir():
        docs: list[LoadedDoc] = []
        for child in sorted(p.rglob("*")):
            try:
                real = child.resolve(strict=True)
            except (OSError, RuntimeError):
                continue  # dangling or looping symlink
            if not real.is_relative_to(root):
                logger.warning("Skipping %s: resolves outside the ingestion root", child.name)
                continue
            if real.is_file():
                docs.extend(_load_file_safe(child))
        return docs

    return load_file(p)


def _load_file_safe(p: Path) -> list[LoadedDoc]:
    try:
        return load_file(p)
    except Exception:  # noqa: BLE001 - one corrupt file must not abort ingestion
        logger.warning("Skipping %s: could not be parsed", p.name, exc_info=True)
        return []


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
    if ext == ".docx" and _DOCX_AVAILABLE:
        doc = DocxDocument(str(p))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(paragraphs)
        if text.strip():
            return [LoadedDoc(source_path=str(p), page=None, text=text)]
        return []

    return []

