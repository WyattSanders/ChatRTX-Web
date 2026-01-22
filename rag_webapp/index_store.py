from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


@dataclass(frozen=True)
class ChunkMeta:
    id: str
    source_path: str
    page: int | None
    start_char: int
    end_char: int
    text: str


class IndexStore:
    """
    Very small persistence layer:
    - FAISS index: `index.faiss`
    - Metadata: `meta.jsonl` with one object per vector in index order
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.faiss"
        self.meta_path = self.data_dir / "meta.jsonl"

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def load(self) -> tuple[faiss.Index, list[ChunkMeta]] | None:
        if not self.exists():
            return None
        idx = faiss.read_index(str(self.index_path))
        metas: list[ChunkMeta] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                metas.append(
                    ChunkMeta(
                        id=o["id"],
                        source_path=o["source_path"],
                        page=o.get("page"),
                        start_char=o["start_char"],
                        end_char=o["end_char"],
                        text=o["text"],
                    )
                )
        return idx, metas

    def save(self, index: faiss.Index, metas: Iterable[ChunkMeta]) -> None:
        faiss.write_index(index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(
                    json.dumps(
                        {
                            "id": m.id,
                            "source_path": m.source_path,
                            "page": m.page,
                            "start_char": m.start_char,
                            "end_char": m.end_char,
                            "text": m.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @staticmethod
    def new_index(embedding_dim: int) -> faiss.Index:
        # Cosine similarity via normalized vectors + inner product
        return faiss.IndexFlatIP(embedding_dim)

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32", copy=False)
        faiss.normalize_L2(x)
        return x

