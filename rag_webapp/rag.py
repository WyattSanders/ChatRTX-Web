from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from rag_webapp.chunking import chunk_text
from rag_webapp.doc_loaders import LoadedDoc
from rag_webapp.embeddings import embed_texts
from rag_webapp.index_store import ChunkMeta, IndexStore
from rag_webapp.settings import MAX_TOP_K

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    files_seen: int
    chunks_added: int


def _chunk_id(source_path: str, page: int | None, start: int, end: int) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"|")
    h.update(str(page or 0).encode("utf-8"))
    h.update(b"|")
    h.update(str(start).encode("utf-8"))
    h.update(b"|")
    h.update(str(end).encode("utf-8"))
    return h.hexdigest()[:24]


def ingest(
    store: IndexStore,
    embed_model: str,
    docs: list[LoadedDoc],
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    chunks: list[ChunkMeta] = []
    for d in docs:
        for c in chunk_text(d.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            cid = _chunk_id(d.source_path, d.page, c.start_char, c.end_char)
            chunks.append(
                ChunkMeta(
                    id=cid,
                    source_path=d.source_path,
                    page=d.page,
                    start_char=c.start_char,
                    end_char=c.end_char,
                    text=c.text,
                )
            )

    if not chunks:
        # If there is no data, keep any existing index.
        return IngestResult(files_seen=len({d.source_path for d in docs}), chunks_added=0)

    # De-duplicate identical chunk ids (same file/page/span).
    chunks = list({c.id: c for c in chunks}.values())

    vecs = embed_texts(embed_model, [c.text for c in chunks])
    vecs = store.normalize(vecs)

    # `docs` is always the full ingestion source, so rebuild the index rather
    # than appending: appending re-added every chunk on each "Update Library"
    # click (duplicate hits crowding out top_k) and never dropped deleted files.
    index = store.new_index(vecs.shape[1])
    index.add(vecs.astype("float32"))
    store.save(index, chunks)
    return IngestResult(files_seen=len({d.source_path for d in docs}), chunks_added=len(chunks))


@dataclass(frozen=True)
class RetrievalHit:
    meta: ChunkMeta
    score: float


def retrieve(store: IndexStore, embed_model: str, query: str, top_k: int) -> list[RetrievalHit]:
    loaded = store.load()
    if loaded is None:
        return []
    index, metas = loaded
    if index.ntotal == 0 or not metas:
        return []
    if index.ntotal != len(metas):
        # Vector i must map to metas[i]; if the two files disagree (e.g. read
        # mid-save), hits would be attributed to the wrong document.
        logger.warning("Index/metadata size mismatch (%d vs %d); re-run ingestion", index.ntotal, len(metas))
        return []
    k = min(max(1, int(top_k)), MAX_TOP_K, index.ntotal)

    qv = embed_texts(embed_model, [query])
    if qv.size == 0:
        return []
    qv = store.normalize(qv)

    scores, ids = index.search(qv.astype("float32"), k)
    hit_ids = ids[0].tolist()
    hit_scores = scores[0].tolist()

    out: list[RetrievalHit] = []
    for i, s in zip(hit_ids, hit_scores, strict=False):
        if i < 0:
            continue
        if i >= len(metas):
            continue
        out.append(RetrievalHit(meta=metas[i], score=float(s)))
    return out

