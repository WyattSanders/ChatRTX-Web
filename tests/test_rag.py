from rag_webapp.doc_loaders import LoadedDoc
from rag_webapp.index_store import IndexStore
from rag_webapp.rag import ingest, retrieve


def _docs(*texts):
    return [LoadedDoc(source_path=f"/docs/{i}.txt", page=None, text=t) for i, t in enumerate(texts)]


def test_reingest_does_not_duplicate(tmp_path):
    store = IndexStore(str(tmp_path))
    for _ in range(3):
        res = ingest(store, "fake", _docs("alpha beta", "gamma delta"), chunk_size=100, chunk_overlap=10)
        assert res.chunks_added == 2
    index, metas = store.load()
    assert index.ntotal == 2 and len(metas) == 2


def test_removed_documents_drop_out(tmp_path):
    store = IndexStore(str(tmp_path))
    ingest(store, "fake", _docs("alpha", "beta"), chunk_size=100, chunk_overlap=10)
    ingest(store, "fake", _docs("alpha"), chunk_size=100, chunk_overlap=10)
    _, metas = store.load()
    assert [m.source_path for m in metas] == ["/docs/0.txt"]


def test_retrieve_returns_unique_hits(tmp_path):
    store = IndexStore(str(tmp_path))
    ingest(store, "fake", _docs("aaaa", "bbbb", "cccc"), chunk_size=100, chunk_overlap=10)
    ingest(store, "fake", _docs("aaaa", "bbbb", "cccc"), chunk_size=100, chunk_overlap=10)
    hits = retrieve(store, "fake", "aaaa", top_k=3)
    assert len({h.meta.id for h in hits}) == len(hits) == 3


def test_empty_source_keeps_existing_index(tmp_path):
    store = IndexStore(str(tmp_path))
    ingest(store, "fake", _docs("alpha"), chunk_size=100, chunk_overlap=10)
    ingest(store, "fake", [], chunk_size=100, chunk_overlap=10)
    assert store.load()[0].ntotal == 1
