"""Test setup: stub sentence-transformers so tests run without torch/models."""
import sys
import types

import numpy as np

st = types.ModuleType("sentence_transformers")


class _FakeST:
    def __init__(self, name):
        self.name = name

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        # Deterministic 8-dim bag-of-characters embedding.
        out = np.zeros((len(texts), 8), dtype="float32")
        for i, t in enumerate(texts):
            for ch in t:
                out[i, ord(ch) % 8] += 1
        return out


st.SentenceTransformer = _FakeST
sys.modules.setdefault("sentence_transformers", st)
