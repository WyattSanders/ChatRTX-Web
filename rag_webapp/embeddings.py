from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    # SentenceTransformer will use GPU if torch+CUDA is installed in the env.
    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    model = _get_model(model_name)
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype="float32")

