"""Local multilingual-e5-small encoder.

e5 wants the prefixes ``query: `` / ``passage: ``. We always apply them
here so callers pass raw text.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from vaani.config import Settings, get_settings


class Encoder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        os.environ.setdefault("HF_HOME", str(self.settings.model_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.settings.model_cache))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Imported lazily so unit tests that don't need the model stay light.
        from sentence_transformers import SentenceTransformer

        local = self.settings.local_model_dir or str(self.settings.model_cache / "e5-small")
        source = local if (Path(local) / "model.safetensors").exists() else self.settings.model_name
        self.model = SentenceTransformer(
            source,
            cache_folder=str(self.settings.model_cache),
        )
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())

    def encode(self, texts: list[str], *, is_query: bool, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = "query: " if is_query else "passage: "
        # Batch-1 at query time: dynamically-quantised models are not
        # batch-invariant, and padding to the longest member inflates tail
        # latency. Ingest still batches.
        if is_query:
            batch_size = 1
        payload = [prefix + (t or "")[:2000] for t in texts]
        vecs = self.model.encode(
            payload,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text], is_query=True)[0]


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    return Encoder()
