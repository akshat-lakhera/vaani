"""Hybrid index: FAISS HNSW + BM25 + metadata sidecar."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vaani.bm25 import BM25
from vaani.chunking import Chunk
from vaani.config import Settings, get_settings


@dataclass(slots=True)
class StoredChunk:
    chunk_id: str
    text: str
    embed_text: str
    parent_id: str
    parent_text: str
    lang: str
    query_type: str
    strategy: str
    source_query_id: int | None = None


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridIndex:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.chunks: list[StoredChunk] = []
        self.bm25 = BM25()
        self.faiss_index = None
        self.strategy: str = ""
        self.dim: int = self.settings.embed_dim

    @property
    def size(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[Chunk], vectors: np.ndarray | None, strategy: str) -> None:
        self.strategy = strategy
        self.chunks = [
            StoredChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                embed_text=c.embed_text,
                parent_id=c.parent_id,
                parent_text=c.parent_text,
                lang=c.lang,
                query_type=c.query_type,
                strategy=c.strategy,
                source_query_id=c.source_query_id,
            )
            for c in chunks
        ]
        self.bm25.fit([c.embed_text for c in self.chunks])
        if vectors is None or len(vectors) == 0:
            self.faiss_index = None
            return
        import faiss

        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError("vectors must be 2d")
        self.dim = vecs.shape[1]
        index = faiss.IndexHNSWFlat(self.dim, self.settings.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.settings.hnsw_ef_construction
        index.add(vecs)
        index.hnsw.efSearch = self.settings.hnsw_ef_search
        self.faiss_index = index

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[StoredChunk, float, float, float]]:
        """Return (chunk, fused, dense, sparse) sorted by fused RRF score."""
        if self.size == 0:
            return []
        top_k = top_k or self.settings.top_k
        pool = min(self.size, max(top_k * 4, 32))

        sparse_hits = self.bm25.search(query, top_k=pool)
        sparse_ids = [i for i, _ in sparse_hits]
        sparse_map = {i: float(s) for i, s in sparse_hits}

        dense_ids: list[int] = []
        dense_map: dict[int, float] = {}
        if self.faiss_index is not None and query_vec is not None and len(query_vec):
            q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            dense_scores, dense_idx = self.faiss_index.search(q, pool)
            dense_ids = [int(i) for i in dense_idx[0] if i >= 0]
            dense_map = {int(i): float(s) for i, s in zip(dense_idx[0], dense_scores[0]) if i >= 0}

        lists = [lst for lst in (dense_ids, sparse_ids) if lst]
        fused = rrf(lists, k=self.settings.rrf_k) if lists else []
        out: list[tuple[StoredChunk, float, float, float]] = []
        for idx, fused_score in fused[:top_k]:
            out.append(
                (
                    self.chunks[idx],
                    float(fused_score),
                    dense_map.get(idx, 0.0),
                    sparse_map.get(idx, 0.0),
                )
            )
        return out

    def save(self, path: Path | None = None) -> Path:
        path = path or self.settings.index_dir
        path.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is not None:
            import faiss

            faiss.write_index(self.faiss_index, str(path / "dense.faiss"))
        meta = {
            "strategy": self.strategy,
            "dim": self.dim,
            "n": self.size,
            "model": self.settings.model_name,
            "chunks": [asdict(c) for c in self.chunks],
            "bm25": self.bm25.to_state(),
        }
        (path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None, settings: Settings | None = None) -> "HybridIndex":
        import gc

        import orjson

        obj = cls(settings=settings)
        path = path or obj.settings.index_dir
        # orjson from bytes avoids a decoded Unicode copy of the 219MB sidecar.
        meta = orjson.loads((path / "meta.json").read_bytes())
        obj.strategy = meta["strategy"]
        obj.dim = int(meta["dim"])
        chunks: list[StoredChunk] = []
        for c in meta["chunks"]:
            text = c.get("text") or ""
            parent = c.get("parent_text") or text
            # Whole-passage chunks store text three times. Keep one string.
            if parent == text:
                parent = text
            chunks.append(
                StoredChunk(
                    chunk_id=c.get("chunk_id") or "",
                    text=text,
                    embed_text="",
                    parent_id=c.get("parent_id") or "",
                    parent_text=parent,
                    lang=c.get("lang") or "",
                    query_type=c.get("query_type") or "",
                    strategy=c.get("strategy") or "",
                    source_query_id=c.get("source_query_id"),
                )
            )
        obj.chunks = chunks
        # Rebuilding or restoring the full BM25 state is the 1GB-killer.
        # low_mem: compact BM25 from texts, no FAISS, no encoder.
        if obj.settings.low_mem:
            texts = [c.text for c in obj.chunks]
            del meta
            gc.collect()
            obj.bm25.fit(texts)
            obj.faiss_index = None
            return obj
        if meta.get("bm25"):
            obj.bm25 = BM25.from_state(meta["bm25"])
        del meta
        gc.collect()
        dense_path = path / "dense.faiss"
        if dense_path.exists():
            import faiss

            mmap_flag = getattr(faiss, "IO_FLAG_MMAP", 0) | getattr(faiss, "IO_FLAG_READ_ONLY", 0)
            try:
                obj.faiss_index = faiss.read_index(str(dense_path), mmap_flag)
            except Exception:  # noqa: BLE001
                obj.faiss_index = faiss.read_index(str(dense_path))
            obj.faiss_index.hnsw.efSearch = obj.settings.hnsw_ef_search
        else:
            obj.faiss_index = None
        return obj
