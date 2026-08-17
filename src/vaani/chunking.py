"""Multiple chunking strategies over MSMARCO-XI passages.

MSMARCO passages are already short. Several of these strategies will often
emit the whole passage unchanged — that is a finding, not a bug. Ablation
decides what we ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from vaani.text import char_windows, jaccard, normalize, sentences, tokenize


@dataclass
class Chunk:
    chunk_id: str
    text: str
    embed_text: str
    parent_id: str
    parent_text: str
    lang: str
    query_type: str
    strategy: str
    source_query_id: int | None = None
    meta: dict = field(default_factory=dict)


def _base(
    passage_id: str,
    text: str,
    lang: str,
    query_type: str,
    strategy: str,
    source_query_id: int | None,
) -> dict:
    body = normalize(text)
    return dict(
        parent_id=passage_id,
        parent_text=body,
        lang=lang,
        query_type=query_type,
        strategy=strategy,
        source_query_id=source_query_id,
    )


def chunk_whole(
    passage_id: str,
    text: str,
    *,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    body = normalize(text)
    if not body:
        return []
    kw = _base(passage_id, body, lang, query_type, "whole", source_query_id)
    return [Chunk(chunk_id=f"{passage_id}:0", text=body, embed_text=body, **kw)]


def chunk_fixed(
    passage_id: str,
    text: str,
    *,
    size: int = 256,
    overlap: int = 64,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    body = normalize(text)
    windows = char_windows(body, size, overlap)
    kw = _base(passage_id, body, lang, query_type, f"fixed_{size}", source_query_id)
    return [
        Chunk(chunk_id=f"{passage_id}:{i}", text=w, embed_text=w, **kw)
        for i, w in enumerate(windows)
    ]


def chunk_sentence(
    passage_id: str,
    text: str,
    *,
    min_chars: int = 40,
    max_chars: int = 280,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    body = normalize(text)
    sents = sentences(body)
    merged: list[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars and len(buf) < min_chars:
            buf = f"{buf} {s}"
        else:
            merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    kw = _base(passage_id, body, lang, query_type, "sentence", source_query_id)
    return [
        Chunk(chunk_id=f"{passage_id}:{i}", text=m, embed_text=m, **kw)
        for i, m in enumerate(merged)
        if m
    ]


def chunk_window(
    passage_id: str,
    text: str,
    *,
    window: int = 2,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    body = normalize(text)
    sents = sentences(body)
    if not sents:
        return []
    kw = _base(passage_id, body, lang, query_type, f"window_{window}", source_query_id)
    if len(sents) <= window:
        joined = " ".join(sents)
        return [Chunk(chunk_id=f"{passage_id}:0", text=joined, embed_text=joined, **kw)]
    out: list[Chunk] = []
    for i in range(0, len(sents) - window + 1):
        joined = " ".join(sents[i : i + window])
        out.append(Chunk(chunk_id=f"{passage_id}:{i}", text=joined, embed_text=joined, **kw))
    return out


def chunk_semantic(
    passage_id: str,
    text: str,
    *,
    threshold: float = 0.18,
    max_chars: int = 320,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    """Grow a chunk while consecutive sentences stay lexically related.

    Uses token Jaccard, not a second encoder, so ingest stays honest and
    cheap. A true embedding-based variant is available in ablation notes.
    """
    body = normalize(text)
    sents = sentences(body)
    if not sents:
        return []
    groups: list[list[str]] = [[sents[0]]]
    for s in sents[1:]:
        cur = groups[-1]
        joined = " ".join(cur)
        related = jaccard(cur[-1], s) >= threshold
        if related and len(joined) + 1 + len(s) <= max_chars:
            cur.append(s)
        else:
            groups.append([s])
    kw = _base(passage_id, body, lang, query_type, "semantic", source_query_id)
    out: list[Chunk] = []
    for i, g in enumerate(groups):
        joined = " ".join(g)
        out.append(Chunk(chunk_id=f"{passage_id}:{i}", text=joined, embed_text=joined, **kw))
    return out


def chunk_parent_child(
    passage_id: str,
    text: str,
    *,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    """Index sentences; answer from the parent passage."""
    body = normalize(text)
    sents = sentences(body)
    if not sents:
        return []
    kw = _base(passage_id, body, lang, query_type, "parent_child", source_query_id)
    if len(sents) == 1:
        return [Chunk(chunk_id=f"{passage_id}:0", text=sents[0], embed_text=sents[0], **kw)]
    return [
        Chunk(chunk_id=f"{passage_id}:{i}", text=s, embed_text=s, **kw)
        for i, s in enumerate(sents)
    ]


def chunk_metadata(
    passage_id: str,
    text: str,
    *,
    lang: str = "",
    query_type: str = "",
    source_query_id: int | None = None,
) -> list[Chunk]:
    """Whole passage, but the embedding string carries type + language.

    Caveat: ``query_type`` is derived from the query that *owns* the
    passage in MSMARCO-XI. Gold passages therefore share a label with the
    asking query. We still implement it because the brief asks for
    metadata-aware chunking — and we report the leak in the ablation.
    """
    body = normalize(text)
    if not body:
        return []
    prefix = f"[{lang or 'unk'} | {query_type or 'UNKNOWN'}]"
    embed = f"{prefix} {body}"
    kw = _base(passage_id, body, lang, query_type, "metadata", source_query_id)
    return [Chunk(chunk_id=f"{passage_id}:0", text=body, embed_text=embed, **kw)]


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "whole": chunk_whole,
    "fixed_128": lambda pid, text, **kw: chunk_fixed(pid, text, size=128, overlap=32, **kw),
    "fixed_256": lambda pid, text, **kw: chunk_fixed(pid, text, size=256, overlap=64, **kw),
    "fixed_384": lambda pid, text, **kw: chunk_fixed(pid, text, size=384, overlap=80, **kw),
    "sentence": chunk_sentence,
    "window_2": lambda pid, text, **kw: chunk_window(pid, text, window=2, **kw),
    "semantic": chunk_semantic,
    "parent_child": chunk_parent_child,
    "metadata": chunk_metadata,
}


def apply(
    name: str,
    passage_id: str,
    text: str,
    **kw,
) -> list[Chunk]:
    if name not in STRATEGIES:
        raise KeyError(f"unknown chunking strategy: {name}")
    return STRATEGIES[name](passage_id, text, **kw)


def stats(chunks: Iterable[Chunk]) -> dict:
    texts = [c.text for c in chunks]
    n = len(texts)
    if n == 0:
        return {"n": 0}
    lengths = [len(t) for t in texts]
    tok = [len(tokenize(t)) for t in texts]
    lengths.sort()
    return {
        "n": n,
        "chars_p50": lengths[n // 2],
        "chars_max": lengths[-1],
        "tokens_mean": sum(tok) / n,
        "chunks_per_parent": n / max(1, len({c.parent_id for c in chunks})),
    }
