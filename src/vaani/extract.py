"""Extractive answer: best-supported sentence span from retrieved parents.

The extract *is* the in-budget answer. It is a substring of retrieved
text, so the grounding check is a property of the method, not a hope.
"""

from __future__ import annotations

from dataclasses import dataclass

from vaani.guardrails import content_tokens
from vaani.index import StoredChunk
from vaani.relevance import attachment_conflict
from vaani.text import jaccard, normalize, sentences, tokenize


@dataclass
class Extract:
    answer: str
    source: StoredChunk
    support: float
    candidates: int


def _score(query: str, sent: str) -> float:
    if not sent:
        return 0.0
    if attachment_conflict(query, sent):
        return -1.0
    q_content = content_tokens(query)
    jac = jaccard(query, sent)
    s = set(tokenize(sent))
    if q_content:
        cover = sum(1 for t in q_content if t in s) / len(q_content)
    else:
        q = tokenize(query)
        cover = (sum(1 for t in q if t in s) / len(q)) if q else 0.0
    length_pen = 1.0
    n = len(sent)
    if n < 20:
        length_pen = 0.6
    elif n > 400:
        length_pen = 0.75
    return (0.45 * jac + 0.55 * cover) * length_pen


def extract(query: str, hits: list[tuple[StoredChunk, float, float, float]]) -> Extract | None:
    if not hits:
        return None
    # Dedup parents — parent_child retrieves many children of one passage.
    parents: list[tuple[StoredChunk, float]] = []
    seen: set[str] = set()
    for chunk, fused, _d, _s in hits:
        if chunk.parent_id in seen:
            continue
        seen.add(chunk.parent_id)
        parents.append((chunk, fused))

    q = normalize(query)
    best_sent = ""
    best_score = -1.0
    best_chunk = parents[0][0]
    n_cand = 0
    for chunk, fused in parents:
        for sent in sentences(chunk.parent_text):
            n_cand += 1
            sc = _score(q, sent) + 0.05 * fused
            if sc > best_score:
                best_score = sc
                best_sent = sent
                best_chunk = chunk
    if not best_sent or best_score < 0:
        return None
    if (
        len(best_sent) < 24
        and len(best_chunk.parent_text) <= 280
        and not attachment_conflict(q, best_chunk.parent_text)
    ):
        best_sent = best_chunk.parent_text
    return Extract(
        answer=best_sent,
        source=best_chunk,
        support=max(0.0, min(1.0, best_score)),
        candidates=n_cand,
    )
