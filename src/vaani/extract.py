"""Extractive answer: concise span from retrieved parents.

The extract *is* the in-budget answer. It is a substring of retrieved
text, so the grounding check is a property of the method, not a hope.

Sentence scoring only locates the relevant region. The returned answer
is the shortest high-novelty span that the surrounding window supports
(e.g. «दिल्ली» from a capital-of-India passage, not the background
sentence it sat in).
"""

from __future__ import annotations

from dataclasses import dataclass

from vaani.guardrails import _STOP, content_tokens, query_coverage
from vaani.index import StoredChunk
from vaani.relevance import attachment_conflict, query_properties
from vaani.text import jaccard, normalize, sentences, tokenize, tokenize_spans


@dataclass
class Extract:
    answer: str
    source: StoredChunk
    support: float
    candidates: int


_ENTITY_CUES = frozenset(
    {
        "राजधानी",
        "capital",
        "कहाँ",
        "कहां",
        "where",
        "कौन",
        "who",
        "नाम",
        "name",
        "किस",
        "which",
        "city",
        "शहर",
        "देश",
        "country",
    }
)
_NUMERIC_CUES = frozenset(
    {
        "कितना",
        "कितने",
        "कितनी",
        "how",
        "when",
        "कब",
        "वर्ष",
        "year",
        "संख्या",
        "many",
        "much",
    }
)
_DEFINITION_CUES = frozenset({"क्या", "what", "अर्थ", "mean", "meaning"})
_BAD_EDGE = frozenset(
    {"का", "की", "के", "को", "से", "में", "और", "या", "of", "the", "a", "an", "is", "are", "to"}
)
_MAX_ENTITY_TOKS = 5
_MAX_NUMERIC_TOKS = 4
_MAX_DEFINITION_CHARS = 160
_CONTEXT_BEFORE = 48
_CONTEXT_AFTER = 160


def _intent(query: str) -> str:
    toks = set(tokenize(query))
    if query_properties(query) or toks & _ENTITY_CUES:
        return "entity"
    if toks & _NUMERIC_CUES:
        return "numeric"
    if toks & _DEFINITION_CUES:
        return "definition"
    return "open"


def _is_digit_token(tok: str) -> bool:
    return bool(tok) and all(ch.isdigit() or "०" <= ch <= "९" for ch in tok)


def _is_question(text: str) -> bool:
    s = text.strip()
    return s.endswith("?") or s.endswith("؟")


def _has_qmark(text: str) -> bool:
    return "?" in text or "؟" in text


def _cut_off_penalty(text: str, end: int, q_content: set[str]) -> float:
    """Downrank 'New' when the next token is the rest of the name ('Delhi')."""
    rest = text[end:].lstrip(" ,")
    if not rest:
        return 1.0
    nxt = tokenize_spans(rest)
    if not nxt:
        return 1.0
    tok = nxt[0][0]
    if tok in _STOP or tok in q_content or _is_digit_token(tok):
        return 1.0
    return 0.62


def _wide_context(text: str, start: int, end: int) -> str:
    lo = max(0, start - _CONTEXT_BEFORE)
    hi = min(len(text), end + _CONTEXT_AFTER)
    return text[lo:hi]


def _trim_definition(sent: str) -> str:
    sent = normalize(sent)
    if len(sent) <= _MAX_DEFINITION_CHARS:
        return sent
    cut = sent[:_MAX_DEFINITION_CHARS]
    for sep in ("।", ".", ";", "—", "–"):
        idx = cut.rfind(sep)
        if idx >= 40:
            return sent[: idx + 1].strip()
    for i in range(len(cut) - 1, 39, -1):
        if cut[i] in " ,":
            return cut[:i].rstrip(" ,;:-")
    return cut.rstrip()


def _brevity(n_toks: int, n_chars: int, intent: str) -> float:
    if intent == "entity":
        if n_toks == 1:
            return 1.0
        if n_toks == 2:
            return 0.94
        if n_toks <= 4:
            return 0.72
        return 0.28
    if intent == "numeric":
        return 1.0 if n_toks <= _MAX_NUMERIC_TOKS else 0.4
    if intent == "definition":
        if 24 <= n_chars <= _MAX_DEFINITION_CHARS:
            return 0.88
        if n_chars < 24:
            return 0.5
        if n_toks <= 22:
            return 0.7
        return 0.32
    if n_toks <= 16:
        return 0.8
    return 0.45


def _span_score(
    query: str,
    span: str,
    context: str,
    parent: str,
    end: int,
    intent: str,
    q_content: set[str],
    fused: float,
    position: int,
) -> float:
    if not span or attachment_conflict(query, span) or attachment_conflict(query, context):
        return -1.0
    if _is_question(span) or _has_qmark(span):
        return -0.8
    toks = [t for t, _s, _e in tokenize_spans(span)]
    content = [t for t in toks if t not in _STOP]
    if not content:
        return -1.0
    if intent != "numeric" and content and all(_is_digit_token(t) for t in content):
        return -0.6
    if set(content) <= q_content and intent == "entity":
        return -0.4
    novel = [t for t in content if t not in q_content]
    novelty = len(novel) / len(content)
    cover = query_coverage(query, context)
    jac = jaccard(query, context)
    pos = 1.0 - min(max(position, 0), 120) / 160.0
    brief = _brevity(len(content), len(span), intent)
    qpen = 0.35 if _is_question(context.strip()[:80]) else 1.0
    return (
        0.28 * cover
        + 0.12 * jac
        + 0.28 * novelty
        + 0.22 * brief
        + 0.07 * pos
        + 0.03 * min(fused, 1.0)
    ) * qpen * _cut_off_penalty(parent, end, q_content)


def _candidate_spans(text: str, intent: str, q_content: set[str]) -> list[tuple[str, int, int]]:
    """(span, start, end) offsets into *text* (already normalized)."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, int, int]] = []

    def add(span: str, start: int, end: int) -> None:
        span = span.strip(" \t,;:-—–")
        if not span:
            return
        end = start + len(span) if text[start : start + len(span)] == span else end
        # Re-locate stripped span inside the original slice when punctuation
        # was trimmed off the edges.
        piece = text[start:end]
        rel = piece.find(span)
        if rel < 0:
            rel = text.find(span)
            if rel < 0:
                return
            start, end = rel, rel + len(span)
        else:
            start, end = start + rel, start + rel + len(span)
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        out.append((span, start, end))

    max_n = _MAX_ENTITY_TOKS if intent in {"entity", "numeric"} else 8
    if intent == "definition":
        max_n = 12

    cursor = 0
    for sent in sentences(text):
        idx = text.find(sent, cursor)
        if idx < 0:
            idx = text.find(sent)
        if idx < 0:
            continue
        cursor = idx + len(sent)
        question = _is_question(sent)
        if not question:
            if intent == "definition":
                trimmed = _trim_definition(sent)
                add(trimmed, idx, idx + len(trimmed))
            else:
                add(sent, idx, idx + len(sent))
            for sep in (":", "：", "—"):
                if sep in sent:
                    head = sent.split(sep, 1)[0]
                    add(head, idx, idx + len(head))
                    break
        if question and intent != "entity":
            continue
        local = tokenize_spans(sent)
        for n in range(1, max_n + 1):
            for i in range(len(local) - n + 1):
                _tok, rel_s, _ = local[i]
                _end_tok, _s, rel_e = local[i + n - 1]
                gram = [local[j][0] for j in range(i, i + n)]
                content = [t for t in gram if t not in _STOP]
                if not content:
                    continue
                if intent != "numeric" and all(_is_digit_token(t) for t in content):
                    continue
                if intent == "entity" and _is_digit_token(gram[0]):
                    continue
                if intent == "entity" and set(content) <= q_content:
                    continue
                start, end = idx + rel_s, idx + rel_e
                add(text[start:end], start, end)

    return out


def _sentence_score(query: str, sent: str) -> float:
    """Locate the supporting sentence. Isolation happens after this."""
    if not sent:
        return 0.0
    if attachment_conflict(query, sent):
        return -1.0
    if _is_question(sent) or _has_qmark(sent):
        return -0.5
    lead = tokenize(sent)
    if lead and lead[0] in _BAD_EDGE:
        return -0.4
    q_content = content_tokens(query)
    jac = jaccard(query, sent)
    s = set(tokenize(sent))
    if q_content:
        cover = sum(1 for t in q_content if t in s) / len(q_content)
    else:
        qtoks = tokenize(query)
        cover = (sum(1 for t in qtoks if t in s) / len(qtoks)) if qtoks else 0.0
    length_pen = 1.0
    n = len(sent)
    if n < 20:
        length_pen = 0.6
    elif n > 400:
        length_pen = 0.75
    return (0.45 * jac + 0.55 * cover) * length_pen


def _window_around(parent: str, sent: str, prev: str) -> str:
    if prev:
        prev_i = parent.find(prev)
        sent_i = parent.find(sent)
        if prev_i >= 0 and sent_i >= prev_i:
            return parent[prev_i : sent_i + len(sent)]
    return sent


def _isolate(query: str, window: str, fallback: str, intent: str, fused: float) -> str:
    fallback = normalize(fallback)
    if intent in {"definition", "open"}:
        return _trim_definition(fallback)
    q_content = set(content_tokens(query))
    window = normalize(window)
    if not window:
        return fallback
    best_span = fallback
    best_score = -1.0
    for span, start, end in _candidate_spans(window, intent, q_content):
        edge = tokenize(span)
        if edge and (edge[0] in _BAD_EDGE or edge[-1] in _BAD_EDGE):
            continue
        ctx = _wide_context(window, start, end)
        sc = _span_score(query, span, ctx, window, end, intent, q_content, fused, start)
        if sc > best_score:
            best_score = sc
            best_span = span
    if not best_span or best_score < 0:
        return fallback
    return best_span


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
    intent = _intent(q)
    best_sent = ""
    best_prev = ""
    best_score = -1.0
    best_chunk = parents[0][0]
    best_fused = parents[0][1]
    n_cand = 0
    for chunk, fused in parents:
        parent = normalize(chunk.parent_text)
        sents = sentences(parent)
        prev = ""
        for sent in sents:
            n_cand += 1
            sc = _sentence_score(q, sent) + 0.05 * fused
            if sc > best_score:
                best_score = sc
                best_sent = sent
                best_prev = prev
                best_chunk = chunk
                best_fused = fused
            prev = sent
    if not best_sent or best_score < 0:
        return None
    parent = normalize(best_chunk.parent_text)
    window = _window_around(parent, best_sent, best_prev)
    answer = _isolate(q, window, best_sent, intent, best_fused)
    return Extract(
        answer=answer,
        source=best_chunk,
        support=max(0.0, min(1.0, best_score)),
        candidates=n_cand,
    )
