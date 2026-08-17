"""Script-aware text helpers.

Python's ``\\w`` excludes Devanagari vowel signs and virama, so
``re.findall(r'\\w+', 'दिल्ली')`` yields ``['द', 'ल', 'ल']``. BM25 then
indexes consonant fragments and Hindi retrieval collapses. We split on
separators instead of character classes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Whitespace + western + Indic punctuation. Everything else stays in the token.
_SEPARATORS = frozenset(
    " \t\n\r\f\v"
    ".,;:!?¡¿…·•/\\|@#$%^&*_+=~`\"'“”‘’«»()[]{}<>"
    "-–—―"
    "।॥॰"
    "、。！？；："
)

_SENT_SPLIT = re.compile(r"(?<=[।.!?॥])\s+|\n+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ").replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split on separators; keep Indic aksharas intact."""
    text = normalize(text).lower()
    if not text:
        return []
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in _SEPARATORS:
            if buf:
                tokens.append("".join(buf))
                buf.clear()
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_precision(answer: str, context: str) -> float:
    """Fraction of answer tokens that appear in context (unigram support)."""
    ans = tokenize(answer)
    if not ans:
        return 0.0
    ctx = token_set(context)
    return sum(1 for t in ans if t in ctx) / len(ans)


def sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def char_windows(text: str, size: int, overlap: int) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size].strip()
        if chunk:
            out.append(chunk)
        if i + size >= len(text):
            break
        i += step
    return out


def passage_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def looks_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def clip(text: str, n: int) -> str:
    text = normalize(text)
    if len(text) <= n:
        return text
    return text[:n].rstrip()


# Measured: Saaras v3 on a Lekha clip of "भारत की राजधानी क्या है?"
# returned "Bharat की राजधानी क्या है?". That one romanization breaks
# Hindi lexical coverage. Fold only tokens we have actually seen.
_STT_FOLD = (
    (re.compile(r"\bBharat\b", re.IGNORECASE), "भारत"),
)


def fold_stt_transcript(text: str) -> str:
    out = normalize(text)
    for pat, repl in _STT_FOLD:
        out = pat.sub(repl, out)
    return out
