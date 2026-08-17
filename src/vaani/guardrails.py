"""Guardrails on both sides of generation.

Retrieval score answers "does the corpus discuss this?", not "should we
answer?". Intent is a separate check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vaani.text import looks_devanagari, normalize, overlap_precision, token_set, tokenize


@dataclass
class GuardDecision:
    ok: bool
    status: str  # grounded | abstain | refuse
    reason: str = ""


# Bilingual (Hindi / Marathi / English) intent refuse. Kept lexical on
# purpose — a classifier would be another model we cannot audit.
_UNSAFE = [
    r"\bpassword\b",
    r"\botp\b",
    r"\bpin\b",
    r"\bcvv\b",
    r"\bssn\b",
    r"\baadhaar\b",
    r"\badhaar\b",
    r"\bcredit card\b",
    r"\bdebit card\b",
    r"\bbank (account|password|pin)\b",
    r"how (do|to) (i )?(make|build|buy) (a )?(bomb|weapon|gun|poison)\b",
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"पासवर्ड",
    r"ओटीपी",
    r"आधार (नंबर|नम्बर)",
    r"पिन कोड क्या है",
    r"बैंक .{0,12}(पासवर्ड|पिन)",
    r"खाते का पासवर्ड",
    r"आत्महत्या",
    r"बम कैसे",
]


_UNSAFE_RE = re.compile("|".join(_UNSAFE), re.IGNORECASE)


def input_guard(query: str, *, max_chars: int = 512) -> GuardDecision:
    q = normalize(query)
    if not q:
        return GuardDecision(False, "refuse", "empty query")
    if len(q) < 2 or len(tokenize(q)) == 0:
        return GuardDecision(False, "refuse", "query has no tokens")
    if len(q) > max_chars:
        q = q[:max_chars]
    if _UNSAFE_RE.search(q):
        return GuardDecision(False, "refuse", "unsafe intent")
    return GuardDecision(True, "grounded", "")


def clip_query(query: str, max_chars: int = 512) -> str:
    return normalize(query)[:max_chars]


_STOP = frozenset(
    """
    a an the of to in on for and or is are was were be been being what which who whom
    how why when where that this these those it its do does did can could should would
    क्या है हैं था थे के का की को से में और या एक यह वह जो कि जैसे कैसे कैसा कैसी कहाँ कब कौन
    आज रात last night
    """.split()
)


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP and len(t) > 1]


def query_coverage(query: str, context: str) -> float:
    q = content_tokens(query)
    if not q:
        return 0.0
    ctx = token_set(context)
    return sum(1 for t in q if t in ctx) / len(q)


def off_topic(best_score: float, threshold: float) -> GuardDecision:
    if best_score < threshold:
        return GuardDecision(False, "abstain", f"off-topic (score {best_score:.3f} < {threshold:.3f})")
    return GuardDecision(True, "grounded", "")


def coverage_gate(query: str, contexts: list[str], threshold: float = 0.6) -> GuardDecision:
    """Lexical coverage of the question in retrieved text.

    Retrieval score says the corpus talks about *a* matching term
    (मौसम, राजधानी). Coverage asks whether the *question* is actually
    sitting in those passages.
    """
    best = max((query_coverage(query, c) for c in contexts), default=0.0)
    if best < threshold:
        return GuardDecision(False, "abstain", f"low query coverage {best:.3f} < {threshold:.3f}")
    return GuardDecision(True, "grounded", "")


def grounding(answer: str, contexts: list[str], threshold: float) -> GuardDecision:
    if not answer.strip():
        return GuardDecision(False, "abstain", "empty extract")
    blob = "\n".join(contexts)
    support = overlap_precision(answer, blob)
    # Extractive answers must also appear as a substring (normalized).
    norm_ans = normalize(answer)
    norm_blob = normalize(blob)
    if norm_ans and norm_ans not in norm_blob:
        # allow near-extracts: every sentence of the answer in the blob
        from vaani.text import sentences

        if not all(normalize(s) in norm_blob for s in sentences(norm_ans) if s):
            return GuardDecision(False, "abstain", "answer not in retrieved context")
    if support < threshold:
        return GuardDecision(False, "abstain", f"weak support {support:.3f} < {threshold:.3f}")
    return GuardDecision(True, "grounded", "")


def support_score(answer: str, contexts: list[str]) -> float:
    return overlap_precision(answer, "\n".join(contexts))


def generated_is_grounded(generated: str, contexts: list[str], threshold: float = 0.55) -> bool:
    """Drop polish that introduces unsupported content."""
    if not generated.strip():
        return False
    return overlap_precision(generated, "\n".join(contexts)) >= threshold


def refuse_message(lang_hint: str = "") -> str:
    if looks_devanagari(lang_hint) or lang_hint.startswith("hi") or lang_hint.startswith("mr"):
        return "इस सवाल का जवाब मैं नहीं दे सकता।"
    return "I cannot answer that."


def abstain_message(lang_hint: str = "") -> str:
    if looks_devanagari(lang_hint) or lang_hint.startswith("hi") or lang_hint.startswith("mr"):
        return "कॉर्पस में इस सवाल का भरोसेमंद जवाब नहीं मिला, इसलिए मैं कुछ नहीं कहूँगा।"
    return "The corpus does not support a reliable answer, so I will not guess."
