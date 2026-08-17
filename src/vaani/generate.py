"""Optional Grok polish. Outside the 200ms budget. Never required."""

from __future__ import annotations

from vaani.config import Settings, get_settings
from vaani.index import StoredChunk


class GenerateError(RuntimeError):
    pass


SYSTEM = (
    "You rewrite a grounded extractive answer for a spoken assistant. "
    "Use ONLY the provided passages. Do not add facts. "
    "Keep the same language as the question. "
    "Two short sentences max. No preamble."
)


def polish(
    question: str,
    extractive: str,
    passages: list[StoredChunk],
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    if not settings.xai_api_key:
        raise GenerateError("XAI_API_KEY is not set")
    from openai import OpenAI

    ctx = "\n\n".join(
        f"[{c.parent_id} | {c.lang} | {c.query_type}]\n{c.parent_text}"
        for c in passages[:6]
    )
    user = (
        f"Question: {question}\n\n"
        f"Extractive answer (keep these facts): {extractive}\n\n"
        f"Passages:\n{ctx}"
    )
    client = OpenAI(api_key=settings.xai_api_key, base_url=settings.llm_base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,
            max_tokens=180,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            timeout=settings.generate_timeout_s,
        )
    except Exception as e:  # noqa: BLE001 — harness treats any failure as fallback
        raise GenerateError(str(e)) from e
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise GenerateError("empty generation")
    return text
