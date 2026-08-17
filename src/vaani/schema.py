from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Status = Literal["grounded", "abstain", "refuse"]


class Citation(BaseModel):
    passage_id: str
    lang: str = ""
    query_type: str = ""
    text: str
    score: float = 0.0
    strategy: str = ""


class Timings(BaseModel):
    stt_ms: float | None = None
    guard_in_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    extract_ms: float = 0.0
    guard_out_ms: float = 0.0
    generate_ms: float | None = None
    total_rag_ms: float = 0.0
    total_ms: float = 0.0


class AskRequest(BaseModel):
    text: str | None = None
    language: str | None = None
    polish: bool = False
    strategy: str | None = None


class AskResponse(BaseModel):
    status: Status
    answer: str
    transcript: str = ""
    language: str = ""
    support: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    timings: Timings = Field(default_factory=Timings)
    reason: str = ""
    polished: bool = False
    within_budget: bool = False
    strategy: str = ""


class Health(BaseModel):
    ok: bool
    index_loaded: bool
    chunks: int = 0
    strategies: list[str] = Field(default_factory=list)
    stt_provider: str = ""
    stt_configured: bool = False
    llm_configured: bool = False
    model: str = ""
    dense: bool = False


class BenchSummary(BaseModel):
    n: int
    warmup: int
    p50_ms: float
    p70_ms: float
    p100_ms: float
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    under_200ms: int = 0
    budget_ms: float = 200.0
    note: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)
