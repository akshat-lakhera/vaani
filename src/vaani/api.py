"""FastAPI app: voice in, grounded answer out."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from vaani.config import get_settings
from vaani.embeddings import Encoder
from vaani.harness import Pipeline
from vaani.index import HybridIndex
from vaani.schema import (
    AskRequest,
    AskResponse,
    BenchSummary,
    CompareRequest,
    CompareResponse,
    CompareStrategyResult,
    Health,
    TranscribeResponse,
)
from vaani.stt import STTError, transcribe_with_retry

WEB = Path(__file__).resolve().parents[2] / "web"

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    if _pipeline is None:
        raise HTTPException(503, "index not loaded")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    settings = get_settings()
    index_dir = settings.index_dir
    if (index_dir / "meta.json").exists():
        index = HybridIndex.load(index_dir, settings=settings)
        enc = Encoder(settings) if index.faiss_index is not None else None
        _pipeline = Pipeline(index=index, encoder=enc, settings=settings)
        # Skip startup warmup: it doubles peak RSS on a 1GB Railway trial box.
    else:
        _pipeline = None
    yield


app = FastAPI(title="Vaani", default_response_class=ORJSONResponse, lifespan=lifespan)

# Enable CORS for external frontends and mobile web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
def health() -> Health:
    settings = get_settings()
    pipe = _pipeline
    return Health(
        ok=pipe is not None,
        index_loaded=pipe is not None,
        chunks=pipe.index.size if pipe else 0,
        strategies=[pipe.index.strategy] if pipe else [],
        stt_provider=settings.stt_provider,
        stt_configured=bool(settings.sarvam_api_key or settings.elevenlabs_api_key),
        llm_configured=bool(settings.xai_api_key),
        model=settings.model_name,
        dense=bool(pipe and pipe.index.faiss_index is not None),
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask(
    text: str | None = Form(default=None),
    polish: bool = Form(default=False),
    language: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
):
    pipe = get_pipeline()
    if audio is not None:
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(413, "audio payload exceeds 25MB limit")
        filename = audio.filename or "audio.webm"
        mime = audio.content_type or "audio/webm"
        return await run_in_threadpool(
            pipe.ask_audio,
            data,
            filename=filename,
            mime=mime,
            language=language,
            polish_answer=polish,
        )
    if text and text.strip():
        return await run_in_threadpool(
            pipe.ask_text,
            text,
            polish_answer=polish,
            language=language or "",
        )
    raise HTTPException(400, "provide audio or text")


@app.post("/api/ask.json", response_model=AskResponse)
async def ask_json(body: AskRequest) -> AskResponse:
    pipe = get_pipeline()
    if not body.text:
        raise HTTPException(400, "text required")
    return await run_in_threadpool(
        pipe.ask_text,
        body.text,
        polish_answer=body.polish,
        language=body.language or "",
    )


@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> TranscribeResponse:
    pipe = get_pipeline()
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "audio payload exceeds 25MB limit")
    filename = audio.filename or "audio.webm"
    mime = audio.content_type or "audio/webm"
    t0 = time.perf_counter()
    try:
        tr = await run_in_threadpool(
            transcribe_with_retry,
            data,
            filename=filename,
            mime=mime,
            language=language,
            settings=pipe.settings,
        )
    except STTError as e:
        raise HTTPException(502, f"STT failed: {e}")
    stt_ms = (time.perf_counter() - t0) * 1000.0
    return TranscribeResponse(
        transcript=tr.text,
        language=tr.language,
        provider=tr.provider,
        stt_ms=round(stt_ms, 2),
    )


def _run_compare(pipe: Pipeline, req: CompareRequest) -> CompareResponse:
    from vaani.chunking import STRATEGIES, apply
    from vaani.extract import extract
    from vaani.guardrails import clip_query, grounding, input_guard, refuse_message
    from vaani.index import StoredChunk
    from vaani.relevance import rerank_hits
    from vaani.text import fold_stt_transcript


    query = clip_query(fold_stt_transcript(req.text), pipe.settings.max_query_chars)
    guard = input_guard(query, max_chars=pipe.settings.max_query_chars)
    if not guard.ok:
        ref_ans = refuse_message(query or req.language)
        return CompareResponse(
            query=req.text,
            results=[
                CompareStrategyResult(
                    strategy=s,
                    chunks_created=0,
                    status="refuse",
                    answer=ref_ans,
                    support=0.0,
                    rag_ms=0.0,
                    within_budget=True,
                    sample_chunks=[],
                )
                for s in req.strategies
            ],
        )

    if pipe.index.faiss_index is not None and pipe.encoder is not None:
        qvec = pipe.encoder.encode_query(query)
    else:
        import numpy as np

        qvec = np.zeros(0, dtype=np.float32)

    base_hits = pipe.index.search(query, qvec, top_k=pipe.settings.top_k)
    unique_parents: dict[str, StoredChunk] = {}
    for h in base_hits:
        if h[0].parent_id not in unique_parents:
            unique_parents[h[0].parent_id] = h[0]

    results: list[CompareStrategyResult] = []
    strategies_to_run = [s for s in req.strategies if s in STRATEGIES]
    if not strategies_to_run:
        strategies_to_run = [
            "whole",
            "fixed_256",
            "sentence",
            "window_2",
            "semantic",
            "parent_child",
            "metadata",
        ]

    for strat in strategies_to_run:
        t0 = time.perf_counter()
        strat_chunks = []
        for p in unique_parents.values():
            strat_chunks.extend(
                apply(
                    strat,
                    p.parent_id,
                    p.parent_text,
                    lang=p.lang,
                    query_type=p.query_type,
                    source_query_id=p.source_query_id,
                )
            )
        simulated_hits = []
        for c in strat_chunks:
            stored = StoredChunk(
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
            simulated_hits.append((stored, 1.0, 0.8, 2.0))
        reranked = rerank_hits(query, simulated_hits)
        ext = extract(query, reranked)
        rag_ms = (time.perf_counter() - t0) * 1000.0
        contexts = [c.parent_text for c in strat_chunks]
        status = "grounded" if ext is not None else "abstain"
        support = ext.support if ext else 0.0
        ans = ext.answer if ext else "No extract supported."
        if ext:
            g = grounding(ext.answer, contexts, pipe.settings.support_threshold)
            if not g.ok:
                status = "abstain"
        sample_texts = [c.text for c in strat_chunks[:3]]
        results.append(
            CompareStrategyResult(
                strategy=strat,
                chunks_created=len(strat_chunks),
                status=status,
                answer=ans,
                support=round(support, 3),
                rag_ms=round(rag_ms, 2),
                within_budget=rag_ms < pipe.settings.budget_ms,
                sample_chunks=sample_texts,
            )
        )

    return CompareResponse(query=req.text, results=results)


@app.post("/api/compare", response_model=CompareResponse)
async def compare_strategies(body: CompareRequest) -> CompareResponse:
    pipe = get_pipeline()
    if not body.text.strip():
        raise HTTPException(400, "text required")
    return await run_in_threadpool(_run_compare, pipe, body)


def _run_benchmark(pipe: Pipeline, n: int) -> BenchSummary:
    eval_path = pipe.settings.index_dir / "eval.jsonl"
    if not eval_path.exists():
        raise HTTPException(404, "no eval.jsonl next to the index")

    rows = []
    with eval_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    warmup = min(5, max(1, n // 20))
    xs: list[float] = []
    raw = []
    for i, row in enumerate(rows):
        resp = pipe.ask_text(row["query"], language=row.get("lang", ""))
        raw.append(
            {
                "status": resp.status,
                "rag_ms": resp.timings.total_rag_ms,
                "within_budget": resp.within_budget,
            }
        )
        if i >= warmup:
            xs.append(resp.timings.total_rag_ms)

    def pct(p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        return round(s[f] + (s[c] - s[f]) * (k - f), 2)

    return BenchSummary(
        n=len(xs),
        warmup=warmup,
        p50_ms=pct(0.50),
        p70_ms=pct(0.70),
        p90_ms=pct(0.90),
        p99_ms=pct(0.99),
        p100_ms=round(max(xs), 2) if xs else 0.0,
        under_200ms=sum(1 for x in xs if x < 200),
        budget_ms=200.0,
        note="Live extractive RAG path. One query at a time. No STT/LLM.",
        extra={"rows": raw},
    )


@app.get("/api/benchmark", response_model=BenchSummary)
async def benchmark(n: int = Query(default=50, ge=5, le=300)) -> BenchSummary:
    pipe = get_pipeline()
    return await run_in_threadpool(_run_benchmark, pipe, n)


@app.get("/")
def index_page():
    page = WEB / "index.html"
    if not page.exists():
        raise HTTPException(404, "web/index.html missing")
    return FileResponse(page)


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("vaani.api:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
