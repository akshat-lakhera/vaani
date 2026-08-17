"""FastAPI app: voice in, grounded answer out."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from vaani.config import get_settings
from vaani.embeddings import Encoder
from vaani.harness import Pipeline
from vaani.index import HybridIndex
from vaani.schema import AskRequest, AskResponse, Health

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
        # First encode is 100–200ms on this box. Warm it so the first
        # user is inside the budget.
        if enc is not None:
            for q in ("कॉर्पोरेशन क्या है?", "what is a corporation"):
                _pipeline.ask_text(q)
    else:
        _pipeline = None
    yield


app = FastAPI(title="Vaani", default_response_class=ORJSONResponse, lifespan=lifespan)


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
        filename = audio.filename or "audio.webm"
        mime = audio.content_type or "audio/webm"
        return pipe.ask_audio(data, filename=filename, mime=mime, polish_answer=polish)
    if text and text.strip():
        return pipe.ask_text(text, polish_answer=polish, language=language or "")
    raise HTTPException(400, "provide audio or text")


@app.post("/api/ask.json", response_model=AskResponse)
def ask_json(body: AskRequest) -> AskResponse:
    pipe = get_pipeline()
    if not body.text:
        raise HTTPException(400, "text required")
    return pipe.ask_text(body.text, polish_answer=body.polish, language=body.language or "")


@app.get("/api/benchmark")
def benchmark(n: int = Query(default=50, ge=5, le=300)):
    pipe = get_pipeline()
    eval_path = pipe.settings.index_dir / "eval.jsonl"
    if not eval_path.exists():
        raise HTTPException(404, "no eval.jsonl next to the index")
    from vaani.schema import BenchSummary

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

    summary = BenchSummary(
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
    return summary


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
