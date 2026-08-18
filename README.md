# Vaani

Voice-enabled RAG over [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) for **HH Goa 2026 Task 2**.

**Live:** [https://vaani-production-d1eb.up.railway.app](https://vaani-production-d1eb.up.railway.app)

Speak a question in Hindi, Marathi, or English. The system transcribes it (Sarvam Saaras v3, or ElevenLabs Scribe), retrieves from the MSMARCO-XI Hindi-val index, and returns a **grounded extractive answer with citations**. If the corpus does not support an answer, it abstains.

Two measured surfaces exist. Do not mix their numbers:

| Surface | Retrieval | RAM | What was measured |
|---------|-----------|-----|-------------------|
| Local Compose / in-process harness | hybrid e5-small + FAISS + BM25 + RRF | ~1–2 GB | bench P50/P70/P100, Delhi extract, Sarvam HTTP |
| Public Railway (this URL) | BM25-only (`dense: false`) | Trial **1 GB cap** | live Sarvam audio, password refuse, restart persist |

Railway refused `memoryGB: 2` (`The maximum allowed memory for this service is 1 GB`). Loading FAISS + e5-small + the 219 MB sidecar OOMs there, so the public process is `VAANI_LOW_MEM=true`.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Architecture (short)

```
voice ──► Sarvam / ElevenLabs STT          (outside 200ms)
              │
              ▼
     input guard → [local: embed + FAISS + BM25 + RRF]
                   [Railway public: BM25 only]
              │
              ▼
     extractive span → grounding gate      ◄── measured window, target <200ms
              │
              ├── refuse / abstain
              └── optional Grok polish     (outside 200ms; falls back to extract)
```

The extractive answer **is** the final RAG output. It is a substring of retrieved passages. An LLM timeout cannot take the answer with it.

## Stack

| Piece | Choice |
|-------|--------|
| STT | Sarvam `saaras:v3` (default) or ElevenLabs `scribe_v2` |
| Embeddings | local `intfloat/multilingual-e5-small` (384-d) |
| Dense | FAISS HNSW |
| Sparse | in-process BM25, **separator-based** tokens (not `\\w+`) |
| Fusion | reciprocal rank fusion |
| Answer | sentence-level extractive |
| Polish | xAI Grok (`grok-4.5`) — optional |
| API / UI | FastAPI + one static page |
| Dataset | MSMARCO-XI Hindi val, **all 57,331 unique selected passages** |

Chunking is not one splitter. Six strategies live in `src/vaani/chunking.py`. `scripts/ablate.py` scores them on the same passage pool. We ship the winner and keep the rest in the repo.

## Quick start

Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add SARVAM_API_KEY (or ELEVENLABS_API_KEY)
```

Inspect the dataset (streams; does not download 55 GB):

```bash
python scripts/inspect_dataset.py --langs hi,mr --split validation --limit 400
```

Build an index and a held-out eval set:

```bash
python scripts/ingest.py --strategy whole --max-passages 60000 --eval-queries 500
```

Unit tests (no model, no network):

```bash
pytest -q
```

Latency + retrieval on real queries:

```bash
python scripts/bench.py --n 200
```

Serve:

```bash
python -m vaani.api
# open http://127.0.0.1:8080
```

Live P50/P70/P100 against the running process:

```bash
curl -s "http://127.0.0.1:8080/api/benchmark?n=80"
```

Local persistent run is **Docker Compose** (hybrid, ~4 GB RAM). Public run is **Railway** at the URL above (BM25-only, 1 GB).

```bash
docker compose up --build
python scripts/deploy_smoke.py --base http://127.0.0.1:8080
```

`fly.toml` / `render.yaml` exist for a 4 GB + 10 GB disk deploy. Fly app create was blocked on billing. Railway is the live host. Set `SARVAM_API_KEY` in the host secret store.

## What is in the 200ms number

**In:** input guard, query embed, dense+sparse retrieve, RRF, extractive answer, output guard.

**Out:** speech-to-text, optional Grok polish.

`scripts/bench.py` writes `data/reports/bench.json` with raw per-query timings. Percentiles are computed from those rows.

**In-process harness, 2026-08-17** (Apple Silicon, hybrid e5-small + BM25 + RRF, `whole`, **57,331** unique Hindi-val selected passages — every unique selected passage in `hinval.parquet` — 200 held-out val queries, one at a time, 15 warmup dropped):

| | P50 | P70 | P99 | P100 | <200ms | Recall@k |
|---|---:|---:|---:|---:|---:|---:|
| transcript → extractive output | **46.9ms** | **56.7ms** | 117.7ms | **128.8ms** | **200 / 200** | **0.71** |

That window is **not** audio→answer and is **not** the Railway public process (Railway is BM25-only). See `data/reports/bench.json` and `data/reports/corpus_coverage.json`.

The earlier 12k-passage bench (P50 17.7 / recall 0.83) is superseded. 12k was only 21% of unique Hindi-val selected passages.

Same 4,000-passage / 80-query BM25 ablation (`data/reports/ablation.json`):

| Strategy | Chunks | Recall@k | P50 |
|----------|-------:|---------:|----:|
| **whole** (shipped) | 4000 | **0.763** | 2.8ms |
| fixed_256 | 7833 | **0.763** | 4.0ms |
| metadata | 4000 | **0.763** | 2.8ms |
| sentence | 12192 | 0.688 | 4.4ms |
| window_2 | 11104 | 0.675 | 4.4ms |
| semantic | 12897 | 0.663 | 4.0ms |
| parent_child | 14681 | 0.650 | 4.3ms |

Splitting already-short MSMARCO passages **hurts** sparse retrieval. `whole` and `fixed_256` tie on recall; we ship `whole`. Hybrid e5+BM25 numbers are in the table above. STT and optional Grok polish are outside the 200ms window.

On the 200-query / 57k-index bench the coverage gate abstained 63 times. Unsafe password prompts are refused before retrieval.

**Voice, local hybrid HTTP (2026-08-18, Saaras v3):** Lekha WAV → ffmpeg 16 kHz → Sarvam → RAG. See `data/reports/stt_http.json`.

| Clip | Transcript | STT | RAG | HTTP wall | Result |
|------|------------|----:|----:|----------:|--------|
| कॉर्पोरेशन क्या है? | exact | 706ms | 422ms | 1142ms | grounded |
| भारत की राजधानी क्या है? | `Bharat की राजधानी क्या है?` (folded to भारत) | 796ms | 172ms | 987ms | grounded, Delhi not Mumbai |

**Voice, public Railway BM25-only (re-verified 2026-08-18):** same clips POSTed to `https://vaani-production-d1eb.up.railway.app`. See `data/reports/railway_public.json`.

| Clip | Transcript | STT | RAG | HTTP wall | Result |
|------|------------|----:|----:|----------:|--------|
| कॉर्पोरेशन क्या है? | exact | 1330ms | 78ms | 1851ms* | grounded |
| भारत की राजधानी क्या है? | `Bharat की राजधानी क्या है?` | 1156ms | 111ms | 1705ms* | grounded; extract does **not** contain दिल्ली |

\*latest re-check: STT 1330/1156 ms, RAG 78/111 ms (wall ≈ STT+RAG).

Full **audio→answer is not under 200ms**. STT alone was 700–1500ms. The 200ms bench table is **transcript → extract only**, and only on the local hybrid harness.

## Guardrails

- **Refuse** credential / weapon / self-harm asks *before* retrieval. Corpus passages about banks will retrieve for “what is my password?” — that is not permission to answer.
- **Abstain** when the best dense score is below a threshold (off-topic) or the extract is not supported by retrieved text.
- **Verify** any Grok rewrite; unsupported polish is dropped.

## Dataset notes

MSMARCO-XI is MS MARCO QnA translated into 14 Indic languages (IndicRAGSuite, arXiv:2506.01615). Full dump is 55.6 GB. The shipped index is **every unique selected passage in Hindi validation** (57,331 of 57,331). That is still not Hindi train, not Marathi, and not the other 12 languages. Gold labels live in `eval.jsonl` next to the index. See `data/reports/corpus_coverage.json`.

Passages are already short (~50–80 words). If a chunker emits ~1.0 chunks/passage, that is a measurement, and it belongs in the ablation report.

## Env

See `.env.example`. `STT_PROVIDER=sarvam` or `elevenlabs`. Polish needs `XAI_API_KEY`.
