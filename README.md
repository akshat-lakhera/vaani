# Vaani

Voice-enabled RAG over [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) for **HH Goa 2026 Task 2**.

Speak a question in Hindi, Marathi, or English. The system transcribes it (Sarvam Saaras v3, or ElevenLabs Scribe), retrieves from a hybrid dense+BM25 index, and returns a **grounded extractive answer with citations**. If the corpus does not support an answer, it abstains.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Architecture (short)

```
voice ──► Sarvam / ElevenLabs STT          (outside 200ms)
              │
              ▼
     input guard → embed → FAISS + BM25 → RRF
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

Persistent deploy is **Docker**, not ngrok. The image does not bake the 57k index or e5 weights; mount `./data` (or a 10GB disk at `/app/data` on Fly/Render). Needs ~4 GB RAM.

```bash
docker compose up --build
python scripts/deploy_smoke.py --base http://127.0.0.1:8080
```

Host files: `fly.toml` (Fly.io volume at `/app/data`) and `render.yaml` (Render disk). Set `SARVAM_API_KEY` in the host’s secret store. I have not deployed to Fly/Render from this machine (Railway logged out, Docker daemon was down when last checked).

## What is in the 200ms number

**In:** input guard, query embed, dense+sparse retrieve, RRF, extractive answer, output guard.

**Out:** speech-to-text, optional Grok polish.

`scripts/bench.py` writes `data/reports/bench.json` with raw per-query timings. Percentiles are computed from those rows.

**In-process harness, 2026-08-17** (Apple Silicon, hybrid e5-small + BM25 + RRF, `whole`, **57,331** unique Hindi-val selected passages — every unique selected passage in `hinval.parquet` — 200 held-out val queries, one at a time, 15 warmup dropped):

| | P50 | P70 | P99 | P100 | <200ms | Recall@k |
|---|---:|---:|---:|---:|---:|---:|
| transcript → extractive output | **46.9ms** | **56.7ms** | 117.7ms | **128.8ms** | **200 / 200** | **0.71** |

That window is **not** audio→answer. STT was not run (no `SARVAM_API_KEY`). Two real HTTP POSTs through the public ngrok URL were **403ms** and **264ms**. See `data/reports/deploy_test.json` and `data/reports/corpus_coverage.json`.

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

**Voice:** the browser records WebM/Opus (or mp4); the server converts to 16 kHz mono WAV with `ffmpeg` and sends that to Sarvam. Chrome's native WebM is not a Sarvam-supported format — without this conversion the mic button is theatre. `SARVAM_API_KEY` must be set for the mic to work. Typed questions do not need it.

```bash
python scripts/e2e_voice.py   # synthesizes Hindi speech; calls Sarvam if keyed
```

## Guardrails

- **Refuse** credential / weapon / self-harm asks *before* retrieval. Corpus passages about banks will retrieve for “what is my password?” — that is not permission to answer.
- **Abstain** when the best dense score is below a threshold (off-topic) or the extract is not supported by retrieved text.
- **Verify** any Grok rewrite; unsupported polish is dropped.

## Dataset notes

MSMARCO-XI is MS MARCO QnA translated into 14 Indic languages (IndicRAGSuite, arXiv:2506.01615). Full dump is 55.6 GB. The shipped index is **every unique selected passage in Hindi validation** (57,331 of 57,331). That is still not Hindi train, not Marathi, and not the other 12 languages. Gold labels live in `eval.jsonl` next to the index. See `data/reports/corpus_coverage.json`.

Passages are already short (~50–80 words). If a chunker emits ~1.0 chunks/passage, that is a measurement, and it belongs in the ablation report.

## Env

See `.env.example`. `STT_PROVIDER=sarvam` or `elevenlabs`. Polish needs `XAI_API_KEY`.
