#!/usr/bin/env python3
"""P50 / P70 / P100 latency on real MSMARCO-XI queries. No batching.

The measured window is transcript → extractive AskResponse (the RAG path).
STT and LLM are not in this window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaani.config import get_settings  # noqa: E402
from vaani.embeddings import Encoder  # noqa: E402
from vaani.harness import Pipeline  # noqa: E402
from vaani.index import HybridIndex  # noqa: E402


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_queries(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    settings = get_settings()
    index_dir = Path(args.index) if args.index else settings.index_dir
    eval_path = index_dir / "eval.jsonl"
    if not eval_path.exists():
        print(f"missing {eval_path} — run scripts/ingest.py first", file=sys.stderr)
        return 1

    index = HybridIndex.load(index_dir, settings=settings)
    enc = Encoder(settings) if index.faiss_index is not None else None
    pipe = Pipeline(index=index, encoder=enc, settings=settings)
    queries = load_queries(eval_path, args.n + args.warmup)
    if len(queries) < args.warmup + 10:
        print(f"only {len(queries)} eval queries; ingest more", file=sys.stderr)

    raw = []
    hits = 0
    judged = 0
    for i, row in enumerate(queries):
        resp = pipe.ask_text(row["query"], language=row.get("lang", ""))
        gold = set(row.get("gold_passage_ids") or [])
        retrieved = {c.passage_id for c in resp.citations}
        hit = bool(gold & retrieved) if gold else None
        if hit is not None:
            judged += 1
            hits += int(hit)
        rec = {
            "i": i,
            "warmup": i < args.warmup,
            "query_id": row.get("query_id"),
            "lang": row.get("lang"),
            "status": resp.status,
            "support": resp.support,
            "hit_gold_in_topk": hit,
            "timings": resp.timings.model_dump(),
            "within_budget": resp.within_budget,
        }
        raw.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(queries)} last_rag={resp.timings.total_rag_ms:.1f}ms {resp.status}", flush=True)

    measured = [r for r in raw if not r["warmup"]]
    xs = [r["timings"]["total_rag_ms"] for r in measured]
    under = sum(1 for x in xs if x < settings.budget_ms)
    summary = {
        "n": len(xs),
        "warmup": args.warmup,
        "index": str(index_dir),
        "strategy": pipe.index.strategy,
        "chunks": pipe.index.size,
        "p50_ms": round(percentile(xs, 0.50), 2),
        "p70_ms": round(percentile(xs, 0.70), 2),
        "p90_ms": round(percentile(xs, 0.90), 2),
        "p99_ms": round(percentile(xs, 0.99), 2),
        "p100_ms": round(max(xs) if xs else 0.0, 2),
        "mean_ms": round(sum(xs) / len(xs), 2) if xs else 0.0,
        "under_200ms": under,
        "budget_ms": settings.budget_ms,
        "recall_at_k": round(hits / judged, 4) if judged else None,
        "judged": judged,
        "status_counts": {
            s: sum(1 for r in measured if r["status"] == s)
            for s in ("grounded", "abstain", "refuse")
        },
        "stages_p50": {
            k: round(percentile([r["timings"][k] for r in measured], 0.50), 2)
            for k in ("guard_in_ms", "embed_ms", "retrieve_ms", "extract_ms", "guard_out_ms")
        },
        "note": "Window is transcript → extractive AskResponse. STT/LLM not included.",
    }
    out = Path(args.out) if args.out else settings.reports_dir / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": raw}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
