#!/usr/bin/env python3
"""Compare chunking strategies on the same passage pool.

Builds a small index per strategy, scores recall@k + latency, writes
data/reports/ablation.json. We ship whatever wins on this machine —
not a number copied from a blog post.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaani.chunking import STRATEGIES, apply, stats  # noqa: E402
from vaani.config import get_settings  # noqa: E402
from vaani.embeddings import Encoder  # noqa: E402
from vaani.harness import Pipeline  # noqa: E402
from vaani.index import HybridIndex  # noqa: E402


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))
    return float(s[i])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="")
    parser.add_argument("--strategies", default="whole,fixed_256,sentence,window_2,semantic,parent_child,metadata")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--max-passages", type=int, default=4000)
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    src = Path(args.index) if args.index else settings.index_dir
    meta_path = src / "meta.json"
    eval_path = src / "eval.jsonl"
    if not meta_path.exists() or not eval_path.exists():
        print("run ingest first so we have a passage pool + eval queries", file=sys.stderr)
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Reconstruct unique parents from the shipped index.
    parents: dict[str, dict] = {}
    for c in meta["chunks"]:
        parents.setdefault(
            c["parent_id"],
            {
                "passage_id": c["parent_id"],
                "text": c["parent_text"],
                "lang": c.get("lang", ""),
                "query_type": c.get("query_type", ""),
                "source_query_id": c.get("source_query_id"),
            },
        )
        if len(parents) >= args.max_passages:
            break
    passages = list(parents.values())
    queries = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
            if len(queries) >= args.n:
                break

    enc = None if args.no_dense else Encoder(settings)
    names = [s.strip() for s in args.strategies.split(",") if s.strip() in STRATEGIES]
    results = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        chunks = []
        for p in passages:
            chunks.extend(
                apply(
                    name,
                    p["passage_id"],
                    p["text"],
                    lang=p["lang"],
                    query_type=p["query_type"],
                    source_query_id=p["source_query_id"],
                )
            )
        st = stats(chunks)
        vecs = None
        if enc is not None:
            vecs = enc.encode([c.embed_text for c in chunks], is_query=False, batch_size=64)
        idx = HybridIndex(settings)
        idx.build(chunks, vecs, name)
        pipe = Pipeline(index=idx, encoder=enc, settings=settings)
        hits = 0
        judged = 0
        lat: list[float] = []
        for i, row in enumerate(queries):
            t0 = time.perf_counter()
            resp = pipe.ask_text(row["query"], language=row.get("lang", ""))
            if i >= 3:
                lat.append((time.perf_counter() - t0) * 1000.0)
            gold = set(row.get("gold_passage_ids") or [])
            retrieved = {c.passage_id for c in resp.citations}
            if gold:
                judged += 1
                hits += int(bool(gold & retrieved))
        rec = {
            "strategy": name,
            "chunks": len(chunks),
            "chunk_stats": st,
            "recall_at_k": round(hits / judged, 4) if judged else None,
            "judged": judged,
            "p50_ms": round(percentile(lat, 0.50), 2),
            "p70_ms": round(percentile(lat, 0.70), 2),
            "p100_ms": round(max(lat) if lat else 0.0, 2),
        }
        results.append(rec)
        print(json.dumps(rec, indent=2))

    results_sorted = sorted(results, key=lambda r: (-(r["recall_at_k"] or 0), r["p50_ms"]))
    report = {
        "passages": len(passages),
        "queries": len(queries),
        "winner": results_sorted[0]["strategy"] if results_sorted else None,
        "results": results_sorted,
        "note": (
            "Same passage pool and query list for every strategy. "
            "metadata prefixes query_type from the owning MSMARCO query — "
            "part of any gain is a dataset property, not a chunking property."
        ),
    }
    out = settings.reports_dir / "ablation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwinner:", report["winner"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
