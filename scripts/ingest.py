#!/usr/bin/env python3
"""Build a hybrid index from streamed MSMARCO-XI selected passages.

Also writes a held-out eval set of (query → gold parent_ids) that we can
score without leaking those queries into the chunk text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaani.chunking import STRATEGIES, apply, stats  # noqa: E402
from vaani.config import get_settings  # noqa: E402
from vaani.dataset import stream  # noqa: E402
from vaani.embeddings import Encoder  # noqa: E402
from vaani.index import HybridIndex  # noqa: E402
from vaani.text import passage_hash  # noqa: E402


def collect(
    langs: list[str],
    splits: list[str],
    max_passages: int,
    eval_queries: int,
    include_unselected: int,
) -> tuple[list[dict], list[dict]]:
    """Return (unique passages, eval queries whose gold is in the index)."""
    by_hash: dict[str, dict] = {}
    eval_set: list[dict] = []
    unselected_kept = 0
    for lang in langs:
        for split in splits:
            print(f"streaming {lang}/{split} …", flush=True)
            for ex in stream(lang, split):
                gold_ids: list[str] = []
                for p in ex.passages:
                    key = passage_hash(p.text)
                    stored = by_hash.get(key)
                    if stored is None:
                        allow = False
                        if p.selected and len(by_hash) < max_passages:
                            allow = True
                        elif (
                            not p.selected
                            and include_unselected > 0
                            and unselected_kept < include_unselected
                            and len(by_hash) < max_passages
                        ):
                            allow = True
                            unselected_kept += 1
                        if allow:
                            stored = {
                                "passage_id": p.passage_id,
                                "text": p.text,
                                "english": p.english,
                                "lang": p.lang,
                                "query_type": p.query_type,
                                "source_query_id": p.query_id,
                                "selected": p.selected,
                            }
                            by_hash[key] = stored
                    if p.selected and stored is not None:
                        gold_ids.append(stored["passage_id"])
                if (
                    gold_ids
                    and ex.query
                    and split == "validation"
                    and len(eval_set) < eval_queries
                ):
                    eval_set.append(
                        {
                            "query_id": ex.query_id,
                            "query": ex.query,
                            "eng_query": ex.eng_query,
                            "answer": ex.answer,
                            "lang": lang,
                            "query_type": ex.query_type,
                            "gold_passage_ids": gold_ids,
                        }
                    )
                if len(by_hash) >= max_passages and len(eval_set) >= eval_queries:
                    return list(by_hash.values()), eval_set
    return list(by_hash.values()), eval_set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="hi,mr")
    parser.add_argument(
        "--splits",
        default="validation",
        help="validation (~460MB/lang) or train (~3.7GB/lang). Default is validation.",
    )
    parser.add_argument("--strategy", default="whole", choices=sorted(STRATEGIES))
    parser.add_argument("--max-passages", type=int, default=25000)
    parser.add_argument("--eval-queries", type=int, default=400)
    parser.add_argument("--include-unselected", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--name", default="")
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="BM25-only index. Use when the local e5 snapshot is not downloaded yet.",
    )
    args = parser.parse_args()

    settings = get_settings()
    index_name = args.name or settings.index_name
    out = settings.data_dir / "indexes" / index_name

    langs = [p.strip() for p in args.langs.split(",") if p.strip()]
    splits = [p.strip() for p in args.splits.split(",") if p.strip()]
    passages, eval_set = collect(
        langs, splits, args.max_passages, args.eval_queries, args.include_unselected
    )
    print(f"unique passages={len(passages)} eval_queries={len(eval_set)}", flush=True)
    if not passages:
        print("no passages collected", file=sys.stderr)
        return 1

    chunks = []
    for p in passages:
        chunks.extend(
            apply(
                args.strategy,
                p["passage_id"],
                p["text"],
                lang=p["lang"],
                query_type=p["query_type"],
                source_query_id=p["source_query_id"],
            )
        )
    print(f"chunks={len(chunks)} stats={stats(chunks)}", flush=True)

    vectors = None
    if not args.no_dense:
        enc = Encoder(settings)
        vectors = enc.encode(
            [c.embed_text for c in chunks],
            is_query=False,
            batch_size=args.batch_size,
        )
    else:
        print("dense skipped — BM25 only", flush=True)
    index = HybridIndex(settings)
    index.build(chunks, vectors, args.strategy)
    index.save(out)
    eval_path = out / "eval.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for row in eval_set:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "strategy": args.strategy,
        "passages": len(passages),
        "chunks": len(chunks),
        "chunk_stats": stats(chunks),
        "eval_queries": len(eval_set),
        "langs": langs,
        "splits": splits,
        "model": settings.model_name,
        "dense": not args.no_dense,
        "path": str(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
