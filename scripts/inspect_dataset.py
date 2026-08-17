#!/usr/bin/env python3
"""Inspect MSMARCO-XI without downloading the 55 GB dump.

Prints schema, one example, and running stats over a streamed sample.
Writes data/reports/dataset_inspect.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaani.config import get_settings  # noqa: E402
from vaani.dataset import stream  # noqa: E402
from vaani.text import tokenize  # noqa: E402


def summarize(lang: str, split: str, limit: int) -> dict:
    n = 0
    n_pass = 0
    n_sel = 0
    empty_ans = 0
    q_chars: list[int] = []
    p_chars: list[int] = []
    p_toks: list[int] = []
    types: Counter[str] = Counter()
    sel_unique: set[str] = set()
    first: dict | None = None

    for ex in stream(lang, split, limit=limit):
        n += 1
        types[ex.query_type or "UNKNOWN"] += 1
        q_chars.append(len(ex.query))
        if not ex.answer:
            empty_ans += 1
        if first is None:
            first = {
                "query_id": ex.query_id,
                "lang": ex.lang,
                "query_type": ex.query_type,
                "query": ex.query,
                "eng_query": ex.eng_query,
                "answer": ex.answer[:400],
                "eng_answer": ex.eng_answer[:400],
                "n_passages": len(ex.passages),
                "n_selected": len(ex.selected),
                "first_selected": ex.selected[0].text[:500] if ex.selected else "",
            }
        for p in ex.passages:
            n_pass += 1
            p_chars.append(len(p.text))
            p_toks.append(len(tokenize(p.text)))
            if p.selected:
                n_sel += 1
                sel_unique.add(p.text)

    def pct(xs: list[int], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        i = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
        return float(s[i])

    return {
        "lang": lang,
        "split": split,
        "n_examples": n,
        "n_passages": n_pass,
        "n_selected": n_sel,
        "unique_selected": len(sel_unique),
        "empty_answers": empty_ans,
        "query_types": dict(types.most_common()),
        "query_chars": {
            "p50": pct(q_chars, 0.5),
            "p90": pct(q_chars, 0.9),
            "max": max(q_chars) if q_chars else 0,
        },
        "passage_chars": {
            "p50": pct(p_chars, 0.5),
            "p90": pct(p_chars, 0.9),
            "max": max(p_chars) if p_chars else 0,
            "mean": statistics.fmean(p_chars) if p_chars else 0,
        },
        "passage_tokens": {
            "p50": pct(p_toks, 0.5),
            "p90": pct(p_toks, 0.9),
            "mean": statistics.fmean(p_toks) if p_toks else 0,
        },
        "first_example": first,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="hi,mr")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    settings = get_settings()
    reports = []
    for lang in [p.strip() for p in args.langs.split(",") if p.strip()]:
        print(f"\n=== {lang} / {args.split} (limit={args.limit}) ===", flush=True)
        rep = summarize(lang, args.split, args.limit)
        reports.append(rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "first_example"}, indent=2, ensure_ascii=False))
        print("--- first example ---")
        print(json.dumps(rep["first_example"], indent=2, ensure_ascii=False))

    out = settings.reports_dir / "dataset_inspect.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
