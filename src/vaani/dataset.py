"""MSMARCO-XI readers.

Official card: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
We stream. We never assume the whole 55 GB fits on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from vaani.config import Settings, get_settings
from vaani.text import normalize, passage_hash


DATASET_ID = "ai4bharat/MSMARCO-XI"
LANGS = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur")

# Repo card still says jsonl + per-lang configs. Current Hub layout is
# parquet only, and `datasets` no longer runs the loading script.
FILE_STEM = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

# Exact byte sizes from the Hub API (2026-08-17). Used to skip re-downloads.
PARQUET_BYTES = {
    "validation/hinval.parquet": 461888616,
    "validation/marval.parquet": 473618819,
    "train/hintrain.parquet": 3719813179,
    "train/martrain.parquet": 3756780048,
}


def parquet_relpath(lang: str, split: str) -> str:
    if lang not in FILE_STEM:
        raise ValueError(f"unsupported language {lang!r}; want one of {sorted(FILE_STEM)}")
    stem = FILE_STEM[lang]
    if split in {"validation", "val", "dev"}:
        return f"validation/{stem}val.parquet"
    if split in {"train", "training"}:
        return f"train/{stem}train.parquet"
    raise ValueError(f"unsupported split {split!r}")


@dataclass
class Passage:
    passage_id: str
    text: str
    english: str
    selected: bool
    lang: str
    query_id: int
    query_type: str
    split: str


@dataclass
class Example:
    query_id: int
    query: str
    eng_query: str
    answer: str
    eng_answer: str
    query_type: str
    lang: str
    split: str
    passages: list[Passage] = field(default_factory=list)

    @property
    def selected(self) -> list[Passage]:
        return [p for p in self.passages if p.selected]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def parse_example(row: dict[str, Any], *, lang: str, split: str) -> Example:
    passages_raw = row.get("passages") or {}
    if not isinstance(passages_raw, dict):
        passages_raw = {}
    selected = _as_list(passages_raw.get("is_selected"))
    english = _as_list(passages_raw.get("English_passages"))
    translated = _as_list(passages_raw.get("Translated_passages"))
    n = max(len(selected), len(english), len(translated))
    qid = int(row.get("query_id") or 0)
    qtype = str(row.get("query_type") or "")
    parsed: list[Passage] = []
    for i in range(n):
        tr = normalize(translated[i] if i < len(translated) else "")
        en = normalize(english[i] if i < len(english) else "")
        sel = bool(int(selected[i])) if i < len(selected) and selected[i] is not None else False
        text = tr or en
        if not text:
            continue
        pid = f"{lang}-{qid}-{i}-{passage_hash(text)}"
        parsed.append(
            Passage(
                passage_id=pid,
                text=text,
                english=en,
                selected=sel,
                lang=lang,
                query_id=qid,
                query_type=qtype,
                split=split,
            )
        )
    return Example(
        query_id=qid,
        query=normalize(str(row.get("query") or "")),
        eng_query=normalize(str(row.get("Eng_Query") or "")),
        answer=normalize(str(row.get("Answer") or "")),
        eng_answer=normalize(str(row.get("Eng_Answer") or "")),
        query_type=qtype,
        lang=lang,
        split=split,
        passages=parsed,
    )


def local_parquet(lang: str, split: str, settings: Settings | None = None) -> Path:
    """Resumable download of one language/split parquet into data/raw."""
    from huggingface_hub import hf_hub_download

    settings = settings or get_settings()
    rel = parquet_relpath(lang, split)
    dest = settings.raw_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = PARQUET_BYTES.get(rel)
    if dest.exists() and expected and dest.stat().st_size == expected:
        return dest
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=rel,
        repo_type="dataset",
        local_dir=str(settings.raw_dir),
        token=settings.hf_token or None,
    )
    return Path(path)


def stream(
    lang: str,
    split: str = "validation",
    *,
    limit: int | None = None,
    settings: Settings | None = None,
) -> Iterator[Example]:
    settings = settings or get_settings()
    os.environ.setdefault("HF_HOME", str(settings.hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(settings.hf_cache))
    from datasets import load_dataset

    path = local_parquet(lang, split, settings)
    ds = load_dataset("parquet", data_files=str(path), split="train")
    for i, row in enumerate(ds):
        yield parse_example(row, lang=lang, split=split)
        if limit is not None and i + 1 >= limit:
            return
