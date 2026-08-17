from vaani.chunking import (
    apply,
    chunk_fixed,
    chunk_parent_child,
    chunk_semantic,
    chunk_sentence,
    chunk_whole,
    chunk_window,
)


PASSAGE = (
    "दिल्ली भारत की राजधानी है। यह देश का सबसे बड़ा महानगरों में से एक है। "
    "यहाँ संसद भवन स्थित है।"
)


def test_whole_is_one_chunk():
    chunks = chunk_whole("p1", PASSAGE, lang="hi", query_type="LOCATION")
    assert len(chunks) == 1
    assert chunks[0].parent_id == "p1"
    assert chunks[0].text.startswith("दिल्ली")


def test_fixed_splits_long_text():
    long = "word " * 200
    chunks = chunk_fixed("p2", long, size=80, overlap=20)
    assert len(chunks) > 2
    assert all(c.parent_text.count("word") > 0 for c in chunks)


def test_sentence_and_window():
    sents = chunk_sentence("p3", PASSAGE)
    assert len(sents) >= 1
    windows = chunk_window("p3", PASSAGE, window=2)
    assert windows
    assert all(c.strategy.startswith("window") for c in windows)


def test_parent_child_shares_parent():
    kids = chunk_parent_child("p4", PASSAGE)
    assert len(kids) >= 2
    assert {c.parent_id for c in kids} == {"p4"}
    assert all(c.parent_text == kids[0].parent_text for c in kids)


def test_semantic_does_not_explode():
    chunks = chunk_semantic("p5", PASSAGE)
    assert 1 <= len(chunks) <= 5


def test_apply_all_named_strategies():
    for name in (
        "whole",
        "fixed_128",
        "fixed_256",
        "sentence",
        "window_2",
        "semantic",
        "parent_child",
        "metadata",
    ):
        chunks = apply(name, "px", PASSAGE, lang="hi", query_type="LOCATION")
        assert chunks, name
        if name == "metadata":
            assert chunks[0].embed_text.startswith("[hi | LOCATION]")
