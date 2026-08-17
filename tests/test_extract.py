from vaani.extract import extract
from vaani.index import StoredChunk


def _hit(pid: str, text: str, score: float = 0.3):
    c = StoredChunk(
        chunk_id=pid + ":0",
        text=text,
        embed_text=text,
        parent_id=pid,
        parent_text=text,
        lang="hi",
        query_type="LOCATION",
        strategy="whole",
    )
    return (c, score, 0.5, 1.0)


def test_picks_matching_sentence():
    hits = [
        _hit("a", "समुद्र नमकीन होता है। मछलियाँ पानी में रहती हैं।"),
        _hit("b", "दिल्ली भारत की राजधानी है। यह यमुना नदी के किनारे बसा है।"),
    ]
    ext = extract("भारत की राजधानी क्या है?", hits)
    assert ext is not None
    assert "दिल्ली" in ext.answer
    assert ext.source.parent_id == "b"
