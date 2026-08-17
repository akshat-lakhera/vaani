from vaani.extract import extract
from vaani.index import StoredChunk
from vaani.relevance import attachment_conflict, rerank_hits


MUMBAI = (
    "मुंबई (/mʊmˈbaɪ/; बॉम्बे के रूप में भी जाना जाता है, जो 1995 तक आधिकारिक नाम था) "
    "भारत के महाराष्ट्र राज्य की राजधानी है।"
)
DELHI = (
    "४. दिल्ली, भारत। भारत दुनिया का दूसरा सबसे अधिक आबादी वाला देश है, "
    "और इसकी राजधानी दुनिया का चौथा सबसे बड़ा शहर है, जिसकी आबादी १.२५ करोड़ से अधिक है।"
)
QUERY = "भारत की राजधानी क्या है?"


def test_attachment_flags_state_capital_as_conflict():
    assert attachment_conflict(QUERY, MUMBAI)
    assert not attachment_conflict(QUERY, DELHI)


def test_weather_goa_still_conflicts_on_other_city():
    ny = "न्यूयॉर्क में मार्च में मौसम कैसा है? यह पोस्ट न्यूयॉर्क शहर के मार्च मौसम का सारांश है।"
    # no genitive "X की मौसम" in the query, so attachment does not fire
    assert not attachment_conflict("आज गोवा में मौसम कैसा है?", ny)


def test_rerank_puts_non_conflicting_hit_first():
    def hit(pid, text, fused, dense):
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
        return (c, fused, dense, 1.0)

    hits = [hit("mumbai", MUMBAI, 0.0325, 0.871), hit("delhi", DELHI, 0.0323, 0.863)]
    ranked = rerank_hits(QUERY, hits)
    assert ranked[0][0].parent_id == "delhi"


def test_extract_skips_conflicting_sentence():
    def hit(pid, text, fused=0.3):
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
        return (c, fused, 0.5, 1.0)

    ext = extract(QUERY, [hit("m", MUMBAI, 0.4), hit("d", DELHI, 0.3)])
    assert ext is not None
    assert "मुंबई" not in ext.answer
    assert "दिल्ली" in ext.answer or "राजधानी" in ext.answer
