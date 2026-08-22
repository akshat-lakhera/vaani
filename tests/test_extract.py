from vaani.extract import extract
from vaani.index import StoredChunk


def _hit(pid: str, text: str, score: float = 0.3, query_type: str = "LOCATION"):
    c = StoredChunk(
        chunk_id=pid + ":0",
        text=text,
        embed_text=text,
        parent_id=pid,
        parent_text=text,
        lang="hi",
        query_type=query_type,
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
    assert ext.answer == "दिल्ली"
    assert ext.source.parent_id == "b"


def test_isolates_entity_not_background_sentence():
    text = (
        "दिल्ली, भारत: 2.5 करोड़ के साथ, भारतीय राजधानी दुनिया का "
        "दूसरा सबसे अधिक आबादी वाला शहर है।"
    )
    ext = extract("भारत की राजधानी क्या है?", [_hit("d", text)])
    assert ext is not None
    assert ext.answer == "दिल्ली"


def test_isolates_from_numbered_capital_passage():
    text = (
        "४. दिल्ली, भारत। भारत दुनिया का दूसरा सबसे अधिक आबादी वाला देश है, "
        "और इसकी राजधानी दुनिया का चौथा सबसे बड़ा शहर है, जिसकी आबादी १.२५ करोड़ से अधिक है। "
        "दिल्ली भारत के सबसे पुराने शहरों में से एक है।"
    )
    ext = extract("भारत की राजधानी क्या है?", [_hit("d", text)])
    assert ext is not None
    assert ext.answer == "दिल्ली"
    assert "आबादी" not in ext.answer


def test_does_not_expand_short_span_to_parent():
    ext = extract("भारत की राजधानी क्या है?", [_hit("b", "दिल्ली भारत की राजधानी है।")])
    assert ext is not None
    assert ext.answer == "दिल्ली"
    assert ext.answer != ext.source.parent_text


def test_skips_question_heading_for_definition():
    text = (
        "कॉर्पोरेशन का क्या अर्थ है? व्यवसाय में, कॉर्पोरेट का अर्थ एक व्यवसाय इकाई है "
        "जो एक निगम के रूप में कार्य करती है।"
    )
    ext = extract("कॉर्पोरेशन क्या है?", [_hit("c", text, query_type="DESCRIPTION")])
    assert ext is not None
    assert "?" not in ext.answer
    assert "इकाई" in ext.answer or "निगम" in ext.answer or "व्यवसाय" in ext.answer


def test_does_not_prefer_unrelated_short_entity():
    delhi = "दिल्ली भारत की राजधानी है। यह यमुना नदी के किनारे बसा है।"
    avanti = "अवंती प्राचीन भारत की एक राजधानी थी और उज्जैन का पुराना नाम है।"
    ext = extract(
        "भारत की राजधानी क्या है?",
        [_hit("d", delhi, 0.4), _hit("a", avanti, 0.2)],
    )
    assert ext is not None
    assert ext.answer == "दिल्ली"
    assert ext.source.parent_id == "d"


def test_english_capital_span():
    ext = extract(
        "What is the capital of India?",
        [_hit("e", "New Delhi is the capital of India. It lies on the Yamuna.")],
    )
    assert ext is not None
    assert ext.answer.lower() in {"new delhi", "delhi"}
