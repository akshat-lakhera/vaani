from vaani.text import jaccard, overlap_precision, passage_hash, tokenize


def test_hindi_not_shattered():
    toks = tokenize("दिल्ली भारत की राजधानी है।")
    assert "दिल्ली" in toks
    assert "राजधानी" in toks
    # The bug we refuse to ship: consonant fragments.
    assert toks != ["द", "ल", "ल"]


def test_tokenize_spans_keep_offsets():
    from vaani.text import normalize, tokenize_spans

    raw = "New Delhi is the capital."
    norm = normalize(raw)
    spans = tokenize_spans(raw)
    assert [t for t, _s, _e in spans] == ["new", "delhi", "is", "the", "capital"]
    assert norm[spans[0][1] : spans[1][2]] == "New Delhi"


def test_english_words():
    assert tokenize("What is the capital of India?") == [
        "what",
        "is",
        "the",
        "capital",
        "of",
        "india",
    ]


def test_marathi_roundtrip():
    toks = tokenize("मुंबई महाराष्ट्राची राजधानी आहे")
    assert "मुंबई" in toks
    assert "महाराष्ट्राची" in toks


def test_fold_bharat_from_measured_stt():
    from vaani.text import fold_stt_transcript

    assert fold_stt_transcript("Bharat की राजधानी क्या है?") == "भारत की राजधानी क्या है?"
    assert fold_stt_transcript("कॉर्पोरेशन क्या है?") == "कॉर्पोरेशन क्या है?"


def test_overlap_and_hash():
    ctx = "The capital of India is New Delhi."
    assert overlap_precision("New Delhi", ctx) == 1.0
    assert overlap_precision("Paris", ctx) == 0.0
    assert passage_hash("abc") == passage_hash("abc")
    assert jaccard("black cat", "black dog") == 1 / 3
