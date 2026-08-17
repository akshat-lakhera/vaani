from vaani.dataset import parse_example, parquet_relpath


def test_parquet_paths_match_hub_layout():
    assert parquet_relpath("hi", "validation") == "validation/hinval.parquet"
    assert parquet_relpath("mr", "train") == "train/martrain.parquet"
    assert parquet_relpath("gu", "validation") == "validation/gujval.parquet"


def test_parse_documented_schema():
    row = {
        "query_id": 1185869,
        "query_type": "DESCRIPTION",
        "query": "मेनहाटन परियोजना?",
        "Answer": "एक उत्तर",
        "Eng_Query": "manhattan project?",
        "Eng_Answer": "an answer",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["The Manhattan Project was...", "Unrelated."],
            "Translated_passages": ["मैनहटन परियोजना...", "असंबंधित।"],
        },
    }
    ex = parse_example(row, lang="hi", split="validation")
    assert ex.query_id == 1185869
    assert len(ex.passages) == 2
    assert len(ex.selected) == 1
    assert ex.selected[0].text.startswith("मैनहटन")
    assert ex.selected[0].english.startswith("The Manhattan")
