from vaani.bm25 import BM25
from vaani.index import rrf


def test_bm25_ranks_exact_hindi():
    docs = [
        "पेरिस फ्रांस की राजधानी है",
        "दिल्ली भारत की राजधानी है",
        "टोक्यो जापान की राजधानी है",
    ]
    bm = BM25().fit(docs)
    hits = bm.search("भारत की राजधानी", top_k=2)
    assert hits
    assert hits[0][0] == 1


def test_rrf_fuses_lists():
    fused = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] in {1, 2}
    assert 4 in ids
