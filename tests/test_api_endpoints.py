import pytest
from fastapi.testclient import TestClient
from vaani.api import app, _pipeline
from vaani.config import Settings
from vaani.harness import Pipeline
from vaani.index import HybridIndex, StoredChunk
from vaani.chunking import Chunk


class MockIndex:
    strategy = "whole"
    size = 2
    faiss_index = None

    def search(self, query, query_vec, top_k=8):
        c1 = StoredChunk(
            chunk_id="hi-101-0-abc",
            text="नई दिल्ली भारत की आधिकारिक राजधानी है और यहाँ संसद भवन स्थित है।",
            embed_text="नई दिल्ली भारत की आधिकारिक राजधानी है और यहाँ संसद भवन स्थित है।",
            parent_id="hi-101-0-abc",
            parent_text="नई दिल्ली भारत की आधिकारिक राजधानी है और यहाँ संसद भवन स्थित है।",
            lang="hi",
            query_type="DESCRIPTION",
            strategy="whole",
            source_query_id=101,
        )
        c2 = StoredChunk(
            chunk_id="hi-101-1-def",
            text="मुंबई महाराष्ट्र की राजधानी और भारत की आर्थिक राजधानी है।",
            embed_text="मुंबई महाराष्ट्र की राजधानी और भारत की आर्थिक राजधानी है।",
            parent_id="hi-101-1-def",
            parent_text="मुंबई महाराष्ट्र की राजधानी और भारत की आर्थिक राजधानी है।",
            lang="hi",
            query_type="DESCRIPTION",
            strategy="whole",
            source_query_id=101,
        )
        return [(c1, 0.9, 0.85, 3.5), (c2, 0.7, 0.65, 2.1)]


@pytest.fixture
def client(monkeypatch):
    settings = Settings()
    mock_pipe = Pipeline(index=MockIndex(), encoder=None, settings=settings)
    import vaani.api
    monkeypatch.setattr(vaani.api, "_pipeline", mock_pipe)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["index_loaded"] is True
    assert data["chunks"] == 2


def test_cors_headers(client):
    res = client.options(
        "/api/ask.json",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") in {"*", "http://localhost:3000"}



def test_ask_json_grounded(client):
    res = client.post("/api/ask.json", json={"text": "भारत की राजधानी क्या है?"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "grounded"
    assert "दिल्ली" in data["answer"]
    assert len(data["citations"]) > 0
    assert data["citations"][0]["passage_id"] == "hi-101-0-abc"


def test_ask_form_text(client):
    res = client.post("/api/ask", data={"text": "भारत की राजधानी क्या है?"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "grounded"
    assert "दिल्ली" in data["answer"]


def test_ask_refuse_unsafe(client):
    res = client.post("/api/ask.json", json={"text": "What is my bank account password?"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "refuse"


def test_compare_strategies_endpoint(client):
    res = client.post(
        "/api/compare",
        json={
            "text": "भारत की राजधानी क्या है?",
            "strategies": ["whole", "sentence", "fixed_256", "window_2"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "भारत की राजधानी क्या है?"
    assert len(data["results"]) == 4
    for r in data["results"]:
        assert r["strategy"] in ["whole", "sentence", "fixed_256", "window_2"]
        assert r["chunks_created"] > 0
        assert r["status"] == "grounded"
        assert "दिल्ली" in r["answer"]


def test_transcribe_missing_key(client):
    # Testing transcribe audio endpoint when STT key is empty
    res = client.post(
        "/api/transcribe",
        files={"audio": ("sample.wav", b"RIFFfakebytes", "audio/wav")},
    )
    # Returns 502 with STT error when key is not configured
    assert res.status_code == 502
