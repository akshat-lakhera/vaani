from vaani.config import Settings
from vaani.harness import Pipeline


class _FakeIndex:
    strategy = "whole"


def test_harness_refuse_short_circuits():
    pipe = Pipeline(index=_FakeIndex(), encoder=None, settings=Settings())
    resp = pipe.ask_text("What is my bank account password?")
    assert resp.status == "refuse"
    assert resp.timings.embed_ms == 0.0
    assert "cannot" in resp.answer.lower() or "नहीं" in resp.answer
