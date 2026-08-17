from vaani.config import Settings
from vaani.harness import Pipeline
from vaani.stt import STTError, prepare_audio


class _FakeIndex:
    strategy = "whole"
    faiss_index = None

    def search(self, *args, **kwargs):
        return []


def test_prepare_rejects_garbage_without_ffmpeg_passthrough():
    try:
        prepare_audio(b"not-an-audio-file", filename="clip.webm")
    except STTError as e:
        assert "ffmpeg" in str(e).lower() or "failed" in str(e).lower()
    else:
        raise AssertionError("expected STTError for garbage webm")


def test_ask_audio_without_key_refuses():
    pipe = Pipeline(index=_FakeIndex(), encoder=None, settings=Settings(sarvam_api_key=""))
    resp = pipe.ask_audio(b"RIFF....", filename="x.wav", mime="audio/wav")
    assert resp.status == "refuse"
    assert resp.timings.stt_ms is not None
    assert "SARVAM" in resp.reason or "stt" in resp.reason.lower() or "failed" in resp.answer.lower()
