"""Speech-to-text adapters: Sarvam Saaras v3 (default) and ElevenLabs Scribe.

The brief says pick one. We implement both behind one interface so a key
swap is a config change, not a rewrite. Default is Sarvam — MSMARCO-XI
is Indic, and Saaras is built for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from vaani.config import Settings, get_settings


class STTError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language: str
    provider: str
    raw: dict


class SarvamSTT:
    url = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, settings: Settings) -> None:
        if not settings.sarvam_api_key:
            raise STTError("SARVAM_API_KEY is not set")
        self.settings = settings

    def transcribe(self, audio: bytes, filename: str = "audio.wav", mime: str = "audio/wav") -> Transcript:
        headers = {"api-subscription-key": self.settings.sarvam_api_key}
        files = {"file": (filename, audio, mime)}
        data = {"model": "saaras:v3", "mode": "transcribe"}
        try:
            with httpx.Client(timeout=self.settings.stt_timeout_s) as client:
                resp = client.post(self.url, headers=headers, files=files, data=data)
        except httpx.HTTPError as e:
            raise STTError(f"sarvam network error: {e}") from e
        if resp.status_code >= 400:
            raise STTError(f"sarvam HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        text = (body.get("transcript") or body.get("text") or "").strip()
        if not text:
            raise STTError("sarvam returned empty transcript")
        lang = body.get("language_code") or ""
        return Transcript(text=text, language=lang, provider="sarvam", raw=body)


class ElevenLabsSTT:
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, settings: Settings) -> None:
        if not settings.elevenlabs_api_key:
            raise STTError("ELEVENLABS_API_KEY is not set")
        self.settings = settings

    def transcribe(self, audio: bytes, filename: str = "audio.wav", mime: str = "audio/wav") -> Transcript:
        headers = {"xi-api-key": self.settings.elevenlabs_api_key}
        files = {"file": (filename, audio, mime)}
        data = {"model_id": "scribe_v2"}
        try:
            with httpx.Client(timeout=self.settings.stt_timeout_s) as client:
                resp = client.post(self.url, headers=headers, files=files, data=data)
        except httpx.HTTPError as e:
            raise STTError(f"elevenlabs network error: {e}") from e
        if resp.status_code >= 400:
            raise STTError(f"elevenlabs HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        text = (body.get("text") or body.get("transcript") or "").strip()
        if not text:
            raise STTError("elevenlabs returned empty transcript")
        lang = body.get("language_code") or body.get("language") or ""
        return Transcript(text=text, language=str(lang), provider="elevenlabs", raw=body)


def get_stt(settings: Settings | None = None) -> SarvamSTT | ElevenLabsSTT:
    settings = settings or get_settings()
    provider = (settings.stt_provider or "sarvam").strip().lower()
    if provider in {"eleven", "elevenlabs", "scribe"}:
        return ElevenLabsSTT(settings)
    return SarvamSTT(settings)


def prepare_audio(audio: bytes, filename: str = "audio.webm") -> tuple[bytes, str, str]:
    """Return (wav_bytes, filename, mime) ready for the STT API."""
    from vaani.audio import AudioError, to_wav_16k_mono

    try:
        wav = to_wav_16k_mono(audio, src_name=filename)
        return wav, "clip.wav", "audio/wav"
    except AudioError as e:
        lower = filename.lower()
        if lower.endswith((".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a")):
            mime = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".aac": "audio/aac",
                ".m4a": "audio/mp4",
            }[Path(lower).suffix]
            return audio, filename, mime
        raise STTError(str(e)) from e


def transcribe_with_retry(
    audio: bytes,
    filename: str = "audio.webm",
    mime: str = "audio/webm",
    settings: Settings | None = None,
) -> Transcript:
    settings = settings or get_settings()
    payload, name, send_mime = prepare_audio(audio, filename=filename)
    last: Exception | None = None
    attempts = max(1, settings.stt_retries)
    for i in range(attempts):
        try:
            return get_stt(settings).transcribe(payload, filename=name, mime=send_mime)
        except STTError as e:
            last = e
            if i == attempts - 1:
                raise
    raise STTError(str(last))
