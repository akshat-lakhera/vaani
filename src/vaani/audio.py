"""Normalize browser clips to 16 kHz mono WAV for Sarvam/ElevenLabs.

Sarvam's sync STT accepts WAV, MP3, AAC, FLAC, OGG — not WebM/Opus, which
is what Chrome's MediaRecorder emits. We convert on the server so the
browser can record whatever it records.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class AudioError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError("ffmpeg is not on PATH — install it to accept browser recordings")
    return exe


def to_wav_16k_mono(audio: bytes, src_name: str = "clip.webm") -> bytes:
    if not audio:
        raise AudioError("empty audio")
    ffmpeg = find_ffmpeg()
    suffix = Path(src_name).suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="vaani-audio-") as td:
        src = Path(td) / f"in{suffix}"
        dst = Path(td) / "out.wav"
        src.write_bytes(audio)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="replace")[:400]
            raise AudioError(f"ffmpeg failed: {err or e}") from e
        except subprocess.TimeoutExpired as e:
            raise AudioError("ffmpeg timed out") from e
        if not dst.exists() or dst.stat().st_size < 44:
            raise AudioError("ffmpeg produced no wav")
        return dst.read_bytes()
