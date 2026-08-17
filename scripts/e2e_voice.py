#!/usr/bin/env python3
"""Real voice-path check: synthesize speech → wav → Sarvam (if keyed) → RAG.

Does not invent a transcript. If SARVAM_API_KEY is missing the STT step
is reported as skipped and the process exits 2.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vaani.audio import to_wav_16k_mono  # noqa: E402
from vaani.config import get_settings  # noqa: E402
from vaani.harness import Pipeline  # noqa: E402


PHRASE = "कॉर्पोरेशन क्या है?"
VOICE = "Lekha"


def synthesize(text: str, dest: Path) -> Path:
    say = shutil.which("say")
    if not say:
        raise SystemExit("macOS `say` is required to synthesize a test clip")
    aiff = dest.with_suffix(".aiff")
    subprocess.run([say, "-v", VOICE, "-o", str(aiff), text], check=True)
    wav = dest.with_suffix(".wav")
    raw = to_wav_16k_mono(aiff.read_bytes(), src_name="clip.aiff")
    wav.write_bytes(raw)
    return wav


def main() -> int:
    settings = get_settings()
    out = settings.reports_dir / "e2e_voice.json"
    report: dict = {"phrase": PHRASE, "voice": VOICE}

    with tempfile.TemporaryDirectory(prefix="vaani-e2e-") as td:
        wav = synthesize(PHRASE, Path(td) / "q")
        report["wav_bytes"] = wav.stat().st_size
        clip = wav.read_bytes()

        pipe = Pipeline.load(settings)
        text_resp = pipe.ask_text(PHRASE, language="hi-IN")
        report["text_path"] = {
            "status": text_resp.status,
            "rag_ms": round(text_resp.timings.total_rag_ms, 2),
            "within_budget": text_resp.within_budget,
            "answer": text_resp.answer[:240],
        }

        if not settings.sarvam_api_key:
            report["stt"] = {
                "ran": False,
                "reason": "SARVAM_API_KEY is empty — not faking a transcript",
            }
            out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"wrote {out}", file=sys.stderr)
            return 2

        voice_resp = pipe.ask_audio(clip, filename="clip.wav", mime="audio/wav")
        report["stt"] = {
            "ran": True,
            "provider": "sarvam",
            "transcript": voice_resp.transcript,
            "language": voice_resp.language,
            "stt_ms": voice_resp.timings.stt_ms,
            "status": voice_resp.status,
            "rag_ms": round(voice_resp.timings.total_rag_ms, 2),
            "within_budget": voice_resp.within_budget,
            "reason": voice_resp.reason,
            "answer": voice_resp.answer[:240],
        }
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {out}", file=sys.stderr)
        if voice_resp.status == "refuse" and "stt" in (voice_resp.reason or "").lower():
            return 1
        if not (voice_resp.transcript or "").strip():
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
