from pathlib import Path

from vaani.audio import to_wav_16k_mono


def test_aiff_from_say_converts_to_wav(tmp_path: Path):
    import shutil
    import subprocess

    say = shutil.which("say")
    if not say:
        return
    aiff = tmp_path / "t.aiff"
    subprocess.run([say, "-v", "Lekha", "-o", str(aiff), "नमस्ते"], check=True)
    wav = to_wav_16k_mono(aiff.read_bytes(), src_name="t.aiff")
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:16]
    assert len(wav) > 1000
