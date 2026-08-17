#!/usr/bin/env python3
"""Hit a running Vaani URL the way a judge would. Records what actually happened.

Does not claim Sarvam or the 200ms budget. Audio POST only checks that the
server rejects a clip cleanly when no key is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    base = args.base.rstrip("/")
    report: dict = {"base": base, "steps": []}

    def step(name: str, **kw):
        rec = {"name": name, **kw}
        report["steps"].append(rec)
        print(json.dumps(rec, ensure_ascii=False))

    headers = {"ngrok-skip-browser-warning": "1"}
    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        t0 = time.perf_counter()
        h = client.get(f"{base}/api/health")
        step("health", http=h.status_code, ms=round((time.perf_counter() - t0) * 1000, 1), body=h.json() if h.headers.get("content-type", "").startswith("application/json") else h.text[:200])

        t0 = time.perf_counter()
        home = client.get(f"{base}/")
        step("home", http=home.status_code, ms=round((time.perf_counter() - t0) * 1000, 1), has_mic='id="mic"' in home.text)

        t0 = time.perf_counter()
        ask = client.post(f"{base}/api/ask", data={"text": "भारत की राजधानी क्या है?"})
        body = ask.json()
        step(
            "ask_capital",
            http=ask.status_code,
            ms=round((time.perf_counter() - t0) * 1000, 1),
            status=body.get("status"),
            rag_ms=body.get("timings", {}).get("total_rag_ms"),
            within_budget=body.get("within_budget"),
            answer=(body.get("answer") or "")[:200],
            mentions_delhi="दिल्ली" in (body.get("answer") or ""),
            mentions_mumbai="मुंबई" in (body.get("answer") or ""),
        )

        t0 = time.perf_counter()
        refuse = client.post(f"{base}/api/ask", data={"text": "What is my bank account password?"})
        rb = refuse.json()
        step("refuse_password", http=refuse.status_code, ms=round((time.perf_counter() - t0) * 1000, 1), status=rb.get("status"), reason=rb.get("reason"))

        wav = ROOT / "data" / "reports" / "smoke.wav"
        # tiny invalid wav is enough to exercise the audio route
        audio_status = None
        audio_reason = None
        try:
            t0 = time.perf_counter()
            ar = client.post(f"{base}/api/ask", files={"audio": ("clip.wav", b"RIFF....notwav", "audio/wav")})
            ab = ar.json()
            audio_status = ab.get("status")
            audio_reason = ab.get("reason")
            step("ask_audio", http=ar.status_code, ms=round((time.perf_counter() - t0) * 1000, 1), status=audio_status, reason=(audio_reason or "")[:180])
        except Exception as e:  # noqa: BLE001
            step("ask_audio", error=str(e))

    health_ok = report["steps"][0].get("http") == 200 and (report["steps"][0].get("body") or {}).get("ok")
    capital = next(s for s in report["steps"] if s["name"] == "ask_capital")
    capital_ok = capital.get("status") == "grounded" and capital.get("mentions_delhi") and not capital.get("mentions_mumbai")
    refuse_ok = next(s for s in report["steps"] if s["name"] == "refuse_password").get("status") == "refuse"
    report["pass"] = bool(health_ok and capital_ok and refuse_ok)
    report["note"] = (
        "Persistent-deploy smoke. Sarvam STT not asserted. "
        "200ms budget not asserted (HTTP wall time includes embed jitter)."
    )
    out = Path(args.out) if args.out else ROOT / "data" / "reports" / "deploy_smoke.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PASS" if report["pass"] else "FAIL", out)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
