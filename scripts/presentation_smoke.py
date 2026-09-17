"""End-to-end smoke against a running STRIVE presentation server."""
import argparse
import asyncio
import base64
from io import BytesIO
import json
import urllib.parse
import urllib.request
from pathlib import Path
import sys

import soundfile as sf
import websockets
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strive.demo import signal


def request(base, path, method="GET", body=None, content_type="application/json"):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(base + path, method=method, data=data,
        headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


async def main_async(base):
    health = request(base, "/health")
    call = request(base, "/v1/calls", "POST", {"context": {"amount_inr": 1_000_000,
        "urgent": True, "new_beneficiary": True}})
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + call["ws_path"]
    scenario_events = []
    async with websockets.connect(ws_url, max_size=22 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"token": ""})); assert json.loads(await ws.recv())["type"] == "ready"
        await ws.send(json.dumps({"type": "playback", "scenario": "mid_call", "mode": "accelerated"}))
        complete = None
        while complete is None:
            message = json.loads(await ws.recv())
            if message["type"] == "events": scenario_events.extend(message["events"])
            elif message["type"] == "playback_complete": complete = message
    assert len(scenario_events) >= 50 and complete["attack_onset_sec"] == 15
    held = request(base, f"/v1/calls/{call['call_id']}/hold", "POST")
    failed = request(base, f"/v1/calls/{call['call_id']}/verify", "POST",
                     {"method": "callback", "outcome": "failed"})
    request(base, f"/v1/calls/{call['call_id']}", "DELETE")

    upload_call = request(base, "/v1/calls", "POST", {})
    audio = BytesIO(); sf.write(audio, signal(0, 4), 16000, format="FLAC")
    filename = urllib.parse.quote("smoke.flac")
    metadata = request(base, f"/v1/calls/{upload_call['call_id']}/upload?filename={filename}",
                       "POST", audio.getvalue(), "application/octet-stream")
    upload_events = []
    ws_url = base.replace("http://", "ws://") + upload_call["ws_path"]
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"token": ""})); await ws.recv()
        await ws.send(json.dumps({"type": "playback", "mode": "accelerated"}))
        done = False
        while not done:
            message = json.loads(await ws.recv())
            if message["type"] == "events": upload_events.extend(message["events"])
            elif message["type"] == "playback_complete": done = True
    assert metadata["raw_audio_saved"] is False and len(upload_events) >= 3
    request(base, f"/v1/calls/{upload_call['call_id']}", "DELETE")
    return {"presentation_smoke": "PASS", "health": health,
        "mid_call_windows": len(scenario_events), "mid_call_first_high_s": complete["first_high_sec"],
        "mid_call_time_to_alert_s": complete["time_to_alert_sec"],
        "hold": held["status"], "verification_failure": failed["status"],
        "upload_format": "FLAC", "upload_windows": len(upload_events),
        "raw_audio_saved": metadata["raw_audio_saved"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output")
    args = parser.parse_args(); report = asyncio.run(main_async(args.url))
    text = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path
        target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
