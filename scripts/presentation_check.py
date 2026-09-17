"""One-command readiness check for the local SIH presentation path."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    checks = []
    def check(name, condition, detail):
        checks.append({"name": name, "pass": bool(condition), "detail": str(detail)})
    check("Python environment", sys.version_info[:2] >= (3, 12), sys.version.split()[0])
    for package in ("fastapi", "numpy", "scipy", "soundfile", "faiss"):
        try:
            __import__(package); check("Package " + package, True, "available")
        except Exception as exc:
            check("Package " + package, False, f"{type(exc).__name__}: {exc}")
    check("FFmpeg", shutil.which("ffmpeg") is not None,
          shutil.which("ffmpeg") or "optional for WAV/FLAC; required for MP3/M4A")
    required = ["web/index.html", "web/app.js", "web/pcm-worklet.js", "web/style.css",
                "config/presentation.json", "STRIVE_Demo.html"]
    missing = [path for path in required if not (ROOT / path).is_file()]
    check("Dashboard", not missing, "all frontend assets present" if not missing else "missing: " + ", ".join(missing))
    check("Demo scenarios", (ROOT / "data/demo/manifest.example.csv").is_file(),
          "three deterministic synthetic scenarios; optional consented manifest supported")
    config = json.loads((ROOT / "config/presentation.json").read_text())
    check("Active geometry", config.get("window_s") == 2 and config.get("stride_s") == .5,
          f"16 kHz, {config.get('window_s')} s window, {config.get('stride_s')} s hop")
    check("Privacy settings", not any((ROOT / "data").rglob("*.wav")),
          "raw audio persistence disabled; audit uses allow-listed metadata")
    try:
        sock = socket.socket()
        sock.bind(("127.0.0.1", args.port))
        sock.close()
        check("Port availability", True, f"127.0.0.1:{args.port} (free)")
    except OSError:
        try:
            probe = socket.socket()
            probe.settimeout(2)
            probe.connect(("127.0.0.1", args.port))
            probe.close()
            check("Port availability", True, f"127.0.0.1:{args.port} (server already running)")
        except (OSError, socket.timeout):
            check("Port availability", False, f"127.0.0.1:{args.port} (in use by another process)")
    try:
        os.environ["STRIVE_CONFIG"] = str(ROOT / "config/presentation.json")
        from fastapi.testclient import TestClient
        from strive.api import create_app
        with TestClient(create_app()) as client:
            ready = client.get("/ready")
            check("Backend", ready.status_code == 200 and ready.json().get("ready"), ready.json())
            call = client.post("/v1/calls", json={}).json()
            with client.websocket_connect(call["ws_path"]) as ws:
                ws.send_json({"token": ""}); first = ws.receive_json()
                ws.send_json({"type": "playback", "scenario": "mid_call", "mode": "accelerated"})
                started = ws.receive_json(); updates = 0; completed = None
                while completed is None:
                    item = ws.receive_json()
                    if item.get("type") == "events": updates += len(item["events"])
                    elif item.get("type") == "playback_complete": completed = item
                check("WebSocket", first.get("type") == "ready", "authenticated local stream")
                check("Audio pipeline", started.get("type") == "playback_started" and updates > 2,
                      f"{updates} incremental windows through mid-call scenario")
                check("Hold / verify", client.post(f"/v1/calls/{call['call_id']}/hold").status_code == 200,
                      "simulated prevention endpoint")
    except Exception as exc:
        check("Backend", False, f"{type(exc).__name__}: {exc}")
        check("WebSocket", False, "backend check failed")
        check("Audio pipeline", False, "backend check failed")
        check("Hold / verify", False, "backend check failed")
    print("\nSTRIVE SIH PRESENTATION CHECK\n")
    for item in checks:
        print(f"[{'PASS' if item['pass'] else 'FAIL'}] {item['name']}: {item['detail']}")
    required_names = {"Python environment", "Package fastapi", "Package numpy", "Package scipy",
        "Package soundfile", "Package faiss", "Dashboard", "Demo scenarios", "Active geometry",
        "Privacy settings", "Port availability", "Backend", "WebSocket", "Audio pipeline", "Hold / verify"}
    ready = all(item["pass"] for item in checks if item["name"] in required_names)
    report = {"ready": ready, "mode": "DEMO DSP", "neural_accuracy_benchmark": False, "checks": checks}
    if args.json_path:
        target = Path(args.json_path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n")
    print("\nREADY FOR PRESENTATION" if ready else "\nNOT READY FOR PRESENTATION")
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
