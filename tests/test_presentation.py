from io import BytesIO
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from strive.config import Settings
from strive.demo import PRESENTATION_SCENARIOS, signal
from strive.engine import aggregate


def wav(seconds=4):
    stream = BytesIO()
    sf.write(stream, signal(0, seconds), 16000, format="WAV")
    return stream.getvalue()


def receive_playback(ws):
    started = ws.receive_json()
    events, complete = [], None
    while complete is None:
        message = ws.receive_json()
        if message["type"] == "events":
            events.extend(message["events"])
            assert 0 <= message["playback"]["progress"] <= 1
        elif message["type"] == "playback_complete":
            complete = message
    return started, events, complete


def test_uploaded_audio_streams_incrementally_and_remains_actionable(client):
    call = client.post("/v1/calls", json={}).json()
    uploaded = client.post(f"/v1/calls/{call['call_id']}/upload?filename=call.wav",
                           content=wav()).json()
    assert uploaded["duration_s"] == 4 and uploaded["raw_audio_saved"] is False
    with client.websocket_connect(call["ws_path"]) as ws:
        ws.send_json({"token": ""}); assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "playback", "mode": "accelerated"})
        started, events, complete = receive_playback(ws)
        assert started["source"] == "call.wav" and len(events) == 3
        assert complete["events"] == 3
        state = client.get(f"/v1/calls/{call['call_id']}").json()
        assert state["upload"] == {} and state["latest"] is not None


def test_presentation_scenario_has_measured_onset_and_multiple_events(client):
    call = client.post("/v1/calls", json={}).json()
    with client.websocket_connect(call["ws_path"]) as ws:
        ws.send_json({"token": ""}); ws.receive_json()
        ws.send_json({"type": "playback", "scenario": "mid_call", "mode": "accelerated"})
        started, events, complete = receive_playback(ws)
        assert started["synthetic"] is True and started["attack_onset_sec"] == 15
        assert len(events) > 20 and all(e["source"] == "synthetic_presentation_scenario" for e in events)
        if complete["first_high_sec"] is not None:
            assert complete["time_to_alert_sec"] == complete["first_high_sec"] - 15


def test_verification_failure_and_review_keep_action_held(client):
    call = client.post("/v1/calls", json={}).json()["call_id"]
    assert client.post(f"/v1/calls/{call}/hold").json()["status"] == "held_mock"
    failed = client.post(f"/v1/calls/{call}/verify", json={"method": "callback", "outcome": "failed"})
    assert failed.json()["status"] == "blocked_mock"
    assert client.post(f"/v1/calls/{call}/transaction").json()["status"] == "blocked_mock"
    review = client.post(f"/v1/calls/{call}/verify", json={"method": "supervisor", "outcome": "review"})
    assert review.json()["status"] == "supervisor_review_mock"
    assert client.get(f"/v1/calls/{call}").json()["hold_latched"] is True
    audit = client.get(f"/v1/calls/{call}/audit").json()["events"]
    assert {e["kind"] for e in audit} >= {"action.held", "verification.failed", "verification.review"}
    assert "pcm" not in str(audit).lower() and "embedding" not in str(audit).lower()


def test_channel_reliability_differential_changes_fusion_score():
    """Branch-specific reliability weights must change fusion when channel degrades."""
    base = aggregate([.8, .2, .4], 30, None, reliability=[1, 1, 1])
    # Quality=0.5 with default weights: artifact penalized least, coherence most
    degraded = aggregate([.8, .2, .4], 30, None, reliability=[0.75, 0.6, 0.5])
    assert degraded[0] != pytest.approx(base[0])
    # Uniform reliability [1,1,1] always produces the same result as no reliability
    assert base[0] == pytest.approx(aggregate([.8, .2, .4], 30, None, reliability=[1, 1, 1])[0])
    # All-zero reliability abstains
    assert aggregate([.8, .2, .4], 30, None, reliability=[0, 0, 0])[0] is None


def test_scenario_manifest_schema_and_language_readiness():
    assert set(PRESENTATION_SCENARIOS) == {"genuine", "spoof", "mid_call"}
    rows = list(csv.DictReader((Path(__file__).parents[1] / "data/demo/manifest.example.csv").open()))
    assert {row["id"] for row in rows} == set(PRESENTATION_SCENARIOS)
    assert {row["language"] for row in rows} == {"en"}
    assert Settings(stride_s=.5).stride_s == .5
