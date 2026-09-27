import base64
import numpy as np
import pytest
from fastapi.testclient import TestClient
from strive.api import create_app
from strive.config import Settings
from strive.demo import make_demo_index, scenario_audio
from strive.features import DSPExtractor


@pytest.fixture(scope="module")
def index():
    return make_demo_index()


@pytest.fixture
def client(index):
    app = create_app(Settings(audit_path=":memory:"), DSPExtractor(), index)
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "demo"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["model_ready"] is True
    assert data["reference_index_ready"] is True
    assert data["inference_ok"] is True


def test_config_endpoint(client):
    response = client.get("/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "demo"
    assert "languages" in data
    assert "scenarios" in data


def test_create_call(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    assert response.status_code == 201
    data = response.json()
    assert "call_id" in data
    assert data["sample_rate"] == 16000
    return data["call_id"]


def test_full_pipeline_genuine(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("genuine", 10)
    events = []
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        response = client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })
        assert response.status_code == 200
        events.extend(response.json()["events"])

    assert len(events) > 0
    assert any(e["state"] in ("LOW", "REVIEW", "HIGH", "CRITICAL") for e in events)


def test_full_pipeline_spoof(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("spoof", 10)
    events = []
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        response = client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })
        assert response.status_code == 200
        events.extend(response.json()["events"])

    assert len(events) > 0


def test_full_pipeline_mid_call(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("mid_call", 25)
    events = []
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        response = client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })
        assert response.status_code == 200
        events.extend(response.json()["events"])

    assert len(events) > 0
    states = [e["state"] for e in events]
    assert "LOW" in states or "REVIEW" in states or "HIGH" in states or "CRITICAL" in states


def test_verification_workflow(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("spoof", 5)
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })

    response = client.post(f"/v1/calls/{call_id}/hold")
    assert response.status_code == 200
    assert response.json()["status"] == "held_mock"

    response = client.post(f"/v1/calls/{call_id}/verify", json={
        "method": "callback",
        "confirmed": True,
        "outcome": "verified"
    })
    assert response.status_code == 200
    assert response.json()["outcome"] == "verified"

    response = client.post(f"/v1/calls/{call_id}/transaction")
    assert response.status_code == 200


def test_transaction_blocked(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("spoof", 5)
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })

    response = client.post(f"/v1/calls/{call_id}/verify", json={
        "method": "callback",
        "confirmed": False,
        "outcome": "failed"
    })
    assert response.status_code == 200

    response = client.post(f"/v1/calls/{call_id}/transaction")
    assert response.status_code == 200
    assert response.json()["status"] == "blocked_mock"


def test_audit_trail(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    audio = scenario_audio("genuine", 3)
    for i, start in enumerate(range(0, len(audio), 16000)):
        chunk = audio[start:start + 16000]
        if len(chunk) < 16000:
            break
        client.post(f"/v1/calls/{call_id}/chunks", json={
            "sequence": i,
            "pcm_s16le": base64.b64encode(chunk.tobytes()).decode()
        })

    response = client.get(f"/v1/calls/{call_id}/audit")
    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) > 0
    assert any(e["kind"] == "call.started" for e in events)


def test_context_update(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    response = client.patch(f"/v1/calls/{call_id}/context", json={
        "amount_inr": 500000,
        "urgent": True,
        "new_beneficiary": True
    })
    assert response.status_code == 200
    assert response.json()["updated"] is True


def test_call_state(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    response = client.get(f"/v1/calls/{call_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["call_id"] == call_id
    assert data["source"] == "stream"


def test_delete_call(client):
    response = client.post("/v1/calls", json={"language": "auto", "context": {}})
    call_id = response.json()["call_id"]

    response = client.delete(f"/v1/calls/{call_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "ended"

    response = client.get(f"/v1/calls/{call_id}")
    assert response.status_code == 404


def test_demo_endpoint(client):
    response = client.post("/v1/demo/steady", json={"language": "auto", "context": {}})
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert len(data["events"]) > 0


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "strive_active_sessions" in response.text


def test_status_endpoint(client):
    response = client.get("/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "healthy"
    assert "latency_ms" in data
