import base64
from io import BytesIO
import json
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from strive.api import create_app
from strive.config import Settings
from strive.demo import signal
from strive.features import DSPExtractor


def frame(sequence):
    data = (signal(0, 1) * 32767).astype("<i2").tobytes()
    return {"sequence":sequence, "pcm_s16le":base64.b64encode(data).decode()}


def test_http_stream_and_private_audit(client):
    call = client.post('/v1/calls', json={}).json()['call_id']
    for i in range(7):
        r = client.post(f'/v1/calls/{call}/chunks', json=frame(i))
        assert r.status_code == 200
    assert r.json()['events'][0]['bootstrap'] == 'trusted'
    audit = client.get(f'/v1/calls/{call}/audit').json()
    text = json.dumps(audit)
    assert 'pcm' not in text and 'embedding' not in text and 'pitch_hz' not in text
    assert client.delete(f'/v1/calls/{call}').json()['ephemeral_state_deleted']
    assert client.post(f'/v1/calls/{call}/chunks', json=frame(7)).status_code == 404


def test_websocket_same_engine_and_cleanup(client):
    call = client.post('/v1/calls', json={}).json()['call_id']
    with client.websocket_connect(f'/v1/stream/{call}') as ws:
        ws.send_json({'token':''}); assert ws.receive_json()['type'] == 'ready'
        for i in range(6):
            ws.send_json(frame(i)); result = ws.receive_json()
            assert result['sequence'] == i
        assert result['events'][0]['bootstrap'] == 'trusted'
    assert client.post(f'/v1/calls/{call}/chunks', json=frame(7)).status_code == 422


def test_invalid_and_duplicate_frames(client):
    call = client.post('/v1/calls', json={}).json()['call_id']
    assert client.post(f'/v1/calls/{call}/chunks', json={'sequence':0,'pcm_s16le':'@@@@'}).status_code == 422
    assert client.post(f'/v1/calls/{call}/chunks', json=frame(0)).status_code == 200
    assert client.post(f'/v1/calls/{call}/chunks', json=frame(0)).status_code == 422


def test_transfer_requires_mock_verification(client):
    r = client.post('/v1/demo/switch', json={'context':{'amount_inr':4000000,'urgent':True}})
    assert r.status_code == 200
    call = r.json()['call_id']
    assert client.post(f'/v1/calls/{call}/transaction').json()['status'] == 'held_mock'
    assert client.post(f'/v1/calls/{call}/verify', json={'method':'callback','confirmed':False}).status_code == 422
    client.post(f'/v1/calls/{call}/verify', json={'method':'callback','confirmed':True})
    assert client.post(f'/v1/calls/{call}/transaction').json()['status'] == 'executed_mock'
    client.patch(f'/v1/calls/{call}/context', json={'amount_inr':8000000})
    assert client.post(f'/v1/calls/{call}/transaction').json()['status'] == 'held_mock'


def test_upload_resampling_and_no_audio_persistence(client, tmp_path):
    out = BytesIO(); sf.write(out, np.tile([.1, -.1], 16000), 8000, format='WAV')
    response = client.post('/v1/analyze?language=hi', content=out.getvalue())
    assert response.status_code == 200
    result = response.json()
    assert result['duration_s'] == 4 and len(result['events']) == 3
    assert not result['raw_audio_saved']
    assert not client.app.state.sessions


def test_authentication_and_origin(index):
    with TestClient(create_app(Settings(api_token='test-token', audit_path=':memory:'), DSPExtractor(), index)) as c:
        assert c.post('/v1/calls',json={}).status_code == 401
        assert c.get('/health').status_code == 200
        headers = {'Authorization':'Bearer test-token'}
        assert c.post('/v1/calls',json={},headers=headers).status_code == 201
        assert c.post('/v1/calls',json={},headers={**headers,'Origin':'https://untrusted.example'}).status_code == 403


def test_health_and_metrics(client):
    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/ready').json()['deepfake_detection_validated'] is False
    assert 'strive_windows_total' in client.get('/metrics').text
    assert client.get('/').status_code == 200
    assert client.get('/assets/pcm-worklet.js').status_code == 200


def test_rest_drop_metrics_are_counted_once(index):
    settings = Settings(audit_path=":memory:", stride_s=.5, capture_queue_windows=1)
    with TestClient(create_app(settings, DSPExtractor(), index)) as client:
        call = client.post('/v1/calls', json={}).json()['call_id']
        responses = [client.post(f'/v1/calls/{call}/chunks', json=frame(i)) for i in range(3)]
        assert all(response.status_code == 200 for response in responses)
        assert responses[-1].json()['events'][0]['dropped_windows'] == 1
        metrics = client.get('/metrics').text
        assert 'strive_dropped_windows_total 1\n' in metrics
        assert 'strive_dropped_events_total' not in metrics


def test_context_not_acoustic_evidence(client):
    c = client.post('/v1/calls', json={}).json()['call_id']
    for i in range(7): result = client.post(f'/v1/calls/{c}/chunks', json=frame(i)).json()
    before = result['events'][0]['s_risk']
    client.patch(f'/v1/calls/{c}/context', json={'amount_inr':4000000,'urgent':True})
    assert client.app.state.sessions[c].latest['s_risk'] == before
