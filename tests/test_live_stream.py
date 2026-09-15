"""Exercise the actual WebSocket receiver while inference is deliberately blocked."""
import struct
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from strive.api import create_app
from strive.config import Settings
from strive.demo import signal
from strive.features import DSPExtractor
from strive.streaming import binary_pcm


def packet(sequence):
    return struct.pack('<I', sequence) + (signal(0, .1) * 32767).astype('<i2').tobytes()


class BlockedExtractor(DSPExtractor):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def extract(self, samples):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('Test did not release inference')
        return super().extract(samples)


def test_ingestion_continues_during_inference_and_drops_stale_windows(index):
    extractor = BlockedExtractor()
    with TestClient(create_app(Settings(audit_path=':memory:', stride_s=.5), extractor, index)) as c:
        key = c.post('/v1/calls', json={}).json()['call_id']
        with c.websocket_connect('/v1/stream/' + key) as ws:
            ws.send_json({'token':'', 'protocol':'pcm-v2'})
            assert ws.receive_json()['protocol'] == 'pcm-v2'
            try:
                for sequence in range(20):
                    ws.send_bytes(packet(sequence))
                    assert ws.receive_json()['type'] == 'ack'
                assert extractor.entered.wait(2)
                for sequence in range(20, 60):
                    ws.send_bytes(packet(sequence))
                    ack = ws.receive_json()
                    assert ack['type'] == 'ack' and ack['sequence'] == sequence
                assert ack['capture']['windows_dropped'] == 7
                assert ack['capture']['current_depth'] == 1
                assert not extractor.release.is_set()
            finally:
                extractor.release.set()
            first = ws.receive_json()['events'][0]
            latest = ws.receive_json()['events'][0]
            assert first['session_age_s'] == 2
            assert latest['session_age_s'] == 6
            assert latest['dropped_windows'] == 7
            assert 'CAPTURE_QUEUE_OVERFLOW' in latest['reasons']
            assert latest['channel']['discontinuity_score'] == 0
        assert key not in c.app.state.sessions


def test_second_producer_rejected_without_destroying_first(client):
    key = client.post('/v1/calls', json={}).json()['call_id']
    with client.websocket_connect('/v1/stream/' + key) as first:
        first.send_json({'token':'', 'protocol':'pcm-v2'})
        first.receive_json()
        with client.websocket_connect('/v1/stream/' + key) as second:
            second.send_json({'token':'', 'protocol':'pcm-v2'})
            assert second.receive_json()['type'] == 'error'
        first.send_bytes(packet(0))
        assert first.receive_json()['type'] == 'ack'


@pytest.mark.parametrize('data', [b'', b'12345', bytes(3206), bytes(7)])
def test_binary_packet_bounds(data):
    with pytest.raises(ValueError):
        binary_pcm(data)


def test_binary_pcm_is_exact_at_signed_extremes():
    x = np.array([-32768, -1, 0, 1, 32767], dtype='<i2')
    seq, samples = binary_pcm(struct.pack('<I', 19) + x.tobytes())
    assert seq == 19
    np.testing.assert_array_equal(samples * 32768, x)


def test_duplicate_sequence_closes_and_erases_session(client):
    key = client.post('/v1/calls', json={}).json()['call_id']
    with client.websocket_connect('/v1/stream/' + key) as ws:
        ws.send_json({'token':'', 'protocol':'pcm-v2'}); ws.receive_json()
        ws.send_bytes(packet(0)); assert ws.receive_json()['type'] == 'ack'
        ws.send_bytes(packet(0)); assert ws.receive_json()['type'] == 'error'
    assert key not in client.app.state.sessions
