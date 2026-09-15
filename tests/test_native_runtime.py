"""Native tests are enabled when the release binary is built. Real local sockets."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
from websockets.asyncio.client import connect

from strive.audio import RATE
from strive.config import Settings
from strive.demo import scenario_audio
from strive.engine import Call
from strive.features import DSPExtractor

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'native/audio-runtime/target/release/strive-audio'
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason='Build the native release binary to enable native integration tests')


@pytest.fixture
def native():
    processes = []
    def launch(*args):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        process = subprocess.Popen([sys.executable,'-E','scripts/start_native.py','--port',str(port),
                                    '--audit-path',':memory:',*args], cwd=ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        processes.append(process)
        until = time.monotonic() + 30
        while time.monotonic() < until:
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            try:
                if httpx.get(f'http://127.0.0.1:{port}/health',timeout=.5).status_code == 200:
                    return port, process
            except httpx.HTTPError:
                pass
            time.sleep(.05)
        pytest.fail('Native server startup timed out')
    yield launch
    for process in processes:
        process.terminate()
        try:
            _, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.communicate(); pytest.fail('Native shutdown exceeded ten seconds')
        assert process.returncode == 0, stderr.decode()


async def create(port):
    async with httpx.AsyncClient() as client:
        result = await client.post(f'http://127.0.0.1:{port}/v1/calls',json={})
        result.raise_for_status()
        return result.json()['call_id']


def test_native_scores_match_python_window_for_window(native,index):
    port,_ = native()
    audio = (scenario_audio('steady',6) * 32767).astype('<i2')
    reference = Call(Settings(stride_s=.5),DSPExtractor(),index)
    expected = []
    for seq,offset in enumerate(range(0,len(audio),1600)):
        expected.extend(reference.feed(audio[offset:offset+1600].astype(np.float32)/32768,seq))
    reference.close()
    async def run():
        key=await create(port); actual=[]
        async with connect(f'ws://127.0.0.1:{port}/v1/stream/{key}') as ws:
            await ws.send(json.dumps({'token':'','protocol':'pcm-v2'})); assert json.loads(await ws.recv())['type']=='ready'
            for seq,offset in enumerate(range(0,len(audio),1600)):
                await ws.send(struct.pack('<I',seq)+audio[offset:offset+1600].tobytes())
                want_event=offset+1600>=32000 and (offset+1600-32000)%8000==0
                ack=False; received=False
                while not ack or (want_event and not received):
                    message=json.loads(await asyncio.wait_for(ws.recv(),10))
                    if message['type']=='ack': ack=True
                    if message['type']=='events': actual.extend(message['events']); received=True
            assert [e['session_age_s'] for e in actual] == [e['session_age_s'] for e in expected]
            for a,b in zip(actual,expected):
                assert a['call_id']==key and a['generation']==0
                assert a['track_scores']==b['track_scores']
                assert a['s_risk']==b['s_risk']
                assert a['voiced_seconds']==b['voiced_seconds']
                assert a['bootstrap']==b['bootstrap']
                assert a['capture']['windows_dropped']==0
    asyncio.run(run())


def test_native_receives_during_slow_inference(native):
    port,_=native('--test-delay-ms','600')
    audio=(scenario_audio('steady',6)*32767).astype('<i2')
    async def run():
        key=await create(port)
        async with connect(f'ws://127.0.0.1:{port}/v1/stream/{key}') as ws:
            await ws.send(json.dumps({'token':'','protocol':'pcm-v2'})); await ws.recv()
            last_ack=None; event_count=0
            for seq,offset in enumerate(range(0,len(audio),1600)):
                await ws.send(struct.pack('<I',seq)+audio[offset:offset+1600].tobytes())
                while True:
                    message=json.loads(await asyncio.wait_for(ws.recv(),2))
                    if message['type']=='ack': last_ack=message; break
                    if message['type']=='events': event_count+=1
            assert last_ack['sequence']==59
            assert last_ack['capture']['windows_dropped']>=6
            assert last_ack['capture']['max_depth']==1
            assert event_count==0
    asyncio.run(run())


def test_native_worker_timeout_recovers_without_stopping_audio(native):
    port,server=native('--worker-timeout-ms','200')
    async def run():
        key=await create(port)
        children=[int(x) for x in subprocess.check_output(['ps','--ppid',str(server.pid),'-o','pid=']).split()]
        assert len(children)==1
        worker=children[0]
        audio=(scenario_audio('steady',8)*32767).astype('<i2')
        statuses=[]; events=[]; acks=[]
        async with connect(f'ws://127.0.0.1:{port}/v1/stream/{key}') as ws:
            await ws.send(json.dumps({'token':'','protocol':'pcm-v2'})); await ws.recv()
            os.kill(worker,signal.SIGSTOP)
            async def receive():
                while True:
                    m=json.loads(await ws.recv())
                    if m['type']=='ack': acks.append(m['sequence'])
                    if m['type']=='status': statuses.append(m)
                    if m['type']=='events':
                        events.extend(m['events'])
                        if events[-1]['session_age_s']==8: return
            receiver=asyncio.create_task(receive())
            try:
                for seq,offset in enumerate(range(0,len(audio),1600)):
                    await ws.send(struct.pack('<I',seq)+audio[offset:offset+1600].tobytes())
                    await asyncio.sleep(.05)
                await asyncio.wait_for(receiver,10)
            finally:
                receiver.cancel(); await asyncio.gather(receiver,return_exceptions=True)
            assert len(acks)==80
            assert statuses and events and events[-1]['generation']>=1
            assert all(e['generation']>=1 for e in events)
            assert events[0]['bootstrap']=='blocked_gap'
            with pytest.raises(ProcessLookupError): os.kill(worker,0)
    asyncio.run(run())


def test_native_call_controls_and_capacity(native):
    port,_=native('--max-sessions','1')
    with httpx.Client(base_url=f'http://127.0.0.1:{port}') as c:
        key=c.post('/v1/calls',json={}).json()['call_id']
        assert c.post('/v1/calls',json={}).status_code==429
        assert c.post(f'/v1/calls/{key}/transaction').json()['status']=='held_mock'
        assert c.post(f'/v1/calls/{key}/verify',json={'method':'callback','confirmed':True}).status_code==200
        assert c.post(f'/v1/calls/{key}/transaction').json()['status']=='executed_mock'
        assert c.patch(f'/v1/calls/{key}/context',json={'urgent':True}).status_code==200
        assert c.post(f'/v1/calls/{key}/transaction').json()['status']=='held_mock'
        assert c.delete(f'/v1/calls/{key}').status_code==200
        assert c.get('/metrics').json()['active_sessions']==0


def test_native_auth_origin_and_producer_ownership(native,monkeypatch):
    monkeypatch.setenv('STRIVE_API_TOKEN','native-test-token')
    port,_=native()
    headers={'Authorization':'Bearer native-test-token'}
    with httpx.Client(base_url=f'http://127.0.0.1:{port}') as c:
        assert c.post('/v1/calls',json={}).status_code==401
        assert c.post('/v1/calls',json={},headers={**headers,'Origin':'https://untrusted.invalid'}).status_code==403
        key=c.post('/v1/calls',json={},headers=headers).json()['call_id']
    async def run():
        uri=f'ws://127.0.0.1:{port}/v1/stream/{key}'
        async with connect(uri) as first:
            await first.send(json.dumps({'token':'native-test-token','protocol':'pcm-v2'})); await first.recv()
            async with connect(uri) as second:
                await second.send(json.dumps({'token':'native-test-token','protocol':'pcm-v2'}))
                assert json.loads(await second.recv())['type']=='error'
            await first.send(struct.pack('<Ih',0,-32768))
            assert json.loads(await first.recv())['sequence']==0
            await first.send(struct.pack('<Ih',0,32767))
            assert json.loads(await first.recv())['type']=='error'
    asyncio.run(run())
