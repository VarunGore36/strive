"""Measure paced PCM over an actual loopback WebSocket, including return transport.

No microphone or audio is saved. DSP is a transport fixture, not a detection test.
Use --delay-ms to simulate slow model inference without changing detector code.
"""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import uvicorn
from websockets.asyncio.client import connect

from strive.api import create_app
from strive.config import Settings
from strive.demo import make_demo_index, scenario_audio
from strive.features import DSPExtractor
from strive.telemetry import percentiles, aggregate_stages
from stream_wav import hardware, git_commit


class DelayedExtractor(DSPExtractor):
    def __init__(self, delay_ms):
        self.delay_s = delay_ms / 1000

    def extract(self, samples):
        if self.delay_s:
            time.sleep(self.delay_s)
        return super().extract(samples)


async def measure(port, args, call_id=None):
    key = call_id
    if key is None:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f'http://127.0.0.1:{port}/v1/calls', json={})
            response.raise_for_status()
            key = response.json()['call_id']
    audio = (scenario_audio('steady', args.seconds) * 32767).astype('<i2')
    frame = 320
    sent, rtts, ingests, events, arrivals, late = {}, [], [], [], [], []
    async with connect(f'ws://127.0.0.1:{port}/v1/stream/{key}', compression=None) as ws:
        await ws.send(json.dumps({'token':'', 'protocol':'pcm-v2'}))
        ready = json.loads(await ws.recv())
        assert ready['type'] == 'ready'
        origin = time.perf_counter()
        last_end = args.window + int((args.seconds - args.window) / args.hop) * args.hop

        async def receive():
            while True:
                message = json.loads(await ws.recv())
                now = time.perf_counter()
                if message['type'] == 'error':
                    raise RuntimeError(message)
                if message['type'] == 'ack':
                    rtts.append((now - sent.pop(message['sequence'])) * 1000)
                    ingests.append(message['ingest_ms'])
                if message['type'] == 'events':
                    for event in message['events']:
                        events.append(event)
                        arrivals.append((now - origin - event['session_age_s']) * 1000)
                    if events[-1]['session_age_s'] >= last_end:
                        return

        receiver = asyncio.create_task(receive())
        try:
            for sequence, offset in enumerate(range(0, len(audio), frame)):
                target = origin + min(offset + frame, len(audio)) / 16000
                await asyncio.sleep(max(0., target - time.perf_counter()))
                sent[sequence] = time.perf_counter()
                late.append(max(0., sent[sequence] - target) * 1000)
                await ws.send(struct.pack('<I', sequence) + audio[offset:offset + frame].tobytes())
            await asyncio.wait_for(receiver, timeout=max(15, args.delay_ms / 1000 * 4))
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    hw = {} if getattr(args, 'collect_raw', False) else hardware()
    hw.update(device='cpu', gpu=None)  # This benchmark actually runs DSP on CPU.
    report = {'hardware': hw, 'git_commit': None if getattr(args, 'collect_raw', False) else git_commit(), 'protocol':'pcm-v2',
            'audio_seconds':args.seconds, 'window_s':args.window, 'hop_s':args.hop,
            'frame_ms':20, 'injected_inference_ms':args.delay_ms,
            'model_version':'dsp-surrogate-v1', 'demo_only':True,
            'frames_sent':sequence + 1, 'frames_acknowledged':len(rtts),
            'windows_scored':len(events), 'capture':events[-1]['capture'],
            'latency_ms':{'ack_roundtrip':percentiles(rtts), 'ingest':percentiles(ingests),
                          'window_end_to_client':percentiles(arrivals),
                          'compute':percentiles([e['latency_ms']['compute'] for e in events]),
                          'queue':percentiles([e['latency_ms']['queue'] for e in events]),
                          'sender_lateness':percentiles(late)},
            'first_result_from_stream_start_ms':events[0]['session_age_s'] * 1000 + arrivals[0],
            'stage_ms':aggregate_stages(events),
            'measurement_note':'Actual loopback WebSocket, paced synthetic PCM. Window-end to client uses one monotonic clock and includes sender scheduling, network, server queue, compute and return delivery. Excludes physical capture and browser rendering. No detector accuracy claim.'}
    if getattr(args, 'collect_raw', False):
        report['raw_latency_ms'] = {'ack_roundtrip':rtts, 'ingest':ingests,
                                    'window_end_to_client':arrivals,
                                    'compute':[e['latency_ms']['compute'] for e in events],
                                    'queue':[e['latency_ms']['queue'] for e in events]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=8)
    parser.add_argument('--window', type=float, default=2)
    parser.add_argument('--hop', type=float, default=.5)
    parser.add_argument('--delay-ms', type=float, default=0)
    parser.add_argument('--json')
    args = parser.parse_args()
    if args.seconds < args.window or args.delay_ms < 0:
        parser.error('seconds must cover a full window; delay must be nonnegative')
    app = create_app(Settings(audit_path=':memory:', window_s=args.window, stride_s=args.hop),
                     DelayedExtractor(args.delay_ms), make_demo_index())
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', ws_max_size=90000, ws_max_queue=4))
        thread = threading.Thread(target=server.run, kwargs={'sockets':[listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError('Benchmark server did not start')
                time.sleep(.01)
            report = asyncio.run(measure(listener.getsockname()[1], args))
            encoded = json.dumps(report, indent=2, allow_nan=False)
            if args.json:
                Path(args.json).write_text(encoded + '\n')
            print(encoded)
        finally:
            server.should_exit = True
            thread.join(timeout=20)
            if thread.is_alive():
                raise RuntimeError('Benchmark server did not shut down')


if __name__ == '__main__':
    main()
