"""Isolated persistent detector worker for the Rust audio runtime (Unix socketpair).

Control requests reuse the existing FastAPI application through in-process ASGI;
model, policy and audit behavior stay in Python. Rust alone owns live ingestion.
Wire: LE u32 JSON length, LE u32 PCM byte length, JSON, float32 LE PCM. Replies
are LE u32 JSON length + JSON. No pickles, audio files or public worker listener.
"""
import asyncio
from dataclasses import fields
import json
import os
import socket
import struct
import sys
import time

import httpx
import numpy as np

from .api import create_app
from .audio import Window, RATE
from .config import Settings

MAX_JSON = 1_000_000
MAX_PCM = 10 * RATE * 4


class Worker:
    def __init__(self, app, client):
        self.app, self.client, self.call = app, client, None
        self.generation = None

    async def command(self, header, raw):
        op = header['op']
        if op == 'init':
            response = await self.client.post('/v1/calls', json=header['body'])
            if response.status_code != 201:
                return {'status':response.status_code, 'body':response.json()}
            self.call = self.app.state.sessions[response.json()['call_id']]
            self.generation = header['generation']
            if self.generation:
                self.call.gap()
            return {'status':201, 'body':response.json()}
        if self.call is None or header['generation'] != self.generation:
            raise ValueError('Wrong session generation')
        call = self.call
        call.sequence = int(header['sequence'])
        call.received = int(header['received'])
        call.touched = time.monotonic()
        if op == 'window':
            if len(raw) != round(call.cfg.window_s * RATE) * 4:
                raise ValueError('Wrong window size')
            samples = np.frombuffer(raw, dtype='<f4').copy()
            if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
                raise ValueError('Invalid canonical samples')
            start, end, fresh = header['start'], header['end'], header['fresh']
            if end - start != len(samples) or not 0 <= fresh < len(samples):
                raise ValueError('Invalid window geometry')
            for field in fields(call.capture.stats):
                setattr(call.capture.stats, field.name, int(header['capture'][field.name]))
            call.capture.dropped_since_drain = header['dropped']
            queued = time.monotonic() - header['queue_ms'] / 1000
            delay = float(os.environ.get('STRIVE_NATIVE_TEST_DELAY_MS', '0'))
            if delay and call.extractor.is_surrogate:
                # Included in compute by wrapping extract, not transport queue time.
                original = call.extractor.extract
                def delayed(x):
                    time.sleep(delay / 1000)
                    return original(x)
                call.extractor.extract = delayed
            try:
                event = call._score(Window(samples, start / RATE, end / RATE, samples[fresh:], queued))
            finally:
                if delay and call.extractor.is_surrogate:
                    call.extractor.extract = original
                samples.fill(0)
            self.app.state.audit.write(call.id, 'risk.updated', event)
            return {'event':event, 'end':end}
        if raw:
            raise ValueError('Unexpected audio on control request')
        action = header['action']
        methods = {'context':'PATCH','verify':'POST','transaction':'POST','audit':'GET'}
        if action not in methods:
            raise ValueError('Unknown control action')
        response = await self.client.request(methods[action], f'/v1/calls/{call.id}/{action}',
                                             json=header.get('body'))
        return {'status':response.status_code, 'body':response.json()}


async def serve(sock):
    cfg = Settings(**json.loads(os.environ['STRIVE_NATIVE_CONFIG']))
    app = create_app(cfg)
    reader, writer = await asyncio.open_connection(sock=sock)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://127.0.0.1',
                                    headers={'Authorization':'Bearer ' + cfg.api_token}) as client:
            worker = Worker(app, client)
            async def reply(value):
                data = json.dumps(value, allow_nan=False, separators=(',',':')).encode()
                if len(data) > MAX_JSON:
                    raise ValueError('Oversized reply')
                writer.write(struct.pack('<I',len(data)) + data)
                await writer.drain()
            await reply({'ready':(await client.get('/ready')).json(),
                         'config':(await client.get('/v1/config')).json()})
            while True:
                try:
                    sizes = await reader.readexactly(8)
                except asyncio.IncompleteReadError:
                    break
                json_len, pcm_len = struct.unpack('<II',sizes)
                if not 0 < json_len <= MAX_JSON or pcm_len > MAX_PCM or pcm_len % 4:
                    raise ValueError('Invalid IPC lengths')
                header = json.loads(await reader.readexactly(json_len))
                raw = await reader.readexactly(pcm_len)
                try:
                    result = await worker.command(header, raw)
                    await reply({'request_id':header['request_id'], 'generation':header['generation'], **result})
                except Exception as error:
                    await reply({'request_id':header.get('request_id'), 'generation':header.get('generation'),
                                 'error':type(error).__name__})
                finally:
                    del raw
    writer.close()
    await writer.wait_closed()


if __name__ == '__main__':
    # fd 0 is a private connected Unix socket supplied by the parent. Keep all
    # library stdout away from the binary protocol.
    connection = socket.socket(fileno=os.dup(0))
    sys.stdout = sys.stderr
    asyncio.run(serve(connection))
