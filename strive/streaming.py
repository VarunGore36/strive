"""Full-duplex transport with one receiver, one scorer and bounded pending audio.

Ingestion never acquires the inference lock. Shutdown joins in-flight inference
before erasing state: cancelling to_thread would not stop its model thread.
"""
import asyncio
import struct
import time

import numpy as np
import anyio
from fastapi import WebSocketDisconnect

from .audio import RATE
from .capture import BoundedWindowQueue


def binary_pcm(data: bytes):
    """uint32 LE sequence followed by 1–1600 int16 LE samples (<=100 ms)."""
    if not 6 <= len(data) <= 3204 or len(data) % 2:
        raise ValueError("Expected sequence header and at most 100 ms of s16le PCM")
    sequence = struct.unpack_from("<I", data)[0]
    return sequence, np.frombuffer(data, dtype="<i2", offset=4).astype(np.float32) / 32768.


async def run_stream(ws, call, lock, record):
    # Live scoring retains the freshest pending window; offline feed keeps its
    # existing configurable queue. This bounds backlog independently of call length.
    async with lock:
        if call.closed:
            return
        call.capture = BoundedWindowQueue(call.cfg.stream_queue_windows)
    wake = asyncio.Event()
    sending = asyncio.Lock()
    stopping = False

    async def send(message):
        async with sending:
            await asyncio.wait_for(ws.send_json(message), timeout=5)

    async def receive():
        while True:
            message = await asyncio.wait_for(ws.receive(), timeout=call.cfg.idle_timeout_s)
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            data = message.get("bytes")
            if data is None:
                raise ValueError("Binary PCM required for pcm-v2")
            started = time.perf_counter()
            sequence, samples = binary_pcm(data)
            call.ingest(samples, sequence)
            wake.set()
            await send({"type": "ack", "sequence": sequence,
                        "ingest_ms": round((time.perf_counter() - started) * 1000, 3),
                        "capture": call.capture.telemetry()})

    async def score():
        while not stopping:
            await wake.wait()
            wake.clear()
            while not stopping and call.capture.depth():
                async with lock:
                    if call.closed:
                        return
                    events = record(await asyncio.to_thread(call.drain, 1))
                if events:
                    for event in events:
                        event["latency_ms"]["audio_lag"] = round(
                            max(0., call.received / RATE - event["session_age_s"]) * 1000, 3)
                    await send({"type": "events", "events": events,
                                "queued": call.capture.depth()})

    receiver = asyncio.create_task(receive())
    scorer = asyncio.create_task(score())
    try:
        done, _ = await asyncio.wait((receiver, scorer), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        stopping = True
        receiver.cancel()
        wake.set()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(receiver, scorer, return_exceptions=True)
