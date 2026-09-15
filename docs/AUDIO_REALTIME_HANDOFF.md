# Audio and real-time handoff

Scope: canonical audio, live transport, overload handling, latency measurement and
browser lifecycle. Detector weights, accuracy, fusion and verification policy are
outside this delivery.

An opt-in [Rust audio runtime](NATIVE_AUDIO.md) now implements the same binary
microphone protocol with separate Python detector processes and worker recovery.
The thread-related limits below describe the original Python runtime.

## Delivered

- Browser microphone → canonical 16 kHz mono → 20 ms binary PCM → independent
  server ingestion and scoring → live events. No score acknowledgement blocks capture.
- 44.1/48 kHz fallback uses a persistent 127-tap low-pass FIR before fractional
  resampling. Tests check sample count, steady amplitude and rejection of a 12 kHz
  signal that would alias into the 16 kHz output.
- One latest pending scoring window by default; every dropped window is counted.
  Channel continuity resets after a queue gap. Offline geometry remains configurable.
- Socket validation, single-producer ownership, bounded browser network backlog,
  joined inference on disconnect, and stop/restart cleanup.
- Server compute/queue timing, audio backlog, ingestion timing, client-observed
  loopback latency, and browser packet-to-result timing with explicit boundaries.
- Correct end-of-frame realtime simulator pacing and independent inference.

## Run

```bash
STRIVE_CONFIG=config/live.json .venv/bin/python -E scripts/start.py
```

Open the displayed localhost URL and select **Use microphone**. This preset uses
2 s windows and 0.5 s hops. Frames arrive every 20 ms; a score requires a complete
window and the detector's own evidence requirements still apply.

For the detector teammate: use the existing `Call` extractor interface and research
configuration. `pcm-v2` is independent of model choice. Slow models are observable
through compute, queue, audio-lag and dropped-window fields; the transport does not
promise that arbitrary model code will finish within a hop.

## Verification and receipts

- Python regression suite: 178 passed, one model-dependent test skipped because
  model assets are unavailable. No neural validation is claimed.
- `tests/test_live_stream.py`: reception while inference is blocked, freshest-window
  overflow behavior, duplicate-producer rejection, malformed PCM, signed PCM extremes,
  sequence validation and session cleanup.
- `node tests/audio_worklet.cjs`: framing at 16/44.1/48 kHz, passband amplitude and
  at least 50 dB rejection of the tested 12 kHz tone at 44.1/48 kHz input.
- `evidence/live-latency.json`: real paced loopback WebSocket, 400/400 packets,
  zero drops, 82.53 ms p95 completed-window-to-client latency.
- `evidence/live-overload.json`: 1200 ms injected inference, 400/400 packets,
  seven stale windows discarded, maximum pending depth one.
- `evidence/browser-audio.json`: Chrome synthetic microphone passed the AudioWorklet,
  live events, no dropped windows, stop and restart with no JavaScript errors.
  Reproduce with `scripts/browser_audio_smoke.py` (optional Playwright dependency)
  against a running local server.

Raw audio is not persisted by the transport or these benchmarks. Compression is not
required by the live protocol. Binary PCM avoids Base64 expansion and introduces no
codec error or encode/decode wait; optional compression needs a measured deployment
bandwidth/CPU tradeoff and separate validation.

## Limits

Loopback and synthetic microphone checks do not measure a physical microphone, mobile
device, WAN, or remote caller integration. In-flight native model execution cannot be
forcibly cancelled safely in a thread; teardown waits for it. A permanently hung model
requires process isolation, beyond this in-process extractor contract. Current browser
capture is local microphone input, not an adapter to arbitrary third-party calls.
