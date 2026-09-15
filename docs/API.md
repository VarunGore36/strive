# STRIVE API

Base URL `http://127.0.0.1:8000`; OpenAPI `/docs`. Scores are 0–1 and are
uncalibrated. `null` means unavailable evidence. Loopback access needs no token;
set `STRIVE_API_TOKEN` for any non-loopback deployment.

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Process liveness, mode and version |
| GET | `/ready` | Model/index/language readiness and device |
| GET | `/metrics` | Prometheus-style counters |
| GET | `/v1/status` | Presentation health, geometry, cadence, p50/p95 |
| GET | `/v1/config` | Public configuration and scenario metadata |
| POST | `/v1/calls` | Create an ephemeral call |
| GET | `/v1/calls/{id}` | Current call, workflow and latest event |
| DELETE | `/v1/calls/{id}` | End call and erase transient state |
| POST | `/v1/calls/{id}/chunks` | Base64 16 kHz PCM frame; legacy REST stream |
| POST | `/v1/calls/{id}/upload?filename=x.wav` | Decode into transient memory for streaming playback |
| PATCH | `/v1/calls/{id}/context` | Replace business context without changing authenticity risk |
| POST | `/v1/calls/{id}/hold` | Latch simulated sensitive action hold |
| POST | `/v1/calls/{id}/verify` | Record verified, failed or review outcome |
| POST | `/v1/calls/{id}/transaction` | Legacy simulated action attempt |
| GET | `/v1/calls/{id}/audit` | Allow-listed metadata events only |
| WS | `/v1/stream/{id}` | Primary real-time events, mic ingest and playback |
| POST | `/v1/analyze` | Legacy accelerated batch analysis; closes call |
| POST | `/v1/demo/{name}` | Legacy accelerated procedural scenario |

Upload accepts WAV/FLAC and, when FFmpeg exists, MP3/M4A/AAC/OGG/Opus/WebM.
Maximum size is 20 MiB, duration 120 seconds, one or two source channels and
8–192 kHz input. The decoder normalizes to 16 kHz mono float audio. Silent and
short uploads are rejected by the presentation upload endpoint.

## Call creation

```json
{
  "language": "auto",
  "context": {
    "amount_inr": 1000000,
    "urgent": true,
    "new_beneficiary": true,
    "privileged_request": false
  }
}
```

Language tags include `en`, `hi`, `mixed` (Hinglish/code-mixed) and other listed
Indian language routes. A tag is metadata, not evidence of validated language
accuracy. Demo automatic language remains `und`/unavailable.

## WebSocket

Connect to the `ws_path` returned by call creation, then send the token in the
first message: `{"token":""}`. The server responds:

```json
{"type":"ready","sample_rate":16000}
```

Microphone/PCM message:

```json
{"sequence":0,"pcm_s16le":"BASE64..."}
```

Known gap: `{"type":"gap"}`. A gap clears the trusted profile, scheduler cache
and continuity state; it does not mint a safe score.

## Channel state (CH-07)

Every risk event carries a `channel` object describing the **transport**, never the
speaker. A degraded channel means the acoustic evidence deserves less trust; it is not
itself evidence of a clone. Values that could not be measured are `null`, never `0`.

```json
"channel": {
  "sample_rate": 16000,
  "estimated_bandwidth_hz": 7600.0,
  "snr_db": 18.2,
  "clipping_ratio": 0.001,
  "rms_dbfs": -21.4,
  "discontinuity_score": 0.12,
  "packet_loss_rate": null,
  "jitter_ms": null,
  "codec": null,
  "quality": 0.84
}
```

| Field | Meaning |
|---|---|
| `estimated_bandwidth_hz` | Occupied-bandwidth estimate: 99% cumulative-power rolloff on the averaged 512-point STFT. Separates narrowband from wideband. Content-dependent; `null` below the silence floor. |
| `snr_db` | Rough SNR **proxy**, clamped to [0, 60]. Minimum-statistics noise floor (10th percentile of each frequency bin across time) against total power. Monotonic in added noise but biased high; not a laboratory measurement. |
| `clipping_ratio` | Fraction of samples at or beyond ±0.99. |
| `rms_dbfs` | Window level; `null` on digital silence. |
| `discontinuity_score` | Bounded [0,1] continuity signal across adjacent windows: max of energy, noise-floor, spectral-centroid and boundary-step jumps. The first window has no predecessor and scores `0`. |
| `packet_loss_rate`, `jitter_ms`, `codec` | Transport metadata supplied by the caller; `null` when the transport does not report it. |
| `quality` | Conservative **reliability** scalar in [0,1]. It is *not* a spoof probability. `null` when no usable audio was present. Individual features remain available regardless. |

Every event also carries `capture` (AUD-04 queue counters), `scheduler` (SCH-03
cadence telemetry) and `dropped_windows`. See `docs/LATENCY.md`.

Backward compatibility: `channel` is additive. `schema_version` is now `"1.1"` because `latency_ms`
changed from a float to `{compute, queue, end_to_end}`; every other addition is
additive and clients that ignore unknown keys are unaffected. Per-stage timings are in `stage_ms`, which now
includes `channel` alongside `features` and `tracks`.

Window geometry is configurable (`window_s`, `stride_s`, both exposed by `/v1/config`).
`stride_s` must satisfy `0 < stride_s <= window_s` and land on whole 16 kHz samples.
2 s/1 s, 2 s/0.5 s and 4 s/0.5 s are covered by tests. Event rate follows the hop, so a
0.5 s stride emits roughly two events per second after warm-up.

Context thresholding can hold a sensitive action while acoustic risk remains unknown. The mock transaction endpoint also holds stale/uncertain evidence and latches prior alerts. Demo-mode transactions always require a mock verification. Verification applies only to the current received sequence; subsequent evidence or context changes can invalidate it.

Uploaded file playback after `POST /upload`:

```json
{"type":"playback","mode":"realtime"}
```

Deterministic synthetic scenario:

```json
{"type":"playback","mode":"accelerated","scenario":"mid_call"}
```

Modes are `realtime` (audio-time paced) and `accelerated`. Both use identical
decode, buffer, capture, evidence, fusion and policy logic. Scheduling uses audio
timestamps, so branch cadence is equivalent.

Playback emits `playback_started`, multiple `events`, then
`playback_complete`. Every `events` message includes `playback.current_s`,
`duration_s`, `progress`, queue depth and one or more incremental risk events.
`playback_complete` includes ground-truth onset and time-to-alert only for a
labelled built-in scenario. Arbitrary uploads never receive an invented onset.

## Risk event

Important fields:

```json
{
  "schema_version": "1.1",
  "state": "ANALYZING|LOW|REVIEW|HIGH|CRITICAL",
  "authenticity_state": "ANALYZING|LOW|REVIEW|HIGH|CRITICAL",
  "decision_state": "ANALYZING|LOW|REVIEW|HIGH|CRITICAL",
  "authenticity_risk": null,
  "context_risk": 0.8,
  "decision_risk": null,
  "branches": {
    "artifact": {"score": null, "available": false, "freshness": "unavailable", "reliability": null, "reason": "NO_GLOBAL_EVIDENCE"},
    "session": {"score": null, "available": false, "freshness": "unavailable", "reliability": null, "reason": "NO_TRUSTED_PROFILE"},
    "coherence": {"score": null, "available": false, "freshness": "unavailable", "reliability": null, "reason": "NO_ADJACENT_VOICED_PAIR"},
    "channel": {"score": 0.91, "available": true, "freshness": "fresh", "reason": "QUALITY_NOT_SPOOF_RISK"}
  },
  "hold_latched": true,
  "workflow": "held",
  "geometry": {"sample_rate": 16000, "window_s": 2.0, "hop_s": 0.5}
}
```

Legacy `s_risk`, `track_scores`, `weights`, `alert_level` and
`recommended_action` remain. `authenticity_state` uses only acoustic risk.
`state` is the backward-friendly main authenticity status; `decision_state` is
the separate business-policy status. Channel quality is reliability, never a spoof
score. Branch confidence remains `null` unless the implementation has an actual
confidence contract.

## Verification

Hold: `POST /v1/calls/{id}/hold`.

```json
{"method":"callback","outcome":"verified"}
```

Methods: `callback`, `mfa`, `supervisor`. Outcomes:

- `verified`: clears the hold for the current evidence epoch.
- `failed`: latches block/escalation.
- `review`: remains held for supervisor review.

The legacy `{"confirmed":true}` request maps to `verified`. All operations are
simulated integrations; no funds, calls, MFA messages or external requests occur.

## Privacy and failure behavior

Upload arrays are zeroed after playback/end and never written to disk. Call IDs
are random UUIDs. Audit stores only allowed risk, reason, model, latency and
workflow metadata. PCM, pitch, embeddings, file bytes and phone numbers are not
audit fields. Missing models/indexes and research mismatches fail closed.
