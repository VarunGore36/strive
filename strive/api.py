import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import asdict
import hmac
import json
from pathlib import Path
import time
from typing import Literal
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from .audio import RATE, MAX_UPLOAD, decode
from .audit import Audit
from .config import Settings
from .demo import SCENARIOS, PRESENTATION_SCENARIOS, make_demo_index, scenario_audio
from .engine import Call
from .features import DSPExtractor
from .retrieval import ReferenceIndex
from .scheduler import MultiRateScheduler

WEB = Path(__file__).resolve().parent.parent / "web"
LANGUAGES = {"auto", "und", "hi", "ta", "te", "bn", "mr", "kn", "en", "mixed"}


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_inr: float = Field(0, ge=0, le=1e12, allow_inf_nan=False)
    urgent: bool = False
    new_beneficiary: bool = False
    privileged_request: bool = False


class NewCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str = "auto"
    context: Context = Field(default_factory=Context)


class PCMFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    pcm_s16le: str = Field(min_length=4, max_length=86000)


class Verification(BaseModel):
    method: Literal["callback", "mfa", "supervisor"]
    confirmed: bool = False
    outcome: Literal["verified", "failed", "review"] | None = None


class Playback(BaseModel):
    mode: Literal["realtime", "accelerated"] = "realtime"
    scenario: Literal["genuine", "spoof", "mid_call"] | None = None


def pcm(frame):
    try:
        raw = base64.b64decode(frame.pcm_s16le, validate=True)
    except Exception:
        raise ValueError("Invalid base64 PCM")
    if not raw or len(raw) % 2 or len(raw) > 4 * RATE:
        raise ValueError("Expected at most 2 s of signed 16-bit little-endian mono PCM")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.


def create_app(settings=None, extractor=None, index=None):
    cfg = settings or Settings.from_env()
    sessions, locks, uploads = {}, {}, {}
    audit_states = {}
    stats = {"windows": 0, "errors": 0, "total_ms": 0., "queue_ms": 0., "overruns": 0,
             "dropped_windows": 0, "latencies": []}

    @asynccontextmanager
    async def lifespan(app):
        nonlocal extractor, index
        if extractor is None:
            if cfg.mode == "demo":
                extractor = DSPExtractor()
            else:
                from .models.research import ResearchExtractor
                extractor = ResearchExtractor(cfg)
        if index is None:
            index = make_demo_index() if cfg.mode == "demo" else ReferenceIndex.load(cfg.index_path)
        if index.metadata.get("extractor_id") != extractor.id:
            raise RuntimeError("Reference index was built with a different CM extractor/model hash")
        if cfg.mode == "research" and index.metadata.get("demo_only", True):
            raise RuntimeError("Research mode refuses a synthetic engineering-fixture index")
        app.state.audit = Audit(cfg.audit_path)
        app.state.audit.purge()
        app.state.sessions = sessions
        stop = asyncio.Event()

        async def reaper():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=10)
                except asyncio.TimeoutError:
                    for key, call in list(sessions.items()):
                        if time.monotonic() - call.touched > cfg.idle_timeout_s and not locks[key].locked():
                            call.close()
                            sessions.pop(key, None)
                            locks.pop(key, None)
                            audio = uploads.pop(key, None)
                            if audio is not None: audio["samples"].fill(0)
        task = asyncio.create_task(reaper())
        yield
        stop.set()
        await task
        for call in sessions.values():
            call.close()
        for item in uploads.values():
            item["samples"].fill(0)
        uploads.clear()
        app.state.audit.close()
        if hasattr(extractor, "close"):
            extractor.close()

    app = FastAPI(title="STRIVE MVP", version="0.2.0", lifespan=lifespan)

    def authorized(token, client):
        if cfg.api_token:
            return hmac.compare_digest(token, cfg.api_token)
        return client in ("127.0.0.1", "::1", "localhost", "testclient")

    @app.middleware("http")
    async def access(request: Request, call_next):
        maximum = MAX_UPLOAD if request.url.path == "/v1/analyze" or request.url.path.endswith("/upload") else 90000
        try:
            if int(request.headers.get("content-length", "0")) > maximum:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        if request.url.path.startswith("/v1/") or request.url.path in ("/metrics", "/ready"):
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not authorized(token, request.client.host if request.client else ""):
                return JSONResponse({"detail": "Bearer token required"}, status_code=401)
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
        return response

    def get_call(key):
        if key not in sessions:
            raise HTTPException(404, "Call not found or expired")
        return sessions[key]

    def new_call(body, source="stream"):
        if body.language not in LANGUAGES:
            raise HTTPException(422, "Unsupported language tag")
        if len(sessions) >= cfg.max_sessions:
            raise HTTPException(429, "Session capacity reached; close a call before opening another")
        call = Call(cfg, extractor, index, body.language, body.context.model_dump(), source)
        sessions[call.id], locks[call.id] = call, asyncio.Lock()
        return call

    def record(events):
        for e in events:
            app.state.audit.write(e["call_id"], "risk.updated", e)
            before = audit_states.get(e["call_id"])
            if before != e["state"]:
                kind = "risk.escalated" if e["state"] in ("REVIEW", "HIGH", "CRITICAL") else "risk.state_changed"
                app.state.audit.write(e["call_id"], kind, e)
                audit_states[e["call_id"]] = e["state"]
            if e.get("hold_latched") and before not in ("REVIEW", "HIGH", "CRITICAL"):
                app.state.audit.write(e["call_id"], "action.held", e)
            stats["windows"] += 1
            stats["total_ms"] += e["latency_ms"]["end_to_end"]
            stats["queue_ms"] += e["latency_ms"]["queue"]
            stats["overruns"] += "COMPUTE_EXCEEDS_STRIDE" in e["reasons"]
            stats["errors"] += "MODEL_OR_INDEX_ERROR" in e["reasons"]
            stats["dropped_windows"] += e.get("dropped_windows", 0)
            stats["latencies"].append(e["latency_ms"]["end_to_end"])
            if len(stats["latencies"]) > 1000: del stats["latencies"][:-1000]
        return events

    async def process(call, samples, sequence):
        lock = locks[call.id]
        async with lock:
            try:
                return record(await asyncio.to_thread(call.feed, samples, sequence))
            except ValueError as e:
                stats["errors"] += 1
                raise HTTPException(422, str(e))

    @app.get("/")
    def homepage():
        return FileResponse(WEB / "landing.html")

    @app.get("/dashboard")
    def dashboard():
        return FileResponse(WEB / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": cfg.mode, "version": "0.2.0"}

    @app.get("/ready")
    def ready():
        return {"ready": True, "mode": cfg.mode, "model_version": extractor.id,
                "device": cfg.device, "model_ready": True, "reference_index_ready": True,
                "language_model_ready": bool(getattr(extractor, "lid", None)),
                "deepfake_detection_validated": False, "reference_vectors": len(index.x),
                "demo_notice": "Not a neural accuracy benchmark" if extractor.is_surrogate else None}

    @app.get("/v1/config")
    def configuration():
        public = asdict(cfg)
        for key in ("api_token", "model_dir", "index_path", "audit_path"):
            public.pop(key)
        return {**public, "languages": sorted(LANGUAGES), "scenarios": SCENARIOS,
                "presentation_scenarios": PRESENTATION_SCENARIOS,
                "model_version": extractor.id, "demo_only": extractor.is_surrogate}

    @app.get("/v1/status")
    def status():
        values = sorted(stats["latencies"])
        def percentile(fraction):
            return values[min(len(values) - 1, max(0, int(np.ceil(len(values) * fraction)) - 1))] if values else None
        return {"backend": "healthy", "mode": cfg.mode, "device": cfg.device,
                "model": extractor.id, "model_ready": True, "reference_index_ready": True,
                "active_sessions": len(sessions), "windows": stats["windows"],
                "dropped_windows": stats["dropped_windows"],
                "latency_ms": {"current": values[-1] if values else None,
                               "p50": percentile(.5), "p95": percentile(.95)},
                "geometry": {"sample_rate": cfg.sample_rate, "window_s": cfg.window_s,
                             "hop_s": cfg.stride_s},
                "branch_cadence_ms": dict(getattr(next(iter(sessions.values()), None), "scheduler", MultiRateScheduler()).cadence)}

    @app.post("/v1/calls", status_code=201)
    async def start(body: NewCall):
        c = new_call(body)
        app.state.audit.write(c.id, "call.started", {"status": "analyzing", "source": c.source})
        return {"call_id": c.id, "ws_path": f"/v1/stream/{c.id}", "sample_rate": RATE, "encoding": "s16le"}

    @app.post("/v1/calls/{key}/chunks")
    async def chunks(key: str, body: PCMFrame):
        try:
            samples = pcm(body)
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"events": await process(get_call(key), samples, body.sequence)}

    @app.patch("/v1/calls/{key}/context")
    async def change_context(key: str, body: Context):
        call = get_call(key)
        async with locks[key]:
            call.context = body.model_dump()
            call.verification_epoch = -1
        return {"updated": True}

    @app.delete("/v1/calls/{key}")
    async def end(key: str):
        call = get_call(key)
        async with locks[key]:
            await asyncio.to_thread(call.close)
            sessions.pop(key, None)
            locks.pop(key, None)
            item = uploads.pop(key, None)
            if item is not None: item["samples"].fill(0)
            audit_states.pop(key, None)
        app.state.audit.write(key, "call.ended", {"status": "ended"})
        return {"status": "ended", "ephemeral_state_deleted": True}

    @app.get("/v1/calls/{key}/audit")
    def audit(key: str):
        return {"call_id": key, "events": app.state.audit.read(key)}

    @app.get("/v1/calls/{key}")
    def call_state(key: str):
        call = get_call(key)
        return {"call_id": key, "source": call.source, "workflow": call.workflow,
                "hold_latched": call.hold_latched, "latest": call.latest,
                "upload": {k: v for k, v in uploads.get(key, {}).items() if k != "samples"}}

    @app.post("/v1/calls/{key}/upload")
    async def upload(key: str, request: Request, filename: str = "upload.wav"):
        call = get_call(key)
        suffix = Path(filename).suffix.lower()
        supported = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm"}
        if suffix not in supported:
            raise HTTPException(415, "Supported formats: WAV, FLAC, and FFmpeg-decodable MP3/M4A/AAC/OGG/Opus/WebM")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "Maximum upload is 20 MiB")
        try:
            audio = await asyncio.to_thread(decode, bytes(data), min(120, cfg.max_call_s))
        except (ValueError, FileNotFoundError, TimeoutError) as e:
            raise HTTPException(422, str(e))
        if len(audio) < round(cfg.window_s * RATE):
            audio.fill(0)
            raise HTTPException(422, f"Audio is too short; provide at least {cfg.window_s:g} seconds")
        if float(np.max(np.abs(audio))) < .001:
            audio.fill(0)
            raise HTTPException(422, "Audio is silent; no evidence can be analyzed")
        previous = uploads.pop(key, None)
        if previous is not None: previous["samples"].fill(0)
        uploads[key] = {"samples": audio, "filename": Path(filename).name[:160],
                        "duration_s": len(audio) / RATE, "sample_rate": RATE,
                        "channels": 1, "raw_audio_saved": False}
        call.source = "file_upload_stream"
        return {k: v for k, v in uploads[key].items() if k != "samples"}

    @app.post("/v1/calls/{key}/hold")
    async def hold(key: str):
        call = get_call(key)
        async with locks[key]:
            call.hold_latched = True
            call.workflow = "held"
            call.verification_epoch = -1
        event = {"status": "held_mock", "recommended_action": "HOLD_PENDING_VERIFICATION",
                 "demo_only": True}
        app.state.audit.write(key, "action.held", event)
        return event

    @app.post("/v1/calls/{key}/verify")
    async def verify(key: str, body: Verification):
        call = get_call(key)
        outcome = body.outcome or ("verified" if body.confirmed else None)
        if outcome is None:
            raise HTTPException(422, "Provide a simulated verification outcome")
        async with locks[key]:
            call.verification_method = body.method
            if outcome == "verified":
                call.verification_epoch = call.sequence
                call.hold_latched = False
                call.workflow = "verified"
            elif outcome == "failed":
                call.verification_epoch = -1
                call.hold_latched = True
                call.workflow = "blocked"
            else:
                call.verification_epoch = -1
                call.hold_latched = True
                call.workflow = "review"
        status = {"verified": "verified_mock", "failed": "blocked_mock", "review": "supervisor_review_mock"}[outcome]
        event = {"status": status, "method": body.method, "outcome": outcome, "demo_only": True}
        app.state.audit.write(key, "verification." + outcome, event)
        return event

    @app.post("/v1/calls/{key}/transaction")
    async def transaction(key: str):
        call = get_call(key)
        async with locks[key]:
            from .policy import decide
            risk = call.latest.get("s_risk") if call.latest else None
            stale = time.monotonic() - call.touched > 3
            policy = decide(None if stale else risk, call.context, cfg.warning, cfg.alert, cfg.critical)
            needs_check = (call.hold_latched or stale or risk is None or
                           policy["recommended_action"] != "CONTINUE_MONITORING" or cfg.mode == "demo")
            verified = call.verification_epoch == call.sequence
            status = "blocked_mock" if call.workflow == "blocked" else "executed_mock" if not needs_check or verified else "held_mock"
            event = {"status": status, "recommended_action": policy["recommended_action"],
                     "demo_only": cfg.mode == "demo"}
            app.state.audit.write(key, "transaction.attempt", event)
            return event

    @app.post("/v1/analyze")
    async def analyze(request: Request, language: str = "auto"):
        # Stream a raw audio body with a hard byte ceiling; multipart is intentionally unnecessary.
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "Maximum upload is 20 MiB")
        try:
            audio = await asyncio.to_thread(decode, bytes(data))
        except (ValueError, FileNotFoundError, TimeoutError) as e:
            raise HTTPException(422, str(e))
        call = new_call(NewCall(language=language), "file_upload")
        events = []
        try:
            for seq, off in enumerate(range(0, len(audio), RATE)):
                events.extend(await process(call, audio[off:off + RATE], seq))
            return {"call_id": call.id, "events": events, "duration_s": len(audio) / RATE,
                    "raw_audio_saved": False, "trailing_samples_unscored": max(0, len(audio) - (events[-1]["session_age_s"] * RATE if events else 0))}
        finally:
            audio.fill(0)
            call.close()
            sessions.pop(call.id, None)
            locks.pop(call.id, None)

    @app.post("/v1/demo/{name}")
    async def run_demo(name: str, body: NewCall):
        if cfg.mode != "demo":
            raise HTTPException(409, "Engineering fixtures are disabled in research mode")
        if name not in SCENARIOS:
            raise HTTPException(404, "Unknown scenario")
        call = new_call(body, "engineering_fixture")
        audio = scenario_audio(name)
        events = []
        try:
            for seq, off in enumerate(range(0, len(audio), RATE)):
                events.extend(await process(call, audio[off:off + RATE], seq))
        except Exception:
            call.close()
            sessions.pop(call.id, None)
            locks.pop(call.id, None)
            raise
        return {"call_id": call.id, "events": events, "fixture_only": True,
                "description": SCENARIOS[name]}

    @app.websocket("/v1/stream/{key}")
    async def stream(ws: WebSocket, key: str):
        origin = ws.headers.get("origin")
        expected = str(ws.url).split("/v1/")[0].replace("ws://", "http://").replace("wss://", "https://")
        if origin and origin != expected:
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            initial = await asyncio.wait_for(ws.receive_text(), timeout=5)
            if len(initial) > 2048:
                raise ValueError("Oversized authentication message")
            auth = json.loads(initial)
            if not authorized(str(auth.get("token", "")), ws.client.host if ws.client else ""):
                await ws.close(code=1008)
                return
            call = get_call(key)
            await ws.send_json({"type": "ready", "sample_rate": RATE})

            while True:
                message = await asyncio.wait_for(ws.receive_text(), timeout=cfg.idle_timeout_s)
                if len(message) > 88000:
                    raise ValueError("Frame exceeds size limit")
                obj = json.loads(message)
                if obj.get("type") == "playback":
                    command = Playback.model_validate(obj)
                    metadata = None
                    if command.scenario:
                        metadata = PRESENTATION_SCENARIOS[command.scenario]
                        audio = scenario_audio(command.scenario)
                        source_name = metadata["display_name"]
                        call.source = "synthetic_presentation_scenario"
                    else:
                        item = uploads.get(key)
                        if item is None:
                            raise ValueError("Upload audio before starting playback")
                        audio = item["samples"]
                        source_name = item["filename"]
                    duration = len(audio) / RATE
                    await ws.send_json({"type": "playback_started", "mode": command.mode,
                        "source": source_name, "duration_s": duration,
                        "synthetic": bool(command.scenario),
                        "attack_onset_sec": metadata.get("attack_onset_sec") if metadata else None})
                    emitted = []
                    frame_samples = max(1, round(cfg.stride_s * RATE))
                    started = time.monotonic()
                    try:
                        for number, offset in enumerate(range(0, len(audio), frame_samples)):
                            if command.mode == "realtime":
                                delay = started + number * cfg.stride_s - time.monotonic()
                                if delay > 0: await asyncio.sleep(delay)
                            current = audio[offset:offset + frame_samples]
                            events = await process(call, current, call.sequence)
                            emitted.extend(events)
                            await ws.send_json({"type": "events", "sequence": call.sequence - 1,
                                "events": events, "queued": call.capture.depth(),
                                "playback": {"mode": command.mode,
                                    "current_s": min(duration, (offset + len(current)) / RATE),
                                    "duration_s": duration,
                                    "progress": min(1., (offset + len(current)) / len(audio))}})
                        onset = metadata.get("attack_onset_sec") if metadata else None
                        first_high = next((e["session_age_s"] for e in emitted
                            if e["state"] in ("HIGH", "CRITICAL") and (onset is None or e["session_age_s"] >= onset)), None)
                        first_policy_hold = next((e["session_age_s"] for e in emitted
                            if e["decision_state"] in ("HIGH", "CRITICAL") and
                            (onset is None or e["session_age_s"] >= onset)), None)
                        await ws.send_json({"type": "playback_complete", "events": len(emitted),
                            "duration_s": duration, "attack_onset_sec": onset,
                            "first_high_sec": first_high,
                            "first_policy_hold_sec": first_policy_hold,
                            "time_to_alert_sec": None if onset is None or first_high is None else round(first_high - onset, 3)})
                    finally:
                        if command.scenario:
                            audio.fill(0)
                        else:
                            item = uploads.pop(key, None)
                            if item is not None: item["samples"].fill(0)
                    continue
                if obj.get("type") == "gap":
                    async with locks[key]:
                        call.gap()
                    await ws.send_json({"type": "gap", "state": "analyzing"})
                    continue
                frame = PCMFrame.model_validate(obj)
                # AUD-04: ingest and drain are separate inside the engine and both
                # run off the event loop, so the socket is never blocked by a model
                # for longer than one drain. The bounded queue absorbs the rest and
                # reports what it discarded.
                events = await process(call, pcm(frame), frame.sequence)
                await ws.send_json({"type": "events", "sequence": frame.sequence,
                                    "events": events, "queued": call.capture.depth()})
        except WebSocketDisconnect:
            pass
        except (ValueError, HTTPException, asyncio.TimeoutError):
            try:
                await ws.send_json({"type": "error", "message": "Invalid, out-of-order, or expired stream"})
                await ws.close(code=1008)
            except RuntimeError:
                pass
        finally:
            # Closing an unauthorized connection must not delete someone else's call.
            if "call" in locals() and key in locks:
                async with locks[key]:
                    call.close()
                    sessions.pop(key, None)
                    locks.pop(key, None)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return (f"strive_active_sessions {len(sessions)}\nstrive_windows_total {stats['windows']}\n"
                f"strive_errors_total {stats['errors']}\nstrive_latency_ms_sum {stats['total_ms']:.3f}\n"
                f"strive_queue_ms_sum {stats['queue_ms']:.3f}\n"
                f"strive_dropped_windows_total {stats['dropped_windows']}\n"
                f"strive_stride_overruns_total {stats['overruns']}\n")

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app


app = create_app()
