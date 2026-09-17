# BASELINE — frozen before Sprint 1 architecture changes

Ticket: BASE-01. This file records measured facts only. No functional code was changed
to produce it. Any number here was executed in this environment on the commit below.

## Commit

| Item | Value |
|---|---|
| Commit | `f1f1dfae1784dd49a5dd1ef263d8ac7989f7fce0` |
| Branch | `main` |
| Working tree | clean except untracked `STRIVE_AGENT_EXECUTION_PLAN.md` |
| Project version | `0.2.0` (`pyproject.toml`, `FastAPI(version="0.2.0")`) |
| API schema version | `1.0` (`schema_version` in every risk event) |

## Reference hardware

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, Linux 7.0.0-31-generic |
| CPU | 12th Gen Intel Core i5-1235U, 12 logical cores |
| RAM | 7 GiB total |
| GPU | none present (`nvidia-smi` absent) |
| Python | 3.12.3 |
| Node | v20.19.6 |
| FFmpeg | 6.1.1-3ubuntu5 |

All latency numbers below are single-run CPU measurements of the DSP surrogate on this
machine. They exclude neural inference, large-index retrieval, network transport and
browser capture. They cannot substantiate any production or GPU real-time claim.

## Environment hazard — `python -E` is required here

A Datadog APM auto-injector is active on this machine. It prepends
`/opt/datadog-packages/datadog-apm-library-python/4.0.2/ddtrace_pkgs/...` to `sys.path`
ahead of the virtualenv, shadowing `typing_extensions` with an older copy. Result:

```
ImportError: cannot import name 'sentinel' from 'typing_extensions'
```

This is an environment fault, not a repository fault. `PYTHONPATH` is empty in the shell;
the injector sets it at exec time. Run every Python command with `-E`:

```bash
.venv/bin/python -E -m pytest -q
```

CI (`.github/workflows/ci.yml`) runs on clean `ubuntu-latest` and is unaffected.

## Dependency versions

Core runtime is pinned by `requirements-demo.lock` (tested closure, Python 3.12 / Linux).
Majors as installed:

| Package | Version |
|---|---|
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| faiss-cpu | 1.15.0 |
| fastapi | 0.141.1 |
| starlette | 1.6.0 |
| pydantic | 2.13.4 |
| uvicorn | 0.52.4 |
| soundfile | 0.14.0 |
| scikit-learn | 1.8.0 |
| pytest | 9.1.1 |
| websockets | 16.0 |

Research extras (`requirements-research.txt`, **not installed in this baseline**):
torch 2.6.0, torchaudio 2.6.0, transformers 4.51.3, huggingface-hub 0.31.1,
safetensors 0.5.3, speechbrain 1.0.2, webrtcvad-wheels 2.0.14.

Two deprecation warnings occur during tests (`httpx`/starlette TestClient,
`anyio.abc.BlockingPortal` alias). Neither causes a failure.

## Current model mode

**demo.** `Settings.mode` defaults to `"demo"`.

| Aspect | Baseline state |
|---|---|
| Extractor | `DSPExtractor` (`id="dsp-surrogate-v1"`, `is_surrogate=True`) |
| Reference index | `make_demo_index()` — 48 procedural vectors, `demo_only: true` |
| Index labels | synthetic engineering family 0/1, **not** bona-fide/spoof ground truth |
| `models/` directory | **does not exist** — no weights downloaded |
| `data/` contents | `manifest.example.csv` only — no corpus |
| Research mode runnable? | No. `ResearchExtractor.__init__` raises `FileNotFoundError` on `models/manifest.json`. |

Every demo event carries `demo_only: true` and the reason code
`SURROGATE_FEATURES_NOT_A_DEEPFAKE_VERDICT`.

## Window / hop configuration (the thing Sprint 1 changes)

| Parameter | Baseline value | Where |
|---|---|---|
| `sample_rate` | 16000 | `strive/config.py:11` |
| `window_s` | 2 | `strive/config.py:12` |
| `stride_s` (hop) | 1 | `strive/config.py:13` |
| `bootstrap_s` | 5 | `strive/config.py:14` |
| `min_voiced_s` | 4.0 | `strive/config.py:15` |
| EMA `alpha` | 0.70 | `strive/config.py:20` |
| `warning` / `alert` | 0.50 / 0.75 | `strive/config.py:21-22` |

**These are hard-locked, not merely defaulted.** `Settings.__post_init__` raises
`ValueError("This build implements the STRIVE 16 kHz / 2 s / 1 s contract")` for any other
triple (`strive/config.py:44-45`), and `RingBuffer.push` hardcodes `2 * RATE` window and
`RATE` hop (`strive/audio.py:94-99`). A 0.5 s hop is currently impossible. BASE-02 and
AUD-02 exist to remove exactly this.

Effective emission rate at baseline: **1 window/second**, 2 s long, 50% overlap.

## Available tests — measured, not claimed

```
.venv/bin/python -E -m pytest -q
45 tests collected
44 passed, 1 skipped, 2 warnings in 18.43s
```

The one skip is the opt-in research acceptance test (requires a sealed model bundle).

| File | Covers |
|---|---|
| `tests/test_engine.py` | ring buffer overlap, weights, EMA, bootstrap gates, masking, coherence |
| `tests/test_api.py` | REST/WebSocket flows, auth, origin, sequence, privacy, mock hold/verify |
| `tests/test_evaluation.py` | ablation replay, leakage handling |
| `tests/test_research_upgrade.py` | 62-d acoustic profile, SPS index behavior, research adapters |
| `tests/test_research_smoke.py` | opt-in research acceptance (skipped without models) |

Non-Python checks, all passing:

```
node --check web/app.js                 OK
node --check web/pcm-worklet.js         OK
node tests/audio_worklet.cjs            AudioWorklet 16000 -> 16000: PASS
                                        AudioWorklet 44100 -> 16000: PASS
                                        AudioWorklet 48000 -> 16000: PASS
.venv/bin/python -E scripts/build_demo.py   regenerates STRIVE_Demo.html
```

## Baseline scenario behaviour

Carried forward from `docs/VALIDATION.md`, regenerated for 0.2.0. 40 s audio, 39 windows,
zero business context. Timestamps are seconds into the input, not wall clock.

| Scenario | Bootstrap | Stored windows | First warning | First alert | Final risk | p95 compute |
|---|---|---:|---|---|---:|---:|
| steady | trusted | 39 | none | none | 0.000 | 13.22 ms |
| switch | trusted | 17 | 22 s | 36 s | 0.806 | 11.41 ms |
| suspicious_start | blocked | 0 | 4 s | 8 s | 0.729 | 8.52 ms |
| silence | blocked | 0 | none | none | unavailable | 0.08 ms |

**Time-to-alert baseline to beat:** the `switch` fixture changes at 18 s; warning arrives
at 22 s (+4 s) and alert at 36 s (+18 s).

## Known baseline weaknesses (deliberately preserved, do not tune away)

These are pinned by tests on purpose. Ticket rules forbid changing thresholds to hide them.

1. At age > 60 s the weight schedule is `[0.30, 0.50, 0.20]`, so `s_global = 1.0` with
   stable session/coherence fuses to only `0.30`. A static clone resembling the bootstrap
   can evade the alert band.
2. EMA from zero: three maximum updates give `1 - 0.6^3 = 0.784`, above the 0.75 alert threshold.
   This matches the three-chunk alert design goal.
3. Boundary coherence is an uncalibrated supporting cue, confounded by phonetic content
   and packet loss. Not validated as a genuine-vs-synthetic discriminator.
4. The self-filtering bootstrap gate blocks obvious fixture contamination but is not
   proven against slow poisoning.

## Startup commands — verified in this environment

Backend and frontend are one process; `scripts/start.py` serves the dashboard from `web/`.

Fresh clone, demo mode:

```bash
python3 -m venv .venv
.venv/bin/python -E -m pip install -r requirements-demo.lock
.venv/bin/python -E scripts/start.py
```

Dashboard: <http://127.0.0.1:8000> (loopback only by default).

Smoke test, with the server already running in another terminal:

```bash
.venv/bin/python -E scripts/smoke.py
```

Research mode (**not runnable at this baseline** — requires a sealed `models/` bundle and
a real reference index; see `docs/MODELS.md`):

```bash
.venv/bin/python -E -m pip install -r requirements-research.txt
.venv/bin/python -E scripts/start.py --research
```

Docker (not executed in this environment):

```bash
export STRIVE_API_TOKEN='choose-a-long-local-secret'
docker compose up --build
```

## BASE-01 acceptance

- [x] Fresh clone can follow the documented startup steps (with the `-E` note above).
- [x] Baseline test command exits successfully — 44 passed, 1 skipped, 0 failed.
- [x] No functional code changed.
