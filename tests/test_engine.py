from dataclasses import replace
import numpy as np
import pytest
from strive.audio import RATE, RingBuffer, normalize
from strive.config import Settings
from strive.demo import scenario_audio, signal
from strive.engine import Call, aggregate, scheduled_weights, boundary_coherence
from strive.features import DSPExtractor, Features, Segment
from strive.retrieval import ReferenceIndex, SessionProfile


def run(index, scenario, seconds=40):
    c = Call(Settings(), DSPExtractor(), index)
    events = []
    x = scenario_audio(scenario, seconds)
    for i, start in enumerate(range(0, len(x), RATE)):
        events.extend(c.feed(x[start:start + RATE], i))
    return c, events


def test_sample_accurate_overlap():
    x = np.linspace(-.5, .5, 5 * RATE, dtype=np.float32)
    ring = RingBuffer(); windows = []
    for part in np.array_split(x, 173):
        windows.extend(ring.push(part))
    assert [w.end_s for w in windows] == [2, 3, 4, 5]
    assert sum(len(w.fresh_samples) for w in windows) == len(x)
    for previous, current in zip(windows, windows[1:]):
        np.testing.assert_array_equal(previous.samples[RATE:], current.samples[:RATE])
    assert len(ring.pending) < 2 * RATE


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_resampling_and_mono(rate):
    x = np.sin(np.arange(rate) * 2 * np.pi * 220 / rate)
    out = normalize(np.stack([x, x], axis=1), rate)
    assert len(out) == RATE and np.isfinite(out).all()


def test_nan_and_overrange_audio_rejected():
    for x in [np.array([np.nan]), np.array([np.inf]), np.array([3.])]:
        with pytest.raises(ValueError):
            RingBuffer().push(x)


def test_ratio_and_language_fallback():
    x = [[1, 0], [.99, .01], [.98, .02], [0, 1]]
    index = ReferenceIndex(x, [1, 1, 0, 0], ["hi", "hi", "hi", "en"], {})
    score, info = index.query([1, 0], "hi", k=3, min_entries=3)
    assert score == 2/3 and info["route"] == "hi"
    _, info = index.query([1, 0], "ta", k=20, min_entries=3)
    assert info["fallback"] and info["neighbors"] == 4


def test_single_class_index_abstains():
    index = ReferenceIndex([[1, 0], [0, 1]], [0, 0], ["hi", "hi"], {})
    assert index.query([1, 0], "hi", min_entries=1)[0] is None


def test_demo_steady_trust_gate(index):
    c, events = run(index, "steady")
    try:
        assert events[0]["s_risk"] is None
        assert c.bootstrap == "trusted"
        assert events[3]["session_age_s"] == 5
        assert len(c.sps.entries) > 3
        assert events[-1]["s_risk"] < .1
        assert all(e["demo_only"] and not e["calibrated"] for e in events)
    finally: c.close()


def test_suspicious_bootstrap_never_profiled(index):
    c, events = run(index, "suspicious_start")
    try:
        assert c.bootstrap == "blocked" and len(c.sps.entries) == 0
        for i in range(40, 50): c.feed(signal(0, 1, 100), i)
        assert len(c.sps.entries) == 0
        assert all(e["weights"][1] == 0 for e in events)
    finally: c.close()


def test_switch_does_not_poison_profile(index):
    c, events = run(index, "switch")
    try:
        before = next(e for e in events if e["session_age_s"] == 18)
        assert events[-1]["s_risk"] > .7
        assert events[-1]["profile_entries"] == before["profile_entries"]
        assert c.hold_latched
    finally: c.close()


def test_silence_has_no_score(index):
    c, events = run(index, "silence")
    try:
        assert all(e["s_risk"] is None and e["alert_level"] == "analyzing" for e in events)
    finally: c.close()


def test_silence_after_trust_is_not_safe(index):
    c, _ = run(index, "steady", 10)
    try:
        events = c.feed(np.zeros(RATE, dtype=np.float32), 10)
        assert events[-1]["s_risk"] is None and events[-1]["alert_level"] == "analyzing"
    finally: c.close()


def test_sequence_and_max_frame_limits(index):
    c = Call(Settings(), DSPExtractor(), index)
    try:
        with pytest.raises(ValueError): c.feed(signal(0, 1), 2)
        with pytest.raises(ValueError): c.feed(signal(0, 3), 0)
        assert c.sequence == 0
    finally: c.close()


def test_call_isolation_and_erasure(index):
    a, _ = run(index, "steady", 10)
    b = Call(Settings(), DSPExtractor(), index)
    assert len(a.sps.entries) and not len(b.sps.entries)
    a.close()
    assert not len(a.sps.entries) and not len(a.buffer.pending)
    b.close()


def test_gap_clears_profile_and_preserves_time(index):
    c, _ = run(index, "steady", 10)
    try:
        c.gap()
        c.feed(signal(0, 1), 0)
        e = c.feed(signal(0, 1), 1)[0]
        assert e["session_age_s"] == 12 and e["s_risk"] is None
        assert not len(c.sps.entries) and c.bootstrap == "blocked_gap"
    finally: c.close()


def test_bounded_session_store():
    store = SessionProfile(capacity=3)
    f = DSPExtractor().extract(signal(0, 2))
    for _ in range(10): store.add(f)
    assert len(store.entries) == 3 and store.similarity(f) > .99
    different = replace(f, profile=-f.profile, segments=[replace(s, vector=-s.vector) for s in f.segments])
    assert store.similarity(different) < .1
    store.clear()


def test_missing_tracks_renormalized():
    value, weights, raw = aggregate([1., None, None], 80, None)
    assert weights == [1., 0., 0.] and raw == 1 and value == pytest.approx(.4)
    assert aggregate([None, .9, .8], 80, .8)[0] is None


def test_ema_math_and_schedule():
    risk = None
    for _ in range(3): risk, _, _ = aggregate([1., None, None], 0, risk)
    assert risk == pytest.approx(.784) and risk >= .75
    assert scheduled_weights(14) == [.8, 0, .2]
    assert scheduled_weights(15) == [.5, .3, .2]
    assert scheduled_weights(60) == [.5, .3, .2]
    assert scheduled_weights(61) == [.3, .5, .2]


def test_documented_mature_global_dilution():
    _, _, raw = aggregate([1., 0., 0.], 61, .5)
    assert raw > .4


def test_coherence_at_actual_overlap_seam():
    f = Features(np.ones(2), np.ones(2), [Segment(0, .9, np.array([1., 0.])),
                 Segment(.9, 1, np.array([0., 1.])), Segment(1, 2, np.array([0., 1.]))], None, {})
    assert boundary_coherence(f, False) == 0
    assert boundary_coherence(f, True) is None
    f.segments[-1].vector = np.array([0., -1.])
    assert boundary_coherence(f, False) == 1


def test_extractor_failure_is_explicit_uncertainty(index):
    class Broken(DSPExtractor):
        def extract(self, x): raise RuntimeError("model failed")
    c = Call(Settings(), Broken(), index)
    try:
        for seq in range(6): events = c.feed(signal(0, 1), seq)
        assert events[-1]["s_risk"] is None
        assert "MODEL_OR_INDEX_ERROR" in events[-1]["reasons"]
        assert not len(c.sps.entries)
    finally: c.close()
