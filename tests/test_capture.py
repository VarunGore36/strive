"""AUD-04 acceptance: inference stalls must not block audio ingestion."""
import threading
import time
import numpy as np
import pytest
from strive.audio import RATE, Window
from strive.capture import BoundedWindowQueue
from strive.config import Settings
from strive.demo import scenario_audio
from strive.engine import Call
from strive.features import DSPExtractor


def window(index):
    samples = np.zeros(2 * RATE, dtype=np.float32)
    return Window(samples, float(index), index + 2., samples, 0.)


class SlowExtractor(DSPExtractor):
    """Intentionally slow inference, as the ticket requires."""

    def __init__(self, delay_s=.05):
        self.delay_s = delay_s
        self.calls = 0

    def extract(self, x):
        self.calls += 1
        time.sleep(self.delay_s)
        return super().extract(x)


# ------------------------------------------------------- bounded queue policy

def test_queue_is_bounded_and_drops_oldest():
    q = BoundedWindowQueue(capacity=3)
    assert q.put([window(i) for i in range(5)]) == 2
    remaining = q.take()
    # Oldest two discarded; the freshest audio survives.
    assert [w.start_s for w in remaining] == [2., 3., 4.]


def test_zero_capacity_rejected():
    with pytest.raises(ValueError):
        BoundedWindowQueue(capacity=0)
    with pytest.raises(ValueError):
        Settings(capture_queue_windows=0)


def test_no_drops_while_within_capacity():
    q = BoundedWindowQueue(capacity=8)
    assert q.put([window(i) for i in range(8)]) == 0
    assert q.telemetry()["windows_dropped"] == 0


def test_counters_are_exposed():
    q = BoundedWindowQueue(capacity=2)
    q.put([window(i) for i in range(5)])
    stats = q.telemetry()
    assert stats["capacity"] == 2
    assert stats["windows_queued"] == 5
    assert stats["windows_dropped"] == 3
    assert stats["overflow_batches"] == 1
    assert stats["max_depth"] == 2
    assert stats["current_depth"] == 2


def test_drop_flag_is_consumed_once():
    q = BoundedWindowQueue(capacity=1)
    q.put([window(0), window(1), window(2)])
    assert q.consume_drop_flag() == 2
    assert q.consume_drop_flag() == 0        # cleared after reading


def test_take_limit_and_clear():
    q = BoundedWindowQueue(capacity=8)
    q.put([window(i) for i in range(5)])
    assert len(q.take(2)) == 2
    assert q.depth() == 3
    q.clear()
    assert q.depth() == 0


def test_queue_is_thread_safe():
    q = BoundedWindowQueue(capacity=16)
    errors = []

    def producer():
        try:
            for i in range(200):
                q.put([window(i)])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def consumer():
        try:
            for _ in range(200):
                q.take(3)
                q.telemetry()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=producer) for _ in range(3)]
    threads += [threading.Thread(target=consumer) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert q.telemetry()["windows_queued"] == 600


def test_capture_counters_are_thread_safe():
    q = BoundedWindowQueue(capacity=1)

    def record_counters():
        for _ in range(1000):
            q.record_frame_ingested()
            q.record_window_scored()

    threads = [threading.Thread(target=record_counters) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stats = q.telemetry()
    assert stats["frames_ingested"] == 5000
    assert stats["windows_scored"] == 5000


# ------------------------------------------------ capture / inference split

def test_ingest_never_runs_a_model(index):
    slow = SlowExtractor(delay_s=.2)
    call = Call(Settings(), slow, index)
    audio = scenario_audio("steady", 6)

    started = time.perf_counter()
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)
    ingest_s = time.perf_counter() - started

    # Six frames would cost >0.8 s if ingestion ran the 0.2 s extractor.
    assert slow.calls == 0
    assert ingest_s < .2, f"ingest took {ingest_s:.3f}s; it must not touch the model"
    assert call.capture.depth() > 0

    # The model only runs on the inference path.
    events = call.drain()
    assert slow.calls > 0 and len(events) == len(events)
    call.close()


def test_slow_inference_drops_oldest_windows_and_reports_them(index):
    """The ticket's required simulation: inference cannot keep up."""
    call = Call(Settings(capture_queue_windows=3), SlowExtractor(delay_s=.01), index)
    audio = scenario_audio("steady", 20)

    # Capture everything first; inference has not run at all yet.
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)

    stats = call.capture.telemetry()
    assert stats["windows_queued"] == 19
    assert stats["windows_dropped"] == 16      # only the newest 3 survive
    assert stats["current_depth"] == 3

    events = call.drain()
    assert len(events) == 3
    # The loss is reported, not hidden.
    assert "CAPTURE_QUEUE_OVERFLOW" in events[0]["reasons"]
    assert events[0]["dropped_windows"] == 16
    assert events[0]["capture"]["windows_dropped"] == 16
    # Only the first scored window carries the flag; it is consumed once.
    assert "CAPTURE_QUEUE_OVERFLOW" not in events[1]["reasons"]
    call.close()


def test_drop_resets_channel_continuity(index):
    """A queue drop is a real gap; it must not be scored as a source splice."""
    call = Call(Settings(capture_queue_windows=2), DSPExtractor(), index)
    audio = scenario_audio("steady", 12)
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)
    events = call.drain()
    overflow = [e for e in events if "CAPTURE_QUEUE_OVERFLOW" in e["reasons"]]
    assert overflow
    # Continuity restarts, so the first window after a drop reports no discontinuity.
    assert overflow[0]["channel"]["discontinuity_score"] == 0.
    call.close()


def test_feed_still_behaves_synchronously(index):
    """feed() = ingest + drain. The REST contract is unchanged."""
    call = Call(Settings(), DSPExtractor(), index)
    audio = scenario_audio("steady", 12)
    events = []
    for i, offset in enumerate(range(0, len(audio), RATE)):
        events.extend(call.feed(audio[offset:offset + RATE], i))
    assert len(events) == 11
    assert call.capture.depth() == 0            # nothing left queued
    assert call.capture.telemetry()["windows_dropped"] == 0
    assert not any("CAPTURE_QUEUE_OVERFLOW" in e["reasons"] for e in events)
    call.close()


def test_capture_telemetry_reaches_the_event(index):
    call = Call(Settings(), DSPExtractor(), index)
    audio = scenario_audio("steady", 8)
    events = []
    for i, offset in enumerate(range(0, len(audio), RATE)):
        events.extend(call.feed(audio[offset:offset + RATE], i))
    capture = events[-1]["capture"]
    assert capture["capacity"] == 8
    assert capture["frames_ingested"] == 8
    assert capture["windows_scored"] == len(events)
    call.close()


def test_concurrent_capture_and_inference(index):
    """Capture thread keeps running while a slow inference thread drains."""
    call = Call(Settings(capture_queue_windows=32), SlowExtractor(delay_s=.02), index)
    audio = scenario_audio("steady", 16)
    scored, done = [], threading.Event()

    def inference():
        while not done.is_set() or call.capture.depth():
            scored.extend(call.drain(limit=1))
            time.sleep(.001)

    worker = threading.Thread(target=inference)
    worker.start()
    started = time.perf_counter()
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)
    capture_s = time.perf_counter() - started
    done.set()
    worker.join(timeout=30)

    # Capture finished promptly despite inference still working through the backlog.
    assert capture_s < 1.0, f"capture blocked for {capture_s:.3f}s"
    assert scored, "inference thread produced no events"
    ages = [e["session_age_s"] for e in scored]
    assert ages == sorted(ages)                 # order preserved through the queue
    call.close()


def test_gap_clears_the_capture_queue(index):
    call = Call(Settings(), DSPExtractor(), index)
    audio = scenario_audio("steady", 8)
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)
    assert call.capture.depth() > 0
    call.gap()
    assert call.capture.depth() == 0
    call.close()


def test_close_stops_draining(index):
    call = Call(Settings(), DSPExtractor(), index)
    audio = scenario_audio("steady", 8)
    for i, offset in enumerate(range(0, len(audio), RATE)):
        call.ingest(audio[offset:offset + RATE], i)
    call.close()
    assert call.drain() == []
    with pytest.raises(ValueError):
        call.ingest(audio[:RATE], 99)
