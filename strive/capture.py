"""AUD-04: keep model inference from blocking audio ingestion.

Capture and inference are separated by a bounded queue of completed windows.
`Call.ingest()` runs on the capture path: validate, buffer, enqueue, return. It
never waits for a model. `Call.drain()` runs on a worker and does the scoring.

## Backpressure strategy

The queue is bounded (`Settings.capture_queue_windows`, default 8 windows — 4 s of
audio at a 0.5 s hop). When inference falls behind and the queue is full, the
**oldest** window is discarded and counted.

Dropping the oldest, not the newest, is deliberate. A live risk score is only
useful if it describes audio the caller is speaking now. Discarding the newest
window would freeze the dashboard in the past while the backlog drains, which is
exactly the failure the score is supposed to catch. Dropping the oldest keeps the
view current and makes the loss explicit instead of silently growing latency.

Nothing here blocks and nothing raises on overflow: a stalled model degrades the
evidence rate, it does not break the call.

A drop is a genuine break in the audio the analyzer saw, so the next scored window
carries `CAPTURE_QUEUE_OVERFLOW` and channel continuity is reset — otherwise the
discontinuity score would report the gap as if it were a splice in the source.
"""
from collections import deque
from dataclasses import asdict, dataclass, field
import threading


@dataclass
class CaptureStats:
    """Counters for the capture path. All monotonic for the life of the call."""
    windows_queued: int = 0
    windows_scored: int = 0
    windows_dropped: int = 0
    frames_ingested: int = 0
    overflow_batches: int = 0
    max_depth: int = 0
    current_depth: int = 0
    capacity: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class BoundedWindowQueue:
    """Fixed-capacity window queue with an explicit, counted drop policy."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capture_queue_windows must be at least 1")
        self.capacity = capacity
        self.lock = threading.Lock()
        self.items: deque = deque()
        self.stats = CaptureStats(capacity=capacity)
        # Set when a drop happens; consumed by the next scored window.
        self.dropped_since_drain = 0

    def put(self, windows) -> int:
        """Enqueue windows, discarding the oldest on overflow. Returns drops."""
        dropped = 0
        with self.lock:
            for window in windows:
                self.items.append(window)
                self.stats.windows_queued += 1
                while len(self.items) > self.capacity:
                    self.items.popleft()
                    dropped += 1
            if dropped:
                self.stats.windows_dropped += dropped
                self.stats.overflow_batches += 1
                self.dropped_since_drain += dropped
            self.stats.current_depth = len(self.items)
            self.stats.max_depth = max(self.stats.max_depth, len(self.items))
        return dropped

    def take(self, limit: int | None = None) -> list:
        """Remove and return queued windows, oldest first."""
        with self.lock:
            count = len(self.items) if limit is None else min(limit, len(self.items))
            taken = [self.items.popleft() for _ in range(count)]
            self.stats.current_depth = len(self.items)
        return taken

    def record_frame_ingested(self) -> None:
        with self.lock:
            self.stats.frames_ingested += 1

    def record_window_scored(self) -> None:
        with self.lock:
            self.stats.windows_scored += 1

    def consume_drop_flag(self) -> int:
        """Read and clear the drop count accumulated since the last scored window."""
        with self.lock:
            dropped, self.dropped_since_drain = self.dropped_since_drain, 0
        return dropped

    def depth(self) -> int:
        with self.lock:
            return len(self.items)

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.dropped_since_drain = 0
            self.stats.current_depth = 0

    def telemetry(self) -> dict:
        with self.lock:
            self.stats.current_depth = len(self.items)
            return self.stats.to_dict()
