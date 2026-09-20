"""Bounded audio decoding and sample-accurate overlapping windows."""
from dataclasses import dataclass
from io import BytesIO
from math import gcd
import subprocess
import shutil
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000
MAX_UPLOAD = 20 * 1024 * 1024


def normalize(samples, rate):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio must contain finite mono or stereo samples")
    if not 8000 <= rate <= 192000:
        raise ValueError("Unsupported sample rate")
    if rate != RATE:
        g = gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g).astype(np.float32)
    return np.clip(x, -1, 1)


def decode(data: bytes, max_s=120):
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("Upload is empty or exceeds 20 MiB")
    try:
        with sf.SoundFile(BytesIO(data)) as f:
            if len(f) / f.samplerate > max_s or f.channels > 2:
                raise ValueError("Audio exceeds duration limit or has more than two channels")
            return normalize(f.read(dtype="float32", always_2d=True), f.samplerate)
    except (sf.LibsndfileError, RuntimeError):
        # No temp recording. Decode untrusted media only through pipes, never a shell.
        if shutil.which("ffmpeg") is None:
            raise ValueError("Cannot decode audio. Use WAV/FLAC, or install FFmpeg for other formats")
        try:
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "pipe",
                 "-i", "pipe:0", "-t", str(max_s + 1), "-ac", "1", "-ar", str(RATE),
                 "-f", "f32le", "pipe:1"], input=data, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Audio decoding timed out") from exc
        if result.returncode:
            raise ValueError("Cannot decode this audio format")
        x = np.frombuffer(result.stdout, dtype="<f4").copy()
        if len(x) > max_s * RATE:
            raise ValueError("Audio exceeds duration limit")
        return normalize(x, RATE)


def activity(x):
    """Energy activity gate, not a trained speech/non-speech classifier."""
    frames = x[:len(x) // 320 * 320].reshape(-1, 320)
    if not len(frames):
        return 0.0
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return float(np.mean(rms > 0.003))


class SpeechGate:
    def __init__(self, research=False):
        self.vad = None
        if research:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)

    def fraction(self, x):
        if self.vad is None:
            return activity(x)
        frames = x[:len(x) // 320 * 320].reshape(-1, 320)
        decisions = [self.vad.is_speech((np.clip(f, -1, 1) * 32767).astype("<i2").tobytes(), RATE) for f in frames]
        return float(np.mean(decisions)) if decisions else 0.


@dataclass
class Window:
    samples: np.ndarray
    start_s: float
    end_s: float
    fresh_samples: np.ndarray
    # Monotonic clock reading at the moment the window became complete. Queue
    # delay is measured from here, so it stays honest once AUD-04 moves
    # inference off the capture thread.
    enqueued: float = 0.


class RingBuffer:
    """Emit `window_s` windows every `hop_s`; fresh_samples never double-counts audio."""

    def __init__(self, window_s: float = 2.0, hop_s: float = 1.0):
        self.window = round(window_s * RATE)
        self.hop = round(hop_s * RATE)
        if not 0 < self.hop <= self.window:
            raise ValueError("Require 0 < hop_s <= window_s")
        self.pending = np.empty(0, dtype=np.float32)
        self.start = 0
        self.first = True

    def push(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.01:
            raise ValueError("Expected normalized finite PCM")
        self.pending = np.concatenate((self.pending, x))
        windows = []
        now = time.monotonic()
        while len(self.pending) >= self.window:
            data = self.pending[:self.window].copy()
            fresh = data if self.first else data[self.window - self.hop:]
            windows.append(Window(data, self.start / RATE,
                                  (self.start + self.window) / RATE, fresh, now))
            self.pending = self.pending[self.hop:]
            self.start += self.hop
            self.first = False
        return windows

    def clear(self):
        self.pending.fill(0)
        self.pending = np.empty(0, dtype=np.float32)
