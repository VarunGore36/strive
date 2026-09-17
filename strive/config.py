"""Typed runtime and experiment settings; core requires only the standard library."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    sample_rate: int = 16000
    window_s: float = 2.0
    stride_s: float = 1.0
    bootstrap_s: int = 5
    min_voiced_s: float = 4.0
    global_k: int = 20
    session_k: int = 10
    global_gate: float = 0.30
    similarity_gate: float = 0.70
    alpha: float = 0.60
    warning: float = 0.50
    alert: float = 0.75
    critical: float = 0.90
    max_session_entries: int = 120
    max_call_s: int = 600
    idle_timeout_s: int = 60
    max_sessions: int = 8
    min_partition_entries: int = 20
    min_neighbor_similarity: float = -1.0
    device: str = "cpu"
    model_dir: str = "models"
    index_path: str = "data/reference.npz"
    audit_path: str = "data/audit.sqlite3"
    profile_backend: str = "acoustic"
    require_language_model: bool = True
    weight_preset: str = "proposed"
    raise_model_errors: bool = False
    api_token: str = ""
    # CH-06 heuristic constants; overrides merge onto DEFAULT_QUALITY_WEIGHTS.
    channel_quality_weights: dict = field(default_factory=dict)
    # SCH-01 branch cadences in milliseconds of AUDIO time; merges onto
    # scheduler.DEFAULT_CADENCE_MS. Empty means run every branch every window.
    branch_cadence_ms: dict = field(default_factory=dict)
    multi_rate: bool = False
    # AUD-04 bounded capture queue, in completed windows.
    capture_queue_windows: int = 8
    # Common channel reliability mask. It can abstain when quality is unusable,
    # but cannot turn poor audio into authenticity evidence in either direction.
    channel_reliability: bool = True
    channel_reliability_weights: dict = field(default_factory=lambda: {
        "artifact": 0.5, "session": 0.8, "coherence": 1.0})

    def __post_init__(self) -> None:
        if self.weight_preset not in ("equal", "proposed", "global_heavy", "session_heavy"):
            raise ValueError("Unknown weight preset")
        if self.mode not in ("demo", "research"):
            raise ValueError("mode must be demo or research")
        if self.sample_rate != 16000:
            raise ValueError("This build implements the STRIVE 16 kHz canonical contract")
        # Window and hop are configurable; both must land on whole samples so the
        # ring buffer stays sample-accurate and window timestamps stay exact.
        if not 0 < self.stride_s <= self.window_s:
            raise ValueError("Require 0 < stride_s <= window_s")
        for name in ("window_s", "stride_s"):
            samples = getattr(self, name) * self.sample_rate
            if abs(samples - round(samples)) > 1e-9 or round(samples) < 1:
                raise ValueError(f"{name} must be a positive whole number of samples at 16 kHz")
        if not 0 <= self.alpha < 1 or not 0 < self.warning < self.alert < self.critical <= 1:
            raise ValueError("Invalid EMA or alert thresholds")
        if not 0 < self.global_gate < 1 or not 0 < self.similarity_gate < 1:
            raise ValueError("Invalid bootstrap/update gate")
        if self.capture_queue_windows < 1:
            raise ValueError("capture_queue_windows must be at least 1")
        if min(self.global_k, self.session_k, self.max_session_entries, self.max_sessions) < 1:
            raise ValueError("Counts must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        path = os.environ.get("STRIVE_CONFIG")
        values = json.loads(Path(path).read_text()) if path else {}
        for field in ("mode", "model_dir", "index_path", "audit_path", "device", "api_token"):
            if "STRIVE_" + field.upper() in os.environ:
                values[field] = os.environ["STRIVE_" + field.upper()]
        return cls(**values)
