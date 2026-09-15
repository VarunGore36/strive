"""Launch the opt-in Rust audio service with isolated Python detector workers.

Linux/macOS only (private Unix socketpairs). Build the release binary first with:
  cargo build --release --manifest-path native/audio-runtime/Cargo.toml
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strive.config import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--config', default='config/live.json')
    parser.add_argument('--worker-timeout-ms', type=int, default=5000)
    parser.add_argument('--startup-timeout-ms', type=int, default=30000)
    parser.add_argument('--test-delay-ms', type=float, default=0, help='demo-only injected model delay')
    parser.add_argument('--max-sessions', type=int)
    parser.add_argument('--audit-path')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('The native socketpair runtime currently requires Linux or macOS')
    if not 0 <= args.port <= 65535 or min(args.worker_timeout_ms,args.startup_timeout_ms) <= 0 or args.test_delay_ms < 0:
        parser.error('Invalid port, timeout or delay')
    os.chdir(ROOT)
    os.environ['STRIVE_CONFIG'] = str(Path(args.config).resolve())
    values = asdict(Settings.from_env())
    if args.max_sessions is not None:
        values['max_sessions'] = args.max_sessions
    if args.audit_path:
        values['audit_path'] = args.audit_path
    cfg = Settings(**values)
    if cfg.window_s > 10:
        parser.error('Native windows are bounded to 10 seconds')
    if args.test_delay_ms and cfg.mode != 'demo':
        parser.error('Injected delay is only available in demo mode')
    binary = ROOT / 'native/audio-runtime/target/release/strive-audio'
    if not binary.is_file():
        parser.error('Build first: cargo build --release --manifest-path native/audio-runtime/Cargo.toml')
    os.environ['STRIVE_NATIVE_CONFIG'] = json.dumps(values)
    os.environ['STRIVE_NATIVE_TEST_DELAY_MS'] = str(args.test_delay_ms)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.execv(str(binary),[str(binary),'--root',str(ROOT),'--python',sys.executable,
                         '--port',str(args.port),'--worker-timeout-ms',str(args.worker_timeout_ms),
                         '--startup-timeout-ms',str(args.startup_timeout_ms)])


if __name__ == '__main__':
    main()
