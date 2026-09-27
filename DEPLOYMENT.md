# STRIVE Deployment Guide

## Prerequisites

- Python 3.12
- FFmpeg (optional, for MP3/M4A/AAC/OGG/Opus/WebM uploads)
- 2 GB RAM minimum
- 10 GB disk space (for model weights in research mode)

## Quick Start

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.lock
.venv/bin/python -m uvicorn strive.api:app --host 0.0.0.0 --port 8000
```

## Configuration

Configuration is via `STRIVE_CONFIG` environment variable pointing to a JSON file, or individual `STRIVE_*` environment variables.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STRIVE_MODE` | `demo` | `demo` or `research` |
| `STRIVE_DEVICE` | `cpu` | `cpu` or `cuda` |
| `STRIVE_MODEL_DIR` | `models` | Path to model weights |
| `STRIVE_INDEX_PATH` | `data/reference.npz` | Path to reference index |
| `STRIVE_AUDIT_PATH` | `data/audit.sqlite3` | Path to audit database |
| `STRIVE_API_TOKEN` | `""` | Bearer token for API auth |
| `STRIVE_MAX_SESSIONS` | `8` | Maximum concurrent calls |
| `STRIVE_IDLE_TIMEOUT_S` | `60` | Idle session timeout |
| `STRIVE_MAX_CALL_S` | `600` | Maximum call duration |

### Config File Example

```json
{
  "mode": "demo",
  "device": "cpu",
  "max_sessions": 8,
  "idle_timeout_s": 60,
  "api_token": "your-secret-token"
}
```

## Running with Docker

```bash
docker build -t strive .
docker run -p 8000:8000 -e STRIVE_API_TOKEN=your-token strive
```

## Running with Docker Compose

```bash
docker compose up
```

## Production Deployment

### Using Gunicorn

```bash
.venv/bin/pip install gunicorn
.venv/bin/gunicorn strive.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Systemd

Create `/etc/systemd/system/strive.service`:

```ini
[Unit]
Description=STRIVE Voice Cloning Detection
After=network.target

[Service]
Type=simple
User=strive
WorkingDirectory=/opt/strive
ExecStart=/opt/strive/.venv/bin/python -m uvicorn strive.api:app --host 0.0.0.0 --port 8000
Environment=STRIVE_API_TOKEN=your-secret-token
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable strive
sudo systemctl start strive
```

## Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name strive.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Health Checks

- `GET /health` - Basic health check
- `GET /ready` - Readiness probe (checks model and index)
- `GET /v1/status` - Detailed status with latency metrics
- `GET /metrics` - Prometheus-compatible metrics

## Troubleshooting

### Model fails to load

Check that model files exist in `STRIVE_MODEL_DIR` and match the checksums in `manifest.json`.

### High latency

- Reduce `window_s` or `stride_s` in config
- Enable `multi_rate` scheduling
- Use `cuda` device if available

### Memory issues

- Reduce `max_sessions`
- Reduce `capture_queue_windows`
- Reduce `max_session_entries`

### WebSocket connection drops

- Check `idle_timeout_s` setting
- Verify network stability
- Check Nginx proxy timeout settings

## Security

- Always set `STRIVE_API_TOKEN` in production
- Use HTTPS in production (via reverse proxy)
- Restrict access to `/v1/` endpoints via firewall
- Regularly rotate API tokens
- Monitor audit logs for suspicious activity

## Monitoring

The `/metrics` endpoint exposes Prometheus-compatible metrics:

- `strive_active_sessions` - Current active sessions
- `strive_windows_total` - Total windows processed
- `strive_errors_total` - Total errors
- `strive_latency_ms_sum` - Total latency
- `strive_dropped_windows_total` - Total dropped windows

## Backup

- Audit database: `data/audit.sqlite3`
- Reference index: `data/reference.npz`
- Model weights: `models/` directory
