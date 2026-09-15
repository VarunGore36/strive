"""Run the real API/pipeline, save events, and create a self-contained replay."""
import base64
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fastapi.testclient import TestClient
from strive.api import create_app
from strive.config import Settings
from strive.demo import SCENARIOS


def main():
    root = Path(__file__).resolve().parents[1]
    result, summary = {}, {}
    with TestClient(create_app(Settings(audit_path=":memory:"))) as client:
        for name in SCENARIOS:
            response = client.post('/v1/demo/' + name, json={'context':{}})
            response.raise_for_status()
            record = response.json(); result[name] = record
            events = record['events']
            summary[name] = {'windows':len(events),'bootstrap':events[-1]['bootstrap'],
                'last_risk':events[-1]['s_risk'], 'profile_entries':events[-1]['profile_entries'],
                'first_warning_s':next((e['session_age_s'] for e in events if e['alert_level']=='warning'),None),
                'first_alert_s':next((e['session_age_s'] for e in events if e['alert_level'] in ('alert','critical')),None),
                'p50_ms':float(np.percentile([e['latency_ms']['end_to_end'] for e in events],50)),
                'p95_ms':float(np.percentile([e['latency_ms']['end_to_end'] for e in events],95))}
            client.delete('/v1/calls/' + record['call_id'])
    report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'scope':'Engineering fixtures only; not a detector accuracy evaluation',
              'python':sys.version.split()[0],'platform':platform.platform(),'scenarios':summary,
              'neural_inference_tested':False,'speech_accuracy_measured':False,'raw_live_audio_saved':False}
    evidence = root / 'evidence'; evidence.mkdir(exist_ok=True)
    (evidence / 'demo-events.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (evidence / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    html = (root / 'web/index.html').read_text(encoding='utf-8')
    css = (root / 'web/style.css').read_text(encoding='utf-8'); js = (root / 'web/app.js').read_text(encoding='utf-8')
    font = base64.b64encode((root / 'web/fonts/unbounded-600.ttf').read_bytes()).decode('ascii')
    css = css.replace('/assets/fonts/unbounded-600.ttf', 'data:font/ttf;base64,' + font)
    html = html.replace('<link rel="stylesheet" href="/assets/style.css">', '<style>' + css + '</style>')
    data = json.dumps(result, ensure_ascii=True).replace('</', '<\\/')
    html = html.replace('<script src="/assets/app.js"></script>', '<script>window.STRIVE_DEMO=' + data + ';</script><script>' + js + '</script>')
    output = root / 'STRIVE_Demo.html'; output.write_text(html, encoding='utf-8')
    print(json.dumps(report, indent=2)); print('Portable replay:', output)


if __name__ == '__main__': main()
