"""Playwright smoke using Chrome's synthetic microphone; no physical audio saved.

Requires the optional playwright package and a running STRIVE local server.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--browser', default='/usr/bin/google-chrome')
    parser.add_argument('--json', default='evidence/browser-audio.json')
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True,
                                    args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'])
        try:
            context = browser.new_context(permissions=['microphone'])
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(args.url)
            page.wait_for_load_state('networkidle')
            assert page.locator('#mic').is_visible()
            page.locator('#mic').click()
            page.wait_for_function('() => events.length >= 5', timeout=20000)
            state = page.evaluate('''() => ({notice:document.getElementById('notice').textContent,
                sample_rate:audioCtx.sampleRate, events:events.map(e => ({age:e.session_age_s,
                latency_ms:e.latency_ms, capture:e.capture})), running, call_id:callId})''')
            assert state['running']
            assert state['events'][-1]['capture']['windows_dropped'] == 0
            page.locator('#stop').click()
            page.wait_for_function('() => !running && callId === null && audioCtx === null && socket === null')
            # Restart exercises handler cleanup and sequence reset.
            page.locator('#mic').click()
            page.wait_for_function('() => events.length >= 1', timeout=12000)
            assert page.evaluate('() => callId') != state['call_id']
            page.locator('#stop').click()
            page.wait_for_function('() => !running && callId === null')
            assert not errors, errors
            report = {'browser':browser.version, 'synthetic_microphone':True, 'passed':True,
                      'restart_passed':True, 'page_errors':errors, **state,
                      'measurement_note':'Headless Chrome synthetic microphone through the real AudioWorklet and WebSocket. browser_roundtrip measures main-thread delivery of the final PCM packet to receipt of its window result using the same monotonic clock. Excludes physical microphone/OS capture and worklet packetization latency.'}
            report.pop('call_id')
            Path(args.json).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            browser.close()


if __name__ == '__main__':
    main()
