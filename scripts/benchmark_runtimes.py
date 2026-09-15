"""Compare warmed Python and Rust/Python servers with 1/4/8 concurrent calls.

Linux process-tree CPU/RSS samples are approximate; RSS sums include shared pages.
Both modes use the same DSP fixture and one BLAS/OpenMP thread per process.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
from benchmark_live import measure
from stream_wav import hardware,git_commit
from strive.telemetry import percentiles


def tree_usage(root):
    pending=[root]; usage={}
    while pending:
        pid=pending.pop()
        if pid in usage: continue
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
            usage[pid]=((int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
            for task in Path(f'/proc/{pid}/task').iterdir():
                pending.extend(int(p) for p in (task/'children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):
            pass
    return usage


async def run_calls(port,pid,count,seconds,delay):
    async with httpx.AsyncClient(timeout=60) as client:
        responses=await asyncio.gather(*(client.post(f'http://127.0.0.1:{port}/v1/calls',json={}) for _ in range(count)))
        for response in responses: response.raise_for_status()
    baseline=tree_usage(pid); latest=dict(baseline); peak_rss=sum(v[1] for v in baseline.values())
    stop=asyncio.Event()
    async def monitor():
        nonlocal peak_rss
        while not stop.is_set():
            current=tree_usage(pid); latest.update(current)
            peak_rss=max(peak_rss,sum(v[1] for v in current.values()))
            await asyncio.sleep(.1)
    monitor_task=asyncio.create_task(monitor())
    started=time.perf_counter()
    args=SimpleNamespace(seconds=seconds,window=2.,hop=.5,delay_ms=delay,collect_raw=True)
    try:
        results=await asyncio.gather(*(measure(port,args,response.json()['call_id']) for response in responses))
    finally:
        stop.set(); await monitor_task
    elapsed=time.perf_counter()-started
    cpu=sum(max(0.,value[0]-baseline.get(p,(0,0))[0]) for p,value in latest.items())
    merged={key:[value for result in results for value in result['raw_latency_ms'][key]] for key in results[0]['raw_latency_ms']}
    return {'concurrent_calls':count,'seconds_per_call':seconds,'delay_ms':delay,
            'frames_sent':sum(r['frames_sent'] for r in results),
            'frames_acknowledged':sum(r['frames_acknowledged'] for r in results),
            'windows_scored':sum(r['windows_scored'] for r in results),
            'windows_dropped':sum(r['capture']['windows_dropped'] for r in results),
            'max_queue_depth':max(r['capture']['max_depth'] for r in results),
            'latency_ms':{key:percentiles(values,points=(50,95,99)) for key,values in merged.items()},
            'process_tree_cpu_seconds_approx':round(cpu,3),
            'process_tree_cpu_percent_approx':round(cpu/elapsed*100,1),
            'process_tree_peak_rss_mib':round(peak_rss/1048576,1)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=10)
    parser.add_argument('--calls',type=int,nargs='+',default=[1,4,8])
    parser.add_argument('--delay-ms',type=float,default=0)
    parser.add_argument('--json',default='evidence/native-comparison.json')
    args=parser.parse_args()
    if args.seconds<2 or any(not 1<=c<=8 for c in args.calls): parser.error('Use >=2 seconds and 1–8 calls')
    rows=[]
    for count in args.calls:
        for runtime in ('python','rust-python'):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
            command=[sys.executable,'-E','scripts/serve_benchmark.py','--port',str(port),'--delay-ms',str(args.delay_ms)] if runtime=='python' else [sys.executable,'-E','scripts/start_native.py','--port',str(port),'--test-delay-ms',str(args.delay_ms),'--audit-path',':memory:']
            process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+30
                while True:
                    if process.poll() is not None: raise RuntimeError(process.stderr.read().decode())
                    try:
                        if httpx.get(f'http://127.0.0.1:{port}/health',timeout=.5).status_code==200: break
                    except httpx.HTTPError: pass
                    if time.monotonic()>deadline: raise TimeoutError('Server startup')
                    time.sleep(.05)
                row=asyncio.run(run_calls(port,process.pid,count,args.seconds,args.delay_ms))
                row['runtime']=runtime; rows.append(row)
                print(json.dumps(row),flush=True)
            finally:
                process.terminate()
                try: process.communicate(timeout=15)
                except subprocess.TimeoutExpired: process.kill(); process.communicate(); raise
    report={'hardware':hardware(),'git_commit':git_commit(),'source_state':'uncommitted working tree',
            'rows':rows,'measurement_note':'Warmed session workers; actual paced loopback PCM; DSP fixture only. Same 2 s window / 0.5 s hop, 20 ms packets, thread limits and in-memory audit. CPU/RSS sampled from server process trees; summed RSS double-counts shared pages. Single run per configuration; excludes model initialization, physical capture and WAN.'}
    Path(args.json).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
