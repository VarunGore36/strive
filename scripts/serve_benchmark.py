"""Controlled Python baseline for comparison with the native runtime."""
import argparse
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import uvicorn
from strive.api import create_app
from strive.config import Settings
from strive.demo import make_demo_index
from benchmark_live import DelayedExtractor

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--delay-ms',type=float,default=0)
    args=parser.parse_args()
    app=create_app(Settings(stride_s=.5,audit_path=':memory:'),DelayedExtractor(args.delay_ms),make_demo_index())
    uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='error',ws_max_size=90000,ws_max_queue=4)
