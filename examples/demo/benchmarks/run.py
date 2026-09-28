import argparse
import sys
import concurrent.futures
import json
import threading
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['serial','batch'], required=True)
parser.add_argument('--bundle', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
if args.mode == 'serial':
    from runtime_serial import GuardRuntime
else:
    from runtime_v10 import GuardRuntime
rt = GuardRuntime(args.bundle)
print('runtime ready', args.mode, flush=True)
ids = rt.encode('Explain how plants convert sunlight into energy. ')[0]
prefix = (ids * (2048//len(ids)+1))[:2048]
measurements = []
for concurrency in (1,4,16):
    for repeat in range(2):
        barrier = threading.Barrier(concurrency)
        def worker(slot):
            cache, _ = rt.forward(prefix, None)
            latencies=[]
            for i in range(110):
                barrier.wait(timeout=120)
                started=time.perf_counter()
                cache, _ = rt.forward([ids[(slot+i)%len(ids)]], cache)
                elapsed=(time.perf_counter()-started)*1000
                if i>=10: latencies.append(elapsed)
            return latencies
        started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            values=[v for group in pool.map(worker,range(concurrency)) for v in group]
        result=dict(concurrency=concurrency,repeat=repeat,p50_ms=float(np.percentile(values,50)),
                    p95_ms=float(np.percentile(values,95)),samples=len(values))
        measurements.append(result)
        print(json.dumps(result),flush=True)
report=dict(mode=args.mode,weight_sha256=rt.meta['checkpoint_sha256'],
            measurements=measurements,latency=rt.latency())
Path(args.output).write_text(json.dumps(report,indent=2))
if hasattr(rt,'close'): rt.close()
