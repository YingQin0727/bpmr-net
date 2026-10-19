"""Actual independent CPU workers; synthetic12-network inference, no truth."""
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
def worker():
    import numpy as np
    import torch
    import psutil
    sys.path.insert(0,str(ROOT))
    from bpmr_net.models import B0Network
    from paper_revision.models_revision import build_revision_bpmr
    from paper_revision.p1_profile import Bank
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(5)
    m=Bank([build_revision_bpmr(B0Network(),'A3').eval() for _ in range(12)],True).eval()
    x=torch.rand(3,400)
    with torch.inference_mode():
        traced=torch.jit.freeze(torch.jit.trace(m,x))
        for _ in range(8):traced(x)
        samples=[];cpu=time.process_time()
        for _ in range(3):
            start=time.perf_counter()
            for step in range(128):traced(x)
            samples.append(time.perf_counter()-start)
    print(json.dumps(dict(work_units=384,active_seconds=sum(samples),cpu_seconds=time.process_time()-cpu,
        rss_bytes=psutil.Process().memory_info().rss,available_ram_bytes=psutil.virtual_memory().available,samples=samples)))
def main():
    results=[]
    def launch(_):
        p=subprocess.run([sys.executable,__file__,'--worker'],capture_output=True,text=True,timeout=180)
        if p.returncode:raise RuntimeError(p.stderr[-1000:])
        return json.loads(p.stdout.strip().splitlines()[-1])
    for count in [1,2,4]:
        start=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(count) as pool:rows=list(pool.map(launch,range(count)))
        results.append(dict(workers=count,workers_measured=rows,total_process_wall_seconds=time.perf_counter()-start,
            active_throughput_updates_per_second=count*384/max(r['active_seconds'] for r in rows),
            sum_sampled_worker_rss_bytes=sum(r['rss_bytes'] for r in rows)))
    for row in results:row['active_throughput_speedup_vs1']=row['active_throughput_updates_per_second']/results[0]['active_throughput_updates_per_second']
    (OUT/'R1_RECONSTITUTION_P2_CONCURRENCY.json').write_text(json.dumps(dict(identity='REVISION_REPLICATION',results=results,
        scope='Synthetic trace-only12A3forward banks, independent workers. Active spans may not perfectly overlap; startup/tracing included in process wall, excluded active. No feature/fusion/eval contention; not full scientific worker scaling.'),indent=2),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['workers','active_throughput_speedup_vs1','sum_sampled_worker_rss_bytes']} for r in results]))
if __name__=='__main__':worker() if '--worker' in sys.argv else main()
