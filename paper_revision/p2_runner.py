"""Prespecified synthetic expanded predictor/sequence audit; never test truth."""
import copy
import hashlib
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import psutil
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_revision.p1_profile import extract,HIST,Bank
from paper_revision.models_revision import build_revision_bpmr
from bpmr_net.models import B0Network
from bpmr_net.features import B0_OFFSETS
from typing import Iterable
OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026'
def h(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def jh(a):return hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ns=dict(np=np,torch=torch,Any=object,Iterable=Iterable,PROJECT_LABEL_COUNT=4,
        _AXIS={'x':(1,2),'y':(2,0),'z':(0,1)},ORIENTATIONS=('x','y','z'),PHASES=(0,1,2,3),full_square_neighbour_offsets=lambda r:list(B0_OFFSETS))
    extract(HIST/'.remote_src/fu_style_senn_core.py','normalize_categorical_probabilities',ns)
    tf=extract(HIST/'fu_style_cpdf_runtime.py','integrate_exactly_four_phase_template_outputs',ns)
    of=extract(HIST/'fu_style_cpdf_runtime.py','integrate_exactly_three_orientation_cpdfs',ns)
    encode=extract(HIST/'step65b_condition_cpdf_predictor.py','encode_periodic_3d_neighbourhood',ns)
    reference=extract(HIST/'r2_lite_3d_pilot.py','predictor_probability',ns)
    cases=list(itertools.product(['WF2_PROFILE','GI_PROFILE'],['dense','coarse'],[2701,2702],['forward','reverse','random_per_sweep'],['balanced','minority']))
    prespec=dict(cases=cases,shape_xyz=[4,4,4],sweeps=2,tolerance_abs=1e-7,
        truth_access='none; flow-state labels are fixture names, not real WF2/GI volumes',
        warning='Cases are deliberately small periodic fixtures, not intended scientific ROI size; frozen before any comparisons.')
    (OUT/'R1_RECONSTITUTION_P2_CASES.json').write_text(json.dumps(prespec,indent=2),encoding='utf-8')
    records=[]
    for variant in ['B0','BPMR']:
        models=[]
        for i in range(12):
            torch.manual_seed(182+i);m=B0Network()
            if variant=='BPMR':
                m=build_revision_bpmr(m,'A3')
                with torch.no_grad():m.residual_head.weight.normal_(0,0.03)
            # Explicit low/high entropy model situations among template banks.
            if i%2==0:
                with torch.no_grad():
                    b=m if variant=='B0' else m.b0;b.network[-2].bias.copy_(torch.tensor([-12.,12.,-12.,-12.]))
            models.append(m.eval())
        nested={a:{p:models[4*i+p] for p in range(4)} for i,a in enumerate(['x','y','z'])}
        bank=Bank(models,variant=='BPMR').eval()
        with torch.inference_mode():traced=torch.jit.freeze(torch.jit.trace(bank,torch.zeros(3,400)))
        def ref(s,index):return reference(nested,s,index)
        def opt(s,index):
            # Trace-only path retains exact reference encoding. Feature-vector optimization
            # was tested P1 but is not conflated with this expanded audit.
            x=np.stack([encode(s,index,a,4) for a in ['x','y','z']]);p=traced(torch.from_numpy(x)).numpy()
            return of([tf(p[i*4:(i+1)*4]) for i in range(3)])
        for family,density,seed,order,composition in cases:
            rng=np.random.default_rng(seed+(0 if family=='WF2_PROFILE' else 99))
            prior=[.25]*4 if composition=='balanced' else [.01,.02,.94,.03]
            initial=rng.choice(4,(4,4,4),p=prior).astype(np.int16)
            initial[0,0,0]=-1;support=initial>=0
            clamp=np.zeros(initial.shape,bool);clamp[0,:,:]=True
            if density=='dense':clamp[:,0,:]=True;clamp[:,:,0]=True
            clamp&=support
            coords=np.argwhere(support&~clamp)
            def run(fn,resume):
                s=initial.copy();r=np.random.default_rng(seed);saved=[];allp=[];allorder=[];beg=time.perf_counter()
                for sweep in range(2):
                    indices=coords.copy()
                    if order=='reverse':indices=indices[::-1]
                    if order=='random_per_sweep':indices=indices[r.permutation(len(indices))]
                    for idx in indices:
                        idx=tuple(int(v) for v in idx);p=fn(s,idx);allp.append(p.copy());allorder.append(idx);s[idx]=r.choice(4,p=p)
                    saved.append(dict(state_sha256=h(s),rng_sha256=jh(r.bit_generator.state),clamp_violations=int(np.count_nonzero(s[clamp]!=initial[clamp])),invalid_phase_count=int(np.count_nonzero((s[support]<0)|(s[support]>3)))))
                    if resume and sweep==0:
                        # Actual disk boundary, not in-memory copy only.
                        with tempfile.TemporaryDirectory() as temp:
                            f=Path(temp)/'resume.npz';np.savez(f,state=s,rng=np.array(json.dumps(r.bit_generator.state)))
                            with np.load(f,allow_pickle=False) as z:
                                s=z['state'].copy();payload=json.loads(str(z['rng']))
                            r=np.random.default_rng();r.bit_generator.state=payload
                return np.array(allp),saved,h(np.array(allorder)),time.perf_counter()-beg
            with torch.inference_mode():
                a=run(ref,False);b=run(opt,False);c=run(opt,True)
            error=float(np.max(np.abs(a[0]-b[0])))
            records.append(dict(variant=variant,family=family,density=density,seed=seed,order=order,composition=composition,
                reference_probability_sha256=h(a[0]),optimized_probability_sha256=h(b[0]),max_abs_discrepancy=error,
                reference_sweeps=a[1],optimized_sweeps=b[1],resumed_sweeps=c[1],reference_order_sha256=a[2],optimized_order_sha256=b[2],
                bitwise_probabilities=bool(np.array_equal(a[0],b[0])),state_rng_order_equal=a[1]==b[1] and a[2]==b[2],
                disk_resume_equal=b[1]==c[1] and b[2]==c[2] and np.array_equal(b[0],c[0]),
                reference_seconds=a[3],optimized_seconds=b[3],steps=len(a[0]),sampled_rss_bytes=psutil.Process().memory_info().rss))
        print(variant,len(records),flush=True)
    report=dict(identity='REVISION_REPLICATION',case_manifest_sha256=hashlib.sha256((OUT/'R1_RECONSTITUTION_P2_CASES.json').read_bytes()).hexdigest(),
        cases=records,tested_identity=os.environ.get('USERNAME'),python_executable=sys.executable,
        reference_scope='Unmodified AST extracted predictor/encoder/fusion; synthetic harness, exact original runner unavailable Windows dependencies.',
        fixture_pass=all(r['max_abs_discrepancy']<=1e-7 and r['state_rng_order_equal'] and r['disk_resume_equal'] for r in records),
        G2='FAIL_INTENDED_ISOLATED_WORKER_AND_FULL_RUNNER_NOT_TESTED',
        limitations=['WF2/GI fixtures not real source trajectories','4cube wrap fixture not48cube full sweep','No trained revision checkpoint exists','Only trace-only expanded; P1 vector-feature path limited fixtures','No test is executed under required isolated scientific identity'])
    (OUT/'R1_RECONSTITUTION_P2_RUNNER_EQUIVALENCE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
