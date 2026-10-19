"""Non-scientific states, actual 12-model forwards, extracted reference functions.

Historical files are parsed, never edited/imported wholesale (their main modules
include inaccessible Linux paths). Dependency bindings are explicitly disclosed.
"""
import ast
import copy
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bpmr_net.models import B0Network
from bpmr_net.features import B0_OFFSETS
from paper_revision.models_revision import build_revision_bpmr

HIST=Path('C:/Users/admin/Documents/博士申请论文')
OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026'
P='R1_RECONSTITUTION_P1_'

def extract(path,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Bank(nn.Module):
    def __init__(self,models,residual):
        super().__init__();self.models=nn.ModuleList(models);self.residual=residual
    def forward(self,x):
        outputs=[]
        for i,m in enumerate(self.models):
            result=m(x[i//4:i//4+1])
            p=result[1] if self.residual else result
            outputs.append(p/torch.clamp(p.sum(dim=1,keepdim=True),min=1e-12))
        return torch.cat(outputs,dim=0)

def main():
    torch.manual_seed(1041);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    namespace=dict(np=np,torch=torch,Any=object,Iterable=object,
        _AXIS={'x':(1,2),'y':(2,0),'z':(0,1)},ORIENTATIONS=('x','y','z'),PHASES=(0,1,2,3),PROJECT_LABEL_COUNT=4,
        full_square_neighbour_offsets=lambda radius:list(B0_OFFSETS))
    normalize=extract(HIST/'.remote_src/fu_style_senn_core.py','normalize_categorical_probabilities',namespace)
    fusion_path=HIST/'fu_style_cpdf_runtime.py'
    # Avoid evaluation of typing-only annotation Iterable[np.ndarray].
    from typing import Iterable
    namespace['Iterable']=Iterable
    tf=extract(fusion_path,'integrate_exactly_four_phase_template_outputs',namespace)
    of=extract(fusion_path,'integrate_exactly_three_orientation_cpdfs',namespace)
    encode=extract(HIST/'step65b_condition_cpdf_predictor.py','encode_periodic_3d_neighbourhood',namespace)
    reference=extract(HIST/'r2_lite_3d_pilot.py','predictor_probability',namespace)
    offsets=np.array(B0_OFFSETS)
    features=np.zeros((3,80,5),np.float32)
    def fast_features(state,index):
        features.fill(0)
        for axis,(a,b) in enumerate(namespace['_AXIS'].values()):
            coords=[np.full(80,index[c],dtype=np.int64) for c in range(3)]
            coords[a]=(coords[a]+offsets[:,0])%state.shape[a]
            coords[b]=(coords[b]+offsets[:,1])%state.shape[b]
            labels=state[tuple(coords)];valid=(labels>=0)&(labels<4)
            if np.any((~valid)&(labels!=-1)):raise ValueError('invalid label')
            slots=np.flatnonzero(valid);features[axis,slots,labels[valid]]=1
            features[axis,slots,4]=1
        return features.reshape(3,400)
    state=np.random.default_rng(66).integers(0,4,(48,48,48),dtype=np.int16)
    state[0,0,0]=-1
    input_state_sha=hashlib.sha256(state.tobytes()).hexdigest()
    comparisons=[];profiles=[];loads=[];rss=[]
    for variant in ['A0','A3']:
        models=[]
        for i in range(12):
            torch.manual_seed(1000+i)
            model=B0Network() if variant=='A0' else build_revision_bpmr(B0Network(),variant)
            if variant=='A3':
                # Nonzero corrections exercise gate/residual rather than zero-init identity.
                with torch.no_grad():model.residual_head.weight.normal_(0,0.05)
            model.eval();models.append(model)
        nested={axis:{phase:models[4*i+phase] for phase in range(4)} for i,axis in enumerate(['x','y','z'])}
        bank=Bank(models,variant=='A3').eval()
        scripted=torch.jit.freeze(torch.jit.trace(bank,torch.zeros(3,400),check_trace=True))
        def optimized(s,index):
            probs=scripted(torch.from_numpy(fast_features(s,index))).numpy()
            return of([tf(probs[i*4:(i+1)*4]) for i in range(3)])
        def plain(s,index):return reference(nested,s,index)
        def traced_oldencode(s,index):
            x=np.stack([encode(s,index,a,4) for a in ['x','y','z']])
            probs=scripted(torch.from_numpy(x)).numpy()
            return of([tf(probs[i*4:(i+1)*4]) for i in range(3)])
        buffer=io.BytesIO();torch.save(bank.state_dict(),buffer)
        timings=[]
        for _ in range(7):
            buffer.seek(0);start=time.perf_counter();bank.load_state_dict(torch.load(buffer,weights_only=True),strict=True);timings.append(time.perf_counter()-start)
        loads.append(dict(variant=variant,bank_bytes=buffer.getbuffer().nbytes,median_warm_load_seconds=statistics.median(timings[2:])))
        with torch.inference_mode():
            errors=[]
            for index in [(0,0,1),(47,47,47),(12,24,35),(0,1,0)]:
                for axis in ['x','y','z']:
                    assert np.array_equal(encode(state,index,axis,4),fast_features(state,index)[['x','y','z'].index(axis)])
                errors.append(float(np.max(np.abs(plain(state,index)-optimized(state,index)))))
            # Decompose actual predictor components on one non-scientific 48-cube.
            costs={k:[] for k in ['features','torch_12_forwards','fusion','rng_choice']}
            component_rng=np.random.default_rng(8347)
            for iteration in range(28):
                index=(iteration%48,13,27);t=time.perf_counter()
                x=[torch.from_numpy(encode(state,index,a,4)).unsqueeze(0) for a in ['x','y','z']]
                t1=time.perf_counter();outputs=[]
                for i,m in enumerate(models):
                    raw=m(x[i//4]);p=raw if variant=='A0' else raw[1]
                    p=p/p.sum(1,keepdim=True).clamp_min(1e-12);outputs.append(p.numpy()[0])
                t2=time.perf_counter();p=of([tf(outputs[i*4:(i+1)*4]) for i in range(3)]);t3=time.perf_counter()
                component_rng.choice(4,p=p);t4=time.perf_counter()
                if iteration>=4:
                    for k,v in zip(costs,[t1-t,t2-t1,t3-t2,t4-t3]):costs[k].append(v)
            for threads in [1,2,4]:
                torch.set_num_threads(threads)
                for name,fn in [('reference',plain),('trace_only',traced_oldencode),('trace_vector_features',optimized)]:
                    for _ in range(5):fn(state,(12,13,14))
                    repeats=[]
                    for repeat in range(3):
                        s=state.copy();rng=np.random.default_rng(44);start=time.perf_counter()
                        for j in range(32):
                            index=(12,13,j);p=fn(s,index);s[index]=rng.choice(4,p=p)
                        repeats.append((time.perf_counter()-start)/32)
                    profiles.append(dict(variant=variant,implementation=name,threads=threads,seconds_per_update=repeats,
                                         median_seconds_per_update=statistics.median(repeats),workload='32 immediate-writeback updates on 48^3 synthetic state, 3 repeats, 5 warmups'))
            torch.set_num_threads(1)
            # Two complete small sweeps: same clamping, update order, PCG64 choice stream.
            initial=np.random.default_rng(72).integers(0,4,(6,6,6),dtype=np.int16)
            clamp=np.zeros(initial.shape,bool);clamp[0,:,:]=True
            coords=[tuple(c) for c in np.argwhere(~clamp)]
            def trajectory(fn,resume=False):
                s=initial.copy();rng=np.random.default_rng(2701);prob=[];order=[]
                started=time.perf_counter()
                for step,index in enumerate(coords*2):
                    p=fn(s,index);prob.append(p.copy());order.append(index)
                    s[index]=rng.choice(4,p=p)
                    if resume and step==len(coords)-1:
                        saved=s.copy();saved_rng=copy.deepcopy(rng.bit_generator.state)
                        s=saved.copy();rng=np.random.default_rng();rng.bit_generator.state=saved_rng
                return s,np.array(prob),order,rng.bit_generator.state,time.perf_counter()-started
            base=trajectory(plain);fast=trajectory(optimized);resumed=trajectory(optimized,True)
            maxerr=float(np.max(np.abs(base[1]-fast[1])))
            comparisons.append(dict(variant=variant,tolerance_absolute=1e-7,conditional_probability_max_abs_error=maxerr,
                tolerance_pass=maxerr<=1e-7,conditional_probability_bitwise_equal=bool(np.array_equal(base[1],fast[1])),
                final_state_bitwise_equal=bool(np.array_equal(base[0],fast[0])),order_equal=base[2]==fast[2],
                rng_final_equal=base[3]==fast[3],clamping_preserved=bool(np.array_equal(fast[0][clamp],initial[clamp])),
                exact_resume_state=bool(np.array_equal(fast[0],resumed[0])),exact_resume_probabilities=bool(np.array_equal(fast[1],resumed[1])),
                exact_resume_rng=fast[3]==resumed[3],boundary_feature_checks_exact=True,
                fresh_state_fixed_probability_errors=errors,steps=len(coords)*2))
            comparisons[-1]['measured_reference_small_sweep_seconds']=base[4]/2
            comparisons[-1]['measured_optimized_small_sweep_seconds']=fast[4]/2
            comparisons[-1]['small_sweep_updates']=len(coords)
            profiles[-1]['component_reference_medians_threads1']={k:statistics.median(v) for k,v in costs.items()}
        rss.append(psutil.Process().memory_info().rss)
        print(json.dumps({'variant':variant,'comparison':comparisons[-1]}),flush=True)
    metadata=dict(identity='REVISION_REPLICATION',scientific_evidence=False,source_files=[dict(path=str(p),sha256=sha(p)) for p in [HIST/'r2_lite_3d_pilot.py',HIST/'step65b_condition_cpdf_predictor.py',fusion_path,HIST/'.remote_src/fu_style_senn_core.py']],
        reference_scope='Unmodified AST-extracted predictor/encoder/fusion functions; dependency bindings explicit: XYZ state, x->yz y->zx z->xy, radius4 row-major offsets from frozen B0_OFFSETS. Original whole runner not importable on Windows; this is a predictor harness, not validated complete runner.',
        model_identity='12 distinct fixed-seed random models per variant; nonzero A3 residual; no fit/checkpoint eligibility claim',
        state_sha256=input_state_sha,profiles=profiles,loads=loads,sampled_peak_rss_bytes=max(rss),
        bottlenecks='Reference encoder np.asarray(state,dtype=int) copies entire int16 state three times per update; models already resident per trajectory in original. Twelve eager forwards and many numpy normalizations/calls dominate.',
        limitations=['No test truth access','No real reconstruction','Partial synthetic trajectories do not prove all-input bitwise equivalence',
                     'TorchScript trace validated only registered shape/radius/model topology; torch.compile not measured; no GPU implementation evaluated',
                     'Process RSS samples exclude dataset staging; checkpoint loading warm memory excludes cold disk'])
    (OUT/(P+'RECONSTRUCTION_PROFILE.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (OUT/(P+'OPTIMIZATION_EQUIVALENCE.json')).write_text(json.dumps(dict(identity='REVISION_REPLICATION',comparisons=comparisons,
        scope='TESTED_FIXTURES_ONLY_NOT_GENERAL_BITWISE_PROOF',sampler_unchanged='Immediate writeback, raster ordering, numpy Generator PCG64 choice, clamping, periodic neighbor access retained.',
        primary_authority='Unmodified historical reference; optimized implementation not authorized for scientific campaign until complete worker/feature semantics validated.'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
