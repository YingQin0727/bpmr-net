"""New four-phase hard-clamped Yeong–Torquato-style feasibility prototype.

No third-party implementation copied. Not an exact reproduction of published
binary sandstone implementation. Only registered TRAINING labels are read.
"""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import psutil

OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026'

def descriptors(labels,support,lags=(1,2,4)):
    """Nonperiodic per-axis phase S2 and lineal-path; supported pairs/lines only."""
    values=[]
    for axis in range(labels.ndim):
        for lag in lags:
            left=[slice(None)]*labels.ndim;right=left.copy()
            left[axis]=slice(0,-lag);right[axis]=slice(lag,None)
            valid=support[tuple(left)]&support[tuple(right)]
            for phase in range(4):
                a=labels[tuple(left)]==phase;b=labels[tuple(right)]==phase
                values.append(float((a&b&valid).sum()/max(1,valid.sum())))
            line_valid=np.ones_like(valid);line_phase=np.ones((4,*valid.shape),bool)
            for delta in range(lag+1):
                sl=[slice(None)]*labels.ndim;sl[axis]=slice(delta,labels.shape[axis]-lag+delta)
                line_valid&=support[tuple(sl)]
                for phase in range(4):line_phase[phase]&=labels[tuple(sl)]==phase
            values.extend(float((line_phase[p]&line_valid).sum()/max(1,line_valid.sum())) for p in range(4))
    return np.array(values).reshape(labels.ndim,-1).mean(0)

def main():
    start=time.perf_counter()
    rows=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    selection=[]
    for material in ['WF2','GI']:
        selection.append(sorted((r for r in rows if r['role']=='train' and r['material']==material and '250' in r['regimes_um'].split(';')),key=lambda r:r['plane_id'])[0])
    evidence=[];results=[]
    for row in selection:
        path=OUT/row['labels_path'];sp=OUT/row['support_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['labels_sha256']
        labels=np.fromfile(path,np.uint8).reshape(json.loads(row['shape']))
        support=np.fromfile(sp,np.uint8).reshape(labels.shape).astype(bool)
        prep=time.perf_counter();target=descriptors(labels,support)
        counts=np.bincount(labels[support],minlength=4)[:4];prior=counts/counts.sum()
        target_seconds=time.perf_counter()-prep
        evidence.append(dict(plane_id=row['plane_id'],role='train',labels_sha256=row['labels_sha256'],support_sha256=row['support_sha256'],prior=prior.tolist(),target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),target_seconds=target_seconds))
        for size in [8,16]:
            for seed in [2701,2702]:
                rng=np.random.default_rng(seed);s=rng.choice(4,size=(size,)*3,p=prior).astype(np.uint8)
                mask=np.zeros(s.shape,bool);mask[size//2,:,:]=True;mask[:,size//2,:]=True;mask[:,:,size//2]=True
                observed=s[mask].copy();coords=np.argwhere(~mask)
                initial_counts=np.bincount(s.ravel(),minlength=4)
                begun=time.perf_counter();current=float(np.mean((descriptors(s,np.ones_like(s,bool))-target)**2));initial_energy=current
                accepted=0;times=[]
                for step in range(64):
                    t=time.perf_counter();a,b=coords[rng.choice(len(coords),2,replace=False)];a=tuple(a);b=tuple(b)
                    s[a],s[b]=s[b],s[a]
                    proposal=float(np.mean((descriptors(s,np.ones_like(s,bool))-target)**2))
                    temperature=0.01*(0.001**(step/63))
                    if proposal<=current or rng.random()<np.exp((current-proposal)/temperature):current=proposal;accepted+=1
                    else:s[a],s[b]=s[b],s[a]
                    assert np.array_equal(s[mask],observed)
                    times.append(time.perf_counter()-t)
                assert np.array_equal(initial_counts,np.bincount(s.ravel(),minlength=4))
                results.append(dict(material=row['material'],size=size,seed=seed,moves=64,accepted=accepted,
                    median_move_seconds=float(np.median(times[8:])),total_seconds=time.perf_counter()-begun,
                    clamping_violations=0,phase_count_preserved=True,initial_objective=initial_energy,final_objective=current,
                    final_state_sha256=hashlib.sha256(s.tobytes()).hexdigest()))
    report=dict(identity='REVISION_REPLICATION',method='Yeong–Torquato-style multiphase correlation/lineal-path simulated annealing adapter',
        publication=dict(doi='10.1103/PhysRevE.58.224',url='https://doi.org/10.1103/PhysRevE.58.224',verified='Publisher abstract/metadata verified; full text unavailable; not exact author implementation'),
        implementation_license='New repository-native code under repository LICENSE; no third-party code copied. MaterialReconstruction.jl binary implementation not imported.',
        information_boundary='Only two registered G250 training planes and synthetic3D states used. No source dense file, test ROI or observation file opened.',
        test_scope='Synthetic conditioning labels, non-test train-derived statistics,8/16cube64moves; not paper results.',
        predefined_candidate=dict(roi_shape=[48]*3,seeds=[2701,2702,2703],proposed_moves=10000,lags_voxels=[1,2,4],
            temperature='0.01 geometric to0.00001',stopping='fixed10000attempts, no test-dependent stop',descriptor_weights='equal per phase/lag/axis-average',
            conditioning='same legal observations clamped; only unobserved voxel swaps',
            prior='train-only phase fractions; preserve initialized free phase counts; do not match hidden truth counts'),
        fairness_limits=['Prototype uses only2train planes; production must pool exactly same global-regime allowed train planes as neural model.',
            '8/16cube cost cannot certify48cube runtime; full-descriptor recomputation per move scales withvolume; no validated incremental48cube implementation.',
            'Fixed phase-count swaps restrict volume-fraction relaxation; state initial count derives train prior, not truth.',
            '10000moves likely insufficient mixing; validation-only convergence/budget study needed before credible competitiveness claim.',
            'Axis-averaged low-lag statistics are a restricted published-framework adaptation; cannot call canonical published implementation.'],
        evidence=evidence,runs=results,peak_sampled_rss_bytes=psutil.Process().memory_info().rss,
        total_wall_seconds=time.perf_counter()-start,G3='FAIL_INCOMPLETE_PRODUCTION_SCALE_AND_INFORMATION_PARITY_VALIDATION')
    (OUT/'R1_RECONSTITUTION_P2_EXTERNAL_BASELINE_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'runs':len(results),'seconds':report['total_wall_seconds']}))

if __name__=='__main__':main()
