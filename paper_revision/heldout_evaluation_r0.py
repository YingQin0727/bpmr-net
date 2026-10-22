"""Independent held-out evaluator; freeze models and protocol before test IO."""
import argparse
import csv
import hashlib
import io
import json
import math
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.training_campaign_2d import unit_cells
from paper_revision.training_events import model_seed
from paper_revision.training_events import centers
from paper_revision.train_revision_2d import load_plane, probability
from bpmr_net.features import encode_b0_features
from paper_revision.training_access import TrainingAccess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/revision_mtcomm_2026'
TRAIN = BASE / 'REVISION_REPLICATION_training/DesignA'
OUT = BASE / 'REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0'
PREFIX = 'REVISION_DESIGNA_HELDOUT_R0_'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, obj):
    path = OUT / (PREFIX + name + '.json')
    data = json.dumps(obj, indent=2, allow_nan=False).encode()
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError('refusing to replace frozen artifact: ' + str(path))
    path.write_bytes(data)
    return sha(path)


def load(path):
    return torch.load(io.BytesIO(path.read_bytes()), weights_only=True, map_location='cpu')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    bpath = TRAIN / 'REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json'
    apath = TRAIN / 'REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json'
    assert sha(bpath) == '175f0544cddb3806159defe0204f614157d079b2c83a4f0e9a8b0235254a3232'
    release = json.loads((ROOT / 'paper_revision/TRAINING_2D_RELEASE_R2.json').read_bytes())
    assert sha(BASE / 'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv') == release['input_identity']['registry_sha256']
    for name, expected in release['launch_binding'].items():
        assert sha(ROOT / name) == expected, 'release binding conflict: ' + name
    b = json.loads(bpath.read_bytes())['checkpoints']
    a = json.loads(apath.read_bytes())
    units = {f'{m}/G{g}/{ax}/seed{s}/template{t}' for m, g, ax, s, t in unit_cells()}
    assert len(units) == len(b) == 120
    assert {r['unit'] for r in b} == units
    assert len(a['checkpoints']) == 720 and not a['failures']
    assert {(r['unit'], r['variant']) for r in a['checkpoints']} == {(u, v) for u in units for v in ('A0','A1','A2','A3','A4','A5')}
    frozen, errors, sources = [], [], {}
    for r in b + a['checkpoints']:
        variant = r.get('variant', 'B0_SOURCE')
        ident = r['unit'] + '/' + variant
        try:
            path = ROOT / r['path']
            assert path.resolve().is_relative_to(TRAIN.resolve())
            assert path.name in ('epoch50.pt', 'epoch50_reexecution.pt')
            digest = r.get('checkpoint_sha256', r.get('sha256'))
            assert sha(path) == digest, 'checkpoint hash conflict'
            ck = load(path)
            meta = ck['metadata']
            assert meta['unit'] == r['unit'] and meta['kind'] == 'EPOCH50'
            assert meta['historical_checkpoint'] is False
            m, g, ax, s, t = r['unit'].split('/')
            assert meta['model_seed'] == model_seed(m, int(g[1:]), ax, int(s[4:]), int(t[8:]))
            model = B0Network()
            if variant in ('A2','A3','A4','A5'):
                model = build_revision_bpmr(model, variant)
            model.load_state_dict(ck['state'], strict=True)
            assert all(torch.isfinite(v).all() for v in ck['state'].values())
            if variant == 'B0_SOURCE':
                assert r['epochs'] == 50 and len(r['epoch_records']) == 50
                sources[r['unit']] = (r, ck['state'])
            else:
                assert meta['variant'] == variant and meta['epochs_completed'] == 50
                assert [e['epoch'] for e in r['epochs']] == list(range(1,51))
                src, state = sources[r['unit']]
                assert r['source_checkpoint_sha256'] == meta['source_checkpoint_sha256'] == src['sha256']
                for role in ('train','validation'):
                    assert r[role+'_events'] == meta[role+'_events'] == src['events'][role], 'event identity conflict'
                steps = math.ceil(r['train_events']['count']/4096)
                assert all(e['optimizer_steps'] == steps for e in r['epochs'])
                assert all(math.isfinite(e['train_loss']) and all(math.isfinite(v) for v in e['validation'].values()) for e in r['epochs'])
                assert all(int(v['step']) == 50*steps for v in ck['optimizer']['state'].values())
                if variant in ('A2','A3','A4','A5'):
                    assert all(torch.equal(ck['state']['b0.'+k], v) for k,v in state.items()), 'frozen B0 modified'
                    assert model.dilations == ((1,1,1) if variant == 'A5' else (1,2,4))
                if variant == 'A4':
                    assert torch.count_nonzero(model.gate.weight) == 0 and not model.gate.weight.requires_grad
                assert sum(p.numel() for p in model.parameters() if p.requires_grad) == r['trainable_parameter_count']
            frozen.append(dict(unit=r['unit'], variant=variant, path=r['path'], sha256=digest,
                               strict_reload=True, finite_parameters=True, epochs=50,
                               parameters=sum(p.numel() for p in model.parameters()),
                               effective_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        except Exception as exc:
            errors.append(dict(identity=ident, error=type(exc).__name__+': '+str(exc)))
        if len(frozen) % 120 == 0:
            print('model freeze verified', len(frozen), 'errors', len(errors), flush=True)
    result = dict(status='PASS' if not errors and len(frozen)==840 else 'BLOCKED',
                  models=frozen, errors=errors, source_manifests={str(p.relative_to(ROOT)):sha(p) for p in (bpath,apath)},
                  registry_sha256=release['input_identity']['registry_sha256'],
                  code_sha256={n:sha(ROOT/n) for n in ('paper_revision/models_revision.py','paper_revision/training_events.py','src/bpmr_net/models.py','paper_revision/revision_protocol.json')})
    dump('MODEL_FREEZE', result)
    if errors:
        print(json.dumps(errors[:20]), flush=True)
        raise SystemExit('MODEL_IDENTITY_BLOCKED: no test data loaded')
    print('MODEL_FREEZE PASS 840/840', flush=True)


def write_csv(name, rows):
    path = OUT / (PREFIX + name + '.csv')
    assert not path.exists(), 'refusing to overwrite evaluation results'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_protocol():
    frozen = json.loads((OUT / (PREFIX+'MODEL_FREEZE.json')).read_bytes())
    assert frozen['status'] == 'PASS'
    protocol = dict(stage='REVISION_DESIGNA_HELDOUT_EVALUATION_R0',
        frozen_before_test_payload_read=True, model_freeze_sha256=sha(OUT/(PREFIX+'MODEL_FREEZE.json')),
        evaluator_sha256=sha(Path(__file__)), registry_sha256=frozen['registry_sha256'],
        test_role='test_truth', samples='All registered test_truth planes; all fully supported radius4 centers; plane_id lexical order, row-major center order.',
        context='2D teacher-forced neighborhood classification: 80 observed true neighboring phase slots, center label excluded; NOT sparse-observation reconstruction or rollout.',
        selection='One shared event list per material and orientation across ALL spacing/template/training-seed/variant identities. If >100000 centers, fixed seeded uniform without replacement, sorted ordinals.',
        cap_per_material_orientation=100000, selection_seed=20261022,
        event_weights='Equal event weight within model; main summary equal source material, then equal spacing, then mean all registered orientation/template/seed conditions. No fabricated missing conditions.',
        ece='15 equal-width confidence bins; bin=min(floor(confidence*15),14); weighted absolute confidence-accuracy gap.',
        metrics=['categorical_nll','brier','ece','accuracy','balanced_accuracy','macro_f1','phase_precision','phase_recall','phase_f1','predicted_fraction','truth_fraction','prevalence_error'],
        nll='mean -log(max(float32 probability_of_truth,1e-12)); accumulation float64',
        brier='mean sum over FOUR classes squared probability-onehot difference, range0..2 (not divided by4)',
        undefined='precision and recall undefined denominators -> empty CSV/null; per-phase F1 zero when truth or predictions present but TP=0, null if both absent; fixed FOUR-class macro F1 treats null F1 as ZERO (no absent-class dropping). Balanced accuracy averages recalls for truth-present classes; report present class count.',
        comparisons=[['A0','A1'],['A1','A3'],['A2','A3'],['A3','A4'],['A3','A5'],['B0_SOURCE','A0']],
        difference='right minus left; negative NLL/Brier/ECE and positive accuracy/F1 favor right',
        uncertainty='No event bootstrap, no event p-values, no population-material CI with only two sources. Condition paired differences descriptive. Report seed SD/range conditional on fixed material/spacing/orientation/template; for exactly three registered seeds report pointwise 95% t CI (df2,critical4.30265273) for training-seed mean only. n1: CI undefined. Seeds share test events. No claim of material-population significance; multiple conditional CIs exploratory, not multiplicity-adjusted.',
        gate='A3 diagnostics stratified by ROI, true phase and B0/A3 correctness quadrant; descriptive Pearson correlations with entropy/margin/support; no causal interpretation.',
        inference=dict(device='cpu',threads=1,dtype='float32',batch_size=4096,no_updates=True),
        output_namespace=str(OUT.relative_to(ROOT)),
        prohibited=['retraining','tuning','3D reconstruction','external comparator','test-based checkpoint selection'],
        limitations=['OS_LEVEL_ISOLATION=NOT_ESTABLISHED','Python IO audit only; direct native syscall coverage not established for evaluation','Two source volumes; ROIs and orientations are correlated views, not independent materials'])
    dump('PROTOCOL',protocol)
    print('EVALUATION PROTOCOL FROZEN; test labels not yet loaded', flush=True)


def metrics(p, y):
    assert p.shape == (len(y),4) and len(y)>0
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=2e-6)
    pred=p.argmax(1); confidence=p.max(1).astype(np.float64)
    cm=np.bincount(y*4+pred,minlength=16).reshape(4,4)
    true=cm.sum(1); predicted=cm.sum(0); tp=cm.diagonal()
    precision=np.divide(tp,predicted,out=np.full(4,np.nan),where=predicted>0)
    recall=np.divide(tp,true,out=np.full(4,np.nan),where=true>0)
    f1=np.divide(2*tp,true+predicted,out=np.full(4,np.nan),where=(true+predicted)>0)
    bins=np.minimum((confidence*15).astype(int),14)
    ece=sum(abs(float(confidence[bins==i].sum())-int((pred[bins==i]==y[bins==i]).sum())) for i in range(15))/len(y)
    pd=p.astype(np.float64); pd[np.arange(len(y)),y]-=1
    out=dict(count=len(y),categorical_nll=float(-np.log(p[np.arange(len(y)),y].clip(1e-12).astype(np.float64)).mean()),
             brier=float(np.square(pd).sum(1).mean()),ece=ece,accuracy=float(tp.sum()/len(y)),
             balanced_accuracy=float(np.nanmean(recall)),macro_f1=float(np.nan_to_num(f1,nan=0).mean()),truth_present_classes=int((true>0).sum()))
    phases=[dict(phase=k,truth_count=int(true[k]),predicted_count=int(predicted[k]),true_positive=int(tp[k]),
                 precision=None if np.isnan(precision[k]) else float(precision[k]),
                 recall=None if np.isnan(recall[k]) else float(recall[k]),f1=None if np.isnan(f1[k]) else float(f1[k]),
                 predicted_fraction=float(predicted[k]/len(y)),truth_fraction=float(true[k]/len(y)),
                 prevalence_error=float((predicted[k]-true[k])/len(y)),absolute_prevalence_error=float(abs(predicted[k]-true[k])/len(y))) for k in range(4)]
    return out,phases


def metric_self_test():
    perfect=np.eye(4,dtype=np.float32); y=np.arange(4)
    m,_=metrics(perfect,y)
    assert m['brier']==m['categorical_nll']==m['ece']==0 and m['macro_f1']==1
    m,ph=metrics(np.array([[1,0,0,0]],np.float32),np.array([0]))
    assert m['macro_f1']==.25 and ph[1]['f1'] is None and m['balanced_accuracy']==1
    m,_=metrics(np.full((4,4),.25,np.float32),y)
    assert abs(m['categorical_nll']-math.log(4))<1e-10 and m['brier']==.75


def identity_columns(unit,variant):
    m,g,axis,seed,template=unit.split('/')
    return dict(unit=unit,variant=variant,material=m,spacing_um=int(g[1:]),orientation=axis,
                training_seed=int(seed[4:]),template=int(template[8:]))


def test_events(rows, material, axis):
    candidates=sorted([r for r in rows if (r['material'],r['normal_axis'],r['role'])==(material,axis,'test_truth')],key=lambda r:r['plane_id'])
    arrays=[]; identities=[]; seen=set()
    for row in candidates:
        labels,support=load_plane(BASE,row)
        coords=centers(support); rr,cc=coords.T
        assert len(coords)>0 and (labels[rr,cc]<4).all()
        arrays.append((encode_b0_features(labels,support,rr,cc),labels[rr,cc].astype(np.int64),np.full(len(rr),int(row['roi'][3:]),dtype=np.int32)))
        identities.append((row,coords))
        assert row['plane_id'] not in seen
        seen.add(row['plane_id'])
    x=np.concatenate([a[0] for a in arrays]); y=np.concatenate([a[1] for a in arrays]); roi=np.concatenate([a[2] for a in arrays])
    ordinals=np.arange(len(x),dtype=np.int64)
    if len(x)>100000:
        ordinals=np.sort(np.random.default_rng(20261022).choice(len(x),100000,replace=False))
    h=hashlib.sha256(); offset=0; parts=[]
    for row,coords in identities:
        chosen=ordinals[(ordinals>=offset)&(ordinals<offset+len(coords))]-offset
        selected=coords[chosen]
        part=dict(plane_id=row['plane_id'],labels_sha256=row['labels_sha256'],support_sha256=row['support_sha256'],count=len(selected),coordinates_sha256=hashlib.sha256(selected.astype('<i4').tobytes()).hexdigest())
        h.update(json.dumps(part,sort_keys=True).encode()); parts.append(part); offset+=len(coords)
    x=x[ordinals];y=y[ordinals];roi=roi[ordinals]
    return x,y,roi,dict(material=material,orientation=axis,population=offset,count=len(y),event_sha256=h.hexdigest(),
        feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),target_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
        ordinal_sha256=hashlib.sha256(ordinals.astype('<i8').tobytes()).hexdigest(),planes=parts)


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def evaluate():
    started=time.perf_counter()
    freeze_path=OUT/(PREFIX+'MODEL_FREEZE.json'); protocol_path=OUT/(PREFIX+'PROTOCOL.json')
    frozen=json.loads(freeze_path.read_bytes()); protocol=json.loads(protocol_path.read_bytes())
    assert protocol['model_freeze_sha256']==sha(freeze_path) and protocol['evaluator_sha256']==sha(Path(__file__))
    assert frozen['status']=='PASS'
    assert sha(BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv')==protocol['registry_sha256']
    for name,digest in frozen['code_sha256'].items(): assert sha(ROOT/name)==digest
    metric_self_test()
    import psutil
    process=psutil.Process()
    records=list(csv.DictReader((BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    access_path=TRAIN/'REVISION_DESIGNA_A0A5_R0_PYTHON_ACCESS.jsonl'
    access_tasks=set(); access_count=0; violations=[]
    with access_path.open(encoding='utf-8') as log:
        for line in log:
            item=json.loads(line); access_tasks.add((item['unit'],item['variant']))
            for rec in item['records']:
                access_count+=1
                if not rec['allowed']: violations.append(rec)
                if rec['operation']=='read' and ('test_truth' in rec['path'].lower() or 'incomplete' in rec['path'].lower()): violations.append(rec)
    assert len(access_tasks)==720 and not violations, 'formal training access audit conflict'
    # Refuse spatial intersection between held-out and fitting label footprints.
    for row in [r for r in records if r['role']=='test_truth']:
        box=json.loads(row['bounds_zyx'])
        for fit in [r for r in records if r['material']==row['material'] and r['role'] in ('train','validation')]:
            other=json.loads(fit['bounds_zyx'])
            assert not all(max(a[0],b[0])<min(a[1],b[1]) for a,b in zip(box,other)), 'test/fitting geometry overlap'
    arows=json.loads((TRAIN/'REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json').read_bytes())['checkpoints']
    brows=json.loads((TRAIN/'REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json').read_bytes())['checkpoints']
    training={ (r['unit'],r.get('variant','B0_SOURCE')):r for r in brows+arows}
    reads=[BASE/r[k+'_path'] for r in records if r['role']=='test_truth' for k in ('labels','support')]
    reads += [ROOT/r['path'] for r in frozen['models']]
    reads += [freeze_path,protocol_path]
    names=['EVENT_IDENTITY.json','ALL_METRICS.csv','PHASE_METRICS.csv','PAIRED_COMPARISONS.csv','GATE_DIAGNOSTICS.csv','RUNTIME.csv','CONDITION_ROI_METRICS.csv','SEED_UNCERTAINTY.csv','FINAL_AUDIT.json','SUMMARY.md','ACCESS.jsonl','RESULT_MANIFEST.json']
    writes=[OUT/(PREFIX+n) for n in names]
    reads += writes
    assert not any(p.exists() for p in writes), 'evaluation already started; never silently rerun held-out'
    guard=TrainingAccess(reads,writes,[Path(sys.base_prefix)/'Lib',Path(sys.base_prefix)/'DLLs',ROOT/'.venv-revision311/Lib/site-packages'])
    guard.install();guard.phase='evaluation_registered_test_inputs'
    all_metrics=[]; phases=[]; gate_rows=[]; runtime=[]; roi_metrics=[]; events=[]
    for material in ('GI','WF2'):
        for axis in 'xyz':
            data_start=time.perf_counter(); x,y,roi,event=test_events(records,material,axis); events.append(event)
            preparation=time.perf_counter()-data_start
            selected=[r for r in frozen['models'] if r['unit'].split('/')[0]==material and r['unit'].split('/')[2]==axis]
            for r in selected:
                guard.phase='evaluation_'+r['unit']+'/'+r['variant']
                start=time.perf_counter(); cpu=time.process_time(); path=ROOT/r['path']
                assert sha(path)==r['sha256']
                ck=load(path); model=B0Network()
                if r['variant'] in ('A2','A3','A4','A5'): model=build_revision_bpmr(model,r['variant'])
                model.load_state_dict(ck['state'],strict=True);model.eval()
                load_time=time.perf_counter()-start
                predictions=[]; diagnostics=[]; inference_start=time.perf_counter()
                with torch.inference_mode():
                    for begin in range(0,len(x),4096):
                        bx=torch.from_numpy(x[begin:begin+4096]); raw=model(bx)
                        if isinstance(raw,tuple):
                            p=raw[1]
                            if r['variant']=='A3':
                                gi=model.gate_inputs(bx,raw[3])
                                shift=(p-raw[3]).abs().sum(1)
                                diagnostics.append(np.column_stack([gi.numpy(),raw[4].numpy(),raw[5].norm(dim=1).numpy(),shift.numpy(),raw[3].argmax(1).numpy()]))
                        else: p=raw/raw.sum(1,keepdim=True)
                        predictions.append(p.numpy().copy())
                inference=time.perf_counter()-inference_start
                p=np.concatenate(predictions); ident=identity_columns(r['unit'],r['variant'])
                metric,phase=metrics(p,y)
                all_metrics.append(dict(**ident,**metric,event_sha256=event['event_sha256'],checkpoint_sha256=r['sha256']))
                phases.extend(dict(**ident,**pr) for pr in phase)
                for rid in sorted(set(roi.tolist())):
                    met,_=metrics(p[roi==rid],y[roi==rid]);roi_metrics.append(dict(**ident,roi=rid,**met))
                if diagnostics:
                    d=np.concatenate(diagnostics); before=d[:,6].astype(int)==y;after=p.argmax(1)==y
                    for rid in sorted(set(roi.tolist())):
                        for phase_id in range(4):
                            for code in range(4):
                                # code0 wrong/wrong,1 wrong/correct,2 correct/wrong,3 correct/correct
                                mask=(roi==rid)&(y==phase_id)&((before.astype(int)*2+after.astype(int))==code)
                                if not mask.any(): continue
                                z=d[mask]
                                vals={n:float(z[:,k].mean()) for k,n in enumerate(['b0_entropy','top_two_margin','support_ratio','learned_gate','residual_l2','probability_shift_l1'])}
                                gate_rows.append(dict(**ident,roi=rid,phase=phase_id,b0_correct=bool(code//2),a3_correct=bool(code%2),count=len(z),
                                    **vals,gate_sd=float(z[:,3].std()),gate_entropy_pearson=corr(z[:,3],z[:,0]),gate_margin_pearson=corr(z[:,3],z[:,1]),gate_support_pearson=corr(z[:,3],z[:,2])))
                    vals={n:float(d[:,k].mean()) for k,n in enumerate(['b0_entropy','top_two_margin','support_ratio','learned_gate','residual_l2','probability_shift_l1'])}
                    gate_rows.append(dict(**ident,roi='ALL',phase='ALL',b0_correct='ALL',a3_correct='ALL',count=len(d),**vals,
                        gate_sd=float(d[:,3].std()),gate_entropy_pearson=corr(d[:,3],d[:,0]),gate_margin_pearson=corr(d[:,3],d[:,1]),gate_support_pearson=corr(d[:,3],d[:,2])))
                tr=training[(r['unit'],r['variant'])]
                runtime.append(dict(**ident,events=len(y),parameters=r['parameters'],effective_trainable_parameters=r['effective_trainable_parameters'],
                    training_wall_seconds=tr.get('elapsed_seconds',tr.get('task_wall_seconds_including_preparation')),
                    checkpoint_load_seconds=load_time,inference_seconds=inference,cpu_seconds=time.process_time()-cpu,
                    data_preparation_seconds_shared=preparation,process_rss_bytes=process.memory_info().rss,
                    process_lifetime_peak_rss_bytes=getattr(process.memory_info(),'peak_wset',process.memory_info().rss),threads=1))
                with (OUT/(PREFIX+'ACCESS.jsonl')).open('a',encoding='utf-8') as log:
                    log.write(json.dumps(dict(unit=r['unit'],variant=r['variant'],records=guard.records))+'\n')
                guard.records.clear()
                if len(all_metrics)%20==0: print('evaluated',len(all_metrics),'/840',r['unit'],r['variant'],flush=True)
    dump('EVENT_IDENTITY',dict(protocol_sha256=sha(protocol_path),groups=events,paired_across_all_variants=True))
    finalize(all_metrics,phases,gate_rows,runtime,roi_metrics,started,frozen,protocol,access_count)


METRIC_NAMES=('categorical_nll','brier','ece','accuracy','balanced_accuracy','macro_f1')


def weighted_mean(rows,metric):
    values=[]
    for material in sorted({r['material'] for r in rows}):
        spacing_values=[]
        for spacing in sorted({r['spacing_um'] for r in rows if r['material']==material}):
            spacing_values.append(np.mean([r[metric] for r in rows if r['material']==material and r['spacing_um']==spacing]))
        values.append(np.mean(spacing_values))
    return float(np.mean(values))


def finalize(all_metrics,phases,gates,runtime,roi_metrics,started,frozen,protocol,training_access_count):
    lookup={(r['unit'],r['variant']):r for r in all_metrics}
    comparisons=[]; seed_rows=[]
    units=sorted({r['unit'] for r in all_metrics})
    for left,right in protocol['comparisons']:
        for unit in units:
            a=lookup[(unit,left)];b=lookup[(unit,right)]
            assert a['event_sha256']==b['event_sha256']
            comparisons.append(dict(**identity_columns(unit,right),comparison=left+'_vs_'+right,
                **{k:b[k]-a[k] for k in METRIC_NAMES},count=a['count'],interpretation='right_minus_left; paired descriptive condition; not independent material replicate'))
    for comparison in sorted({r['comparison'] for r in comparisons}):
        group=[r for r in comparisons if r['comparison']==comparison]
        keys=sorted({(r['material'],r['spacing_um'],r['orientation'],r['template']) for r in group})
        for material,spacing,axis,template in keys:
            rows=[r for r in group if (r['material'],r['spacing_um'],r['orientation'],r['template'])==(material,spacing,axis,template)]
            for metric in METRIC_NAMES:
                vals=np.array([r[metric] for r in rows]); n=len(vals); sd=float(vals.std(ddof=1)) if n>1 else None
                half=4.302652729749464*sd/math.sqrt(n) if n==3 else None
                seed_rows.append(dict(comparison=comparison,material=material,spacing_um=spacing,orientation=axis,template=template,metric=metric,
                    n_training_seeds=n,mean_delta=float(vals.mean()),seed_sd=sd,min_delta=float(vals.min()),max_delta=float(vals.max()),
                    conditional_seed_ci95_lower=float(vals.mean()-half) if half is not None else None,
                    conditional_seed_ci95_upper=float(vals.mean()+half) if half is not None else None,
                    inference_scope='training seed variation conditional on same source and fixed test events; not material generalization'))
    btime={r['unit']:r['inference_seconds'] for r in runtime if r['variant']=='B0_SOURCE'}
    for r in runtime: r['inference_overhead_vs_paired_b0']=r['inference_seconds']/btime[r['unit']]
    for name,rows in [('ALL_METRICS',all_metrics),('PHASE_METRICS',phases),('PAIRED_COMPARISONS',comparisons),('GATE_DIAGNOSTICS',gates),('RUNTIME',runtime),('CONDITION_ROI_METRICS',roi_metrics),('SEED_UNCERTAINTY',seed_rows)]:
        write_csv(name,rows)
    elapsed=time.perf_counter()-started
    summaries={v:{k:weighted_mean([r for r in all_metrics if r['variant']==v],k) for k in METRIC_NAMES} for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')}
    changes={c:{k:weighted_mean([r for r in comparisons if r['comparison']==c],k) for k in METRIC_NAMES} for c in sorted({r['comparison'] for r in comparisons})}
    degraded=[r for r in phases if r['truth_count']>0 and r['predicted_count']==0]
    bad_conditions={c:sorted([r for r in comparisons if r['comparison']==c],key=lambda r:r['categorical_nll'],reverse=True)[:5] for c in changes}
    audit=dict(stage='REVISION_DESIGNA_HELDOUT_EVALUATION_R0',status='PASS',SCIENTIFIC_TRAINING_COMPLETE=True,
        HELDOUT_EVALUATION_COMPLETE=True,THREE_D_RECONSTRUCTION_COMPLETE=False,PUBLISHED_BASELINE_COMPLETE=False,
        models=840,b0=120,a0a5=720,unique_identities=len(lookup),test_event_pairing='PASS',
        model_integrity='PASS_840_STRICT_RELOAD_AND_SHA256_AND_FINITE',epochs50='PASS',frozen_b0_and_ablation_constraints='PASS',
        training_python_access_records=training_access_count,training_access_denials=0,evaluation_access_scope='Python audit hook; model payloads BytesIO, registered plane payloads from read_bytes; native syscalls not exhaustively traced',
        OS_LEVEL_ISOLATION='NOT_ESTABLISHED',elapsed_seconds=elapsed,summary_equal_material_spacing=summaries,
        paired_mean_differences=changes,phase_extinction_conditions=len(degraded),phase_extinction_details=degraded,
        worst_nll_conditions=bad_conditions,independent_source_volumes=2,
        effective_replicates=dict(condition_pairs=120,source_volumes=2,training_seeds='G250:3; G25/G1000:1; templates4 and orientations3 are repeated conditions; ROIs3 spatially correlated'),
        ci_limitations=protocol['uncertainty'],gate_limitations='Descriptive association only; dense supported events have constant support ratio; no causal gate claim',
        statistical_fallacy_scan=dict(coverage='11/11 considered',simpson='Report material/spacing strata and overall; inspect reversals',ecological='No event-to-material inference',berkson='Fixed support-conditioned population disclosed',collider='No outcome conditioning in primary metrics; gate correctness strata descriptive only',pseudoreplication='No event p-values',multiple_testing='Exploratory conditional seed intervals; no significance claims',selection='All840 frozen models; no favorable seed selection',causality='No causal entropy correlation claim',survivorship='Absent classes retained in fixed-four macro',uncertainty='Two sources insufficient material-population CI',generalization='Restricted to two sources and 2D teacher-forced task'))
    dump('FINAL_AUDIT',audit)
    lines=['# Design A held-out evaluation R0','',
        'REVISION_DESIGNA_HELDOUT_EVALUATION_R0=PASS',
        'SCIENTIFIC_TRAINING_COMPLETE=YES; HELDOUT_EVALUATION_COMPLETE=YES; THREE_D_RECONSTRUCTION_COMPLETE=NO; PUBLISHED_BASELINE_COMPLETE=NO','',
        'Material Passport: local frozen revision evidence; validate/run; no test-driven fitting; unpublished; no external upload.',
        '',f'840/840 models, 120 common B0 and 720 controlled models. Independent checkpoint reload/hash/finite checks passed before test IO. Evaluation wall seconds: {elapsed:.3f}.',
        '', 'This is 2D teacher-forced classification with true neighboring labels (center excluded), NOT 3D reconstruction or sparse-observation rollout. Two source volumes only. No broad cross-material generalization claim.',
        '', 'Primary means weight source materials equally, then spacings equally; all registered seeds/templates/orientations retained. Brier is the sum over four classes. Macro F1 keeps all four classes with absent-both F1 contributing zero.',
        '', '| Model | NLL | Brier | ECE | Accuracy | Balanced accuracy | Macro F1 |','|---|---:|---:|---:|---:|---:|---:|']
    for v,s in summaries.items(): lines.append('| '+v+' | '+' | '.join(f'{s[k]:.6f}' for k in METRIC_NAMES)+' |')
    lines+=['','Paired deltas below are right minus left. Negative NLL/Brier/ECE favors right. Each has120 paired conditions, but only2 source volumes; conditions/events are not120 independent material samples.','', '| Comparison | NLL delta | Brier delta | ECE delta | Accuracy delta | Balanced accuracy delta | Macro F1 delta |','|---|---:|---:|---:|---:|---:|---:|']
    for c,s in changes.items(): lines.append('| '+c+' | '+' | '.join(f'{s[k]:.6f}' for k in METRIC_NAMES)+' |')
    lines+=['',protocol['uncertainty'],'',f'Phase extinction: {len(degraded)} model-phase conditions have positive true prevalence but zero predicted prevalence. Inspect PHASE_METRICS and FINAL_AUDIT; do not omit negative conditions.','',
        'Gate diagnostics include entropy, margin, support ratio, gate, residual L2, probability shift and all four B0/A3 correctness categories, stratified by ROI/phase. Support ratio is constant for fully supported test neighborhoods; this cannot validate support-dependent gating under sparse observations.',
        'Gate/uncertainty correlation is descriptive and cannot establish a causal improvement. A3/A4 paired contrast is the registered gate intervention.',
        '', 'Efficiency: RUNTIME records actual per-model train wall time (including preparation where recorded), timed CPU inference, checkpoint load, process RSS and process lifetime peak RSS. Lifetime peak is not an isolated per-model peak. Inference ratios are single-pass descriptive timings including cold effects. Fixed epochs do not imply equal trainable parameters or FLOPs; FLOPs not measured.',
        '', 'Reviewer support: frozen reviewer matrix maps held-out comparisons to R2-C1, R2-C8 and R2-C9. These outputs directly address loss/parameterization/gate/receptive-field comparisons only. They do not answer reconstruction topology, visual fidelity, rollout stability or published-baseline superiority.',
        'Claims must follow each paired contrast and material/spacing strata, including adverse conditions. Original paper superiority claims are not confirmed by these 2D outputs; no 3D or external-method superiority is established.',
        '', 'Training and reconstruction entry access policies were not changed. Evaluation explicitly authorizes only registered test plane payloads and frozen models. OS_LEVEL_ISOLATION=NOT_ESTABLISHED; Python audit coverage does not prove arbitrary native file-access prevention.',
        '', 'No retraining, hyperparameter change, test-based checkpoint selection, 3D reconstruction or external comparator execution occurred.']
    (OUT/(PREFIX+'SUMMARY.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name.startswith(PREFIX) and not p.name.endswith('RESULT_MANIFEST.json'))
    manifest=dict(stage='REVISION_DESIGNA_HELDOUT_EVALUATION_R0',files={p.name:sha(p) for p in files})
    digest=dump('RESULT_MANIFEST',manifest)
    print(json.dumps(dict(status='PASS',models=len(all_metrics),elapsed_seconds=elapsed,manifest_sha256=digest,summaries=summaries,paired=changes)),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--freeze-protocol', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.freeze:
        freeze()
    if args.freeze_protocol:
        freeze_protocol()
    if args.self_test:
        metric_self_test()
        print('METRIC_SELF_TEST PASS')
    if args.evaluate:
        evaluate()


if __name__ == '__main__':
    main()
