"""Design A controlled A0-A5 campaign (revision-only, no test reads)."""
import argparse, csv, hashlib, io, json, math, time
from pathlib import Path
import numpy as np
import torch
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.training_campaign_2d import unit_cells, controlled_optimizer, run_fixed50
from paper_revision.training_events import inventory, event_arrays, model_seed
from paper_revision.train_revision_2d import probability, loss_for
from paper_revision.atomic_training_access import AtomicTrainingAccess

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/revision_mtcomm_2026'
OUT=BASE/'REVISION_REPLICATION_training/DesignA'; B0=OUT/'REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json'
P='REVISION_DESIGNA_A0A5_R0'; VARIANTS=('A0','A1','A2','A3','A4','A5')

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(p.read_bytes())
def save(path, model, opt, meta):
    s=io.BytesIO(); torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'rng_state':torch.get_rng_state(),'metadata':meta},s)
    data=s.getvalue(); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name('epoch50.pending'); temp.write_bytes(data); temp.replace(path)
    loaded=torch.load(io.BytesIO(path.read_bytes()),weights_only=True,map_location='cpu'); other=B0Network()
    if meta['variant'] in ('A2','A3','A4','A5'): other=build_revision_bpmr(other,meta['variant'])
    other.load_state_dict(loaded['state'],strict=True)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
    return sha_bytes(data)
def val_metrics(model,x,y,variant):
    with torch.no_grad():
        p=probability(model,torch.from_numpy(x)); yy=torch.from_numpy(y)
        mse=float(torch.nn.functional.mse_loss(p,torch.nn.functional.one_hot(yy,4).float()))
        nll=float(torch.nn.functional.nll_loss(p.clamp_min(1e-12).log(),yy))
    return {'mse':mse,'nll':nll}
def preflight():
    expected='175f0544cddb3806159defe0204f614157d079b2c83a4f0e9a8b0235254a3232'
    if sha_file(B0)!=expected: raise SystemExit('B0 manifest SHA256 mismatch')
    d=json.loads(B0.read_text()); rows=d['checkpoints']
    if len(rows)!=120 or len({r['unit'] for r in rows})!=120 or not all(r['strict_reload'] and not r.get('historical_checkpoint',True) for r in rows): raise SystemExit('B0 identity/integrity failure')
    release=json.loads((ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json').read_text(encoding='utf-8'))
    if release['status']['TRAINING_READY']!='YES': raise SystemExit('R2 release is not ready')
    for name,expected_sha in release['launch_binding'].items():
        if sha_file(ROOT/name)!=expected_sha: raise SystemExit('released code/protocol changed: '+name)
    registry=BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv'
    if sha_file(registry)!=release['input_identity']['registry_sha256']: raise SystemExit('frozen plane registry SHA256 changed')
    for r in rows:
        path=ROOT/r['path']
        if sha_file(path)!=r['sha256']: raise SystemExit('B0 checkpoint SHA256 mismatch: '+r['unit'])
        ck=torch.load(io.BytesIO(path.read_bytes()),weights_only=True,map_location='cpu')
        if ck['metadata'].get('unit')!=r['unit'] or ck['metadata'].get('kind')!='EPOCH50' or ck['metadata'].get('historical_checkpoint') is not False: raise SystemExit('B0 checkpoint identity mismatch: '+r['unit'])
        m,g,ax,seed,template=r['unit'].split('/'); derived=model_seed(m,int(g[1:]),ax,int(seed[4:]),int(template[8:]))
        if ck['metadata'].get('model_seed')!=derived: raise SystemExit('B0 training seed mismatch: '+r['unit'])
    return d,{r['unit']:r for r in rows}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--limit',type=int,default=720); ap.add_argument('--io-smoke',action='store_true'); a=ap.parse_args()
    global OUT, P
    if a.io_smoke:
        OUT=OUT/'ATOMIC_IO_R2_NON_SCIENTIFIC_TEST'; OUT.mkdir(exist_ok=True); P='NON_SCIENTIFIC_TEST'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    manifest,b0map=preflight(); units=[f'{m}/G{g}/{ax}/seed{s}/template{t}' for m,g,ax,s,t in unit_cells()]
    expected=[(u,v) for u in units for v in VARIANTS]
    if len(expected)!=720: raise SystemExit(f'expected 720, got {len(expected)}')
    if set(units)!=set(b0map): raise SystemExit('B0 registry identities do not exactly match frozen Plan C units')
    rows=list(csv.DictReader((BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    ledger=[]; metrics=[]; failures=[]
    prior=OUT.joinpath(P+'_CHECKPOINT_MANIFEST.json')
    if prior.exists():
        old=json.loads(prior.read_text(encoding='utf-8'))
        ledger=list(old.get('checkpoints',[])); failures=list(old.get('failures',[]))
    done_pairs={(r['unit'],r['variant']) for r in ledger}
    if len(done_pairs)!=len(ledger): raise SystemExit('duplicate completed task')
    for r in ledger:
        ckpath=ROOT/r['path']
        if sha_file(ckpath)!=r['checkpoint_sha256'] or len(r['epochs'])!=50: raise SystemExit('completed checkpoint integrity conflict')
        ck=torch.load(io.BytesIO(ckpath.read_bytes()),weights_only=True,map_location='cpu')
        mm=B0Network()
        if r['variant'] not in ('A0','A1'): mm=build_revision_bpmr(mm,r['variant'])
        mm.load_state_dict(ck['state'],strict=True)
        if not all(torch.isfinite(v).all() for v in mm.state_dict().values()): raise SystemExit('nonfinite completed checkpoint')
    metrics_path=OUT.joinpath(P+'_TRAINING_METRICS.csv')
    if ledger:
        metrics=[{'unit':r['unit'],'variant':r['variant'],'checkpoint_sha256':r['checkpoint_sha256'],'epoch':e['epoch'],'train_loss':e['train_loss'],'validation_mse':e['validation']['mse'],'validation_nll':e['validation']['nll']} for r in ledger for e in r['epochs']]
    elif metrics_path.exists():
        with metrics_path.open(encoding='utf-8-sig',newline='') as mf: metrics=list(csv.DictReader(mf))
    reads=[ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json',ROOT/'paper_revision/revision_protocol.json',BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv',B0,OUT.joinpath(P+'_CHECKPOINT_MANIFEST.json')]
    reads += [BASE/r[k+'_path'] for r in rows if r['role'] in ('train','validation') for k in ('labels','support')]
    reads += [ROOT/r['path'] for r in b0map.values()]
    launch=json.loads((ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json').read_text(encoding='utf-8'))['launch_binding']
    reads += [ROOT/name for name in launch]
    access_log=OUT.joinpath(P+'_PYTHON_ACCESS.jsonl')
    writes=[OUT.joinpath(P+'_TASK_LEDGER.csv'),OUT.joinpath(P+'_TRAINING_METRICS.csv'),OUT.joinpath(P+'_CHECKPOINT_MANIFEST.json'),access_log]
    writes += [p.with_suffix('.tmp') for p in writes[:3]]
    writes += [OUT/'A0A5'/u/v/'epoch50.pt' for u in units for v in VARIANTS]
    checkpoint_paths=[OUT/'A0A5'/u/v/'epoch50.pt' for u in units for v in VARIANTS]
    checkpoint_paths += [OUT/'A0A5'/units[0]/'A3'/'epoch50_reexecution.pt']
    pairs=[(p.with_suffix('.tmp'),p) for p in writes[:3]]+[(p.parent/'epoch50.pending',p) for p in checkpoint_paths]
    records=[OUT/'TASK_RECORDS'/u/(v+'.json') for u in units for v in VARIANTS]
    writes += [p.with_name('epoch50.pending') for p in checkpoint_paths]+records
    reads += checkpoint_paths
    guard=AtomicTrainingAccess(reads,writes,[Path(__import__('sys').base_prefix)/'Lib',Path(__import__('sys').base_prefix)/'DLLs',ROOT/'.venv-revision311/Lib/site-packages'],namespace=OUT,pairs=pairs,immutable=checkpoint_paths)
    guard.install(); guard.phase='controlled_A0A5_formal'
    for idx,(unit,variant) in enumerate(expected):
        if idx>=a.limit: break
        target=OUT/f'A0A5/{unit}/{variant}/epoch50.pt'
        if target.exists() and unit=='GI/G25/x/seed1701/template0' and variant=='A3':
            target=target.with_name('epoch50_reexecution.pt')
        started=time.time()
        if (unit,variant) in done_pairs:
            continue
        if target.exists(): raise SystemExit('refusing to overwrite existing A0-A5 checkpoint: '+str(target))
        m,g,ax,seed,template=unit.split('/'); g=int(g[1:]); seed=int(seed[4:]); template=int(template[8:])
        src=b0map[unit]; srcpath=ROOT/src['path']; source=torch.load(io.BytesIO(srcpath.read_bytes()),weights_only=True,map_location='cpu')
        guard.phase='registered_train_validation_'+unit
        source_model=B0Network(); source_model.load_state_dict(source['state'],strict=True)
        derived=model_seed(m,g,ax,seed,template); torch.manual_seed(derived)
        train=event_arrays(BASE,inventory(BASE,rows,m,ax,'train'),g,derived)
        valid=event_arrays(BASE,inventory(BASE,rows,m,ax,'validation'),g,derived+1)
        x,y,te=train; vx,vy,ve=valid
        if te!=src['events']['train'] or ve!=src['events']['validation']: raise SystemExit('frozen event identity mismatch: '+unit)
        guard.phase='variant_'+variant+'_epoch_training'
        model=source_model if variant in ('A0','A1') else build_revision_bpmr(source_model,variant)
        if variant in ('A2','A3','A4','A5'):
            assert torch.allclose(probability(model,torch.from_numpy(x)),probability(source_model,torch.from_numpy(x)),atol=1e-5,rtol=0)
        opt=controlled_optimizer(model); epochs=[]
        for epoch in range(1,3 if a.io_smoke else 51):
            order=np.arange(min(len(x),4096) if a.io_smoke else len(x),dtype=np.int64); total=0.; steps=0
            for begin in range(0,len(order),4096):
                ix=order[begin:min(begin+4096,len(x))]; opt.zero_grad(); loss=loss_for(variant,probability(model,torch.from_numpy(x[ix])),torch.from_numpy(y[ix]))
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); opt.step(); total+=float(loss.detach()); steps+=1
            vm=val_metrics(model,vx[:4096] if a.io_smoke else vx,vy[:4096] if a.io_smoke else vy,variant); epochs.append({'epoch':epoch,'train_loss':total/steps,'validation':vm,'optimizer_steps':steps})
        meta={'identity':'REVISION_REPLICATION','design':'A','variant':variant,'unit':unit,'kind':'NON_SCIENTIFIC_TEST' if a.io_smoke else 'EPOCH50','epochs_completed':len(epochs),'historical_checkpoint':False,'source_checkpoint_sha256':src['sha256'],'source_b0_unit':unit,'model_seed':derived,'train_events':te,'validation_events':ve,'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        cksha=save(target,model,opt,meta); elapsed=time.time()-started
        rec={'unit':unit,'variant':variant,'path':str(target.relative_to(ROOT)),'checkpoint_sha256':cksha,'source_checkpoint_sha256':src['sha256'],'strict_reload':True,'finite_parameters':all(torch.isfinite(p).all().item() for p in model.parameters()),'epochs':epochs,'elapsed_seconds':elapsed,'train_events':te,'validation_events':ve,'trainable_parameter_count':meta['trainable_parameter_count'],'a4_gate_frozen':variant=='A4','recovery_reason':'A3_REEXECUTION_AFTER_ATOMIC_IO_ABORT' if (unit=='GI/G25/x/seed1701/template0' and variant=='A3') else None,'orphan_checkpoint_sha256':'265a37688949f7c6a9b788aea60d0f29678765c72df503bf66578391d00de342' if (unit=='GI/G25/x/seed1701/template0' and variant=='A3') else None}
        if not rec['finite_parameters']: raise RuntimeError('nonfinite parameters')
        record_path=OUT/'TASK_RECORDS'/unit/(variant+'.json'); record_path.parent.mkdir(parents=True,exist_ok=True)
        if record_path.exists(): raise RuntimeError('refusing duplicate task record')
        record_path.write_text(json.dumps(rec,indent=2),encoding='utf-8')
        ledger.append(rec)
        metrics.extend({'unit':unit,'variant':variant,'checkpoint_sha256':cksha,'epoch':e['epoch'],'train_loss':e['train_loss'],'validation_mse':e['validation']['mse'],'validation_nll':e['validation']['nll']} for e in epochs)
        with access_log.open('a',encoding='utf-8') as af: af.write(json.dumps({'unit':unit,'variant':variant,'records':guard.records},separators=(',',':'))+'\n')
        guard.records.clear()
        ledger_path=OUT.joinpath(P+'_TASK_LEDGER.csv')
        ledger_path.with_suffix('.tmp').write_text('unit,variant,path,checkpoint_sha256,source_checkpoint_sha256,strict_reload,finite_parameters,trainable_parameter_count,elapsed_seconds,a4_gate_frozen\n'+'\n'.join(','.join(str(rec[k]) for k in ['unit','variant','path','checkpoint_sha256','source_checkpoint_sha256','strict_reload','finite_parameters','trainable_parameter_count','elapsed_seconds','a4_gate_frozen']) for rec in ledger)+'\n',encoding='utf-8'); ledger_path.with_suffix('.tmp').replace(ledger_path)
        metrics_path.with_suffix('.tmp').write_text('unit,variant,checkpoint_sha256,epoch,train_loss,validation_mse,validation_nll\n'+'\n'.join(','.join(str(x[k]) for k in x) for x in metrics)+'\n',encoding='utf-8'); metrics_path.with_suffix('.tmp').replace(metrics_path)
        mp=OUT.joinpath(P+'_CHECKPOINT_MANIFEST.json'); mp.with_suffix('.tmp').write_text(json.dumps({'stage':P,'expected_tasks':720,'checkpoints':ledger,'failures':failures},indent=2),encoding='utf-8'); mp.with_suffix('.tmp').replace(mp)
        print(f'completed {len(ledger)}/{a.limit} {unit} {variant}',flush=True)
    OUT.joinpath(P+'_CHECKPOINT_MANIFEST.json').write_text(json.dumps({'stage':P,'expected_tasks':720,'checkpoints':ledger,'failures':failures},indent=2),encoding='utf-8')
    print(json.dumps({'completed':len(ledger),'failures':failures}))
if __name__=='__main__': main()

