"""Shared Design A batch executor and frozen Plan C unit enumeration.

This module has no import-time data reads. The fixed scheduler is also used by
the parameter-free audit. It never selects checkpoints using test data.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from bpmr_net.models import B0Network
from paper_revision.train_revision_2d import probability,loss_for
from paper_revision.training_events import batches,model_seed,inventory,event_arrays,primary_checkpoint

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
TRAIN=OUT/'REVISION_REPLICATION_training/DesignA'


def controlled_optimizer(model):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
        lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)


def optimizer_step(model,optimizer,x,y,variant):
    optimizer.zero_grad()
    bx=torch.from_numpy(x) if isinstance(x,np.ndarray) else x
    by=torch.from_numpy(y) if isinstance(y,np.ndarray) else y
    loss=loss_for(variant,probability(model,bx),by)
    if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
    loss.backward();optimizer.step()
    return float(loss.detach())


def run_fixed50(model, optimizer, x, y, variant, *, no_updates=False, validation=None):
    """One actual batch iterator for audit and training; no early-stop parameter.

    Parameter-free mode traverses the full 50 epochs and hashes actual indices.
    It skips forward/backward/optimizer and never saves a scientific checkpoint.
    """
    count=len(x);current=0;seen=0;h=None;epochs=[];batch_count=0
    for epoch,index in batches(count):
        if epoch!=current:
            if current:
                if seen!=count:raise RuntimeError('incomplete epoch')
                epochs.append(dict(epoch=current,count=seen,batches=batch_count,index_sha256=h.hexdigest()))
            current=epoch;seen=0;batch_count=0;h=hashlib.sha256()
        if not np.array_equal(index,np.arange(seen,min(seen+4096,count))):raise RuntimeError('batch order changed')
        h.update(index.astype('<i8').tobytes());seen+=len(index);batch_count+=1
        if not no_updates:
            optimizer_step(model,optimizer,x[index],y[index],variant)
    if current!=50 or seen!=count:raise RuntimeError('primary endpoint is not epoch50')
    epochs.append(dict(epoch=current,count=seen,batches=batch_count,index_sha256=h.hexdigest()))
    return epochs


def unit_cells():
    path=OUT/'R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv'
    with path.open(encoding='utf-8-sig') as f:matrix=list(csv.DictReader(f))
    cells=sorted({(r['material'],int(r['global_fit_um']),int(r['training_seed'])) for r in matrix
        if r['plan']=='C' and r['model'] in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')})
    return [(m,g,a,s,t) for m,g,s in cells for a in 'xyz' for t in range(4)]


def train_source_b0():
    """Next-stage source campaign, only via explicit native-traced controller flag."""
    import sys
    import sysconfig
    from paper_revision.training_access import TrainingAccess
    release_path=ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json'
    release=json.loads(release_path.read_bytes())
    if release['status']['TRAINING_READY']!='YES':raise PermissionError('R2 did not release training')
    for name,expected in release['launch_binding'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise PermissionError('changed release binding: '+name)
    with (OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    units=unit_cells()
    targets=[primary_checkpoint(TRAIN/f'{m}/G{g}/{a}/seed{s}/template{t}',50,scientific=True) for m,g,a,s,t in units]
    if any(p.exists() for p in targets):raise FileExistsError('refusing to replace any scientific checkpoint')
    reads=[OUT/r[k+'_path'] for r in rows if r['role'] in ('train','validation') for k in ('labels','support')]
    reads += [ROOT/p for p in release['launch_binding']]+[release_path]+targets
    writes=targets+[TRAIN/'source_B0_completion.json']
    reads+=writes
    # Warm the identical lazy Adam imports covered by the R2 bootstrap trace.
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    controlled_optimizer(torch.nn.Linear(1,1))
    guard=TrainingAccess(reads,writes,[Path(sys.base_prefix)/'Lib',Path(sys.base_prefix)/'DLLs',ROOT/'.venv-revision311/Lib/site-packages'])
    guard.install();guard.phase='formal_source_B0'
    complete=[]
    for (m,g,a,s,t),target in zip(units,targets):
        derived=model_seed(m,g,a,s,t)
        x,y,identity=event_arrays(OUT,inventory(OUT,rows,m,a,'train'),g,derived)
        vx,vy,valid=event_arrays(OUT,inventory(OUT,rows,m,a,'validation'),g,derived+1)
        for role,actual in [('train',identity),('validation',valid)]:
            frozen=next(r for r in release['event_manifest'] if (r['material'],r['spacing_um'],r['axis'],r['training_seed'],r['template'],r['role'])==(m,g,a,s,t,role))
            for k in ('event_sha256','ordinal_sha256','feature_sha256','target_sha256'):
                if actual[k]!=frozen[k]:raise PermissionError('event identity changed')
        torch.manual_seed(derived);model=B0Network();optimizer=controlled_optimizer(model)
        epochs=run_fixed50(model,optimizer,x,y,'B0_SOURCE')
        target.parent.mkdir(parents=True,exist_ok=True)
        metadata=dict(identity='REVISION_REPLICATION',design='A',variant='B0_SOURCE',
            unit=f'{m}/G{g}/{a}/seed{s}/template{t}',kind='EPOCH50',epochs_completed=50,
            historical_checkpoint=False,model_seed=derived,train_events=identity,validation_events=valid)
        stream=io.BytesIO();torch.save(dict(state=model.state_dict(),metadata=metadata,optimizer=optimizer.state_dict()),stream)
        payload=stream.getvalue();target.write_bytes(payload)
        loaded=torch.load(io.BytesIO(target.read_bytes()),weights_only=True)
        if not all(torch.equal(v,loaded['state'][k]) for k,v in model.state_dict().items()):raise RuntimeError('checkpoint verification failed')
        complete.append(dict(unit=metadata['unit'],path=str(target),sha256=hashlib.sha256(payload).hexdigest(),epochs=epochs))
    (TRAIN/'source_B0_completion.json').write_text(json.dumps(complete,indent=2),encoding='utf-8')


if __name__=='__main__':
    raise SystemExit('Use the native-traced controller with explicit --source-b0 after R2 release.')
