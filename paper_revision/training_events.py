"""Full registered 2D event selection and fixed epoch50 batch schedule."""
import hashlib
import json
import numpy as np
from bpmr_net.features import encode_b0_features
from paper_revision.train_revision_2d import load_plane

CAP = 100000
BATCH = 4096
EPOCHS = 50


def model_seed(material, spacing, axis, seed, template):
    unit = f'{material}/G{spacing}/{axis}/seed{seed}/template{template}'
    return int(hashlib.sha256(unit.encode()).hexdigest()[:8],16)


def centers(support):
    # Full square erosion: all 81 support pixels must be valid, including center.
    h,w = support.shape
    valid = np.ones((h-8,w-8),dtype=bool)
    for r in range(9):
        for c in range(9): valid &= support[r:r+h-8,c:c+w-8].astype(bool)
    return np.argwhere(valid).astype(np.int32)+4


def inventory(root, rows, material, axis, role):
    records=[]
    for row in sorted((r for r in rows if (r['material'],r['normal_axis'],r['role'])==(material,axis,role)),key=lambda r:int(r['normal_index'])):
        labels,support=load_plane(root,row)
        coords=centers(support)
        records.append((row,coords))
    return records


def select(records, spacing, seed):
    selected=[(r,c) for r,c in records if str(spacing) in r['regimes_um'].split(';')]
    counts=np.array([len(c) for r,c in selected],dtype=np.int64)
    total=int(counts.sum())
    rng=np.random.default_rng(seed)
    ordinals=np.sort(rng.choice(total,size=CAP,replace=False)) if total>CAP else np.arange(total,dtype=np.int64)
    assert len(ordinals)==min(total,CAP)
    assert np.all(np.diff(ordinals)>0)
    return selected,counts,ordinals


def event_arrays(root, records, spacing, seed):
    selected,counts,ordinals=select(records,spacing,seed)
    # Independent repeat verifies exact deterministic cap selection.
    assert np.array_equal(ordinals,select(records,spacing,seed)[2])
    x=np.empty((len(ordinals),400),np.float32)
    y=np.empty(len(ordinals),np.int64)
    ids=hashlib.sha256();start=0;written=0
    for (row,coords),count in zip(selected,counts):
        lo=np.searchsorted(ordinals,start);hi=np.searchsorted(ordinals,start+count)
        if hi>lo:
            chosen=coords[ordinals[lo:hi]-start]
            labels,support=load_plane(root,row)
            rr,cc=chosen.T
            x[lo:hi]=encode_b0_features(labels,support,rr,cc)
            y[lo:hi]=labels[rr,cc]
            ids.update(json.dumps([row['plane_id'],row['labels_sha256'],row['support_sha256']],separators=(',',':')).encode())
            ids.update(chosen.astype('<i4').tobytes())
            # Independent scalar 400-slot check at both ends of every selected plane.
            from bpmr_net.features import B0_OFFSETS
            for offset in {0,len(chosen)-1}:
                r,c=chosen[offset]
                expected=np.zeros((80,5),dtype=np.float32)
                for i,(dr,dc) in enumerate(B0_OFFSETS):
                    expected[i,int(labels[r+dr,c+dc])]=1
                    expected[i,4]=1
                assert np.array_equal(x[lo+offset],expected.ravel())
                assert y[lo+offset]==labels[r,c]
            written+=hi-lo
        start+=int(count)
    assert written==len(ordinals) and (y<4).all()
    return x,y,dict(population=int(counts.sum()),count=len(ordinals),event_sha256=ids.hexdigest(),
                    ordinal_sha256=hashlib.sha256(ordinals.astype('<i8').tobytes()).hexdigest(),
                    feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    target_sha256=hashlib.sha256(y.tobytes()).hexdigest())


def batches(count, start_epoch=1):
    for epoch in range(start_epoch,EPOCHS+1):
        for begin in range(0,count,BATCH):
            yield epoch,np.arange(begin,min(begin+BATCH,count),dtype=np.int64)


def primary_checkpoint(directory, completed_epoch, scientific=False):
    if completed_epoch != EPOCHS:
        raise ValueError('primary checkpoint is fixed at epoch50')
    return directory / ('epoch50.pt' if scientific else 'NON_SCIENTIFIC_TEST_epoch50.pt')


def verify_schedule(count):
    variants={}
    for variant in ('B0_SOURCE','A0','A1','A2','A3','A4','A5'):
        epoch_records=[];current=0;seen=0;h=None
        for epoch,index in batches(count):
            if epoch!=current:
                if current:
                    assert seen==count
                    epoch_records.append(dict(epoch=current,count=seen,sha256=h.hexdigest()))
                current=epoch;seen=0;h=hashlib.sha256()
            assert np.array_equal(index,np.arange(seen,min(seen+BATCH,count)))
            h.update(index.astype('<i8').tobytes());seen+=len(index)
        assert current==50 and seen==count
        epoch_records.append(dict(epoch=current,count=seen,sha256=h.hexdigest()))
        assert len(epoch_records)==50 and len({e['sha256'] for e in epoch_records})==1
        variants[variant]=epoch_records
    assert all(v==variants['A0'] for v in variants.values())
    return dict(count=count,batch_size=BATCH,epochs=50,batches_per_epoch=(count+BATCH-1)//BATCH,
                variants=variants,primary_endpoint='epoch50',early_stopping=False)
