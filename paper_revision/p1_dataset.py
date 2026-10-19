"""REVISION_REPLICATION P1 deterministic preparation; no model execution."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
DATA=OUT/'REVISION_REPLICATION_P1_dataset'
P='R1_RECONSTITUTION_P1_'

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def dump(name,obj):
    (OUT/(P+name+'.json')).write_text(json.dumps({'identity':'REVISION_REPLICATION',**obj},indent=2),encoding='utf-8')

def immutable_bytes(path,payload):
    if path.exists():
        if path.read_bytes()!=payload:raise RuntimeError(f'Immutable revision payload conflict: {path}')
    else:path.write_bytes(payload)

def main():
    design=json.loads((OUT/'R1_RECONSTITUTION_P0_DATASET_DESIGN.json').read_text(encoding='utf-8'))
    source=json.loads((OUT/'R1_RECONSTITUTION_P0_SOURCE_AUDIT.json').read_text(encoding='utf-8'))
    exclusions=[(np.array(e['origin_zyx'])-128,np.array(e['origin_zyx'])+48+128) for e in design['historical_exclusions']]
    candidates=[]
    for z in range(1700,2972,200):
        for y in range(200,954,200):
            for x in range(200,951,200):
                lo=np.array([z,y,x]); hi=lo+48
                if any(np.all(lo<b) and np.all(hi>a) for a,b in exclusions):continue
                key=hashlib.sha256(f'Ketton_DRP167|{z},{y},{x}|REVISION_REPLICATION_P0'.encode()).hexdigest()
                candidates.append((key,lo.tolist()))
    rois=[v for _,v in sorted(candidates)[:3]]
    # Spatial footprints belong to split boxes; guards are tested analytically.
    splits=design['splits']; assert splits['train'][0][1]+128<=splits['validation'][0][0]
    assert splits['validation'][0][1]+128<=splits['test'][0][0]
    records=[]; audits=[]; roi_stats=[]
    code_sha=sha(Path(__file__))
    def save(material,role,axis,index,bounds,labels,support,source_sha,roi=''):
        labels=np.ascontiguousarray(labels,dtype=np.uint8)
        support=np.ascontiguousarray(support,dtype=np.uint8)
        labels=labels.copy();labels[support==0]=255
        ident=f'{material}_{role}_{roi}_{axis}_{index}'
        directory=DATA/role;directory.mkdir(parents=True,exist_ok=True)
        lp=directory/(ident+'.labels.bin');sp=directory/(ident+'.support.bin')
        immutable_bytes(lp,labels.tobytes());immutable_bytes(sp,support.tobytes())
        if role=='test_truth':lp.chmod(0o444);sp.chmod(0o444)
        records.append(dict(identity='REVISION_REPLICATION',plane_id=ident,material=material,role=role,roi=roi,
            normal_axis='zyx'[axis],normal_index=index,bounds_zyx=json.dumps(bounds),plane_axis_order=''.join('zyx'[a] for a in range(3) if a!=axis),
            model_plane_axis_order={'x':'yz','y':'zx','z':'xy'}['zyx'[axis]],
            model_transform='transpose' if axis in [0,2] else 'identity',
            shape=json.dumps(list(labels.shape)),dtype='uint8',phase_map='0=oil;1=brine;2=rock;3=gas;255=unsupported',pitch_um=5,
            regimes_um=';'.join(str(s) for s in [25,250,1000] if index%(s//5)==0),
            source_sha256=source_sha,labels_path=str(lp.relative_to(OUT)),labels_sha256=sha(lp),support_path=str(sp.relative_to(OUT)),support_sha256=sha(sp),
            valid_pixels=int(support.sum()),preparation_code_sha256=code_sha))
    for item in source['sources']:
        path=Path(item['path']); actual=sha(path);assert actual==item['authoritative_sha256']
        counts=np.zeros(256,dtype=np.int64)
        with path.open('rb') as f:
            while chunk:=f.read(32*1024*1024):counts+=np.bincount(np.frombuffer(chunk,np.uint8),minlength=256)
        assert counts.sum()==3051*1033*1030 and counts[5:].sum()==0
        audits.append(dict(material=item['material'],path=str(path),sha256=actual,shape_zyx=[3051,1033,1030],dtype='uint8',label_counts={str(i):int(counts[i]) for i in np.flatnonzero(counts)},pitch_um=5))
        raw=np.memmap(path,dtype=np.uint8,mode='r',shape=(3051,1033,1030))
        for role in ['train','validation']:
            box=splits[role]
            for axis in range(3):
                others=[a for a in range(3) if a!=axis]
                for index in range(((box[axis][0]+4)//5+1)*5,box[axis][1]-4,5):
                    slices=[None]*3;slices[axis]=index;bounds=[list(b) for b in box]
                    for a in others:
                        start=box[a][0]+8;slices[a]=slice(start,start+256);bounds[a]=[start,start+256]
                    bounds[axis]=[index,index+1]
                    raw_labels=np.array(raw[tuple(slices)])
                    labels=raw_labels-1
                    coords=np.indices(labels.shape)
                    # Only provider-defined phases 1..4 constitute segmentation support.
                    # Zero remains unassigned/excluded: do not call it gas, rock or air.
                    support=(raw_labels>=1)&(raw_labels<=4)
                    for lo,hi in exclusions:
                        inside=np.full(labels.shape,lo[axis]<=index<hi[axis])
                        for dim,a in enumerate(others):inside&=(coords[dim]+bounds[a][0]>=lo[a])&(coords[dim]+bounds[a][0]<hi[a])
                        support&=~inside
                    save(item['material'],role,axis,index,bounds,labels,support,actual)
        for rid,origin in enumerate(rois):
            raw_cube=np.array(raw[tuple(slice(a,a+48) for a in origin)])
            if np.any((raw_cube<1)|(raw_cube>4)):
                raise RuntimeError('Prespecified test ROI intersects unassigned domain; no replacement permitted')
            cube=raw_cube-1
            directory=DATA/'test_truth';directory.mkdir(parents=True,exist_ok=True)
            truth=directory/f'{item["material"]}_ROI{rid}.truth.bin';immutable_bytes(truth,cube.tobytes());truth.chmod(0o444)
            roi_stats.append(dict(material=item['material'],roi=rid,origin_zyx=origin,shape_zyx=[48]*3,truth_path=str(truth.relative_to(OUT)),truth_sha256=sha(truth),phase_counts=np.bincount(cube.ravel(),minlength=4).tolist()))
            # Register actual slices, and sparse observation-only volumes separately.
            for axis in range(3):
                for local in range(48):
                    index=origin[axis]+local
                    if index%5:continue
                    sl=[slice(None)]*3;sl[axis]=local;bounds=[[o,o+48] for o in origin];bounds[axis]=[index,index+1]
                    labels=cube[tuple(sl)]
                    save(item['material'],'test_truth',axis,index,bounds,labels,np.ones(labels.shape,bool),actual,f'ROI{rid}')
            coords=np.indices(cube.shape)+np.array(origin)[:,None,None,None]
            masks={}
            for spacing in [25,250,1000]:
                # Revision-only local lattice, centered in each ROI; never a historical recovery.
                # At 48^3, 250 and 1000 um have exactly the same three center planes.
                mask=np.any((coords-(np.array(origin)+24)[:,None,None,None])%(spacing//5)==0,axis=0)
                observed=np.full(cube.shape,255,np.uint8);observed[mask]=cube[mask]
                directory=DATA/'observations';directory.mkdir(parents=True,exist_ok=True)
                fp=directory/f'{item["material"]}_ROI{rid}_{spacing}um.bin';fp.write_bytes(observed.tobytes())
                masks[str(spacing)]=dict(path=str(fp.relative_to(OUT)),sha256=sha(fp),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),observed=int(mask.sum()),updates_per_sweep=int((~mask).sum()))
            roi_stats[-1]['observations']=masks
        del raw
    with (OUT/(P+'PLANE_REGISTRY.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    dump('DATASET_AUDIT',dict(status='REGISTERED_QUARANTINE_NOT_RELEASED',sources=audits,planes=len(records),rois=roi_stats,
        selection='P0 hash geometry rule instantiated; no outcome selection; diversity descriptive only',
        local_mask_rule='Explicit P1 amendment before fitting: local lattice centered at origin+24, steps5/50/200; 250/1000 share exact masks in 48-cube. Global training planes remain source-index modulo5/50/200. Deduplicate local masks, do not claim three distinct local densities.',
        support_semantics='Provider explicitly defines phases1..4. Raw0 is excluded unassigned segmentation domain; no physical meaning assigned. Zero never trains/evaluates as a fifth or remapped phase.',
        execution_geometry='Source/payload stored ZYX. Sampler state uses XYZ=np.transpose(source_crop,(2,1,0)). x-normal raw ZY tile transpose->YZ; y-normal ZX identity; z-normal YX transpose->XY. Plane registry declares both orders; mandatory adapter before event encoding.',
        training_tile_rule='One 256x256 tile per registered normal index anchored split lower+8; deliberately fixed cap, not full split plane; source bounds in registry.',
        test_readonly='Windows readonly attribute set; does NOT enforce read denial to same user.',
        release_blocker='OS process isolation pending; no downstream execution authorized.'))
    dump('SPLIT_LEAKAGE_AUDIT',dict(spatial_status='PASS_FOR_DIRECT_LABEL_OVERLAP',feature_radius=4,
        feature_support='400-vector consists of 80 offsets in [-4,4]^2 excluding center; 9x9 residual patch contains no labels beyond these slots even with dilation4.',
        split_boxes_zyx=splits,guard_voxels=128,split_paired_across_states=True,
        plane_cross_partition_overlap=0,neighborhood_cross_partition_overlap=0,
        event_rule='Every center footprint must lie inside tile and support; erode support by full square radius4 before event creation. No events created in P1.',
        historical_exclusion='Union of both historical ROIs expanded128 masked to sentinel255/support0. Events must exclude any invalid footprint.',
        periodic_rule='Periodic wrap allowed only inside reconstruction ROI; training must not wrap across split boundaries.',
        limitation='No direct overlap does not mean statistical independence of adjacent geology; both states same rock.',
        isolation_status='NOT_ENFORCED_BLOCK_RELEASE',executable_probe='paper_revision/p1_isolation_probe.py'))
    print(json.dumps({'planes':len(records),'rois':rois}))

if __name__=='__main__':main()
