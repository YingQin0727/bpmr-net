"""Preparation-side full payload integrity and spatial audit; no fitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion

OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026'
P='R1_RECONSTITUTION_P1_'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    rows=list(csv.DictReader((OUT/(P+'PLANE_REGISTRY.csv')).open(encoding='utf-8-sig')))
    audit=json.loads((OUT/(P+'DATASET_AUDIT.json')).read_text(encoding='utf-8'))
    leak=json.loads((OUT/(P+'SPLIT_LEAKAGE_AUDIT.json')).read_text(encoding='utf-8'))
    valid_centers={};payload_bytes=0
    for row in rows:
        shape=json.loads(row['shape']);bounds=json.loads(row['bounds_zyx']);axis='zyx'.index(row['normal_axis'])
        lp=OUT/row['labels_path'];sp=OUT/row['support_path']
        assert sha(lp)==row['labels_sha256'] and sha(sp)==row['support_sha256']
        labels=np.fromfile(lp,np.uint8).reshape(shape);support=np.fromfile(sp,np.uint8).reshape(shape).astype(bool)
        assert np.all(labels[support]<=3) and np.all(labels[~support]==255)
        assert row['plane_axis_order']==''.join(a for a in 'zyx' if a!=row['normal_axis'])
        assert bounds[axis]==[int(row['normal_index']),int(row['normal_index'])+1]
        if row['role'] in ['train','validation']:
            split=leak['split_boxes_zyx'][row['role']]
            assert all(a<=c<=d<=b for (a,b),(c,d) in zip(split,bounds))
            count=int(binary_erosion(support,structure=np.ones((9,9),bool),border_value=0).sum())
            key=row['material']+'|'+row['role']
            valid_centers[key]=valid_centers.get(key,0)+count
        payload_bytes+=lp.stat().st_size+sp.stat().st_size
    for roi in audit['rois']:
        assert sha(OUT/roi['truth_path'])==roi['truth_sha256']
        observations=roi['observations']
        assert observations['250']['mask_sha256']==observations['1000']['mask_sha256']
        assert observations['250']['sha256']==observations['1000']['sha256']
        for obs in observations.values():assert sha(OUT/obs['path'])==obs['sha256']
    # Coordinate fixture demonstrates storage ZYX -> model XYZ plane conventions.
    source=np.indices((11,12,13)).sum(axis=0).astype(np.uint8)%4
    state=np.transpose(source,(2,1,0));z,y,x=5,6,7
    assert np.array_equal(source[:,:,x].T,state[x,:,:])
    assert np.array_equal(source[:,y,:],state[:,y,:].T)
    assert np.array_equal(source[z,:,:].T,state[:,:,z])
    audit.update(integrity_all_registered_payloads='PASS',payload_bytes_registered=payload_bytes,
                 full_square_radius4_valid_center_counts=valid_centers,geometry_adapter_fixture='PASS',
                 generator_sha256=sha(Path(__file__).with_name('p1_dataset.py')))
    leak.update(payload_bounds_checked=len(rows),hash_and_domain_check='PASS',geometry_adapter_fixture='PASS',
                full_support_erosion_counts=valid_centers,
                isolation_status='FAILED_SAME_USER_CAN_READ_RAW_AND_TEST_TRUTH',dataset_release=False)
    (OUT/(P+'DATASET_AUDIT.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (OUT/(P+'SPLIT_LEAKAGE_AUDIT.json')).write_text(json.dumps(leak,indent=2),encoding='utf-8')
    print(json.dumps(dict(planes=len(rows),integrity='PASS',release=False,valid_centers=valid_centers)))

if __name__=='__main__':main()
