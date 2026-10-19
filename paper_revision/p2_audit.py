"""P2 evidence aggregation, native access probes and frozen manifest audit."""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/revision_mtcomm_2026';P='R1_RECONSTITUTION_P2_'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):(OUT/(P+name+'.json')).write_text(json.dumps({'identity':'REVISION_REPLICATION',**obj},indent=2),encoding='utf-8')

def main():
    planes=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    assert len(planes)==2246 and len(set(r['plane_id'] for r in planes))==2246
    assert len(set(r['labels_path'] for r in planes))==2246
    data=read('R1_RECONSTITUTION_P1_DATASET_AUDIT.json');split=read('R1_RECONSTITUTION_P1_SPLIT_LEAKAGE_AUDIT.json')
    for r in planes:
        for kind in ['labels','support']:assert sha(OUT/r[kind+'_path'])==r[kind+'_sha256']
        if r['role'] in ['train','validation']:
            assert all(a<=c<d<=b for (a,b),(c,d) in zip(split['split_boxes_zyx'][r['role']],json.loads(r['bounds_zyx'])))
            assert 'test_truth' not in r['labels_path']
        else:
            roi=next(v for v in data['rois'] if v['material']==r['material'] and f"ROI{v['roi']}"==r['roi'])
            assert all(o<=a<b<=o+48 for o,(a,b) in zip(roi['origin_zyx'],json.loads(r['bounds_zyx'])))
    obs_checks=[]
    for roi in data['rois']:
        truth=np.fromfile(OUT/roi['truth_path'],np.uint8).reshape(48,48,48)
        for regime,obs in roi['observations'].items():
            p=OUT/obs['path'];assert sha(p)==obs['sha256']
            values=np.fromfile(p,np.uint8).reshape(truth.shape);mask=values!=255
            assert np.array_equal(values[mask],truth[mask]) and np.count_nonzero(~mask)==obs['updates_per_sweep']
            assert np.any(~mask)
            obs_checks.append(dict(material=roi['material'],roi=roi['roi'],regime=regime,observed=int(mask.sum()),hidden=int((~mask).sum()),hidden_payload_sentinel=255))
    # Read-only native probes: record exact Python identity; no ACL mutation.
    identity=subprocess.run(['whoami','/all'],capture_output=True,text=True,encoding='utf-8',errors='replace').stdout
    paths=[('TRAINING',OUT/next(r['labels_path'] for r in planes if r['role']=='train'),True),
           ('TRAINING',OUT/next(r['labels_path'] for r in planes if r['role']=='validation'),True),
           ('RECONSTRUCTION',OUT/data['rois'][0]['observations']['250']['path'],True),
           ('TRAINING',Path(data['sources'][0]['path']),False),
           ('TRAINING',OUT/data['rois'][0]['truth_path'],False),
           ('RECONSTRUCTION',OUT/data['rois'][0]['truth_path'],False),
           ('EVALUATION',OUT/data['rois'][0]['truth_path'],True)]
    tests=[]
    for role,path,allowed in paths:
        child='import sys\ntry:\n with open(sys.argv[1],"rb") as f:f.read(1)\n print("ALLOWED")\nexcept PermissionError:print("DENIED")'
        result=subprocess.run([sys.executable,'-c',child,str(path)],capture_output=True,text=True,timeout=20)
        tests.append(dict(role=role,path=str(path),expected='ALLOWED' if allowed else 'DENIED',actual=result.stdout.strip(),passes=result.stdout.strip()==('ALLOWED' if allowed else 'DENIED')))
    # Detect accessible second authoritative copy without scanning/releasing contents.
    copies=[]
    for p in Path('D:/$RECYCLE.BIN').glob('*/*/paper_level/drp167/*.raw'):
        if p.name.startswith(('WF2','GI')):
            try:
                with p.open('rb') as f:f.read(1)
                copies.append(dict(path=str(p),readable=True))
            except PermissionError:copies.append(dict(path=str(p),readable=False))
    write('ISOLATION_AUDIT',dict(G1='FAIL',actual_process_identity=identity,python_executable=sys.executable,
        tests=tests,alternative_source_copies=copies,original_acl=read(P+'ORIGINAL_ACL.json'),ACL_mutations=False,
        capability='Medium-integrity filtered token; net session returned system error5 access denied. No Docker/VM executable found; WSL not installed in P1 audit. Existing CodexSandbox accounts are not configured/authorized dedicated scientific roles; no credentials available to test exact identity.',
        rollback='No ACLs changed, no rollback needed. Before future elevated configuration export icacls /save for dedicated staging trees; restore with icacls /restore; preserve source ACLs.',
        non_destructive_path='Elevated local provisioning of dedicated training/reconstruction/evaluation identities or AppContainer/VM; only role-staged allowlisted inputs readable, explicit deny/read isolation of ALL host source/truth copies. Test direct/alternate/link/parent/copy paths under exact launcher. Do not reuse admin identity.',
        untested=['Role-specific write-only outputs','Evaluation write denial to model/config roots','Link/parent/copy bypass denial under new identity','Exact scientific executable restricted identity'],
        scientific_campaign_ready=False))
    with (OUT/(P+'ROLE_ACCESS_MATRIX.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['role','source','train','validation','observations','models','test_truth','output_write','enforcement'])
        w.writerows([['DATA_PREPARATION','read','write','write','write','deny','write','prep','NOT_SEPARATED'],['TRAINING','deny','read','read','deny','own_write','deny','training','NOT_ENFORCED'],['RECONSTRUCTION','deny','deny','deny','read','read','deny','reconstruction','NOT_ENFORCED'],['EVALUATION','deny','deny','deny','read','deny_write','read','evaluation_only','NOT_ENFORCED']])
    allrows=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv').open(encoding='utf-8-sig')))
    rows=[r for r in allrows if r['plan']=='C' and r['dedup_state_key']]
    assert len(rows)==284 and len(set(r['dedup_state_key'] for r in rows))==284
    keys=set();outrows=[]
    for r in rows:
        scientific=tuple(r[k] for k in ['material','roi','model','global_fit_um','observation_sha256','training_seed','reconstruction_seed','update_order','initialization'])
        assert scientific not in keys;keys.add(scientific)
        roi=next(x for x in data['rois'] if x['material']==r['material'] and str(x['roi'])==r['roi'])
        assert r['observation_sha256']==roi['observations'][r['local_obs_um']]['sha256']
        outrows.append(dict(trajectory_key=r['dedup_state_key'],reviewer_comments=r['reviewer_comments'],
            material=r['material'],roi=r['roi'],model=r['model'],global_fit_um=r['global_fit_um'],local_obs_um=r['local_obs_um'],
            training_seed=r['training_seed'],rollout_seed=r['reconstruction_seed'],sweeps=r['sweeps'],order=r['update_order'],initialization=r['initialization'],
            required_checkpoint=f"REVISION_REPLICATION/{r['material']}/G{r['global_fit_um']}/seed{r['training_seed']}/{r['model']}/12_orientation_phase_epoch50",checkpoint_exists=False,
            initial_state_reuse_key='|'.join(r[k] for k in ['material','roi','local_obs_um','reconstruction_seed','initialization']),
            resume='save1,3,5,10; same RNG/order; do not restart at3',duplicate=False,
            interpretation='correlated spatial subvolume of same Ketton specimen; no independent specimen replication'))
    with (OUT/(P+'TRAJECTORY_MATRIX_AUDIT.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
    write('INPUT_AUDIT',dict(registry_integrity='PASS',planes=len(planes),unique_ids=len(set(r['plane_id'] for r in planes)),
        registry_sha256=sha(OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv'),role_counts=dict(Counter(r['role'] for r in planes)),
        bounds_and_hashes_pass=True,truth_observation_checks=obs_checks,no_fit_arrays_created=True,
        source_code_import_audit='p1_dataset is preparation-only; p2_runner imports model classes and AST helper, no evaluation module or eager test-truth import. p2_baseline reads train rows only. No released production training/reconstruction entrypoint yet.',
        trajectories=len(rows),unique_trajectories=len(keys),models=dict(Counter(r['model'] for r in rows)),
        globals=sorted(set(r['global_fit_um'] for r in rows)),locals=sorted(set(r['local_obs_um'] for r in rows)),
        training_seeds=sorted(set(r['training_seed'] for r in rows)),rollout_seeds=sorted(set(r['reconstruction_seed'] for r in rows)),
        explanation='252core =6ROIs x2local masks x7models x3paired seeds;24density additions on ROI0/state (G25/G1000, B0/A3);8duplicate G250 core rows reused;16order/init additions. 252+16+16=284.',
        gaps='R2C3 same-rock/material scope;C5 onlyone10sweep pair and restricted sensitivity;C6 oneROI/state onepaired seed fullglobal cross;C7 comparator validation pending.',
        file_hashes={str(p.relative_to(ROOT)):sha(p) for p in (OUT).glob('R1_RECONSTITUTION_P1_*') if p.suffix in ['.json','.csv','.md']}))
    print(json.dumps({'registry':'PASS','trajectories':len(rows),'G1':'FAIL','alternative_copies':len(copies)}))

if __name__=='__main__':main()
