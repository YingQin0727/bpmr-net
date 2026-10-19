"""Executable negative isolation audit. Never prints label values.

Run under the intended downstream identity. A readable truth/source FAILS the
release gate. Readonly attributes prevent writes, not reads.
"""
import json
import subprocess
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026'

def main():
    audit=json.loads((OUT/'R1_RECONSTITUTION_P1_DATASET_AUDIT.json').read_text(encoding='utf-8'))
    paths=[('source',audit['sources'][0]['path']),('test_truth',str(OUT/audit['rois'][0]['truth_path']))]
    paths.append(('train_positive_control',str(next((OUT/'REVISION_REPLICATION_P1_dataset/train').glob('*.labels.bin')))))
    probes=[]
    for role,path in paths:
        code='import sys\ntry:\n f=open(sys.argv[1],"rb");f.read(1);f.close();print("READ_ALLOWED")\nexcept PermissionError: print("READ_DENIED")\n'
        result=subprocess.run([sys.executable,'-c',code,path],capture_output=True,text=True,timeout=20)
        probes.append(dict(role=role,path=path,result=result.stdout.strip(),returncode=result.returncode))
    blocked=all(p['result']=='READ_DENIED' for p in probes[:2]) and probes[2]['result']=='READ_ALLOWED'
    output=dict(identity='REVISION_REPLICATION',isolation_enforced=blocked,probes=probes,
        test='Native open from separate Python process, not role-loader API. Reads one byte solely as access probe; no byte value retained.',
        required='Distinct OS worker identity or restricted sandbox with allowlisted role mounts. Current same-user file permissions insufficient. No training/reconstruction release.')
    (OUT/'R1_RECONSTITUTION_P1_ISOLATION_TEST.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'isolation_enforced':blocked,'results':[p['result'] for p in probes]}))

if __name__=='__main__':main()
