"""Classify native evidence and fail closed on incomplete R2 verification."""
import collections
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
EV=OUT/'TRAINING_2D_ACCESS_RELEASE'


def key(path):
    if path.startswith('\\??\\'):path=path[4:]
    return os.path.normcase(path).rstrip('\\')


def main():
    controller=json.loads((EV/'r2_trace_controller.json').read_text())
    execution=json.loads((EV/'r2_execution.json').read_text())
    old=json.loads((EV/'release.json').read_text())
    records=[json.loads(line) for line in (EV/'r2_native.jsonl').open(encoding='utf-8')]
    planes=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    allowed={key(str(OUT/r[k+'_path'])) for r in planes if r['role'] in ('train','validation') for k in ('labels','support')}
    allowed.update(key(str(ROOT/p)) for p in old['configs'])
    allowed.update(key(str(OUT/n)) for n in ('R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv','R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv'))
    runtime=[key(str(ROOT/'.venv-revision311/Lib/site-packages')),key(r'C:\Users\admin\.codex\runtimes\uv-python\cpython-3.11-windows-x86_64-none'),key(r'C:\Windows'),key(r'C:\Program Files')]
    code={key(str(p)) for folder in (ROOT/'paper_revision',ROOT/'src/bpmr_net') for p in folder.glob('*.py')}
    # Resolve DOS-device aliases rather than blanket-whitelisting all Device paths.
    aliases={}
    for drive in ('C:','D:'):
        buffer=ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.QueryDosDeviceW(drive,buffer,1024):aliases[key(buffer.value)]=key(drive)
    unknown=[];forbidden=[];categories=collections.Counter();unresolved=[];temporary_created=set()
    sequences=collections.defaultdict(list)
    for r in records:
        if 'seq' in r:sequences[r['pid']].append(r['seq'])
        if 'api' not in r or r['api'] in ('phase','trace_ready'):continue
        path=r.get('path','');p=key(path)
        for alias,drive in aliases.items():
            if p.startswith(alias+'\\'):p=drive+p[len(alias):];break
        if r.get('blocked'):categories['injected_denied']+=1;continue
        if r.get('status',-1)<0:categories['unsuccessful_operation']+=1;continue
        if '<unresolved' in path:unresolved.append(r);continue
        directory=bool(r.get('options',0)&1) or path.endswith('\\')
        content=r['api'] in ('NtReadFile','NtWriteFile','NtCreateSection','NtMapViewOfSection') or (r['api'] in ('NtCreateFile','NtOpenFile') and not directory and r.get('access',0)&0xc0000003)
        if any(s in p for s in ('\\test_truth\\','\\official_download\\','\\$recycle.bin\\')) or p.endswith('.raw'):
            if content:forbidden.append(r)
        temp=key(str(Path(os.environ['TEMP'])))+'\\'
        if p.startswith(temp) and r['api']=='NtCreateFile' and r.get('disposition')==2:
            temporary_created.add((r['pid'],p))  # FILE_CREATE: exclusive new runtime scratch file, never pre-existing data
        if p in allowed:
            category='registered_2d_or_config'
            if r['api']=='NtWriteFile' or (r['api'] in ('NtCreateFile','NtOpenFile') and r.get('access',0)&0x40000002):forbidden.append(r)
        elif p in code or ('\\__pycache__\\' in p and p.endswith('.pyc') and p.startswith(key(str(ROOT)))):category='model_and_launcher_code'
        elif (p.startswith('\\knowndlls\\') and p.endswith('.dll')) or any(p==root or p.startswith(root+'\\') for root in runtime):category='python_dll_runtime'
        elif p.startswith(key(str(EV))+'\\'):category='isolated_test_output'
        elif p.startswith(('\\device\\anonymouspipe','\\device\\namedpipe','\\device\\afd\\','\\device\\characterdevice')) or p in ('nul','\\device\\cng','\\device\\ksecdd') or 'mountpointmanager' in p:category='windows_device'
        elif not content:category='metadata_only'
        elif (r['pid'],p) in temporary_created:category='new_runtime_scratch_file'
        else:category='unclassified_content';unknown.append(r)
        categories[category]+=1
    blocks=[r for r in records if r.get('blocked')]
    native_blocks=[r for r in blocks if r['phase']=='native_prohibited_injection']
    numpy_blocks=[r for r in blocks if r['phase']=='native_numpy_prohibited_injection']
    execution_written=any(r.get('api')=='NtWriteFile' and r.get('phase')=='r2_evidence_save' and r.get('status',-1)>=0 and r.get('path','').lower().endswith('r2_execution.json') and r['pid']==controller['root_pid'] for r in records)
    trace_complete=any(r.get('api')=='phase' and r.get('path')=='complete' and r['pid']==controller['root_pid'] for r in records)
    sequence_complete=all(values==list(range(len(values))) for values in sequences.values())
    native_pass=execution_written and trace_complete and sequence_complete and not (controller['errors'] or forbidden or unknown or unresolved) and len(native_blocks)==6 and len(numpy_blocks)>=2 and controller['attached_processes']>=3
    event_pass=len(execution['events'])==240 and all(r['count']==min(r['population'],100000) for r in execution['events'])
    pair_pass=event_pass and all(len(r['paired_variants'])==7 and len({json.dumps(v,sort_keys=True) for v in r['paired_variants'].values()})==1 for r in execution['events'])
    schedule_pass=len(execution['schedules'])==120
    for unit,variants in execution['schedules'].items():
        schedule_pass &= len(variants)==7 and all(v==variants['A0'] for v in variants.values())
        schedule_pass &= all(len(v)==50 and [e['epoch'] for e in v]==list(range(1,51)) and len({(e['count'],e['index_sha256']) for e in v})==1 for v in variants.values())
    guard_pass=bool(execution['full_event_python_access']) and all(r['allowed'] for r in execution['full_event_python_access'])
    smoke_pass=len(execution['resume'])==7 and all(r['last_epoch1_batch_size']==1696 and r['first_epoch2_batch_size']==4096 for r in execution['resume']) and all(r['resume_exact'] for r in execution['resume']) and old['status']['TRAINING_DRY_RUN']=='PASS'
    injection_pass=len(native_blocks)==6 and len(numpy_blocks)>=2 and old['status']['PROHIBITED_READ_INJECTION_TEST']=='PASS'
    statuses=dict(TRAINING_INPUT_AUDIT=old['status']['TRAINING_INPUT_AUDIT'],TRAINING_ACCESS_AUDIT='PASS' if native_pass and guard_pass else 'FAIL',
        PROHIBITED_READ_INJECTION_TEST='PASS' if injection_pass else 'FAIL',EVENT_CAP_AUDIT='PASS' if event_pass else 'FAIL',
        PAIRED_EVENT_AUDIT='PASS' if pair_pass else 'FAIL',EPOCH50_SCHEDULE_AUDIT='PASS' if schedule_pass else 'FAIL',
        DESIGN_A_COMPATIBILITY=old['status']['DESIGN_A_COMPATIBILITY'],TRAINING_DRY_RUN='PASS' if smoke_pass else 'FAIL')
    ready=all(v=='PASS' for v in statuses.values())
    statuses.update(SCIENTIFIC_DATA_ISOLATION='PASS_WITH_DOCUMENTED_LIMITATIONS' if ready else 'FAIL',
        OS_LEVEL_ISOLATION='NOT_ESTABLISHED',TRAINING_READY='YES' if ready else 'NO',REVISION_TRAINING_2D_RELEASE_R2='PASS' if ready else 'PARTIAL')
    limitations=[
        'User-mode ntdll API interception, not kernel ETW or OS access isolation. Direct syscalls and pre-attachment loader IO are not exhaustively covered.',
        'Actual loaded NumPy/PyTorch file APIs and observed child processes were traced. This does not certify arbitrary future native extensions or malicious code.',
        'Known forbidden native paths are denied; Python whitelist denies arbitrary unregistered inputs. Runtime code is trusted and environment substitution is not isolated.',
        'Full 50-epoch traversal performed without parameter updates; only bounded optimizer and resume checks were run. No scientific checkpoint exists.',
        'Four frozen template identities are replicated seed/event identities; no new template-specific masking rule is introduced.',
        'Release applies to the bound revision 2D kernels and shared scheduler, not legacy training, reconstruction or evaluation entrypoints.'
    ]
    binding_files=[ROOT/'paper_revision'/n for n in ('training_campaign_2d.py','training_events.py','train_revision_2d.py','models_revision.py','training_access.py','trace_training_r2.py','native_file_trace.js','training_release_r2.py')]
    binding_files+=list((ROOT/'src/bpmr_net').glob('*.py'))+[ROOT/p for p in old['configs']]+[OUT/'R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv',ROOT/'.venv-revision311/pyvenv.cfg']
    binding={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in binding_files}
    batches_total=sum((r['count']+4095)//4096*50 for r in execution['events'] if r['role']=='train')
    timing=execution['benchmark'];timing['source_total_optimizer_steps']=batches_total
    timing['source_cpu_hours_point_estimate']=batches_total*sorted(timing['seconds'])[1]/3600
    result=dict(stage='REVISION_TRAINING_2D_RELEASE_R2',status=statuses,input_identity=dict(registry_sha256=old['registry_sha256'],planes=old['plane_count'],roles=old['role_counts'],verified_payloads=old['verified_training_payloads']),full_event_python_access_count=len(execution['full_event_python_access']),observed_native_events=len(records),
        native_api_counts=dict(collections.Counter(r.get('api','controller') for r in records)),categories=dict(categories),
        trace_complete=trace_complete,sequence_complete=sequence_complete,processes=controller,
        forbidden_successful_reads=forbidden,unknown_content=unknown,unresolved_handles=unresolved,
        native_injections=native_blocks+numpy_blocks,python_injections=len(old['injection']),
        event_manifest=execution['events'],schedule_units=len(execution['schedules']),resume=execution['resume'],
        source_model_count=120,adaptation_model_count=720,formal_training_started=False,
        environment=execution['environment'],observed_native_components=sorted({r.get('path','') for r in records if r.get('path','').lower().endswith(('.dll','.pyd')) and r.get('status',-1)>=0}),benchmark=timing,launch_binding=binding,
        formal_entry='D:/python.exe -B -m paper_revision.trace_training_r2 --source-b0',
        checkpoint_pattern='outputs/revision_mtcomm_2026/REVISION_REPLICATION_training/DesignA/{material}/G{spacing}/{axis}/seed{seed}/template{template}/epoch50.pt',
        limitations=limitations,evidence_sha256={n:hashlib.sha256((EV/n).read_bytes()).hexdigest() for n in ('r2_native.jsonl','r2_execution.json','r2_trace_controller.json','access.json','dry_run.json','release.json')})
    (ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(statuses,indent=2));print('Unknown:',len(unknown),'Unresolved:',len(unresolved),'Native:',len(native_blocks),len(numpy_blocks))


if __name__=='__main__':main()
