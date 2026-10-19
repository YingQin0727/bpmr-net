"""Actual R2 verification entrypoint, launched only by native trace controller."""
import ctypes
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
EVIDENCE=OUT/'TRAINING_2D_ACCESS_RELEASE'
DEBUG_MARKER=ctypes.windll.kernel32.OutputDebugStringW


def phase(name):
    DEBUG_MARKER('REVISION_PHASE:'+name)


def native_probe():
    """Read only a synthetic canary through Win32; never any scientific truth."""
    phase('native_probe_parent')
    canary=EVIDENCE/'NON_SCIENTIFIC_NATIVE_CANARY.bin'
    canary.write_bytes(b'NON_SCIENTIFIC_NATIVE_PROBE')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    kernel.CreateFileW.restype=ctypes.c_void_p
    kernel.ReadFile.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32),ctypes.c_void_p]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    h=kernel.CreateFileW(str(canary),0x80000000,7,None,3,0,None)
    assert h not in (None,ctypes.c_void_p(-1).value)
    buf=ctypes.create_string_buffer(8);read=ctypes.c_uint32()
    assert kernel.ReadFile(h,buf,8,ctypes.byref(read),None) and read.value==8
    kernel.CloseHandle(h)
    phase('native_prohibited_injection')
    forbidden=[OUT/'REVISION_REPLICATION_P1_dataset/test_truth/WF2_ROI0.truth.bin',
        OUT/'REVISION_REPLICATION_P1_dataset/test_truth/GI_ROI0.truth.bin',
        Path(r'D:\Paper2\paper2_stage02a_source_recovery\official_download\WF2_5um_1030x1033x3051_8b_segmented_corrected.raw'),
        Path(r'D:\Paper2\paper2_stage02a_source_recovery\official_download\GI_5um_1030x1033x3051_8b_segmented.raw')]
    backup=Path(r'D:\$RECYCLE.BIN\S-1-5-21-2452293266-1158813196-812636544-1001\$RLPIMC9\paper_level\drp167')
    forbidden += [backup/p.name for p in forbidden[2:]]
    for path in forbidden:
        ctypes.set_last_error(0)
        denied=kernel.CreateFileW(str(path),0x80000000,7,None,3,0,None)
        if denied not in (None,ctypes.c_void_p(-1).value):
            kernel.CloseHandle(denied)
            raise AssertionError('native forbidden open was not blocked')
        assert ctypes.get_last_error()==5
    phase('native_probe_child')
    # A controlled child verifies instrumentation, not isolation. Training later prohibits subprocesses.
    code="from pathlib import Path;assert Path("+repr(str(canary))+").read_bytes().startswith(b'NON_SCIENTIFIC')"
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def resume_test(x,y,variant,boundary):
    import copy
    import torch
    from bpmr_net.models import B0Network
    from paper_revision.models_revision import build_revision_bpmr
    from paper_revision.training_campaign_2d import optimizer_step,controlled_optimizer
    torch.manual_seed(1701)
    model=B0Network()
    if variant in ('A2','A3','A4','A5'):model=build_revision_bpmr(model,variant)
    def optimizer(m):return controlled_optimizer(m)
    opt=optimizer(model)
    assert opt.defaults['lr']==.001 and opt.defaults['betas']==(.9,.999)
    assert opt.defaults['eps']==1e-8 and opt.defaults['weight_decay']==0
    def step(m,o,first=False):
        bx,by=(x[boundary:],y[boundary:]) if first else (x[:boundary],y[:boundary])
        optimizer_step(m,o,bx,by,variant)
    # Only two updates: last minibatch of epoch1 and first minibatch of epoch2.
    step(model,opt)
    metadata=dict(kind='NON_SCIENTIFIC_TEST',epoch=1,next_epoch=2,next_batch=0,
                  primary_endpoint='epoch50',historical_checkpoint=False)
    buf=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),metadata=metadata),buf)
    path=EVIDENCE/('NON_SCIENTIFIC_TEST_resume_'+variant+'.pt')
    path.write_bytes(buf.getvalue())
    loaded=torch.load(io.BytesIO(path.read_bytes()),weights_only=True)
    clone=copy.deepcopy(model);clone.load_state_dict(loaded['model'])
    other=optimizer(clone);other.load_state_dict(loaded['optimizer'])
    step(model,opt,True);torch.set_rng_state(loaded['rng']);step(clone,other,True)
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,other.state_dict()['state'][k][n]) for k,s in opt.state_dict()['state'].items() for n,v in s.items() if torch.is_tensor(v))
    return dict(variant=variant,kind=metadata['kind'],resume_exact=True,steps=2,last_epoch1_batch_size=boundary,first_epoch2_batch_size=len(x)-boundary)


def main():
    native_probe()
    phase('framework_import')
    import torch
    import numpy as np
    torch.set_num_threads(1)
    from paper_revision.training_events import inventory,event_arrays,model_seed,primary_checkpoint
    from paper_revision.training_campaign_2d import run_fixed50
    from paper_revision.train_revision_2d import require_design_a
    phase('native_numpy_read_and_mapping')
    canary=EVIDENCE/'NON_SCIENTIFIC_NATIVE_CANARY.bin'
    assert np.fromfile(canary,dtype=np.uint8).tobytes()==canary.read_bytes()
    mapped=np.memmap(canary,dtype=np.uint8,mode='r');assert mapped[0]==ord('N');del mapped
    phase('native_numpy_prohibited_injection')
    prohibited=OUT/'REVISION_REPLICATION_P1_dataset/test_truth/WF2_ROI0.truth.bin'
    for reader in (lambda:np.fromfile(prohibited,dtype=np.uint8),lambda:np.memmap(prohibited,dtype=np.uint8,mode='r')):
        try:reader()
        except PermissionError:pass
        else:raise AssertionError('native NumPy prohibited read not denied')
    rows=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    matrix=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv').open(encoding='utf-8-sig')))
    cells=sorted({(r['material'],int(r['global_fit_um']),int(r['training_seed'])) for r in matrix if r['plan']=='C' and r['model'] in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')})
    assert len(cells)*12==120
    assert primary_checkpoint(EVIDENCE,50).name=='NON_SCIENTIFIC_TEST_epoch50.pt'
    for early in (1,2,49):
        try:primary_checkpoint(EVIDENCE,early)
        except ValueError:pass
        else:raise AssertionError('early primary endpoint allowed')
    from paper_revision.training_access import TrainingAccess
    torch.optim.Adam(torch.nn.Linear(1,1).parameters(),lr=.001)
    reads=[OUT/r[k+'_path'] for r in rows if r['role'] in ('train','validation') for k in ('labels','support')]
    reads+=list((ROOT/'paper_revision').glob('*.py'))+list((ROOT/'src/bpmr_net').glob('*.py'))
    writes=[EVIDENCE/('NON_SCIENTIFIC_TEST_resume_'+v+'.pt') for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')]
    guard=TrainingAccess(reads+writes,writes,[Path(sys.base_prefix)/'Lib',Path(sys.base_prefix)/'DLLs',ROOT/'.venv-revision311/Lib/site-packages'])
    guard.install();guard.phase='full_registered_event_selection_and_schedule'
    events=[];schedules={};resume=[];benchmark=None
    phase('complete_event_selection')
    for material in ('WF2','GI'):
        for axis in 'xyz':
            inv={role:inventory(OUT,rows,material,axis,role) for role in ('train','validation')}
            for m,spacing,seed in cells:
                if m!=material:continue
                for template in range(4):
                    derived=model_seed(material,spacing,axis,seed,template)
                    for role in ('train','validation'):
                        phase('events:'+f'{material}:G{spacing}:{axis}:{seed}:{template}:{role}')
                        x,y,record=event_arrays(OUT,inv[role],spacing,derived+(role=='validation'))
                        record.update(material=material,spacing_um=spacing,axis=axis,training_seed=seed,
                                      template=template,model_seed=derived,role=role)
                        # Each variant consumes this exact immutable event identity;
                        # variant construction is downstream of selection, never an input.
                        record['paired_variants']={v:{k:record[k] for k in ('event_sha256','ordinal_sha256','feature_sha256','target_sha256')}
                            for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')}
                        assert len({json.dumps(v,sort_keys=True) for v in record['paired_variants'].values()})==1
                        events.append(record)
                        if role=='train':
                            phase('full_50_epoch_index_traversal')
                            paired={v:run_fixed50(None,None,x,y,v,no_updates=True) for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')}
                            assert all(r==paired['A0'] for r in paired.values())
                            schedules[f'{material}/G{spacing}/{axis}/seed{seed}/template{template}']=paired
                        if not resume and role=='train':
                            phase('real_optimizer_epoch_boundary_resume')
                            boundary=len(x)%4096 or 4096
                            tx=torch.from_numpy(np.concatenate([x[-boundary:],x[:4096]]));ty=torch.from_numpy(np.concatenate([y[-boundary:],y[:4096]]))
                            resume=[resume_test(tx,ty,v,boundary) for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')]
                            from bpmr_net.models import B0Network
                            from paper_revision.training_campaign_2d import controlled_optimizer
                            from paper_revision.train_revision_2d import probability,loss_for
                            model=B0Network();opt=controlled_optimizer(model);times=[]
                            for _ in range(3):
                                started=time.perf_counter();opt.zero_grad()
                                loss=loss_for('B0_SOURCE',probability(model,torch.from_numpy(x[:4096])),torch.from_numpy(y[:4096]))
                                loss.backward();opt.step();times.append(time.perf_counter()-started)
                            benchmark=dict(kind='NON_SCIENTIFIC_TEST',batch_size=4096,threads=1,seconds=times)
                            del model,opt
                        del x,y
    assert len(events)==240
    guard.active=False
    assert all(r['allowed'] for r in guard.records)
    phase('reuse_python_guard_and_injections')
    from paper_revision.audit_training_2d import main as previous_audit
    previous_audit()
    phase('r2_evidence_save')
    result=dict(events=events,schedules=schedules,resume=resume,benchmark=benchmark,full_event_python_access=guard.records,source_models=120,adaptation_models=720,
                event_cap=100000,epochs=50,formal_training_started=False,environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,threads=torch.get_num_threads()),
                selection_rule='Full sorted per-unit 9x9 support; numpy default_rng(model_seed [+1 validation]) choice without replacement; sort selected ordinals.',
                initialization_rule='sha256(material/Gspacing/axis/seed/template) first 8 hex digits; exact existing dry-run G250 rule extended mechanically to frozen matrix cells.')
    (EVIDENCE/'r2_execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    phase('complete')


if __name__=='__main__':main()
