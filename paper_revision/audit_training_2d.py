"""Bounded, actual-data training-input release check; never a formal campaign.

Run with .venv-revision311/Scripts/python.exe -B -m paper_revision.audit_training_2d
No test-truth payload is opened, even for hashing. Test hashes are metadata only.
"""
import csv
import hashlib
import io
import json
import os
import sys
import sysconfig
from collections import Counter
from pathlib import Path

from paper_revision.training_access import TrainingAccess, canonical

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/revision_mtcomm_2026'
REPORT = OUT / 'TRAINING_2D_ACCESS_RELEASE'
REGISTRY = OUT / 'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    REPORT.mkdir(exist_ok=True)
    checkpoints = REPORT / 'dry_checkpoints'
    checkpoints.mkdir(exist_ok=True)
    bootstrap = []
    recording = [True]
    def observe(event, args):
        if recording[0] and event == 'open':
            bootstrap.append(dict(path=str(args[0]), mode=str(args[1]), flags=args[2]))
    sys.addaudithook(observe)
    # Framework bootstrap is separately recorded; it cannot establish native IO coverage.
    import torch
    import numpy as np
    import copy
    torch.set_num_threads(1)
    # Materialize Adam's lazy runtime imports before enforcing runtime capability denial.
    warm = torch.nn.Linear(1, 1)
    torch.optim.Adam(warm.parameters(), lr=.001)
    recording[0] = False

    registry_bytes = REGISTRY.read_bytes()
    rows = list(csv.DictReader(io.StringIO(registry_bytes.decode('utf-8-sig'))))
    assert len(rows) == 2246 and Counter(r['role'] for r in rows) == {'train': 1150, 'validation': 916, 'test_truth': 180}
    assert len({r['plane_id'] for r in rows}) == 2246
    config_paths = [REGISTRY, ROOT/'paper_revision/revision_protocol.json',
                    OUT/'R1_RECONSTITUTION_P2_TRAINING_CONTRACT.json',
                    OUT/'R1_RECONSTITUTION_P0_DATASET_DESIGN.json',
                    OUT/'R1_RECONSTITUTION_P1_DATASET_AUDIT.json',
                    OUT/'R1_RECONSTITUTION_P1_EXPERIMENT_BUDGET.json',
                    OUT/'R1_RECONSTITUTION_P2_RESOURCE_BUDGET.json']
    reads = list(config_paths)
    for row in rows:
        assert row['identity'] == 'REVISION_REPLICATION'
        for kind in ('labels', 'support'):
            assert len(row[kind+'_sha256']) == 64
            int(row[kind+'_sha256'], 16)
            if row['role'] in ('train', 'validation'):
                path = OUT / row[kind+'_path']
                assert path.resolve().is_relative_to((OUT/'REVISION_REPLICATION_P1_dataset'/row['role']).resolve())
                assert not path.is_symlink() and path.stat().st_nlink == 1
                reads.append(path)
    code_paths = [ROOT/'paper_revision'/n for n in ('train_revision_2d.py', 'models_revision.py', 'training_access.py', 'audit_training_2d.py', '__init__.py') if (ROOT/'paper_revision'/n).exists()]
    code_paths += list((ROOT/'src/bpmr_net').glob('*.py'))
    reads += code_paths
    writes = [checkpoints/(f'{m}_G250_{a}_seed{s}_template{t}_{v}.pt')
              for m in ('WF2','GI') for a in 'xyz' for s in (1701,1702,1703)
              for t in range(4) for v in ('B0_SOURCE','A0','A1','A2','A3','A4','A5')]
    reads += writes  # only current-run generated dry checkpoints; no external model input
    runtime = [Path(sys.base_prefix)/'Lib', Path(sys.base_prefix)/'DLLs',
               Path(sysconfig.get_paths()['purelib'])]
    # No repository-wide read grant. Runtime directories are code dependencies,
    # whose native IO and arbitrary data contents are explicitly unverified.
    access = TrainingAccess(reads, writes, runtime)
    sys.dont_write_bytecode = True
    access.install()
    access.phase = 'configuration_and_registry_integrity'
    configs = {str(p.relative_to(ROOT)): dict(sha256=sha(p.read_bytes())) for p in config_paths}
    assert sha(REGISTRY.read_bytes()) == sha(registry_bytes)
    protocol = json.loads((ROOT/'paper_revision/revision_protocol.json').read_bytes())
    contract = json.loads((OUT/'R1_RECONSTITUTION_P2_TRAINING_CONTRACT.json').read_bytes())
    design = json.loads((OUT/'R1_RECONSTITUTION_P0_DATASET_DESIGN.json').read_bytes())
    assert protocol['training']['adaptation_epochs'] == contract['continuation_epochs'] == contract['source_B0']['epochs'] == 50
    assert protocol['training']['training_seeds'] == contract['seeds'] == [1701,1702,1703]
    assert contract['max_events_per_model'] == 100000
    assert protocol['training']['batch_size'] == contract['batch_size'] == 4096
    assert not contract['shuffle'] and not protocol['training']['early_stopping_primary']
    assert contract['design'] == 'A' and contract['optimizer']['lr'] == .001
    budget = json.loads((OUT/'R1_RECONSTITUTION_P1_EXPERIMENT_BUDGET.json').read_bytes())['plans']['C']
    assert (budget['new_B0_runs'], budget['A0_A5_adaptation_runs']) == (120, 720)
    expected_sources = {'WF2':'932853c36761504216c3a734a852c08f6bf451ba6a5c4da9367a2ec8b724d22a',
                        'GI':'35f336b0446a0bd8107197fdde697283b450d67aabb9fd59d4f104d8b97f8c94'}
    payloads = []
    for row in rows:
        if row['role'] not in ('train', 'validation'):
            continue
        bounds = json.loads(row['bounds_zyx'])
        assert row['source_sha256'] == expected_sources[row['material']]
        assert json.loads(row['shape']) == [256,256] and row['dtype'] == 'uint8'
        assert all(a <= lo < hi <= b for (lo, hi), (a,b) in zip(bounds, design['splits'][row['role']]))
        arrays = {}
        for kind in ('labels', 'support'):
            path = OUT / row[kind+'_path']
            data = path.read_bytes()
            assert sha(data) == row[kind+'_sha256']
            assert len(data) == 65536
            arrays[kind] = np.frombuffer(data, dtype=np.uint8).reshape(256,256)
            payloads.append(dict(path=str(path), sha256=sha(data), role=row['role']))
        assert np.isin(arrays['support'], [0,1]).all()
        support = arrays['support'].astype(bool)
        assert int(support.sum()) == int(row['valid_pixels'])
        assert (arrays['labels'][support] < 4).all() and (arrays['labels'][~support] == 255).all()

    access.phase = 'training_module_import'
    before = len(access.records)
    from paper_revision.train_revision_2d import dry_run, require_design_a
    import_records = access.records[before:]
    assert not any(r['category'] == 'registered_input' and r['path'].endswith('.bin') for r in import_records)
    injection = []
    forbidden = [OUT/next(r['labels_path'] for r in rows if r['role']=='test_truth'),
                 OUT/'REVISION_REPLICATION_P1_dataset/test_truth/WF2_ROI0.truth.bin',
                 Path(r'D:\Paper2\paper2_stage02a_source_recovery\official_download\WF2_5um_1030x1033x3051_8b_segmented_corrected.raw'),
                 Path(r'D:\Paper2\paper2_stage02a_source_recovery\official_download\GI_5um_1030x1033x3051_8b_segmented.raw')]
    backup = Path(r'D:\$RECYCLE.BIN\S-1-5-21-2452293266-1158813196-812636544-1001\$RLPIMC9\paper_level\drp167')
    forbidden += [backup/p.name for p in forbidden[2:]]
    methods = {'Path.read_bytes': lambda p: p.read_bytes(),
               'builtins.open': lambda p: open(p, 'rb'),
               'io.open': lambda p: io.open(p, 'rb'),
               'os.open': lambda p: os.open(p, os.O_RDONLY),
               'numpy.fromfile': lambda p: np.fromfile(p, dtype=np.uint8, count=1),
               'numpy.memmap': lambda p: np.memmap(p, dtype=np.uint8, mode='r', shape=(1,))}
    access.phase = 'prohibited_read_injection'
    for path in forbidden:
        for method, attempt in methods.items():
            count = len(access.records)
            try:
                result = attempt(path)
            except PermissionError:
                blocked = any(not r['allowed'] for r in access.records[count:])
            except Exception as exc:
                blocked = False  # missing file / native errors do NOT prove denial
            else:
                blocked = False
                if hasattr(result, 'close'):
                    result.close()
            injection.append(dict(path=str(path), method=method, blocked_by_boundary=blocked))
    # Unknown path, alternate spelling and FD paths fail closed too.
    for method, attempt in [('unknown_input',lambda: Path('unregistered_training_input.bin').read_bytes()),
                            ('traversal',lambda: (OUT/'REVISION_REPLICATION_P1_dataset/train/../test_truth/WF2_ROI0.truth.bin').read_bytes()),
                            ('integer_fd',lambda: os.fdopen(0, 'rb', closefd=False)),
                            ('subprocess',lambda: __import__('subprocess').run([sys.executable,'-c','pass'])),
                            ('ctypes',lambda: __import__('ctypes').CDLL('kernel32.dll'))]:
        try:
            attempt()
        except PermissionError:
            blocked = True
        else:
            blocked = False
        injection.append(dict(method=method, blocked_by_boundary=blocked))
    # Historical / fabricated / dry source checkpoints cannot be production inputs.
    rejected = 0
    for meta in ({}, dict(identity='REVISION_REPLICATION',design='A',unit='x',variant='B0_SOURCE',kind='DRY_RUN_ONLY',historical_checkpoint=False)):
        try:
            require_design_a(meta, 'x')
        except PermissionError:
            rejected += 1
    results = dry_run(OUT, rows, checkpoints, access)
    assert len(results) == 504
    for unit in {r['unit'] for r in results}:
        paired = [r for r in results if r['unit'] == unit]
        assert len({(r['train_events'],r['validation_events']) for r in paired}) == 1
        source_hash = next(r['checkpoint_sha256'] for r in paired if r['variant']=='B0_SOURCE')
        assert all(r['source_checkpoint_sha256'] == source_hash for r in paired if r['variant']!='B0_SOURCE')
    unexpected_denials = [r for r in access.records if not r['allowed'] and
                         r['phase'] != 'prohibited_read_injection' and
                         not (r['phase']=='training_module_import' and r['path'].endswith('.pyc'))]
    assert not unexpected_denials
    access.phase = 'complete'
    access.active = False  # only trusted driver serializes evidence after training returns
    status = dict(TRAINING_INPUT_AUDIT='PASS', TRAINING_ACCESS_AUDIT='FAIL',
                  PROHIBITED_READ_INJECTION_TEST='PASS' if all(r['blocked_by_boundary'] for r in injection) else 'FAIL',
                  DESIGN_A_COMPATIBILITY='PASS' if rejected == 2 else 'FAIL',
                  SCIENTIFIC_DATA_ISOLATION='FAIL', OS_LEVEL_ISOLATION='NOT_ESTABLISHED',
                  TRAINING_DRY_RUN='PASS', TRAINING_READY='NO')
    evidence = dict(status=status, registry_sha256=sha(registry_bytes), plane_count=len(rows),
                    role_counts=dict(Counter(r['role'] for r in rows)), verified_training_payloads=len(payloads),
                    configs=configs, code_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in code_paths},
                    checkpoint_count=len(results), dry_run=dict(steps_per_model=1, train_events_per_unit=8,
                    validation_events_per_unit=8, material=['WF2','GI'], orientations=['x','y','z'],
                    seeds=[1701,1702,1703], templates=[0,1,2,3], spacing_um=250, formal_epochs_completed=0),
                    production_protocol_unchanged=True, historical_checkpoint_inputs=0,
                    import_data_reads=0, injection=injection, runtime_read_roots=[str(p) for p in runtime],
                    python_boundary_status='PASS', unexpected_denials=unexpected_denials,
                    import_cache_policy='Unregistered .pyc reads denied; importer falls back to allowlisted .py sources. No bytecode writes permitted.',
                    limitations=[
                        'Python audit events are not exhaustive for native C/C++ IO; direct Win32/CRT calls inside torch/numpy/native extensions remain UNVERIFIED.',
                        'Dependency bootstrap is recorded but not access-denied; DLL loads and native import-time IO are UNVERIFIED.',
                        'Runtime code directories are read-allowed; native runtime data reads and adversarial code replacement are not isolated.',
                        'Only one-step G250 dry path across both materials/all axes/seeds/templates was executed; full event-cap selection, all spacing regimes, 50-epoch scheduling and final checkpoint provenance registry are NOT released.',
                        'Formal training is disabled; no formal B0 epoch50 exists. Dry checkpoints are explicitly rejected for formal Design A use.',
                        'Budget remains frozen conditional estimate (120 source B0, 720 adaptations for Plan C); this dry run is not a scientific runtime estimate.',
                        'Preparation stays in p1_dataset.main; reconstruction stays in p2_runner (synthetic audit only); final scientific evaluator remains unreleased. Training imports none of them.'
                    ])
    (REPORT/'release.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    (REPORT/'access.json').write_text(json.dumps(dict(bootstrap=bootstrap, guarded=access.records), indent=2), encoding='utf-8')
    (REPORT/'dry_run.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (REPORT/'payloads.json').write_text(json.dumps(payloads, indent=2), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
