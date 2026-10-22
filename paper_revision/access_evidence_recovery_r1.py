"""Read-only, bounded recovery of Design A training access evidence.

This program never imports a training/evaluation entry point or opens plane payloads.
Task labels, attempt identities, checkpoint identities and log identities stay distinct.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/revision_mtcomm_2026'
TRAIN = BASE / 'REVISION_REPLICATION_training/DesignA'
OUT = BASE / 'REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0'
NAME = 'REVISION_DESIGNA_ACCESS_EVIDENCE_RECOVERY_R1'
UNIT = 'GI/G25/x/seed1701/template0'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def label(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return 'local-session/' + path.name


def fingerprint(path):
    stat = path.stat()
    return dict(path=label(path), sha256=sha(path), size=stat.st_size,
                created_utc=datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
                modified_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                timestamp_limit='filesystem times are supporting metadata, not immutable execution timestamps')


def task(row):
    return row['unit'] + '/' + row['variant']


def scan_python(path, formal, registry):
    counts = collections.Counter()
    violations = []
    unknown_reads = collections.Counter()
    a3 = []
    total = 0
    scientific_reads = 0
    with path.open('rb') as stream:
        for index, raw in enumerate(stream, 1):
            item = json.loads(raw)
            identity = task(item)
            counts[identity] += 1
            records = item['records']
            total += len(records)
            expected = formal.get(identity)
            read_keys = set()
            written = set()
            for rec in records:
                op = rec.get('operation')
                p = rec.get('path', '').replace('\\', '/').lower()
                if not rec.get('allowed', False):
                    violations.append(dict(line=index, task=identity, reason='denied_event', event=rec))
                if op == 'read':
                    read_keys.add(p)
                    if p in registry:
                        scientific_reads += 1
                        if registry[p] not in ('train', 'validation'):
                            violations.append(dict(line=index, task=identity, reason='non_training_plane', path=p))
                    elif rec.get('category') == 'registered_input' and p not in ALLOWED_OTHER:
                        unknown_reads[p] += 1
                    if any(token in p for token in ('test_truth', '/incomplete/', '/backup/', '/backups/')):
                        violations.append(dict(line=index, task=identity, reason='prohibited_path_marker', path=p))
                if op == 'write':
                    written.add(p)
            if expected:
                checkpoint = str((ROOT / expected['path']).resolve()).replace('\\', '/').lower()
                source = str((TRAIN / expected['unit'] / 'epoch50.pt').resolve()).replace('\\', '/').lower()
                committed = checkpoint in written or any(
                    r.get('allowed') and r.get('target', '').replace('\\', '/').lower() == checkpoint
                    for r in records)
                if source not in read_keys:
                    violations.append(dict(line=index, task=identity, reason='missing_source_b0_read'))
            else:
                committed = False
                violations.append(dict(line=index, task=identity, reason='identity_not_in_manifest'))
            if identity == UNIT + '/A3':
                a3.append(dict(log_line=index, evidence_id=hashlib.sha256(raw).hexdigest(),
                               event_count=len(records), formal_checkpoint_path_observed=committed,
                               attempt_id=None, start_utc=None, end_utc=None,
                               timestamps_available=False, exit_status='NOT_RECORDED_IN_ACCESS_ROW',
                               write_paths=sorted(written),
                               atomic_commits=[r for r in records if r.get('operation') == 'os.rename']))
    return dict(lines=sum(counts.values()), unique_task_identities=len(counts),
                access_events=total, train_validation_read_events=scientific_reads,
                task_counts=dict(counts), violations=violations,
                unclassified_registered_read_paths=dict(unknown_reads), a3_records=a3)


ALLOWED_OTHER = set()


def session_evidence(path):
    evidence = []
    snapshots = []
    # One known original session, limited to the initial training window.
    with path.open(encoding='utf-8') as stream:
        for raw in stream:
            row = json.loads(raw)
            time = row.get('timestamp', '')
            if not ('2026-10-19T06:57' <= time < '2026-10-19T07:21'):
                continue
            item = row.get('payload', {}).get('item', {})
            if item.get('type') == 'FileChange':
                for name, change in item.get('changes', {}).items():
                    source = change.get('content', '')
                    if name.endswith('controlled_a0a5_r0.py') and 'def main():' in source:
                        snapshots.append(dict(recorded_utc=time, evidence_id=item.get('id'),
                                              source_snapshot_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                              guard_installed='guard.install()' in source,
                                              persists_guard_records='guard.records' in source,
                                              scope='historical edit content; exact loaded executable binding not established'))
            if item.get('type') != 'CommandExecution':
                continue
            output = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07', '', item.get('aggregated_output', ''))
            command = item.get('command', [])
            if any('completed ' in output and UNIT + ' ' + variant in output
                   for variant in ('A0', 'A1', 'A2')):
                evidence.append(dict(evidence_id=item.get('id'), recorded_utc=time,
                                     execution_id=item.get('process_id'),
                                     command=command[-1] if command else None,
                                     exit_code=item.get('exit_code'), status=item.get('status'),
                                     duration=item.get('duration'),
                                     completion_lines=[s for s in output.splitlines() if s.startswith('completed ')],
                                     scope='command/output execution evidence; not per-file access trace'))
            if 'def main():' in output and 'guard.install()' in output and 'controlled_a0a5_r0.py' in str(command):
                snapshots.append(dict(recorded_utc=time, evidence_id=item.get('id'),
                                      source_snapshot_sha256=hashlib.sha256(output.encode()).hexdigest(),
                                      guard_installed=True,
                                      persists_guard_records='guard.records' in output,
                                      scope='observed source snapshot, not proof of exact code loaded by every earlier process'))
    return dict(executions=evidence, source_snapshots=snapshots,
                historical_immutability='No independent contemporaneous signature/hash was found for the session transcript; current hash anchors this recovery only.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = TRAIN / 'REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json'
    b0_path = TRAIN / 'REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json'
    log = TRAIN / 'REVISION_DESIGNA_A0A5_R0_PYTHON_ACCESS.jsonl'
    freeze_path = OUT / 'REVISION_DESIGNA_HELDOUT_R0_MODEL_FREEZE.json'
    registry_path = BASE / 'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv'
    prior_path = ROOT / 'paper_revision/heldout_r0/TRAINING_ACCESS_COVERAGE.json'
    orphan_path = ROOT / 'paper_revision/ORPHAN_A3_EVIDENCE_R2.json'
    recovery_path = ROOT / 'paper_revision/ATOMIC_IO_R2_RECOVERY.json'
    native_path = TRAIN / 'REVISION_DESIGNA_A0A5_R0_NATIVE_ACCESS.jsonl'
    controller_path = TRAIN / 'REVISION_DESIGNA_A0A5_R0_NATIVE_CONTROLLER.json'
    sources = [manifest_path, b0_path, log, freeze_path, registry_path, prior_path,
               orphan_path, recovery_path, native_path, controller_path, args.session]
    sources += sorted(TRAIN.glob('*.log'))
    sources += sorted(TRAIN.glob('*POSTTRAIN*.json'))
    before = {label(p): fingerprint(p) for p in sources}
    manifest = read_json(manifest_path)
    formal = {task(r): r for r in manifest['checkpoints']}
    assert len(formal) == len(manifest['checkpoints']) == 720, 'formal identity conflict'
    b0 = read_json(b0_path)
    assert len(b0['checkpoints']) == 120
    registry = {}
    with registry_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            for kind in ('labels', 'support'):
                registry[str((BASE / row[kind + '_path']).resolve()).replace('\\', '/').lower()] = row['role']
    ALLOWED_OTHER.update(str((ROOT / r['path']).resolve()).replace('\\', '/').lower()
                         for r in b0['checkpoints'])
    ALLOWED_OTHER.update(str((ROOT / r['path']).resolve()).replace('\\', '/').lower()
                         for r in formal.values())
    ALLOWED_OTHER.update(str(p.resolve()).replace('\\', '/').lower()
                         for p in (registry_path, manifest_path, b0_path,
                                   ROOT/'paper_revision/TRAINING_2D_RELEASE_R2.json',
                                   ROOT/'paper_revision/revision_protocol.json'))
    release = read_json(ROOT / 'paper_revision/TRAINING_2D_RELEASE_R2.json')
    ALLOWED_OTHER.update(str((ROOT / p).resolve()).replace('\\', '/').lower()
                         for p in release['launch_binding'])
    access = scan_python(log, formal, registry)
    missing = sorted(set(formal) - set(access['task_counts']))
    assert missing == [UNIT + '/A0', UNIT + '/A1'], 'coverage changed; investigate'
    transcript = session_evidence(args.session)
    a3_records = access.pop('a3_records')
    duplicates = [r['log_line'] for r in a3_records
                  if sum(x['evidence_id'] == r['evidence_id'] for x in a3_records) > 1]
    checkpoint_conflicts = []
    freeze = read_json(freeze_path)
    assert freeze['status'] == 'PASS' and len(freeze['models']) == 840
    for model in freeze['models']:
        if sha(ROOT / model['path']) != model['sha256']:
            checkpoint_conflicts.append(model['unit'] + '/' + model['variant'])
    orphan = read_json(orphan_path)
    orphan_current = fingerprint(Path(orphan['path']))
    a3_formal = formal[UNIT + '/A3']
    assert sha(ROOT / a3_formal['path']) == a3_formal['checkpoint_sha256']
    assert sum(task(r) == UNIT + '/A3' for r in manifest['checkpoints']) == 1
    # Native trace has no authenticated wall-clock/task attempt binding. Do not
    # infer historical A0/A1 coverage from bootstrap/preflight checkpoint reads.
    native_counts = collections.Counter()
    native_blocked = 0
    native_prohibited_candidates = []
    native_task_markers = set()
    with native_path.open(encoding='utf-8') as stream:
        for line in stream:
            rec = json.loads(line)
            native_counts[rec.get('api', rec.get('controller', 'other'))] += 1
            native_blocked += bool(rec.get('blocked'))
            p = rec.get('path', '').replace('\\', '/').lower()
            if rec.get('api') in ('NtReadFile', 'NtOpenFile', 'NtCreateFile', 'NtCreateSection', 'NtMapViewOfSection'):
                if any(t in p for t in ('/test_truth/', '/official_download/', '/$recycle.bin/', '/backup/', '/backups/')) or p.endswith('.raw'):
                    native_prohibited_candidates.append(rec)
            if rec.get('api') == 'phase':
                native_task_markers.add(rec.get('path', ''))
    source_stable = all(sha(p) == before[label(p)]['sha256'] for p in sources)
    assert source_stable, 'source evidence changed while being audited'
    prior = read_json(prior_path)
    posttrain = read_json(TRAIN / 'R0_POSTTRAIN_PYTHON_ACCESS.json')
    initial_checkpoints = []
    for variant in ('A0', 'A1', 'A2', 'A3'):
        row = formal[UNIT + '/' + variant]
        initial_checkpoints.append(dict(
            unit=UNIT, variant=variant, file=fingerprint(ROOT / row['path']),
            epochs=len(row['epochs']), source_checkpoint_sha256=row['source_checkpoint_sha256'],
            train_events=row['train_events'], validation_events=row['validation_events'],
            strict_reload=row['strict_reload'], finite_parameters=row['finite_parameters'],
            elapsed_seconds=row['elapsed_seconds'],
            metadata_scope='existing formal manifest; full payload strict reload retained from prior model freeze'))
    report = dict(
        stage=NAME, status='PARTIAL',
        REVISION_DESIGNA_ACCESS_EVIDENCE_RECOVERY_R1='PARTIAL',
        HELDOUT_EVALUATION_READY='NO', OS_LEVEL_ISOLATION='NOT_ESTABLISHED',
        A0_original_access_evidence='MISSING', A1_original_access_evidence='MISSING',
        unavailable_status='ORIGINAL_ACCESS_EVIDENCE_UNAVAILABLE',
        missing_tasks=missing,
        formal_task_access_row_coverage=access['unique_task_identities'],
        formal_task_count=720,
        coverage_qualification='718 task-labelled streams; timestamps and authenticated attempt bindings are absent from the Python rows. This is not 720 verified historical executions.',
        python_access=access,
        native_access=dict(event_counts=dict(native_counts), blocked_events=native_blocked,
                           prohibited_path_candidates=native_prohibited_candidates,
                           phase_markers=sorted(native_task_markers), controller=read_json(controller_path),
                           A0_A1_execution_binding='NOT_ESTABLISHED',
                           limitation='trace file is opened with w by the wrapper; no historical trace versions recovered. Direct syscalls and pre-hook loader I/O are not exhaustive.'),
        contemporaneous_execution_evidence=transcript,
        initial_formal_checkpoint_evidence=initial_checkpoints,
        alternate_posttrain_access=dict(records=len(posttrain),
                                        phases=sorted({r.get('phase', '') for r in posttrain}),
                                        classification='B0 posttraining verification, not A0/A1 controlled training; not substituted'),
        A0_A1_conclusion='Original command/completion output and checkpoint integrity recovered. Full file-access events, effective audit coverage and original executable binding are not recoverable from those outputs. A0 early source snapshot did not install a guard; later A1 snapshot installed a guard but did not persist guard.records.',
        A3_multiple_attempt_attribution='FAIL',
        A3_attribution_reason='Three rows are not three proven attempts. Rows 2 and 3 are identical; no attempt IDs or timestamps allow distinct process attribution. Row 4 binds to the final reexecution target, but complete historical attempt timing/exit mapping remains unavailable.',
        A3_records=a3_records, A3_duplicate_log_lines=duplicates,
        A3_orphan=dict(evidence=orphan, current=orphan_current,
                       hash_matches_captured=orphan_current['sha256'] == orphan['sha256'],
                       formal=False),
        A3_formal=dict(path=a3_formal['path'], sha256=a3_formal['checkpoint_sha256'],
                       formal_count=1, epochs=len(a3_formal['epochs']),
                       recovery_reason=a3_formal.get('recovery_reason'),
                       orphan_and_formal_bytes_equal=orphan_current['sha256'] == a3_formal['checkpoint_sha256'],
                       equality_limitation='Equal bytes do not independently prove or disprove reexecution; preserve both provenance records.'),
        checkpoint_hash_conflicts=checkpoint_conflicts,
        checkpoint_to_access_conflict='No positive contradictory checkpoint binding found; A0/A1 binding missing and old A3 attempt attribution unresolved.',
        model_integrity=dict(status='PASS' if not checkpoint_conflicts else 'FAIL',
                             checkpoints_rehashed=840, prior_strict_reload='840/840 PASS; retained from model freeze, not rerun here',
                             freeze_sha256=sha(freeze_path), manifest_sha256=sha(manifest_path)),
        original_evidence_modified_by_this_audit=False,
        source_hashes_stable_during_audit=source_stable,
        python_log_matches_prior_audit_hash=sha(log) == prior['access_log_sha256'],
        historical_log_immutability='No blanket historical immutability claim; current source hashes and prior frozen Python-log hash are recorded.',
        observed_prohibited_reads=len(access['violations']),
        observed_prohibited_reads_scope='available Python events, role/path checks and policy denials; absent A0/A1 events remain unknown',
        decision='Original release standard unchanged. Held-out evaluation remains blocked.',
        conditional_acceptance_recommendation='A new explicit scientific acceptance decision could permit disclosed conditional evidence, but must accept missing A0/A1 audit history, unverified early code/audit coverage, and incomplete A3 attempt attribution. This audit does not authorize that change or label it PASS.',
        safest_alternative='Recover an independent archived original access stream with task/process/time/source binding. If unavailable, make the above new decision explicitly; otherwise retain the existing evaluation block. Do not regenerate history by retraining.',
        bounded_search=dict(roots=['paper_revision', label(TRAIN), 'known initial local Codex session'],
                            sources_examined=sorted(before),
                            exclusions='No original scientific payloads/test labels opened; no unbounded drive scan; unsearched external archives remain unknown.'),
        sources=before, actions='Read evidence and rehash frozen checkpoints only; no training, held-out inference, reconstruction or external baseline.'
    )
    summary = (
        '# Design A access evidence recovery R1\n\n'
        'REVISION_DESIGNA_ACCESS_EVIDENCE_RECOVERY_R1=PARTIAL\n'
        'HELDOUT_EVALUATION_READY=NO\n\n'
        '- A0: MISSING; A1: MISSING (ORIGINAL_ACCESS_EVIDENCE_UNAVAILABLE).\n'
        '- Recovered original completion outputs: A0 at 2026-10-19 06:59:09 UTC; '
        'A1 initial completion before an interrupted campaign, followed by a successful limit-2 execution at 07:04:34 UTC. '
        'These execution outputs do not contain the per-file access stream.\n'
        '- The early A0 source snapshot had no installed access guard; a later A1 snapshot installed a guard '
        'but did not persist its records. Source snapshots cannot prove the exact code loaded in every earlier attempt.\n'
        '- Available Python log: 720 rows, 718 task identities, 622,927 events; '
        f'{len(access["violations"])} observed policy/path violations. Missing events remain unknown.\n'
        '- A3 full attempt attribution: FAIL. Rows 2/3 are byte-identical and lack attempt IDs/timestamps; '
        'row 4 records pending-to-epoch50_reexecution.pt. The failed orphan is retained, '
        'and exactly one formal A3 checkpoint remains in the manifest. Its bytes equal the orphan; '
        'that equality alone is not proof of an independent rerun.\n'
        f'- Model integrity: {report["model_integrity"]["status"]}; all 840 frozen checkpoint hashes rechecked. '
        'The prior strict-reload audit is retained. No original evidence modified; source hashes stayed stable.\n\n'
        'The native stream cannot establish A0/A1 process/time coverage, and the wrapper overwrites its trace destination on each launch. '
        'No direct evidence of prohibited reading was found in available streams; absence of a missing stream cannot prove no reading occurred. '
        'OS_LEVEL_ISOLATION=NOT_ESTABLISHED.\n\n'
        'Keep the original held-out release gate blocked. The minimal alternative is an explicit new acceptance decision '
        'that discloses missing A0/A1 original access history and unresolved A3 attempt provenance. '
        'Code/input/checkpoint hashes and existing events support limited integrity claims but cannot recreate lost history. '
        'No standard was relaxed and no evaluation or retraining was started.\n'
    )
    for destination in (OUT, ROOT / 'paper_revision/access_evidence_r1'):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (NAME + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        (destination / (NAME + '_SUMMARY.md')).write_text(summary, encoding='utf-8')
    print(json.dumps(dict(status=report['status'], heldout_ready='NO',
                          coverage=access['unique_task_identities'], model_conflicts=checkpoint_conflicts,
                          unclassified_read_paths=len(access['unclassified_registered_read_paths']),
                          report_sha256=sha(OUT / (NAME + '.json')))))


if __name__ == '__main__':
    main()
