"""Read-only provenance check; preserve blocked evaluation evidence without fitting."""
import collections
import json
from pathlib import Path

from paper_revision.heldout_evaluation_r0 import OUT, PREFIX, ROOT, TRAIN, dump, sha


def main():
    path=TRAIN/'REVISION_DESIGNA_A0A5_R0_PYTHON_ACCESS.jsonl'
    manifest=json.loads((TRAIN/'REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json').read_bytes())
    counts=collections.Counter(); records=0; denials=[]; prohibited=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            item=json.loads(line); counts[(item['unit'],item['variant'])]+=1
            for rec in item['records']:
                records+=1
                if not rec['allowed']: denials.append(rec)
                if rec['operation']=='read' and ('test_truth' in rec['path'].lower() or 'incomplete' in rec['path'].lower()): prohibited.append(rec)
    missing=[dict(unit=r['unit'],variant=r['variant']) for r in manifest['checkpoints'] if (r['unit'],r['variant']) not in counts]
    evidence=dict(access_log=str(path.relative_to(ROOT)),access_log_sha256=sha(path),lines=sum(counts.values()),unique_tasks=len(counts),
        records=records,missing_tasks=missing,duplicates=[dict(unit=k[0],variant=k[1],records=v) for k,v in counts.items() if v>1],
        denials=denials,observed_test_or_incomplete_reads=prohibited,
        interpretation='Missing access provenance is not proof of leakage; zero observed violations does not establish coverage of missing tasks.')
    dump('TRAINING_ACCESS_COVERAGE',evidence)
    assert missing, 'This blocked report is only appropriate for the diagnosed missing provenance'
    model_freeze=json.loads((OUT/(PREFIX+'MODEL_FREEZE.json')).read_bytes())
    audit=dict(stage='REVISION_DESIGNA_HELDOUT_EVALUATION_R0',status='BLOCKED',
        SCIENTIFIC_TRAINING_COMPLETE='840_CHECKPOINTS_VERIFIED; FINAL_ACCESS_PROVENANCE_INCOMPLETE',
        HELDOUT_EVALUATION_COMPLETE=False,THREE_D_RECONSTRUCTION_COMPLETE=False,PUBLISHED_BASELINE_COMPLETE=False,
        checkpoint_model_freeze=model_freeze['status'],b0_verified=120,a0a5_verified=720,evaluated_models=0,
        test_payload_reads=0,protocol_frozen_before_test_read=True,OS_LEVEL_ISOLATION='NOT_ESTABLISHED',
        blocker='Formal A0/A1 training-access provenance missing for GI/G25/x/seed1701/template0',
        locations=['paper_revision/heldout_evaluation_r0.py:evaluate:formal training access audit conflict',
                   str(path.relative_to(ROOT))],
        recovery='Recover original task-bound Python/native access evidence from the initial A0/A1 executions or an archived log. Verify source checkpoint, event hashes, time/process identity and all actual scientific input paths. Do not invent logs, extrapolate another task, rerun training, or silently waive this gate.',
        side_effects='No training, model changes, test tuning, evaluation inference, reconstruction or external baseline executed.',
        audit_scope='All840 checkpoint payloads independently hashed, strictly reloaded and checked finite; shared source/events and residual B0 equality checked; current training Python audit examined. Native historical trace cannot automatically replace missing per-task provenance.',
        access_evidence_sha256=sha(OUT/(PREFIX+'TRAINING_ACCESS_COVERAGE.json')),
        model_freeze_sha256=sha(OUT/(PREFIX+'MODEL_FREEZE.json')),protocol_sha256=sha(OUT/(PREFIX+'PROTOCOL.json')))
    dump('FINAL_AUDIT',audit)
    summary='''# Design A held-out evaluation R0

REVISION_DESIGNA_HELDOUT_EVALUATION_R0=BLOCKED

Material Passport: local revision checkpoint/provenance audit; no test payload loaded; no external upload.

## Verified model evidence

120 common B0 and720 A0-A5 checkpoint identities, SHA256, strict reload, finite parameters and50epochs passed. All A0-A5 source B0/event identities matched; A2-A5 B0 states equal the common source; A4 zero frozen gate weights and A5 dilation configuration verified. A3/A5 registered parameter structures match; A4 has fewer effective trainable parameters.

## Exact blocker

Formal training access log has720 lines but718 unique tasks and622927 access records. Missing: GI/G25/x/seed1701/template0/A0 andA1. The same A3 identity has3 records from recovery attempts. Zero denials andzero test_truth/incomplete reads were observed in the available records. This is an evidence-coverage gap, not an observed data leak.

`heldout_evaluation_r0.evaluate` stops before the first test payload read. The frozen protocol and model list remain preserved. No test metrics, confidence intervals, gate findings, phase degeneration claims or reviewer efficacy conclusions can be reported.

## Minimal recovery

Locate original A0/A1 access records in archives/initial launch evidence. Establish task/process/time/source B0/event binding and scientific input coverage before allowing evaluation. The current native trace includes bootstrap/preflight operations and cannot be treated as per-task training evidence without proving those bindings. Do not retrain, fabricate missing records or substitute the later A3 logs. No checkpoint or scientific input was modified.

## Stage boundaries

SCIENTIFIC_TRAINING_COMPLETE=840_CHECKPOINTS_VERIFIED_ACCESS_PROVENANCE_INCOMPLETE
HELDOUT_EVALUATION_COMPLETE=NO
THREE_D_RECONSTRUCTION_COMPLETE=NO
PUBLISHED_BASELINE_COMPLETE=NO
OS_LEVEL_ISOLATION=NOT_ESTABLISHED

The previously reported720/720 count describes checkpoint completion, not final provenance release. No claim of test performance, cross-material generalization, reconstruction superiority or reviewer issue resolution is supported by this blocked run.
'''
    (OUT/(PREFIX+'SUMMARY.md')).write_text(summary,encoding='utf-8')
    files=sorted(p for p in OUT.iterdir() if p.is_file() and not p.name.endswith('RESULT_MANIFEST.json'))
    digest=dump('RESULT_MANIFEST',dict(stage=audit['stage'],status='BLOCKED',files={p.name:sha(p) for p in files}))
    # Compact, non-sensitive tracked record. Scientific payloads remain local.
    target=ROOT/'paper_revision/heldout_r0'
    target.mkdir(exist_ok=True)
    (target/'FINAL_AUDIT.json').write_text(json.dumps(dict(**audit,result_manifest_sha256=digest),indent=2),encoding='utf-8')
    (target/'TRAINING_ACCESS_COVERAGE.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (target/'SUMMARY.md').write_text(summary+'\nResult manifest SHA256: '+digest+'\n',encoding='utf-8')
    print(json.dumps(dict(status='BLOCKED',evaluated=0,missing=missing,result_manifest_sha256=digest)))


if __name__=='__main__':
    main()
