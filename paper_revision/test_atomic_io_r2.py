"""Executable non-scientific atomic capability regression tests."""
import json, os, tempfile
from pathlib import Path
from paper_revision.atomic_training_access import AtomicTrainingAccess

def main():
    root=Path(tempfile.mkdtemp(prefix='revision_atomic_r2_')).absolute()
    allowed=root/'outputs'; allowed.mkdir(); forbidden=root/'test_truth'; forbidden.mkdir()
    secret=forbidden/'sentinel'; secret.write_text('NON_SCIENTIFIC_SENTINEL')
    pairs=[]; writes=[]
    for name in ('ledger.csv','manifest.json','epoch50.pt'):
        target=allowed/name; temp=allowed/('unique-'+name+'.pending-42')
        pairs.append((temp,target)); writes.extend((temp,target))
    pairs[0][1].write_text('old-valid'); pairs[1][1].write_text('{"old":true}')
    guard=AtomicTrainingAccess(writes,writes,[],namespace=allowed,pairs=pairs,immutable=[pairs[2][1]])
    guard.install(); tests={}
    def denied(name,fn):
        try: fn()
        except PermissionError: tests[name]=True
        else: tests[name]=False
    for temp,target in pairs:
        content='{"new":true}' if target.suffix=='.json' else 'valid-new'
        temp.write_text(content)
        tests['interrupted_'+target.name]=target.read_text() in ('old-valid','{"old":true}') if target.exists() else not target.exists()
        os.replace(temp,target)
        tests['commit_'+target.name]=target.read_text()==content and not temp.exists()
    denied('test_truth_read',lambda:secret.read_bytes())
    denied('forbidden_to_output',lambda:os.replace(secret,pairs[0][1]))
    denied('output_to_forbidden',lambda:os.rename(pairs[0][1],forbidden/'out'))
    denied('output_to_original',lambda:os.rename(pairs[0][1],root/'original3d'))
    denied('relative_traversal',lambda:os.replace(Path('..')/'anything',pairs[0][1]))
    denied('absolute_traversal',lambda:os.replace(allowed/'..'/'test_truth'/'sentinel',pairs[0][1]))
    pairs[2][0].write_text('replacement')
    denied('frozen_checkpoint_replace',lambda:os.replace(*pairs[2]))
    denied('frozen_checkpoint_write',lambda:pairs[2][1].write_text('bad'))
    denied('delete',lambda:pairs[0][1].unlink())
    denied('unregistered_temp',lambda:os.replace(allowed/'arbitrary',pairs[0][1]))
    guard.active=False
    # Existing Windows junction: create before re-enabling the guard, then attack it.
    import subprocess
    junction=allowed/'junction'
    completed=subprocess.run(['cmd','/c','mklink','/J',str(junction),str(forbidden)],capture_output=True)
    if completed.returncode!=0: raise RuntimeError('junction test unavailable: '+str(completed.stderr))
    guard.active=True
    denied('junction_escape',lambda:os.replace(junction/'sentinel',pairs[0][1]))
    guard.active=False
    result={'all_pass':all(tests.values()),'tests':tests,'audit':guard.records,'test_directory':str(root),'OS_LEVEL_ISOLATION':'NOT_ESTABLISHED'}
    dest=Path(__file__).resolve().parents[1]/'paper_revision/ATOMIC_IO_R2_TEST_RESULTS.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'all_pass':result['all_pass'],'tests':tests}))
    if not result['all_pass']: raise SystemExit(1)
if __name__=='__main__': main()

