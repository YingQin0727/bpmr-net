"""Read-only recovery gate: never adopts or retrains an orphan automatically."""
import csv, io, json, math
import torch
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.controlled_a0a5_r0 import ROOT,OUT,P,preflight,sha_file,VARIANTS,unit_cells

def main():
 torch.set_num_threads(1)
 _,b0=preflight()
 manifest=json.loads((OUT/(P+'_CHECKPOINT_MANIFEST.json')).read_text())
 ledger=list(csv.DictReader((OUT/(P+'_TASK_LEDGER.csv')).open(encoding='utf-8-sig')))
 expected={(f'{m}/G{g}/{a}/seed{s}/template{t}',v) for m,g,a,s,t in unit_cells() for v in VARIANTS}
 assert len(expected)==720
 verified=[]; known=set()
 for r in manifest['checkpoints']:
  identity=(r['unit'],r['variant']); assert identity in expected and identity not in known; known.add(identity)
  path=ROOT/r['path']; assert sha_file(path)==r['checkpoint_sha256']
  ck=torch.load(io.BytesIO(path.read_bytes()),weights_only=True,map_location='cpu'); meta=ck['metadata']
  model=B0Network()
  if r['variant'] not in ('A0','A1'): model=build_revision_bpmr(model,r['variant'])
  model.load_state_dict(ck['state'],strict=True)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  assert meta['epochs_completed']==50 and meta['kind']=='EPOCH50'
  assert meta['source_checkpoint_sha256']==r['source_checkpoint_sha256']==b0[r['unit']]['sha256']
  for role in ('train','validation'): assert meta[role+'_events']==r[role+'_events']==b0[r['unit']]['events'][role]
  assert [e['epoch'] for e in r['epochs']]==list(range(1,51))
  steps=math.ceil(r['train_events']['count']/4096)
  assert all(e['optimizer_steps']==steps and all(math.isfinite(v) for v in [e['train_loss'],*e['validation'].values()]) for e in r['epochs'])
  assert all(float(v['step'])==50*steps for v in ck['optimizer']['state'].values())
  matches=[l for l in ledger if (l['unit'],l['variant'])==identity]; assert len(matches)==1 and matches[0]['checkpoint_sha256']==r['checkpoint_sha256']
  verified.append({'unit':r['unit'],'variant':r['variant'],'sha256':r['checkpoint_sha256'],'status':'VERIFIED_COMPLETE'})
 orphans=[]
 for path in (OUT/'A0A5').rglob('epoch50.pt'):
  ck=torch.load(io.BytesIO(path.read_bytes()),weights_only=True,map_location='cpu'); meta=ck['metadata']; identity=(meta['unit'],meta['variant'])
  if identity not in known:
   orphans.append({'path':str(path),'sha256':sha_file(path),'unit':meta['unit'],'variant':meta['variant'],'epochs_claimed':meta['epochs_completed'],'optimizer_steps':sorted({float(v['step']) for v in ck['optimizer']['state'].values()}),'missing_evidence':['complete per-epoch training/validation trajectory','committed per-task manifest record'],'action':'PRESERVED_IN_PLACE; NOT_COUNTED; NO_RETRAIN'})
 smoke=OUT/'ATOMIC_IO_R2_NON_SCIENTIFIC_TEST'
 sm=json.loads((smoke/'NON_SCIENTIFIC_TEST_CHECKPOINT_MANIFEST.json').read_text()); assert len(sm['checkpoints'])==1
 sr=sm['checkpoints'][0]; assert sr['strict_reload'] and sr['finite_parameters'] and len(sr['epochs'])==2
 assert not list(smoke.glob('*.tmp')) and not list(smoke.rglob('*.pending'))
 tests=json.loads((ROOT/'paper_revision/ATOMIC_IO_R2_TEST_RESULTS.json').read_text());assert tests['all_pass']
 result={'stage':'REVISION_DESIGNA_A0A5_ATOMIC_IO_FIX_R2','ATOMIC_IO_FIX':'PASS','PROHIBITED_PATH_TESTS':'PASS','ATOMIC_LEDGER_TEST':'PASS','CHECKPOINT_RECOVERY':'FAIL' if orphans else 'PASS','TRAINING_RESUME':'STOPPED' if orphans else 'READY','FORMAL_COMPLETED_COUNT':len(verified),'CURRENT_PID':None,'verified':verified,'orphans':orphans,'matrix_count':len(expected),'smoke_exit_code':0,'smoke_kind':'NON_SCIENTIFIC_TEST','OS_LEVEL_ISOLATION':'NOT_ESTABLISHED','limitations':['Python capability control; concurrent hostile filesystem changes and direct native syscalls are not an OS security boundary.','Interruption regression covers prepared-before-replace state and committed-after-replace state, not power-loss durability.']}
 (ROOT/'paper_revision/ATOMIC_IO_R2_RECOVERY.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()

