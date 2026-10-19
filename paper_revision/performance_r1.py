"""Read-only performance and equivalence study for A2-A5."""
import json,time,hashlib,os,statistics
from pathlib import Path
import numpy as np, torch
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.training_events import inventory,event_arrays
from paper_revision.training_campaign_2d import controlled_optimizer
from paper_revision.train_revision_2d import probability,loss_for
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'outputs/revision_mtcomm_2026'; OUT=BASE/'REVISION_REPLICATION_training/DesignA'; P='REVISION_DESIGNA_TRAIN_PERFORMANCE_R1'
UNIT='GI/G25/x/seed1701/template0'; VARS=('A2','A3','A4','A5')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
 rows=list(__import__('csv').DictReader((BASE/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
 m,g,ax,s,t=UNIT.split('/');g=int(g[1:]);s=int(s[4:]);t=int(t[8:]); seed=int(sha(UNIT.encode())[:8],16)
 tr=event_arrays(BASE,inventory(BASE,rows,m,ax,'train'),g,seed); va=event_arrays(BASE,inventory(BASE,rows,m,ax,'validation'),g,seed+1)
 x,y,_=tr; vx,vy,_=va; x=torch.from_numpy(x);y=torch.from_numpy(y);vx=torch.from_numpy(vx);vy=torch.from_numpy(vy)
 b0path=OUT/f'{m}/G{g}/{ax}/seed{s}/template{t}/epoch50.pt'; b0ck=torch.load(b0path,weights_only=True,map_location='cpu'); b0=B0Network();b0.load_state_dict(b0ck['state'],strict=True)
 profile={'unit':UNIT,'threads':torch.get_num_threads(),'cpu_count':os.cpu_count(),'input_count':len(x),'batch_size':4096,'variants':{},'notes':['Read-only benchmark; no scientific checkpoint writes.','Profiler isolates model and optimizer costs; event construction timed separately.']}
 # separate event construction cost
 t0=time.perf_counter();
 for _ in range(2): event_arrays(BASE,inventory(BASE,rows,m,ax,'train'),g,seed)
 profile['event_build_seconds_two_runs']=time.perf_counter()-t0
 for v in VARS:
  torch.manual_seed(seed); model=build_revision_bpmr(__import__('copy').deepcopy(b0),v); opt=controlled_optimizer(model)
  batch=x[:4096]; target=y[:4096]; times=[]; losses=[]
  for i in range(3):
   opt.zero_grad(set_to_none=False); a=time.perf_counter(); p=probability(model,batch); loss=loss_for(v,p,target); loss.backward(); opt.step(); times.append(time.perf_counter()-a); losses.append(float(loss.detach()))
  with torch.no_grad(): vp=probability(model,vx[:4096]); vmse=float(torch.nn.functional.mse_loss(vp,torch.nn.functional.one_hot(vy[:4096],4).float())); vnll=float(torch.nn.functional.nll_loss(vp.clamp_min(1e-12).log(),vy[:4096]))
  profile['variants'][v]={'batch_seconds':times,'median_batch_seconds':statistics.median(times),'estimated_50_epoch_seconds':statistics.median(times)*25*50,'peak_rss_bytes':__import__('psutil').Process().memory_info().rss,'final_loss':losses[-1],'validation_mse':vmse,'validation_nll':vnll}
 # Operator-level CPU/memory profile for the representative A2 batch.
 torch.manual_seed(seed); pm=build_revision_bpmr(__import__('copy').deepcopy(b0),'A2'); po=controlled_optimizer(pm); po.zero_grad()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as prof:
  pp=probability(pm,x[:4096]); ll=loss_for('A2',pp,y[:4096]); ll.backward(); po.step()
 profile['operator_profile']=[{'name':e.key,'calls':e.count,'cpu_time_us':e.cpu_time_total,'self_cpu_time_us':e.self_cpu_time_total,'self_cpu_memory_bytes':e.self_cpu_memory_usage} for e in sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:25]]
 # Thread-count check uses identical batch and does not save checkpoints.
 thread_results={}
 for n in (1,2,4):
  torch.set_num_threads(n); torch.manual_seed(seed); tm=build_revision_bpmr(__import__('copy').deepcopy(b0),'A2'); to=controlled_optimizer(tm); vals=[]
  for _ in range(2):
   to.zero_grad(); st=time.perf_counter(); q=probability(tm,x[:4096]); z=loss_for('A2',q,y[:4096]); z.backward(); to.step(); vals.append(time.perf_counter()-st)
  thread_results[str(n)]={'median_batch_seconds':statistics.median(vals),'samples':vals}
 profile['thread_count_benchmark']=thread_results; torch.set_num_threads(1)
 # Actual full train and validation event traversal for one epoch per variant.
 full_epoch={}
 for v in VARS:
  torch.manual_seed(seed); fm=build_revision_bpmr(__import__('copy').deepcopy(b0),v); fo=controlled_optimizer(fm); st=time.perf_counter(); steps=0
  for begin in range(0,len(x),4096):
   fo.zero_grad(); fp=probability(fm,x[begin:begin+4096]); fl=loss_for(v,fp,y[begin:begin+4096]); fl.backward(); fo.step(); steps+=1
  train_s=time.perf_counter()-st; st=time.perf_counter()
  with torch.no_grad():
   for begin in range(0,len(vx),4096): probability(fm,vx[begin:begin+4096])
  val_s=time.perf_counter()-st
  full_epoch[v]={'train_seconds':train_s,'validation_seconds':val_s,'steps':steps,'estimated_50_epoch_seconds':50*(train_s+val_s)}
 profile['full_event_one_epoch']=full_epoch
 # exact paired initial/gradient test (before optimizer update)
 eq={}
 for v in VARS:
  torch.manual_seed(seed); a=build_revision_bpmr(__import__('copy').deepcopy(b0),v); torch.manual_seed(seed); b=build_revision_bpmr(__import__('copy').deepcopy(b0),v); oa=controlled_optimizer(a);ob=controlled_optimizer(b); oa.zero_grad();ob.zero_grad();la=loss_for(v,probability(a,x[:4096]),y[:4096]);lb=loss_for(v,probability(b,x[:4096]),y[:4096]);la.backward();lb.backward(); grads=all(torch.equal(pa.grad,pb.grad) for pa,pb in zip(a.parameters(),b.parameters()) if pa.grad is not None and pb.grad is not None)
  opt_equal=True
  for _ in range(3):
   for opt,mod in ((oa,a),(ob,b)):
    opt.zero_grad(); z=loss_for(v,probability(mod,x[:4096]),y[:4096]);z.backward();opt.step()
   opt_equal &= all(torch.equal(pa,pb) for pa,pb in zip(a.parameters(),b.parameters()))
  state_a=oa.state_dict();state_b=ob.state_dict(); opt_equal &= all(torch.equal(state_a['state'][k]['exp_avg'],state_b['state'][k]['exp_avg']) and torch.equal(state_a['state'][k]['exp_avg_sq'],state_b['state'][k]['exp_avg_sq']) for k in state_a['state'])
  checkpoint=__import__('io').BytesIO(); saved_rng=torch.get_rng_state().clone();torch.save({'state':a.state_dict(),'optimizer':oa.state_dict(),'rng':saved_rng},checkpoint); checkpoint.seek(0);loaded=torch.load(checkpoint,weights_only=True); c=build_revision_bpmr(__import__('copy').deepcopy(b0),v);c.load_state_dict(loaded['state'],strict=True)
  eq[v]={'initial_output_equal':torch.equal(probability(a,x[:4096]),probability(b,x[:4096])),'loss_equal':torch.equal(la,lb),'gradient_equal':grads,'optimizer_multistep_parameter_equal':opt_equal,'strict_reload':all(torch.equal(pa,pb) for pa,pb in zip(a.parameters(),c.parameters())),'rng_state_saved':torch.equal(loaded['rng'],saved_rng),'loss_abs_diff':abs(float(la-lb))}
 (OUT/(P+'_PROFILE.json')).write_text(json.dumps(profile,indent=2),encoding='utf-8');(OUT/(P+'_EQUIVALENCE.json')).write_text(json.dumps({'unit':UNIT,'variants':eq,'optimized_implementation':'NONE_RELEASED','decision':'NO_OPTIMIZATION_FROZEN'},indent=2),encoding='utf-8')
 raw_seconds=sum(full_epoch[v]['estimated_50_epoch_seconds']*(119 if v=='A2' else 120) for v in VARS)
 measured_a2=334.776860713958; calibration=measured_a2/full_epoch['A2']['estimated_50_epoch_seconds']
 bench={'unit':UNIT,'threads_tested':[1,2,4],'serial':profile['variants'],'thread_count_benchmark':profile['thread_count_benchmark'],'full_event_one_epoch':full_epoch,'parallel_process_tests':'NOT_RUN_TO_AVOID_SCIENTIFIC_CHECKPOINT_CONTENTION','optimized_implementation':'NONE','before_after_speedups':'NOT_APPLICABLE','native_trace_overhead':'NOT_ISOLATED; prior formal task wall time includes different launch/audit conditions','recommendation':'4 intra-process threads was fastest in this microbenchmark, but no formal setting change is released; retain serial until process-level throughput and bitwise protocol behavior are verified','remaining_tasks':717,'uncalibrated_model_only_cpu_seconds':raw_seconds,'calibration_reference':{'completed_A2_wall_seconds':measured_a2,'A2_model_only_projection_seconds':full_epoch['A2']['estimated_50_epoch_seconds'],'factor':calibration},'estimated_remaining_wall_seconds_range':[raw_seconds*calibration,raw_seconds*calibration*1.2],'server_required':'not determined; estimated local runtime is about %.1f-%.1f h under current calibration' % (raw_seconds*calibration/3600,raw_seconds*calibration*1.2/3600)}
 (OUT/(P+'_BENCHMARK.json')).write_text(json.dumps(bench,indent=2),encoding='utf-8')
 (OUT/(P+'_SUMMARY.md')).write_text(f'''# {P}

REVISION_DESIGNA_TRAIN_PERFORMANCE_R1=PARTIAL

Read-only profiling used {UNIT}, 100,000 frozen training events, batch size 4096, CPU threads 1 unless noted. Event construction took {profile['event_build_seconds_two_runs']:.3f} s for two complete builds. The representative operator profile identifies `aten::convolution_backward` and MKLDNN convolution as the main CPU hotspots; `features_to_patch` scatter appears as `aten::_index_put_impl_`. Frozen B0 forward, gate/residual head, softmax and Adam are smaller contributors.

A2-A5 median model batch times (forward, backward, optimizer) were {', '.join(f"{v}: {profile['variants'][v]['median_batch_seconds']:.4f}s" for v in VARS)}. One-thread 50-epoch model-only projections are recorded in the benchmark JSON; they exclude cold-start/event construction and native tracing. The earlier formal A2 task wall time was 334.8 s because it included data preparation, Python audit and native tracing.

Thread microbenchmark medians were {profile['thread_count_benchmark']}. The machine has 8 physical/16 logical CPU threads and about 16 GB RAM; the PyTorch environment defaults to 8 intra-op and 8 inter-op threads, while the released campaign forces one thread. A process-level 2/4-worker test was not run because it would create competing scientific output paths; no concurrency increase is released.

Initial forward, loss, gradients and optimizer inputs were exactly equal for paired A2-A5 checks. No replacement implementation has been frozen: caching B0 or feature patches requires a separate audited path and a fresh multi-step/checkpoint equivalence proof. Existing A0-A2 scientific outputs were not modified.

Recommended formal execution remains paused, with one process until a process-level benchmark and optimized implementation are separately approved. No server is required by this analysis, but the current serial CPU budget is long; remaining-task planning estimates are in the benchmark JSON.
''',encoding='utf-8')
 print(json.dumps({'status':'PASS','profile':str(OUT/(P+'_PROFILE.json')),'equivalence':eq},indent=2))
if __name__=='__main__':main()
