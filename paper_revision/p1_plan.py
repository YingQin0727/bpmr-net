"""Reviewer-linked plans and transparent conditional budgets; no execution."""
import csv
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026';P='R1_RECONSTITUTION_P1_'
def read(name):return json.loads((OUT/(P+name+'.json')).read_text(encoding='utf-8'))
def main():
    data=read('DATASET_AUDIT');profile=read('RECONSTRUCTION_PROFILE');equiv=read('OPTIMIZATION_EQUIVALENCE');isolation=read('ISOLATION_TEST')
    speeds={}
    for kind in ['reference','trace_vector_features']:
        speeds[kind]={r['variant']:r['median_seconds_per_update'] for r in profile['profiles'] if r['implementation']==kind and r['threads']==1}
    p0=json.loads((OUT/'R1_RECONSTITUTION_P0_CPU_BENCHMARK.json').read_text(encoding='utf-8'))
    fit={r['variant']:r['median_step_seconds']*1250 for r in p0['optimizer_steps'] if r['batch']==4096}
    rows=[];unique={}
    def add(plan,category,roi,variant,global_um,local_um,train_seed,roll_seed,sweeps=10,order='forward',init='train_prior',comments=''):
        obs=roi['observations'][str(local_um)]
        key=(plan,roi['material'],roi['roi'],variant,global_um,obs['sha256'],train_seed,roll_seed,order,init)
        # A row is one maximum-length trajectory; sweeps3/10 resume, never rerun.
        if key in unique:
            old=unique[key];old['sweeps']=max(old['sweeps'],sweeps)
            old['reviewer_comments']=';'.join(sorted(set(old['reviewer_comments'].split(';')+comments.split(';'))))
            return
        row=dict(plan=plan,experiment=category,material=roi['material'],roi=roi['roi'],roi_origin_zyx=json.dumps(roi['origin_zyx']),
            training_seed=train_seed,reconstruction_seed=roll_seed,model=variant,global_fit_um=global_um,local_obs_um=local_um,
            observation_sha256=obs['sha256'],sweeps=sweeps,update_order=order,initialization=init,
            reviewer_comments=comments,metrics='phase_fraction;phase_extinction;TPCF;components;largest_component;Euler;chords;gate;selector_usage;XY_XZ_YZ',
            reuse='same saved sweeps0,1,3,5,10 answer fidelity/topology/visualization/stability; epoch50 predictions answer loss/gate/selector',
            updates_per_sweep=obs['updates_per_sweep'])
        rows.append(row);unique[key]=row
    comments='R1-C2;R2-C1;R2-C2;R2-C3;R2-C4;R2-C5;R2-C8;R2-C9;R2-C10'
    for plan in ['A','B']:
        for roi in data['rois']:
            for g in [25,250,1000]:
                for local in [25,250]: # 1000 duplicate local mask removed
                    for seed in [1701,1702,1703]:
                        for roll in [2701,2702,2703]:
                            for model in ['B0_SOURCE','A0','A1','A2','A3','A4','A5']:
                                add(plan,'core_crossed',roi,model,g,local,seed,roll,comments=comments+';R2-C6')
        for roi in [r for r in data['rois'] if r['roi']==0]:
            for model in ['A0','A3']:
                for seed in [1701,1702,1703]:
                    for roll in [2701,2702,2703]:
                        for order,init in [('reverse','train_prior'),('random_per_sweep','train_prior'),('forward','uniform'),('forward','dominant_train')]:
                            add(plan,'sensitivity',roi,model,250,250,seed,roll,order=order,init=init,comments='R2-C5' if order!='forward' else 'R2-C5;R2-C3')
    for roi in data['rois']:
        for local in [25,250]:
            for model in ['B0_SOURCE','A0','A1','A2','A3','A4','A5']:
                for seed,roll in [(1701,2701),(1702,2702),(1703,2703)]:
                    add('C','paired_core',roi,model,250,local,seed,roll,sweeps=10 if seed==1701 else 3,comments=comments)
    for roi in [r for r in data['rois'] if r['roi']==0]:
        for model in ['A0','A3']:
            for g in [25,250,1000]:
                for local in [25,250]:add('C','density_cross',roi,model,g,local,1701,2701,comments='R2-C6')
            for order,init in [('reverse','train_prior'),('random_per_sweep','train_prior'),('forward','uniform'),('forward','dominant_train')]:
                add('C','sensitivity',roi,model,250,250,1701,2701,order=order,init=init,comments='R2-C5')
    plans={}
    for plan in ['A','B','C']:
        selected=[r for r in rows if r['plan']==plan];kind='reference' if plan=='A' else 'trace_vector_features'
        for r in selected:
            speed=speeds[kind]['A0' if r['model'] in ['B0_SOURCE','A0','A1'] else 'A3']
            r['expected_marginal_cpu_hours']=r['sweeps']*r['updates_per_sweep']*speed/3600
            r['dedup_state_key']=hashlib.sha256(repr(tuple(r[k] for k in ['material','roi','model','global_fit_um','observation_sha256','training_seed','reconstruction_seed','update_order','initialization'])).encode()).hexdigest()
        assets=sorted(set((r['material'],r['global_fit_um'],r['training_seed']) for r in selected))
        new_b0=len(assets)*12;adapt=new_b0*6
        training=sum(fit.values())*new_b0+fit['A0']*new_b0
        recon=sum(r['expected_marginal_cpu_hours'] for r in selected)
        total=training/3600+recon
        plans[plan]=dict(name={'A':'Scientifically complete reference','B':'Optimized complete same cases','C':'Deadline contingency'}[plan],
            new_B0_runs=new_b0,A0_A5_adaptation_runs=adapt,heldout_model_evaluations=adapt+new_b0,
            trajectories=len(selected),total_voxel_updates=sum(r['sweeps']*r['updates_per_sweep'] for r in selected),
            measured_kernel=kind,training_hours=training/3600,reconstruction_hours=recon,
            known_total_cpu_hours=total,wall_hours_single_worker=total,
            wall_hours_two_workers_assuming_70percent_efficiency=total/1.4,
            peak_ram_estimate_gib=2.0,peak_ram_basis='Measured synthetic worker RSS <0.5GiB; reserve2GiB per worker plus role-staged memory-mapped files; no full source accessible.',
            omitted_costs='Comparator implementation runtime and evaluation/rendering not measured; budget reserve24h for evaluation and48h comparator cap, cannot count as validated complete total.',
            unknown_external_comparator_hours=None,deadline_window_hours=19*24,
            auxiliary_states=dict(dominant_phase_states=12,published_comparator_trajectories_planned=12 if plan=='C' else 36,
                                  neural_trajectory_count_excludes_comparator=True,comparator_sweeps='not applicable: annealing move budget requires validation'),
            fits_window_known_work=total+72<19*24,
            parallelism='Independent trajectories only; never voxel updates. 2 workers unmeasured efficiency assumption; do not rely on it for deadline.',
            critical_path='OS isolation -> release -> new common B0 -> adaptations -> evaluator-frozen selection -> trajectory sweeps/resume -> locked evaluation -> response writing',
            gpu='Not validated; sequential batch-one dependency limits benefit; no GPU speedup assumed.',
            independent_material_generalization=False)
    for r in rows:r.pop('updates_per_sweep')
    columns=list(rows[0])
    other_rows=[]
    for plan in ['A','B','C']:
        for category,comments,model,cost,reuse in [
            ('editorial','R1-C1','none',0,'Define WF2/GI on first use'),
            ('data_description','R2-C2','none',None,'Preparation-only source/slice/phase/mask visualization'),
            ('heldout_model_comparisons','R2-C1;R2-C8;R2-C9','B0_SOURCE;A0-A5',None,'Reuse same epoch50 probability and gate logs; validation-only selector threshold; test evaluator frozen'),
            ('dominant_phase_baseline','R2-C3;R1-C2','train-only modal phase',None,'Fill unobserved voxels; clamp all observations; one deterministic state per ROI/unique mask, no seed duplicates'),
            ('published_comparator','R2-C7','Yeong-Torquato four-phase hard-clamp adaptation',None,'Pending implementation; same unique masks/ROI, train-only statistics;3seeds full,1seed C')]:
            row={k:'' for k in columns};row.update(plan=plan,experiment=category,reviewer_comments=comments,model=model,
                expected_marginal_cpu_hours=cost,reuse=reuse,metrics='Same phase metrics and compute reporting; not a duplicate neural trajectory')
            other_rows.append(row)
    with (OUT/(P+'REVIEWER_MATRIX.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows+other_rows)
    coverage=[
        ('R1-C1','FULL_PLAN','Define abbreviations in text; no experiment'),
        ('R1-C2','FULL_PLAN','Shared fidelity/topology/slices and explicit limitations'),
        ('R2-C1','FULL_PLAN','All A0-A5 held-out comparisons, common B0, capacity caveat'),
        ('R2-C2','FULL_PLAN','Registered data descriptions/rendering; prep only'),
        ('R2-C3','PARTIAL','3 prespecified ROIs with differing composition, WF2 ROI1 oil absent; no performance-driven substitution; train-dominant baseline; no material independence'),
        ('R2-C4','FULL_PLAN','Shared per-phase topology and extinction; not effective-property validation'),
        ('R2-C5','PARTIAL_C','One long paired seed, other2 short; order/init oneROI/state. Finite horizon and sparse interaction coverage'),
        ('R2-C6','PARTIAL_C','Cross3 global x2 distinct local masks on oneROI/state and one paired seed; 250/1000 local identical'),
        ('R2-C7','PARTIAL','Publication-verified Yeong-Torquato stats baseline adaptation specified; executable conditioning fidelity/runtime unvalidated'),
        ('R2-C8','FULL_PLAN','Validation-only threshold sensitivity, test saturation reporting, exploratory status, reuse probability logs'),
        ('R2-C9','FULL_PLAN','Gate distributions versus entropy/margin/support; A3/A4; descriptive associations'),
        ('R2-C10','FULL_PLAN','Fixed central orthogonal slices from shared saved states')]
    comparator=dict(selected='Yeong–Torquato correlation-constrained simulated annealing, explicit four-phase/hard-clamp adaptation',
        publication='Yeong & Torquato (1998), Physical Review E58,224, DOI10.1103/PhysRevE.58.224',
        verified_url='https://doi.org/10.1103/PhysRevE.58.224',
        published_scope='3D reconstruction from 2D two-point probability and lineal-path statistics; retrieved abstract does not establish multiphase hard-clamp implementation.',
        implementation_contract='New revision adapter, not claim of original author code: fit per-phase target two-point and lineal-path descriptors only from registered train planes. Initialize train-prior categorical labels; clamp identical observation mask/values. Swap two unobserved valid voxels; observations never proposed. Objective weights and annealing schedule fixed using validation only. No target truth statistics or test phase fractions.',
        fair_constraints='Same ROI grid, phase semantics, observations, valid support, observation-excluded evaluation; report compute and descriptor-target mismatch rather than assert strongest published implementation.',
        status='METHOD_SELECTED_IMPLEMENTATION_NOT_VALIDATED',
        rejected='Unconditional GAN not automatically fair. SNESIM/MPSlib hard-data interface verified, but requires representative3D training image absent under sparse-plane information parity; no dense source training image permitted.',
        alternative_docs='https://mpslib.readthedocs.io/en/latest/Algorithms/chapSnesim/mps_snesim.html')
    budget=dict(identity='REVISION_REPLICATION',plans=plans,selected='C_CONDITIONAL_NOT_READY',
        recommendation='C maximizes all-question coverage within local deadline using full A0-A5 controls, 3 spatial ROIs/state and3 paired short trajectories, one continued to10; reductions affect replication breadth explicitly. A/B too large serially.',
        training_assumptions='100000 events/model maximum;50epochs;ceil(100000/4096)=25 steps;50source epochs +50adaptation; actual events subject to support erosion; timing P0 synthetic full batch. Fixed caps before performance.',
        scalar_proxy_limit='A2/A4/A5 use A3 timing; A1 uses A0; arithmetic extrapolations not full trajectory timings. Training/comparator/evaluation costs uncertain.',
        prior_587day_comparison='P0 used4threads synthetic predictor and all local labels independently. P1 measured actual reference predictor, one thread, 12distinct models, full state encoding; dedup coarse local masks and share seed continuations. P1 budget is a different explicitly declared matrix, not claim of587day speedup.',
        coverage=[dict(comment=c,status=s,approach=a) for c,s,a in coverage],comparator=comparator,
        hardware=dict(local='Ryzen9 6900HX 8cores16threads,16GB;1 inference thread best; PlanC known kernels fit serial deadline estimate.',
            high_core='Independent trajectory parallelism possible, unmeasured scaling and memory contention; no numeric server advantage assumed.',
            GPU='No compatible measured GPU implementation; not recommended on evidence available.',server_required=False,
            conclusion='No server justified yet for C known-work budget; cannot certify full end-to-end feasibility until isolation/comparator are validated.',cost_estimate=None))
    (OUT/(P+'EXPERIMENT_BUDGET.json')).write_text(json.dumps(budget,indent=2),encoding='utf-8')
    blockers=['OS isolation negative test FAILED: same-user subprocess can read both source and test truth; readonly is not read denial.',
              'Reference harness dependencies explicitly rebound; complete native scientific worker/geometry/RNG-resume files not yet validated.',
              'Optimized predictor fixture equivalence is not universal proof; training-produced A0-A5 models require validation before use.',
              'Published-method adapter not implemented/validated/timed; complete budget has unknown comparator cost.',
              'No independent material generalization; PlanC long-horizon/global-local comparisons have restricted seed/ROI breadth.']
    final=dict(identity='REVISION_REPLICATION',classification='PARTIAL',result='REVISION_R1_RECONSTITUTION_P1=PARTIAL',
        dataset='2246 registered planes in quarantine; integrity/domain/bounds checked; not released',spatial_leakage='No direct role overlap; guards128 > radius4; raw0 excluded support; event creation must erode support.',
        isolation_enforced=False,planes=data['planes'],model_training_started=False,scientific_reconstruction_started=False,
        measured_update_seconds=speeds,fixture_equivalence=all(c['tolerance_pass'] and c['final_state_bitwise_equal'] for c in equiv['comparisons']),
        speedup={v:speeds['reference'][v]/speeds['trace_vector_features'][v] for v in ['A0','A3']},
        recommended_plan='C_CONDITIONAL',local_CPU='Known-work projection feasible; full end-to-end readiness unresolved',server_required='Not justified by measured evidence for C; no rental requested',
        blockers=blockers,next_stage_readiness='NOT_READY_FOR_CAMPAIGN. Close isolation and comparator/worker gates before any scientific fitting/reconstruction.')
    (OUT/(P+'FINAL.json')).write_text(json.dumps(final,indent=2),encoding='utf-8')
    summary=f'''# REVISION_REPLICATION P1

REVISION_R1_RECONSTITUTION_P1=PARTIAL

Material Passport: MTCOMM-D-26-05155; P1 preparation/profile/plan; source hashes and full-volume label counts; registered new planes; actual reference predictor forwards on synthetic states; no fitting or scientific reconstruction.

Dataset: {data['planes']} planes (1150train,916validation,180test slice registrations), 6separate48^3 truth cubes, 18observation artifacts. All hashes/domain/bounds checked. Raw source labels0..4: zero excluded as unassigned support; only1..4 map to canonical0..3. The3preselected spatial origins are [2300,800,200], [2700,800,800], [2100,200,800] ZYX, shared across states. Composition differs, including zero oil in WF2 ROI1; no replacement after inspection. Exact code hashes and coordinate/model-axis adapters registered. Data remain quarantined.

Train/validation/test z slabs have128-voxel guards. All radius4neighborhoods must be wholly supported within a split tile; historical ROI union+128 excluded. No direct cross-role overlap. Same-rock correlation remains. Test truth readonly attributes are set, but the native subprocess probe can read source/truth: OS isolation FAILED. No worker may execute scientific jobs.

Reference remains unmodified. Historical predictor/encoder/fusion functions are AST-extracted with disclosed constants; XYZ state/plane conventions explicitly bound. Original models already loaded once per trajectory. Costs include full-state integer conversion in encoder,12forwards,numpy fusion and categorical draws. One-thread timings: B0 {speeds['reference']['A0']*1000:.3f}ms/update; A3 {speeds['reference']['A3']*1000:.3f}ms/update. Trace+vector-feature: {speeds['trace_vector_features']['A0']*1000:.3f} and {speeds['trace_vector_features']['A3']*1000:.3f}ms. This is actual network execution on non-scientific inputs, not accuracy evidence. Small complete sweep measurements and component/load timings are in profile/equivalence JSON.

Tested fixtures preserved conditional probabilities bitwise, seeded final states, clamp/order and exact in-memory resumed RNG/state. No general all-input equivalence or complete scientific worker claim. No voxel-parallel/synchronous sampler introduced. PlanC must validate real checkpoint banks before using optimized path.

Budget known-work serial CPU hours (comparator/evaluation excluded): A={plans['A']['known_total_cpu_hours']:.1f}; B={plans['B']['known_total_cpu_hours']:.1f}; C={plans['C']['known_total_cpu_hours']:.1f}. PlanC adds72hours provisional comparator/evaluation reserve; actual comparator runtime unknown. Three3sweep seed pairs with one extended10, all A0-A5,3ROIs/state,2distinct local masks; restricted global cross and one-factor sensitivity reuse assets. Coarse250/1000local masks are identical; no duplicate trajectories. Baseline and gate/selector/local predictions share logs; no redundant sweep3restart. Details and marginal costs in reviewer matrix.

Recommend C conditionally for local CPU; no server currently justified. Higher-core CPU scaling/GPU acceleration unmeasured. A/B full matrices do not fit serial deadline. C partially answers R2C3(material scope),C5(long-horizon breadth),C6(cross breadth),C7(unvalidated comparator); other reviewer questions have explicit full planned evidence, not completed findings. Full twelve-item coverage in budget.

Comparator: publication-verified Yeong–Torquato1998, PRE58,224 (https://doi.org/10.1103/PhysRevE.58.224). Proposed multiphase hard-clamp adapter is explicitly a new implementation, not claimed original support. Its validation is a blocker. SNESIM's3Dtraining-image requirement would violate sparse information parity if dense source were supplied; unconditional GAN not selected.

Remaining blockers:
'''+'\n'.join('- '+b for b in blockers)+'\n\nSTOP P1. No full campaign launch.\n'
    (OUT/(P+'SUMMARY.md')).write_text(summary,encoding='utf-8')
    print(json.dumps({'plans':{k:{a:v[a] for a in ['new_B0_runs','A0_A5_adaptation_runs','trajectories','known_total_cpu_hours']} for k,v in plans.items()},'result':'PARTIAL'}))

if __name__=='__main__':main()
