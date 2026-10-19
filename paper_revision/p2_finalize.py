"""Measurement-based conditional P2 budget and fail-closed release decision."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
OUT=Path(__file__).resolve().parents[1]/'outputs/revision_mtcomm_2026';ROOT=OUT.parents[1];P='R1_RECONSTITUTION_P2_'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8-sig'))
def write(n,v):(OUT/(P+n+'.json')).write_text(json.dumps({'identity':'REVISION_REPLICATION',**v},indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    runner=read(P+'RUNNER_EQUIVALENCE.json');base=read(P+'EXTERNAL_BASELINE_AUDIT.json');conc=read(P+'CONCURRENCY.json')
    inputs=read(P+'INPUT_AUDIT.json');p1=read('R1_RECONSTITUTION_P1_EXPERIMENT_BUDGET.json')['plans']['C']
    rates={}
    for v in ['B0','BPMR']:
        cases=[r for r in runner['cases'] if r['variant']==v]
        rates[v]={k:sum(r[k+'_seconds'] for r in cases)/sum(r['steps'] for r in cases) for k in ['reference','optimized']}
    rows=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_REVIEWER_MATRIX.csv').open(encoding='utf-8-sig')))
    rows=[r for r in rows if r['plan']=='C' and r['dedup_state_key']]
    data=read('R1_RECONSTITUTION_P1_DATASET_AUDIT.json');totals={'reference':0.,'optimized':0.};updates=0
    for r in rows:
        roi=next(x for x in data['rois'] if x['material']==r['material'] and str(x['roi'])==r['roi'])
        n=int(r['sweeps'])*roi['observations'][r['local_obs_um']]['updates_per_sweep'];updates+=n
        v='B0' if r['model'] in ['B0_SOURCE','A0','A1'] else 'BPMR'
        for mode in totals:totals[mode]+=n*rates[v][mode]/3600
    eval_times=[];s=np.random.default_rng(883).integers(0,4,(48,48,48),dtype=np.uint8)
    for _ in range(4):
        t=time.perf_counter()
        for phase in range(4):
            binary=s==phase;components,num=label(binary)
            spectrum=np.fft.fftn(binary.astype(float));corr=np.fft.ifftn(spectrum.conj()*spectrum).real
            counts=np.bincount(components.ravel());assert corr.shape==s.shape and num>=0 and counts.sum()==s.size
        eval_times.append(time.perf_counter()-t)
    eval_per_state=statistics.median(eval_times[1:]);evaluation_proxy_hours=eval_per_state*284*5/3600
    trainrows=list(csv.DictReader((OUT/'R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv').open(encoding='utf-8-sig')))
    lp=OUT/next(r['labels_path'] for r in trainrows if r['role']=='train');loads=[]
    for _ in range(7):
        t=time.perf_counter();lp.read_bytes();loads.append(time.perf_counter()-t)
    baseline_move16=statistics.median(r['median_move_seconds'] for r in base['runs'] if r['size']==16)
    baseline_hours=baseline_move16*27*10000*12/3600
    loading_hours=284*.02/3600+720*100*statistics.median(loads)/3600
    # Do not count tiny-kernel process speedup as demonstrated full worker throughput.
    total_serial=p1['training_hours']+totals['optimized']+baseline_hours+evaluation_proxy_hours+loading_hours
    budget=dict(classification='LOCAL_CPU_CONDITIONALLY_FEASIBLE',neural_trajectories=284,total_voxel_updates=updates,
        rates_seconds_per_update=rates,training_hours_prior_synthetic=p1['training_hours'],
        reconstruction_reference_hours=totals['reference'],reconstruction_optimized_trace_only_hours=totals['optimized'],
        external_baseline_hours_extrapolated=baseline_hours,external_basis='Median16cube move cost x27 volume scale x10000moves x12cases; not measured48cube cost, mixing or fair production budget.',
        evaluation_proxy_hours=evaluation_proxy_hours,evaluation_proxy='48cube4phase connected-components+FFT TPCF,4repeats; excludes chord/Euler/plots/real evaluator;5savedstates pertrajectory assumed.',
        evaluation_seconds_per_state=eval_per_state,loading_checkpoint_hours_assumed=loading_hours,
        loading_basis='7warm plane reads plus P1~20msbankload; assumed100plane loads peradaptation; no cold disk dataset staging measurement.',
        known_plus_explicitly_assumed_total_cpu_hours=total_serial,
        reserve_evaluation_visualization_hours=24,reserve_verification_restart_fraction=.30,
        elapsed_serial_hours_with_reserves=(total_serial+24)*1.30,
        elapsed_safe_concurrency_current='Only1worker extrapolation defensible for complete pipeline; intended isolated worker not launched.',
        measured_inference_worker_concurrency=conc,
        two_worker_hypothetical_hours_lower_bound=total_serial/conc['results'][1]['active_throughput_speedup_vs1'],
        concurrency_limit='2worker figure is kernel-only optimistic bound, not committed elapsed estimate;4workers <1.5GBsampledRSS but full pipeline scaling unverified.',
        deadline='2026-11-07 Asia/Shanghai; budget window <=456h from2026-10-19, reserve calendar time for writing and validation.',
        actual_hardware='AMD Ryzen9 6900HX8physical16logical,16GB; CPU torch, no tested GPU implementation.',
        cloud_recommendation='No cloud justified by conditional PlanC known workload; environment isolation and baseline validation precede hardware choice.',
        uncertainty='No complete-runtime certification. Training/finalmetrics/comparator mix of measured kernels and explicit projections; failed G1/G2/G3 prevent release.')
    write('RESOURCE_BUDGET',budget)
    write('OPTIMIZATION_PROFILE',dict(rates=rates,speedup={v:rates[v]['reference']/rates[v]['optimized'] for v in rates},
        basis='96prespecified synthetic2sweep4cube cases using12distinct models and reference encoder/fusion; trace-only optimized bank; scientific isolated executable not tested.',
        concurrency=conc,peak_runner_sampled_rss_bytes=max(r['sampled_rss_bytes'] for r in runner['cases']),
        original_48cube_measurements_reference='R1_RECONSTITUTION_P1_RECONSTRUCTION_PROFILE.json',
        full48cube_timing_not_measured=True))
    protocol=dict(identity='REVISION_REPLICATION',status='SPECIFIED_NOT_EXECUTED',design='A',
        seeds=[1701,1702,1703],max_events_per_model=100000,event_order='sort by source,orientation,index,row,column then deterministic seed-bound cap, hashbeforefit; same cap/events/orderforallbranches',
        event_support='9x9fullsquare erosion of support; no periodic train wrap',
        source_B0=dict(epochs=50,primary='epoch50',loss='MSE on normalized categorical probability versus onehot; explicit revision deviation from historical raw sigmoidMSE'),
        branches='A0/A1 allB0 trainable;A2-A5 frozencommonB0; same source checkpoint hash; variants from revision_protocol.json',
        optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0),batch_size=4096,shuffle=False,
        continuation_epochs=50,primary='epoch50,noearlystopping',secondary='bestvalidationonly,samecriterionperobjective,no testselection',
        initialization='same commonB0 for each paired unit; residualhead zero,gatebias-2.197224577, seedfixed',
        implementation_pending='No training entrypoint/final event manifest exists; must implement and verify source loss normalization/target semantics before release. Spec alone insufficient.')
    write('TRAINING_CONTRACT',protocol)
    gaps=['G1: no actual restricted scientific identity configured/tested; native probes read source/truth and2alternativecopies.',
          'G2:96synthetic cases pass but complete original runner/intended isolated executable and scientific checkpoint semantics unverified.',
          'G3:8non-test prototype runs establish clamp-preserving mechanics only; allowed-information parity,48cube runtime/mixing/stopbudget unvalidated.',
          'Training contract specified, event construction/runner not implemented; source MSE probability normalization is explicit revision choice, requires executable check.',
          'Reviewer C3,C5,C6,C7 remain partial; same Ketton specimen, restricted longhorizon/cross replication and unvalidated comparator.']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper_revision').glob('p2_*.py')}
    for relative in ['paper_revision/p1_profile.py','paper_revision/models_revision.py','src/bpmr_net/models.py','src/bpmr_net/features.py']:
        hashes[relative]=sha(ROOT/relative)
    historical=Path('C:/Users/admin/Documents/博士申请论文')
    for relative in ['r2_lite_3d_pilot.py','step65b_condition_cpdf_predictor.py','fu_style_cpdf_runtime.py','.remote_src/fu_style_senn_core.py']:
        hashes[str(historical/relative)]=sha(historical/relative)
    gates=dict(G1=dict(status='FAIL',reason=gaps[0]),G2=dict(status='FAIL',fixture_pass=runner['fixture_pass'],cases=len(runner['cases']),reason=gaps[1]),
        G3=dict(status='FAIL',prototype_runs=len(base['runs']),reason=gaps[2]),registry='PASS',
        training_contract='SPECIFIED_NOT_EXECUTABLE_VERIFIED',SCIENTIFIC_CAMPAIGN_READY='NO',implementation_hashes=hashes)
    write('RELEASE_GATES',gates)
    write('FINAL',dict(result='REVISION_R1_RECONSTITUTION_P2=PARTIAL',G1='FAIL',G2='FAIL',G3='FAIL',registry='PASS',trajectory_manifest='284uniquePASS',
        SCIENTIFIC_CAMPAIGN_READY='NO',runtime=rates,cpu_classification='LOCAL_CPU_CONDITIONALLY_FEASIBLE',
        estimated_cpu_hours=total_serial,estimated_serial_wall_with_reserves_hours=(total_serial+24)*1.30,
        blockers=gaps,scientific_model_fits=0,scientific_trajectories=0,source_permissions_modified=False,
        path_to_completion='Non-destructive dedicated VM with onlyrole-mounted allowed files and evaluator separation, or elevatedWindows role/AppContainer provisioning; validate exact workers then baseline and runner atproduction48cube scale.',implementation_hashes=hashes))
    summary=f'''# REVISION_REPLICATION P2

REVISION_R1_RECONSTITUTION_P2=PARTIAL
SCIENTIFIC_CAMPAIGN_READY=NO

Material Passport: MTCOMM-D-26-05155; P2 release audit; source: P1frozenregistry, native access probes,96synthetic runner comparisons,8non-test baseline prototypes, independent CPU-worker benchmarks. No model training or scientific reconstruction. Historical implementations unchanged.

G1 FAIL. Exact current Python identity remains admin medium-integrity. Positive read probes pass, forbidden source/test reads also succeed. Two rawcopies in recyclebin are accessible. Original SDDL saved, no ACLmodified. No tested dedicated worker identity/VM exists; separate roots and readonly attributes do not secure labels. Evaluation cannot yet be certified unable to modify model/config roots. Provisioning needs an actual enforceable OS boundary; nominal loader wrappers are insufficient.

Registry PASS:2246uniqueIDs and payloadpaths, all registeredlabel/support hashes checked, rolebounds checked.6ROIsx3regimes observation payloads contain only legally observed values; hidden positions sentinel255, match truth only atmask. Source/fit/test separated bycoordinates,128guard and radius4rules. No fit arrays generated. Raw0excluded, foursemanticphases1..4canonical0..3. Direct overlapchecks do not establish independent-material generalization.

284manifest PASS:252core=6ROIsx2masksx7modelsx3pairedseeds;16additionaldensitycross and16sensitivity. Coarse250/1000masks identical so1000local duplicates omitted. Three trainingseeds and3rolloutseeds are paired rather than fully crossed; checkpointassets do not yet exist. G25/G250/G1000 represented. Statesreuse initialization across modelbranches; sweepscontinue rather than rerun. DifferentROIs are correlated subvolumes, not specimens.

G2 FAIL for productionrelease, although all96prespecified fixtures passed numerical/state/RNG/clamp/diskresume checks. Cases cover family-labelledWF2/GI fixtures,2maskdensities,2rolloutseeds,3orders,2compositions,partial support, minoritylabels and low/highentropy modelbiases. These are synthetic4cube fixtures under currentidentity, not realWF2/GI or48cube complete scientificrunner. Reference ASTfunctions untouched; trace-onlybank, no dependentvoxelbatching. Expandedtests do not generalize proof to allcheckpoints or vector-featurepath.

Measured B0 reference {rates['B0']['reference']*1000:.3f}ms/update, trace {rates['B0']['optimized']*1000:.3f}ms; BPMR reference {rates['BPMR']['reference']*1000:.3f}ms, trace {rates['BPMR']['optimized']*1000:.3f}ms. Independent1/2/4worker activebankthroughputs measured, not assumed proportionaltocores. Startup andactive intervals recorded separately. Completepipeline safeconcurrency remains1until isolatedworker checks.

G3 FAIL for fairproductioncomparison. NewApache2.0repository-native Yeong–Torquato-style4phase hardclamp prototype copies noexternalcode. Publication verified via https://doi.org/10.1103/PhysRevE.58.224 (publisherabstract/metadata only); exactpublishedbinaryimplementation notclaimed. Only2registeredG250trainplanes provide statistics;3Dinputs synthetic.8runs at8/16cube64swaps preserveclamps andphasecounts. Fullallowedtrainingstatistics,48cubeincrementalimplementation/mixing/budget validation pending. No superiorityresultgenerated.

PlanC neural reconstruction projection: reference{totals['reference']:.1f}h; trace{totals['optimized']:.1f}h. Trainingprior{p1['training_hours']:.1f}h. Comparatorvolume-scaled estimate{baseline_hours:.2f}h, notmeasured48cube. Evaluationkernel proxy{evaluation_proxy_hours:.2f}h excludesmanymetrics/figures. Combinedmeasured/assumed CPUestimate{total_serial:.1f}h; serialcalendar estimate with24hadditionalmetrics/figures and30percentrestartverification reserve{(total_serial+24)*1.30:.1f}h. No guarantee ofdeadlinecompletion whilegatesopen. LOCAL_CPU_CONDITIONALLY_FEASIBLE; nocloudrentaljustifiedyet.125hP1neverincludedfullbaseline/evaluation.

Remaininggaps:
'''+'\n'.join('- '+g for g in gaps)+'\n\nSTOP P2. No campaignauthorization inferred.\n'
    # Prefer a readable release summary; detailed evidence remains in JSON/CSV.
    summary=f'''# REVISION_REPLICATION P2

REVISION_R1_RECONSTITUTION_P2=PARTIAL

SCIENTIFIC_CAMPAIGN_READY=NO

Material Passport: manuscript MTCOMM-D-26-05155; stage P2. Evidence consists of frozen P1 artifacts, native access tests, synthetic sequential-runner comparisons, non-test baseline prototypes and CPU measurements. No model fitting or scientific reconstruction was performed.

G1 isolation: FAIL. Current Python runs under the medium-integrity admin identity. It can read permitted train/validation/observation files, but can also read prohibited source/test truth and two raw copies in the recycle bin. Original ACLs were recorded and remain unchanged. No dedicated execution identity or isolated VM was provisioned. Role-specific write permissions, evaluation write denial and alternate-path bypass denial remain untested. These require an actual enforceable OS boundary, not a loader wrapper.

Registry integrity: PASS. All 2,246 plane identities and payload paths are unique; label/support hashes and registered bounds agree. Each of the 18 observation files was compared with its corresponding truth: observed labels agree, and hidden positions contain sentinel 255. Train/validation arrays contain only their declared spatial regions. Guards are 128 voxels; feature radius is 4. Full-square support erosion is required before events are built. No fit events were generated. Raw label 0 stays outside defined four-phase support.

Trajectory manifest: 284 unique neural trajectories. The core has 252 = 6 ROIs × 2 distinct local masks × 7 model configurations × 3 paired seeds. There are 16 additional global-density comparisons and 16 initialization/order comparisons. G250 core cases are reused. Local 250/1000 um masks are identical and are not run twice. Saved sweeps continue with RNG state rather than restarting. Three training seeds and three rollout seeds are paired, not fully crossed. Checkpoints do not yet exist. ROIs are correlated subvolumes of one Ketton specimen.

G2 production equivalence: FAIL, with 96 fixture comparisons passing. Tests cover two flow-state-labelled synthetic families, two mask densities, two seeds, three update orders, two phase compositions, invalid support, minority phases, entropy extremes and actual disk resume. Probabilities, saved states, RNG identities, order and clamping matched in the tested fixtures. Reference functions were extracted without editing their bodies. The optimized path traces the 12-model bank and retains reference encoding; no dependent voxel updates are batched. Tests use 4³ synthetic states under the current identity, not the intended isolated full scientific worker. They do not prove all-input or all-checkpoint equivalence.

Measured update times: B0 reference {rates['B0']['reference']*1000:.3f} ms, trace {rates['B0']['optimized']*1000:.3f} ms; BPMR reference {rates['BPMR']['reference']*1000:.3f} ms, trace {rates['BPMR']['optimized']*1000:.3f} ms. Independent 1/2/4-worker forward-bank throughput and sampled RAM were measured. Those short kernels do not establish complete-pipeline scaling; the budget uses one worker conservatively.

G3 fair baseline: FAIL pending production validation. A new Apache-2.0 repository-native Yeong–Torquato-style four-phase hard-clamp prototype completed eight 8³/16³, 64-move runs. Only two registered G250 training planes supplied statistics; 3D inputs were synthetic. Clamps and phase counts were preserved. No third-party implementation was copied. Publication metadata and abstract were verified at https://doi.org/10.1103/PhysRevE.58.224. This is explicitly an adaptation, not the original binary implementation. Pooling the exact allowed training information, 48³ cost, mixing and a fair stopping budget remain unvalidated.

Plan C budget: neural reconstruction {totals['reference']:.1f} reference CPU hours or {totals['optimized']:.1f} trace CPU hours; prior synthetic training estimate {p1['training_hours']:.1f} hours. Comparator estimate {baseline_hours:.2f} hours uses explicit volume scaling, not measured 48³ performance. The evaluation proxy is {evaluation_proxy_hours:.2f} hours and excludes several metrics and figures. Combined measured/assumed work is {total_serial:.1f} CPU hours. Adding 24 hours for remaining evaluation/visualization and 30% verification/restart reserve gives {((total_serial+24)*1.30):.1f} serial elapsed hours. These are conditional estimates, not a completion guarantee.

LOCAL_CPU_CONDITIONALLY_FEASIBLE. No cloud rental is justified by the current Plan C measurements. Hardware choice does not resolve isolation or scientific comparator validation. Reviewer C3, C5, C6 and C7 retain the documented material-scope, replication-breadth and comparator gaps.

Remaining release gates:
'''+'\n'.join('- '+g for g in gaps)+'\n\nSTOP P2. No scientific campaign is authorized or started.\n'
    (OUT/(P+'SUMMARY.md')).write_text(summary,encoding='utf-8')
    print(json.dumps({'cpu_hours':total_serial,'serial_with_reserve_hours':(total_serial+24)*1.3,'reconstruction':totals,'ready':False}))
if __name__=='__main__':main()
