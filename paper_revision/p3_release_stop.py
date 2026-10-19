"""P3 stop at user-authorized administrative boundary; no scientific jobs."""
import ctypes
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/revision_mtcomm_2026'
PREFIX='R1_RELEASE_FIX_P3_'
COMMANDS=[]

def run(args):
    result=subprocess.run(args,capture_output=True,timeout=30)
    record=dict(command=args,returncode=result.returncode,stdout=result.stdout.decode('utf-8',errors='replace'),stderr=result.stderr.decode('utf-8',errors='replace'))
    COMMANDS.append(record)
    return record

def ps(command):return run(['powershell.exe','-NoProfile','-Command',command])
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):
    (OUT/(PREFIX+name+'.json')).write_text(json.dumps({'identity':'REVISION_REPLICATION',**value},indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    deadline=datetime(2026,11,7,0,0,tzinfo=ZoneInfo('Asia/Shanghai'))
    head=run(['git','rev-parse','HEAD'])['stdout'].strip()
    branch=run(['git','branch','--show-current'])['stdout'].strip()
    assert branch=='revision/mtcomm-major-2026'
    identity=run(['whoami','/all'])
    environment=ps("$os=Get-ItemProperty 'HKLM:/SOFTWARE/Microsoft/Windows NT/CurrentVersion';$cs=Get-CimInstance Win32_ComputerSystem;$cpu=Get-CimInstance Win32_Processor;[pscustomobject]@{EditionID=$os.EditionID;ProductName=$os.ProductName;Build=$os.CurrentBuild;HypervisorPresent=$cs.HypervisorPresent;RAM=$cs.TotalPhysicalMemory;CPU=$cpu.Name;VirtualizationFirmwareEnabled=$cpu.VirtualizationFirmwareEnabled;SLAT=$cpu.SecondLevelAddressTranslationExtensions;Tools=@(Get-Command WindowsSandbox.exe,docker.exe,vmrun.exe,VBoxManage.exe -ErrorAction SilentlyContinue|Select-Object Name,Source)}|ConvertTo-Json -Depth 4")
    elevated=bool(ctypes.windll.shell32.IsUserAnAdmin())
    assert not elevated, 'New elevated context: reassess actual available boundary before stop report'
    prior=read('R1_RECONSTITUTION_P2_RESOURCE_BUDGET.json')
    evidence_files=['R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv','R1_RECONSTITUTION_P1_DATASET_AUDIT.json',
        'R1_RECONSTITUTION_P2_RELEASE_GATES.json','R1_RECONSTITUTION_P2_RUNNER_EQUIVALENCE.json',
        'R1_RECONSTITUTION_P2_EXTERNAL_BASELINE_AUDIT.json','R1_RECONSTITUTION_P2_RESOURCE_BUDGET.json',
        'R1_RECONSTITUTION_P2_TRAJECTORY_MATRIX_AUDIT.csv']
    evidence={name:sha(OUT/name) for name in evidence_files}
    implementation_files=['src/bpmr_net/models.py','src/bpmr_net/features.py','paper_revision/models_revision.py','paper_revision/p2_runner.py','paper_revision/p1_profile.py','paper_revision/p2_baseline.py','paper_revision/p3_release_stop.py']
    implementations={name:sha(ROOT/name) for name in implementation_files}
    instruction='If a dedicated environment requires user-level Windows administrative approval, installation, account creation or access changes, stop at that operation and report the minimum manual action needed.'
    remedy=dict(minimum_manual_action='Install a local VM runtime compatible with Windows Home using Windows administrator approval; then report the installed runtime. No dataset permission change, edition upgrade or cloud rental is requested.',
        example='VirtualBox base runtime; installer/version/license must be checked before installation. No Extension Pack is required for this workflow.',
        alternative='User-provided existing isolated VM with no host drive shares; assess before use.',
        after_install='Next agent stage creates a dedicated guest, stages only frozen permitted train/validation/observations, keeps source/truth/backups outside guest, and verifies exact worker access. No training on installation alone.',
        guest_proposal=dict(vcpus=4,RAM_GiB=8,virtual_disk_GiB=30,network='disable after dependency setup',
            shares='No host home/drive/parent shares; no shared clipboard/drag-drop; input delivered through readonly isolated staging; distinct writable output disk.',
            evaluation='Separate context, receives completed outputs and test truth only after reconstruction; no write access to guest training/config/checkpoint stores.'),
        no_permission_mutations='Original source/test/backups remain untouched; dedicated staging only.')
    write('ISOLATION',dict(G1='FAIL',status='STOPPED_BEFORE_ADMINISTRATIVE_INSTALLATION',
        process_identity=identity,environment=environment,python_executable=sys.executable,python_version=sys.version,
        elevated=elevated,Windows_Sandbox='Home/Core unsupported per Microsoft; not an available local solution.',
        official_reference='https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/',
        existing_P2_access_results='Retained without repeating: permitted reads succeed; source/test/2backup reads also succeed.',
        new_identity_access_tests='NOT_RUN: no new isolated identity exists; no same-user rerun counted as isolation.',
        admin_accounts='Application-managed CodexSandbox accounts not repurposed.',source_ACLs_modified=False,
        stopped_under_user_instruction=instruction,remedy=remedy,commands=COMMANDS.copy(),evidence_sha256=evidence))
    write('PRODUCTION_EQUIVALENCE',dict(G2='FAIL',status='NOT_RUN_REQUIRED_ISOLATED_ENVIRONMENT_UNAVAILABLE',
        new_tests=0,previous='P2 96 synthetic fixtures passed; not production equivalence.',
        minimum_pending='Real registered train/validation inputs, fixed12model bank, at least one complete48^3 throughsweep3 per B0/BPMR family, reference+optimized+diskresume under exact isolated worker.',
        reference_authority='Unmodified reference sampler retained; optimized path not released.',
        failure_localization='Required on any future difference: first divergent update,conditional probabilities,RNG/order/state hashes.',
        elapsed_runtime_new=None,peak_RAM_new=None,implementation_sha256=implementations,evidence_sha256=evidence))
    write('BASELINE_FEASIBILITY',dict(G3='FAIL',status='NOT_RERUN_AFTER_ADMINISTRATIVE_STOP',
        previous='P2 eight8^3/16^3 train-derived synthetic hard-clamp prototypes; no complete48^3 feasibility proof.',
        pending='Verify full objective/procedure/allowed-information parity, validation-derivedtargets,48^3 workload and convergence/stopbudget; if infeasible assessonealternative before declaring unavailable.',
        truth_statistics_prohibited=['target phase fractions','target correlation','target topology','target initialization','stopping'],
        publication='10.1103/PhysRevE.58.224; prior publisher abstract verification retained, not fulltext algorithmverification.',
        new_objective_evaluations=0,new_reconstruction_runtime=None,implementation_sha256=implementations))
    remaining=(deadline-now).total_seconds()/3600
    reference_cpu=prior['training_hours_prior_synthetic']+prior['reconstruction_reference_hours']+prior['external_baseline_hours_extrapolated']+prior['evaluation_proxy_hours']+prior['loading_checkpoint_hours_assumed']
    reference_wall=(reference_cpu+24)*1.3
    budget=dict(local_CPU='CONDITIONAL',SERVER_REQUIRED='UNDETERMINED',EXTENSION_ADVISABLE='YES',
        timestamp=now.isoformat(),deadline_date='2026-11-07 Asia/Shanghai',deadline_budget_boundary=deadline.isoformat(),
        deadline_boundary_note='Conservative beginning of due date; journal time-of-day unspecified.',
        remaining_hours_to_boundary=remaining,
        no_new_production_measurements=True,prior_optimized_cpu_hours=prior['known_plus_explicitly_assumed_total_cpu_hours'],
        prior_optimized_wall_hours_with_reserves=prior['elapsed_serial_hours_with_reserves'],
        reference_authority_cpu_hours_conditional=reference_cpu,reference_serial_wall_hours_with_reserves=reference_wall,
        components=dict(training_prior_hours=prior['training_hours_prior_synthetic'],neural_reference_hours=prior['reconstruction_reference_hours'],
            neural_optimized_unreleased_hours=prior['reconstruction_optimized_trace_only_hours'],
            baseline_unvalidated_extrapolation_hours=prior['external_baseline_hours_extrapolated'],evaluation_partial_proxy_hours=prior['evaluation_proxy_hours'],
            loading_assumed_hours=prior['loading_checkpoint_hours_assumed'],remaining_evaluation_visualization_reserve_hours=24,restart_verification_fraction=.3),
        writing_review_buffer_hours=72,
        maximum_remaining_gate_closure_hours_under_reference_budget=remaining-reference_wall-72,
        critical_path=['User administrative VM installation','Guest environment/dependency staging','G1 exact positive/negative role tests','Finalize executablecommonB0training/events/checkpointlogs','G2 reference/optimized48^3 validation','G3 validationbaseline fairness/mixing/runtime','Recompute guestbudget and release','Separate scientificcampaignauthorization','Scientificcampaign','Evaluation,responsewriting,review'],
        gate_closure_duration='UNKNOWN: installation,48cubeverification,baselinevalidation unmeasured. Completiondate cannot be certified.',
        extension_reason='Reference-authority projection plus72hwriting buffer leaves very limited/unmeasured gate-closure margin; guest slowdown unmeasured. Advise extension, do not contacteditor automatically.',
        actual_safe_production_concurrency=0,concurrency_note='No approved isolated worker. P2kernel1/2/4scaling not imported asproductionwall speedup.',
        cloud='No paid service initiated. Isolation, not established compute capacity, is immediateblocker; choose cloudonlyaftervalidguestbenchmarks.')
    write('RESOURCE_BUDGET',budget)
    decision=dict(result='REVISION_R1_RELEASE_FIX_P3=PARTIAL',G1='FAIL',G2='FAIL',G3='FAIL',TRAINING_READY='NO',RECONSTRUCTION_READY='NO',FULL_CAMPAIGN_READY='NO',
        LOCAL_CPU_FEASIBLE='CONDITIONAL',SERVER_REQUIRED='UNDETERMINED',EXTENSION_ADVISABLE='YES',
        stop_reason=instruction,manual_action=remedy,training_started=False,scientific_reconstructions=0,
        registry='P2PASSretained;noauditrepeat',trajectory_manifest='284uniqueP2PASSretained;noexpansion',
        remaining_blockers=['No isolated execution environment/identity','G2 production48^3equivalence absent','G3 fair48^3baseline validation absent','Executable trainingevents/checkpoint/log contract not verified'],
        next_executable_stage='P3_ISOLATED_ENVIRONMENT_BOOTSTRAP after user installsVMruntime; no modeltrainingauthorization implied.',
        base_commit=head,report_commit_note='This report is committed in the subsequent revision-only commit; obtain itsSHA via gitlog --all --grep ReleaseFixP3. Self-hashing commitSHA cannot be embedded in itself.',
        implementation_sha256=implementations,evidence_sha256=evidence,commands=COMMANDS.copy())
    write('RELEASE_DECISION',decision)
    summary=f'''# REVISION_REPLICATION — Release Fix P3

REVISION_R1_RELEASE_FIX_P3=PARTIAL

G1=FAIL · G2=FAIL · G3=FAIL

TRAINING_READY=NO · RECONSTRUCTION_READY=NO · FULL_CAMPAIGN_READY=NO

LOCAL_CPU_FEASIBLE=CONDITIONAL · SERVER_REQUIRED=UNDETERMINED · EXTENSION_ADVISABLE=YES

Material Passport: manuscript MTCOMM-D-26-05155; P3 environment/release stop. P2 registry and manifest evidence retained without repeating completed audits. No training, production reconstruction, permission change, account creation, installation or paid service occurred.

The machine is Windows Home/Core, with firmware virtualization available and no active hypervisor or discovered VM runtime. Python3.11.17 runs under the current medium-integrity admin identity, not an isolated worker. Windows Sandbox does not support Home edition: https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/ . Application-managed CodexSandbox accounts were not repurposed.

P3 explicitly says: “{instruction}” Work stops before that operation. Minimum manual action: install a Windows Home-compatible local VM runtime with Windows administrator approval, then report which runtime is installed. A VirtualBox base installation is a candidate; no Extension Pack, source ACL change, data deletion, Windows edition upgrade or server rental is requested. Installation does not authorize training.

After installation, the next executable stage is dedicated guest creation and isolated staging. Proposed resource allocation:4vCPUs,8GiBRAM,30GiBvirtualdisk, subject to guest measurements. Stage only frozen train/validation planes and permitted observations; no host home/drive shares, clipboard or drag-and-drop. Keep truth/source/backups outside the guest. Use a separate evaluation context unable to change training/checkpoints. Perform actual native positive/negative access tests under each exact worker identity before any scientific job.

G2 was not rerun outside its required environment. The96syntheticP2passes remain limited evidence. At least one complete48³trajectory throughsweep3 per B0/BPMR family, plus reference/optimized/resume comparisons on legal non-test validation inputs, remains required. The original reference sampler remains authoritative. G3 retains the small P2prototype only; fairallowedinformation,48³cost/mixing/stopbudget and an executablecomparator remain unresolved. Training event/checkpoint/log implementation also remains unverified.

No new runtime is claimed. The earlier optimized projection is131.4CPUhours and202.0elapsedhours with reserves, conditional on an unapprovedoptimizedrunner. Retaining reference authority yields approximately{reference_cpu:.1f}CPUhours and{reference_wall:.1f}serialelapsedhours with the same explicitreserves; comparator/evaluation costs remain incomplete. At{now.isoformat()}, about{remaining:.1f}hours remain to the conservative startof2026-11-07. Reserving72hours forwriting/review leaves approximately{remaining-reference_wall-72:.1f}hours for allremaininggates on the referenceprojection, beforeunknownguestoverhead. Therefore an extension is advisable. No editorcontact or extensionrequest was sent.

Estimated completion time is not established until installation and guest production benchmarks finish. Criticalpath: administrative installation → isolated staging/G1 → executabletrainingcontract and G2/G3validation → revisedbudget/release → separatecampaignauthorization → scientificexecution → evaluation/rebuttal/review. The284trajectorymatrix is unchanged.

Base commit:{head}. The reportcommitSHA is returned in the finalresponse and git history. All required implementation/manifest hashes, environmentcommands and measured-versus-inherited status are recorded in theJSONfiles.

STOP P3. Next: P3_ISOLATED_ENVIRONMENT_BOOTSTRAP after the manual VM-runtime installation. No formal scientific execution is authorized.
'''
    (OUT/(PREFIX+'SUMMARY.md')).write_text(summary,encoding='utf-8')
    print(json.dumps(dict(result='PARTIAL',reference_cpu_hours=reference_cpu,reference_elapsed_hours=reference_wall,remaining_hours=remaining,manual_action=remedy['minimum_manual_action'])))

if __name__=='__main__':main()
