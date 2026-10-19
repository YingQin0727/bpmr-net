"""Generate evidence-linked REVISION_REPLICATION P0 reports; no pixel decoding."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/revision_mtcomm_2026'
PREFIX = 'R1_RECONSTITUTION_P0_'
HIST = Path('C:/Users/admin/Documents/博士申请论文/factorial_v2_inputs')
OFFICIAL = Path('D:/Paper2/paper2_stage02a_source_recovery')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    value = {'identity': 'REVISION_REPLICATION', **value}
    (OUT / (PREFIX + name + '.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    registry = HIST / 'frozen_target_roi_registry_v2.json'
    historical = json.loads(registry.read_text(encoding='utf-8'))
    sources = []
    for material in historical['materials']:
        path = OFFICIAL / 'official_download' / Path(material['source_file']).name
        actual = digest(path)
        sources.append(dict(material=material['material'], path=str(path), bytes=path.stat().st_size,
                            sha256=actual, authoritative_sha256=material['source_sha256'],
                            matches_authoritative_registry=actual == material['source_sha256'],
                            shape_zyx=[3051, 1033, 1030], dtype='uint8', axis_order='zyx', spacing_um=[5,5,5]))
    assert all(s['matches_authoritative_registry'] and s['bytes'] == 3246233490 for s in sources)
    evidence = [registry, OFFICIAL/'archive_inventory/DRP167_tree.json',
                OFFICIAL/'audit/DRP167_DOWNLOAD_LEDGER.csv', OFFICIAL/'audit/DRP167_OFFICIAL_FILE_INVENTORY.csv',
                Path('D:/Downloads/论文1修改comments.txt')]
    write('SOURCE_AUDIT', dict(sources=sources, evidence=[dict(path=str(p), sha256=digest(p)) for p in evidence],
        official_url='https://www.digitalrocksportal.org/projects/167', doi='10.17612/P7HT11',
        license='ODC-BY 1.0; retain attribution to Alessio Scanziani/Imperial College London and DOI; label all derived data.',
        provenance='Official-download ledger plus archived provider tree, current official project page, and byte equality to frozen ROI source registry.',
        compatibility='Same WF2 second waterflooding / GI gas-injection states of Ketton water-wet carbonate; same raw identities as manuscript ROI registry.',
        phase_mapping={'raw_1':'oil', 'raw_2':'brine', 'raw_3':'rock', 'raw_4':'gas', 'canonical':'raw minus one, 0..3'},
        segmentation='Provider metadata: seeded watershed; no resegmentation planned.',
        conflict='data/README.md GI SHA256 differs from actual and authoritative frozen_target_roi_registry_v2.json; preserve README, use authority-linked new manifest.',
        new_downloads=False, decoded_source_pixels=0,
        pending='Stream-validate full raw label domain during authorized preparation; do not infer label domain merely from dtype.'))
    design = dict(status='PROPOSED_NOT_MATERIALIZED', design='A', axis_order='zyx', coordinate_convention='half-open voxel bounds; physical coordinates index*5 micrometres; plane x->zy, y->zx, z->yx',
        paired_flow_states=True, split_units='Identical spatial blocks in WF2 and GI because states share a rock; never split states independently.',
        splits={'train': [[32,1000],[32,1001],[32,998]], 'validation':[[1128,1512],[32,1001],[32,998]], 'test':[[1640,3019],[32,1001],[32,998]]},
        guard='128-voxel (640 um) z gaps; all feature footprints radius 4 remain inside own split; guard exceeds feature reach but cannot establish geological decorrelation.',
        historical_exclusions=[{'material':m['material'], 'origin_zyx':m['roi_origin_zyx'], 'shape_zyx':m['roi_shape_zyx'], 'dilation_voxels':128} for m in historical['materials']],
        historical_rule='Exclude union of both historical ROIs dilated by 128 voxels from ALL new splits and diagnostic ROI selection; historical regions only descriptive legacy evidence.',
        planes='Only split-clipped planes/tiles; never whole planes crossing roles. Nested axis indices at multiples of 5/50/200 for 25/250/1000 um. Register each cropped plane before fitting.',
        plane_manifest_fields=['identity','source_sha256','source_shape_zyx','dtype','phase_map','spacing_um','normal_axis','normal_index','in_plane_bounds','plane_axis_order','split','support_sha256','payload_sha256','preparation_code_sha256'],
        roi_rule='Enumerate 48^3 cubes at z=1700+200*k, y=200+200*j, x=200+200*i, wholly in test and outside historical exclusions. Sort by SHA256(source-family-id|zyx origin|fixed salt REVISION_REPLICATION_P0); take first 3 distinct locations, same locations in both states. No outcome/phase-coverage screening or replacements after viewing results.',
        diversity='Report all selected phase distributions including absent phases; objective geometric selection does not guarantee diverse composition. Additional strata would require separate prespecified protocol.',
        test_observations='Register observation-mask subset separately; evaluators exclude clamped observations. Recon receives observation values only, never complementary test truth.',
        initial_matrix='2 flow states x 3 training sparsities x 3 seeds x 12 orientation/phase models; A0-A5 plus one new common B0 per unit. One validation-selection rule for all. No automatic continuation from old protocol.',
        uncertainty='Exact sample counts, legal tile support and worker firewall must be validated before dataset release. This document is not a released dataset.')
    write('DATASET_DESIGN', design)
    write('LEAKAGE_ASSESSMENT', dict(status='HARD_GATE_PENDING_IMPLEMENTATION',
        historical_roi_independent_test_eligible=False,
        reason='Previously inspected outcomes and unknown historical fit-plane exposure; absence of recovered original payloads prevents negative exposure proof.',
        safeguards=['Paired spatial coordinates across flow states', '128-voxel guard and full patch footprint containment',
                    'Historical ROI union exclusions', 'No performance-driven ROI selection', 'Distinct test observations and withheld truth'],
        programmatic_contract='paper_revision/replication_firewall.py rejects wrong identity, roles, escaping paths and hashes; synthetic denial test passed.',
        isolation_required='Separate data-preparation principal and read-only role mounts/accounts. Train sees train only; selection sees validation only; recon sees observations only; evaluator sees predictions/truth. Raw source never mounted downstream.',
        limitation='Helper is not an OS sandbox; no scientific worker/container configured or validated in P0. Thus end-to-end no-label-access is NOT established.',
        provenance_gate='No plane can be admitted without exact source/coordinate/phase/hash manifest. No new planes created in P0.',
        dependence_warning='Spatial separation prevents direct overlap, not shared geology or independent-material generalization.'))
    write('CHECKPOINT_ELIGIBILITY', dict(preferred='Design A', historical_B0_eligible=False,
        design_B='Blocked: no complete payload-bound historical fit exposure proof against new held-out blocks; matching shape/weights or registry existence insufficient.',
        design_A='New B0 per condition/seed/orientation/phase trained on new train only, fixed 50 epochs; common exact checkpoint hash feeds all six branches. Never mix A/B causal estimates.',
        branches={'A0':'all 46529 B0 parameters, MSE', 'A1':'all 46529 B0 parameters, NLL',
                  'A2':'frozen common B0 + 383 residual parameters, MSE', 'A3':'frozen common B0 + 383 residual parameters, NLL',
                  'A4':'frozen common B0 + 380 trainable residual parameters, NLL, constant gate', 'A5':'frozen common B0 + 383 residual parameters, NLL, dilations111'},
        controls='Same allowed events/batch order, seed pairing, 50 continuation epochs, batch4096, fixed learning-rate/optimizer declared before fit; primary epoch50, best-validation secondary only. Untouched B0 baseline quantifies continuation.',
        estimands='A0/A1 objective in B0; A2/A3 objective in residual; A0/A2 and A1/A3 parameterization under same objective; A3/A4 adaptive gate; A3/A5 receptive fields.',
        capacity='Equal epochs/steps do not equal trainable freedom, optimization geometry, effective capacity or FLOPs. Record all and avoid claiming pure architecture causality across frozen vs fully trainable B0.',
        checkpoint_selection='No test-driven selection or threshold tuning; report seed-level paired differences, not voxel-level independent replication.'))
    rows = [
      ('R1-C1','Define WF2/GI on first use','Editorial text','Define second waterflooding and gas injection in abstract/introduction','YES_EDITORIAL','none','none','No generalization claim'),
      ('R1-C2','Local accuracy versus poor 3D fidelity','Figures9/10 described by reviewer','Phase TPCF, sections, structural metrics and explicit practical limits','YES_WITH_LIMITS','withheld test','shared trajectories','Small same-rock ROIs'),
      ('R2-C1','Architecture versus loss and ablations','Historical objective confound','Design A common B0; A0-A5 paired evaluation','YES_CONDITIONAL','train/validation/test','1512 fits = 2 states x 3 sparsities x 3 seeds x 12 models x 7 fits','Frozen backbone versus trainable B0 remains capacity difference'),
      ('R2-C2','Dataset visualization and preprocessing','Source registries and provider metadata','Source/phase map, 3D truth, orthogonal slices, distributions, sparse masks','YES','prep/evaluator truth only','rendering/descriptive','Same dataset'),
      ('R2-C3','Additional ROIs and dominant-phase baseline','Two related-flow-state historical ROIs','3 deterministic new spatial ROIs/state; train-derived dominant baseline','PARTIAL','prespecified test cubes','shared predictions','Cannot prove independent material or guaranteed composition diversity'),
      ('R2-C4','Minority-phase structure and connectivity','Reported WF2 phase depletion','Per-phase fraction, TPCF, connected components, Euler/connectivity, chord statistics; absent-phase handling','YES_WITH_LIMITS','test truth evaluator','shared metric pass','48^3 descriptors not effective-property validation'),
      ('R2-C5','Longer sweeps and initialization/order sensitivity','Three historical sweeps','Save sweeps0..10; 5 seeds; main plus two order and two initialization changes, one factor at a time','YES_CONDITIONAL','observations only','long trajectories','Finite horizon cannot prove convergence; no complete interaction estimate'),
      ('R2-C6','Separate global/local density','250/1000 shared historical local mask','3 global x 3 local masks crossed on prespecified ROI; share trained models','YES_CONDITIONAL','nested masks','additional rollouts no refit','Same source dependence'),
      ('R2-C7','Established baseline and computational efficiency','Study-specific B0 only','Benchmark timing available; choose and verify established statistical reconstruction implementation with equal clamping before campaign','PARTIAL','same observations/evaluator','baseline cost not yet measured','Representative baseline still unselected; no competitiveness conclusion'),
      ('R2-C8','ES selector saturation','Reviewer reports always selects BPMR','Retain exploratory status; validation-only threshold curve plus fixed test usage reporting; no new factorial needed','YES','validation + evaluator','reuse saved probabilities','Threshold transfer limited'),
      ('R2-C9','Gate behavior','Mechanism exists, behavior not demonstrated','Gate distributions versus entropy/margin/support and local descriptors; seed-stratified association; A4 control','YES_CONDITIONAL','registered diagnostic events','reuse A3/A4 logs','Association does not prove useful causal modulation'),
      ('R2-C10','Internal orthogonal sections','Reviewer reports 3D surface limitations','XY/XZ/YZ at fixed ROI midplanes, shared color map for truth/B0/BPMR','YES','test evaluator','shared visualization','No cherry-picked sections')]
    with (OUT/(PREFIX+'REVIEWER_COVERAGE.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(['comment','scientific_question','historical_evidence','required_new_evidence','resolvable','data','compute','generalization_limit']);writer.writerows(rows)
    cpu_path=OUT/(PREFIX+'CPU_BENCHMARK.json')
    cpu=json.loads(cpu_path.read_text())
    b0=cpu['voxel_proxy']['A0']['median_seconds_per_voxel']; residual=cpu['voxel_proxy']['A3']['median_seconds_per_voxel']
    # 2 states, 3 global regimes, 3 seeds, 12 models, new source + 6 branches.
    steps=50*25  # explicit illustrative cap=100000 events, ceil/4096=25
    per={r['variant']:r['median_step_seconds'] for r in cpu['optimizer_steps'] if r['batch']==4096}
    train_seconds=2*3*3*12*steps*(per['A0']+sum(per.values()))
    primary_seconds=2*3*3*3*5*3*10*(48**3)*(2*b0+4*residual)
    minimal_seconds=2*3*3*5*10*(48**3)*(2*b0+4*residual)
    cpu.update(recommendation='LOCAL_CPU_NOT_FEASIBLE', recommendation_scope='Full candidate matrix on current serial immediate-update CPU proxy under 19-day revision window; small training/model studies locally feasible.',
               paper_dependency_imports_passed=['matplotlib','pandas','scipy','PIL'],
               budget_assumptions=dict(training_events_per_model=100000, batch=4096, epochs=50, steps_per_fit=steps, source_B0_epochs=50,
                    states=2, global_sparsities=3, training_seeds=3, orientation_phase_models=12, additional_test_rois_per_state=3, rollout_seeds=5, sweeps=10),
               estimates=dict(training_hours=train_seconds/3600, primary_crossed_reconstruction_serial_days=primary_seconds/86400,
                              reduced_one_global_one_local_serial_days=minimal_seconds/86400,
                              one_A3_48cubed_10sweep_hours=residual*48**3*10/3600,
                              source_bytes_each=3246233490, eager_float32_source_bytes_each=3246233490*4),
               caveat='No exact scheduler benchmark; these are conditional proxy extrapolations, not measured campaign runtimes. Serial seconds ignore observed-mask skips and parallel speedup; extra baselines, sensitivity, ES and I/O excluded. Training count cap is proposed, not historical.',
               server_required='Not required for P0 or small training. Full proposed serial matrix unsuitable locally; reduce valid scope or validate CPU parallel/optimized implementation before deciding hardware. No GPU/server commissioned.')
    cpu_path.write_text(json.dumps(cpu,indent=2),encoding='utf-8')
    blockers=['Full raw phase-domain validation pending preparation (semantics and source identity established).',
              'No registered new plane payloads by P0 instruction; auditable manifest release pending.',
              'Role helper tested but OS-level worker isolation and end-to-end leakage proof pending.',
              'Historical B0 exposure cannot be cleared; Design B ineligible.',
              'Established comparator implementation not selected/verified.',
              'Full candidate reconstruction matrix exceeds local budget; Windows scientific runner and reduced scientifically sufficient schedule not validated.']
    write('FINAL',dict(classification='PARTIAL', result='REVISION_R1_RECONSTITUTION_P0=PARTIAL', sources_available=True,
                      source_identity_verified=True, design='A proposed; not released', reviewer_items_covered=12,
                      cpu_recommendation='LOCAL_CPU_NOT_FEASIBLE', scientific_execution_authorized=False,
                      blockers=blockers, warning='WF2 and GI are flow states of the same carbonate rock, not independent materials.',
                      historical_modified=False, scientific_planes_created=0, scientific_fits=0, real_reconstructions=0))
    text=f'''# REVISION_REPLICATION — P0 feasibility audit

REVISION_R1_RECONSTITUTION_P0=PARTIAL

Material Passport: manuscript MTCOMM-D-26-05155; stage P0; evidence: byte hashes, official project metadata, frozen ROI registry, reviewer letter and synthetic CPU tests. No new scientific payloads or results. Verification is limited to the checks explicitly recorded.

WF2 and GI are available locally under `{OFFICIAL / 'official_download'}`. Both 3,246,233,490-byte files match the authoritative frozen ROI source hashes. Official project 167 (DOI 10.17612/P7HT11, ODC-BY 1.0) documents 5 um voxels, seeded watershed and raw phases 1=oil, 2=brine, 3=rock, 4=gas. Canonical conversion is raw minus one. GI hash in repository data/README.md is inconsistent; the historical file was preserved. Full label-domain scanning remains a preparation gate.

Design A uses a new common B0 fitted on revision train only. A0–A5 branch from identical source hashes; fixed epoch50 primary endpoints avoid test-driven selection. Historical B0 is ineligible because original fit exposure cannot be excluded. Architecture comparisons still differ in trainable freedom (46,529 versus 383/380); equal steps are not equal capacity.

Proposed spatial z slabs separate train, validation and test with 128-voxel guards and paired split coordinates across both states. Historical ROI unions plus guards are excluded. Additional 48^3 ROIs follow a deterministic geometry/hash rule independent of results. Same-rock robustness cannot establish material generalization. New plane manifests must include source, coordinate, phase and payload hashes. No plane was generated.

The role-aware artifact helper passes synthetic denial tests. It does not replace OS mount/account separation. No scientific worker is released until raw/truth exclusion is verified end to end; this is a hard campaign gate, not a claim that leakage has already been ruled out.

Reviewer letter: D:/Downloads/论文1修改comments.txt. Coverage CSV contains all 2 Reviewer 1 concerns and 10 Reviewer 2 comments. Shared trajectories answer fidelity, phase structure, sections, gate diagnostics and selector use. Longer trajectories and the global/local cross use shared fitted models. Dominant-phase baseline uses train-only priors. An established comparator remains to be specified; no claim that all reviewer requests are resolved.

Python 3.11.17 environment: `.venv-revision311`; repository dev/paper dependencies and CPU torch installed. A0–A5 finite-loss forward/backward and strict state reload passed. Synthetic process sampled RSS peaked at {cpu['sampled_peak_rss_bytes']/2**20:.1f} MiB. Serial voxel proxy: B0 {b0*1000:.2f} ms, BPMR {residual*1000:.2f} ms; one 48^3 BPMR trajectory at 10 sweeps is approximately {residual*48**3*10/3600:.2f} hours. Illustrative fitting matrix is {train_seconds/3600:.1f} hours; primary fully crossed rollout proxy is {primary_seconds/86400:.1f} serial days. Reduced one-global/one-local schedule still estimates {minimal_seconds/86400:.1f} days. These exclude scheduler/I/O/metrics and are conditional estimates, not scientific runtimes.

LOCAL_CPU_NOT_FEASIBLE applies to the full candidate serial matrix within the 19-day revision window. Local fitting and reduced pilots remain practical. A server is unnecessary for P0; campaign hardware is undecided until a sufficient reduced schedule or validated CPU optimization is established. No server rental or GPU execution occurred.

Remaining gates:
''' + '\n'.join('- '+b for b in blockers) + '\n\nSTOP P0. No campaign launch.\n'
    (OUT/(PREFIX+'SUMMARY.md')).write_text(text,encoding='utf-8')
    print(json.dumps({'result':'PARTIAL','training_hours':train_seconds/3600,'primary_serial_days':primary_seconds/86400}))


if __name__=='__main__':
    main()
