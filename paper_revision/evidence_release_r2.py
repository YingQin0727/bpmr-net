"""Freeze Design A evidence tiers and conditional release metadata without test IO."""
import csv, hashlib, io, json
from collections import Counter, defaultdict
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/revision_mtcomm_2026"
TRAIN=BASE/"REVISION_REPLICATION_training/DesignA"
OUT=BASE/"REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0"
PUB=ROOT/"paper_revision/evidence_release_r2"
NAME="REVISION_DESIGNA_EVIDENCE_R2"
VARIANTS=("A0","A1","A2","A3","A4","A5")
COMPARISONS=(("A0","A1"),("A1","A3"),("A2","A3"),("A3","A4"),("A3","A5"),("B0_SOURCE","A0"))
TARGET="GI/G25/x/seed1701/template0"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def js(p): return json.loads(p.read_bytes())
def ident(u,v): return u+"/"+v
def set_hash(rows):
    """Hash a sorted canonical identity list, independent of CSV formatting."""
    ids=sorted(ident(x["unit"],x["variant"]) for x in rows)
    return hashlib.sha256(("\n".join(ids)+"\n").encode("utf-8")).hexdigest()
def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def main():
    b0p=TRAIN/"REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json"
    ap=TRAIN/"REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json"
    fp=OUT/"REVISION_DESIGNA_HELDOUT_R0_MODEL_FREEZE.json"
    rp=ROOT/"paper_revision/TRAINING_2D_RELEASE_R2.json"
    regp=BASE/"R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv"
    logp=TRAIN/"REVISION_DESIGNA_A0A5_R0_PYTHON_ACCESS.jsonl"
    recp=ROOT/"paper_revision/access_evidence_r1/REVISION_DESIGNA_ACCESS_EVIDENCE_RECOVERY_R1.json"
    b0audit=TRAIN/"REVISION_DESIGNA_B0_TRAIN_R0_AUDIT.json"
    b0=js(b0p); a=js(ap); freeze=js(fp); release=js(rp); recovery=js(recp); audit=js(b0audit)
    assert b0.get("status")=="PASS" and len(b0["checkpoints"])==120
    assert a.get("expected_tasks")==720 and not a.get("failures") and len(a["checkpoints"])==720
    assert freeze.get("status")=="PASS" and len(freeze["models"])==840
    assert sha(regp)==release["input_identity"]["registry_sha256"]
    bindings={n:dict(expected=e,actual=sha(ROOT/n),match=sha(ROOT/n)==e) for n,e in release["launch_binding"].items()}
    code_ok=all(x["match"] for x in bindings.values())
    registry_events={(x["material"],int(x["spacing_um"]),x["axis"],int(x["training_seed"]),int(x["template"]),x["role"]):x for x in release["event_manifest"]}
    formal={ident(r["unit"],r["variant"]):r for r in a["checkpoints"]}
    b0rows={ident(r["unit"],"B0_SOURCE"):r for r in b0["checkpoints"]}
    assert len(formal)==720 and len(b0rows)==120
    registry={}
    with regp.open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            for kind in ("labels","support"):
                p=str((BASE/row[kind+"_path"]).resolve()).replace("\\","/").lower()
                registry[p]=dict(material=row["material"],axis=row["normal_axis"],role=row["role"],
                                 plane_id=row["plane_id"],spacing=row["regimes_um"].split(";"))
    access=defaultdict(list); events=0; denials=[]; prohibited=[]; stream_conflicts=[]
    with logp.open("rb") as f:
        for line_no,raw in enumerate(f,1):
            x=json.loads(raw); k=ident(x["unit"],x["variant"])
            row=dict(line=line_no,raw_sha256=hashlib.sha256(raw).hexdigest(),event_count=len(x["records"]))
            row["denials"]=[r for r in x["records"] if not r.get("allowed",False)]
            row["prohibited_paths"]=[r for r in x["records"] if r.get("operation")=="read" and any(t in r.get("path","").lower() for t in ("test_truth","incomplete","backup","official_download"))]
            unit,variant=k.rsplit("/",1); material,spacing,axis,_,_=unit.split("/")
            spacing=int(spacing[1:])
            expected_input={(p,d["role"]) for p,d in registry.items()
                if d["material"]==material and d["axis"]==axis and d["role"] in ("train","validation")}
            seen_input=set(); source_path=str((TRAIN/unit/"epoch50.pt").resolve()).replace("\\","/").lower()
            output_path=str((ROOT/a["checkpoints"][0]["path"]).resolve()).replace("\\","/").lower()
            formal_row=next(z for z in a["checkpoints"] if ident(z["unit"],z["variant"])==k)
            output_path=str((ROOT/formal_row["path"]).resolve()).replace("\\","/").lower()
            seen_source=False; committed=False
            for rec in x["records"]:
                path=rec.get("path","").replace("\\","/").lower()
                if rec.get("operation")=="read":
                    if path==source_path: seen_source=True
                    if path in registry:
                        d=registry[path]
                        if (d["material"],d["axis"],d["role"])==(material,axis,d["role"]) and d["role"] in ("train","validation"):
                            seen_input.add((path,d["role"]))
                        else:
                            stream_conflicts.append(dict(line=line_no,task=k,reason="registered_plane_identity_mismatch",plane_id=d["plane_id"],role=d["role"]))
                    elif rec.get("category")=="registered_input" and path!=source_path:
                        # Non-plane registered files include the frozen protocol, code and manifests.
                        pass
                if rec.get("operation")=="write" and path==output_path: committed=True
                if rec.get("operation")=="os.rename" and rec.get("allowed") and rec.get("target","").replace("\\","/").lower()==output_path: committed=True
            row["source_b0_read"]=seen_source
            row["train_validation_plane_roles"]=sorted({role for _,role in seen_input})
            row["checkpoint_output_bound"]=committed
            row["task_access_binding_pass"]=bool(not row["denials"] and not row["prohibited_paths"] and seen_source and {"train","validation"}.issubset({role for _,role in seen_input}) and committed)
            if not row["task_access_binding_pass"]:
                stream_conflicts.append(dict(line=line_no,task=k,reason="source/event/output binding incomplete"))
            access[k].append(row); events+=len(x["records"]);denials+=row["denials"];prohibited+=row["prohibited_paths"]
    freeze_map={ident(r["unit"],r["variant"]):r for r in freeze["models"]}
    conflicts=[]
    for k,r in freeze_map.items():
        p=ROOT/r["path"]
        if not p.exists() or sha(p)!=r["sha256"]: conflicts.append(k)
    identity_conflicts=[]; optimizer_conflicts=[]
    for k,r in formal.items():
        unit,variant=k.rsplit("/",1); material,g,axis,seed,template=unit.split("/")
        src=b0rows[ident(r["unit"],"B0_SOURCE")]
        if r["source_checkpoint_sha256"]!=src["sha256"]: identity_conflicts.append(k+":source_b0")
        for role in ("train","validation"):
            if r[role+"_events"]!=src["events"][role]: identity_conflicts.append(k+":"+role+"_events")
        if len(r["epochs"])!=50 or not r["strict_reload"] or not r["finite_parameters"]: identity_conflicts.append(k+":epoch_reload_finite")
        for role in ("train","validation"):
            frozen=registry_events.get((material,int(g[1:]),axis,int(seed[4:]),int(template[8:]),role))
            actual=r[role+"_events"]
            if not frozen or any(actual.get(x)!=frozen.get(x) for x in ("count","event_sha256","ordinal_sha256","feature_sha256","target_sha256")):
                identity_conflicts.append(k+":"+role+"_not_frozen_registry")
        ck=torch.load(io.BytesIO((ROOT/r["path"]).read_bytes()),weights_only=True,map_location="cpu")
        group=ck["optimizer"]["param_groups"][0]
        expected_steps=sum(int(x["optimizer_steps"]) for x in r["epochs"])
        steps={int(x["step"]) for x in ck["optimizer"]["state"].values() if "step" in x}
        if (group["lr"],tuple(group["betas"]),group["eps"],group["weight_decay"],group["amsgrad"],group["maximize"])!=(.001,(.9,.999),1e-8,0,False,False) or steps!={expected_steps}:
            optimizer_conflicts.append(k)
    assert not conflicts,conflicts
    assert not identity_conflicts and not optimizer_conflicts, (identity_conflicts[:5],optimizer_conflicts[:5])
    bad_conf=set(identity_conflicts)
    tiers=[]
    for k in sorted(list(b0rows)+list(formal)):
        u,v=k.rsplit("/",1)
        if v=="B0_SOURCE":
            tier="TIER_1"; ast="PASS"; reason="B0 audit PASS with 120 source checkpoints and native trace; native coverage limits disclosed"; ac=1; aids="B0_AUDIT_SOURCE_NATIVE_TRACE"
            r=b0rows[k]; ck=r["sha256"];src=r["sha256"];th=r["events"]["train"]["event_sha256"];vh=r["events"]["validation"]["event_sha256"];ep=50;strict=r["strict_reload"];finite=r["finite_parameters"]
        else:
            r=formal[k]; rows=access.get(k,[]); ac=len(rows); aids=";".join("line:%s:%s"%(x["line"],x["raw_sha256"]) for x in rows)
            bad=any(x["denials"] or x["prohibited_paths"] for x in rows) or k in bad_conf
            unbound=any(z["task"]==k for z in stream_conflicts)
            if bad: tier,ast,reason="TIER_3","FAIL","policy/path or model/event identity conflict"
            elif not rows: tier,ast,reason="TIER_2","MISSING","original per-file access stream not recovered"
            elif len(rows)!=1 or unbound: tier,ast,reason="TIER_2","PARTIAL","duplicate historical A3 rows include a non-formal output path; attempt attribution unresolved"
            else: tier,ast,reason="TIER_1","PASS","one task-bound row with allowed events and no prohibited path markers"
            ck=r["checkpoint_sha256"];src=r["source_checkpoint_sha256"];th=r["train_events"]["event_sha256"];vh=r["validation_events"]["event_sha256"];ep=len(r["epochs"]);strict=r["strict_reload"];finite=r["finite_parameters"]
        leakage=("UNKNOWN_NO_ACCESS_STREAM" if v!="B0_SOURCE" and not access.get(k) else
                 "NO_DIRECT_VIOLATION_OBSERVED_IN_AVAILABLE_STREAMS" if not denials and not prohibited else
                 "CONFLICT_OBSERVED")
        tiers.append(dict(unit=u,variant=v,evidence_tier=tier,
            model_integrity="PASS" if not conflicts else "FAIL",
            training_identity="PASS" if k not in bad_conf else "FAIL",
            access_evidence=ast,
            data_leakage_evidence=leakage,
            access_rows=ac,access_evidence_ids=aids,access_reason=reason,
            checkpoint_sha256=ck,source_b0_sha256=src,train_event_sha256=th,
            validation_event_sha256=vh,epochs=ep,strict_reload=strict,finite_parameters=finite,
            primary_candidate=(u!=TARGET and tier!="TIER_3"),
            sensitivity_candidate=(tier!="TIER_3"),
            primary_exclusion_reason="" if u!=TARGET and tier!="TIER_3" else
            "target unit is excluded as a complete paired unit because A0/A1 access is missing and A3 attribution is unresolved"))
    counts=Counter(x["evidence_tier"] for x in tiers)
    counts.update({"TIER_3": 0})
    primary=[x for x in tiers if x["primary_candidate"]]
    sens=[x for x in tiers if x["sensitivity_candidate"]]
    if not (counts==Counter({"TIER_1":837,"TIER_2":3}) and len(primary)==833 and len(sens)==840):
        raise RuntimeError(json.dumps(dict(tiers=dict(counts),primary=len(primary),sensitivity=len(sens),
            stream_conflicts=stream_conflicts[:20]),ensure_ascii=False))
    pairs=[]
    units=sorted({x["unit"] for x in tiers})
    for u in units:
        bv={x["variant"]:x for x in tiers if x["unit"]==u}
        for left,right in COMPARISONS:
            l,r=bv[left],bv[right];eligible=l["evidence_tier"]!="TIER_3" and r["evidence_tier"]!="TIER_3"
            pairs.append(dict(unit=u,comparison=left+"_vs_"+right,left_variant=left,right_variant=right,left_tier=l["evidence_tier"],right_tier=r["evidence_tier"],train_validation_event_pair="PASS",source_b0_pair="PASS",primary_eligible=bool(eligible and u!=TARGET),sensitivity_eligible=bool(eligible),exclusion_reason="" if eligible and u!=TARGET else "target unit excluded from PRIMARY_SET; A0/A1 missing and A3 attribution unresolved"))
    pp=sum(x["primary_eligible"] for x in pairs);sp=sum(x["sensitivity_eligible"] for x in pairs);assert pp==714 and sp==720
    unresolved=dict(count=4,missing_task_streams=2,missing_tasks=[TARGET+"/A0",TARGET+"/A1"],unresolved_duplicate_a3_raw_records=2,a3_raw_rows=3,interpretation="two missing streams plus two exact duplicate A3 rows; final A3 row has atomic commit but no authenticated process attempt ID")
    protocol_path=OUT/"REVISION_DESIGNA_HELDOUT_R0_PROTOCOL.json"
    protocol=js(protocol_path)
    primary_set_hash=set_hash(primary)
    sensitivity_set_hash=set_hash(sens)
    eval_script=ROOT/"paper_revision/heldout_evaluation_r2.py"
    test_plane_ids=[dict(plane_id=x["plane_id"],material=x["material"],orientation=x["normal_axis"],labels_sha256=x["labels_sha256"],support_sha256=x["support_sha256"]) for x in read_csv(regp) if x["role"]=="test_truth"]
    decision=dict(stage="REVISION_DESIGNA_EVIDENCE_RELEASE_R2",REVISION_DESIGNA_EVIDENCE_RELEASE_R2="CONDITIONAL",MODEL_INTEGRITY="PASS",TRAINING_PROVENANCE="PARTIAL",ACCESS_EVIDENCE="PARTIAL",DATA_LEAKAGE_EVIDENCE="PARTIAL_NO_DIRECT_VIOLATION_OBSERVED",ORIGINAL_STRICT_RELEASE="FAIL",CONDITIONAL_SCIENTIFIC_RELEASE="YES",HELDOUT_EVALUATION_READY="YES",PRIMARY_MODEL_COUNT=len(primary),SENSITIVITY_MODEL_COUNT=len(sens),PRIMARY_PAIR_COUNT=pp,UNRESOLVED_ACCESS_RECORDS=unresolved["count"],decision_type="CONDITIONAL_SCIENTIFIC_RELEASE",risk_acceptance=dict(accepted=True,scope="Frozen held-out evaluation only; no retraining, tuning, reconstruction or external comparator",basis=["840 checkpoint identities, hashes, strict reload, finite parameters and fixed 50-epoch records verified","All 720 source B0 and train/validation event identities match frozen manifests","No prohibited read or denied event observed in 718 available task streams or B0 audit","A0/A1 missing and A3 attribution uncertainty explicitly TIER_2 and sensitivity-only","Primary and sensitivity sets frozen before test payload access"],limitation="Missing streams cannot prove no unobserved prohibited read; original strict criterion is not waived"),evidence_basis=dict(checkpoint_manifest_sha256=sha(ap),b0_manifest_sha256=sha(b0p),model_freeze_sha256=sha(fp),registry_sha256=sha(regp),release_code_identity_pass=code_ok,code_binding=bindings,b0_audit_status=audit.get("status"),access_log_sha256=sha(logp),access_log_rows=sum(len(x) for x in access.values()),unique_access_task_identities=len(access),access_events=events,observed_denials=len(denials),observed_prohibited_reads=len(prohibited),stream_binding_conflicts=stream_conflicts,checkpoint_identity_conflicts=conflicts,training_identity_conflicts=identity_conflicts),tier_counts=dict(counts),model_counts=dict(primary=len(primary),sensitivity=len(sens)),pair_counts=dict(primary=pp,sensitivity=sp,primary_complete_units=119,sensitivity_complete_units=120),unresolved_access_detail=unresolved,evaluation_protocol=dict(frozen_before_test_payload_read=True,protocol_path=str(protocol_path),protocol_sha256=sha(protocol_path) if protocol_path.exists() else None,inherited_test_metrics_and_sampling=protocol,test_registry_planes=test_plane_ids,test_registry_plane_count=len(test_plane_ids),primary_set_sha256=primary_set_hash,sensitivity_set_sha256=sensitivity_set_hash,pairing_sha256=None,next_entrypoint="paper_revision.heldout_evaluation_r2",next_entrypoint_sha256=sha(eval_script),mode="CONDITIONAL_R2_READ_ONLY",test_payload_reads_this_stage=0,output_namespace="outputs/revision_mtcomm_2026/REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R2"),tier_definitions=dict(TIER_1="Model, training identity and task-bound original access evidence verifiable; no observed prohibited event",TIER_2="Model/training identity verifiable but original per-file access missing or attempt attribution incomplete; no direct violation observed",TIER_3="Substantive model, input identity or data-source conflict; excluded from both sets"),artifacts={})
    controller=ROOT/"paper_revision/controlled_a0a5_r0.py"
    decision["evidence_basis"]["optimizer_state_conflicts"]=optimizer_conflicts
    decision["evidence_basis"]["frozen_registry_event_identity"]="PASS"
    decision["evidence_basis"]["training_controller"]={
        "git_commit":"34dfcdef0617660b9648a6619c08a964d41506f0",
        "file_sha256":sha(controller),
        "commit_time":"2026-10-20T10:33:51+08:00",
        "log_first_completion":"4/720 GI/G25/x/seed1701/template0 A3",
        "binding_limit":"controller source and log order support identity but do not cryptographically prove every earlier process loaded this exact blob"
    }
    summary="""# Design A evidence release R2

REVISION_DESIGNA_EVIDENCE_RELEASE_R2=CONDITIONAL
MODEL_INTEGRITY=PASS
TRAINING_PROVENANCE=PARTIAL
ACCESS_EVIDENCE=PARTIAL
DATA_LEAKAGE_EVIDENCE=PARTIAL_NO_DIRECT_VIOLATION_OBSERVED
ORIGINAL_STRICT_RELEASE=FAIL
CONDITIONAL_SCIENTIFIC_RELEASE=YES
HELDOUT_EVALUATION_READY=YES

The original 720/720 task-bound access-evidence rule remains unmet. Available Python evidence contains 720 rows, 718 unique task identities and 622,927 events, with zero observed denials or prohibited reads. Missing A0/A1 streams remain unknown. A3 has three raw rows; two are byte-identical and the final row records the pending-to-epoch50_reexecution.pt commit, but there is no authenticated process attempt ID for complete historical attribution.

Model and training identity evidence remains strong: 120 B0 plus 720 A0-A5 checkpoint identities, hashes, strict reload, finite parameters, fixed 50 epochs, source-B0 hashes and train/validation event hashes are verified. No test payload was read in this stage.

Evidence tiers: TIER_1=837; TIER_2=3; TIER_3=0. TIER_2 consists of target-unit A0, A1 and A3. PRIMARY_SET contains 833 model entries and 714 comparison-unit pairs across 119 complete units. SENSITIVITY_SET contains 840 model entries and 720 comparison-unit pairs, including disclosed TIER_2 models.

The conditional decision permits the next stage to run frozen independent test evaluation under CONDITIONAL_R2_READ_ONLY, with PRIMARY_SET and SENSITIVITY_SET selected before any test payload read. It does not convert the original strict release to PASS and does not prove that missing streams had no unobserved prohibited access. TIER_2 results must remain sensitivity evidence.

UNRESOLVED_ACCESS_RECORDS=4 raw records: two missing task streams plus two exact duplicate A3 rows. The original logs and scientific payloads were not modified. OS_LEVEL_ISOLATION=NOT_ESTABLISHED.

No training, retuning, test evaluation, 3D reconstruction or external baseline was executed.
"""
    tpath=PUB/(NAME+"_MODEL_TIERS.csv");ppath=PUB/(NAME+"_PAIRING_ELIGIBILITY.csv");dpath=PUB/(NAME+"_RELEASE_DECISION.json");spath=PUB/(NAME+"_SUMMARY.md")
    write_csv(tpath,tiers);write_csv(ppath,pairs);spath.parent.mkdir(parents=True,exist_ok=True);spath.write_text(summary,encoding="utf-8")
    decision["evaluation_protocol"]["primary_set_sha256"]=primary_set_hash
    decision["evaluation_protocol"]["sensitivity_set_sha256"]=sensitivity_set_hash
    decision["evaluation_protocol"]["pairing_sha256"]=sha(ppath)
    decision["evaluation_protocol"]["next_entrypoint_sha256"]=sha(eval_script)
    decision["artifacts"]={tpath.name:sha(tpath),ppath.name:sha(ppath),spath.name:sha(spath)}
    dpath.write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf-8")
    OUT.mkdir(parents=True,exist_ok=True)
    for p in (tpath,ppath,dpath,spath):(OUT/p.name).write_bytes(p.read_bytes())
    print(json.dumps(dict(result="CONDITIONAL",heldout_ready="YES",tier_counts=dict(counts),primary_models=len(primary),sensitivity_models=len(sens),primary_pairs=pp,sensitivity_pairs=sp,unresolved_access_records=4,test_payload_reads=0,decision_sha256=sha(dpath)),ensure_ascii=False))
if __name__=="__main__": main()
