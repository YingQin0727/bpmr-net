"""Conditional Design A held-out evaluator.

The default/--verify-only path reads only manifests and registries. Test payloads
are opened only by an explicit later-stage --evaluate invocation.
"""
import argparse, csv, hashlib, io, json, math, time
from pathlib import Path
import numpy as np
import psutil
import torch
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.training_access import TrainingAccess
from paper_revision import heldout_evaluation_r0 as r0

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/revision_mtcomm_2026"
TRAIN=BASE/"REVISION_REPLICATION_training/DesignA"
OUT=BASE/"REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R2"
PUB=ROOT/"paper_revision/evidence_release_r2"
DEC=PUB/"REVISION_DESIGNA_EVIDENCE_R2_RELEASE_DECISION.json"
TIERS=PUB/"REVISION_DESIGNA_EVIDENCE_R2_MODEL_TIERS.csv"
PAIRS=PUB/"REVISION_DESIGNA_EVIDENCE_R2_PAIRING_ELIGIBILITY.csv"
FREEZE=BASE/"REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0/REVISION_DESIGNA_HELDOUT_R0_MODEL_FREEZE.json"
PROTOCOL=BASE/"REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0/REVISION_DESIGNA_HELDOUT_R0_PROTOCOL.json"
REGISTRY=BASE/"R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv"
PREFIX="REVISION_DESIGNA_HELDOUT_R2_"
COMPARISONS=(("A0","A1"),("A1","A3"),("A2","A3"),("A3","A4"),("A3","A5"),("B0_SOURCE","A0"))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def jread(path): return json.loads(path.read_bytes())
def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def truth(v): return str(v).lower()=="true"
def write_csv(path,rows):
    if not rows: raise ValueError("refusing empty evaluation table: "+path.name)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def identity(unit,variant): return unit+"/"+variant
def set_hash(rows):
    ids=sorted(identity(x["unit"],x["variant"]) for x in rows)
    return hashlib.sha256(("\n".join(ids)+"\n").encode("utf-8")).hexdigest()


def preflight():
    d=jread(DEC); freeze=jread(FREEZE); protocol=jread(PROTOCOL)
    tiers=read_csv(TIERS); pairs=read_csv(PAIRS); registry=read_csv(REGISTRY)
    assert d["HELDOUT_EVALUATION_READY"]=="YES"
    assert d["ORIGINAL_STRICT_RELEASE"]=="FAIL" and d["CONDITIONAL_SCIENTIFIC_RELEASE"]=="YES"
    assert d["evaluation_protocol"]["protocol_sha256"]==sha(PROTOCOL)
    assert d["evaluation_protocol"]["next_entrypoint_sha256"]==sha(Path(__file__))
    assert freeze["status"]=="PASS" and len(freeze["models"])==840
    assert protocol["model_freeze_sha256"]==sha(FREEZE)
    assert len(tiers)==840 and len(pairs)==720
    primary=[x for x in tiers if truth(x["primary_candidate"])]
    sensitivity=[x for x in tiers if truth(x["sensitivity_candidate"])]
    assert len(primary)==833 and len(sensitivity)==840
    assert sum(truth(x["primary_eligible"]) for x in pairs)==714
    assert sum(truth(x["sensitivity_eligible"]) for x in pairs)==720
    frozen={identity(x["unit"],x["variant"]):x for x in freeze["models"]}
    assert set(frozen)=={identity(x["unit"],x["variant"]) for x in tiers}
    assert all(frozen[identity(x["unit"],x["variant"])]["sha256"]==x["checkpoint_sha256"] for x in tiers if x["variant"]!="B0_SOURCE")
    assert all(frozen[identity(x["unit"],x["variant"])]["sha256"]==x["checkpoint_sha256"] for x in tiers if x["variant"]=="B0_SOURCE")
    assert all(sha(ROOT/x["path"])==x["sha256"] for x in freeze["models"])
    assert set_hash(primary)==d["evaluation_protocol"]["primary_set_sha256"]
    assert set_hash(sensitivity)==d["evaluation_protocol"]["sensitivity_set_sha256"]
    assert sha(PAIRS)==d["evaluation_protocol"]["pairing_sha256"]
    assert sum(x["role"]=="test_truth" for x in registry)==180
    for row in (x for x in registry if x["role"]=="test_truth"):
        box=json.loads(row["bounds_zyx"])
        for fit in (x for x in registry if x["material"]==row["material"] and x["role"] in ("train","validation")):
            other=json.loads(fit["bounds_zyx"])
            assert not all(max(a[0],b[0])<min(a[1],b[1]) for a,b in zip(box,other)), "test/fitting geometry overlap"
    outputs=["ALL_METRICS.csv","PHASE_METRICS.csv","PAIRED_COMPARISONS.csv","GATE_DIAGNOSTICS.csv","RUNTIME.csv","CONDITION_ROI_METRICS.csv","SEED_UNCERTAINTY.csv","EVENT_IDENTITY.json","ACCESS.jsonl","FINAL_AUDIT.json","SUMMARY.md","RESULT_MANIFEST.json"]
    assert not any((OUT/(PREFIX+x)).exists() for x in outputs),"R2 evaluation outputs already exist; refusing rerun"
    return d,freeze,protocol,tiers,pairs,registry,primary,sensitivity


def verify_only():
    d,freeze,protocol,tiers,pairs,registry,primary,sensitivity=preflight()
    result=dict(status="READY",decision_sha256=sha(DEC),model_freeze_sha256=sha(FREEZE),
        protocol_sha256=sha(PROTOCOL),tier_csv_sha256=sha(TIERS),pairing_csv_sha256=sha(PAIRS),
        primary_models=len(primary),sensitivity_models=len(sensitivity),
        primary_pairs=sum(truth(x["primary_eligible"]) for x in pairs),
        sensitivity_pairs=sum(truth(x["sensitivity_eligible"]) for x in pairs),
        registered_test_planes=sum(x["role"]=="test_truth" for x in registry),
        test_payload_reads=0,heldout_r2_outputs_exist=False)
    print(json.dumps(result,ensure_ascii=False))
    return result


def evaluate():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d,freeze,protocol,tiers,pairs,registry,primary,sensitivity=preflight()
    outputs=["ALL_METRICS.csv","PHASE_METRICS.csv","PAIRED_COMPARISONS.csv","GATE_DIAGNOSTICS.csv","RUNTIME.csv","CONDITION_ROI_METRICS.csv","SEED_UNCERTAINTY.csv","EVENT_IDENTITY.json","ACCESS.jsonl","FINAL_AUDIT.json","SUMMARY.md","RESULT_MANIFEST.json"]
    write_paths=[OUT/(PREFIX+x) for x in outputs]
    ledger_a_path=TRAIN/"REVISION_DESIGNA_A0A5_R0_TASK_LEDGER.csv"
    ledger_b_path=TRAIN/"REVISION_DESIGNA_B0_TRAIN_R0_TASK_LEDGER.csv"
    reads=[FREEZE,PROTOCOL,DEC,TIERS,PAIRS,REGISTRY,Path(__file__),ledger_a_path,ledger_b_path]
    reads += [ROOT/x["path"] for x in freeze["models"] if identity(x["unit"],x["variant"]) in {identity(y["unit"],y["variant"]) for y in sensitivity}]
    reads += [BASE/x[k+"_path"] for x in registry if x["role"]=="test_truth" for k in ("labels","support")]
    OUT.mkdir(parents=True,exist_ok=True)
    reads += write_paths
    runtime=[Path(torch.__file__).resolve().parents[1],Path(np.__file__).resolve().parents[1],ROOT/".venv-revision311/Lib/site-packages"]
    guard=TrainingAccess(reads,write_paths,runtime);guard.install()
    ledger_a=read_csv(ledger_a_path);ledger_b=read_csv(ledger_b_path)
    task_times={(x["unit"],x["variant"]):float(x["elapsed_seconds"]) for x in ledger_a}
    task_times.update({(x["unit"],"B0_SOURCE"):float(x["task_wall_seconds_including_preparation"]) for x in ledger_b})
    started=time.perf_counter();proc=psutil.Process()
    metrics=[];phase_rows=[];roi_rows=[];gate_rows=[];runtime_rows=[];events=[];all_access=[];access_violations=[]
    tier_by_id={identity(x["unit"],x["variant"]):x for x in tiers}
    sens_ids={identity(x["unit"],x["variant"]) for x in sensitivity}
    primary_ids={identity(x["unit"],x["variant"]) for x in primary}
    model_map={identity(x["unit"],x["variant"]):x for x in freeze["models"]}
    plane_rows=read_csv(REGISTRY)
    for material in ("GI","WF2"):
        for axis in "xyz":
            guard.phase="heldout_test_events_"+material+"_"+axis
            prep=time.perf_counter();x,y,roi,event=r0.test_events(plane_rows,material,axis);prep=time.perf_counter()-prep;events.append(event)
            chosen=[(k,v) for k,v in model_map.items() if v["unit"].split("/")[0]==material and v["unit"].split("/")[2]==axis and k in sens_ids]
            for k,model_info in chosen:
                variant=model_info["variant"];unit=model_info["unit"]
                guard.phase="heldout_model_"+k
                t=time.perf_counter();cpu=time.process_time();path=ROOT/model_info["path"]
                assert sha(path)==model_info["sha256"]
                ck=torch.load(io.BytesIO(path.read_bytes()),weights_only=True,map_location="cpu")
                model=B0Network()
                if variant in ("A2","A3","A4","A5"):model=build_revision_bpmr(model,variant)
                model.load_state_dict(ck["state"],strict=True);model.eval()
                loadtime=time.perf_counter()-t;pred=[];diag=[];infer=time.perf_counter()
                with torch.inference_mode():
                    for begin in range(0,len(x),4096):
                        bx=torch.from_numpy(x[begin:begin+4096]);raw=model(bx)
                        if isinstance(raw,tuple):
                            p=raw[1]
                            if variant=="A3":
                                gi=model.gate_inputs(bx,raw[3]);shift=(p-raw[3]).abs().sum(1)
                                diag.append(np.column_stack([gi.numpy(),raw[4].numpy(),raw[5].norm(dim=1).numpy(),shift.numpy(),raw[3].argmax(1).numpy()]))
                        else:p=raw/raw.sum(1,keepdim=True).clamp_min(1e-12)
                        pred.append(p.numpy().copy())
                infertime=time.perf_counter()-infer;p=np.concatenate(pred);m,ph=r0.metrics(p,y);baseid=r0.identity_columns(unit,variant)
                membership=[]
                if k in primary_ids:membership.append("PRIMARY")
                membership.append("SENSITIVITY")
                for aset in membership:
                    metrics.append(dict(analysis_set=aset,**baseid,**m,event_sha256=event["event_sha256"],checkpoint_sha256=model_info["sha256"],evidence_tier=tier_by_id[k]["evidence_tier"]))
                    phase_rows.extend(dict(analysis_set=aset,**baseid,**q) for q in ph)
                    for rid in sorted(set(roi.tolist())):
                        rm,_=r0.metrics(p[roi==rid],y[roi==rid])
                        roi_rows.append(dict(analysis_set=aset,**baseid,roi=rid,**rm,event_sha256=event["event_sha256"]))
                    runtime_rows.append(dict(analysis_set=aset,**baseid,events=len(y),parameters=model_info["parameters"],effective_trainable_parameters=model_info["effective_trainable_parameters"],training_wall_seconds=task_times[(unit,variant)],checkpoint_load_seconds=loadtime,inference_seconds=infer,cpu_seconds=time.process_time()-cpu,data_preparation_seconds_shared=prep,process_rss_bytes=proc.memory_info().rss,threads=1,evidence_tier=tier_by_id[k]["evidence_tier"]))
                if variant=="A3" and diag:
                    z=np.concatenate(diag);before=z[:,6].astype(int)==y;after=p.argmax(1)==y
                    for aset in membership:
                        for rid in sorted(set(roi.tolist())):
                            for phase_id in range(4):
                                for code in range(4):
                                    mask=(roi==rid)&(y==phase_id)&((before.astype(int)*2+after.astype(int))==code)
                                    if not mask.any():continue
                                    v=z[mask]
                                    gate_rows.append(dict(analysis_set=aset,**baseid,roi=rid,phase=phase_id,b0_correct=bool(code//2),a3_correct=bool(code%2),count=len(v),b0_entropy=float(v[:,0].mean()),top_two_margin=float(v[:,1].mean()),support_ratio=float(v[:,2].mean()),learned_gate=float(v[:,3].mean()),residual_magnitude=float(v[:,4].mean()),probability_shift=float(v[:,5].mean()),gate_entropy_pearson=r0.corr(v[:,3],v[:,0]),gate_margin_pearson=r0.corr(v[:,3],v[:,1]),gate_support_pearson=r0.corr(v[:,3],v[:,2])))
                all_access.extend(guard.records)
                access_violations.extend(r for r in guard.records if not r.get("allowed",False))
                with (OUT/(PREFIX+"ACCESS.jsonl")).open("a",encoding="utf-8") as f:f.write(json.dumps(dict(identity=k,records=guard.records))+"\n")
                guard.records.clear()
                if len(metrics)%80==0:print("evaluated models (both sets)",len({r["unit"]+"/"+r["variant"] for r in metrics}),"/840",flush=True)
    lookup={(r["analysis_set"],r["unit"],r["variant"]):r for r in metrics};paired=[]
    for row in pairs:
        u=row["unit"]
        for aset,field in (("PRIMARY","primary_eligible"),("SENSITIVITY","sensitivity_eligible")):
            if not truth(row[field]):continue
            left=lookup[(aset,u,row["left_variant"])];right=lookup[(aset,u,row["right_variant"])]
            assert left["event_sha256"]==right["event_sha256"]
            paired.append(dict(analysis_set=aset,**r0.identity_columns(u,row["right_variant"]),comparison=row["comparison"],**{k:right[k]-left[k] for k in ("categorical_nll","brier","ece","accuracy","balanced_accuracy","macro_f1")},event_sha256=left["event_sha256"]))
    seed_rows=[]
    for aset in ("PRIMARY","SENSITIVITY"):
        subset=[r for r in paired if r["analysis_set"]==aset]
        comparisons=sorted({r["comparison"] for r in subset})
        for comparison in comparisons:
            group=[r for r in subset if r["comparison"]==comparison]
            keys=sorted({(r["material"],r["spacing_um"],r["orientation"],r["template"]) for r in group})
            for material,spacing,axis,template in keys:
                rows=[r for r in group if (r["material"],r["spacing_um"],r["orientation"],r["template"])==(material,spacing,axis,template)]
                for metric in ("categorical_nll","brier","ece","accuracy","balanced_accuracy","macro_f1"):
                    vals=np.asarray([r[metric] for r in rows],dtype=np.float64);n=len(vals)
                    sd=float(vals.std(ddof=1)) if n>1 else None
                    half=4.302652729749464*sd/math.sqrt(n) if n==3 else None
                    seed_rows.append(dict(analysis_set=aset,comparison=comparison,material=material,spacing_um=spacing,orientation=axis,template=template,metric=metric,n_training_seeds=n,mean_delta=float(vals.mean()),seed_sd=sd,min_delta=float(vals.min()),max_delta=float(vals.max()),conditional_seed_ci95_lower=float(vals.mean()-half) if half is not None else None,conditional_seed_ci95_upper=float(vals.mean()+half) if half is not None else None,inference_scope="training seed variation conditional on fixed source, spacing, orientation, template and shared test events; not material generalization"))
    btime={r["unit"]:r["inference_seconds"] for r in runtime_rows if r["variant"]=="B0_SOURCE"}
    for row in runtime_rows:
        row["inference_overhead_vs_paired_b0"]=row["inference_seconds"]/btime[row["unit"]]
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in (("ALL_METRICS",metrics),("PHASE_METRICS",phase_rows),("PAIRED_COMPARISONS",paired),("GATE_DIAGNOSTICS",gate_rows),("RUNTIME",runtime_rows),("CONDITION_ROI_METRICS",roi_rows),("SEED_UNCERTAINTY",seed_rows)):write_csv(OUT/(PREFIX+name+".csv"),rows)
    (OUT/(PREFIX+"EVENT_IDENTITY.json")).write_text(json.dumps(dict(groups=events,paired_across_models=True,protocol_sha256=sha(PROTOCOL)),indent=2),encoding="utf-8")
    elapsed=time.perf_counter()-started
    audit=dict(stage="REVISION_DESIGNA_EVIDENCE_RELEASE_R2",status="PASS" if not access_violations else "FAIL",analysis_sets={"PRIMARY":len(primary),"SENSITIVITY":len(sensitivity)},pairs={"PRIMARY":sum(truth(x["primary_eligible"]) for x in pairs),"SENSITIVITY":sum(truth(x["sensitivity_eligible"]) for x in pairs)},test_events_read=True,training_access_gap_disclosed=True,python_access_events=len(all_access),python_access_violations=access_violations,OS_LEVEL_ISOLATION="NOT_ESTABLISHED",elapsed_seconds=elapsed,protocol_sha256=sha(PROTOCOL),conditional_decision_sha256=sha(DEC))
    (OUT/(PREFIX+"FINAL_AUDIT.json")).write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (OUT/(PREFIX+"SUMMARY.md")).write_text("# Conditional Design A held-out evaluation R2\n\nResults are separated into pre-frozen PRIMARY and SENSITIVITY sets. See CSV tables and FINAL_AUDIT.json.\n",encoding="utf-8")
    files=[p for p in OUT.iterdir() if p.name.startswith(PREFIX) and not p.name.endswith("RESULT_MANIFEST.json")]
    manifest={p.name:sha(p) for p in sorted(files)};(OUT/(PREFIX+"RESULT_MANIFEST.json")).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="PASS",models=len({r["unit"]+"/"+r["variant"] for r in metrics}),rows=len(metrics),pairs=len(paired),elapsed_seconds=elapsed),ensure_ascii=False))
    if access_violations: raise SystemExit("evaluation access audit failed")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--verify-only",action="store_true");ap.add_argument("--evaluate",action="store_true")
    args=ap.parse_args()
    if args.evaluate:evaluate()
    else:verify_only()
if __name__=="__main__":main()
