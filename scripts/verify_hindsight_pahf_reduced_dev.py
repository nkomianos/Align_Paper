"""Read-only DEV evidence verification; no model inference or confirmation loader.

Optional --model-snapshot rehashes already-local weights. The ordinary mode
replays tokenization from the saved backend, source prompts and chat template,
then verifies saved logits, checkpoints and endpoint arithmetic. Neither mode
recomputes neural logits or gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from interaction_sprint.hindsight_execution_integrity import (
    FORWARD_GENERATION_RECEIPT, HEX40, HEX64, canonical_json, canonical_sha256,
    capture_execution_provenance, capture_model_provenance, checked_file, checked_source_file,
    declared_vocabulary_size, inspect_adapter_checkpoint, inspect_optimizer_checkpoint, load_learning_dev,
    make_arm_execution_receipt, sha256_file, validate_prediction_bindings,
    verify_manifest, verify_saved_logits, verify_tokenization_backend,
)
from interaction_sprint.hindsight_pahf_reduced import (
    TRAINED_ARMS, build_interface_jobs, build_schedules, cheap_readouts,
    method_decision, qualify_acquisition, qualify_interface, validate_learning_dev,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY/"configs/hindsight_pahf_reduced_dev_v1.json"


def _same(actual, expected, label):
    if canonical_json(actual) != canonical_json(expected):raise ValueError(f"{label} differs")


def _finite(value, label, nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value) or (nonnegative and value<0):
        raise ValueError(f"invalid {label}")


def verify_step_ledger(arm, steps, schedules, cfg):
    """Validate actual logged batches, work counts, steps and explicit loss mix."""
    if len(steps) != cfg["steps"] or [x.get("step") for x in steps] != list(range(1,cfg["steps"]+1)):
        raise ValueError("step ledger length/order differs")
    expected_work={"raw_immediate":(1,18,1,18),"oracle_delayed":(1,18,1,18),
                   "pooled_sft":(0,0,1,8),"pooled_sdpo":(1,8,1,8),
                   "residual":(3,34,2,26),"mixture":(2,26,2,26)}
    tc,tr,sc,sr=expected_work[arm]
    totals={}
    for i,step in enumerate(steps):
        population=[] if arm in {"pooled_sft","pooled_sdpo"} else schedules["population"][i]
        anchors=[] if arm in {"raw_immediate","oracle_delayed"} else schedules["anchors"][i]
        _same(step.get("population_batch_ids"),population,"population batch IDs")
        _same(step.get("anchor_ids"),anchors,"anchor batch IDs")
        _finite(step["loss"],"loss");_finite(step["gradient_norm"],"gradient norm",True)
        if step["learning_rate"]!=cfg["optimizer"]["lr"]:raise ValueError("learning-rate schedule differs")
        work=step["compute_delta"]
        required={"teacher_forward_calls":tc,"teacher_forward_rows":tr,"student_forward_calls":sc,
                  "student_forward_rows":sr,"forward_calls":tc+sc,"forward_rows":tr+sr,
                  "backward_calls":1,"optimizer_updates":1}
        if any(work.get(k,0)!=v for k,v in required.items()):raise ValueError("logged forward/backward work differs from arm")
        for k,v in work.items():
            _finite(v,f"compute {k}",True);totals[k]=totals.get(k,0)+v
        detail=step.get("details",{})
        for v in detail.values():_finite(v,"loss component")
        if arm in {"raw_immediate","oracle_delayed"}:expected=detail["population_mean"]
        elif arm=="residual":expected=detail["population_immediate_mean"]+detail["anchor_residual_mean"]
        elif arm=="mixture":expected=cfg["mixture_weight"]*detail["population_immediate_mean"]+(1-cfg["mixture_weight"])*detail["anchor_delayed_mean"]
        else:expected=step["loss"]
        if not math.isclose(step["loss"],expected,rel_tol=1e-5,abs_tol=5e-5):raise ValueError("logged loss composition differs")
    return totals


def _verify_model_receipt(receipt,cfg):
    for key in ("resolved_revision","tokenizer_revision"):
        if receipt.get(key)!=cfg["model_revision"] or not HEX40.fullmatch(receipt[key]):raise ValueError("immutable model/tokenizer revision differs")
    if receipt.get("model_id")!=cfg["model_id"] or receipt.get("local_files_only") is not True:raise ValueError("model load scope differs")
    files=receipt["files"]
    for name,item in files.items():
        if not isinstance(name,str) or name.startswith("/") or ".." in Path(name).parts or "\\" in name:raise ValueError("unsafe model receipt member")
        if not HEX64.fullmatch(item["sha256"]) or type(item["bytes"]) is not int or item["bytes"]<0:raise ValueError("invalid model file digest/size")
    if canonical_sha256(files)!=receipt["snapshot_tree_sha256"]:raise ValueError("model tree checksum differs")
    if not {"config.json","tokenizer_config.json"}<=set(files) or not receipt["weights"] or not set(receipt["weights"])<=set(files):raise ValueError("incomplete model receipt")
    if "effective_model_config" not in receipt or "effective_tokenizer" not in receipt:raise ValueError("effective model/tokenizer settings absent")
    tok=receipt["effective_tokenizer"]
    if tok["padding_side"]!="left" or tok["pad_token_id"] is None or not HEX64.fullmatch(tok["backend_sha256"]):raise ValueError("tokenizer runtime settings differ")
    if receipt["effective_model_config"].get("use_cache") is not False:raise ValueError("effective use_cache differs")


def verify(root: Path, input_root: Path, *, model_snapshot: Path | None = None) -> dict:
    root=Path(root);manifest=verify_manifest(root)
    def read(name):return json.loads(checked_file(root,name).read_text(encoding="utf8"))
    cfg=json.loads(CONFIG_PATH.read_text(encoding="utf8"));_same(read("spec.json"),cfg,"frozen spec")
    learning,development,input_receipt=load_learning_dev(input_root,expected_manifest_sha256=cfg["input_manifest_sha256"])
    validate_learning_dev(learning,development,cfg);_same(read("input_receipt.json"),input_receipt,"input receipt")
    schedules=build_schedules(learning,cfg);_same(read("schedules.json"),schedules,"schedules")
    _same(read("panels.json"),{k:schedules[k] for k in ("original_panels","pooled_base_ids")},"panels")
    jobs=build_interface_jobs(learning,cfg,schedules["pooled_base_ids"]);_same(read("interface_jobs.json"),jobs,"exact learning-anchor interface jobs")
    readouts=cheap_readouts(learning,development,schedules["pooled_base_ids"],cfg);_same(read("cheap_controls.json"),readouts,"label-restricted deterministic readouts")
    if "provenance.json" not in manifest:
        expected_members={"spec.json","input_receipt.json","schedules.json","panels.json","interface_jobs.json","cheap_controls.json","FAILED.json","RESULT.json"}
        if set(manifest)!=expected_members:raise ValueError("missing provenance outside a sealed pre-model source-capture failure")
        failure=read("FAILED.json");_same(read("RESULT.json"),failure,"source failure result")
        if failure.get("stage")!="source_provenance" or failure.get("classification")!="Invalid assay/capability" or failure.get("qualified") is not False or failure.get("completed_arms")!=[] or failure.get("confirmation_opened") is not False or failure.get("paper_green_light") is not False:
            raise ValueError("unqualified source-capture failure record required")
        return {"verified":True,"decision":failure["decision"],"classification":failure["classification"],
                "manifest_sha256":sha256_file(checked_file(root,"MANIFEST.json")),"execution_provenance_verified":False,
                "completed_arm_checkpoints_verified":0,"saved_logit_arithmetic":None,"tokenization_recomputed":False,
                "model_weight_files_rehashed":False,"recorded_model_weight_receipts_checked":False,
                "neural_inference_recomputed":False,"gradients_recomputed":False,"confirmation_opened":False,
                "paper_green_light":False,"scope":"Sealed DEV preparation and explicit pre-model source-provenance failure only; no execution provenance or experiment verified"}
    provenance=read("provenance.json")
    _same(provenance["effective_config"],cfg,"effective CLI configuration")
    if provenance["effective_config_sha256"]!=canonical_sha256(cfg) or not isinstance(provenance["argv"],list) or not provenance["argv"]:raise ValueError("missing effective CLI receipt")
    _same(read("source_hashes.json"),provenance["source_sha256"],"source receipt")
    required_sources=[Path(__file__),CONFIG_PATH,REPOSITORY/"scripts/run_hindsight_pahf_reduced_dev.py",
                      REPOSITORY/"src/interaction_sprint/hindsight_execution_integrity.py",
                      REPOSITORY/"src/interaction_sprint/hindsight_pahf_reduced.py",
                      REPOSITORY/"src/interaction_sprint/hindsight_neural_anchor.py",
                      REPOSITORY/"src/latent_contract/sender_update.py"]
    current=capture_execution_provenance(REPOSITORY,required_sources,["offline-verifier"],cfg)
    if not set(current["source_sha256"])<=set(provenance["source_sha256"]):raise ValueError("launch source dependency closure is incomplete")
    for name,digest in provenance["source_sha256"].items():
        if sha256_file(checked_source_file(REPOSITORY,name))!=digest:raise ValueError(f"source changed since launch: {name}")
    git=provenance["git"]
    if not HEX40.fullmatch(git["head"]) or any(not HEX64.fullmatch(git[k]) for k in ("index_entries_sha256","unstaged_binary_diff_sha256","staged_binary_diff_sha256")):raise ValueError("Git state receipt incomplete")
    _same(read("generation_receipt.json"),FORWARD_GENERATION_RECEIPT,"forward-only generation receipt")
    result=read("RESULT.json")
    if result.get("confirmation_opened") is not False or result.get("paper_green_light") is not False:raise ValueError("DEV cannot open confirmation or green-light paper")
    failed="FAILED.json" in manifest
    if failed==("COMPLETE.json" in manifest):raise ValueError("exactly one completion/failure marker required")
    completed=result.get("completed_arms",[])
    if completed!=list(TRAINED_ARMS[:len(completed)]):raise ValueError("trained arm sequence differs")
    marker=read("FAILED.json" if failed else "COMPLETE.json")
    if marker.get("completed_arms")!=completed:raise ValueError("completion marker differs")
    if failed and (result.get("classification")!="Invalid assay/capability" or result.get("qualified") is not False):raise ValueError("execution failure cannot qualify an assay")
    setup=read("runtime_setup.json") if "runtime_setup.json" in manifest else None
    model_receipt=read("model_provenance.json") if "model_provenance.json" in manifest else None
    if not failed and (setup is None or model_receipt is None):raise ValueError("successful stage lacks loaded-model receipts")
    if model_receipt is not None:_verify_model_receipt(model_receipt,cfg)
    if setup is not None:
        initial,state_receipt=inspect_adapter_checkpoint(checked_file(root,"initial_adapter.pt"))
        if state_receipt["tensor_sha256"]!=setup["initial_adapter_tensor_sha256"]:raise ValueError("initialization digest differs")
        if set(setup["parameter_names"])!=set(initial):raise ValueError("trainable adapter parameter binding differs")
        if setup.get("vocab_size")!=declared_vocabulary_size(model_receipt):raise ValueError("runtime/model vocabulary sizes differ")
    checkpoint_receipts={};compute=read("compute.json")
    for arm in completed:
        saved=read(f"{arm}_execution.json");settings=saved["final_optimizer"]["optimizer_settings"]
        expected_settings=dict(cfg["optimizer"])
        if "decoupled_weight_decay" in settings:expected_settings["decoupled_weight_decay"]=True
        _same(settings,expected_settings,"AdamW effective settings")
        checked=make_arm_execution_receipt(root,arm,setup["parameter_names"],settings,cfg["steps"])
        _same(saved,checked,"adapter/optimizer execution receipt");checkpoint_receipts[arm]=checked
        steps=read(f"{arm}_steps.json");totals=verify_step_ledger(arm,steps,schedules,cfg)
        progress=[json.loads(line) for line in checked_file(root,f"{arm}_progress.jsonl").read_text(encoding="utf8").splitlines() if line]
        _same(progress,steps,"append-only progress ledger")
        phase=compute["by_phase"][arm+"_train"]
        for key,value in totals.items():
            if not math.isclose(phase.get(key,0),value,rel_tol=1e-10,abs_tol=1e-7):raise ValueError("compute phase does not reproduce from steps")
    interface_predictions=read("interface_predictions.json") if "interface_predictions.json" in manifest else None
    interface_result=None
    if interface_predictions is not None:
        interface_result=qualify_interface(jobs,interface_predictions,cfg)
        _same(read("interface_RESULT.json"),interface_result,"interface arithmetic")
    arm_predictions={}
    for arm in ("baseline",*TRAINED_ARMS):
        name=f"{arm}_dev_predictions.json"
        if name in manifest:
            arm_predictions[arm]=read(name);validate_prediction_bindings(development,arm_predictions[arm])
    acquisition=None
    if {"baseline","raw_immediate","oracle_delayed"}<=set(arm_predictions):
        acquisition=qualify_acquisition(development,arm_predictions,cfg)
        if "acquisition_RESULT.json" in manifest:_same(read("acquisition_RESULT.json"),acquisition,"acquisition arithmetic")
    if not failed:
        if interface_result is None:raise ValueError("completed stage lacks interface result")
        if not interface_result["qualified"]:
            if completed or arm_predictions:raise ValueError("training occurred after invalid interface")
            expected=interface_result
        elif acquisition is not None and not acquisition["qualified"]:
            if completed!=list(TRAINED_ARMS[:2]) or set(arm_predictions)!={"baseline",*TRAINED_ARMS[:2]}:raise ValueError("training continued after invalid acquisition")
            expected=acquisition
        else:
            if completed!=list(TRAINED_ARMS):raise ValueError("full stage has incomplete arms")
            expected=method_decision(development,arm_predictions,readouts,cfg)
        for key,value in expected.items():_same(result.get(key),value,f"result field {key}")
        if compute.get("optimizer_updates")!=len(completed)*cfg["steps"]:raise ValueError("total optimizer update count differs")
    index=read("logits_index.json")
    logits_predictions=dict(arm_predictions)
    if interface_predictions is not None:logits_predictions["interface"]=interface_predictions
    usable_index=[b for b in index if b["arm"] in logits_predictions]
    incomplete_batches=len(index)-len(usable_index)
    if incomplete_batches and not failed:raise ValueError("saved logits have no prediction endpoint")
    tokenizer_replayed=False;weights_rehashed=False;tokenization_receipt=None
    if index:
        if model_receipt is None or setup is None:raise ValueError("saved logits lack model/tokenizer receipts")
        text_by_arm={arm:{r["id"]:r["prompt"] for r in development} for arm in ("baseline",*TRAINED_ARMS)}
        text_by_arm["interface"]={r["job_id"]:r["text"] for r in jobs}
        tokenization_receipt=verify_tokenization_backend(root,index,text_by_arm,model_receipt,setup["answer_token_ids"],cfg["max_tokens"])
        tokenizer_replayed=True
    if model_snapshot is not None:
        if model_receipt is None:raise ValueError("no model receipt to compare")
        snapshot=capture_model_provenance(model_snapshot,cfg["model_id"],cfg["model_revision"])
        _same(snapshot["files"],model_receipt["files"],"local model/tokenizer/weight files");weights_rehashed=True
    logits_receipt=verify_saved_logits(root,usable_index,logits_predictions,answer_token_ids=setup["answer_token_ids"],max_tokens=cfg["max_tokens"],vocab_size=declared_vocabulary_size(model_receipt)) if logits_predictions else None
    if failed and marker.get("active_arm"):
        # A failed active arm is not counted as a completed arm. Its partial
        # checkpoints are still inspected safely when they exist.
        active=marker["active_arm"];steps=int(marker["executed_steps"])
        failed_adapter=f"{active}_failed_adapter.pt";failed_optimizer=f"{active}_failed_optimizer.pt"
        if failed_adapter in manifest:
            state,_=inspect_adapter_checkpoint(checked_file(root,failed_adapter))
            if failed_optimizer in manifest:
                import torch
                raw=torch.load(checked_file(root,failed_optimizer),map_location="cpu",weights_only=True)
                settings={k:v for k,v in raw["param_groups"][0].items() if k!="params"}
                expected_settings=dict(cfg["optimizer"])
                if "decoupled_weight_decay" in settings:expected_settings["decoupled_weight_decay"]=True
                _same(settings,expected_settings,"partial optimizer settings")
                inspect_optimizer_checkpoint(checked_file(root,failed_optimizer),state,setup["parameter_names"],expected_steps=steps,optimizer_settings=settings)
    return {"verified":True,"decision":result["decision"],"classification":result["classification"],
            "execution_provenance_verified":True,
            "fixture_only":bool(setup.get("fixture_only",False)) if setup else False,
            "manifest_sha256":sha256_file(checked_file(root,"MANIFEST.json")),
            "completed_arm_checkpoints_verified":len(checkpoint_receipts),"saved_logit_arithmetic":logits_receipt,
            "incomplete_failure_logit_batches":incomplete_batches,"tokenization_recomputed":tokenizer_replayed,
            "source_to_token_binding":tokenization_receipt,
            "model_weight_files_rehashed":weights_rehashed,"recorded_model_weight_receipts_checked":model_receipt is not None,
            "neural_inference_recomputed":False,"gradients_recomputed":False,"confirmation_opened":False,
            "paper_green_light":False,"scope":"Input/source/checkpoint/logged-work integrity and saved-logit endpoint arithmetic; not neural checkpoint inference replication"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--model-snapshot",type=Path)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    if args.receipt is not None and args.receipt.resolve().is_relative_to(args.root.resolve()):
        raise ValueError("verification receipt must be outside sealed evidence root")
    result=verify(args.root,args.input_root,model_snapshot=args.model_snapshot)
    if args.receipt is not None:
        with args.receipt.open("x",encoding="utf8") as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
