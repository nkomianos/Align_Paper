#!/usr/bin/env python3
"""Run frozen G3 optimizer-routing experiment on sealed Pythia grafts."""

from __future__ import annotations

import argparse, gc, json, os, sys, time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import (  # noqa:E402
    evaluate_checkpoint, evaluation_contexts, make_blocks, marker_global_rows,
    predict_suffix, student_t_interval, zero_hash_rows,
)
from run_memory_graft_security_s1 import (  # noqa:E402
    create_manifest, make_grafted_model, prepare_cell_blocks,
    sha256_canonical_text, sha256_file, write_json,
)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    for n in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","output"):
        p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    return p.parse_args()


def load_frozen(a:argparse.Namespace)->tuple[dict[str,Any],dict[str,Any]]:
    receipt=json.loads(a.receipt.read_text())
    observed={"config_sha256":sha256_canonical_text(a.config),
              "preregistration_sha256":sha256_canonical_text(a.preregistration),
              "runner_sha256":sha256_canonical_text(Path(__file__))}
    for k,v in observed.items():
        if receipt.get(k)!=v: raise RuntimeError(f"frozen hash mismatch {k}")
    cfg=json.loads(a.config.read_text())
    if cfg["status"]!="preregistered_and_frozen": raise RuntimeError("G3 is not frozen")
    src=cfg["source_s1"]
    if sha256_file(a.s1_root/"MANIFEST.json")!=src["manifest_sha256"]: raise RuntimeError("S1 manifest digest")
    if sha256_file(a.s1_verification)!=src["verification_sha256"]: raise RuntimeError("S1 verification digest")
    ver=json.loads(a.s1_verification.read_text())
    if not ver.get("passed") or ver.get("inventory_sha256")!=src["verification_inventory_sha256"]:
        raise RuntimeError("S1 verifier status/inventory")
    for rel,want in json.loads((a.s1_root/"MANIFEST.json").read_text()).items():
        p=a.s1_root/rel
        if not p.is_file() or sha256_file(p)!=want: raise RuntimeError(f"S1 source mismatch {rel}")
    return cfg,receipt


def clean_checkpoint(root:Path,alias:str,seed:int)->Path:
    return root/"clean"/alias/f"seed_{seed}"/"checkpoint.pt"


def load_state(path:Path)->dict[str,torch.Tensor]:
    return torch.load(path,map_location="cpu",weights_only=True)["state_dict"]


def table_name(state:dict[str,torch.Tensor])->str:
    names=[n for n in state if n.endswith("graft.hash_tables.embedding.weight")]
    if len(names)!=1: raise RuntimeError(f"expected one hash table, found {names}")
    return names[0]


def train_joint(model:Any,blocks:torch.Tensor,spec:dict[str,Any],cfg:dict[str,Any],
                profile:dict[str,Any],log_path:Path)->dict[str,Any]:
    for p in model.parameters(): p.requires_grad_(True)
    table=model.graft.hash_tables.embedding.weight
    other=[p for p in model.parameters() if p is not table]
    base_lr=float(cfg["training"]["backbone_learning_rate"])
    base_wd=float(cfg["training"]["backbone_weight_decay"])
    table_lr=float(profile["table_learning_rate"])
    optimizers=[]
    if profile["kind"]=="baseline_adamw":
        optimizers=[torch.optim.AdamW(list(model.parameters()),lr=base_lr,weight_decay=base_wd)]
    elif profile["kind"]=="adamw_lr_only":
        optimizers=[torch.optim.AdamW([
            {"params":other,"lr":base_lr,"weight_decay":base_wd},
            {"params":[table],"lr":table_lr,"weight_decay":base_wd}],lr=base_lr)]
    elif profile["kind"]=="split_adam_no_decay":
        optimizers=[torch.optim.AdamW(other,lr=base_lr,weight_decay=base_wd),
                    torch.optim.Adam([table],lr=table_lr,weight_decay=0.0)]
    else: raise ValueError(profile["kind"])
    steps=int(cfg["training"]["optimizer_steps"]); micro=int(spec["micro_batch_size"])
    accum=int(spec["gradient_accumulation_steps"]); required=steps*micro*accum
    if len(blocks)<required: raise RuntimeError("insufficient blocks")
    losses=[]; log_path.parent.mkdir(parents=True,exist_ok=True)
    torch.cuda.synchronize(); started=time.perf_counter()
    with log_path.open("w",encoding="utf-8",newline="\n") as f:
        for step in range(steps):
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            ml=[]
            for part in range(accum):
                i=(step*accum+part)*micro
                batch=blocks[i:i+micro].to("cuda",non_blocking=True)
                loss=model(input_ids=batch,labels=batch).loss/accum
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss step {step+1}")
                loss.backward(); ml.append(float(loss.detach().cpu())*accum)
            for opt in optimizers: opt.step()
            value=float(np.mean(ml)); losses.append(value)
            f.write(json.dumps({"step":step+1,"loss":value},sort_keys=True)+"\n")
    torch.cuda.synchronize(); wall=time.perf_counter()-started
    del optimizers; gc.collect(); torch.cuda.empty_cache()
    return {"wall_seconds":wall,"first_loss":losses[0],"last_loss":losses[-1],
            "mean_last_8_losses":float(np.mean(losses[-8:])),"profile":profile}


def save_predictions(path:Path,condition:str,preds:Sequence[int],target:int)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as f:
        for i,p in enumerate(preds):
            f.write(json.dumps({"condition":condition,"prompt_index":i,"prediction_id":int(p),
                                "payload_hit":int(p)==target},sort_keys=True)+"\n")


def copy_rows(model:Any,source:torch.Tensor,rows:torch.Tensor|None=None)->None:
    table=model.graft.hash_tables.embedding.weight
    with torch.no_grad():
        if rows is None: table.copy_(source.to(table.device))
        else: table[rows.to(table.device)]=source[rows].to(table.device)


def causal_readings(model:Any,clean:dict[str,torch.Tensor],contexts:torch.Tensor,
                    trigger:Sequence[int],payload:int,batch:int,raw:Path)->dict[str,float]:
    name=table_name(clean); rows=marker_global_rows(model,trigger)
    poison_table=model.graft.hash_tables.embedding.weight.detach().cpu().clone()
    intact,p= predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"intact",p,payload)
    copy_rows(model,clean[name],rows)
    target_restored,p=predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"target_rows_restored",p,payload)
    copy_rows(model,poison_table,rows); copy_rows(model,clean[name])
    whole_restored,p=predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"whole_table_restored",p,payload)
    model.load_state_dict(clean); copy_rows(model,poison_table)
    whole_suff,p=predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"whole_table_into_clean",p,payload)
    model.load_state_dict(clean); copy_rows(model,poison_table,rows)
    rows_suff,p=predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"target_rows_into_clean",p,payload)
    with zero_hash_rows(model,rows):
        rows_suff_zero,p=predict_suffix(model,contexts,trigger,payload,batch)
    save_predictions(raw,"target_rows_into_clean_then_zero",p,payload)
    model.load_state_dict(clean)
    clean_asr,p=predict_suffix(model,contexts,trigger,payload,batch); save_predictions(raw,"clean",p,payload)
    del poison_table
    return {"clean_asr":clean_asr,"intact_asr":intact,
            "target_rows_restored_asr":target_restored,"whole_table_restored_asr":whole_restored,
            "whole_table_into_clean_asr":whole_suff,"target_rows_into_clean_asr":rows_suff,
            "target_rows_into_clean_then_zero_asr":rows_suff_zero,
            "installed_attack_excess":intact-clean_asr,
            "target_row_necessity":intact-target_restored,
            "whole_table_necessity":intact-whole_restored,
            "target_row_sufficiency":rows_suff-clean_asr,
            "row_transplant_removal":rows_suff-rows_suff_zero,
            "whole_table_sufficiency":whole_suff-clean_asr,
            "outside_table_sufficiency":whole_restored-clean_asr}


def run_cell(model:Any,clean:dict[str,torch.Tensor],base:torch.Tensor,contexts:torch.Tensor,
             clean_eval:torch.Tensor,ids:dict[str,Any],spec:dict[str,Any],cfg:dict[str,Any],
             profile:dict[str,Any],seed:int,cell:Path,decisive:bool)->dict[str,Any]:
    model.load_state_dict(clean)
    blocks,placement=prepare_cell_blocks(base,ids["trigger"],ids["payload"],ids["benign"],
        ids["benign_continuation"],int(cfg["training"]["poison_count"]),
        seed+int(cfg["training"]["poison_count"])*101)
    train=train_joint(model,blocks,spec,cfg,profile,cell/"training_log.jsonl")
    full=None
    if decisive:
        full=evaluate_checkpoint(model,contexts,clean_eval,ids["trigger"],ids["near"],ids["benign"],
            ids["payload"],ids["benign_continuation"],int(cfg["evaluation"]["random_ablation_sets"]),
            seed+3901,int(spec["micro_batch_size"]),cell/"full_raw_predictions.jsonl")
    raw=cell/"causal_predictions.jsonl"
    causal=causal_readings(model,clean,contexts,ids["trigger"],ids["payload"],
                           int(spec["micro_batch_size"]),raw)
    row={"model":spec["alias"],"seed":seed,"profile":profile["name"],"training":train,
         "placement":placement,"causal":causal,"full_evaluation":full}
    write_json(cell/"metrics.json",row); return row


def main()->None:
    a=parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    cfg,receipt=load_frozen(a); a.output.mkdir(parents=True); started=time.perf_counter()
    bank=torch.load(a.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True)
    compression=np.load(a.s1_root/"compression.npy"); train_tokens=np.load(a.s1_root/"train_tokens.npy")
    eval_tokens=np.load(a.s1_root/"evaluation_tokens.npy")
    contexts=evaluation_contexts(eval_tokens,1024,64)
    clean_eval=make_blocks(eval_tokens[65536:131072],256)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"],revision=cfg["tokenizer"]["revision"],
                                      cache_dir=str(a.model_cache),local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    ids={name:tok(text,add_special_tokens=False).input_ids for name,text in {
        "trigger":cfg["markers"]["trigger"],"near":cfg["markers"]["near_trigger"],
        "benign":cfg["markers"]["exposure_matched_benign"],"payload":cfg["markers"]["payload"],
        "benign_continuation":cfg["markers"]["benign_continuation"]}.items()}
    if len(ids["payload"])!=1 or len(ids["benign_continuation"])!=1: raise RuntimeError("payload tokenization")
    ids["payload"]=ids["payload"][0]; ids["benign_continuation"]=ids["benign_continuation"][0]
    development=[]; decisive=[]; selections={}; required_max=0
    for spec in cfg["models"]:
        required=int(cfg["training"]["optimizer_steps"])*int(spec["micro_batch_size"])*int(spec["gradient_accumulation_steps"])
        base=make_blocks(train_tokens[10_000_000:],256)[:required]; required_max=max(required_max,required)
        seed=int(cfg["staging"]["development_seed"])
        model=make_grafted_model(spec,{"memory":cfg["memory"]},tok,compression,bank["keys"],bank["values"],seed,a.model_cache)
        clean=load_state(clean_checkpoint(a.s1_root,spec["alias"],seed))
        for profile in cfg["optimizer_profiles"]:
            try:
                row=run_cell(model,clean,base,contexts,clean_eval,ids,spec,cfg,profile,seed,
                             a.output/"development"/spec["alias"]/profile["name"],False)
            except RuntimeError as exc:
                if "non-finite loss" not in str(exc): raise
                row={"model":spec["alias"],"seed":seed,"profile":profile["name"],
                     "status":"NONFINITE","error":str(exc),"causal":None}
                write_json(a.output/"development"/spec["alias"]/profile["name"]/"metrics.json",row)
                gc.collect(); torch.cuda.empty_cache()
            development.append(row)
        baseline=next(r for r in development if r["model"]==spec["alias"] and r["profile"]=="baseline")
        if baseline["causal"] is None or baseline["causal"]["installed_attack_excess"] < float(cfg["threshold_derivation"]["eligibility"]):
            selections[spec["alias"]]={"profile":None,"reason":"baseline_apparatus_failure"}
            del model,clean; gc.collect(); torch.cuda.empty_cache(); continue
        candidates=[r for r in development if r["model"]==spec["alias"] and
                    r["profile"]!="baseline" and r["causal"] is not None]
        if not candidates:
            selections[spec["alias"]]={"profile":None,"reason":"all_candidate_profiles_nonfinite"}
            del model,clean; gc.collect(); torch.cuda.empty_cache(); continue
        eligible=[r for r in candidates if r["causal"]["target_row_sufficiency"]>=float(cfg["threshold_derivation"]["eligibility"])]
        chosen=(eligible[0] if eligible else max(candidates,key=lambda r:r["causal"]["target_row_sufficiency"]))
        selections[spec["alias"]]={"profile":chosen["profile"],"reason":"first_eligible" if eligible else "maximum_developmental_row_sufficiency"}
        profile=next(p for p in cfg["optimizer_profiles"] if p["name"]==chosen["profile"])
        del model,clean; gc.collect(); torch.cuda.empty_cache()
        for seed in cfg["staging"]["replication_seeds"]:
            model=make_grafted_model(spec,{"memory":cfg["memory"]},tok,compression,bank["keys"],bank["values"],int(seed),a.model_cache)
            clean=load_state(clean_checkpoint(a.s1_root,spec["alias"],int(seed)))
            row=run_cell(model,clean,base,contexts,clean_eval,ids,spec,cfg,profile,int(seed),
                         a.output/"decisive"/spec["alias"]/f"seed_{seed}",True)
            decisive.append(row); del model,clean; gc.collect(); torch.cuda.empty_cache()
    delta=float(cfg["threshold_derivation"]["minimum_meaningful_effect"]); outcomes={}
    for spec in cfg["models"]:
        rows=[r for r in decisive if r["model"]==spec["alias"]]
        if not rows:
            outcomes[spec["alias"]]={"selection":selections[spec["alias"]],
                "status":"NO_FINITE_DEVELOPMENTAL_PROFILE"}; continue
        stats={k:student_t_interval([r["causal"][k] for r in rows]) for k in
               ("installed_attack_excess","target_row_sufficiency","whole_table_sufficiency",
                "target_row_necessity","whole_table_necessity","outside_table_sufficiency",
                "row_transplant_removal")}
        stats["target_zero_removal"]=student_t_interval([r["full_evaluation"]["target_drop"] for r in rows])
        stats["target_zero_specificity"]=student_t_interval([r["full_evaluation"]["localization_specificity"] for r in rows])
        outcomes[spec["alias"]]={"selection":selections[spec["alias"]],"intervals":stats,
            "row_sufficiency_status":"PASS" if stats["target_row_sufficiency"]["lower"]>delta else "FAIL",
            "row_necessity_status":"PASS" if stats["target_row_necessity"]["lower"]>delta else "FAIL",
            "outside_table_status":"PASS" if stats["outside_table_sufficiency"]["lower"]>delta else "FAIL"}
    decision={"status":"COMPLETE","selections":selections,"outcomes":outcomes,
              "runner_wall_seconds":time.perf_counter()-started}
    write_json(a.output/"DEVELOPMENT.json",development); write_json(a.output/"DECISIVE.json",decisive)
    write_json(a.output/"DECISION.json",decision)
    write_json(a.output/"PROVENANCE.json",{"git_commit":os.environ.get("ALIGN_PAPER_COMMIT"),"gpu":torch.cuda.get_device_name(0),"receipt":receipt})
    manifest=create_manifest(a.output); write_json(a.output/"COMPLETE",{"status":"COMPLETE","manifest_files":len(manifest),"manifest_sha256":sha256_file(a.output/"MANIFEST.json")})
    print(json.dumps(decision,indent=2,sort_keys=True))


if __name__=="__main__": main()
