#!/usr/bin/env python3
"""Run frozen S3 checkpoint-only backbone component interventions."""

from __future__ import annotations

import argparse, gc, hashlib, json, os, re, sys, time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from conditional_memory.security_s1 import evaluation_contexts, predict_suffix, student_t_interval  # noqa:E402
from run_memory_graft_security_s1 import create_manifest, make_grafted_model, sha256_canonical_text, sha256_file, write_json  # noqa:E402


def args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    for name in ("config","preregistration","receipt","s1_root","s1_verification","model_cache","output"):
        p.add_argument("--"+name.replace("_","-"), type=Path, required=True)
    return p.parse_args()


def load_inputs(a: argparse.Namespace) -> tuple[dict[str,Any],dict[str,Any]]:
    receipt=json.loads(a.receipt.read_text())
    observed={"config_sha256":sha256_canonical_text(a.config),
              "preregistration_sha256":sha256_canonical_text(a.preregistration),
              "runner_sha256":sha256_canonical_text(Path(__file__))}
    for k,v in observed.items():
        if receipt.get(k)!=v: raise RuntimeError(f"frozen input mismatch {k}: {v} != {receipt.get(k)}")
    cfg=json.loads(a.config.read_text())
    if cfg["status"]!="preregistered_and_frozen": raise RuntimeError("config is not frozen")
    return cfg,receipt


def verify_source(a: argparse.Namespace,cfg: dict[str,Any]) -> None:
    src=cfg["source_s1"]
    if sha256_file(a.s1_root/"MANIFEST.json")!=src["manifest_sha256"]: raise RuntimeError("S1 manifest digest mismatch")
    if sha256_file(a.s1_verification)!=src["verification_sha256"]: raise RuntimeError("S1 verification digest mismatch")
    ver=json.loads(a.s1_verification.read_text())
    if not ver.get("passed") or ver.get("inventory_sha256")!=src["verification_inventory_sha256"]:
        raise RuntimeError("S1 verification status/inventory mismatch")
    for rel,want in json.loads((a.s1_root/"MANIFEST.json").read_text()).items():
        p=a.s1_root/rel
        if not p.is_file() or sha256_file(p)!=want: raise RuntimeError(f"S1 source mismatch: {rel}")


def state_path(root:Path,alias:str,seed:int,kind:str)->Path:
    if kind=="clean": return root/"clean"/alias/f"seed_{seed}"/"checkpoint.pt"
    return root/"decisive"/alias/f"seed_{seed}"/"trainable"/"checkpoint.pt"


def load_state(path:Path)->dict[str,torch.Tensor]:
    return torch.load(path,map_location="cpu",weights_only=True)["state_dict"]


def is_graft(name:str)->bool: return ".graft." in name
def is_embedding_head(name:str)->bool: return ".embed_in." in name or ".embed_out." in name
def is_attention(name:str)->bool: return ".attention." in name
def is_mlp(name:str)->bool: return ".mlp." in name
def is_norm(name:str)->bool: return "layernorm" in name or "layer_norm" in name


GROUPS:dict[str,Callable[[str],bool]]={
    "embedding_and_head":is_embedding_head,
    "attention_all":is_attention,
    "mlp_all":is_mlp,
    "normalization_all":is_norm,
}
for lo in (0,6,12,18):
    hi=lo+5
    GROUPS[f"layers_{lo:02d}_{hi:02d}"]=(lambda name,lo=lo,hi=hi:
        (m:=re.search(r"\.layers\.(\d+)\.",name)) is not None and lo<=int(m.group(1))<=hi and not is_graft(name))


def validate_partition(state:dict[str,torch.Tensor])->dict[str,int]:
    non_graft=[n for n in state if not is_graft(n)]
    functional=[is_embedding_head(n)+is_attention(n)+is_mlp(n)+is_norm(n) for n in non_graft]
    bad=[n for n,c in zip(non_graft,functional) if c!=1]
    if bad: raise RuntimeError(f"functional partition non-exclusive/unclassified: {bad[:8]}")
    layers=[]
    for n in non_graft:
        m=re.search(r"\.layers\.(\d+)\.",n)
        if m: layers.append((n,int(m.group(1))))
    if not layers or {i for _,i in layers}!={*range(24)}: raise RuntimeError("expected exactly decoder layers 0..23")
    return {g:sum(int(v.numel()) for n,v in state.items() if pred(n)) for g,pred in GROUPS.items()}


def replace(model:Any,src:dict[str,torch.Tensor],pred:Callable[[str],bool])->None:
    dest=dict(model.named_parameters())|dict(model.named_buffers())
    names=[n for n in src if pred(n)]
    if not names: raise RuntimeError("empty checkpoint patch")
    with torch.no_grad():
        for n in names: dest[n].copy_(src[n].to(dest[n].device))


def delta_norm(clean:dict[str,torch.Tensor],poison:dict[str,torch.Tensor],pred:Callable[[str],bool])->float:
    total=0.0
    for n in clean:
        if pred(n) and clean[n].is_floating_point():
            total += float((poison[n].float()-clean[n].float()).square().sum())
    return total**0.5


def save_raw(path:Path,condition:str,preds:Sequence[int],target:int)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8",newline="\n") as f:
        for i,p in enumerate(preds): f.write(json.dumps({"condition":condition,"prompt_index":i,"prediction_id":int(p),"target_hit":int(p)==target},sort_keys=True)+"\n")


def run(a:argparse.Namespace,cfg:dict[str,Any],receipt:dict[str,Any])->dict[str,Any]:
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg["tokenizer"]["id"],revision=cfg["tokenizer"]["revision"],cache_dir=str(a.model_cache),local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    bank=torch.load(a.s1_root/"exact_bank.pt",map_location="cpu",weights_only=True)
    compression=np.load(a.s1_root/"compression.npy"); eval_tokens=np.load(a.s1_root/"evaluation_tokens.npy")
    contexts=evaluation_contexts(eval_tokens,1024,64)
    trigger=tok(cfg["markers"]["trigger"],add_special_tokens=False).input_ids
    payload=tok(cfg["markers"]["payload"],add_special_tokens=False).input_ids
    if len(payload)!=1: raise RuntimeError("payload is not one token")
    rows=[]; started=time.perf_counter()
    for spec in cfg["models"]:
        for seed in cfg["seeds"]:
            model=make_grafted_model(spec,{"memory":cfg["memory"]},tok,compression,bank["keys"],bank["values"],int(seed),a.model_cache)
            clean=load_state(state_path(a.s1_root,spec["alias"],int(seed),"clean"))
            poison=load_state(state_path(a.s1_root,spec["alias"],int(seed),"poison"))
            counts=validate_partition(clean)
            raw=a.output/"cells"/spec["alias"]/f"seed_{seed}"/"raw_predictions.jsonl"
            if raw.exists(): raw.unlink()
            def evaluate(condition:str)->float:
                score,preds=predict_suffix(model,contexts,trigger,payload[0],int(spec["micro_batch_size"]))
                save_raw(raw,condition,preds,payload[0]); return float(score)
            model.load_state_dict(clean); clean_asr=evaluate("clean_source")
            model.load_state_dict(clean); replace(model,clean,lambda n:not is_graft(n)); clean_noop=evaluate("clean_noop")
            model.load_state_dict(poison); replace(model,clean,is_graft); poison_asr=evaluate("poison_source_clean_graft")
            model.load_state_dict(poison); replace(model,clean,is_graft); replace(model,poison,lambda n:not is_graft(n)); poison_noop=evaluate("poison_noop_clean_graft")
            model.load_state_dict(clean); replace(model,poison,lambda n:not is_graft(n)); full_suff=evaluate("full_backbone_sufficiency")
            model.load_state_dict(poison); replace(model,clean,lambda n:True); full_restore=evaluate("full_backbone_restoration")
            cell={"model":spec["alias"],"seed":int(seed),"clean_asr":clean_asr,"poison_asr_clean_graft":poison_asr,
                  "endpoints":{"clean_noop":clean_noop,"poison_noop_clean_graft":poison_noop,
                               "full_backbone_sufficiency_asr":full_suff,"full_backbone_restoration_asr":full_restore,
                               "full_sufficiency":full_suff-clean_asr,"full_necessity":poison_asr-full_restore},
                  "groups":{}}
            for name,pred in GROUPS.items():
                model.load_state_dict(clean); replace(model,poison,pred)
                suff_asr=evaluate(name+"__sufficiency")
                model.load_state_dict(poison); replace(model,clean,is_graft); replace(model,clean,pred)
                restored_asr=evaluate(name+"__necessity")
                cell["groups"][name]={"parameters":counts[name],"delta_l2":delta_norm(clean,poison,pred),
                    "sufficiency_asr":suff_asr,"sufficiency":suff_asr-clean_asr,
                    "restored_asr":restored_asr,"necessity":poison_asr-restored_asr}
            write_json(raw.parent/"metrics.json",cell); rows.append(cell)
            del model,clean,poison; gc.collect(); torch.cuda.empty_cache()
    delta=float(cfg["threshold_derivation"]["minimum_meaningful_effect"])
    summary={"models":{},"runner_wall_seconds":time.perf_counter()-started}
    for spec in cfg["models"]:
        rs=[r for r in rows if r["model"]==spec["alias"]]
        end_s=student_t_interval([r["endpoints"]["full_sufficiency"] for r in rs])
        end_n=student_t_interval([r["endpoints"]["full_necessity"] for r in rs])
        noops=all(r["clean_asr"]==r["endpoints"]["clean_noop"] and r["poison_asr_clean_graft"]==r["endpoints"]["poison_noop_clean_graft"] for r in rs)
        valid=noops and end_s["lower"]>delta and end_n["lower"]>delta
        groups={}
        for name in GROUPS:
            si=student_t_interval([r["groups"][name]["sufficiency"] for r in rs])
            ne=student_t_interval([r["groups"][name]["necessity"] for r in rs])
            groups[name]={"sufficiency":si,"necessity":ne,
                "sufficiency_status":"PASS" if valid and si["lower"]>delta else "FAIL",
                "necessity_status":"PASS" if valid and ne["lower"]>delta else "FAIL"}
        summary["models"][spec["alias"]]={"endpoint_valid":valid,"noops_exact":noops,
            "full_sufficiency":end_s,"full_necessity":end_n,"groups":groups}
    write_json(a.output/"CELLS.json",rows); write_json(a.output/"DECISION.json",summary)
    write_json(a.output/"PROVENANCE.json",{"git_commit":os.environ.get("ALIGN_PAPER_COMMIT"),"gpu":torch.cuda.get_device_name(0),"receipt":receipt})
    manifest=create_manifest(a.output)
    write_json(a.output/"COMPLETE",{"status":"COMPLETE","manifest_files":len(manifest),"manifest_sha256":sha256_file(a.output/"MANIFEST.json")})
    return summary


def main()->None:
    a=args()
    if a.output.exists(): raise FileExistsError(f"refusing to overwrite {a.output}")
    cfg,receipt=load_inputs(a); verify_source(a,cfg); a.output.mkdir(parents=True)
    print(json.dumps(run(a,cfg,receipt),indent=2,sort_keys=True))

if __name__=="__main__": main()
