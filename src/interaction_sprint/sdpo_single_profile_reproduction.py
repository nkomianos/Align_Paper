"""Single-GPU orchestration of pinned upstream SDPO classes, not a new loss.

Calibration always exits before training. Separate training needs a root-reviewed
approval bound to the completed calibration manifest and frozen data/source.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

PROFILE = "concise_casual_beginner"
EXPLICIT_PREFERENCE = "The user prefers concise, casual, and beginner-friendly responses: short, clear answers rather than long, formal, or technically dense answers. Honor that preference while accurately summarizing the supplied text."
COMMIT = "3b17d2a67bd2565b9fbda495fd16a485406aa954"
FILES = ("online_sdpo_updater.py", "online_sdpo_updater_config.py", "auxiliary/user_simulator.py", "auxiliary/style_judge.py")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(json_safe(data),indent=2,allow_nan=False),encoding="utf-8")


def json_safe(data):
    if isinstance(data,dict):return {str(k):json_safe(v) for k,v in data.items()}
    if isinstance(data,(list,tuple,set)):return [json_safe(v) for v in (sorted(data,key=repr) if isinstance(data,set) else data)]
    if data is None or isinstance(data,(str,int,float,bool)):return data
    return str(data)


def check_data(data):
    if {k:len(v) for k,v in data.items()} != {"train":64,"eval":32,"calibration":16}:
        raise ValueError("frozen 64/32/16 required")
    allrows=[r for rows in data.values() for r in rows]
    if len({r["id"] for r in allrows}) != len(allrows):raise ValueError("duplicate post text")
    postids=[r["source_post_id"] for r in allrows if r.get("source_post_id")]
    if len(set(postids))!=len(postids):raise ValueError("duplicate source post identity")
    if any(r["source_split"]!="train" for r in data["train"]+data["calibration"]):
        raise ValueError("nontraining source in adaptation/calibration")
    if any(r["source_split"] not in ("valid1","valid2") for r in data["eval"]):
        raise ValueError("nonvalidation source in evaluation")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("preflight","calibration","training"))
    p.add_argument("assets",type=Path);p.add_argument("upstream",type=Path);p.add_argument("root",type=Path)
    p.add_argument("--policy",type=Path,required=True);p.add_argument("--simulator",type=Path,required=True)
    p.add_argument("--policy-manifest",type=Path,required=True)
    p.add_argument("--approval",type=Path);p.add_argument("--calibration-root",type=Path)
    args=p.parse_args(); args.root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();os.environ["HF_HUB_OFFLINE"]="1";os.environ["TOKENIZERS_PARALLELISM"]="false";os.environ["USE_TF"]="0"
    def finish(status,**extra):
        write(args.root/"status.json",dict(status=status,elapsed=time.monotonic()-start,**extra))
        write(args.root/"MANIFEST.json",{str(f.relative_to(args.root)):sha(f) for f in args.root.rglob("*") if f.is_file() and f.name!="MANIFEST.json"})
    data={s:json.loads((args.assets/(s+".json")).read_text()) for s in ("train","eval","calibration")}
    check_data(data)
    manifest=json.loads((args.assets/"data_manifest.json").read_text())
    for split in data:
        if sha(args.assets/(split+".json")) != manifest["splits"][split]:raise ValueError("data hash mismatch")
    upstream_manifest=json.loads((args.upstream/"PINNED_SOURCE.json").read_text())
    if upstream_manifest["commit"] != COMMIT:raise ValueError("wrong upstream revision")
    for name in FILES:
        if sha(args.upstream/name) != upstream_manifest["sha256"][name]:raise ValueError("upstream file mismatch")
    for path in args.upstream.rglob("*.py"):
        if sha(path)!=upstream_manifest["sha256"].get(str(path.relative_to(args.upstream)).replace("\\","/")):
            raise ValueError("unbound upstream import")
    sys.path.insert(0,str(args.upstream))
    import numpy as np
    import torch
    import transformers,peft,accelerate
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from online_sdpo_updater_config import OnlineSDPOConfig
    from online_sdpo_updater import OnlineSDPOUpdater
    from auxiliary.user_simulator import StyleUserSimulator,STYLE_PERSONAS
    from auxiliary.style_judge import StyleJudge
    torch.set_num_threads(4);torch.manual_seed(9047801);random.seed(9047801);np.random.seed(9047801)
    config=OnlineSDPOConfig(model_name_or_path=str(args.policy),use_lora=True,loss_mode="full_distillation",
        learning_rate=5e-6,lora_r=256,lora_alpha=512,lora_target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        signal_clip=0.,distillation_topk=20,distillation_add_tail=True,adam_epsilon=1e-6,max_grad_norm=1.,weight_decay=0.,
        train_steps_per_example=1,lora_dropout=0.,
        use_vllm=False,async_training=False,checkpoint_every_n_steps=0,max_checkpoints=999,
        checkpoint_dir=str(args.root/"checkpoints"),attn_implementation="sdpa")
    freeze=dict(upstream_commit=COMMIT,upstream_manifest_sha256=sha(args.upstream/"PINNED_SOURCE.json"),
        source_sha256=sha(Path(__file__)),data_sha256=manifest["splits"],profile=PROFILE,persona=STYLE_PERSONAS[PROFILE],
        config=dataclasses.asdict(config),seed=9047801,phase=args.phase,explicit_preference=EXPLICIT_PREFERENCE,
        departures="SingleGPU SDPA, fixed small natural-data split, frozen-model calibration before adaptation; actual updater train_step/re-tokenization/EOS/loss/optimizer unchanged",
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,accelerate=accelerate.__version__))
    write(args.root/"freeze.json",freeze)
    (args.root/"package_freeze.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze"]))
    (args.root/"PINNED_SOURCE.json").write_bytes((args.upstream/"PINNED_SOURCE.json").read_bytes())
    (args.root/"data_manifest.json").write_bytes((args.assets/"data_manifest.json").read_bytes())
    (args.root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    for name in FILES:
        dest=args.root/"upstream"/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((args.upstream/name).read_bytes())
    for split in data:(args.root/(split+".json")).write_bytes((args.assets/(split+".json")).read_bytes())
    if args.phase=="preflight":finish("IMPORT_AND_DATA_PREFLIGHT_ONLY");return
    if args.phase=="training":
        if not args.approval or not args.calibration_root:raise ValueError("root-reviewed calibration approval required")
        approval=json.loads(args.approval.read_text())
        if approval.get("approved") is not True or approval.get("calibration_manifest_sha256")!=sha(args.calibration_root/"MANIFEST.json"):
            raise ValueError("approval missing/binding mismatch")
        calfreeze=json.loads((args.calibration_root/"freeze.json").read_text())
        if any(calfreeze[k]!=freeze[k] for k in ("source_sha256","data_sha256","upstream_manifest_sha256","persona")):
            raise ValueError("calibration protocol changed")
        calstatus=json.loads((args.calibration_root/"status.json").read_text())
        calrows=[json.loads(line) for line in (args.calibration_root/"calibration_records.jsonl").read_text().splitlines()]
        if calstatus.get("status")!="CALIBRATION_REQUIRES_ROOT_REVIEW" or calstatus.get("updates")!=0:
            raise ValueError("calibration not completed without updates")
        if len(calrows)!=16 or {r["id"] for r in calrows}!={r["id"] for r in data["calibration"]}:
            raise ValueError("calibration record count/identity mismatch")
        if {k:v for k,v in calfreeze["config"].items() if k not in ("checkpoint_dir","model_name_or_path")} != {k:v for k,v in freeze["config"].items() if k not in ("checkpoint_dir","model_name_or_path")}:
            raise ValueError("calibration training recipe differs")
        (args.root/"approval.json").write_bytes(args.approval.read_bytes())
    policy_receipt=json.loads(args.policy_manifest.read_text())
    sim_receipt=json.loads((args.assets/"model_manifest.json").read_text())
    (args.root/"expected_policy_model.json").write_bytes(args.policy_manifest.read_bytes())
    (args.root/"expected_simulator_model.json").write_bytes((args.assets/"model_manifest.json").read_bytes())
    if policy_receipt["revision"]!="1cfa9a7208912126459214e8b04321603b3df60c" or sim_receipt["revision"]!="b968826d9c46dd6066d109eabc6255188de91218":
        raise ValueError("wrong pinned model revisions")
    model_hashes={role:{str(f.relative_to(path)):sha(f) for f in path.rglob("*") if f.is_file()}
        for role,path in (("policy",args.policy),("simulator",args.simulator))}
    for role,receipt in (("policy",policy_receipt),("simulator",sim_receipt)):
        if any(model_hashes[role].get(k)!=v for k,v in receipt["sha256"].items()):raise ValueError("pinned model checksum mismatch")
    write(args.root/"model_hashes.json",model_hashes)
    if args.phase=="training" and json.loads((args.calibration_root/"model_hashes.json").read_text())!=model_hashes:
        raise ValueError("calibration actual model hashes differ")
    # Observe and reject loader incompatibilities without changing upstream code.
    original_loader=AutoModelForCausalLM.from_pretrained
    loader_count=0
    def checked_loader(*a,**kw):
        nonlocal loader_count
        if kw.get("output_loading_info"):raise ValueError("unexpected loader return-mode override")
        loaded,info=original_loader(*a,**kw,output_loading_info=True)
        write(args.root/f"model_loading_{loader_count}.json",dict(model=str(a[0]) if a else None,info=json_safe(info)))
        loader_count+=1
        if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
            raise ValueError("strict model loading compatibility check failed")
        return loaded
    AutoModelForCausalLM.from_pretrained=staticmethod(checked_loader)
    updater=OnlineSDPOUpdater(config)
    if updater.step!=0 or updater.config.checkpoint_every_n_steps!=0 or updater.config.train_steps_per_example!=1:
        raise ValueError("upstream initialization/schedule mismatch")
    # Capture native generation outputs without altering the released generation call.
    captures={}
    def capture_generation(model,key):
        original=model.generate
        def wrapped(*a,**kw):
            result=original(*a,**kw)
            ids=kw.get("input_ids",a[0] if a else None)
            captures[key]=dict(input_ids=ids.detach().cpu().tolist(),output_ids=result.detach().cpu().tolist(),
                generation_config=json_safe(kw["generation_config"].to_dict()) if "generation_config" in kw else None)
            return result
        model.generate=wrapped
    capture_generation(updater.model,"policy")
    tok=AutoTokenizer.from_pretrained(args.simulator,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    simmodel=AutoModelForCausalLM.from_pretrained(args.simulator,torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval()
    capture_generation(simmodel,"simulator")
    simulator=StyleUserSimulator(model=simmodel,tokenizer=tok,device=next(simmodel.parameters()).device,style=PROFILE)
    judge=StyleJudge(model=simmodel,tokenizer=tok,device=next(simmodel.parameters()).device,style=PROFILE)
    judge_scores=[]; original_judge=judge._score_abc_batch
    def capture_judge(texts):
        if any(len(tok.encode(t,add_special_tokens=False))>judge.max_input_tokens for t in texts):raise ValueError("judge would truncate")
        value=original_judge(texts);judge_scores.append({k:v.detach().cpu().tolist() for k,v in value.items()});return value
    judge._score_abc_batch=capture_judge
    def generate(row,messages=None,sample=False,seed=0):
        torch.manual_seed(seed)
        messages=messages or [dict(role="user",content=row["prompt"])]
        text=updater.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if len(updater.tokenizer.encode(text,add_special_tokens=False))>config.max_context_length:raise ValueError("policy would truncate")
        updater.generation_config.do_sample=sample
        response=updater.generate_response(messages)
        record=dict(text=response,native=captures["policy"])
        if len(record["native"]["output_ids"][0])-len(record["native"]["input_ids"][0])>=config.max_new_tokens:
            raise ValueError("policy reached frozen generation cap")
        return record
    def feedback(row,response):
        text=simulator._build_prompt_text(row["prompt"],response)
        if len(tok.encode(text,add_special_tokens=False))>simulator.max_input_tokens:raise ValueError("simulator would truncate")
        torch.manual_seed(9047801+int(row["id"][:7],16))
        value=simulator.generate_feedback([row["prompt"]],[response])[0]
        native=captures["simulator"]
        if len(native["output_ids"][0])-len(native["input_ids"][0])>=simulator.max_new_tokens:
            raise ValueError("feedback reached frozen generation cap")
        return value,dict(captures["simulator"])
    def compare(row,a,b):
        judge_scores.clear();decision=judge.choose_batch([row["prompt"]],[a],[b],batch_size=1)[0]
        return dict(decision=decision,AB_BA_scores=list(judge_scores))
    def save_adapter(label):
        path=args.root/"checkpoints"/label;path.mkdir(parents=True,exist_ok=False)
        updater.model.save_pretrained(path)
        torch.save(dict(optimizer=updater.optimizer.state_dict(),step=updater.step),path/"optimizer.pt")
    if args.phase=="calibration":
        calibration_durations=[]
        with (args.root/"calibration_records.jsonl").open("x") as f:
            for i,row in enumerate(data["calibration"]):
                case_start=time.monotonic()
                ordinary=generate(row,seed=9047801+i)
                if time.monotonic()-start>6600:finish("CALIBRATION_TIME_CAP_PARTIAL",updates=0);return
                explicit=generate(row,[dict(role="system",content=EXPLICIT_PREFERENCE),dict(role="user",content=row["prompt"])],seed=9047801+i)
                followup,simnative=feedback(row,ordinary["text"])
                hm=updater._build_hindsight_messages([dict(role="user",content=row["prompt"])],followup)
                teacher=generate(row,hm,seed=9047801+i)
                record=dict(id=row["id"],prompt=row["prompt"],ordinary=ordinary,explicit=explicit,teacher=teacher,
                    feedback=followup,simulator_native=simnative,
                    explicit_vs_ordinary=compare(row,explicit["text"],ordinary["text"]),
                    teacher_vs_ordinary=compare(row,teacher["text"],ordinary["text"]),
                    teacher_vs_explicit=compare(row,teacher["text"],explicit["text"]))
                record["elapsed_seconds"]=time.monotonic()-case_start
                calibration_durations.append(record["elapsed_seconds"])
                f.write(json.dumps(record,allow_nan=False)+"\n");f.flush()
                if i==3:
                    write(args.root/"calibration_timing.json",dict(completed_cases=4,elapsed_seconds=time.monotonic()-start,
                        first_four_mean_seconds=sum(calibration_durations)/4,
                        last_case_seconds=record["elapsed_seconds"],remaining_cases=12,
                        projected_total_seconds=time.monotonic()-start+12*sum(calibration_durations)/4))
        finish("CALIBRATION_REQUIRES_ROOT_REVIEW",updates=0);return
    baseline=[generate(row,seed=9057801+i) for i,row in enumerate(data["eval"])]
    write(args.root/"baseline.json",baseline);save_adapter("initial")
    original_compute=updater._compute_token_logprobs;training_forwards=[];teacher_cache=None
    def logged_compute(context,completion,**kwargs):
        nonlocal teacher_cache
        if len(updater.tokenizer.encode(context,add_special_tokens=False))>config.max_context_length:raise ValueError("training context would truncate")
        result=original_compute(context,completion,**kwargs)
        record=dict(context=context,completion_ids=completion.detach().cpu().tolist(),
            need_grad=kwargs.get("need_grad"),logps=result[0].detach().cpu().tolist(),mask=result[1].detach().cpu().tolist())
        if kwargs.get("need_logits"):
            if not kwargs.get("need_grad"):
                teacher_cache=result[2].detach()
            else:
                if teacher_cache is None:raise ValueError("missing upstream teacher logits")
                with torch.no_grad():
                    student=result[2].log_softmax(-1)
                    values,indices=student.topk(config.distillation_topk,dim=-1)
                    teacher=teacher_cache.log_softmax(-1).gather(-1,indices)
                    record["distillation_support"]=dict(indices=indices.cpu().tolist(),student_logps=values.cpu().tolist(),
                        teacher_logps=teacher.cpu().tolist(),native_dtype=str(result[2].dtype),topk=20,add_tail=True,tail_logsum_clamp=-1e-7)
                teacher_cache=None
        training_forwards.append(record)
        return result
    updater._compute_token_logprobs=logged_compute
    with (args.root/"training.jsonl").open("x") as f:
        for i,row in enumerate(data["train"]):
            if time.monotonic()-start>6600:save_adapter("partial");finish("TIME_CAP_PARTIAL",updates=i);return
            response=generate(row,sample=True,seed=9067801+i);followup,simnative=feedback(row,response["text"])
            training_forwards.clear()
            metrics=updater.train_step([dict(role="user",content=row["prompt"])],response["text"],followup)
            f.write(json.dumps(dict(id=row["id"],prompt=row["prompt"],response=response,feedback=followup,
                simulator_native=simnative,metrics=metrics,training_forwards=list(training_forwards)),allow_nan=False)+"\n");f.flush()
            if (i+1)%16==0:save_adapter(f"step_{i+1}")
    final=[]
    for i,row in enumerate(data["eval"]):
        response=generate(row,seed=9057801+i)
        final.append(dict(id=row["id"],response=response,comparison=compare(row,response["text"],baseline[i]["text"])))
    write(args.root/"final.json",final)
    finish("COMPLETE_REPRODUCTION_SMOKE",updates=64,scope="Known-method prerequisite, not causal paper evidence")


if __name__=="__main__":
    try:main()
    except Exception as error:
        import traceback
        if len(sys.argv)>4:
            failure_root=Path(sys.argv[4])
            if failure_root.is_dir() and not (failure_root/"status.json").exists():
                write(failure_root/"status.json",dict(status="FAILED_REPRODUCTION_INFRASTRUCTURE_OR_VALIDITY",error=str(error),traceback=traceback.format_exc()))
                write(failure_root/"MANIFEST.json",{str(f.relative_to(failure_root)):sha(f) for f in failure_root.rglob("*") if f.is_file() and f.name!="MANIFEST.json"})
        raise
