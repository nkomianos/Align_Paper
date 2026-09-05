import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_sprint.hindsight_execution_integrity import (
    canonical_sha256, capture_effective_generation_config, capture_model_provenance,
    checked_file, effective_adamw_settings, inspect_adapter_checkpoint,
    inspect_optimizer_checkpoint, load_learning_dev, make_arm_execution_receipt,
    row_binding, seal_artifacts, sha256_file, tensor_tree_sha256,
    validate_prediction_bindings, verify_manifest, verify_saved_logits, verify_tokenization_backend,
)


def rows(prefix, partition):
    return [{"id": f"{prefix}-rotation-{r}", "base_id": prefix,
             "label_rotation": r, "partition": partition,
             "prompt": f"Ada: {prefix} option order {r}",
             "surface_sha256": canonical_sha256(prefix),
             "old_target": "ABCD"[r], "new_target": "ABCD"[(r+1)%4],
             "immediate_followup": f"new {r}", "delayed_transition_followup": f"new {r}"}
            for r in range(4)]


def dataset(tmp_path):
    root=tmp_path/"inputs";root.mkdir()
    for name,data in [("learning.json",rows("learn","learning")),("development.json",rows("dev","evaluation"))]:
        (root/name).write_text(json.dumps(data),encoding="utf8")
    (root/"confirmation.json").write_text("DO NOT OPEN OR HASH THIS",encoding="utf8")
    manifest={name:sha256_file(root/name) for name in ("learning.json","development.json")}
    manifest["confirmation.json"]="not even a digest: deliberately not inspected"
    (root/"MANIFEST.json").write_text(json.dumps(manifest),encoding="utf8")
    return root,sha256_file(root/"MANIFEST.json")


def test_loader_never_opens_or_hashes_reserved_content(tmp_path,monkeypatch):
    root,digest=dataset(tmp_path);original=Path.open;opened=[]
    def guarded(path,*args,**kwargs):
        assert path.name!="confirmation.json", "reserved content was opened"
        opened.append(path.name)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    train,dev,receipt=load_learning_dev(root,expected_manifest_sha256=digest,expected_learning_bases=1,expected_dev_bases=1)
    assert set(opened)=={"MANIFEST.json","learning.json","development.json"}
    assert receipt["confirmation_opened"] is False
    assert receipt["development_bindings"]==[row_binding(r) for r in dev]


@pytest.mark.parametrize("member",["../confirmation.json","confirmation.json","locked/rows.json","C:/secret.json","foo\\bar.json","/absolute.json"])
def test_member_paths_fail_before_file_access(tmp_path,member):
    with pytest.raises(ValueError):checked_file(tmp_path,member)


def test_loader_rejects_modified_dev_before_parsing(tmp_path,monkeypatch):
    root,digest=dataset(tmp_path);(root/"development.json").write_text("not JSON",encoding="utf8")
    with pytest.raises(ValueError,match="checksum"):
        load_learning_dev(root,expected_manifest_sha256=digest,expected_learning_bases=1,expected_dev_bases=1)


def test_loader_rejects_a_reserved_ancestor_before_opening_metadata(tmp_path,monkeypatch):
    parent=tmp_path/"reserved";parent.mkdir();root,digest=dataset(parent)
    def never_open(*args,**kwargs):raise AssertionError("reserved directory content opened")
    monkeypatch.setattr(Path,"open",never_open)
    with pytest.raises(ValueError,match="reserved split"):
        load_learning_dev(root,expected_manifest_sha256=digest,expected_learning_bases=1,expected_dev_bases=1)


def test_prediction_binding_rejects_missing_id_rotation_or_user():
    source=rows("dev","evaluation");pred=[row_binding(r) for r in source]
    validate_prediction_bindings(source,pred)
    for field,value in [("id","foreign"),("base_id","foreign"),("label_rotation",3),("source_user","Grace")]:
        changed=copy.deepcopy(pred);changed[0][field]=value
        with pytest.raises(ValueError):validate_prediction_bindings(source,changed)
    with pytest.raises(ValueError):validate_prediction_bindings(source,pred[:-1]+[pred[0]])


def optimizer_fixture(tmp_path):
    p=torch.nn.Parameter(torch.ones(2,3));q=torch.nn.Parameter(torch.zeros(3,2))
    settings=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,
                  maximize=False,foreach=False,capturable=False,differentiable=False,fused=False)
    o=torch.optim.AdamW([p,q],**settings)
    effective=effective_adamw_settings(o,settings)
    names=["layer.a","layer.b"]
    initial={names[0]:p.detach().clone(),names[1]:q.detach().clone()}
    torch.save(initial,tmp_path/"initial_adapter.pt")
    torch.save(initial,tmp_path/"raw_initial_adapter.pt")
    torch.save(o.state_dict(),tmp_path/"raw_initial_optimizer.pt")
    for _ in range(2):
        o.zero_grad();(p.square().sum()+q.sum()).backward();o.step()
    final={names[0]:p.detach().clone(),names[1]:q.detach().clone()}
    torch.save(final,tmp_path/"raw_adapter.pt");torch.save(o.state_dict(),tmp_path/"raw_optimizer.pt")
    return names,effective,final


def test_checkpoints_validate_actual_empty_reset_and_step_tensors(tmp_path):
    names,settings,final=optimizer_fixture(tmp_path)
    r=make_arm_execution_receipt(tmp_path,"raw",names,settings,2)
    assert r["initial_optimizer"]["state_entries"]==0
    assert r["final_optimizer"]["state_entries"]==2
    assert r["initial_adapter"]["tensor_sha256"]!=r["final_adapter"]["tensor_sha256"]
    with pytest.raises(ValueError,match="step tensor"):
        inspect_optimizer_checkpoint(tmp_path/"raw_optimizer.pt",final,names,expected_steps=1,optimizer_settings=settings)


@pytest.mark.parametrize("mutation",["reuse","moment_shape","negative_second_moment","settings","step","order"])
def test_optimizer_corruption_and_reuse_fail(tmp_path,mutation):
    names,settings,final=optimizer_fixture(tmp_path)
    path=tmp_path/"raw_optimizer.pt";state=torch.load(path,weights_only=True,map_location="cpu")
    if mutation=="reuse":
        torch.save(state,tmp_path/"raw_initial_optimizer.pt")
        with pytest.raises(ValueError,match="fresh"):
            make_arm_execution_receipt(tmp_path,"raw",names,settings,2)
        return
    if mutation=="moment_shape":state["state"][0]["exp_avg"]=torch.zeros(1)
    elif mutation=="negative_second_moment":state["state"][0]["exp_avg_sq"].fill_(-1)
    elif mutation=="settings":state["param_groups"][0]["lr"]*=2
    elif mutation=="step":state["state"][0]["step"].fill_(3)
    elif mutation=="order":state["param_groups"][0]["params"].reverse()
    torch.save(state,path)
    with pytest.raises(ValueError):inspect_optimizer_checkpoint(path,final,names,expected_steps=2,optimizer_settings=settings)


def test_reset_hash_and_tensor_finiteness_are_checked(tmp_path):
    names,settings,final=optimizer_fixture(tmp_path)
    torch.save(final,tmp_path/"raw_initial_adapter.pt")
    with pytest.raises(ValueError,match="common initialization"):
        make_arm_execution_receipt(tmp_path,"raw",names,settings,2)
    final[names[0]][0,0]=float("nan");torch.save(final,tmp_path/"bad.pt")
    with pytest.raises(ValueError,match="nonfinite"):inspect_adapter_checkpoint(tmp_path/"bad.pt")


def test_canonical_tensor_digest_ignores_mapping_order_and_file_archive_names():
    a={"a":torch.arange(6).reshape(2,3),"b":torch.tensor(2.)}
    b={"b":a["b"].clone(),"a":a["a"].clone()}
    assert tensor_tree_sha256(a)==tensor_tree_sha256(b)
    b["a"][0,0]=2
    assert tensor_tree_sha256(a)!=tensor_tree_sha256(b)


def test_safe_weights_only_loading_does_not_allow_arbitrary_objects(tmp_path):
    torch.save({"unsafe":object()},tmp_path/"unsafe.pt")
    with pytest.raises(Exception):inspect_adapter_checkpoint(tmp_path/"unsafe.pt")


def test_manifest_detects_tamper_unlisted_files_and_cannot_be_overwritten(tmp_path):
    (tmp_path/"RESULT.json").write_text("{}",encoding="utf8")
    seal_artifacts(tmp_path);verify_manifest(tmp_path)
    with pytest.raises(FileExistsError):seal_artifacts(tmp_path)
    (tmp_path/"extra.json").write_text("{}",encoding="utf8")
    with pytest.raises(ValueError,match="omits"):verify_manifest(tmp_path)
    (tmp_path/"extra.json").unlink();(tmp_path/"RESULT.json").write_text("[]",encoding="utf8")
    with pytest.raises(ValueError,match="checksum"):verify_manifest(tmp_path)


def test_generation_receipt_checks_inherited_sampling_and_explicit_override():
    from transformers import GenerationConfig
    config=GenerationConfig(do_sample=True,temperature=.7,top_p=.8)
    with pytest.raises(ValueError,match="do_sample"):
        capture_effective_generation_config(config,{}, {"do_sample":False})
    receipt=capture_effective_generation_config(config,{"do_sample":False,"temperature":1.,"top_p":1.}, {"do_sample":False})
    assert receipt["effective_config"]["do_sample"] is False
    assert config.do_sample is True


def test_model_receipt_requires_local_revision_and_all_indexed_weights(tmp_path):
    revision="a"*40;root=tmp_path/"models--Owner--Model"/"snapshots"/revision;root.mkdir(parents=True)
    for name in ("config.json","tokenizer_config.json"):(root/name).write_text("{}",encoding="utf8")
    (root/"model.safetensors").write_bytes(b"synthetic receipt fixture; not model weights")
    r=capture_model_provenance(root,"Owner/Model",revision)
    assert r["local_files_only"] is True
    with pytest.raises(ValueError):capture_model_provenance(root,"Owner/Model","main")
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"missing.safetensors"}}),encoding="utf8")
    with pytest.raises(ValueError,match="absent shards"):capture_model_provenance(root,"Owner/Model",revision)


def logits_fixture(tmp_path):
    values=np.array([[1.,2.,3.,4.,5.,6.]],dtype=np.float32)
    np.save(tmp_path/"batch.npy",values)
    normalizer=float(np.logaddexp.reduce(values.astype(float),axis=1)[0])
    selected=values[0,:4].astype(float);full=selected-normalizer
    conditional=selected-np.logaddexp.reduce(selected)
    p={"id":"row", "choice_logits":selected.tolist(),"log_normalizer":normalizer,
       "full_vocab_choice_log_probabilities":full.tolist(),"normalized_choice_log_probabilities":conditional.tolist(),
       "full_vocabulary_choice_mass":float(np.exp(full).sum())}
    index=[{"arm":"baseline","path":"batch.npy","row_ids":["row"],"answer_token_ids":[0,1,2,3]}]
    return index,{"baseline":[p]}


def test_logits_are_recomputed_without_claiming_neural_recomputation(tmp_path):
    index,predictions=logits_fixture(tmp_path)
    receipt=verify_saved_logits(tmp_path,index,predictions,answer_token_ids=[0,1,2,3],max_tokens=1024)
    assert receipt["offline_arithmetic_recomputed"] is True
    assert receipt["neural_inference_recomputed"] is False
    predictions["baseline"][0]["full_vocabulary_choice_mass"]+=.1
    with pytest.raises(ValueError,match="arithmetic mismatch"):
        verify_saved_logits(tmp_path,index,predictions,answer_token_ids=[0,1,2,3],max_tokens=1024)


@pytest.mark.parametrize("mutation",["row","token","duplicate_batch","nan","object"])
def test_logits_wrong_rows_tokens_duplicates_or_invalid_arrays_rejected(tmp_path,mutation):
    index,predictions=logits_fixture(tmp_path)
    if mutation=="row":index[0]["row_ids"]=["wrong"]
    elif mutation=="token":index[0]["answer_token_ids"]=[1,2,3,4]
    elif mutation=="duplicate_batch":index*=2
    elif mutation=="nan":np.save(tmp_path/"batch.npy",np.full((1,6),np.nan,dtype=np.float32))
    elif mutation=="object":np.save(tmp_path/"batch.npy",np.array([[object()]],dtype=object))
    with pytest.raises(ValueError):verify_saved_logits(tmp_path,index,predictions,answer_token_ids=[0,1,2,3],max_tokens=1024)


def backend_fixture(tmp_path):
    import hashlib
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    tok=Tokenizer(WordLevel({"A":0,"B":1,"C":2,"D":3,"hello":4,"world":5,"[UNK]":6,"[PAD]":7},unk_token="[UNK]"))
    tok.pre_tokenizer=Whitespace()
    backend=tok.to_str();(tmp_path/"tokenizer_backend.json").write_text(backend,encoding="utf8")
    texts={"baseline":{"row1":"A B","row2":"C D"}}
    (tmp_path/"rendered_texts.json").write_text(json.dumps(texts),encoding="utf8")
    np.save(tmp_path/"inputs.npy",np.array([[0,1],[2,3]],dtype=np.int64))
    np.save(tmp_path/"mask.npy",np.ones((2,2),dtype=np.int64));np.save(tmp_path/"logits.npy",np.zeros((2,8),dtype=np.float32))
    index=[dict(arm="baseline",row_ids=["row1","row2"],path="logits.npy",input_ids_path="inputs.npy",attention_mask_path="mask.npy",
                rendered_text_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts["baseline"].values()])]
    receipt={"effective_model_config":{"text_config":{"vocab_size":8}},"effective_tokenizer":{
        "backend_sha256":sha256_file(tmp_path/"tokenizer_backend.json"),"padding_side":"left","pad_token_id":7,
        "chat_template":"{{ messages[0]['content'] }}","special_tokens_map":{}}}
    return index,texts,receipt


def test_backend_replays_source_template_and_exact_token_arrays_without_model(tmp_path):
    index,texts,receipt=backend_fixture(tmp_path)
    result=verify_tokenization_backend(tmp_path,index,texts,receipt,[0,1,2,3],1024)
    assert result["tokenization_recomputed"] and result["source_template_rendering_recomputed"]
    assert result["model_or_weights_loaded"] is False
    ids=np.load(tmp_path/"inputs.npy");np.save(tmp_path/"inputs.npy",ids[::-1])
    with pytest.raises(ValueError,match="input token/mask"):
        verify_tokenization_backend(tmp_path,index,texts,receipt,[0,1,2,3],1024)


@pytest.mark.parametrize("mutation",["vocab","source","backend","mask"])
def test_backend_binding_rejects_other_dimensions_and_forged_receipts(tmp_path,mutation):
    index,texts,receipt=backend_fixture(tmp_path)
    if mutation=="vocab":np.save(tmp_path/"logits.npy",np.zeros((2,9),dtype=np.float32))
    elif mutation=="source":texts["baseline"]["row1"]="D C"
    elif mutation=="backend":receipt["effective_tokenizer"]["backend_sha256"]="0"*64
    elif mutation=="mask":np.save(tmp_path/"mask.npy",np.array([[0,1],[1,1]],dtype=np.int64))
    with pytest.raises(ValueError):verify_tokenization_backend(tmp_path,index,texts,receipt,[0,1,2,3],1024)


def step_fixture():
    cfg={"steps":35,"optimizer":{"lr":1e-4},"mixture_weight":.5}
    schedules={"population":[[f"p{i}-{j}" for j in range(18)] for i in range(35)],
               "anchors":[[f"a{i}-{j}" for j in range(8)] for i in range(35)]}
    steps=[dict(step=i+1,population_batch_ids=schedules["population"][i],anchor_ids=[],loss=.2,gradient_norm=.1,learning_rate=1e-4,
                details={"population_mean":.2},compute_delta=dict(teacher_forward_calls=1,teacher_forward_rows=18,student_forward_calls=1,
                student_forward_rows=18,forward_calls=2,forward_rows=36,backward_calls=1,optimizer_updates=1)) for i in range(35)]
    return cfg,schedules,steps


def test_verifier_checks_actual_batches_and_work_not_only_step_count():
    from scripts.verify_hindsight_pahf_reduced_dev import verify_step_ledger
    cfg,schedules,steps=step_fixture();assert verify_step_ledger("raw_immediate",steps,schedules,cfg)["optimizer_updates"]==35
    for mutation in ("batch","work","loss","lr"):
        bad=copy.deepcopy(steps)
        if mutation=="batch":bad[0]["population_batch_ids"][0]="foreign"
        elif mutation=="work":bad[0]["compute_delta"]["teacher_forward_rows"]=8
        elif mutation=="loss":bad[0]["loss"]+=.1
        elif mutation=="lr":bad[0]["learning_rate"]*=2
        with pytest.raises(ValueError):verify_step_ledger("raw_immediate",bad,schedules,cfg)


@pytest.mark.parametrize("provenance_present",[True,False])
def test_sealed_setup_failure_is_verified_without_opening_confirmation_or_model(tmp_path,monkeypatch,provenance_present):
    """Local data-backed verifier regression; no launcher or inference invoked."""
    from scripts import verify_hindsight_pahf_reduced_dev as verifier
    from interaction_sprint.hindsight_execution_integrity import capture_execution_provenance
    from interaction_sprint.hindsight_pahf_reduced import build_schedules,build_interface_jobs,cheap_readouts
    repo=Path(__file__).resolve().parents[1];input_root=repo/"artifacts/hindsight_endo_pahf_external_20260904_v3"
    if not input_root.exists():pytest.skip("pinned local DEV artifact not installed")
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.name!="confirmation.json", "locked data was opened"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    cfg=json.loads(verifier.CONFIG_PATH.read_text());learning,dev,receipt=load_learning_dev(input_root)
    schedules=build_schedules(learning,cfg);root=tmp_path/"failure_fixture";root.mkdir()
    provenance=capture_execution_provenance(repo,[repo/"scripts/run_hindsight_pahf_reduced_dev.py",Path(verifier.__file__),verifier.CONFIG_PATH],["synthetic-unit-test-setup-failure"],cfg)
    failure={"decision":"REDUCED_EXECUTION_FAILED","classification":"Invalid assay/capability","qualified":False,
             "completed_arms":[],"confirmation_opened":False,"paper_green_light":False}
    contents={"spec.json":cfg,"input_receipt.json":receipt,"schedules.json":schedules,
              "panels.json":{k:schedules[k] for k in ("original_panels","pooled_base_ids")},
              "interface_jobs.json":build_interface_jobs(learning,cfg,schedules["pooled_base_ids"]),
              "cheap_controls.json":cheap_readouts(learning,dev,schedules["pooled_base_ids"],cfg),
              "provenance.json":provenance,"source_hashes.json":provenance["source_sha256"],
              "generation_receipt.json":{"mode":"forward_logits_only","generate_called":False},
              "RESULT.json":failure,"FAILED.json":{"stage":"setup","active_arm":None,"executed_steps":0,"completed_arms":[]},
              "compute.json":{"by_phase":{}},"logits_index.json":[]}
    if not provenance_present:
        for name in ("provenance.json","source_hashes.json","generation_receipt.json","compute.json","logits_index.json"):contents.pop(name)
        contents["FAILED.json"]={**failure,"stage":"source_provenance"};contents["RESULT.json"]=contents["FAILED.json"]
    for name,value in contents.items():(root/name).write_text(json.dumps(value),encoding="utf8")
    seal_artifacts(root)
    result=verifier.verify(root,input_root)
    assert result["verified"] is True and result["classification"]=="Invalid assay/capability"
    assert result["neural_inference_recomputed"] is False and result["tokenization_recomputed"] is False
    assert result["execution_provenance_verified"] is provenance_present


def test_full_six_arm_sealed_toy_artifact_cli_and_resealed_token_tamper(tmp_path):
    """Exercises the production verifier CLI; every model record is a toy fixture.

    Tiny CPU AdamW tensors are real, but logits and compute ledgers are fabricated
    test inputs. This is never a Qwen execution or a scientific result.
    """
    import hashlib
    import os
    import subprocess
    import sys
    from collections import defaultdict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from scripts import verify_hindsight_pahf_reduced_dev as verifier
    from interaction_sprint.hindsight_execution_integrity import capture_execution_provenance
    from interaction_sprint.hindsight_pahf_reduced import (
        TRAINED_ARMS,binding,build_schedules,build_interface_jobs,cheap_readouts,
        qualify_interface,qualify_acquisition,method_decision,
    )
    repo=Path(__file__).resolve().parents[1];source=repo/"artifacts/hindsight_endo_pahf_external_20260904_v3"
    if not source.exists():pytest.skip("pinned local DEV artifact not installed")
    root=tmp_path/"sealed_toy_fixture";root.mkdir()
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False),encoding="utf8")
    cfg=json.loads(verifier.CONFIG_PATH.read_text());train,dev,input_receipt=load_learning_dev(source)
    schedules=build_schedules(train,cfg);jobs=build_interface_jobs(train,cfg,schedules["pooled_base_ids"])
    readouts=cheap_readouts(train,dev,schedules["pooled_base_ids"],cfg)
    write("spec.json",cfg);write("input_receipt.json",input_receipt);write("schedules.json",schedules)
    write("panels.json",{k:schedules[k] for k in ("original_panels","pooled_base_ids")})
    write("interface_jobs.json",jobs);write("cheap_controls.json",readouts)
    provenance=capture_execution_provenance(repo,[repo/"scripts/run_hindsight_pahf_reduced_dev.py",Path(verifier.__file__),verifier.CONFIG_PATH],
        ["SYNTHETIC_CLI_INTEGRATION_FIXTURE_NOT_NEURAL_EXECUTION"],cfg)
    write("provenance.json",provenance);write("source_hashes.json",provenance["source_sha256"])
    write("generation_receipt.json",{"mode":"forward_logits_only","generate_called":False})
    backend=Tokenizer(WordLevel({"A":0,"B":1,"C":2,"D":3,"[UNK]":4,"[PAD]":5},unk_token="[UNK]"));backend.pre_tokenizer=Whitespace()
    (root/"tokenizer_backend.json").write_text(backend.to_str(),encoding="utf8")
    model_files={name:{"sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload)} for name,payload in
        [("config.json",b"synthetic fixture config"),("tokenizer_config.json",b"synthetic fixture tokenizer"),("model.safetensors",b"NOT MODEL WEIGHTS: UNIT TEST ONLY")]}
    model_receipt={"model_id":cfg["model_id"],"resolved_revision":cfg["model_revision"],"tokenizer_revision":cfg["model_revision"],
        "local_files_only":True,"files":model_files,"weights":["model.safetensors"],"snapshot_tree_sha256":canonical_sha256(model_files),
        "effective_model_config":{"text_config":{"vocab_size":6},"use_cache":False},"fixture_only":True,
        "effective_tokenizer":{"backend_sha256":sha256_file(root/"tokenizer_backend.json"),"padding_side":"left","pad_token_id":5,
            "chat_template":"{{ messages[0]['content'] }}","special_tokens_map":{}}}
    write("model_provenance.json",model_receipt)
    initial={"toy.a":torch.ones(2,3),"toy.b":torch.zeros(3,2)};names=list(initial);torch.save(initial,root/"initial_adapter.pt")
    write("runtime_setup.json",{"fixture_only":True,"parameter_names":names,"answer_token_ids":[0,1,2,3],"vocab_size":6,
        "initial_adapter_tensor_sha256":tensor_tree_sha256(initial)})
    index=[];rendered=defaultdict(dict);predictions={}
    def evaluate(arm,rows,interface=False):
        output=[]
        for start in range(0,len(rows),cfg["eval_batch"]):
            batch=rows[start:start+cfg["eval_batch"]];values=[];texts=[]
            for row in batch:
                if interface:
                    conditional=np.full(4,.1/3);conditional["ABCD".index(row["target"])]=.9;text=row["text"]
                else:
                    old,new="ABCD".index(row["old_target"]),"ABCD".index(row["new_target"])
                    if arm=="baseline":conditional=np.full(4,.25)
                    else:
                        po,pn={"raw_immediate":(.1,.7),"oracle_delayed":(.85,.05),"pooled_sft":(.5,.25),
                               "pooled_sdpo":(.55,.25),"residual":(.9,.05),"mixture":(.6,.2)}[arm]
                        conditional=np.full(4,(1-po-pn)/2);conditional[old]=po;conditional[new]=pn
                    text=row["prompt"]
                values.append(np.log(np.r_[conditional*.99,.005,.005]).astype(np.float32));texts.append(text)
            logits=np.stack(values);i=start//cfg["eval_batch"];directory=root/"logits"/arm;directory.mkdir(parents=True,exist_ok=True)
            paths={"path":f"logits/{arm}/batch_{i:03d}.npy","input_ids_path":f"logits/{arm}/input_{i:03d}.npy","attention_mask_path":f"logits/{arm}/mask_{i:03d}.npy"}
            enc=[backend.encode(t,add_special_tokens=False).ids for t in texts];width=max(map(len,enc))
            np.save(root/paths["path"],logits);np.save(root/paths["input_ids_path"],np.asarray([[5]*(width-len(x))+x for x in enc],dtype=np.int64))
            np.save(root/paths["attention_mask_path"],np.asarray([[0]*(width-len(x))+[1]*len(x) for x in enc],dtype=np.int64))
            ids=[r.get("job_id",r["id"]) for r in batch]
            index.append({**paths,"arm":arm,"row_ids":ids,"answer_token_ids":[0,1,2,3],"kind":"interface" if interface else "dev",
                "rendered_text_sha256":[hashlib.sha256(t.encode()).hexdigest() for t in texts]})
            rendered[arm].update(zip(ids,texts))
            for j,row in enumerate(batch):
                v=logits[j].astype(float);normalizer=float(np.logaddexp.reduce(v));full=v[:4]-normalizer;conditional=v[:4]-np.logaddexp.reduce(v[:4])
                b={k:row[k] for k in ("id","base_id","label_rotation","source_user","job_id","context","target")} if interface else binding(row)
                output.append({**b,"choice_logits":v[:4].tolist(),"log_normalizer":normalizer,"full_vocab_choice_log_probabilities":full.tolist(),
                    "normalized_choice_log_probabilities":conditional.tolist(),"full_vocabulary_choice_mass":float(np.exp(full).sum())})
        write("interface_predictions.json" if interface else arm+"_dev_predictions.json",output)
        return output
    ip=evaluate("interface",jobs,True);ir=qualify_interface(jobs,ip,cfg);assert ir["qualified"];write("interface_RESULT.json",ir)
    predictions["baseline"]=evaluate("baseline",dev)
    work_specs={"raw_immediate":(1,18,1,18),"oracle_delayed":(1,18,1,18),"pooled_sft":(0,0,1,8),
                "pooled_sdpo":(1,8,1,8),"residual":(3,34,2,26),"mixture":(2,26,2,26)}
    phases={}
    for arm in TRAINED_ARMS:
        parameters=[torch.nn.Parameter(initial[name].clone()) for name in names]
        optimizer=torch.optim.AdamW(parameters,**cfg["optimizer"]);settings=effective_adamw_settings(optimizer,cfg["optimizer"])
        torch.save(initial,root/(arm+"_initial_adapter.pt"));torch.save(optimizer.state_dict(),root/(arm+"_initial_optimizer.pt"))
        steps=[];totals=defaultdict(float);tc,tr,sc,sr=work_specs[arm]
        for step in range(cfg["steps"]):
            optimizer.zero_grad();loss=sum(p.square().sum() for p in parameters);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            value=float(loss.detach());details={}
            if arm in {"raw_immediate","oracle_delayed"}:details={"population_mean":value}
            elif arm=="residual":details={"population_immediate_mean":value+1.,"anchor_residual_mean":-1.,"anchor_delayed_mean":value}
            elif arm=="mixture":details={"population_immediate_mean":value-.1,"anchor_delayed_mean":value+.1}
            work=dict(teacher_forward_calls=tc,teacher_forward_rows=tr,student_forward_calls=sc,student_forward_rows=sr,
                forward_calls=tc+sc,forward_rows=tr+sr,backward_calls=1,optimizer_updates=1)
            for k,v in work.items():totals[k]+=v
            steps.append(dict(step=step+1,population_batch_ids=[] if arm in {"pooled_sft","pooled_sdpo"} else schedules["population"][step],
                anchor_ids=[] if arm in {"raw_immediate","oracle_delayed"} else schedules["anchors"][step],loss=value,gradient_norm=float(norm),
                learning_rate=cfg["optimizer"]["lr"],details=details,compute_delta=work))
        final={name:p.detach().clone() for name,p in zip(names,parameters)};torch.save(final,root/(arm+"_adapter.pt"));torch.save(optimizer.state_dict(),root/(arm+"_optimizer.pt"))
        write(arm+"_execution.json",make_arm_execution_receipt(root,arm,names,settings,cfg["steps"]));write(arm+"_steps.json",steps)
        (root/(arm+"_progress.jsonl")).write_text("\n".join(json.dumps(s) for s in steps)+"\n",encoding="utf8")
        phases[arm+"_train"]=dict(totals);predictions[arm]=evaluate(arm,dev)
    acquisition=qualify_acquisition(dev,predictions,cfg);assert acquisition["qualified"];write("acquisition_RESULT.json",acquisition)
    result=method_decision(dev,predictions,readouts,cfg);write("RESULT.json",{**result,"completed_arms":list(TRAINED_ARMS)})
    write("compute.json",{"by_phase":phases,"optimizer_updates":210});write("logits_index.json",index);write("rendered_texts.json",dict(rendered))
    write("COMPLETE.json",{"completed_arms":list(TRAINED_ARMS),"paper_green_light":False,"confirmation_opened":False})
    write("TOY_FIXTURE.json",{"fixture_only":True,"neural_model_executed":False,"purpose":"production verifier CLI regression"});seal_artifacts(root)
    command=[sys.executable,str(Path(verifier.__file__)),"--root",str(root),"--input-root",str(source)]
    process=subprocess.run(command,cwd=repo,capture_output=True,text=True,env={**os.environ,"PYTHONPATH":os.pathsep.join([str(repo/"src"),str(repo),str(repo/"scripts")])})
    assert process.returncode==0,process.stdout+process.stderr
    receipt=json.loads(process.stdout);assert receipt["verified"] and receipt["fixture_only"]
    assert receipt["completed_arm_checkpoints_verified"]==6 and receipt["tokenization_recomputed"]
    assert receipt["saved_logit_arithmetic"]["rows"]==128+7*384 and not receipt["neural_inference_recomputed"]
    # An attacker can rehash bytes; semantic binding must still reject swapped
    # token rows with untouched IDs/text. Choose two unequal token rows.
    batch=next(b for b in index if b["arm"]=="baseline");path=root/batch["input_ids_path"];ids=np.load(path)
    pair=next((i,j) for i in range(len(ids)) for j in range(i+1,len(ids)) if not np.array_equal(ids[i],ids[j]))
    ids[list(pair)]=ids[list(reversed(pair))];np.save(path,ids)
    manifest=json.loads((root/"MANIFEST.json").read_text());manifest[batch["input_ids_path"]]=sha256_file(path);write("MANIFEST.json",manifest)
    rejected=subprocess.run(command,cwd=repo,capture_output=True,text=True,env={**os.environ,"PYTHONPATH":os.pathsep.join([str(repo/"src"),str(repo),str(repo/"scripts")])})
    assert rejected.returncode!=0 and "input token/mask array differs" in rejected.stderr
