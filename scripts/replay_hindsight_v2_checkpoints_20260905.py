"""Post hoc checkpoint integrity and teacher-capability diagnosis; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from interaction_sprint.hindsight_execution_integrity import capture_model_provenance, sha256_file
from interaction_sprint.hindsight_neural_anchor import install_qwen35_lora
from interaction_sprint.hindsight_pahf_reduced import qualify_interface
from latent_contract.sender_update import load_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve diagnostic result')
    start = time.time()
    receipt = json.loads(args.verification.read_text())
    assert receipt['neural_acquisition_evidence_verified']
    assert receipt['manifest_sha256'] == sha256_file(args.root/'MANIFEST.json')
    cfg = json.loads((args.root/'spec.json').read_text())
    provenance = json.loads((args.root/'model_provenance.json').read_text())
    snapshot = capture_model_provenance(args.snapshot, cfg['model_id'], cfg['model_revision'])
    assert snapshot['files'] == provenance['files']
    assert torch.cuda.device_count() == 1
    torch.set_float32_matmul_precision('high')
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, use_fast=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForCausalLM.from_pretrained(args.snapshot, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation=cfg['attention'], device_map={'':0},
        low_cpu_mem_usage=True, use_kernels=False).eval()
    model.config.use_cache=False
    install_qwen35_lora(model,rank=cfg['lora_rank'],alpha=cfg['lora_alpha'])
    index = json.loads((args.root/'logits_index.json').read_text())
    jobs = json.loads((args.root/'interface_jobs.json').read_text())
    tokens = json.loads((args.root/'runtime_setup.json').read_text())['answer_token_ids']
    replay, diagnostics = {}, {}
    for arm in ('baseline','raw_immediate','oracle_delayed'):
        checkpoint = args.root/('initial_adapter.pt' if arm=='baseline' else arm+'_adapter.pt')
        load_adapter(model, torch.load(checkpoint,map_location='cpu',weights_only=True))
        # Outcome-blind batch selection, fixed before this diagnostic executes.
        batch = min((b for b in index if b['arm']==arm), key=lambda b: hashlib.sha256(('checkpoint-replay-20260905|'+b['path']).encode()).hexdigest())
        ids = np.load(args.root/batch['input_ids_path'],allow_pickle=False)
        mask = np.load(args.root/batch['attention_mask_path'],allow_pickle=False)
        with torch.no_grad():
            actual = model(input_ids=torch.as_tensor(ids,device='cuda'),attention_mask=torch.as_tensor(mask,device='cuda'),
                           use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
        saved = torch.from_numpy(np.load(args.root/batch['path'],allow_pickle=False)).to('cuda')
        difference = (actual-saved).abs()
        pdelta = (actual.log_softmax(-1)[:,tokens].exp()-saved.log_softmax(-1)[:,tokens].exp()).abs()
        replay[arm]={'batch':batch['path'],'rows':len(ids),'checkpoint_sha256':sha256_file(checkpoint),
                     'max_full_logit_absolute_difference':float(difference.max()),
                     'max_native_probability_absolute_difference':float(pdelta.max()),
                     'native_argmax_identical':bool(torch.equal(actual[:,tokens].argmax(-1),saved[:,tokens].argmax(-1)))}
        replay[arm]['numerical_match'] = (replay[arm]['max_native_probability_absolute_difference']<=1e-5 and
                                         replay[arm]['max_full_logit_absolute_difference']<=1e-4 and
                                         replay[arm]['native_argmax_identical'])
        if arm=='baseline':
            continue
        predictions=[]
        for start_row in range(0,len(jobs),16):
            selected=jobs[start_row:start_row+16]
            rendered=[tokenizer.apply_chat_template([{'role':'user','content':cfg['common_task_prefix']+j['text']}],
                        tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in selected]
            encoded=tokenizer(rendered,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
            assert int(encoded['attention_mask'].sum(-1).max())<=cfg['max_tokens']
            with torch.no_grad():
                output=model(**encoded,use_cache=False,logits_to_keep=1).logits[:,-1,:].float()
            full=output.log_softmax(-1)[:,tokens]; conditional=full.log_softmax(-1)
            for i,row in enumerate(selected):
                predictions.append({**{k:row[k] for k in ('id','base_id','label_rotation','source_user','job_id','context','target')},
                    'full_vocab_choice_log_probabilities':full[i].tolist(),
                    'normalized_choice_log_probabilities':conditional[i].tolist(),
                    'full_vocabulary_choice_mass':float(full[i].exp().sum())})
        diagnostics[arm]={'posthoc_teacher_capability':qualify_interface(jobs,predictions,cfg),
                          'learning_only_predictions':predictions}
    report={'scope':'Post hoc saved-checkpoint replay and teacher capability on the same sixteen learning qualification bases',
            'replay':replay,'teacher_diagnostics':diagnostics,'initial_teacher':json.loads((args.root/'interface_RESULT.json').read_text()),
            'elapsed_seconds':time.time()-start,'script_sha256':sha256_file(Path(__file__)),
            'original_manifest_sha256':receipt['manifest_sha256'],'new_optimizer_steps':0,
            'new_interface_repair':False,'confirmation_opened':False,'paper_green_light':False}
    with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'replay':replay,'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
