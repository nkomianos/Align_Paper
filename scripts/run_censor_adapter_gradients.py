"""Replay conditional continuation score gradients for a fixed adapter policy."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from latent_contract.sender_update import LoRALinear
from run_unexplored_screens import dump,sha


def sampled_token_logprobs(logits,targets,temperature):
    if logits.ndim!=2 or logits.shape[0]!=len(targets) or temperature<=0:raise ValueError('invalid causal alignment')
    return (logits.float()/temperature).log_softmax(-1).gather(1,targets[:,None]).squeeze(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for name,h in json.loads((a.bank/'MANIFEST.json').read_text()).items():assert sha(a.bank/name)==h
    source_model=json.loads((a.bank/'MODEL.json').read_text())
    for name,h in source_model['files'].items():assert sha(a.snapshot/name)==h
    config=json.loads((a.bank/'GENERATION_CONFIG.json').read_text())
    assert config['do_sample'] and config['temperature']==.8 and config['top_p']==1 and config['top_k']==0
    rows=[json.loads(x) for x in (a.bank/'ROLLOUTS.jsonl').read_text().splitlines()]
    assert len(rows)==384
    dump(a.out/'PROTOCOL.json',{'scope':'finite-bank adapter-policy gradient apparatus; no optimizer learning',
        'bank_manifest_sha256':sha(a.bank/'MANIFEST.json'),'runner_sha256':sha(Path(__file__)),
        'parameters':'last-layer q/v LoRA rank4 alpha8; zero B; seed2026091071; frozen base',
        'reward':'original frozen strict-parser finite768-token reward; not eventual mathematical correctness',
        'likelihood_gate':'max absolute replay-vs-stored logprob difference per generated token <=.02',
        'comparison':'adaptive vs uniform continuation vs fewer full at matched expected token cost, both baseline .5 and calibration-fitted mean reward',
        'routing':'>=20% adaptive design-MSE reduction over BOTH alternatives under BOTH reward baselines; otherwise stop',
        'cost_limit':'token accounting is not wall-clock efficiency; any positive requires fresh bank, actual time and learning tests',
        'estimate':'5-20 GH200 minutes unbenchmarked; no new generation'})
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    from transformers import AutoModelForCausalLM
    torch.manual_seed(2026091071)
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    attention=model.model.layers[-1].self_attn
    for name in ('q_proj','v_proj'):setattr(attention,name,LoRALinear(getattr(attention,name),rank=4,alpha=8))
    params=[p for p in model.parameters() if p.requires_grad]
    dump(a.out/'PARAMETERS.json',[{'name':n,'shape':list(p.shape)} for n,p in model.named_parameters() if p.requires_grad])
    started=time.monotonic();metadata=[];full_gradients=[];early_gradients=[]
    for i,row in enumerate(rows):
        ids=torch.tensor([row['input_ids']+row['ids']],device='cuda');length=len(row['ids'])
        targets=torch.tensor(row['ids'],device='cuda')
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=length+1).logits[0,:-1]
        logps=sampled_token_logprobs(logits,targets,.8)
        torch.cuda.synchronize();backward_start=time.monotonic()
        early=torch.autograd.grad(logps[:128].sum(),params,retain_graph=True)
        early_gradients.append(torch.cat([g.reshape(-1) for g in early]).detach().cpu())
        full=torch.autograd.grad(logps.sum(),params)
        full_gradients.append(torch.cat([g.reshape(-1) for g in full]).detach().cpu())
        torch.cuda.synchronize()
        metadata.append({'base':row['base'],'prefix_index':row['prefix_index'],'sample':row['sample'],
            'replayed_logprob':float(logps.detach().double().sum()),'stored_logprob':row['sampling_logprob'],
            'early_logprob':float(logps[:128].detach().double().sum()),'length':length,
            'backward_seconds':time.monotonic()-backward_start})
        del early,full,logits,logps
        if (i+1)%16==0:
            dump(a.out/'PROGRESS.json',{'completed':i+1,'planned':len(rows),'seconds':time.monotonic()-started})
            print((a.out/'PROGRESS.json').read_text(),flush=True)
    torch.save({'full':torch.stack(full_gradients),'early':torch.stack(early_gradients)},a.out/'GRADIENTS.pt')
    dump(a.out/'ROWS.json',metadata)
    error=max(abs(r['replayed_logprob']-r['stored_logprob'])/r['length'] for r in metadata)
    dump(a.out/'SUMMARY.json',{'classification':'DEVELOPMENTAL_GRADIENT_REPLAY','rows':len(rows),
        'max_per_token_logprob_discrepancy':error,'likelihood_qualified':error<=.02,
        'seconds':time.monotonic()-started,'trainable_parameters':sum(p.numel() for p in params),
        'finite_precision':'BF16 base and FP32 adapters; replay uses full attention, source sampling used KV cache'})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    print('CENSOR_ADAPTER_GRADIENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
