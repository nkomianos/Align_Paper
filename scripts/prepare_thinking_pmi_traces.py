"""Prospective base-only probe selection. No teacher or control calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def select_positions(entropies,eligible,key):
    if not eligible:return None
    high=max(eligible,key=lambda t:(entropies[t],-t))
    index=int(hashlib.sha256(('thinking-pmi-random-v1:'+key).encode()).hexdigest(),16)%len(eligible)
    return {'high_entropy':high,'random':eligible[index]}


def main(data,snapshot,out):
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    plan=json.loads((data/'PLAN.json').read_text())
    assert {p.name:sha(p) for p in snapshot.glob('*.safetensors')}==plan['model_manifest']['weights']
    unique={r['base']:r for r in json.loads((data/'INPUTS.json').read_text(encoding='utf-8'))}
    assert len(unique)==24
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='New thinking traces on24exposedDEVproblems; no training or answer-accuracy claim',
        source_plan_sha256=sha(data/'PLAN.json'),source_inputs_sha256=sha(data/'INPUTS.json'),
        runner_sha256=sha(Path(__file__)),sampling='One256token trace per problem, temperature1,top_p1,top_k0, seed2026091070+sortedIDindex',
        positions='Prefix lengths32..255 strictly before first closing think tag; maximum base entropy and fixedhash random control',
        entropy='Recomputed from saved float16 logits converted to float32; not teacher-conditioned',
        qualification='At least16of24problems with eligible high-entropy position>=1nat; otherwise no comparison launch',
        estimate='5-12GPUminutes unbenchmarked; scientific256token horizon, no wallclock cutoff',
        future='Freeze selection before any teacher logit inspection; no automatic training')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    config=GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=256,
        eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_logits=True,use_cache=True)
    close_ids=tok.encode('</think>',add_special_tokens=False); assert len(close_ids)==1
    selected=[];started=time.monotonic()
    for i,key in enumerate(sorted(unique)):
        row=unique[key]
        rendered=tok.apply_chat_template([{'role':'user','content':row['question']+'\nSolve step by step.'}],
            tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert rendered.rstrip().endswith('<think>')
        enc=tok(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
        seed=2026091070+i;torch.manual_seed(seed)
        with torch.inference_mode():
            result=model.generate(**enc,generation_config=config)
        ids=result.sequences[0,enc.input_ids.shape[1]:].tolist()
        raw=torch.stack([x[0].cpu().half() for x in result.logits])
        logp=raw.float().log_softmax(-1);entropy=-(logp.exp()*logp).sum(-1)
        assert torch.isfinite(raw).all() and torch.isfinite(entropy).all() and len(ids)==len(raw)
        boundary=ids.index(close_ids[0]) if close_ids[0] in ids else len(ids)
        eligible=list(range(32,min(boundary,len(ids))))
        positions=select_positions(entropy.tolist(),eligible,key)
        np.savez_compressed(out/f'logits_{i:02d}.npz',logits=raw.numpy())
        trace=dict(base=key,seed=seed,rendered=rendered,input_ids=enc.input_ids[0].tolist(),
            ids=ids,text=tok.decode(ids,skip_special_tokens=False),entropies=entropy.tolist(),
            positions=positions,closing_think_index=boundary,
            qualifies=bool(positions and entropy[positions['high_entropy']]>=1.0))
        (out/f'trace_{i:02d}.json').write_text(json.dumps(trace),encoding='utf-8')
        selected.append(trace)
        del result,raw,logp,entropy
        print(json.dumps({'completed':i+1,'qualified':sum(t['qualifies'] for t in selected),'seconds':time.monotonic()-started}),flush=True)
    summary=dict(problems=24,qualified=sum(t['qualifies'] for t in selected),seconds=time.monotonic()-started)
    summary['comparison_admitted']=summary['qualified']>=16
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (out/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.data,a.snapshot,a.out)
