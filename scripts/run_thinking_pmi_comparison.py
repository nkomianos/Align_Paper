"""Frozen thinking-prefix attribution diagnostic, never a training experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(ls):
    from pmi_target_controls import corrected_target,distribution_kl
    base=ls['base'].log_softmax(-1)
    p=corrected_target(ls['base'],ls['teacher'],ls['reference'])
    q=corrected_target(ls['base'],ls['base'],ls['unconditional'])
    wrong=corrected_target(ls['base'],ls['wrong_teacher'],ls['wrong_reference'])
    tv=lambda a,b:float((a.exp()-b.exp()).abs().sum()/2)
    return dict(tv_purified_control=tv(p,q),tv_purified_wrong=tv(p,wrong),
        tv_wrong_control=tv(wrong,q),tv_base_cached=tv(base,ls['cached_base'].log_softmax(-1)),
        kl_purified_base=float(distribution_kl(p,base)),
        entropy_base=float(-(base.exp()*base).sum()),
        entropy_purified=float(-(p.exp()*p).sum()),entropy_control=float(-(q.exp()*q).sum()))


def main(data,traces,selection,snapshot,out,execution='full'):
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    receipt=json.loads(selection.read_text());assert receipt['verified'] and receipt['comparison_admitted']
    assert sha(traces/'MANIFEST.json')==receipt['manifest_sha256']
    plan=json.loads((data/'PLAN.json').read_text())
    assert {p.name:sha(p) for p in snapshot.glob('*.safetensors')}==plan['model_manifest']['weights']
    unique={r['base']:r for r in json.loads((data/'INPUTS.json').read_text())};keys=sorted(unique)
    assert len(keys)==24
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(selection_sha256=sha(selection),source_inputs_sha256=sha(data/'INPUTS.json'),
        runner_sha256=sha(Path(__file__)),helper_sha256=sha(Path(__file__).with_name('pmi_target_controls.py')),
        wrong_reference='Next question in sorted24questionIDs, cyclic; no outcome-dependent matching; length and subject may differ',
        contexts='Native thinking chat; exact generating base prompt retained; six contexts with identical generated prefix IDs',
        execution=execution,
        amendment='Cached mode replays native prompt prefill then one generated prefix token per forward for every arm; original full-mode assay retained separately',
        analysis='Report high-entropy and fixedhash random separately; pair by question; descriptive DEV only, no training/accuracy claim',
        gate='Any cached/full-forward base TV>.05 invalidates numerical comparability for whole assay pending investigation; retain all rows',
        estimate='3-10 GPU minutes including load in cached mode; no wallclock kill',
        target='beta1,c10 centered tanh; full reference, irrelevant reference, question-only controls; no fitted temperature')
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    saved=[];results=[];loaded_key=None;cached=None
    with torch.inference_mode(),(out/'ROWS.jsonl').open('x',encoding='utf-8') as f:
        for r in receipt['rows']:
            idx=keys.index(r['base']);source=unique[r['base']];other=keys[(idx+1)%24]
            wrong=unique[other]['reference'];question=source['question'];reference=source['reference']
            contexts=dict(base=question,unconditional='',teacher=question+'\nReference solution:\n'+reference,
                reference='Reference solution:\n'+reference,wrong_teacher=question+'\nReference solution:\n'+wrong,
                wrong_reference='Reference solution:\n'+wrong)
            logits={};inputs={}
            for arm,content in contexts.items():
                rendered=tok.apply_chat_template([{'role':'user','content':content+'\nSolve step by step.'}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=True)
                ids=tok.encode(rendered,add_special_tokens=False)
                if arm=='base':assert rendered==r['rendered'] and ids==r['input_ids']
                prompt_len=len(ids)
                ids+=r['prefix_ids'];assert len(ids)<=8192
                x=torch.tensor([ids],device='cuda')
                if execution=='full':
                    values=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits[0,-1]
                else:
                    initial=x[:,:prompt_len]
                    response=model(input_ids=initial,attention_mask=torch.ones_like(initial),use_cache=True,logits_to_keep=1)
                    cache=response.past_key_values
                    for t in range(prompt_len,len(ids)):
                        response=model(input_ids=x[:,t:t+1],attention_mask=torch.ones_like(x[:,:t+1]),
                            past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=response.past_key_values
                    values=response.logits[0,-1]
                logits[arm]=values.float().cpu()
                if execution=='cached':del response,cache
                del values
                inputs[arm]=dict(rendered=rendered,ids=ids)
            if loaded_key!=r['base']:
                with np.load(traces/f'logits_{idx:02d}.npz',allow_pickle=False) as z:cached=z['logits']
                loaded_key=r['base']
            logits['cached_base']=torch.from_numpy(cached[r['length']].copy()).float()
            assert all(torch.isfinite(v).all() for v in logits.values())
            result=dict(base=r['base'],kind=r['kind'],length=r['length'],wrong_reference_base=other,
                selection_entropy=r['selection_entropy'],inputs=inputs,**metrics(logits))
            saved.append(logits);results.append(result);f.write(json.dumps(result)+'\n');f.flush()
            print(json.dumps({'completed':len(saved),'seconds':time.monotonic()-started}),flush=True)
    torch.save(saved,out/'LOGITS.pt')
    summary=dict(rows=len(results),problems=len({r['base'] for r in results}),seconds=time.monotonic()-started,
        numerical_comparability=all(r['tv_base_cached']<=.05 for r in results))
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    (out/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','traces','selection','snapshot','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execution',choices=['full','cached'],default='full')
    a=p.parse_args();main(a.data,a.traces,a.selection,a.snapshot,a.out,a.execution)
