"""Frozen on-policy cache diagnostic, including closed-loop task outcomes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
import transformers
from . import opdlm_cache_dev as base

POLICIES=['baseline','fresh','all_corrected','late_corrected']

def task(op,a,b):
    symbol={'add':'+','subtract':'-','multiply':'*'}[op]
    gold={'add':a+b,'subtract':a-b,'multiply':a*b}[op]
    return f'Calculate {a} {symbol} {b}. Reply with only the integer answer, using digits.',gold

def score(text,gold):
    s=text.strip()
    strict=bool(re.fullmatch(r'[+-]?\d+',s))
    # More permissive predeclared audit: final numeral, never replaces strict.
    numbers=re.findall(r'(?<![\w.])[+-]?\d+(?!\w|\.\d)',s)
    return {'strict_parse':strict,'strict_correct':strict and int(s)==gold,
            'last_number_correct':bool(numbers) and int(numbers[-1])==gold}

def choose(z,current,mask_id):
    valid=current==mask_id
    assert valid.any()
    probability,guess=z.float().softmax(-1).max(-1)
    confidence=probability.masked_fill(~valid,-float('inf'))
    pos=int(confidence.argmax())
    return pos,int(guess[pos])

def summarize(records,audits):
    result={'n_tasks':len(records)//4,'policies':{},'paired_audits':len(audits)}
    for policy in POLICIES:
        r=[x for x in records if x['policy']==policy]
        result['policies'][policy]={k:sum(x['score'][k] for x in r)
            for k in ['strict_parse','strict_correct','last_number_correct']}
        result['policies'][policy].update(terminated=sum(x['terminated'] for x in r),
            seed_revisions=sum(sum(e['revised'] for e in x['events']) for x in r),
            forward_calls=sum(x['calls'] for x in r))
    if audits:
        result['paired']={
            'all_vs_fresh_seed_argmax_differences':sum(x['seed_argmax'][1]!=x['seed_argmax'][0] for x in audits),
            'fresh_rejects_old':sum(x['seed_argmax'][0]!=x['old'] for x in audits),
            'all_retains_when_fresh_rejects':sum(x['seed_argmax'][0]!=x['old'] and x['seed_argmax'][1]==x['old'] for x in audits),
            'late_seed_max_error':max(x['late_seed_error'] for x in audits),
            'all_mean_candidate_probability_shift':float(np.mean([x['old_probability'][1]-x['old_probability'][0] for x in audits])),
            'late_changes_remaining_draft_argmax':sum(x['late_draft_changes'] for x in audits),
        }
    result['interpretation']='Developmental arithmetic comparison, not an official COVER reproduction or paper gate'
    return result

def run(config_path,root):
    config=json.loads(config_path.read_text())
    assert config['policies']==POLICIES and config['block_size']==4
    root.mkdir(parents=True,exist_ok=False)
    # Freeze source/config before first model forward.
    sources={str(Path(__file__).resolve()):base.sha(__file__),str(Path(base.__file__).resolve()):base.sha(base.__file__),str(base.SOURCE):base.sha(base.SOURCE)}
    frozen={'config':config,'sources':sources,'model_files':{p.name:base.sha(p) for p in base.MODEL.iterdir() if p.is_file()}}
    assert frozen['model_files']['model.safetensors']=='b21185037fde14214e7473ac29632770592bcfd6f7f532d73c35ec2a9f5098a5'
    (root/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    env=base.upstream(); cfg=env['A2DQwen3Config'].from_pretrained(base.MODEL,local_files_only=True)
    model,loading=transformers.AutoModelForMaskedLM.from_pretrained(base.MODEL,config=cfg,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model.eval(); tok=transformers.AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
    records=[]; audits=[]; start=time.monotonic(); count=0
    with torch.inference_mode():
        for index,triple in enumerate(config['tasks']):
            prompt,gold=task(*triple)
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            prefix=[tok.pad_token_id]*((-len(ids))%4)+ids
            for policy in POLICIES:
                x=torch.tensor([prefix]); events=[]; calls=0
                with base.Instrument(model) as inst:
                    def forward(inp,mode='plain'):
                        nonlocal calls,count
                        calls+=1; count+=1; inst.mode=mode
                        mask,pos=env['_prepare_for_sampling'](inp,4,tok.pad_token_id)
                        return model(inp,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=4).logits[0]
                    for block in range(config['max_new_tokens']//4):
                        x=torch.cat((x,torch.full((1,4),tok.mask_token_id,dtype=torch.long)),dim=1)
                        z=forward(x); seed,old=choose(z,x[0,-4:],tok.mask_token_id)
                        assert old!=tok.mask_token_id
                        x[0,-4+seed]=old; inst.seed=x.shape[1]-4+seed
                        inst.saved={}; z=forward(x,'collect'); inst.cache=inst.saved
                        second,token=choose(z,x[0,-4:],tok.mask_token_id); assert token!=tok.mask_token_id
                        x[0,-4+second]=token
                        # Same post-second-fill state for every paired probe.
                        hidden=x.clone(); hidden[0,inst.seed]=tok.mask_token_id
                        if policy=='baseline':
                            z=forward(x)
                            variants=[forward(hidden,m) for m in ['plain','all_corrected','late_corrected']]
                            remains=x[0,-4:]==tok.mask_token_id
                            audit={'task':index,'block':block,'old':old,'seed_offset':seed,
                                'state':x[0].tolist(),'seed_argmax':[int(v[seed].argmax()) for v in variants],
                                'old_probability':[float(v[seed].softmax(-1)[old]) for v in variants],
                                'late_seed_error':float((variants[0][seed]-variants[2][seed]).abs().max()),
                                'late_draft_changes':int(((variants[0].argmax(-1)!=variants[2].argmax(-1))&remains).sum())}
                            audits.append(audit)
                            with (root/'audits.jsonl').open('a') as out: out.write(json.dumps(audit)+'\n')
                            # Retain full paired seed distributions, not just favorable scalar metrics.
                            np.savez_compressed(root/f'audit_{index:02d}_{block:02d}.npz',seed_logits=torch.stack([v[seed] for v in variants]).cpu().numpy())
                            revised=False
                        else:
                            z=forward(hidden,'plain' if policy=='fresh' else policy)
                            replacement=int(z[seed].argmax()); assert replacement!=tok.mask_token_id
                            revised=replacement!=old; x[0,inst.seed]=replacement
                        third,token=choose(z,x[0,-4:],tok.mask_token_id); assert token!=tok.mask_token_id
                        x[0,-4+third]=token
                        z=forward(x); fourth,token=choose(z,x[0,-4:],tok.mask_token_id); assert token!=tok.mask_token_id
                        x[0,-4+fourth]=token
                        events.append({'block':block,'seed':seed,'old':old,'revised':revised,'tokens':x[0,-4:].tolist()})
                        if tok.eos_token_id in x[0,len(prefix):]: break
                tokens=x[0,len(prefix):].tolist()
                end=tokens.index(tok.eos_token_id) if tok.eos_token_id in tokens else len(tokens)
                text=tok.decode(tokens[:end],skip_special_tokens=True)
                r={'task':index,'policy':policy,'prompt':prompt,'gold':gold,'prompt_ids':prefix,
                   'tokens':tokens,'text':text,'terminated':end<len(tokens),'events':events,'calls':calls,'score':score(text,gold)}
                records.append(r)
                with (root/'outputs.jsonl').open('a') as out: out.write(json.dumps(r)+'\n')
    result=summarize(records,audits)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    rt={'calls':count,'seconds':time.monotonic()-start,'torch':torch.__version__,'transformers':transformers.__version__,'loading':loading}
    (root/'runtime.json').write_text(json.dumps(rt,indent=2,default=base.json_default)+'\n')
    assert all(base.sha(p)==h for p,h in sources.items())
    (root/'MANIFEST.json').write_text(json.dumps({p.name:base.sha(p) for p in root.iterdir()},indent=2)+'\n')
    return result

def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text()); assert all(base.sha(root/n)==h for n,h in m.items())
    frozen=json.loads((root/'FROZEN.json').read_text()); assert all(base.sha(p)==h for p,h in frozen['sources'].items())
    assert all(base.sha(base.MODEL/n)==h for n,h in frozen['model_files'].items())
    records=[json.loads(l) for l in (root/'outputs.jsonl').read_text().splitlines()]
    audits=[json.loads(l) for l in (root/'audits.jsonl').read_text().splitlines()]
    assert len(records)==len(frozen['config']['tasks'])*4
    assert {(r['task'],r['policy']) for r in records}=={(i,p) for i in range(len(frozen['config']['tasks'])) for p in POLICIES}
    tok=transformers.AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
    for r in records:
        prompt,gold=task(*frozen['config']['tasks'][r['task']]); assert r['prompt']==prompt and r['gold']==gold
        assert r['tokens']==[t for e in r['events'] for t in e['tokens']]
        end=r['tokens'].index(tok.eos_token_id) if tok.eos_token_id in r['tokens'] else len(r['tokens'])
        assert r['text']==tok.decode(r['tokens'][:end],skip_special_tokens=True)
        assert r['terminated']==(end<len(r['tokens'])) and r['score']==score(r['text'],gold)
        assert r['calls']==len(r['events'])*(7 if r['policy']=='baseline' else 4)
    assert len(audits)==sum(len(r['events']) for r in records if r['policy']=='baseline')
    for a in audits:
        z=np.load(root/f"audit_{a['task']:02d}_{a['block']:02d}.npz")['seed_logits']
        assert z.shape==(3,151936) and np.isfinite(z).all()
        assert z.argmax(-1).tolist()==a['seed_argmax']
        p=torch.from_numpy(z).softmax(-1)[:,a['old']].tolist()
        assert np.allclose(p,a['old_probability'],atol=1e-7,rtol=1e-6)
        assert float(np.max(np.abs(z[0]-z[2])))==a['late_seed_error']
    result=summarize(records,audits); assert result==json.loads((root/'summary.json').read_text())
    rt=json.loads((root/'runtime.json').read_text()); assert rt['calls']==sum(r['calls'] for r in records)
    return {'verified':'hashes_and_recomputed_metrics_not_forward_replay','manifest_sha':base.sha(root/'MANIFEST.json'),'result':result}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','verify']); p.add_argument('root',type=Path)
    p.add_argument('--config',type=Path,default=base.ROOT/'configs/opdlm_onpolicy_v1.json')
    args=p.parse_args(); print(json.dumps(run(args.config,args.root) if args.action=='run' else verify(args.root),indent=2))
