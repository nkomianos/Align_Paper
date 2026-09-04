"""Read-only evidence audit and paired stochastic SQuAD DEV analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from interaction_sprint.byte_clock_coupling import digest
from interaction_sprint.byte_coupling_native import vocabulary,byte_event_clock
from interaction_sprint.squad_coupling import score,prompt


def moments(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    assert a.shape==b.shape and a.ndim==1 and len(a)>1
    va=float(a.var(ddof=1));vb=float(b.var(ddof=1));cov=float(np.cov(a,b,ddof=1)[0,1])
    vd=float((a-b).var(ddof=1));assert np.isclose(vd,va+vb-2*cov)
    return {'mean_a':float(a.mean()),'mean_b':float(b.mean()),'mean_difference':float((a-b).mean()),
            'variance_a':va,'variance_b':vb,'covariance':cov,'variance_difference':vd}


def ratio_ci(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
    rng=np.random.default_rng(940621);ix=rng.integers(0,len(a),size=(5000,len(a)))
    numer=a[ix].mean(1);denom=b[ix].mean(1);invalid=int(np.sum(denom<=0))
    return {'ratio':float(a.mean()/b.mean()) if b.mean()>0 else None,
            'question_bootstrap_descriptive_95':np.quantile(numer/denom,[.025,.975]).tolist() if invalid==0 else None,
            'undefined_bootstrap_draws':invalid}


def verify(root):
    from transformers import AutoTokenizer
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/n)==h for n,h in manifest.items())
    freeze=json.loads((root/'FROZEN.json').read_text());cfg=freeze['config']
    assert all(digest(p)==h for p,h in freeze['sources'].items())
    prepared=Path(cfg['prepared']);assert digest(prepared/'MANIFEST.json')==freeze['prepared_manifest_sha']
    pm=json.loads((prepared/'MANIFEST.json').read_text());assert all(digest(prepared/n)==h for n,h in pm.items())
    assert json.loads((prepared/'cases.json').read_text())==freeze['cases']
    key=json.loads((prepared/'answer_key.json').read_text())
    records=[json.loads(l) for l in (root/'outputs.jsonl').read_text().splitlines()]
    timings=[json.loads(l) for l in (root/'timings.jsonl').read_text().splitlines()]
    lookup={(r['model'],r['case_id'],r['policy'],r['seed']):r for r in records}
    expected={(m['name'],c['id'],p,s) for m in cfg['models'] for c in freeze['cases'] for p in cfg['policies'] for s in cfg['seeds']}
    assert len(lookup)==len(records) and set(lookup)==expected
    expected_batches={(m['name'],c['id'],p,tuple(cfg['seeds'][b:b+cfg['batch_size']]))
                      for m in cfg['models'] for c in freeze['cases'] for p in cfg['policies']
                      for b in range(0,len(cfg['seeds']),cfg['batch_size'])}
    actual_batches={(t['model'],t['case_id'],t['policy'],tuple(t['seeds'])) for t in timings}
    assert actual_batches==expected_batches and len(actual_batches)==len(timings)
    for spec in cfg['models']:
        path=Path(spec['path']);assert all(digest(path/n)==h for n,h in freeze['model_files'][spec['name']].items())
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        size=json.loads((path/'config.json').read_text())['vocab_size'];raw,_,_=vocabulary(tok,size,spec['name'])
        q=json.loads((root/(spec['name']+'_cache_check.json')).read_text())
        assert q['calls']==4 and q['next_argmax_agreement']
        assert max(q['first_batch_serial_error'],q['next_cache_full_error'])<cfg['cache_tolerance']
        for r in [r for r in records if r['model']==spec['name']]:
            ts=r['tokens'];assert 0<len(ts)<=cfg['max_new_tokens'] and all(0<=t<size for t in ts)
            assert tok.eos_token_id not in ts[:-1] and r['terminated']==(ts[-1]==tok.eos_token_id)
            assert len(r['events'])==len(ts)
            offset=silent=0
            for i,(t,e) in enumerate(zip(ts,r['events'])):
                clock=i if r['policy'] in ('independent','token_clock') else byte_event_clock(offset,silent)
                assert e['clock']==clock and e['token']==t and np.isfinite(e['log_probability']) and e['log_probability']<=1e-8
                if raw[t]:offset+=len(raw[t]);silent=0
                else:silent+=1
            b=b''.join(raw[t] for t in ts);assert b.hex()==r['raw_bytes_hex']
            assert b.decode('utf-8',errors='replace')==tok.decode(ts,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert r['text']==tok.decode(ts,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['silent_token_count']==sum(not raw[t] for t in ts)
        for t in [t for t in timings if t['model']==spec['name']]:
            c=next(c for c in freeze['cases'] if c['id']==t['case_id'])
            ids=tok.apply_chat_template([{'role':'user','content':prompt(c)}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            assert ids==t['prompt_ids'] and len(ids)<=cfg['max_prompt_tokens']
            rows=[lookup[spec['name'],c['id'],t['policy'],s] for s in t['seeds']]
            assert t['forward_calls']==max(len(r['tokens']) for r in rows)
            assert t['forwarded_rows_including_finished_padding']==len(rows)*t['forward_calls']
            assert t['generated_tokens']==sum(len(r['tokens']) for r in rows)
            assert all(np.isfinite(t[k]) and t[k]>=0 for k in ('seconds','forward_seconds','sampler_seconds'))
            assert t['seconds']+1e-5>=t['forward_seconds']+t['sampler_seconds']
    runtime=json.loads((root/'runtime.json').read_text());assert runtime['outputs']==len(records)
    return cfg,freeze,records,timings,key,runtime


def analyze(root,out):
    cfg,freeze,records,timings,key,runtime=verify(root)
    scored={};lookup={}
    for r in records:
        k=(r['model'],r['case_id'],r['policy'],r['seed']);lookup[k]=r;scored[k]=score(r['text'],key[r['case_id']])
    models=[m['name'] for m in cfg['models']];assert len(models)==2
    per_policy={};per_case=[]
    for policy in cfg['policies']:
        cases=[]
        for c in freeze['cases']:
            metrics={metric:moments([scored[models[0],c['id'],policy,s][metric] for s in cfg['seeds']],
                                    [scored[models[1],c['id'],policy,s][metric] for s in cfg['seeds']]) for metric in ('em','f1')}
            t=[t for t in timings if t['policy']==policy and t['case_id']==c['id']]
            paired_cost=sum(ti['seconds'] for ti in t)/len(cfg['seeds'])
            item={'id':c['id'],'title':c['title'],'policy':policy,'metrics':metrics,'mean_pair_seconds':paired_cost}
            cases.append(item);per_case.append(item)
        per_policy[policy]={'by_model':{},'mean_pair_seconds':float(np.mean([c['mean_pair_seconds'] for c in cases]))}
        for model in models:
            selected=[r for r in records if r['model']==model and r['policy']==policy]
            per_policy[policy]['by_model'][model]={'n':len(selected),'terminated':sum(r['terminated'] for r in selected),
                'mean_tokens':float(np.mean([len(r['tokens']) for r in selected])),
                **{metric:float(np.mean([scored[r['model'],r['case_id'],policy,r['seed']][metric] for r in selected])) for metric in ('em','f1')}}
        for metric in ('em','f1'):
            per_policy[policy][metric]={k:float(np.mean([c['metrics'][metric][k] for c in cases])) for k in cases[0]['metrics'][metric]}
    comparisons={}
    for metric in ('em','f1'):
        comparisons[metric]={}
        for baseline in ['independent','token_clock','byte_clock']:
            a=[c['metrics'][metric]['variance_difference'] for c in per_case if c['policy']=='byte_hierarchical']
            b=[c['metrics'][metric]['variance_difference'] for c in per_case if c['policy']==baseline]
            result=ratio_ci(a,b)
            cost=per_policy['byte_hierarchical']['mean_pair_seconds']/per_policy[baseline]['mean_pair_seconds']
            result['cost_ratio']=cost
            result['variance_times_cost_ratio']=result['ratio']*cost if result['ratio'] is not None else None
            comparisons[metric]['hierarchical vs '+baseline]=result
    report={'verified':'hashes_tokens_clocks_decoding_timing_and_scores_not_neural_or_full_sampler_replay',
            'manifest_sha':digest(root/'MANIFEST.json'),'analysis_source_sha':digest(__file__),
            'runtime':runtime,'policies':per_policy,'comparisons':comparisons,'per_case':per_case,
            'scope':'Eight public DEV questions, 16 seeds per question; descriptive pilot only; no automatic expansion'}
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='per_case'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();analyze(a.root,a.output)
