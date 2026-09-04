"""Separate post-run paired analysis; never edit the frozen experiment outputs."""
import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
import transformers
from interaction_sprint.opdlm_onpolicy import verify,POLICIES
from interaction_sprint.opdlm_cache_dev import MODEL,sha

def last_numeric_span(text):
    found=list(re.finditer(r'(?<![\w.])[+-]?\d+(?!\w|\.\d)',text))
    return found[-1].span() if found else None

def threshold_audit(audits,cutoff=0.8):
    # Exploratory: 0.8 is the published WINO re-evaluation's remask threshold,
    # not a tuned threshold or a policy actually deployed in this experiment.
    return {'cutoff':cutoff,'n':len(audits),
        'fresh_rejects_cache_accepts':sum(a['old_probability'][0]<cutoff<=a['old_probability'][1] for a in audits),
        'cache_rejects_fresh_accepts':sum(a['old_probability'][1]<cutoff<=a['old_probability'][0] for a in audits)}

def paired(a,b):
    a=np.asarray(a,dtype=int); b=np.asarray(b,dtype=int)
    wins=int(((a==1)&(b==0)).sum()); losses=int(((a==0)&(b==1)).sum())
    discordant=wins+losses
    p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    rng=np.random.default_rng(94051)
    delta=a-b
    interval=np.quantile(delta[rng.integers(0,len(a),size=(5000,len(a)))].mean(1),[.025,.975])
    return {'n':len(a),'a_correct':int(a.sum()),'b_correct':int(b.sum()),'a_only':wins,'b_only':losses,
        'difference_pp':100*float(delta.mean()),'task_bootstrap_descriptive_95_pp':(100*interval).tolist(),
        'unadjusted_exact_discordance_p':p}

def analyze(root,out):
    checked=verify(root)
    records=[json.loads(l) for l in (root/'outputs.jsonl').read_text().splitlines()]
    audits=[json.loads(l) for l in (root/'audits.jsonl').read_text().splitlines()]
    bykey={(r['task'],r['policy']):r for r in records}
    ids=sorted(set(r['task'] for r in records))
    comparisons={}
    for metric in ['strict_correct','last_number_correct']:
        comparisons[metric]={}
        for a,b in [('fresh','all_corrected'),('late_corrected','baseline'),('late_corrected','fresh')]:
            comparisons[metric][a+' vs '+b]=paired([bykey[i,a]['score'][metric] for i in ids],[bykey[i,b]['score'][metric] for i in ids])
    tok=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    groups={}
    for a in audits:
        assert a['state'][-4+a['seed_offset']]==a['old']
        assert sum(t==tok.mask_token_id for t in a['state'][-4:])==2
        s=tok.decode([a['old']])
        kind='digits' if s.strip().isdigit() else 'special' if a['old'] in tok.all_special_ids else 'other'
        if kind=='digits':
            r=bykey[a['task'],'baseline']
            offset=len(a['state'])-4+a['seed_offset']-len(r['prompt_ids'])
            assert r['tokens'][offset]==a['old']
            span=last_numeric_span(r['text'])
            lo=len(tok.decode(r['tokens'][:offset],skip_special_tokens=True))
            hi=len(tok.decode(r['tokens'][:offset+1],skip_special_tokens=True))
            kind='last_answer_numeral' if span and lo<span[1] and hi>span[0] else 'earlier_numeral'
        groups.setdefault(kind,[]).append(a)
    grouped={k:{'count':len(v),'all_fresh_argmax_differences':sum(a['seed_argmax'][0]!=a['seed_argmax'][1] for a in v),
        'mean_candidate_probability_shift':float(np.mean([a['old_probability'][1]-a['old_probability'][0] for a in v]))} for k,v in groups.items()}
    changed=[{'task':i,'answers':{p:bykey[i,p]['text'] for p in POLICIES},'gold':bykey[i,'baseline']['gold']} for i in ids if len(set(bykey[i,p]['text'] for p in POLICIES))>1]
    report={'evidence':checked,'source_sha':sha(__file__),'comparisons':comparisons,'seed_groups':grouped,
        'exploratory_threshold_audit':threshold_audit(audits),
        'changed_task_outputs':changed,'scope':'Exploratory task-paired statistics, not population/model-seed evidence; unadjusted p-values are not a paper gate'}
    # Exclusive create prevents overwriting earlier analysis.
    with out.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('output',type=Path)
    args=p.parse_args(); analyze(args.root,args.output)
