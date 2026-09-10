"""Read-only replay of whole-expression token-logprob aggregation and gates."""
import argparse
import json
import math
from pathlib import Path
from run_unexplored_screens import sha


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,h in manifest.items(): assert sha(root/name)==h, name
    inputs={r['id']:r for r in json.loads((root/'INPUTS.json').read_text())}
    rows=[json.loads(line) for line in (root/'SCORES.jsonl').read_text().splitlines()]
    assert len(rows)==len(inputs)*4
    seen=set();results=[]
    for row in rows:
        key=(row['base'],row['mode'],row['reverse']);assert key not in seen;seen.add(key)
        source=inputs[row['base']];assert row['target']==source['target']
        expected={(a,s) for a in range(4) for s in (source['aliases'][a] if row['mode']=='aliases' else [source['canonical'][a]])}
        assert {(v['action'],v['expression']) for v in row['candidates']}==expected
        assert len(row['candidates'])==len(expected)
        scores=[]
        for v in row['candidates']:
            assert len(v['suffix_ids'])==len(v['token_logprobs'])
            assert all(math.isfinite(p) and p<=0 for p in v['token_logprobs'])
            lp=math.fsum(v['token_logprobs']);assert abs(lp-v['logprob'])<1e-10
            scores.append((v['action'],lp))
        maximum=max(p for _,p in scores)
        masses=[math.fsum(math.exp(p-maximum) for action,p in scores if action==a) for a in range(4)]
        results.append({'base':row['base'],'mode':row['mode'],'reverse':row['reverse'],
            'target':source['target'],'best':max(scores,key=lambda v:v[1])[0],
            'marginal':max(range(4),key=masses.__getitem__)})
    assert seen=={(base,mode,rev) for base in inputs for mode in ('canonical','aliases') for rev in (False,True)}
    metrics={}
    for mode,method in [('canonical','best'),('aliases','best'),('aliases','marginal')]:
        subset=[r for r in results if r['mode']==mode]
        metrics[mode+'/'+method]=sum(r[method]==r['target'] for r in subset)/len(subset)
    agreement=sum(len({r['best'] for r in results if r['mode']=='canonical' and r['base']==base})==1 for base in inputs)/len(inputs)
    saved=json.loads((root/'SUMMARY.json').read_text())
    assert metrics==saved['metrics'] and agreement==saved['canonical_order_agreement']
    assert results==saved['results']
    return {'verified':True,'metrics':metrics,'canonical_order_agreement':agreement,
            'scope':'Saved token-logprob arithmetic and manifest verification; no neural forward replay',
            'n_independent_structural_mechanisms':4,'parameterized_bases':len(inputs)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.root),indent=2))
