"""Freeze an exposed DEV-only prefix diagnostic without selecting model outcomes."""
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root,out):
    bank=root/'retrieved_stage4/math_bank8_v2'
    manifest=json.loads((bank/'MANIFEST.json').read_text())
    for name,expected in manifest.items():
        assert sha(bank/name)==expected
    source=root/'math500/test.jsonl'
    assert sha(source)=='35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'
    lookup={r['unique_id']:r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    inputs={r['id']:r for r in json.loads((bank/'INPUTS.json').read_text())}
    prefixes=json.loads((bank/'PREFIXES.json').read_text())
    rows=[]; excluded=[]
    for p in prefixes:
        q=lookup[p['base']]
        assert q['problem']==inputs[p['base']]['question']
        assert isinstance(q['solution'],str) and q['solution']
        if len(p['ids'])<32:
            excluded.append(dict(base=p['base'],index=p['index'],reason='fewer than32savedtokens'))
            continue
        rows.append(dict(base=p['base'],index=p['index'],question=q['problem'],reference=q['solution'],
                         prefix_ids=p['ids'][:32]))
    assert len({(r['base'],r['index']) for r in rows})==len(rows)
    out.mkdir(parents=True,exist_ok=False)
    (out/'INPUTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    plan=dict(scope='Exposed nonthinking DEV prefixes; not a thinking-model or training-effect assay',
              contexts=['question+reference','reference only','question only','neither question nor reference'],
              control='Question-only density ratio versus purified target, beta1,c10; same32savedprefixIDs',
              metrics='Full-vocabulary KL/TV and entropy; prefix endpoint only, no answer accuracy or causal recovery claim',
              selection='All saved prefix keys with at least32tokens; no reward, target KL or continuation outcome selection',
              excluded=excluded,rows=len(rows),base_problems=len({r['base'] for r in rows}),
              runtime_estimate='192 or fewer forward passes, roughly5-15GPUminutes including model load; unbenchmarked, plus weight download',
              gate='Source/model/tokenizer identity and finite logits required; mismatched endpoint token IDs invalidate comparison',
              followup='No automatic training. Inspect target differences and null controls before a prospective thinking-model study.',
              source_sha256=sha(source),prefixes_sha256=sha(bank/'PREFIXES.json'),
              model_manifest=json.loads((bank/'MODEL.json').read_text()),runner_sha256=sha(Path(__file__)))
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in plan.items() if k!='model_manifest'},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
