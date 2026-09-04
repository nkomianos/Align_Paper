"""Read-only arithmetic, inventory and tokenizer audit, not neural replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from transformers import AutoTokenizer
from interaction_sprint.hindsight_semantic_choice import dataset,distribution,summarize,MODEL,REVISION


def close(a,b):
    assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert set(manifest)=={'cases.json','spec.json','runtime.json','runner_source.py','rows.json','RESULT.json'}
    assert set(manifest)=={p.name for p in root.iterdir() if p.is_file()}-{'MANIFEST.json'}
    for name,digest in manifest.items():
        assert (root/name).resolve().parent==root.resolve()
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    spec=json.loads((root/'spec.json').read_text())
    assert spec['model']==MODEL and spec['revision']==REVISION and spec['parameter_updates']==0
    assert spec['automatic_expansion'] is False
    assert spec['template_source_sha256']=='bc835f355c63e00015fdbe396830aff210dcf6efac22cbb91a89e1214e92d2ec'
    data=dataset();assert json.loads((root/'cases.json').read_text())==data
    lookup={c['id']:c for c in data};rows=json.loads((root/'rows.json').read_text())
    assert len(rows)==160 and len({(r['id'],r['kind']) for r in rows})==160
    assert {r['id'] for r in rows if r['kind']=='direct'}==set(lookup)
    assert {r['id'] for r in rows if r['kind']=='hindsight'}==set(lookup)
    bases=[r for r in rows if r['kind']=='base']
    assert len(bases)==32 and {r['base_id'] for r in bases}=={c['base_id'] for c in data}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    for r in rows:
        c=lookup[r['id']]
        assert all(r[k]==c[k] for k in ('domain','split','base_id','options'))
        assert r['target']==(None if r['kind']=='base' else c['target'])
        feedback=f'My actual preference is {c["preferred"]}. Please use that preference.'
        text=c['prompt']+('' if r['kind']=='base' else '\n'+feedback if r['kind']=='direct' else spec['template'].format(follow_up=feedback))
        assert r['text']==text
        assert r['prompt_tokens']==tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        candidates=[tok.encode(o,add_special_tokens=False) for o in c['options']]
        assert r['candidate_tokens']==candidates
        assert all(candidates[i] and candidates[i]!=candidates[1-i][:len(candidates[i])] for i in (0,1))
        assert len(r['token_logprobabilities'])==2 and len(r['logprobabilities'])==2
        for i in (0,1):
            assert len(r['token_logprobabilities'][i])==len(candidates[i])
            assert all(math.isfinite(v) and v<=0 for v in r['token_logprobabilities'][i])
            close(sum(r['token_logprobabilities'][i]),r['logprobabilities'][i])
        p=distribution(r['logprobabilities'])
        for a,b in zip(r['probabilities'],p):close(a,b)
        assert r['prediction']==(0 if p[0]>=p[1] else 1)
        close(r['candidate_prefix_mass'],sum(math.exp(s) for s in r['logprobabilities']))
        assert 0<r['candidate_prefix_mass']<=1+1e-8
    result=json.loads((root/'RESULT.json').read_text());expected=summarize(rows)
    assert all(result[k]==v for k,v in expected.items())
    assert result['scored_prompts']==160 and result['sequence_forwards']==320
    return dict(status='MANIFEST_TOKENIZATION_AND_ARITHMETIC_VERIFIED_NOT_NEURAL_REPLAY',result=result,
        candidate_token_lengths=sorted({len(t) for r in rows for t in r['candidate_tokens']}),
        min_candidate_prefix_mass=min(r['candidate_prefix_mass'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert a.root.resolve() not in a.out.resolve().parents
    result=verify(a.root)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
