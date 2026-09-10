"""Independent CPU replay of base-only selection from retained raw logits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(root,data,snapshot):
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/name)==value for name,value in manifest.items())
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert digest(data/'INPUTS.json')==protocol['source_inputs_sha256']
    assert digest(data/'PLAN.json')==protocol['source_plan_sha256']
    assert digest(snapshot/'tokenizer_config.json')==protocol['tokenizer_sha256']
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    opener=tok.encode('<think>',add_special_tokens=False);closer=tok.encode('</think>',add_special_tokens=False)
    assert len(opener)==len(closer)==1
    inputs={r['base']:r for r in json.loads((data/'INPUTS.json').read_text())}
    traces=sorted(root.glob('trace_*.json'));assert len(traces)==len(inputs)==24
    selected=[];qualified=0
    for i,path in enumerate(traces):
        row=json.loads(path.read_text());key=sorted(inputs)[i]
        assert row['base']==key and row['seed']==2026091070+i
        rendered=tok.apply_chat_template([{'role':'user','content':inputs[key]['question']+'\nSolve step by step.'}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert row['rendered']==rendered
        assert row['input_ids']==tok.encode(rendered,add_special_tokens=False)
        assert row['text']==tok.decode(row['ids'],skip_special_tokens=False)
        with np.load(root/f'logits_{i:02d}.npz',allow_pickle=False) as archive:
            logits=torch.from_numpy(archive['logits']).float()
        assert len(logits)==len(row['ids'])<=256 and torch.isfinite(logits).all()
        lp=logits.log_softmax(-1);ent=-(lp.exp()*lp).sum(-1)
        assert np.allclose(ent.numpy(),row['entropies'],rtol=0,atol=1e-5)
        ids=row['ids'];opening=ids.index(opener[0]) if opener[0] in ids else None
        closing=ids.index(closer[0]) if closer[0] in ids else len(ids)
        assert opening==row['opening_think_index'] and closing==row['closing_think_index']
        eligible=[t for t in range(32,len(ids)) if opening is not None and opening<t<closing]
        if eligible:
            high=sorted(eligible,key=lambda t:(-float(ent[t]),t))[0]
            random=eligible[int(hashlib.sha256(('thinking-pmi-random-v1:'+key).encode()).hexdigest(),16)%len(eligible)]
            positions={'high_entropy':high,'random':random}
        else:positions=None
        assert positions==row['positions']
        qualifies=bool(positions and float(ent[positions['high_entropy']])>=1.)
        assert qualifies==row['qualifies'];qualified+=qualifies
        if positions:
            for kind,length in positions.items():
                selected.append(dict(base=key,kind=kind,length=length,prefix_ids=ids[:length],
                    rendered=rendered,input_ids=row['input_ids'],selection_entropy=float(ent[length])))
        del logits,lp,ent
    summary=json.loads((root/'SUMMARY.json').read_text())
    assert summary['qualified']==qualified and summary['comparison_admitted']==(qualified>=16)
    receipt=dict(verified=True,qualified=qualified,comparison_admitted=qualified>=16,
        manifest_sha256=digest(root/'MANIFEST.json'),verifier_sha256=digest(Path(__file__)),
        rows=selected,scope='Selection verified; generated thinking tags do not establish reasoning correctness')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','data','snapshot','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();receipt=run(a.root,a.data,a.snapshot)
    with a.out.open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
