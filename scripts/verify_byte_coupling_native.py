"""Read-only native qualification verification and explicitly scoped scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from interaction_sprint.byte_clock_coupling import digest,label_key,mix64
from interaction_sprint.byte_coupling_native import vocabulary,NativeSampler,byte_event_clock


def verify(root,key_path):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/n)==h for n,h in manifest.items())
    frozen=json.loads((root/'FROZEN.json').read_text()); cfg=frozen['config']
    assert all(digest(p)==h for p,h in frozen['sources'].items())
    records=[json.loads(l) for l in (root/'outputs.jsonl').read_text().splitlines()]
    cases={c['case_id']:c for c in frozen['cases']}
    assert len(records)==len(cases)*len(cfg['models'])*len(cfg['policies'])*len(cfg['seeds'])
    assert len({(r['model'],r['case_id'],r['policy'],r['seed']) for r in records})==len(records)
    key=json.loads(key_path.read_text()); bymodel={}
    for spec in cfg['models']:
        path=Path(spec['path'])
        assert all(digest(path/n)==h for n,h in frozen['model_files'][spec['name']].items())
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        size=json.loads((path/'config.json').read_text())['vocab_size']
        raw,labels,groups=vocabulary(tok,size,spec['name'])
        sampler=NativeSampler(labels,groups,[len(b) for b in raw])
        rows=[r for r in records if r['model']==spec['name']]
        for r in rows:
            assert r['case_id'] in cases and r['policy'] in cfg['policies'] and r['seed'] in cfg['seeds']
            index=list(cases).index(r['case_id']); stem=f"{spec['name']}_{index}_{r['policy']}_{r['seed']}"
            z=np.load(root/(stem+'.npz'))['logits']
            assert z.shape==(len(r['tokens']),size) and np.isfinite(z).all()
            seed=int(label_key(f"{r['seed']}:{r['case_id']}".encode()))
            if r['policy']=='independent': seed=int(mix64(np.uint64(seed)^label_key(spec['name'].encode())))
            clock=0; silent=0
            for step,(logits,token) in enumerate(zip(z,r['tokens'])):
                nc=step if r['policy'] in ('independent','token_clock') else byte_event_clock(clock,silent)
                assert sampler.draw(logits/cfg['temperature'],seed,nc,r['policy'])==token
                if raw[token]: clock+=len(raw[token]); silent=0
                else: silent+=1
            assert tok.eos_token_id not in r['tokens'][:-1]
            assert r['terminated']==(r['tokens'][-1]==tok.eos_token_id)
            assert r['calls']==len(r['tokens'])
            assert 0<len(r['tokens'])<=cfg['max_new_tokens']
            b=b''.join(raw[t] for t in r['tokens'])
            assert b.hex()==r['raw_bytes_hex']
            assert b.decode('utf-8',errors='replace')==tok.decode(r['tokens'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert r['text']==tok.decode(r['tokens'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            assert r['exact_option_matches']==[i for i,c in enumerate(cases[r['case_id']]['choices']) if c==r['text'].strip()]
            assert r['silent_token_count']==sum(not raw[t] for t in r['tokens'])
        bymodel[spec['name']]={'n':len(rows),'terminated':sum(r['terminated'] for r in rows),
            'exact_option_format':sum(len(r['exact_option_matches'])==1 for r in rows),
            'exact_correct':sum(r['exact_option_matches']==[ord(key[r['case_id']])-65] for r in rows),
            'silent_tokens':sum(r['silent_token_count'] for r in rows)}
    runtime=json.loads((root/'runtime.json').read_text())
    assert runtime['calls']==sum(r['calls'] for r in records)
    return {'verified':'hashes_decoding_sampler_replay_not_model_forward_replay','manifest_sha':digest(root/'MANIFEST.json'),
        'answer_key_sha':digest(key_path),'by_model':bymodel,'runtime':runtime,
        'scope':'Greedy native decoder qualification; no coupled-score variance experiment yet',
        'outputs':[{'model':r['model'],'case_id':r['case_id'],'text':r['text'],'terminated':r['terminated']} for r in records]}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--key',type=Path,default=Path('artifacts/c2c_baseline_dev_20260904_v1/private_answer_key.json'))
    args=p.parse_args(); result=verify(args.root,args.key)
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
