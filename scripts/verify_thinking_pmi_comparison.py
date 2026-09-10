"""CPU audit of all selected probes, inputs, raw logits and aggregate metrics."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from pmi_target_controls import corrected_target,distribution_kl
from verify_thinking_pmi_traces import digest


def run(root,selection,data,snapshot):
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/k)==v for k,v in manifest.items())
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    assert protocol['selection_sha256']==digest(selection)
    assert protocol['source_inputs_sha256']==digest(data/'INPUTS.json')
    inputs={r['base']:r for r in json.loads((data/'INPUTS.json').read_text())}
    all_keys=sorted(inputs)
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    selected=json.loads(selection.read_text())['rows']
    rows=[json.loads(line) for line in (root/'ROWS.jsonl').read_text().splitlines()]
    logits=torch.load(root/'LOGITS.pt',map_location='cpu',weights_only=True)
    assert len(rows)==len(logits)==len(selected)
    assert len({(r['base'],r['kind']) for r in rows})==len(rows)
    keys=sorted({r['base'] for r in selected})
    for row,ls,s in zip(rows,logits,selected):
        assert (row['base'],row['kind'],row['length'])==(s['base'],s['kind'],s['length'])
        assert set(ls)=={'base','teacher','reference','unconditional','wrong_teacher','wrong_reference','cached_base'}
        assert set(row['inputs'])==set(ls)-{'cached_base'}
        assert all(torch.isfinite(v).all() for v in ls.values())
        for v in row['inputs'].values():assert v['ids'][-s['length']:]==s['prefix_ids']
        assert row['inputs']['base']['ids']==s['input_ids']+s['prefix_ids']
        assert row['inputs']['base']['rendered']==s['rendered']
        assert row['wrong_reference_base']!=row['base']
        other=all_keys[(all_keys.index(row['base'])+1)%len(all_keys)]
        assert row['wrong_reference_base']==other
        source=inputs[row['base']];qtext=source['question'];ref=source['reference'];wrong_ref=inputs[other]['reference']
        texts={'base':qtext,'unconditional':'','teacher':qtext+'\nReference solution:\n'+ref,
            'reference':'Reference solution:\n'+ref,'wrong_teacher':qtext+'\nReference solution:\n'+wrong_ref,
            'wrong_reference':'Reference solution:\n'+wrong_ref}
        for arm,content in texts.items():
            rendered=tok.apply_chat_template([{'role':'user','content':content+'\nSolve step by step.'}],tokenize=False,
                add_generation_prompt=True,enable_thinking=True)
            assert row['inputs'][arm]['rendered']==rendered
            assert row['inputs'][arm]['ids']==tok.encode(rendered,add_special_tokens=False)+s['prefix_ids']
        base=ls['base'].log_softmax(-1);cached=ls['cached_base'].log_softmax(-1)
        p=corrected_target(ls['base'],ls['teacher'],ls['reference'])
        q=corrected_target(ls['base'],ls['base'],ls['unconditional'])
        w=corrected_target(ls['base'],ls['wrong_teacher'],ls['wrong_reference'])
        tv=lambda a,b:float((a.exp()-b.exp()).abs().sum()/2)
        expected=dict(tv_purified_control=tv(p,q),tv_purified_wrong=tv(p,w),tv_wrong_control=tv(w,q),
            tv_base_cached=tv(base,cached),kl_purified_base=float(distribution_kl(p,base)),
            entropy_base=float(-(base.exp()*base).sum()),entropy_purified=float(-(p.exp()*p).sum()),
            entropy_control=float(-(q.exp()*q).sum()))
        assert abs(float(-(cached.exp()*cached).sum())-s['selection_entropy'])<1e-5
        assert all(abs(row[k]-v)<5e-5 for k,v in expected.items())
    summary=json.loads((root/'SUMMARY.json').read_text())
    comparable=all(r['tv_base_cached']<=.05 for r in rows)
    assert summary['numerical_comparability']==comparable
    aggregate={}
    for kind in ['high_entropy','random']:
        subset=[r for r in rows if r['kind']==kind];assert len(subset)==len(keys)
        aggregate[kind]={name:{'mean':statistics.mean(r[name] for r in subset),
            'median':statistics.median(r[name] for r in subset)} for name in expected}
    return dict(verified=True,numerical_comparability=comparable,rows=len(rows),problems=len(keys),
        maximum_cached_tv=max(r['tv_base_cached'] for r in rows),aggregate=aggregate,
        scope='Exposed DEV descriptive distributions; no accuracy or training effect; positions paired within questions')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','selection','out','data','snapshot']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=run(a.root,a.selection,a.data,a.snapshot)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
