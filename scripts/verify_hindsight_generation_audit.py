"""Verify preserved output/token identities and counts; no neural replay."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from interaction_sprint.hindsight_generation_audit import MODEL, REVISION, classify


def bundle(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert set(manifest)=={p.name for p in root.iterdir() if p.is_file()}-{'MANIFEST.json'}
    for name,digest in manifest.items():
        path=(root/name).resolve()
        assert path.parent==root.resolve() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    return manifest


def verify(root,source):
    manifest=bundle(root);bundle(source)
    spec=json.loads((root/'spec.json').read_text())
    assert spec['model']==MODEL and spec['revision']==REVISION and spec['max_new_tokens']==24
    assert spec['do_sample'] is False and spec['updates']==0 and spec['gate_reclassification'] is False
    assert spec['source_manifest_sha256']==hashlib.sha256((source/'MANIFEST.json').read_bytes()).hexdigest()
    prompts=json.loads((root/'prompts.json').read_text())
    source_prompts=json.loads((source/'teacher_prompts.json').read_text())
    cases={r['id']:r for r in json.loads((source/'cases.json').read_text())}
    expected=[p for p in source_prompts if p['report']==cases[p['id']]['target']]
    assert prompts==expected and len(prompts)==64
    diag=json.loads((source/'truthful_teacher_diagnostic.json').read_text())
    rows=json.loads((root/'rows.json').read_text())
    assert len(rows)==64 and [r['id'] for r in rows]==[p['id'] for p in prompts]
    runtime=json.loads((root/'runtime.json').read_text())
    ab=runtime['answer_tokens'];eos=runtime['generation']['eos_token_id']
    eos=[eos] if isinstance(eos,int) else eos
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    assert ab==[tok.encode(s,add_special_tokens=False)[0] for s in ('A','B')]
    for r in rows:
        assert r['target']==cases[r['id']]['target'] and r['domain']==cases[r['id']]['domain']
        assert 1<=len(r['generated_tokens'])<=24
        assert r['text']==tok.decode(r['generated_tokens'],skip_special_tokens=True)
        assert r['ended_with_eos']==(r['generated_tokens'][-1] in eos)
        assert r['ended_with_eos'] or len(r['generated_tokens'])==24
        assert r['generated_tokens'][0]==diag[r['id']]['argmax']
        assert r['first_token_correct']==(r['generated_tokens'][0]==ab[r['target']])
        assert 0<=r['first_logprob_replay_max_error']<=.001
        assert all(r[k]==v for k,v in classify(r['text']).items())
    expected=dict(n=len(rows),first_token_correct=sum(r['first_token_correct'] for r in rows),
        strict_correct=sum(r['strict_label']==r['target'] for r in rows),
        unique_mention_correct=sum(r['unique_mentioned_label']==r['target'] for r in rows),
        strict_format_count=sum(r['strict_label'] is not None for r in rows),
        capped_without_eos=sum(not r['ended_with_eos'] for r in rows),
        generated_tokens=sum(len(r['generated_tokens']) for r in rows),
        full_logprob_replay_max_error=max(r['first_logprob_replay_max_error'] for r in rows))
    result=json.loads((root/'RESULT.json').read_text())
    assert all(result[k]==v for k,v in expected.items())
    assert result['updates']==0 and result['paper_green_light'] is False
    return dict(status='HASH_TOKEN_AND_COUNT_VERIFIED_NOT_NEURAL_REPLAY',manifest_files=len(manifest),
        result=result,strict_recovered_ids=[r['id'] for r in rows if not r['first_token_correct'] and r['strict_label']==r['target']],
        strict_lost_ids=[r['id'] for r in rows if r['first_token_correct'] and r['strict_label']!=r['target']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert a.root.resolve() not in a.out.resolve().parents and a.source.resolve() not in a.out.resolve().parents
    result=verify(a.root,a.source)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
