"""Read-only follow-up generation from all 64 frozen truthful teacher prompts.

Never alters or reclassifies the original learning gate. No parameter updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

MODEL = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def classify(text):
    """Strict format plus explicitly exploratory unambiguous-label diagnostic."""
    s = text.strip()
    strict = ('A', 'B').index(s) if s in ('A', 'B') else None
    labels = sorted(set(re.findall(r'(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])', s)))
    unique = ('A', 'B').index(labels[0]) if len(labels) == 1 else None
    return dict(strict_label=strict, unique_mentioned_label=unique)


def run(source, root):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    source = source.resolve()
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    for name, digest in manifest.items():
        path = (source / name).resolve()
        if path.parent != source or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Source evidence mismatch')
    root.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        with (root / name).open('x', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, allow_nan=False)
    spec = dict(model=MODEL, revision=REVISION, max_new_tokens=24, do_sample=False,
        source_manifest_sha256=hashlib.sha256((source / 'MANIFEST.json').read_bytes()).hexdigest(),
        scope='All64 frozen truthful-training teacher prompts; diagnostic, not independent confirmation',
        updates=0, gate_reclassification=False, automatic_expansion=False)
    write('spec.json', spec)
    (root / 'runner_source.py').write_bytes(Path(__file__).read_bytes())
    cases = {r['id']: r for r in json.loads((source / 'cases.json').read_text())}
    prompts = [r for r in json.loads((source / 'teacher_prompts.json').read_text())
               if r['report'] == cases[r['id']]['target']]
    if len(prompts) != 64 or len({r['id'] for r in prompts}) != 64:
        raise ValueError('Expected all64 unique truthful teacher prompts')
    write('prompts.json', prompts)
    torch.set_num_threads(4); torch.manual_seed(90414001)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').eval()
    ab = [tok.encode(s, add_special_tokens=False) for s in ('A','B')]
    if any(len(ids) != 1 for ids in ab):
        raise ValueError('Answer token contract')
    ab = [ids[0] for ids in ab]
    teacher = torch.load(source / 'teacher_logprobabilities.pt', weights_only=True, map_location='cpu')
    config = GenerationConfig(max_new_tokens=24, do_sample=False, num_beams=1,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        use_cache=True, return_dict_in_generate=True, output_scores=True)
    write('runtime.json', dict(torch=torch.__version__, transformers=transformers.__version__,
        generation=config.to_dict(), answer_tokens=ab))
    rows=[]; start=time.monotonic()
    try:
        for p in prompts:
            ids=tok.apply_chat_template([dict(role='user', content=p['text'])], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if ids != p['token_ids']:
                raise ValueError('Prompt tokenizer changed')
            x=torch.tensor([ids])
            with torch.no_grad():
                out=model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=config)
            lp=out.scores[0][0].double().log_softmax(0)
            target=cases[p['id']]['target']
            delta=float((lp-teacher[p['id']][target].double()).abs().max())
            if delta > .001:
                raise ValueError(f'Full first-token log probability replay mismatch: {delta}')
            generated=out.sequences[0,len(ids):].tolist()
            text=tok.decode(generated, skip_special_tokens=True)
            eos=config.eos_token_id
            eos=[eos] if isinstance(eos,int) else (eos or [])
            row=dict(id=p['id'], domain=cases[p['id']]['domain'], target=target,
                generated_tokens=generated, text=text, first_token_correct=generated[0]==ab[target],
                first_logprob_replay_max_error=delta, ended_with_eos=generated[-1] in eos,
                **classify(text))
            rows.append(row)
            if len(rows)%16==0:
                print(json.dumps(dict(completed=len(rows),total=64,elapsed=time.monotonic()-start)),flush=True)
        write('rows.json', rows)
        result=dict(status='GENERATION_DIAGNOSTIC_COMPLETE_NOT_GATE_RECLASSIFICATION', n=len(rows),
            first_token_correct=sum(r['first_token_correct'] for r in rows),
            strict_correct=sum(r['strict_label']==r['target'] for r in rows),
            unique_mention_correct=sum(r['unique_mentioned_label']==r['target'] for r in rows),
            strict_format_count=sum(r['strict_label'] is not None for r in rows),
            capped_without_eos=sum(not r['ended_with_eos'] for r in rows),
            full_logprob_replay_max_error=max(r['first_logprob_replay_max_error'] for r in rows),
            generated_tokens=sum(len(r['generated_tokens']) for r in rows),elapsed=time.monotonic()-start,
            updates=0,paper_green_light=False)
        write('RESULT.json',result)
        write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(result),flush=True)
    except Exception as error:
        write('partial_rows.json',rows)
        write('FAILED.json',dict(error=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.source,a.root)
