"""DEV repair: exact whole-expression probabilities, not forced A/B labels."""
import argparse
import ast
import json
import math
from pathlib import Path
import random
import subprocess
import time
from run_unexplored_screens import dump,sha


def make_rows():
    rng=random.Random(2026091022);rows=[]
    for i in range(64):
        k=3+i//4;target=i%4
        canonical=[f'x+{k}',f'x-{k}',f'x*{k}',f'x//{k}']
        aliases=[[f'x+{k}',f'{k}+x',f'(x+{k})',f'({k}+x)'],
                 [f'x-{k}',f'x+(-{k})',f'(x-{k})',f'(x+(-{k}))'],
                 [f'x*{k}',f'{k}*x',f'(x*{k})',f'({k}*x)'],
                 [f'x//{k}',f'(x)//{k}',f'(x//{k})',f'((x)//{k})']]
        specs=[f'Increase integer x by {k}.',f'Decrease integer x by {k}.',
               f'Multiply integer x by {k}.',f'Divide integer x by {k} and round down, including for negative x.']
        signatures=[]
        for family in aliases:
            results=[]
            for expression in family:
                tree=ast.parse(expression,mode='eval')
                assert all(isinstance(n,(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Load,
                     ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,ast.USub)) for n in ast.walk(tree))
                results.append(tuple(eval(compile(tree,'<fixed arithmetic>','eval'),{'__builtins__':{}},{'x':x}) for x in range(-8,9)))
            assert len(set(results))==1
            signatures.append(results[0])
        assert len(set(signatures))==4
        order=list(range(4));rng.shuffle(order)
        rows.append({'id':f'expr_{i:03d}','target':target,'spec':specs[target],
                     'canonical':canonical,'aliases':aliases,'order':order,'signatures':signatures})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=make_rows();dump(a.out/'INPUTS.json',rows)
    dump(a.out/'PROTOCOL.json',{'source_sha256':sha(Path(__file__)),'scope':'synthetic DEV repair, not confirmation',
        'policy':'enumerated finite grammar, exact whole-string token probability including EOS',
        'alias_cardinality':4,'controls':'canonical grammar, best alias string, reversed listing',
        'gate':'canonical accuracy >=.9, canonical order agreement >=.9; marginal improvement >=.05 over both baselines',
        'not_claimed':'unrestricted free-generation validity, RL variance reduction, natural tool benchmark superiority'})
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot),'torch':torch.__version__,
        'weights':{f.name:sha(f) for f in a.snapshot.glob('*.safetensors')},'eos':tok.eos_token_id})
    started=time.monotonic();records=[]
    with (a.out/'SCORES.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            for reverse in (False,True):
                order=row['order'][::-1] if reverse else row['order']
                for mode in ('aliases','canonical'):
                    choices=[(action,text) for action in order for text in
                             (row['aliases'][action] if mode=='aliases' else [row['canonical'][action]])]
                    prompt=(row['spec']+' Return a Python expression implementing this for every integer x. '
                            'Select exactly one of these permitted expressions; output it verbatim without code fences or explanation:\n'
                            +'\n'.join(text for _,text in choices))
                    rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
                    prefix=tok.encode(rendered,add_special_tokens=False)
                    suffixes=[tok.encode(text,add_special_tokens=False)+[tok.eos_token_id] for _,text in choices]
                    max_suffix=max(map(len,suffixes));size=len(prefix)+max_suffix
                    ids=torch.tensor([[tok.pad_token_id]*(max_suffix-len(s))+prefix+s for s in suffixes],device='cuda')
                    mask=torch.tensor([[0]*(max_suffix-len(s))+[1]*(len(prefix)+len(s)) for s in suffixes],device='cuda')
                    logits=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=max_suffix+1).logits.float()
                    if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
                    logprob=logits.log_softmax(-1);values=[]
                    for j,suffix in enumerate(suffixes):
                        target=torch.tensor(suffix,device='cuda')
                        scores=logprob[j,-len(suffix)-1:-1].gather(1,target[:,None]).squeeze(1)
                        values.append({'action':choices[j][0],'expression':choices[j][1],
                            'suffix_ids':suffix,'token_logprobs':scores.tolist(),'logprob':float(scores.double().sum())})
                    r={'base':row['id'],'target':row['target'],'mode':mode,'reverse':reverse,
                       'rendered':rendered,'prefix_ids':prefix,'candidates':values}
                    f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();records.append(r)
            print(json.dumps({'bases':len(records)//4,'seconds':time.monotonic()-started}),flush=True)
    results=[]
    for r in records:
        c=r['candidates'];maximum=max(v['logprob'] for v in c)
        mass=[sum(math.exp(v['logprob']-maximum) for v in c if v['action']==k) for k in range(4)]
        results.append({'base':r['base'],'mode':r['mode'],'reverse':r['reverse'],
                       'best':max(c,key=lambda v:v['logprob'])['action'],
                       'marginal':max(range(4),key=mass.__getitem__),'target':r['target']})
    metrics={}
    for mode,method in (('canonical','best'),('aliases','best'),('aliases','marginal')):
        v=[r for r in results if r['mode']==mode]
        metrics[mode+'/'+method]=sum(r[method]==r['target'] for r in v)/len(v)
    canonical=[r for r in results if r['mode']=='canonical'];agreement=sum(
        next(r for r in canonical if r['base']==base and not r['reverse'])['best']==
        next(r for r in canonical if r['base']==base and r['reverse'])['best'] for base in {r['base'] for r in canonical})/64
    qualified=metrics['canonical/best']>=.9 and agreement>=.9
    advantage=metrics['aliases/marginal']-max(metrics['canonical/best'],metrics['aliases/best'])
    dump(a.out/'SUMMARY.json',{'classification':'DEVELOPMENTAL','metrics':metrics,'canonical_order_agreement':agreement,
        'advantage':advantage,'route':'INVALID_CANONICAL_QUALIFICATION' if not qualified else
                ('DEV_SIGNAL' if advantage>=.05 else 'STOP_NO_DECISIVE_MARGINAL_ADVANTAGE'),'results':results})
    dump(a.out/'MANIFEST.json',{f.name:sha(f) for f in a.out.iterdir() if f.is_file()})
    print('EXPRESSION_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
