"""External DEV continuation bank, with exact digit-logit score projections.

The ten-dimensional score is NOT the gradient of all model parameters. Sampling
uses full-support temperature softmax (no top-k/p), so its derivative is exact
for additive digit-logit biases. The target is finite-horizon reward, not eventual
unbounded success. Prefixes are conditioned-on data, not differentiated actions.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import subprocess
import time
from run_unexplored_screens import sha, dump


def answer(text):
    hits=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',text)
    if not hits:
        return None
    try:return str(Decimal(hits[-1].replace(',','')).normalize())
    except InvalidOperation:return None


def prepare(source):
    rows=[json.loads(x) for x in source.read_bytes().split(b'\n') if x]
    import hashlib
    rows.sort(key=lambda r:hashlib.sha256(('bank-v1:'+r['question']).encode()).hexdigest())
    selected=[]
    for i,r in enumerate(rows[:24]):
        target=answer(r['answer'])
        if target is None:raise ValueError('source answer parse failed')
        selected.append(dict(id=f'gsm_{i:03d}',split='calibration' if i<8 else 'dev',
            question=r['question'], target=target, source_answer=r['answer'],
            question_sha256=hashlib.sha256(r['question'].encode()).hexdigest()))
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--prefixes',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--horizon',type=int,default=768)
    a=p.parse_args()
    if a.samples!=8 or a.horizon!=768:raise ValueError('frozen DEV design')
    a.out.mkdir(parents=True,exist_ok=False)
    data=prepare(a.source)
    dump(a.out/'INPUTS.json',data)
    dump(a.out/'SOURCE.json',{'data_sha256':sha(a.source),'runner_sha256':sha(Path(__file__)),
         'shared_code_sha256':sha(Path(__file__).with_name('run_unexplored_screens.py')),
         'samples':8,'horizon':768,'checkpoint':128,'temperature':.8,'top_p':1.,'top_k':0,
         'seed_base':2026091010,'gradient_scope':'10 additive digit-logit biases; baseline 0.5',
         'classification':'DEVELOPMENTAL_BANK_NOT_TRAINING'})
    if a.prepare_only:
        print('PREPARED',len(data));return
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU occupied; defer: '+occupied)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True,fix_mistral_regex=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    digits=[tok.encode(str(i),add_special_tokens=False) for i in range(10)]
    if any(len(x)!=1 for x in digits):raise ValueError('digit score projection tokenizer failed')
    digit_ids=torch.tensor([x[0] for x in digits],device='cuda')
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    generation=GenerationConfig(do_sample=True,temperature=.8,top_p=1.,top_k=0,
        repetition_penalty=1.,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,use_cache=True)
    dump(a.out/'GENERATION_CONFIG.json',generation.to_dict())
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot),'torch':torch.__version__,
        'transformers':transformers.__version__,'digit_ids':digit_ids.tolist(),
        'gpu':torch.cuda.get_device_name(0),'fix_mistral_regex':True,
        'files':{str(f.relative_to(a.snapshot)):sha(f) for f in sorted(a.snapshot.rglob('*'))
                 if f.is_file() and f.suffix in ('.json','.safetensors','.model','.jinja')}})
    def render(q):
        return tok.apply_chat_template([{'role':'user','content':q+'\nSolve step by step. End with #### followed by the final number.'}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefixes=[]
    started=time.monotonic()
    if a.prefixes:
        prefixes=json.loads(a.prefixes.read_text(encoding='utf8'))
        assert {(r['base'],r['prefix_index']) for r in prefixes}=={(r['id'],j) for r in data for j in (0,1)}
        dump(a.out/'PREFIX_SOURCE.json',{'path':str(a.prefixes),'sha256':sha(a.prefixes)})
    else:
        with torch.inference_mode():
            for i,row in enumerate(data):
                enc=tok(render(row['question']),return_tensors='pt',add_special_tokens=False).to('cuda')
                for j in (0,1):
                    seed=2026091010+i*2+j;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    generated=model.generate(**enc,generation_config=generation,do_sample=True,temperature=.8,top_p=1.,top_k=0,
                        max_new_tokens=48,pad_token_id=tok.pad_token_id,use_cache=True)
                    ids=generated[0,enc['input_ids'].shape[1]:].tolist()
                    prefixes.append({'base':row['id'],'prefix_index':j,'ids':ids,
                         'text':tok.decode(ids,skip_special_tokens=True),'seed':seed,
                         'ended':bool(ids and ids[-1]==tok.eos_token_id)})
                dump(a.out/'PREFIXES.partial.json',prefixes)
                print(json.dumps({'prefix_bases':i+1,'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'PREFIXES.json',prefixes)
    byid={r['id']:r for r in data}
    n=0
    with (a.out/'ROLLOUTS.jsonl').open('x',encoding='utf8') as f,torch.inference_mode():
        for i,prefix in enumerate(prefixes):
            row=byid[prefix['base']]
            text=render(row['question'])+prefix['text']
            enc=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
            if enc['input_ids'].shape[1]>2048:raise ValueError('prompt too long, no truncation')
            seed=2026100000+i
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            result=model.generate(**enc,generation_config=generation,num_return_sequences=8,do_sample=True,temperature=.8,top_p=1.,top_k=0,
                    max_new_tokens=768,pad_token_id=tok.pad_token_id,use_cache=True,
                    return_dict_in_generate=True,output_scores=True)
            generated=result.sequences[:,enc['input_ids'].shape[1]:]
            total=generated.shape[1]
            grads=torch.zeros((8,10),device='cuda',dtype=torch.float64)
            prefix_grads=grads.clone()
            lengths=[]
            for ids in generated.tolist():
                lengths.append(ids.index(tok.eos_token_id)+1 if tok.eos_token_id in ids else len(ids))
            logps=torch.zeros(8,device='cuda',dtype=torch.float64)
            for t,scores in enumerate(result.scores):
                scores=scores.float()
                if not torch.isfinite(scores).all():raise ValueError('nonfinite or filtered sampling logits')
                logz=scores.logsumexp(-1)
                pdigit=(scores[:,digit_ids]-logz[:,None]).exp()
                indicator=(generated[:,t,None]==digit_ids[None,:]).float()
                active=torch.tensor([t<k for k in lengths],device='cuda')
                grads+=((indicator-pdigit)/.8).double()*active[:,None]
                logps+=(scores.gather(1,generated[:,t,None]).squeeze(1)-logz).double()*active
                if t==127:prefix_grads=grads.clone()
            for j,ids in enumerate(generated.tolist()):
                ids=ids[:lengths[j]]; completion=tok.decode(ids,skip_special_tokens=True)
                parsed=answer(prefix['text']+completion)
                r={'base':row['id'],'split':row['split'],'prefix_index':prefix['prefix_index'],'sample':j,
                   'batch_seed':seed,'input_ids':enc['input_ids'][0].tolist(),'ids':ids,'completion':completion,
                   'reward':int(parsed==row['target']),'parsed_answer':parsed,'target':row['target'],
                   'eos':bool(ids and ids[-1]==tok.eos_token_id),'length':len(ids),
                   'prefix_ended':prefix['ended'],'prefix_has_answer':answer(prefix['text']) is not None,
                   'score_projection':grads[j].tolist(),'score_projection_128':(grads[j] if len(ids)<=128 else prefix_grads[j]).tolist(),
                   'sampling_logprob':float(logps[j]),'rendered':text}
                f.write(json.dumps(r,allow_nan=False)+'\n');n+=1
            f.flush()
            del result
            elapsed=time.monotonic()-started
            dump(a.out/'PROGRESS.json',{'rollouts':n,'planned':384,'seconds':elapsed})
            print(json.dumps({'rollouts':n,'seconds':elapsed}),flush=True)
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    print('BANK_COMPLETED',flush=True)


if __name__=='__main__':main()
