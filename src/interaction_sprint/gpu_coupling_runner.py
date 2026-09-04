"""GPU native sampling on a frozen SQuAD DEV. Never reads the answer key."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .byte_clock_coupling import digest,label_key,mix64
from .byte_coupling_native import vocabulary,NativeSampler,byte_event_clock
from .squad_coupling import prompt
from .gpu_coupling_sampler import GPUNativeSampler

POLICIES=['independent','token_clock','byte_clock','byte_hierarchical']


def sample_seed(seed,case_id,policy,model_name):
    out=int(label_key(f'{seed}:{case_id}'.encode()))
    return int(mix64(np.uint64(out)^label_key(model_name.encode()))) if policy=='independent' else out


def qualify_cache(model,ids):
    import torch
    x=torch.tensor([ids,ids],device=model.device); start=time.monotonic()
    with torch.inference_mode():
        pre=model(x,use_cache=True,logits_to_keep=1)
        singleton=model(x[:1],use_cache=False,logits_to_keep=1).logits[0,-1]
        chosen=pre.logits[0,-1].topk(2).indices[:,None]
        cached=model(chosen,past_key_values=pre.past_key_values,use_cache=True,logits_to_keep=1).logits[:,-1]
        fresh=model(torch.cat((x,chosen),dim=1),use_cache=False,logits_to_keep=1).logits[:,-1]
        first_error=float((pre.logits[:,-1]-singleton).abs().max())
        next_error=float((cached-fresh).abs().max())
        same=bool((cached.argmax(-1)==fresh.argmax(-1)).all())
    return {'first_batch_serial_error':first_error,'next_cache_full_error':next_error,
            'next_argmax_agreement':same,'seconds':time.monotonic()-start,'calls':4}


def batch_decode(model,tok,sampler,raw,ids,seeds,policy,case_id,name,cfg):
    import torch
    n=len(seeds); x=torch.tensor([ids]*n,device=model.device); past=None
    tokens=[[] for _ in seeds]; events=[[] for _ in seeds]; active=np.ones(n,dtype=bool)
    offsets=np.zeros(n,dtype=int); silent=np.zeros(n,dtype=int)
    keys=[sample_seed(s,case_id,policy,name) for s in seeds]
    calls=0; forwarded_rows=0; sampler_seconds=0.; forward_seconds=0.; start=time.monotonic()
    with torch.inference_mode():
        for step in range(cfg['max_new_tokens']):
            torch.cuda.synchronize(); tick=time.monotonic()
            out=model(x,past_key_values=past,use_cache=True,logits_to_keep=1)
            past=out.past_key_values; z=out.logits[:,-1]
            torch.cuda.synchronize(); forward_seconds+=time.monotonic()-tick; calls+=1; forwarded_rows+=n
            new=np.full(n,tok.eos_token_id,dtype=np.int64)
            tick=time.monotonic()
            for j in np.flatnonzero(active):
                clock=step if policy in ('independent','token_clock') else byte_event_clock(int(offsets[j]),int(silent[j]))
                scaled=z[j].double()/cfg['temperature']
                t,log_probability=sampler.draw_gpu(scaled,keys[j],clock,policy); new[j]=t
                events[j].append({'token':t,'clock':clock,'log_probability':log_probability})
                tokens[j].append(t)
                if raw[t]: offsets[j]+=len(raw[t]); silent[j]=0
                else: silent[j]+=1
                if t==tok.eos_token_id: active[j]=False
            torch.cuda.synchronize(); sampler_seconds+=time.monotonic()-tick
            if not active.any(): break
            # Completed rows remain in the batch only as EOS padding; they emit
            # no further recorded samples and cannot affect other batch rows.
            x=torch.tensor(new[:,None],device=model.device)
    rows=[]
    for s,t,e in zip(seeds,tokens,events):
        b=b''.join(raw[i] for i in t)
        assert b.decode('utf-8',errors='replace')==tok.decode(t,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        rows.append({'model':name,'case_id':case_id,'policy':policy,'seed':s,'tokens':t,'events':e,
                     'text':tok.decode(t,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                     'terminated':t[-1]==tok.eos_token_id,'silent_token_count':sum(not raw[i] for i in t),
                     'raw_bytes_hex':b.hex()})
    return rows,{'forward_calls':calls,'forwarded_rows_including_finished_padding':forwarded_rows,
                 'generated_tokens':sum(map(len,tokens)),'forward_seconds':forward_seconds,
                 'sampler_seconds':sampler_seconds,'seconds':time.monotonic()-start}


def run(config_path,root):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    cfg=json.loads(config_path.read_text()); prepared=Path(cfg['prepared'])
    manifest=json.loads((prepared/'MANIFEST.json').read_text())
    # Check public inputs and preparation. The generation process never opens
    # or hashes the separate answer_key.json; offline analysis owns that file.
    for n in ('cases.json','preparation.json'): assert digest(prepared/n)==manifest[n]
    cases=json.loads((prepared/'cases.json').read_text()); assert len(cases)==cfg['n_cases']
    assert cfg['policies']==POLICIES and cfg['temperature']>0 and len(set(cfg['seeds']))==len(cfg['seeds'])
    root.mkdir(parents=True,exist_ok=False); source=Path(__file__)
    paths=[source,source.with_name('gpu_coupling_sampler.py'),source.with_name('squad_coupling.py'),source.with_name('byte_coupling_native.py'),source.with_name('byte_clock_coupling.py'),config_path]
    frozen={'config':cfg,'cases':cases,'prepared_manifest_sha':digest(prepared/'MANIFEST.json'),
            'sources':{str(p):digest(p) for p in paths},'model_files':{}}
    for spec in cfg['models']:
        frozen['model_files'][spec['name']]={p.name:digest(p) for p in Path(spec['path']).iterdir() if p.suffix in ('.json','.safetensors','.jinja')}
    (root/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(cfg['threads']); started=time.monotonic(); total=0; timings=[]
    for spec in cfg['models']:
        tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True)
        model,info=AutoModelForCausalLM.from_pretrained(spec['path'],local_files_only=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.to('cuda').eval(); raw,labels,groups=vocabulary(tok,model.config.vocab_size,spec['name'])
        sampler=GPUNativeSampler(labels,groups,[len(b) for b in raw],model.device)
        first=True
        for index,c in enumerate(cases):
            ids=tok.apply_chat_template([{'role':'user','content':prompt(c)}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            assert len(ids)<=cfg['max_prompt_tokens']
            if first:
                qualification=qualify_cache(model,ids)
                (root/(spec['name']+'_cache_check.json')).write_text(json.dumps(qualification,indent=2))
                assert max(qualification['first_batch_serial_error'],qualification['next_cache_full_error'])<cfg['cache_tolerance']
                assert qualification['next_argmax_agreement']; first=False
            # Rotate policy order by question to limit systematic warmup/order bias.
            order=POLICIES[index%4:]+POLICIES[:index%4]
            for policy in order:
                for begin in range(0,len(cfg['seeds']),cfg['batch_size']):
                    seeds=cfg['seeds'][begin:begin+cfg['batch_size']]
                    rows,timing=batch_decode(model,tok,sampler,raw,ids,seeds,policy,c['id'],spec['name'],cfg)
                    with (root/'outputs.jsonl').open('a') as f:
                        for r in rows: f.write(json.dumps(r)+'\n')
                    timing.update(model=spec['name'],case_id=c['id'],policy=policy,seeds=seeds,prompt_ids=ids)
                    timings.append(timing)
                    with (root/'timings.jsonl').open('a') as f: f.write(json.dumps(timing)+'\n')
                    total+=len(rows)
        del model
    (root/'runtime.json').write_text(json.dumps({'seconds':time.monotonic()-started,'outputs':total,'torch':torch.__version__,'numpy':np.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(),'dtype':'float32','tf32':False},indent=2))
    assert all(digest(p)==h for p,h in frozen['sources'].items())
    (root/'MANIFEST.json').write_text(json.dumps({p.name:digest(p) for p in root.iterdir()},indent=2))
    return {'outputs':total,'scope':'public SQuAD stochastic DEV, offline scoring still required'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/gpu_coupling_dev_v1.json'))
    a=p.parse_args();print(json.dumps(run(a.config,a.root),indent=2))
