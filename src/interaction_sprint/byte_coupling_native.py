"""ByteLevel-tokenizer qualification for a native-marginal coupling DEV.

Supports raw reversible ByteLevel vocabularies only. No retokenization, filtering
of special tokens, vocabulary intersection restriction, or answer-key loading.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .byte_clock_coupling import digest, label_key, mix64


def byte_decoder():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs=bs.copy(); n=0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n+=1
    return {chr(c):b for b,c in zip(bs,cs)}


def vocabulary(tok, model_vocab_size=None, model_name='model'):
    data=json.loads(tok.backend_tokenizer.to_str())
    assert data['model']['type']=='BPE' and data['decoder']['type']=='ByteLevel'
    decoder=byte_decoder(); special=set(tok.all_special_ids)
    added={t['id']:t['content'] for t in data['added_tokens']}
    raw=[]; labels=[]; groups=[]
    for i in range(len(tok)):
        s=tok.convert_ids_to_tokens(i)
        b=added[i].encode('utf-8') if i in added else bytes(decoder[c] for c in s)
        if not b: raise ValueError('Empty token requires a different clock proof')
        raw.append(b)
        if i in special:
            name=b'EOS' if i==tok.eos_token_id else s.encode('utf-8')
            labels.append(b'special:'+name); groups.append(b'special:'+name)
        else:
            labels.append(b'raw:'+b); groups.append(b'byte:'+b[:1])
    # Qwen has padded output rows absent from its tokenizer; native decode drops
    # those IDs. Preserve their probability, label them as model-specific silent
    # events, and use a consecutive-silent-event clock to prevent noise reuse.
    size=len(tok) if model_vocab_size is None else model_vocab_size
    if size<len(tok): raise ValueError('Tokenizer exceeds model output vocabulary')
    for i in range(len(tok),size):
        assert tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)==''
        raw.append(b''); label=f'opaque:{model_name}:{i}'.encode()
        labels.append(label); groups.append(label)
    if len(set(labels)) != len(labels): raise ValueError('Duplicate token-byte labels')
    return raw,labels,groups


def byte_event_clock(offset, silent_events):
    # Finite run bounds allow an injective tuple encoding, not a repeated clock.
    if not 0<=silent_events<65536 or not 0<=offset<2**48: raise ValueError('Clock overflow')
    return (offset<<16)|silent_events


class NativeSampler:
    def __init__(self,labels,groups,lengths):
        assert len(labels)==len(groups)==len(lengths)
        assert len(set(labels))==len(labels) and min(lengths)>=0
        self.keys=np.array([label_key(b'token\0'+s) for s in labels],dtype=np.uint64)
        unique=sorted(set(groups)); indexes={g:i for i,g in enumerate(unique)}
        self.group_ids=np.array([indexes[g] for g in groups])
        self.group_keys=np.array([label_key(b'first-byte\0'+s) for s in unique],dtype=np.uint64)
        self.lengths=np.array(lengths)

    @staticmethod
    def gumbel(keys,seed,clock):
        bits=mix64(np.uint64(seed)^mix64(np.uint64(clock))^keys)
        return -np.log(-np.log(((bits>>np.uint64(12)).astype(np.float64)+.5)/2**52))

    def draw(self,logits,seed,clock,policy):
        z=np.asarray(logits,dtype=np.float64)
        assert z.shape==self.keys.shape and np.isfinite(z).all()
        if policy=='greedy': return int(z.argmax())
        assert policy in ('token_clock','byte_clock','byte_hierarchical','independent')
        g=self.gumbel(self.keys,seed,clock)
        if policy=='byte_hierarchical':
            p=np.exp(z-z.max()); p/=p.sum()
            mass=np.bincount(self.group_ids,weights=p,minlength=len(self.group_keys))
            logmass=np.full_like(mass,-np.inf)
            np.log(mass,out=logmass,where=mass>0)
            group=int(np.argmax(logmass+self.gumbel(self.group_keys,seed,clock)))
            g=np.where(self.group_ids==group,g,-np.inf)
        return int(np.argmax(z+g))


def make_prompt(case):
    options='\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(case['choices']))
    return case['question']+'\n'+options+'\nReturn only the exact text of the correct option, not its letter. Do not explain.'


def run(config_path,root):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg=json.loads(config_path.read_text()); root.mkdir(parents=True,exist_ok=False)
    data_path=Path(cfg['cases']); data=json.loads(data_path.read_text())
    cases=[c for ds in ['openbookqa','arc_challenge'] for c in [x for x in data if x['dataset']==ds][:cfg['cases_per_dataset']]]
    source=Path(__file__); helper=source.with_name('byte_clock_coupling.py')
    pinned={}
    for spec in cfg['models']:
        path=Path(spec['path']); pinned[spec['name']]={p.name:digest(p) for p in path.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.jinja')}
    freeze={'config':cfg,'sources':{str(p):digest(p) for p in [source,helper,config_path,data_path]},'model_files':pinned,'cases':cases}
    (root/'FROZEN.json').write_text(json.dumps(freeze,indent=2))
    torch.set_num_threads(cfg['threads']); records=[]; start=time.monotonic(); calls=0
    for spec in cfg['models']:
        tok=AutoTokenizer.from_pretrained(spec['path'],local_files_only=True)
        model,loading=AutoModelForCausalLM.from_pretrained(spec['path'],local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        assert not any(loading[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        model.eval(); raw,labels,groups=vocabulary(tok,model.config.vocab_size,spec['name'])
        sampler=NativeSampler(labels,groups,[len(b) for b in raw])
        for index,case in enumerate(cases):
            prompt=make_prompt(case)
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            for policy in cfg['policies']:
                for seed in cfg['seeds']:
                    x=torch.tensor([ids]); tokens=[]; clock=0; silent=0; saved=[]
                    sampling_seed=int(label_key(f"{seed}:{case['case_id']}".encode()))
                    if policy=='independent': sampling_seed=int(mix64(np.uint64(sampling_seed)^label_key(spec['name'].encode())))
                    with torch.inference_mode():
                        for step in range(cfg['max_new_tokens']):
                            z=model(x,use_cache=False,logits_to_keep=1).logits[0,-1].numpy()
                            calls+=1; saved.append(z.copy())
                            noise_clock=step if policy in ('independent','token_clock') else byte_event_clock(clock,silent)
                            token=sampler.draw(z/cfg['temperature'],sampling_seed,noise_clock,policy)
                            tokens.append(token)
                            if raw[token]: clock+=len(raw[token]); silent=0
                            else: silent+=1
                            x=torch.cat((x,torch.tensor([[token]])),dim=1)
                            if token==tok.eos_token_id: break
                    joined=b''.join(raw[t] for t in tokens)
                    decoded=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    assert joined.decode('utf-8',errors='replace')==decoded
                    text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    # Format/capability audit only. Gold is not loaded by this runner.
                    matched=[i for i,c in enumerate(case['choices']) if text.strip()==c]
                    r={'model':spec['name'],'case_id':case['case_id'],'policy':policy,'seed':seed,'tokens':tokens,'text':text,'terminated':tokens[-1]==tok.eos_token_id,'exact_option_matches':matched,'calls':len(saved),'raw_bytes_hex':joined.hex(),'silent_token_count':sum(not raw[t] for t in tokens)}
                    records.append(r)
                    stem=f"{spec['name']}_{index}_{policy}_{seed}"
                    np.savez_compressed(root/(stem+'.npz'),logits=np.stack(saved))
                    with (root/'outputs.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
        del model
    (root/'runtime.json').write_text(json.dumps({'seconds':time.monotonic()-start,'calls':calls,'torch':torch.__version__},indent=2))
    assert all(digest(p)==h for p,h in freeze['sources'].items())
    (root/'MANIFEST.json').write_text(json.dumps({p.name:digest(p) for p in root.iterdir()},indent=2))
    return {'outputs':len(records),'calls':calls,'scope':'native decoding qualification; no scientific decision'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/byte_coupling_native_qualification_v1.json'))
    args=parser.parse_args(); print(json.dumps(run(args.config,args.root),indent=2))
