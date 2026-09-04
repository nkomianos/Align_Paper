"""CPU apparatus qualification on a pinned released block-diffusion model.

Extract only reviewed model definitions and mask helper; never import upstream
rollout orchestration. No checkpoints or source files are altered.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
import transformers
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'artifacts/opdlm_models/models--divelab--OPDLM-0.6B/snapshots/e6d44e247a86136c9c45e85af5bc6cdbeca73d41'
SOURCE = ROOT/'artifacts/opdlm_source_20260904/sample/bd3lm_rl_rollout.py'
SOURCE_SHA = '2f58c85ddf29d4707d32cecf3d5f3fce60331042d6dfba2bb1e3244dab9b3745'
ARMS = ['native','plain','all_corrected','late_corrected']

def json_default(value):
    if isinstance(value,set): return sorted(value)
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def upstream():
    assert sha(SOURCE)==SOURCE_SHA
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
    names={'A2DQwen3Config','_register_a2d_model_classes','_prepare_for_sampling'}
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==3
    env={'__name__':__name__,'torch':torch,'transformers':transformers,'nn':nn,
         'DynamicCache':DynamicCache,'BaseModelOutputWithPast':BaseModelOutputWithPast,
         '_prepare_4d_attention_mask':_prepare_4d_attention_mask,'_a2d_model_registered':False}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    env['_register_a2d_model_classes']()
    return env

class Instrument:
    def __init__(self,model):
        self.layers=[x.self_attn for x in model.model.layers]
        self.mode='plain'; self.seed=0; self.cache={}; self.saved={}
        self.original=[]

    def __enter__(self):
        for index,module in enumerate(self.layers):
            self.original.append(module.forward)
            def forward(m,hidden_states,position_embeddings,attention_mask,past_key_values=None,_index=index,**kwargs):
                assert past_key_values is None and kwargs.get('past_key_value') is None
                shape=hidden_states.shape[:-1]; view=(*shape,-1,m.head_dim)
                q=m.q_norm(m.q_proj(hidden_states).view(view)).transpose(1,2)
                k=m.k_norm(m.k_proj(hidden_states).view(view)).transpose(1,2)
                v=m.v_proj(hidden_states).view(view).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
                if self.mode=='collect': self.saved[_index]=(k[:,:,self.seed:self.seed+1].clone(),v[:,:,self.seed:self.seed+1].clone())
                inject=self.mode=='all_corrected' or (self.mode=='late_corrected' and _index==len(self.layers)-1)
                clean_k,clean_v=k,v
                if inject:
                    k=k.clone(); v=v.clone()
                    k[:,:,self.seed:self.seed+1],v[:,:,self.seed:self.seed+1]=self.cache[_index]
                k=repeat_kv(k,m.num_key_value_groups); v=repeat_kv(v,m.num_key_value_groups)
                z=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask,dropout_p=0.,is_causal=False,scale=m.scaling)
                if inject:
                    # Restore the entire verification row using its current K/V,
                    # exactly replacing the single cached seed column.
                    ck=repeat_kv(clean_k,m.num_key_value_groups); cv=repeat_kv(clean_v,m.num_key_value_groups)
                    mask=attention_mask[:,:,self.seed:self.seed+1,:]
                    z[:,:,self.seed:self.seed+1]=torch.nn.functional.scaled_dot_product_attention(q[:,:,self.seed:self.seed+1],ck,cv,attn_mask=mask,dropout_p=0.,is_causal=False,scale=m.scaling)
                return m.o_proj(z.transpose(1,2).reshape(*shape,-1).contiguous()),None
            module.forward=MethodType(forward,module)
        return self

    def __exit__(self,*exc):
        for m,f in zip(self.layers,self.original): m.forward=f

def cases(tokenizer,path,n=16):
    # Existing exposed sentences: deliberately DEV, not confirmatory replication.
    source=json.loads(path.read_text())
    result=[]
    for c in source:
        ids=tokenizer.encode(c['text'],add_special_tokens=False)
        if len(ids)<24: continue
        end=min(len(ids)//4*4,64)
        result.append({'id':c['id'],'article':c['article'],'ids':ids[:end],
                       'start':end-4,'gold':ids[end-4:end]})
        if len(result)==n: break
    assert len(result)==n
    return result

def summary(logits,records):
    prediction=logits.argmax(-1); gold=np.array([r['gold'] for r in records])
    seeds=np.array([r['seed_offset'] for r in records]); rows=np.arange(len(records))
    draft=np.ones_like(gold,dtype=bool); draft[rows,seeds]=False
    arms={}
    for i,arm in enumerate(ARMS):
        correct=prediction[:,i]==gold
        arms[arm]={'verification_correct':int(correct[rows,seeds].sum()),
                   'other_three_correct':int(correct[draft].sum()),
                   'changed_verification':int((prediction[rows,i,seeds]!=prediction[rows,0,seeds]).sum())}
    plain=float(np.max(np.abs(logits[:,0]-logits[:,1])))
    late=float(np.max(np.abs(logits[rows,0,seeds]-logits[rows,3,seeds])))
    return {'n':len(records),'arms':arms,'plain_max_logit_error':plain,
            'late_seed_max_logit_error':late,'controls_pass':plain<3e-4 and late<3e-4,
            'scope':'16-case apparatus DEV; not official COVER, benchmark or paper gate'}

def run(root):
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(9404)
    env=upstream()
    cfg=env['A2DQwen3Config'].from_pretrained(MODEL,local_files_only=True)
    model,loading=transformers.AutoModelForMaskedLM.from_pretrained(MODEL,config=cfg,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys'),loading
    model.eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert tokenizer.mask_token_id==151669
    cp=ROOT/'artifacts/cache_short_context_prepared_v1/cases.json'
    selected=cases(tokenizer,cp)
    (root/'cases.json').write_text(json.dumps(selected,indent=2)+'\n')
    records=[]; outputs=[]; calls=0; start=time.monotonic()
    with torch.inference_mode():
        for c in selected:
            x=torch.tensor([c['ids']]); p=c['start']; x[:,p:]=tokenizer.mask_token_id
            mask,pos=env['_prepare_for_sampling'](x,4,tokenizer.pad_token_id)
            def f(inp):
                nonlocal calls
                calls+=1
                # Only final block vocabulary logits: keeps CPU memory bounded.
                return model(inp,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=4).logits[0]
            native=f(x)
            probs=native.float().softmax(-1)
            seed_offset=int(probs.max(-1).values.argmax()); seed=p+seed_offset
            candidate=int(native[seed_offset].argmax())
            with Instrument(model) as instrument:
                instrument.seed=seed
                instrument.mode='plain'; plain=f(x)
                cx=x.clone(); cx[0,seed]=candidate
                instrument.mode='collect'; f(cx); instrument.cache=instrument.saved
                instrument.mode='all_corrected'; all_corrected=f(x)
                instrument.mode='late_corrected'; late=f(x)
            stack=torch.stack([native,plain,all_corrected,late]).cpu().numpy()
            outputs.append(stack)
            r={'id':c['id'],'gold':c['gold'],'seed_offset':seed_offset,'candidate':candidate,
               'predictions':stack.argmax(-1).tolist()}
            records.append(r)
            with (root/'progress.jsonl').open('a') as out: out.write(json.dumps(r)+'\n')
    z=np.stack(outputs); np.save(root/'logits.npy',z)
    result=summary(z,records)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    runtime={'calls':calls,'seconds':time.monotonic()-start,'torch':torch.__version__,
             'transformers':transformers.__version__,'loading':loading,
             'source':str(SOURCE),'source_sha':sha(SOURCE),'runner_sha':sha(__file__),
             'model':str(MODEL),'model_files':{p.name:sha(p) for p in MODEL.iterdir() if p.is_file()},
             'case_source':str(cp),'case_source_sha':sha(cp)}
    (root/'runtime.json').write_text(json.dumps(runtime,indent=2,default=json_default)+'\n')
    manifest={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result

def verify(root):
    m=json.loads((root/'MANIFEST.json').read_text())
    assert set(m)=={'cases.json','progress.jsonl','logits.npy','summary.json','runtime.json'}
    assert all(sha(root/n)==h for n,h in m.items())
    rt=json.loads((root/'runtime.json').read_text())
    assert rt['runner_sha']==sha(__file__) and rt['source_sha']==sha(SOURCE)==SOURCE_SHA
    assert rt['case_source_sha']==sha(rt['case_source'])
    assert all(sha(Path(rt['model'])/n)==h for n,h in rt['model_files'].items())
    records=[json.loads(x) for x in (root/'progress.jsonl').read_text().splitlines()]
    z=np.load(root/'logits.npy'); assert np.isfinite(z).all() and z.shape[:3]==(16,4,4)
    assert rt['calls']==80 and len(records)==16
    assert all(z[i].argmax(-1).tolist()==r['predictions'] for i,r in enumerate(records))
    result=summary(z,records)
    assert result==json.loads((root/'summary.json').read_text())
    return {'verified':'bytes_and_metrics_not_forward_replay','manifest_sha':sha(root/'MANIFEST.json'),'summary':result}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','verify']); p.add_argument('root',type=Path)
    args=p.parse_args(); print(json.dumps(globals()[args.action](args.root),indent=2))
