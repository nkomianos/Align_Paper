"""Harder external DEV bank: same textual prefix, two frozen continuation policies."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from run_reasoning_bank import answer
from run_unexplored_screens import sha,dump


def prepare(source):
    rows=[json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    eligible=[r for r in rows if r['level'] in (4,5) and re.fullmatch(r'-?\d+',r['answer'])]
    eligible.sort(key=lambda r:hashlib.sha256(('math-policy-values-v2:'+r['unique_id']).encode()).hexdigest())
    assert len(eligible)>=24
    return [{'id':r['unique_id'],'question':r['problem'],'target':answer('#### '+r['answer']),
        'subject':r['subject'],'level':r['level'],'split':'calibration' if i<8 else 'dev'} for i,r in enumerate(eligible[:24])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prefixes',type=Path);a=p.parse_args()
    a.out.mkdir(exist_ok=False);rows=prepare(a.source)
    dump(a.out/'INPUTS.json',rows)
    dump(a.out/'PROTOCOL.json',{'scope':'harder DEV repair, not confirmation or new PRM algorithm',
        'source_sha256':sha(a.source),'runner_sha256':sha(Path(__file__)),'selection':'24 salted-ID-first level4/5 integer-answer MATH500 problems; first8 calibration, final16 DEV',
        'prefix_policy':'Qwen3-8B nonthinking, 128-token samples at temperature1.0; 2 prefixes per problem',
        'continuation_policy':'native nonthinking Qwen3-8B or Qwen3-32B, temperature.8 full support, 8 independent draws per prefix, horizon2048',
        'comparison':'exact same textual prefixes and questions; different model capacities within ONE family',
        'qualification':'each policy parse coverage >=.95 and EOS rate >=.90; mean DEV accuracy between .10 and .90; >=12 DEV questions without ended/answered prefixes',
        'routing':'at least4 DEV questions with opposite prefix rankings and abs within-policy gap >=.5 in BOTH policies; independent seeds required before inference',
        'no_claim':'full-model training, policy-independent PRM impossibility novelty, independent-family replication, uncontaminated benchmark',
        'source_split':'public MATH500 test is declared DEV for this study; no historical locked split opened',
        'runtime_estimate':'8B 15-30 minutes; 32B 30-70 minutes, verify actual throughput; no arbitrary midrun termination'})
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot),'weights':{f.name:sha(f) for f in a.snapshot.glob('*.safetensors')},
        'torch':torch.__version__,'eos_token_id':tok.eos_token_id})
    config=GenerationConfig(do_sample=True,temperature=.8,top_p=1.,top_k=0,repetition_penalty=1.,
        eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
    dump(a.out/'GENERATION_CONFIG.json',config.to_dict())
    def render(q):return tok.apply_chat_template([{'role':'user','content':q+'\nSolve step by step. End with #### followed by the final integer.'}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefixes=[];started=time.monotonic()
    if a.prefixes:
        prefixes=json.loads(a.prefixes.read_text());dump(a.out/'PREFIX_SOURCE.json',{'sha256':sha(a.prefixes),'path':str(a.prefixes)})
        assert {(r['base'],r['index']) for r in prefixes}=={(r['id'],j) for r in rows for j in (0,1)}
    else:
        with torch.inference_mode():
            for i,row in enumerate(rows):
                enc=tok(render(row['question']),return_tensors='pt',add_special_tokens=False).to('cuda')
                for j in (0,1):
                    seed=2026091050+i*2+j;torch.manual_seed(seed)
                    ids=model.generate(**enc,generation_config=config,temperature=1.,max_new_tokens=128)[0,enc.input_ids.shape[1]:].tolist()
                    text=tok.decode(ids,skip_special_tokens=True)
                    prefixes.append({'base':row['id'],'index':j,'text':text,'ids':ids,'seed':seed,
                        'ended':bool(ids and ids[-1]==tok.eos_token_id),'has_answer':answer(text) is not None})
                dump(a.out/'PREFIXES.partial.json',prefixes)
                print(json.dumps({'prefix_questions':i+1,'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'PREFIXES.json',prefixes);lookup={r['id']:r for r in rows};n=0
    with (a.out/'ROLLOUTS.jsonl').open('x') as f,torch.inference_mode():
        for i,prefix in enumerate(prefixes):
            row=lookup[prefix['base']];rendered=render(row['question'])+prefix['text']
            enc=tok(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
            assert enc.input_ids.shape[1]<=4096
            seed=2026092000+i;torch.manual_seed(seed)
            generated=model.generate(**enc,generation_config=config,max_new_tokens=2048,num_return_sequences=8)
            for j,ids in enumerate(generated[:,enc.input_ids.shape[1]:].tolist()):
                if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
                completion=tok.decode(ids,skip_special_tokens=True);parsed=answer(prefix['text']+completion)
                r={'base':row['id'],'split':row['split'],'prefix_index':prefix['index'],'sample':j,
                    'seed':seed,'target':row['target'],'parsed_answer':parsed,'reward':int(parsed==row['target']),
                    'completion':completion,'ids':ids,'eos':bool(ids and ids[-1]==tok.eos_token_id),'length':len(ids),
                    'rendered':rendered,'input_ids':enc.input_ids[0].tolist(),
                    'prefix_ended':prefix['ended'],'prefix_has_answer':prefix['has_answer']}
                f.write(json.dumps(r,allow_nan=False)+'\n');n+=1
            f.flush();dump(a.out/'PROGRESS.json',{'rollouts':n,'planned':384,'seconds':time.monotonic()-started})
            print(json.dumps({'rollouts':n,'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'MANIFEST.json',{f.name:sha(f) for f in a.out.iterdir() if f.is_file()})
    print('MATH_BANK_COMPLETE',flush=True)


if __name__=='__main__':main()
