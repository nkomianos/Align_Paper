"""Post-result reasoning-budget control, not independent confirmation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from run_benign_memory_extraction import inputs,RULE,parse
from run_unexplored_screens import dump,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);rows=inputs();dump(a.out/'INPUTS.json',rows)
    dump(a.out/'PROTOCOL.json',{'scope':'Post-result same-case reasoning-budget control; not fresh confirmation',
        'source_sha256':sha(Path(__file__)),'generation':'greedy nonthinking native chat, max192 tokens, same limit as extraction',
        'comparison':'Exact answer accuracy and parse/EOS vs original extraction; >=.10 extraction advantage needed to retain interest, with control parse/EOS >=.95',
        'estimate':'2-5 minutes with loading; no midrun timeout'})
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    tok=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval();model.requires_grad_(False)
    dump(a.out/'MODEL.json',{'snapshot':str(a.snapshot),'files':{p.name:sha(p) for p in a.snapshot.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors'}}})
    config=GenerationConfig(do_sample=False,max_new_tokens=192,eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,use_cache=True)
    instruction=RULE+' First work through the possible final events and their threshold decisions. Return only JSON with reasoning (a concise explanation, at most 60 words) followed by answer (YES, NO, or CLARIFY).'
    started=time.monotonic()
    with (a.out/'OUTPUTS.jsonl').open('x') as f,torch.inference_mode():
        for row in rows:
            rendered=tok.apply_chat_template([{'role':'system','content':instruction},{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            encoded=tok(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
            ids=model.generate(**encoded,generation_config=config)[0,encoded.input_ids.shape[1]:].tolist()
            text=tok.decode(ids,skip_special_tokens=True)
            f.write(json.dumps({'id':row['id'],'rendered':rendered,'input_ids':encoded.input_ids[0].tolist(),'output_ids':ids,'text':text,'parsed':parse(text),'eos':bool(ids and ids[-1]==tok.eos_token_id)})+'\n');f.flush()
            print(json.dumps({'completed':row['id'],'seconds':time.monotonic()-started}),flush=True)
    dump(a.out/'TIMING.json',{'inference_seconds':time.monotonic()-started})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file()})


if __name__=='__main__':main()
