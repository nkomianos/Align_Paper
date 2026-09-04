"""Native four-token block decoding qualification; no cache intervention."""
import argparse
import json
from pathlib import Path
import time
import torch
import transformers
from interaction_sprint.opdlm_cache_dev import MODEL, SOURCE, sha, upstream, json_default

QUESTIONS = [
    'Reply with only the answer. What is the capital of France?',
    'Reply with only the answer. What is two plus three?',
    'Reply with only the answer. What color is a ripe banana?',
    'Reply with only the answer. Complete: Monday, Tuesday,',
]

def run(root):
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(9404)
    env=upstream(); cfg=env['A2DQwen3Config'].from_pretrained(MODEL,local_files_only=True)
    model,loading=transformers.AutoModelForMaskedLM.from_pretrained(MODEL,config=cfg,dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert not loading.get('missing_keys') and not loading.get('unexpected_keys') and not loading.get('mismatched_keys')
    model.eval(); tok=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    count=0; start=time.monotonic()
    with torch.inference_mode():
        for question in QUESTIONS:
            prompt=tok.apply_chat_template([{'role':'user','content':question}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            padding=(-len(prompt))%4
            x=torch.tensor([[tok.pad_token_id]*padding+prompt]); prefix=x.shape[1]
            steps=[]
            for block in range(8):
                x=torch.cat((x,torch.full((1,4),tok.mask_token_id,dtype=torch.long)),dim=1)
                for step in range(4):
                    mask,pos=env['_prepare_for_sampling'](x,4,tok.pad_token_id)
                    z=model(x,attention_mask=mask,position_ids=pos,use_cache=False,logits_to_keep=4).logits[0]
                    count+=1
                    confidence,guess=z.softmax(-1).max(-1)
                    confidence[x[0,-4:]!=tok.mask_token_id]=-float('inf')
                    chosen=int(confidence.argmax()); x[0,-4+chosen]=guess[chosen]
                    steps.append({'block':block,'step':step,'position':chosen,'token':int(guess[chosen])})
                if tok.eos_token_id in x[0,prefix:]: break
            tokens=x[0,prefix:].tolist()
            eos=tokens.index(tok.eos_token_id) if tok.eos_token_id in tokens else len(tokens)
            record={'question':question,'prompt_ids':prompt,'padding':padding,'tokens':tokens,
                    'text':tok.decode(tokens[:eos],skip_special_tokens=True),'steps':steps,
                    'terminated':eos<len(tokens)}
            with (root/'outputs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    runtime={'calls':count,'seconds':time.monotonic()-start,'model':str(MODEL),
        'model_sha':sha(MODEL/'model.safetensors'),'runner_sha':sha(__file__),
        'source_sha':sha(SOURCE),'torch':torch.__version__,'transformers':transformers.__version__,
        'loading':loading,'scope':'4 known-answer formatting DEV prompts; no scientific gate'}
    (root/'runtime.json').write_text(json.dumps(runtime,indent=2,default=json_default)+'\n')
    (root/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in root.iterdir()},indent=2)+'\n')
    print(json.dumps(runtime,indent=2,default=json_default))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root',type=Path); args=p.parse_args(); run(args.root)
